## Research: Technical Paper Reviews

### Paper: An Improvement of the Matrix-Matrix Multiplication Speed using 2D-Tiling and AVX512 Intrinsics
**Authors:** Nwe Zin Oo and Panyayot Chaikan (2021)
**Reference:** see `references.bib`

#### Memory Bottleneck
* Matrix multiplication performance is often limited by memory bandwidth. As matrices grow, the CPU cannot keep all of the data in its fast internal caches.
* **Cache levels:** CPUs use a hierarchy: L1 (fastest, smallest), L2, and L3 (largest, slowest, usually shared between cores).
* When the working set exceeds L3 capacity, the CPU must fetch from RAM (main memory), which takes far more clock cycles than an L1 hit, so the processor stalls while it waits for data.

#### 2D Loop Tiling
* **Tiling:** dividing large matrices into smaller tiles (blocks) and computing one tile at a time.
* **L1 optimization:** tiles are sized (e.g. $128 \times 64$) so the data being worked on fits in the L1 data cache.
* While working on a small tile, the CPU reuses each value many times before it is evicted, instead of repeatedly fetching it from L2, L3 or RAM. This raises arithmetic intensity (useful work per byte loaded).

#### AVX-512 (SIMD)
* Instead of computing one number at a time, AVX-512 processes 16 single-precision floats in a single instruction.
* **512-bit registers:** each register holds 16 floats, so one instruction operates on 16 data points at once.
* **FMA:** `_mm512_fmadd_ps` (fused multiply-add) computes `a * b + c` for all 16 lanes in one instruction, with a single rounding step.
* **Latency vs. throughput:** an FMA takes roughly 4 cycles before its result is ready (latency), but a core can *start* one or two FMAs every cycle (throughput). Reaching peak therefore requires several independent FMAs in flight at once, which is what loop unrolling provides.

#### Zero Padding
* A cache is split into *sets*, and each memory address maps to exactly one set, determined by bits of the address. Each set holds only a few lines (its associativity).
* If a row length is a power of two (e.g. 4096 floats), consecutive rows map to the same set. When walking down a column, those rows compete for the same few slots and evict each other (conflict misses), even while most of the cache is empty.
* Padding each row with a few extra unused elements shifts the row start addresses so they map to different sets, spreading the data across the whole cache.

#### OpenMP and Loop Unrolling
* `#pragma omp parallel for` distributes the outer loops across all physical cores.
* Unrolling processes several rows or vectors per iteration, giving the CPU multiple independent FMAs to overlap and hiding FMA latency.

### Applied to MiniTensor: Implementation Goals

| Concept from Paper | Planned Implementation in MiniTensor |
| :--- | :--- |
| **2D loop tiling** | The GEMM kernel will accept `tile_size` parameters chosen to fit L1/L2 based on the host CPU's cache sizes. |
| **AVX-512 (SIMD)** | An intrinsics header will detect CPU features at runtime and use `_mm512_fmadd_ps` in the innermost product loops, with a scalar fallback. |
| **Zero padding** | The `Tensor` allocator will over-allocate rows for power-of-two shapes, so logical dimensions are backed by cache-friendly physical strides. |
| **Loop unrolling** | Micro-kernels will compute 2–4 rows of the output at a time to keep enough independent FMAs in flight to hide latency. |

---

## Research: Other Notes
* A load that misses every cache level and goes to RAM costs on the order of ~200 cycles, while an AVX-512 FMA has a latency of about 4 cycles. A single RAM access can therefore cost as much time as roughly 50 back-to-back FMAs, which is why data reuse (tiling) matters more than raw arithmetic speed.
* Test machine: AMD Ryzen 9 9900X (Zen 5), which supports AVX-512, so the kernels above can be benchmarked locally.

---

## Implementation Notes: Fixing the Tiled AVX-512 GEMM

The first tiled AVX-512 version of `Tensor::matmul` crashed with heap corruption when tested against the baseline on non-aligned shapes. A comparison test against `baseline::matmul` exposed the bugs below.

### How the kernel works
* **Loop order:** the outer `ii` / `jj` / `kk` loops only choose which 64×64 tile to work on. Inside a tile the loops run `i`, then `k`, then `j` (the i-k-j order from the tiling paper), so rows of B and C are walked contiguously.
* **The core operation:** for each `A[i][k]`, add `A[i][k] × B[k][j]` into `C[i][j]` for every `j` in the tile.
* **Broadcast:** `_mm512_set1_ps(a_val)` copies the single value `A[i][k]` into all 16 lanes.
* **Row pointers:** `b_row` and `c_row` point at the start of row `k` of B and row `i` of C, so each load is just `row + j` instead of recomputing `k * cols + j` every time.
* **SIMD step:** load 16 values of C, load 16 values of B, `_mm512_fmadd_ps` computes all 16 `C += a × B` at once, then store. 64 multiply-adds take 4 instructions instead of 64.
* **Indexing rule:** for a row-major matrix, element `[row][col]` is at `row * row_length + col`. Every index in the kernel is this formula, sometimes split into "row start pointer + col".
* **`static_cast<std::size_t>`:** makes `i * cols` a 64-bit multiplication, so very large matrices cannot overflow the index before it becomes a memory offset.

### Bugs found and why the fixes work
* **Tile coverage:** the inner loops ran 16 rows and 16 `k` values (the vector width) instead of the full 64-wide tile, so most of each tile was never computed. Fix: inner loops run to `i_end`, `k_end` and `j_end`, the end of the tile clamped to the matrix size.
* **Vector stepping:** `j` stepped by 1 while each load and store covered 16 floats, so neighbouring iterations overlapped and added into the same outputs repeatedly. Fix: `j` steps by `VEC` (16).
* **Edges (masked tail):** when fewer than 16 values remain in a row, a full 16-wide load or store reads and writes past the end of the array, which caused the crash. Fix: a 16-bit lane mask, `(1u << remaining) - 1`, where each bit switches one lane on. `_mm512_maskz_loadu_ps` only reads memory for active lanes (inactive lanes become zero), and `_mm512_mask_storeu_ps` only writes active lanes. Writing zeros past the edge instead would corrupt memory that belongs to the next row or another variable.
* **Scalar fallback:** the old non-AVX branch used undefined names and could not compile. Fix: one shared scalar loop after the AVX block. With AVX-512 the vector loop and masked tail finish the row and leave `j == j_end`, so the scalar loop does nothing. Without AVX-512 the preprocessor removes the AVX block, `j` still starts at `jj`, and the scalar loop does all the work. `j` is the handoff point, so there is no second copy of the function to keep in sync.
* **OpenMP data race:** `collapse(2)` covered the `i` and `k` loops, but `k` is the dimension being summed over. Two threads with the same `i` and different `k` blocks both did `C[i][j] += ...` on the same element at the same time, so one thread's addition could overwrite the other's. Atomics would fix correctness but make every add synchronized and slow. Fix: parallelize over the output tiles (`ii` / `jj`) only. Each thread owns its own block of C and runs the whole `k` loop for it, so no two threads ever write the same element. Rule: parallelize over the outputs, never over the dimension being summed.

### Testing approach
* Naive kernels live in `src_baseline/` as `baseline::` free functions that use the same `Tensor` class, so only the kernel differs between the two sides.
* Tests compare `Tensor::matmul` against `baseline::matmul` element by element with a relative tolerance, since summing in a different order changes floating-point rounding.
* Shapes are deliberately awkward (e.g. 17×33 × 33×9, 100×70 × 70×130) so they cross tile and vector boundaries, plus aligned shapes (64×64, 128×256) to check the main path on its own.

### Next optimization
* The current kernel loads and stores each 16-value chunk of C for every `k`. A register-blocked micro-kernel keeps a small block of C in registers across the whole `k` loop, so C is loaded and stored once per block instead of once per `k`. This is the next step toward peak FLOP/s.