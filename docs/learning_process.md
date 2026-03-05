## Research Phase 1: SIMD and Cache-Aware Tiling

### Paper Summary: "An Improvement of the Matrix-Matrix Multiplication Speed using 2D-Tiling and AVX512 Intrinsics" 
**Authors:** Nwe Zin Oo and Panyayot Chaikan (2021)
**Reference:**

#### Memory Bottleneck
* Matrix multiplication performance is often limited by memory bandwidth. As matrices grow, the CPU cannot keep all data in its fast internal caches.
* **Cache Levels:** CPUs utilize a hierarchy: L1 (fastest/smallest), L2, and L3 (shared/slowest).
* When data size exceeds L3 capacity, the CPU must fetch from RAM (Main Memory), which can take significantly more clock cycles compared to L1, causing the processor to slow down while waiting for data.

#### 2D-Loop Tiling
* **Tiling:** The process of dividing large matrices into smaller "tiles" or blocks.
* **L1 Optimization:** Tiles are sized (e.g., $128 \times 64$) to fit entirely within the L1 data cache.
* Focusing on a small tile, the CPU reuses values multiple times without returning to slow RAM, maximizing throughput.

#### AVX-512 (SIMD)
* Instead of calculating one number at a time, AVX-512 allows the CPU to process 16 single-precision numbers in a single instruction.
* **512-bit Registers:** Utilizes ultra-wide registers to perform math across multiple data points simultaneously.
* **FMA:** Implements `_mm512_fmadd_ps` (Fused Multiply-Add) to perform a multiplication and an addition in a single clock cycle.

#### Zero Padding
* CPUs use mathematical formulas to decide where in the cache specific data is located.
* If a matrix size is a power of 2 (like 4096), rows often compete for the same cache set, kicking each other out even if the rest of the cache is empty.
* Zero Padding adds extra empty columns to shift memory addresses, "tricking" the CPU into spreading rows across all available cache floors.

#### OpenMP & Unrolling
* Use `#pragma omp parallel for` to distribute workload across all physical CPU cores.
* Processes multiple AVX lines at once to keep the internal execution units fully saturated.

---

### **Applied to MiniTensor: Implementation Goals**

| Concept from Paper | Planned Implementation in MiniTensor |
| :--- | :--- |
| **2D-Loop Tiling** | The GEMM kernel will accept `tile_size` parameters to stay within L1/L2 cache boundaries based on host CPU specifications. |
| **AVX-512 (SIMD)** | Implementation of an "Intrinsics" header to detect CPU features and utilize `_mm512_fmadd_ps` for inner-most product loops. |
| **Zero Padding** | The `Tensor` allocator will "over-allocate" memory for power-of-two matrices, ensuring logical dimensions are backed by cache-friendly physical strides. |
| **Loop Unrolling** | Micro-kernels will be written to process 2–4 rows of the result matrix simultaneously to hide instruction latency. |