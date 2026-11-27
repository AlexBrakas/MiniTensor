# MiniTensor

[![C++ CI Build and Test](https://github.com/AlexBrakas/MiniTensor/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/AlexBrakas/MiniTensor/actions/workflows/build-and-test.yml)

A dependency-free C++20 tensor library built from scratch, focused on how tensor operations map onto real hardware: memory layout, cache behaviour and SIMD. The long-term goal is a small neural-network library; the current work is the high-performance foundation underneath it.

## Status: Paused

Development is paused while I focus on another project.

**Implemented:** the N-dimensional \Tensor` core and a tiled, OpenMP-parallel AVX-512 matrix multiply, verified against a baseline implementation with unit tests and CI.`

**Next, when work resumes:** a benchmark harness measuring the tiled AVX-512 GEMM against the baseline and the CPU's theoretical peak, then register-blocked micro-kernels.

Design research and paper notes are kept in the [Research Log](./docs/learning_process.md).

## Approach

Each component goes through four stages:

1. **Baseline:** implement the straightforward version in plain C++ and test it.
2. **Research:** identify the bottleneck and study established optimizations (tiling, packing, SIMD), drawing on work such as GotoBLAS, BLIS and TVM.
3. **Implementation:** rewrite the hot kernels with cache-aware layouts and intrinsics (AVX-512).
4. **Benchmarking:** measure each step against the baseline and record what changed and why.

## Features

* **N-dimensional tensor:** templated `Tensor` class.
    * Construct from a shape or from nested `std::vector` data.
    * `tensor(i, j, k)` element access for any number of dimensions.
    * Element-wise operators (`+`, `-`, `*`, `/`) for tensor–tensor and tensor–scalar operations.
* **Matrix multiply:** tiled AVX-512 GEMM with masked edge handling and OpenMP threading, plus a scalar fallback; verified against a naive baseline.
* **Build:** CMake with presets.
* **Tests:** unit tests for the tensor module, run with CTest.
* **CI:** build and test on every push with GitHub Actions.

## Roadmap

* **Optimized GEMM:** cache tiling, operand packing, AVX-512 FMA micro-kernels, zero padding for power-of-two shapes, and multi-threading.
* **Benchmarks:** GFLOP/s and percentage of theoretical peak for each kernel version.
* **Neural-network layers:** layers, loss functions and optimizers built on the optimized tensor core.
* **RISC-V Vector (exploratory):** following the schedule/compute separation from TVM (Chen et al., 2018), a vector-length-agnostic RVV back-end for the core kernels.

## Getting Started

### Prerequisites

* A C++20 compiler (GCC, Clang or MSVC)
* [CMake](https://cmake.org/download/) (see `CMakeLists.txt` for the minimum version)

### Build and test

```bash
# 1. Clone the repository
git clone https://github.com/AlexBrakas/MiniTensor.git
cd MiniTensor

# 2. Configure
cmake -S . -B build

# 3. Build
cmake --build build

# 4. Run the tests
ctest --test-dir build
```

## License

MIT
