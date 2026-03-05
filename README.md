# MiniTensor

[![C++ CI Build and Test](https://github.com/AlexBrakas/MiniTensor/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/AlexBrakas/MiniTensor/actions/workflows/build-and-test.yml)

A lightweight, C++20 neural network library built from scratch. This project is a deep dive into the core components of deep learning, including N-dimensional tensors, automatic differentiation (coming soon), and dynamic computation graphs.

## 🚩 Current Project Status: Research for Tensor Optimization
**Current Focus:** Low-Level Optimization & Memory Hierarchy Analysis

MiniTensor is currently in the **Deep Research Phase**. Before any high-level neural network components are implemented, the project is focusing on the mathematical and hardware-level foundation.

* **Academic Benchmarking:** Studying foundational papers (GotoBLAS, BLIS, TVM) to architect a cache-aware GEMM kernel.
* **Hardware Profiling:** Analyzing AVX-512 vectorization and L1/L2/L3 cache occupancy strategies.
* **Documentation:** Establishing a [Research Log](./docs/learning_process.md) to track the transition from theory to C++ implementation.

## 🧠 Project Philosophy

MiniTensor is built on a four-stage iterative cycle designed to bridge the gap between high-level ML theory and low-level hardware execution:

1. **Baseline:** Implement the mathematical version in pure C++.
2. **Research:** Identify bottlenecks and study state-of-the-art optimizations (SIMD, Tiling, Packing).
3. **Implementation:** Rewrite core kernels using intrinsics (AVX-512) and cache-aware memory management.
4. **Benchmarking:** Quantify the performance increase to verify engineering gains.

## 🚀 Features
* **N-Dimensional Tensor:** A powerful, templated `Tensor` class.
    * Constructors from shape or N-dimensional `std::vector` data.
    * Python-like `tensor(i, j, k)` accessor for any number of dimensions.
    * Full operator overloading (`+`, `-`, `*`, `/`) for tensors and scalars.
* **Build System:** Professional setup using CMake.
* **Unit Tests:** Complete test suite for the Tensor module using CTest.
* **CI/CD:** Automatic testing on every push using GitHub Actions.

## 🏁 Getting Started

### Prerequisites
* A C++20 compliant compiler (e.g., MSVC, GCC)
* [CMake](https://cmake.org/download/) (Version 3.8 or higher)

### Building the Project
You can build the library, example application, and tests using CMake:

```bash
# 1. Clone the repository
git clone [https://github.com/AlexBrakas/MiniTensor.git](https://github.com/AlexBrakas/MiniTensor.git)
cd MiniTensor

# 2. Configure CMake
cmake -S . -B build

# 3. Build the project
cmake --build build
