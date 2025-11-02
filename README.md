# MiniTensor

[![C++ CI Build and Test](httpshttps://github.com/AlexBrakas/MiniTensor/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/AlexBrakas/MiniTensor/actions/workflows/build-and-test.yml)

A lightweight, C++20 neural network library built from scratch. This project is a deep dive into the core components of deep learning, including N-dimensional tensors, automatic differentiation (coming soon), and dynamic computation graphs.

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