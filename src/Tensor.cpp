// Tensor.cpp : Defines the core class required for creating tensors.
//

#include "Tensor.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstddef>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

Tensor::Tensor(const std::vector<int>& shape) : _shape(shape) {
    int total_size = 1;
    for (int dim : _shape) {
        total_size *= dim;
    }
    _data.resize(total_size, 0.0f);
}

const std::vector<int>& Tensor::getShape() const {
    return _shape;
}

int Tensor::getSize() const {
    // return 32-bit instead of 64-bit
    return static_cast<int>(_data.size());
}

const std::vector<float>& Tensor::getData() const {
    return _data;
}

Tensor Tensor::operator+(const Tensor& other) const {
    assert(_shape == other.getShape() && "Tensor shapes must match for addition.");

    Tensor result(_shape);
    const std::vector<float>& other_data = other.getData();

    for (int i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] + other_data[i];
    }

    return result;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor result(_shape);

    for (int i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] + scalar;
    }

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    assert(_shape == other.getShape() && "Tensor shapes must match for subtraction.");

    Tensor result(_shape);
    const std::vector<float>& other_data = other.getData();

    for (int i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] - other_data[i];
    }

    return result;
}

Tensor Tensor::operator-(float scalar) const {
    Tensor result(_shape);

    for (int i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] - scalar;
    }

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    assert(_shape == other.getShape() && "Tensor shapes must match for multiplication.");

    Tensor result(_shape);
    const std::vector<float>& other_data = other.getData();

    for (int i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] * other_data[i];
    }

    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(_shape);

    for (int i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] * scalar;
    }

    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    assert(_shape == other.getShape() && "Tensor shapes must match for division.");

    Tensor result(_shape);
    const std::vector<float>& other_data = other.getData();

    for (int i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] / other_data[i];
    }

    return result;
}

Tensor Tensor::operator/(float scalar) const {
    Tensor result(_shape);

    for (int i = 0; i < _data.size(); ++i) {
        result._data[i] = _data[i] / scalar;
    }

    return result;
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    assert(a.getShape().size() == 2 && "Matrix A must be 2D");
    assert(b.getShape().size() == 2 && "Matrix B must be 2D");

    const int rows = a.getShape()[0];  // rows of A and C
    const int inner = a.getShape()[1]; // cols of A == rows of B
    const int cols = b.getShape()[1];  // cols of B and C

    assert(inner == b.getShape()[0] && "Inner dimensions must match for matmul");

    Tensor result({ rows, cols }); // constructor zero-initializes

    const float* A = a._data.data();
    const float* B = b._data.data();
    float* C = result._data.data();

    // 64x64 floats = 16 KB per tile
    constexpr int TILE_I = 64;
    constexpr int TILE_J = 64;
    constexpr int TILE_K = 64;
    #if defined(__AVX512F__)
        constexpr int VEC = 16; // floats per 512-bit register
    #endif
	    // Change loop ordering: i, j, k -> i, k, j to improve cache locality
     // Parallelize over output tiles only, so each thread owns its own block of C
#pragma omp parallel for collapse(2) schedule(static)
        for (int ii = 0; ii < rows; ii += TILE_I) {
            for (int jj = 0; jj < cols; jj += TILE_J) {
                const int i_end = std::min(ii + TILE_I, rows);
                const int j_end = std::min(jj + TILE_J, cols);

                for (int kk = 0; kk < inner; kk += TILE_K) {
                    const int k_end = std::min(kk + TILE_K, inner);

                    for (int i = ii; i < i_end; ++i) {
                        float* c_row = C + static_cast<std::size_t>(i) * cols;

                        for (int k = kk; k < k_end; ++k) {
                            const float a_val = A[static_cast<std::size_t>(i) * inner + k];
                            const float* b_row = B + static_cast<std::size_t>(k) * cols;
                            int j = jj;

#if defined(__AVX512F__)
                            const __m512 a_vec = _mm512_set1_ps(a_val); // broadcast A[i][k] to all 16 lanes
                            for (; j + VEC <= j_end; j += VEC) {
                                __m512 c_vec = _mm512_loadu_ps(c_row + j);
                                const __m512 b_vec = _mm512_loadu_ps(b_row + j);
                                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                _mm512_storeu_ps(c_row + j, c_vec);
                            }
                            if (j < j_end) {
                                // Mask off lanes past the tile edge so loads and stores stay in bounds
                                const __mmask16 mask = static_cast<__mmask16>((1u << (j_end - j)) - 1u);
                                __m512 c_vec = _mm512_maskz_loadu_ps(mask, c_row + j);
                                const __m512 b_vec = _mm512_maskz_loadu_ps(mask, b_row + j);
                                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                _mm512_mask_storeu_ps(c_row + j, mask, c_vec);
                                j = j_end;
                            }
#endif
                            for (; j < j_end; ++j) { // scalar path (no AVX-512)
                                c_row[j] += a_val * b_row[j];
                            }
                        }
                    }
                }
            }
        }
        return result;
}

Tensor Tensor::transpose(const Tensor& a) {
    // Currently assuming 2D.
    assert(a.getShape().size() == 2 && "Transpose currently supports 2D matrices.");

    int rows = a.getShape()[0];
    int cols = a.getShape()[1];

    Tensor result({ cols, rows });

    // L1 cache lines.
    const int BLOCK = 32;

    // block by block
    for (int i = 0; i < rows; i += BLOCK) {
        for (int j = 0; j < cols; j += BLOCK) {

            // Transpose within block
            // std::min handles multiples of BLOCK
            int i_max = std::min(i + BLOCK, rows);
            int j_max = std::min(j + BLOCK, cols);

            for (int i_block = i; i_block < i_max; ++i_block) {
                for (int j_block = j; j_block < j_max; ++j_block) {
                    result._data[j_block * rows + i_block] = a._data[i_block * cols + j_block];
                }
            }
        }
    }

    return result;
}