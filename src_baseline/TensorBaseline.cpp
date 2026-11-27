// Tensor.cpp : Defines the core class required for creating tensors.
//

#include "TensorBaseline.h"
#include <algorithm>
#include <cassert>

namespace baseline {

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

    // Naive i-j-k triple loop with no tiling, SIMD or threading.
    Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
        assert(a.getShape().size() == 2 && "Matrix A must be 2D");
        assert(b.getShape().size() == 2 && "Matrix B must be 2D");

        int n = a.getShape()[0]; // rows of a
        int m = a.getShape()[1]; // cols of a
        int p = b.getShape()[1]; // cols of b

        assert(m == b.getShape()[0] && "Inner dimensions must match for matmul");

        Tensor result({ n, p });

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < p; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < m; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
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
}