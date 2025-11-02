// Tensor.cpp : Defines the core class required for creating tensors.
//

#include "Tensor.h"


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