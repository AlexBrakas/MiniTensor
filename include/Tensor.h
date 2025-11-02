// Tensor.h : Define the Tensor class required and include
// details about the methods in this library

#pragma once

#include <iostream>
#include <vector>
#include <cassert> 

class Tensor {
public:
    /**
     * @brief Constructs a Tensor with a given shape.
     * @param shape A vector of integers defining the dimensions of the tensor.
     */
    Tensor(const std::vector<int>& shape);

    /**
     * @brief Constructs a Tensor from N-dimensional nested vector data.
     * @tparam T The nested vector type.
     * @param data The nested vector containing the initial data.
     */
    template<typename T>
    Tensor(const T& data);

    
    /**
     * @brief Gets the shape of the tensor.
     * @return A constant reference to the internal vector representing the shape.
     */
    const std::vector<int>& getShape() const;

    /**
     * @brief Gets the total number of elements in the tensor.
     * @return The total size as an integer.
     */
    int getSize() const;

    /**
     * @brief Gets the data in the tensor.
     * @return The data of the tensor.
     */
    const std::vector<float>& getData() const;

    /**
     * @brief Provides read/write access using N-dimensional coordinates.
     * @tparam Args A pack of integer types.
     * @param coords The coordinates for each dimension.
     * @return A reference to the element at the given coordinates.
     */
    template<typename... Args>
    float& operator()(Args... coords);

    /**
     * @brief Performs element-wise addition with another tensor.
     * @param other The tensor on the right-hand side of the '+' operator.
     * @return A new Tensor containing the result of the addition.
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * @brief Performs addition with a float to all elements.
     * @param scalar The float on the right-hand side of the '+' operator.
     * @return A new Tensor containing the result of the addition.
     */
    Tensor operator+(float scalar) const;

    /**
     * @brief Performs element-wise subtraction with another tensor.
     * @param other The tensor on the right-hand side of the '-' operator.
     * @return A new Tensor containing the result of the subtraction.
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * @brief Performs subtraction with a float to all elements.
     * @param scalar The float on the right-hand side of the '-' operator.
     * @return A new Tensor containing the result of the subtraction.
     */
    Tensor operator-(float scalar) const;

    /**
     * @brief Performs element-wise multiplication with another tensor.
     * @param other The tensor on the right-hand side of the '*' operator.
     * @return A new Tensor containing the result of the multiplication.
     */
    Tensor operator*(const Tensor& other) const;

    /**
     * @brief Performs mutiplication with a float to all elements.
     * @param scalar The float on the right-hand side of the '*' operator.
     * @return A new Tensor containing the result of the multiplication.
     */
    Tensor operator*(float scalar) const;

    /**
     * @brief Performs element-wise division with another tensor.
     * @param other The tensor on the right-hand side of the '/' operator.
     * @return A new Tensor containing the result of the division.
     */
    Tensor operator/(const Tensor& other) const;

    /**
     * @brief Performs division with a float to all elements.
     * @param scalar The float on the right-hand side of the '/' operator.
     * @return A new Tensor containing the result of the division.
     */
    Tensor operator/(float scalar) const;


private:
    /**
     * @brief Recursively infers the tensor's shape from nested vector data.
     * @tparam T The current nested vector type.
     * @param data The nested vector data to infer the shape from.
     */
    template<typename T>
    void infer_shape(const T& data);

    /**
     * @brief Recursively flattens nested vector data into the internal _data vector.
     * @tparam T The current nested vector type.
     * @param data The nested vector data to flatten.
     */
    template<typename T>
    void flatten_data(const T& data);

    std::vector<int> _shape;
    std::vector<float> _data;
};


template<typename T>
Tensor::Tensor(const T& data) {
    infer_shape(data);

    int total_size = 1;
    for (int dim : _shape) {
        total_size *= dim;
    }
    _data.reserve(total_size);

    flatten_data(data);
}

// Recursive helper to infer the shape
template<typename T>
void Tensor::infer_shape(const T& data) {
    _shape.push_back(static_cast<int>(data.size()));

    if constexpr (!std::is_same_v<typename T::value_type, float>) {
        infer_shape(data[0]);
    }
}

// Recursive helper to flatten the data
template<typename T>
void Tensor::flatten_data(const T& data) {
    if constexpr (std::is_same_v<typename T::value_type, float>) {
        _data.insert(_data.end(), data.begin(), data.end());
    }
    else {
        for (const auto& item : data) {
            flatten_data(item);
        }
    }
}

template<typename... Args>
float& Tensor::operator()(Args... coords) {
    static_assert((std::is_same_v<Args, int> && ...), "All coordinates must be integers.");

    assert(sizeof...(coords) == _shape.size() && "Incorrect number of dimensions provided.");

    int coords_array[] = { coords... };
    int index = 0;
    int stride = 1;

    for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
        index += coords_array[i] * stride;
        stride *= _shape[i];
    }

    assert(index >= 0 && index < _data.size() && "Tensor index is out of range");
    return _data[index];
}