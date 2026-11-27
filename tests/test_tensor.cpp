// test_tensor.cpp : tests the main functionality of the tensor class.
//  verify the output is correct

#include "Tensor.h"
#include "TensorBaseline.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

// Test case 1: Verify the shape-based constructor
void test_constructor_by_shape() {
    std::cout << "Running test: constructor_by_shape..." << std::endl;
    std::vector<int> shape = { 2, 3 };
    Tensor t(shape);
    assert(t.getShape() == std::vector<int>({ 2, 3 }) && "Test failed, incorrect shape (shape)");
    assert(t.getSize() == 6 && "Test failed, inccorect flatten data (shape)");
    std::cout << "PASS" << std::endl;
}

// Test case 2: Verify the data-based constructor for a 1D vector
void test_constructor_by_1D_data() {
    std::cout << "Running test: constructor_by_1D_data..." << std::endl;

    // Pass 1D data
    std::vector<float> my_data = { 1.0f, 2.0f, 3.0f };
    Tensor t(my_data);

    // Assert for the correct 1D shape
    assert(t.getShape() == std::vector<int>({ 3 }) && "Test failed, incorrect 1D shape");
    assert(t.getSize() == 3 && "Test failed, incorrect 1D size");
    assert(t(1) == 2.0f && "Test failed, incorrect 1D data access at t(1)");
    std::cout << "PASS" << std::endl;
}

// Test case 3: Verify the data-based constructor for a 2D vector
void test_constructor_by_2D_data() {
    std::cout << "Running test: constructor_by_2D_data..." << std::endl;
    std::vector<std::vector<float>> my_data = { {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f} };
    Tensor t(my_data);
    assert(t.getShape() == std::vector<int>({ 3, 2 }) && "Test failed, incorrect 2D shape");
    assert(t.getSize() == 6 && "Test failed, incorrect 2D size");
    assert(t(1, 0) == 3.0f && "Test failed, incorrect 2D data access at t(1,0)");
    std::cout << "PASS" << std::endl;
}

// Test case 4: Verify the data-based constructor for a 3D vector
void test_constructor_by_3D_data() {
    std::cout << "Running test: constructor_by_3D_data..." << std::endl;

    // 3D data (2x3x2)
    std::vector<std::vector<std::vector<float>>> my_data = {
        {
            {1.0f, 2.0f},
            {3.0f, 4.0f},
            {5.0f, 6.0f}
        },
        {
            {7.0f, 8.0f},
            {9.0f, 10.0f},
            {11.0f, 12.0f}
        }
    };

    Tensor t(my_data);

    assert(t.getShape() == std::vector<int>({ 2, 3, 2 }) && "Test failed, incorrect 3D shape");
    assert(t.getSize() == 12 && "Test failed, incorrect 3D size");
    assert(t(1, 2, 1) == 12.0f && "Test failed, incorrect 3D data access");
    std::cout << "PASS" << std::endl;
}

// Test case 5: Verify tensor-scalar addition
void test_scalar_addition() {
    std::cout << "Running test: scalar_addition..." << std::endl;
    std::vector<std::vector<float>> my_data = { {1.0f, 2.0f}, {3.0f, 4.0f} };
    Tensor t(my_data);
    Tensor result = t + 5.0f;
    assert(result(0, 0) == 6.0f && "Test failed, incorrect addition of t(0, 0)");
    assert(result(1, 1) == 9.0f && "Test failed, incorrect addition of t(1, 1)");
    std::cout << "PASS" << std::endl;
}

// Fill a 2D tensor (Tensor or baseline::Tensor) with varied, deterministic values
template <typename T>
T make_matrix(int rows, int cols, int seed) {
    T t({ rows, cols });
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            t(i, j) = static_cast<float>(((i * 31 + j * 17 + seed * 7) % 23) - 11) * 0.25f;
    return t;
}

// Compare two result buffers element by element; prints the first mismatch
bool results_match(const std::vector<float>& expected, const std::vector<float>& actual, int cols, const char* label) {
    if (expected.size() != actual.size()) {
        std::cout << "FAIL " << label << ": size mismatch" << std::endl;
        return false;
    }
    for (std::size_t idx = 0; idx < expected.size(); ++idx) {
        const float e = expected[idx], r = actual[idx];
        if (std::fabs(e - r) > 1e-3f * std::max(1.0f, std::fabs(e))) { // relative tolerance for reordered float sums
            std::cout << "FAIL " << label << " at (" << idx / cols << ", " << idx % cols
                << "): expected " << e << ", got " << r << std::endl;
            return false;
        }
    }
    return true;
}

// Test case 6: Verify optimized matmul against the baseline on one shape
bool check_matmul(int n, int m, int p) {
    Tensor a = make_matrix<Tensor>(n, m, 1);
    Tensor b = make_matrix<Tensor>(m, p, 2);
    baseline::Tensor ba = make_matrix<baseline::Tensor>(n, m, 1);
    baseline::Tensor bb = make_matrix<baseline::Tensor>(m, p, 2);

    Tensor actual = Tensor::matmul(a, b);
    baseline::Tensor expected = baseline::Tensor::matmul(ba, bb);

    std::cout << "Running test: matmul " << n << "x" << m << " * " << m << "x" << p << "..." << std::endl;
    if (!results_match(expected.getData(), actual.getData(), p, "matmul")) return false;
    std::cout << "PASS" << std::endl;
    return true;
}

// Run tests
int main() {
    test_constructor_by_shape();
    test_constructor_by_1D_data();
    test_constructor_by_2D_data();
    test_constructor_by_3D_data();
    test_scalar_addition();

    bool ok = true;
    ok &= check_matmul(1, 1, 1);
    ok &= check_matmul(17, 33, 9);      // nothing aligned to 16 or 64
    ok &= check_matmul(64, 64, 64);     // exactly one tile
    ok &= check_matmul(100, 70, 130);   // crosses tile boundaries with ragged edges
    ok &= check_matmul(128, 256, 48);   // multiple full tiles

    if (!ok) {
        std::cout << "\nTensor tests FAILED" << std::endl;
        return 1;
    }
    std::cout << "\nAll Tensor tests passed!" << std::endl;
    return 0;
}