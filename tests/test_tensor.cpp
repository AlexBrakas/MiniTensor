// test_tensor.cpp : tests the main functionality of the tensor class.
//  verify the output is correct

#include "Tensor.h"
#include <cassert>  
#include <vector>
#include <iostream>

// Test case 1: Verify the shape-based constructor
void test_constructor_by_shape() {
    std::cout << "Running test: constructor_by_shape..." << std::endl;
    std::vector<int> shape = {2, 3};
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
    assert(result(0,0) == 6.0f && "Test failed, incorrect addition of t(0, 0)");
    assert(result(1,1) == 9.0f && "Test failed, incorrect addition of t(1, 1)");
    std::cout << "PASS" << std::endl;
}

// Run tests
int main() {
    test_constructor_by_shape();
    test_constructor_by_1D_data();
    test_constructor_by_2D_data();
    test_constructor_by_3D_data();
    test_scalar_addition();

    std::cout << "\nAll Tensor tests passed!" << std::endl;
    return 0;
}