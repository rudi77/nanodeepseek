#pragma once

#include <vector>
#include <iostream>

/// @brief A class representing a multi-dimensional tensor which keeps weights and gradients.
class Tensor {
public:
    int rows;
    int cols;

    std::vector<float> data; // forward pass
    std::vector<float> grad; // backward pass

    Tensor(int rows, int cols)
        : rows(rows), cols(cols), data(rows * cols, 0.0f), grad(rows * cols, 0.0f) 
    {}

    
    /// @brief Accessor for data at position (r, c).
    inline float& at(int r, int c) {return data[r * cols + c];}

    /// @brief Accessor for gradient at position (r, c).
    inline float& grad_at(int r, int c) {return grad[r * cols + c];}

    /// @brief Zeros out the gradient tensor.
    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
};
