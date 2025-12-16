#include <iostream>
#include <omp.h>
#include "tensor.hpp"


// parallel matrix multiplication of two tensors
void matmul_forward_parallel(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
        throw std::invalid_argument("Incompatible tensor dimensions for matmul.");
    }

    #pragma omp parallel for
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            C.data[i * C.cols + j] = sum;
        }
    }
}

//sequential matrix multiplication of two tensors
void matmul_forward_sequential(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
        throw std::invalid_argument("Incompatible tensor dimensions for matmul.");
    }

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            C.data[i * C.cols + j] = sum;
        }
    }
}

int main() {
    std::cout << "Nutze OpenMP mit max. Threads: " << omp_get_max_threads() << std::endl;
    
    Tensor A(1024, 1024);
    Tensor B(1024, 1024);
    Tensor C(1024, 1024);

    // Initialize A and B with some values
    std::fill(A.data.begin(), A.data.end(), 1.0f);
    std::fill(B.data.begin(), B.data.end(), 1.0f);

    // measure time
    double start = omp_get_wtime();
    matmul_forward_parallel(A, B, C);
    double end = omp_get_wtime();
    std::cout << "C[0,0]: " << C.at(0, 0) << " expected: 1024" << std::endl;
    std::cout << "Elapsed time: " << (end - start) << " seconds" << std::endl;

    
    // measure time for sequential
    start = omp_get_wtime();
    matmul_forward_sequential(A, B, C);
    end = omp_get_wtime();
    std::cout << "C[0,0]: " << C.at(0, 0) << " expected: 1024" << std::endl;
    std::cout << "Elapsed time (sequential): " << (end - start) << " seconds" << std::endl;

    return 0;
}

