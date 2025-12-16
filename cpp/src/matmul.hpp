#pragma once

#include <vector>

// B: [KxN] -> Bt: [NxK]
void transpose_kn_to_nk(
    const std::vector<float>& B, 
    std::vector<float>& Bt, 
    int K, int N);

// C = A [MxK] * B [KxN] (intern with Bt)
void matmul_mk_kn_mn(
    const std::vector<float>& A,
    const std::vector<float>& Bt,
    std::vector<float>& C, 
    int M, int K, int N);

// C = A [M×K] * Bt [N×K] (B bereits transponiert)
void matmul_mk_nk_mn_bt(
    const std::vector<float>& A,
    const std::vector<float>& Bt,
    std::vector<float>& C,
    int M, int K, int N);