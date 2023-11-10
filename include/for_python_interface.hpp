#pragma once
#include "1_bit_vs_1_bit.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <immintrin.h>
#include <tuple>
#include "cmath"
#include <cstdlib>
#include <bitset>
#include <cstring>
#include <assert.h>
using namespace std;

//TODO: Rewrite using a template.. input type could be of any kind that supported signed values

vector<uint64_t> pack_int64_into_bits(const vector<int64_t> &int_vec)
{
    size_t new_size = int_vec.size() / 64;

    vector<uint64_t> binary_vec(new_size);
    uint64_t tbin, sign;
    size_t n_bits = 64, idx = 0;

    for (size_t i = 0; i < int_vec.size(); i += n_bits, idx++)
    {
        tbin = 0;

        for (size_t j = 0; j < n_bits; j++)
        {
            sign = (uint64_t) (int_vec[i + j] > 0); //here, we should check if the value is equal
            // j-th bit from the left (|-->j----|)
            tbin |= sign << ((n_bits - 1) - j);
            // j-th bit from the right (|---j<----|)
            // tbin |= sign << j;
        }
        binary_vec[idx] = tbin;
    }
    return binary_vec;
}

// Ensure that m, k, n are multiple of 512.

int64_t * binary_matrix_multiplication(const vector<int64_t> &A, const vector<int64_t> &B, const size_t m, const size_t k, const size_t n)
{

    size_t k_bin = k / 64;

    // Pack A and copy it into an aligned memory location.

    // TODO: This should be removed by replacing the current load instructions with loadu.
    auto A_bin = pack_int64_into_bits(A);
    uint64_t *A_bin_aligned = (uint64_t *)std::aligned_alloc(64, sizeof(uint64_t) * m * k_bin);
    memcpy(A_bin_aligned, A_bin.data(), m * k_bin * sizeof(uint64_t));

    // Reorganize the elements of B for a memory register-friendly access
    auto B_packed = vector<int64_t>(B.size());
    for (size_t i = 0; i < k; i += 64)
    {
        for (size_t j = 0; j < n; j++)
        {
            for (size_t p = 0; p < 64; p++)
            {
                B_packed[(i / 64) * n * 64 + p + j * 64] = B[i * n + p * n + j];
            }
        }
    }

    //Pack B and copy it into an aligned memory location.

    // TODO: same as for A.
    auto B_packed_bin = pack_int64_into_bits(B_packed);
    uint64_t *B_bin_aligned_packed = (uint64_t *)std::aligned_alloc(64, sizeof(uint64_t) * n * k_bin);
    memcpy(B_bin_aligned_packed, B_packed_bin.data(), n * k_bin * sizeof(uint64_t));

    int64_t *C_aligned = (int64_t *)std::aligned_alloc(64, 8 * m * n);

    const size_t mr = 8;
    const size_t nr = 16;

    for (size_t x = 0; x < m; x += mr)
    {
        for (size_t y = 0; y < n; y += nr)
        {
            kernel_bin_8x16(A_bin_aligned, B_bin_aligned_packed, C_aligned, m, k_bin, n, x, y);
        }
    }

    std::free(A_bin_aligned);
    std::free(B_bin_aligned_packed);

    return C_aligned;
}

// vector<uint64_t> binary_multiplication(const vector<float> &A, const vector<float> &B, vector<float> &C, const size_t m, const size_t k, const size_t n, const size_t mr, const size_t nr)
// {

//     size_t k_bin = k / 64;

//     auto A_bin = convert_float_to_bin_64bit(A);

//     uint64_t *A_bin_aligned = (uint64_t *)std::aligned_alloc(64, sizeof(uint64_t) * m * k_bin);
//     memcpy(A_bin_aligned, A_bin.data(), m * k_bin * sizeof(uint64_t));

//     auto B_packed = vector<float>(B.size());
//     // B packed has shape (k/64, N*64)
//     // N should be a multiple of 8
//     for (size_t i = 0; i < k; i += 64)
//     {
//         for (size_t j = 0; j < n; j++)
//         {
//             for (size_t p = 0; p < 64; p++)
//             {
//                 B_packed[(i / 64) * n * 64 + p + j * 64] = B[i * n + p * n + j];
//             }
//         }
//     }

//     auto B_packed_bin = convert_float_to_bin_64bit(B_packed);
//     uint64_t *B_bin_aligned_packed = (uint64_t *)std::aligned_alloc(64, sizeof(uint64_t) * n * k_bin);
//     memcpy(B_bin_aligned_packed, B_packed_bin.data(), n * k_bin * sizeof(uint64_t));

//     int64_t *C_aligned = (int64_t *)std::aligned_alloc(64, 8 * m * n);

//     vector<uint64_t> elapsed_times;

//     for (size_t x = 0; x < m; x += mr){
//         for (size_t y = 0; y < n; y += nr)
//         {
//             kernel_bin_8x16(A_bin_aligned, B_bin_aligned_packed, C_aligned, m, k_bin, n, x, y);
//         }
//     }
//     for (size_t i = 0; i < m * n; i++)
//     {
//         C[i] = (float)C_aligned[i];
//     }

//     std::free(A_bin_aligned);
//     std::free(B_bin_aligned_packed);

//     return elapsed_times;
// }