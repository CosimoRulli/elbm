#pragma once

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
#include "1_bit_vs_1_bit.hpp"
#include "omp.h"

using namespace std;





inline void trivial_dense_multiplication(const vector<float> &A, const vector<float> &B, const size_t M, const size_t K, const size_t N, vector<float> &C) {
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}


vector<float> generate_fp32_mat(const size_t M, const size_t N){
    vector<float> vec(M*N);
    random_device rd;
    mt19937 e(rd());
    uniform_real_distribution<> dist_float(-5, 5);
    generate(vec.begin(), vec.end(), [&]() {return floor(dist_float(e));});
    return vec;
}

vector<float> generate_bin_mat(const size_t M, const size_t N){

    random_device rd;
    mt19937 e(rd());
    vector<float> float_v (M*N);
    uniform_real_distribution<> dist_float(-5, 5);

    float p;
    for (size_t i=0; i <M*N; i++){
        p = dist_float(e);
        float_v[i] = (float)(p >= 0 ? 1 : -1);
    }

    return float_v;
}


inline uint64_t are_equal(const float a, const float b, const float toll){
    uint64_t return_val;
    if (abs(a - b) >= toll){
        return_val = 0;
    }else{
        return_val = 1;
    }
    return return_val;
}


template<typename T>
void print_mat(const vector<T> &mat, const size_t M, const size_t N){
    for(size_t i=0; i <M; i++){
        for (size_t j=0; j<N; j++){
            cout<<mat[i*N +j]<<" ";
        }
        cout<<"\n";
    }
}


template<typename T>
void print_mat(const T *mat, const size_t M, const size_t N){
    for(size_t i=0; i <M; i++){
        for (size_t j=0; j<N; j++){
            cout<<mat[i*N +j]<<" ";
        }
        cout<<"\n";
    }
}

vector<float> transpose_matrix(const vector<float> &data, const size_t K, const size_t N){
    vector<float> transposed_data(K*N);
    for (size_t j = 0; j < N; j++){
        for (size_t k = 0; k < K; k++){
            transposed_data[j*K + k] = data[k * N + j];
        }
    }
    return transposed_data;
}

template <typename  T1, typename  T2>
size_t check_equality(const vector<T1> &A, const vector<T2> & B, const size_t M, const size_t N, const float toll){
    size_t count=0;
    for (size_t i=0; i< M; i++){
        for (size_t j=0; j< N; j++){
            if (abs(A[i*N +j] - (T1) B[i*N +j]) >= toll) count++;
        }
    }
    return count;

}

template <typename  T1, typename  T2>
size_t check_equality(const T1 *A, const T2* B, const size_t M, const size_t N, const float toll){
    size_t count=0;
    for (size_t i=0; i< M; i++){
        for (size_t j=0; j< N; j++){
            if (abs(A[i*N +j] - (T1) B[i*N +j]) >= toll) count++;
        }
    }
    return count;

}



vector<uint64_t> convert_float_to_bin_64bit(const vector<float>& int_vec) {
    size_t new_size = int_vec.size() / 64;

    vector<uint64_t> binary_vec(new_size);
    uint64_t tbin;
    uint64_t sign;
    size_t n_bits = 64;
    size_t idx=0;
    for (size_t i=0; i< int_vec.size(); i+=n_bits, idx++){
        tbin = 0;

        for (size_t j = 0; j < n_bits; j++) {
            sign = (uint64_t) (int_vec[i+j] > 0);
            //j-th bit from the left (|-->j----|)
            tbin |= sign << ((n_bits - 1) - j);
            //j-th bit from the right (|---j<----|)
            //tbin |= sign << j;

        }
        //std::bitset<16> x (tbin);
        //cout<<x<<"\n";
        binary_vec[idx] = tbin;

    }
    return binary_vec;

}

//todo i convert vanno riscritti con un template typename 
vector<uint32_t> convert_float_to_bin_32bit(const vector<float>& int_vec) {
    size_t new_size = int_vec.size() / 32;
    cout<<"New size "<<new_size<<"\n";

    vector<uint32_t> binary_vec(new_size);
    uint32_t tbin;
    uint32_t sign;
    size_t n_bits = 32;
    size_t idx = 0;
    for (size_t i=0; i< int_vec.size(); i+=n_bits, idx++){
        tbin = 0;

        for (size_t j = 0; j < n_bits; j++) {
            sign = (uint32_t) (int_vec[i+j] > 0);
            //j-th bit from the left (|-->j----|)
            tbin |= sign << ((n_bits - 1) - j);
            //j-th bit from the right (|---j<----|)
            //tbin |= sign << j;

        }
        //std::bitset<16> x (tbin);
        //cout<<x<<"\n";
        binary_vec[idx] = tbin;

    }
    return binary_vec;

}


vector<uint16_t> convert_float_to_bin_16bit(const vector<float>& int_vec) {
    size_t new_size = int_vec.size() / 16;
    cout<<"New size "<<new_size<<"\n";

    vector<uint16_t> binary_vec(new_size);
    uint16_t tbin;
    int sign;
    size_t n_bits = 16;
    size_t idx=0;
    for (size_t i=0; i< int_vec.size(); i+=n_bits, idx++){
        tbin = 0;
        for (size_t j = 0; j < n_bits; j++) {
            sign = (int) (int_vec[i+j] > 0);
            //j-th bit from the left (|-->j----|)
            //tbin |= sign << ((n_bits - 1) - j);
            //j-th bit from the right (|---j<----|)
            tbin |= sign << j;

        }
        //std::bitset<16> x (tbin);
        //cout<<x<<"\n";
        binary_vec[idx] = tbin;

    }
    return binary_vec;

}


vector<uint64_t> measure_time_bin_bin_transposed(const vector<float> &A, const vector<float> &B, vector<float> &C, const size_t m, const size_t k, const size_t n, const size_t n_run){

    //todo A and B are passed as float but must be in -1, 1
    auto A_bin = convert_float_to_bin_64bit(A);
    auto B_transposed = transpose_matrix(B, k, n);
    auto B_bin_transposed = convert_float_to_bin_64bit(B_transposed);

    size_t k_bin = k / 64;

    uint64_t * A_bin_aligned = (uint64_t*) std::aligned_alloc(64, sizeof(uint64_t)*m*k_bin);
    uint64_t * B_bin_aligned_transposed = (uint64_t*) std::aligned_alloc(64, sizeof(uint64_t)*n*k_bin);
    
    memcpy(A_bin_aligned, A_bin.data(), m*k_bin* sizeof(uint64_t));
    memcpy(B_bin_aligned_transposed, B_bin_transposed.data(), n * k_bin * sizeof(uint64_t));

    vector<int64_t> C_bin(m*n, 0);
    vector<uint64_t> elapsed_times;

    for (size_t run=0; run < n_run; run ++){
        auto start = std::chrono::high_resolution_clock::now();
        mult_bin_avx512_transposed((uint64_t *) A_bin_aligned, (uint64_t *) B_bin_aligned_transposed, C_bin.data(), m, k_bin, n);
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
        elapsed_times.push_back(elapsed);
    }

    for (size_t i=0; i<C_bin.size(); i++){
        C[i] = (float) C_bin[i];
    }

    std::free(A_bin_aligned);
    std::free(B_bin_aligned_transposed);

   return elapsed_times;

}




template<auto kernel>
vector<uint64_t> measure_time_bin_bin_kernel(const vector<float> &A, const vector<float> &B, vector<float> &C, const size_t m, const size_t k, const size_t n, const size_t mr, const size_t nr, const size_t n_run){

    size_t k_bin = k / 64;

    auto A_bin = convert_float_to_bin_64bit(A);
    uint64_t* A_bin_aligned = (uint64_t*) std::aligned_alloc(64, sizeof(uint64_t)*m*k_bin);
    memcpy(A_bin_aligned, A_bin.data(), m*k_bin* sizeof(uint64_t));


    auto B_packed = vector<float>(B.size());
    // B packed has shape (k/64, N*64)
    // N should be a multiple of 8
    for (size_t i=0; i< k; i+=64){
        for (size_t j=0; j<n; j++){
            for (size_t p=0; p < 64; p++) {
                B_packed[(i / 64) * n*64 + p + j*64] = B[i * n + p * n + j];
            }
        }
    }

    auto B_packed_bin = convert_float_to_bin_64bit(B_packed);
    uint64_t* B_bin_aligned_packed = (uint64_t*) std::aligned_alloc(64, sizeof(uint64_t)*n*k_bin);
    memcpy(B_bin_aligned_packed, B_packed_bin.data(), n * k_bin * sizeof(uint64_t));

    int64_t * C_aligned = (int64_t*) std::aligned_alloc(64, 8*m*n);

    vector<uint64_t> elapsed_times;

    for (size_t run=0; run < n_run; run ++){
        auto start = std::chrono::high_resolution_clock::now();
            for (size_t x = 0; x < m; x += mr)
                for (size_t y = 0; y < n; y += nr){
                    kernel(A_bin_aligned, B_bin_aligned_packed, C_aligned, m, k_bin, n, x, y);   
                }
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
        elapsed_times.push_back(elapsed);
    }

    for (size_t i=0; i<m*n; i++){
            C[i] = (float) C_aligned[i];
        }

    std::free(A_bin_aligned);
    std::free(B_bin_aligned_packed);

   return elapsed_times;

}



template<auto kernel>
vector<uint64_t> measure_time_bin_bin_kernel_16bit(const vector<float> &A, const vector<float> &B, vector<float> &C, const size_t m, const size_t k, const size_t n, const size_t mr, const size_t nr, const size_t n_run){

    size_t k_bin = k / 16;

    auto A_bin = convert_float_to_bin_16bit(A);
    uint16_t* A_bin_aligned = (uint16_t*) std::aligned_alloc(64, sizeof(uint16_t)*m*k_bin);
    memcpy(A_bin_aligned, A_bin.data(), m*k_bin* sizeof(uint16_t));


    auto B_packed = vector<float>(B.size());
    // B packed has shape (k/16, N*16)
    // N should be a multiple of 16
    for (size_t i=0; i< k; i+=16){
        for (size_t j=0; j<n; j++){
            for (size_t p=0; p < 16; p++) {
                B_packed[(i / 16) * n*16 + p + j*16] = B[i * n + p * n + j];
            }
        }
    }

    auto B_packed_bin = convert_float_to_bin_16bit(B_packed);
    uint16_t* B_bin_aligned_packed = (uint16_t*) std::aligned_alloc(64, sizeof(uint16_t)*n*k_bin);
    memcpy(B_bin_aligned_packed, B_packed_bin.data(), n * k_bin * sizeof(uint16_t));

    int16_t * C_aligned = (int16_t*) std::aligned_alloc(64, sizeof(int16_t)*m*n);

    vector<uint64_t> elapsed_times;

    for (size_t run=0; run < n_run; run ++){
        auto start = std::chrono::high_resolution_clock::now();
            for (size_t x = 0; x < m; x += mr)
                for (size_t y = 0; y < n; y += nr){
                    kernel(A_bin_aligned, B_bin_aligned_packed, C_aligned, m, k_bin, n, x, y);   
                }
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
        elapsed_times.push_back(elapsed);
    }

    for (size_t i=0; i<m*n; i++){
            C[i] = (float) C_aligned[i];
        }

    std::free(A_bin_aligned);
    std::free(B_bin_aligned_packed);

   return elapsed_times;

}




