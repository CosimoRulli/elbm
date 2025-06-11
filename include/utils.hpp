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
#include "1_bit_vs_1_bit_kernels.hpp"
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

tuple<vector<uint64_t>, vector<uint64_t> , vector<uint64_t>> generate_2bits_masks(const vector<float>& float_vec, const float s ){
    size_t new_size = float_vec.size() / 64;


    vector<uint64_t> m1(new_size);
    vector<uint64_t> m2(new_size);
    vector<uint64_t> m3(new_size);

    float toll = 0.01;
    uint64_t tbin_m1, tbin_m2, tbin_m3; 
    uint64_t sign_m1, sign_m2, sign_m3;

    size_t n_bits = 64;
    size_t idx = 0;

    for (size_t i=0; i< float_vec.size(); i+=n_bits, idx++){
        tbin_m1 = 0;
        tbin_m2 = 0;
        tbin_m3 = 0;  


        for (size_t j = 0; j < n_bits; j++) {
            
            sign_m1 = (uint64_t) (are_equal(float_vec[i+j], 3*s/2, toll));
            tbin_m1 |= sign_m1 << ((n_bits - 1) - j);

            sign_m2 = (uint64_t) (are_equal(abs(float_vec[i+j]), 3*s/2, toll));
            tbin_m2 |= sign_m2 << ((n_bits - 1) - j);

            sign_m3 = (uint64_t) (are_equal(float_vec[i+j], s/2, toll));
            tbin_m3 |= sign_m3 << ((n_bits - 1) - j);

        }
        //std::bitset<16> x (tbin);
        //cout<<x<<"\n";
        m1[idx] = tbin_m1;
        m2[idx] = tbin_m2;
        m3[idx] = tbin_m3;

    }
    return {m1, m2, m3};

}
tuple<vector<uint32_t>, vector<uint32_t> , vector<uint32_t>> generate_2bits_masks_32_bits(const vector<float>& float_vec, const float s ){
    size_t new_size = float_vec.size() / 32;


    vector<uint32_t> m1(new_size);
    vector<uint32_t> m2(new_size);
    vector<uint32_t> m3(new_size);

    float toll = 0.01;
    uint32_t tbin_m1, tbin_m2, tbin_m3; 
    uint32_t sign_m1, sign_m2, sign_m3;

    size_t n_bits = 32;
    size_t idx = 0;

    for (size_t i=0; i< float_vec.size(); i+=n_bits, idx++){
        tbin_m1 = 0;
        tbin_m2 = 0;
        tbin_m3 = 0;  


        for (size_t j = 0; j < n_bits; j++) {
            
            sign_m1 = (uint32_t) (are_equal(float_vec[i+j], 3*s/2, toll));
            tbin_m1 |= sign_m1 << ((n_bits - 1) - j);

            sign_m2 = (uint32_t) (are_equal(abs(float_vec[i+j]), 3*s/2, toll));
            tbin_m2 |= sign_m2 << ((n_bits - 1) - j);

            sign_m3 = (uint32_t) (are_equal(float_vec[i+j], s/2, toll));
            tbin_m3 |= sign_m3 << ((n_bits - 1) - j);

        }
        //std::bitset<16> x (tbin);
        //cout<<x<<"\n";
        m1[idx] = tbin_m1;
        m2[idx] = tbin_m2;
        m3[idx] = tbin_m3;

    }
    return {m1, m2, m3};

}


vector<uint32_t> compute_mask_counter_float_matrix(const vector<float> &B, const size_t k, const float s){
    size_t n = B.size()/k;
    float toll = 0.01;
    vector<uint32_t> mask_counter(n, 0);
    for(size_t i=0; i< k; i++){
        for (size_t j=0; j<n; j++){
            if (are_equal(abs(B[i*n +j]), 3*s/2 , toll) ){
                mask_counter[j]+=1;
            }
        }
    }
    return mask_counter;
}

vector<float> generate_2bits_mat(const size_t M, const size_t N, const float s){

    random_device rd;
    mt19937 e(rd());
    vector<float> float_v (M*N);
    uniform_real_distribution<> dist_float(-5, 5);
    float a_neg = -s * 3/2;
    float a_pos = s * 3/2;
    float b_neg = -s / 2;
    float b_pos = s / 2;


    float p;
    for (size_t i=0; i <M*N; i++){
        p = dist_float(e);
        if (p>=0){
            float_v[i] = (float)(p >= 2.5 ? a_pos : b_pos);
        }else{
            float_v[i] = (float)(p <= -2.5 ? a_neg : b_neg);
        }
        
    }

    return float_v;
}


inline void compute_mask_counter_float_matrix_avx(const uint64_t *B_packed, const size_t k, const size_t n, const uint64_t * mask_counter){
    __m512i acc{}, val{};
    size_t k_bin = k/64;    
    for(size_t j=0; j<n; j+=8){
        acc = _mm512_setzero_si512();
        for(size_t i=0; i < k_bin; i++){
            val = _mm512_load_si512((const void *) &(B_packed[j + i*n]));
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(val));
            }  
        _mm512_store_epi64((void *) &(mask_counter[j]), acc );
        }
    }


// inline uint64_t are_equal(const float a, const float b, const float toll){
//     uint64_t return_val;
//     if (abs(a - b) >= toll){
//         return_val = 0;
//     }else{
//         return_val = 1;
//     }
//     return return_val;
// }


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
    size_t count = 0;
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




vector<float> pack_matrix_for_binary_mul(const vector<float> &B, const size_t k, const size_t n){
    auto B_packed = vector<float>(B.size());
    // for (size_t i=0; i< k; i+=64){
    //     for (size_t j=0; j<n; j++){
    //         for (size_t p=0; p < 64; p++) {
    //             //B_packed[(i // 64) * n*64 + i%64 + j*64] = B[i * n + p * n + j];

    //             B_packed[(i / 64) * n*64 + p + j*64] = B[i * n + p * n + j];
    //         }
    //     }
    // }
    for (size_t i=0; i< k; i++){
        for (size_t j=0; j<n; j++){
                B_packed[(i / 64) * n*64 + i%64 + j*64] = B[i * n + j];
        }
    }

    return B_packed;

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






template<auto routine>
vector<uint64_t> measure_time_1_2_transposed(const vector<float> &A, const vector<float> &B, vector<float> &C, const size_t m, const size_t k, const size_t n, const float s, const size_t n_run){
        auto A_bin = convert_float_to_bin_64bit(A);
        auto B_transposed = transpose_matrix(B, k, n);
        auto [alpha, mask, beta] = generate_2bits_masks(B_transposed, s);

        uint64_t * a = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(A_bin.size()));
        memcpy(a, A_bin.data(), sizeof(uint64_t)*(A_bin.size()));

        uint64_t *alpha_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(alpha.size()));
        memcpy(alpha_aligned, alpha.data(), sizeof(uint64_t)*(alpha.size()));

        uint64_t *mask_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(mask.size()));
        memcpy(mask_aligned, mask.data(), sizeof(uint64_t)*(mask.size()));

        uint64_t *beta_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(beta.size()));
        memcpy(beta_aligned, beta.data(), sizeof(uint64_t)*(beta.size()));
        vector<uint64_t> elapsed_times;

        for (size_t run=0; run < n_run; run ++){
            auto start = std::chrono::high_resolution_clock::now();
            routine(a, alpha_aligned, mask_aligned, beta_aligned, C.data(), m, k, n, s);
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
            elapsed_times.push_back(elapsed);
        }
        
        std::free(a);
        std::free(alpha_aligned);
        std::free(mask_aligned);
        std::free(beta_aligned);

   return elapsed_times;

}


template<auto kernel>
vector<uint64_t> measure_time_1_2_kernel_64_bit(const vector<float> &A, const vector<float> &B, vector<float> &C, const size_t m, const size_t k, const size_t n, const float s, const size_t mr, size_t const nr, const size_t n_run){
    auto A_bin = convert_float_to_bin_64bit(A);

    vector<int64_t> integer_C(C.size());
    std::transform(C.begin(), C.end(), integer_C.begin(), [](float x) { return (int64_t)x;});
    auto B_packed = pack_matrix_for_binary_mul(B, k, n);
    //auto B_packed = vector<float>(B.size());
        // for (size_t i=0; i< k; i+=64){
    //     for (size_t j=0; j<n; j++){
    //         for (size_t p=0; p < 64; p++) {
    //             B_packed[(i / 64) * n*64 + p + j*64] = B[i * n + p * n + j];
    //         }
    //     }
    // }

    auto [alpha, mask, beta] = generate_2bits_masks(B_packed, s);

    uint64_t * a = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(A_bin.size()));
    memcpy(a, A_bin.data(), sizeof(uint64_t)*(A_bin.size()));

    uint64_t *alpha_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(alpha.size()));
    memcpy(alpha_aligned, alpha.data(), sizeof(uint64_t)*(alpha.size()));

    uint64_t *mask_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(mask.size()));
    memcpy(mask_aligned, mask.data(), sizeof(uint64_t)*(mask.size()));

    uint64_t *beta_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(beta.size()));
    memcpy(beta_aligned, beta.data(), sizeof(uint64_t)*(beta.size()));

    int64_t *c = (int64_t * ) std::aligned_alloc(64, sizeof(int64_t)*(integer_C.size()));
    memcpy(c, integer_C.data(),  sizeof(int64_t)*(integer_C.size()));

    vector<uint32_t> mask_counter_gt;
    mask_counter_gt = compute_mask_counter_float_matrix(B, k, s);    
    uint64_t* mask_counter = (uint64_t *) std::aligned_alloc(64, sizeof(uint64_t)*n);


    //compute_mask_counter_float_matrix_avx(mask_aligned, k, n, mask_counter);

    vector<uint64_t> elapsed_times;
    for (size_t run=0; run < n_run; run ++){

        auto start = std::chrono::high_resolution_clock::now();
        compute_mask_counter_float_matrix_avx(mask_aligned, k, n, mask_counter);
        // for (size_t i=0; i< k; i+=64){
        //     for (size_t j=0; j<n; j++){
        //         for (size_t p=0; p < 64; p++) {
        //             B_packed[(i / 64) * n*64 + p + j*64] = B[i * n + p * n + j];
        //         }
        //     }
        // }

        for (size_t x = 0; x < m; x += mr){
            for (size_t y = 0; y < n; y += nr){
                kernel(a, alpha_aligned, mask_aligned, beta_aligned, c, m, k, n, x, y, mask_counter);   
            }
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
        elapsed_times.push_back(elapsed);
    }

  
    for (size_t i=0; i<m; i++){
        for(size_t j=0;j< n; j++ ){
            C[i*n +j ] = ((float)c[i*n + j]) * s/2;   
        }
    } 
          
    std::free(a);
    std::free(alpha_aligned);
    std::free(mask_aligned);
    std::free(beta_aligned);
    std::free(c);

   return elapsed_times;

}


template<auto kernel>
vector<uint64_t> measure_time_2_2_kernel_64_bit(const vector<float> &A, const vector<float> &B, vector<float> &C, const size_t m, const size_t k, const size_t n, const float s, const size_t mr, size_t const nr, const size_t n_run){
     auto [alpha_a, mask_a, beta_a] = generate_2bits_masks(A, s);

    vector<int64_t> integer_C(C.size());
    std::transform(C.begin(), C.end(), integer_C.begin(), [](float x) { return (int64_t)x;});
    auto B_packed = pack_matrix_for_binary_mul(B, k, n);

    auto [alpha_b, mask_b, beta_b] = generate_2bits_masks(B_packed, s);

    //Aligning A 
    uint64_t * alpha_a_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(alpha_a.size()));
    memcpy(alpha_a_aligned, alpha_a.data(), sizeof(uint64_t)*(alpha_a.size()));

    uint64_t * mask_a_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(mask_a.size()));
    memcpy(mask_a_aligned, mask_a.data(), sizeof(uint64_t)*(mask_a.size()));

    uint64_t * beta_a_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(beta_a.size()));
    memcpy(beta_a_aligned, beta_a.data(), sizeof(uint64_t)*(beta_a.size()));


    //Aligning B vectors
    uint64_t * alpha_b_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(alpha_b.size()));
    memcpy(alpha_b_aligned, alpha_b.data(), sizeof(uint64_t)*(alpha_b.size()));

    uint64_t *mask_b_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(mask_b.size()));
    memcpy(mask_b_aligned, mask_b.data(), sizeof(uint64_t)*(mask_b.size()));

    uint64_t *beta_b_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(beta_b.size()));
    memcpy(beta_b_aligned, beta_b.data(), sizeof(uint64_t)*(beta_b.size()));
    vector<uint64_t> elapsed_times;

    int64_t *c = (int64_t * ) std::aligned_alloc(64, sizeof(int64_t)*(integer_C.size()));
    memcpy(c, integer_C.data(),  sizeof(int64_t)*(integer_C.size()));

    for (size_t run=0; run < n_run; run ++){

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t x = 0; x < m; x += mr){
            for (size_t y = 0; y < n; y += nr){
                kernel(alpha_a_aligned, beta_a_aligned, mask_a_aligned, alpha_b_aligned, beta_b_aligned, mask_b_aligned, c, m, k, n, x, y);   
            }
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
        elapsed_times.push_back(elapsed);
    }

  
    for (size_t i=0; i<m; i++){
        for(size_t j=0;j< n; j++ ){
            //cout<<c[i*n + j]<< "\n";
            C[i*n +j ] = ((float)c[i*n + j]) * s*s/4;   
            //cout<<C[i*n +j ]<< "\n";
        }
    } 
          
    std::free(alpha_a_aligned);
    std::free(beta_a_aligned);
    std::free(mask_a_aligned);

    std::free(alpha_b_aligned);
    std::free(beta_b_aligned);
    std::free(mask_b_aligned);


   return elapsed_times;

}



template<auto routine>
vector<uint64_t> measure_time_2_2_transposed(const vector<float> &A, const vector<float> &B, vector<float> &C, const size_t m, const size_t k, const size_t n, const float s, const size_t n_run){


        auto [alpha_a, mask_a, beta_a] = generate_2bits_masks(A, s);
        auto B_transposed = transpose_matrix(B, k, n);
        auto [alpha_b, mask_b, beta_b] = generate_2bits_masks(B_transposed, s);


        //Aligning A 
        uint64_t * alpha_a_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(alpha_a.size()));
        memcpy(alpha_a_aligned, alpha_a.data(), sizeof(uint64_t)*(alpha_a.size()));

        uint64_t * mask_a_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(mask_a.size()));
        memcpy(mask_a_aligned, mask_a.data(), sizeof(uint64_t)*(mask_a.size()));

        uint64_t * beta_a_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(beta_a.size()));
        memcpy(beta_a_aligned, beta_a.data(), sizeof(uint64_t)*(beta_a.size()));


        //Aligning B vectors
        uint64_t * alpha_b_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(alpha_b.size()));
        memcpy(alpha_b_aligned, alpha_b.data(), sizeof(uint64_t)*(alpha_b.size()));

        uint64_t *mask_b_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(mask_b.size()));
        memcpy(mask_b_aligned, mask_b.data(), sizeof(uint64_t)*(mask_b.size()));

        uint64_t *beta_b_aligned = (uint64_t * ) std::aligned_alloc(64, sizeof(uint64_t)*(beta_b.size()));
        memcpy(beta_b_aligned, beta_b.data(), sizeof(uint64_t)*(beta_b.size()));
        vector<uint64_t> elapsed_times;


        for (size_t run=0; run < n_run; run ++){
            auto start = std::chrono::high_resolution_clock::now();
            routine(alpha_a_aligned, beta_a_aligned, mask_a_aligned, alpha_b_aligned, beta_b_aligned, mask_b_aligned, C.data(), m, k, n, s);
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
            elapsed_times.push_back(elapsed);
        }
        

        std::free(alpha_a_aligned);
        std::free(beta_a_aligned);
        std::free(mask_a_aligned);

        std::free(alpha_b_aligned);
        std::free(beta_b_aligned);
        std::free(mask_b_aligned);

   return elapsed_times;

}