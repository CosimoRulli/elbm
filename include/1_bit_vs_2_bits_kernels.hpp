#pragma once
#include <cstring>
//#include "simd.hpp"
#include <immintrin.h>
#include <1_bit_vs_1_bit_kernels.hpp>

inline void mult_2bits_avx512_transposed(
        const uint64_t *a, const uint64_t *alphas,const uint64_t *masks, 
        const uint64_t *betas,
        float *c,
        //const vec512_64 * a, const vec512_64 *b, int64_t * c,
        size_t m,
        size_t k,
        size_t n, 
        float s
) {

    float L = 512;
    k = k /64;
    __m512i xor1{}, xor2{}, sum1{}, sum2{}, diff1{}, diff2{}, mask_counter{};
    __m512i w, alpha, mask, beta;
    float alpha_scale = 1.5*s;
    float beta_scale = 0.5*s;
    float mask_counter_sum = 0;
    
    for(size_t i = 0; i < m; ++i) {
        for (size_t j=0; j< n; ++j){
            sum1 = _mm512_setzero_si512();
            sum2 = _mm512_setzero_si512();
            mask_counter = _mm512_setzero_si512();
        
        
            for (size_t p =0; p < k; p+=8){
                
                w = _mm512_load_si512((const void *) &(a[i*k + p]));

                alpha = _mm512_load_si512((const void *) &(alphas[j*k + p]));
                mask = _mm512_load_si512((const void *) &(masks[j*k + p]));
                beta = _mm512_load_si512((const void *) &(betas[j*k + p]));

                xor1 = _mm512_xor_si512(w, alpha);
                sum1 = _mm512_add_epi64(_mm512_popcnt_epi64(xor1), sum1);
                diff1 = _mm512_popcnt_epi64(_mm512_andnot_si512(mask, xor1));
                sum1 = _mm512_sub_epi64(sum1, diff1);
                
                xor2 = _mm512_xor_si512(w, beta);
                sum2 = _mm512_add_epi64(_mm512_popcnt_epi64(xor2), sum2);
                diff2 = _mm512_popcnt_epi64(_mm512_and_si512(mask, xor2));
                sum2 = _mm512_sub_epi64(sum2, diff2);

                mask_counter = _mm512_add_epi64(_mm512_popcnt_epi64(mask), mask_counter);
                // xor1 = _mm512_xor_si512(w, alpha);
                // //sum1 = _mm512_add_epi64(_mm512_popcnt_epi64(xor), sum1);
                // sum1 = _mm512_popcnt_epi64(xor1);
                // //diff1 = _mm512_add_epi64(_mm512_andnot_si512(mask, xor1), diff1);
                // diff1 = _mm512_popcnt_epi64(_mm512_andnot_si512(mask, xor1));

                // xor2 = _mm512_xor_si512(w, beta);
                // sum2 = _mm512_popcnt_epi64(xor2);

                // diff2 = _mm512_popcnt_epi64(_mm512_and_si512(mask, xor2));

                // //sum2 = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(w, m3)), sum2);
                
                // float alpha_sum = 3*s/2 * (float)(L - (L - hsum(_mm512_popcnt_epi64(mask))) - 2* (hsum(sum1) - hsum(diff1)));

                // float beta_sum = s/2 * (float) (L - hsum(_mm512_popcnt_epi64(mask)) - 2* (hsum(sum2) - hsum(diff2)));

                //c[i*n + j]+=  beta_sum;

            }
            mask_counter_sum = (float) (hsum(mask_counter));

            c[i*n + j]+=  alpha_scale * (float) (mask_counter_sum -2 * (hsum(sum1))) + beta_scale*(float) (L - mask_counter_sum - 2*hsum(sum2));


            
        }
    }
}


inline void mult_2bits_avx512_transposed_ternary(
        const uint64_t *a, const uint64_t *alphas,const uint64_t *masks, 
        const uint64_t *betas,
        float *c,
        //const vec512_64 * a, const vec512_64 *b, int64_t * c,
        size_t m,
        size_t k,
        size_t n, 
        float s
) {

    float L = 512;
    k = k /64;
    __m512i sum1{}, sum2{};
    __m512i w, alpha, mask, beta;
    float alpha_scale = 1.5*s;
    float beta_scale = 0.5*s;
    float mask_counter_sum = 0;

    //compute mask_counters
    __m512i acc{}, val{};
    float mask_counters [n];
    for (size_t i=0; i<n; i++){
        acc = _mm512_setzero_si512();
        for (size_t j=0; j< k; j+=8){
            val = _mm512_load_si512((const void *) &(masks[i*k + j]));
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(val));
        }
        mask_counters[i] = hsum(acc);
    }
    
    for(size_t i = 0; i < m; ++i) {
        for (size_t j=0; j< n; ++j){
            sum1 = _mm512_setzero_si512();
            sum2 = _mm512_setzero_si512();
            //mask_counter = _mm512_setzero_si512();
        
        
            for (size_t p =0; p < k; p+=8){
                
                w = _mm512_load_si512((const void *) &(a[i*k + p]));

                alpha = _mm512_load_si512((const void *) &(alphas[j*k + p]));
                mask = _mm512_load_si512((const void *) &(masks[j*k + p]));
                beta = _mm512_load_si512((const void *) &(betas[j*k + p]));
                
                //00101000 -> 0x28
                sum1 = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(w, alpha, mask, 0x28)), sum1);

                //00010100 -> 0x14
                sum2 = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(w, beta, mask, 0x14)), sum2);

                //mask_counter = _mm512_add_epi64(_mm512_popcnt_epi64(mask), mask_counter);

            }
//            mask_counter_sum = (float) (hsum(mask_counter));
            mask_counter_sum = mask_counters[j];

            c[i*n + j]+=  alpha_scale * (float) (mask_counter_sum -2 * (hsum(sum1))) + beta_scale*(float) (L - mask_counter_sum - 2*hsum(sum2));


            
        }
    }
}


// inline void compute_mask_counter(const uint32_t* mask, const size_t k, const size_t n, float *mask_counter){
//      __m512i acc{}, val{};

//     size_t new_k = k/32;
//     size_t new_n = N * 32 ;

//     for (size_t i=0; i<new_k; i++){
//         acc = _mm512_setzero_si512();
//         for (size_t j=0; j< new_n; j+=32){
//             val = _mm512_load_si512((const void *) &(masks[i*new_n + j]));
//             acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(val));
//         }
//         mask_counters[i] = hsum(acc);
//     }
// }





inline void kernel_1_2_8x8(
        const uint64_t *a, const uint64_t *alphas, const uint64_t *masks, 
        const uint64_t *betas,
        int64_t *c,
        const size_t m,
        size_t k,
        const size_t n, 
        const size_t x, 
        const size_t y,
        const uint64_t* mask_counter

        //const float s
        
) {
    // m true numbers of row of A
    // k numbers of cols of A divided by 64 due to bit packing == numbers of row of B (due to reshaping)
    // n true numbers of cols of B

    int64_t L = 512;
    __m512i sum1[8] {}; // accumulators for alphas
    __m512i sum2[8] {};
    k = k/64;

    __m512i w, alpha, mask, beta;

    // // float alpha_scale = 1.5*s;
    // // float beta_scale = 0.5*s;
    // float mask_counter_sum = 0;

    for (size_t p=0; p<k; p++){ 
        alpha = _mm512_load_si512((const void *)&(alphas[(p * n) + y])); // 512 values of alphas are packed here
        mask = _mm512_load_si512((const void *)&(masks[(p * n) + y]));
        beta = _mm512_load_si512((const void *)&(betas[(p * n) + y]));
        for (size_t i = 0; i < 8; i++) {
            //Broadcast a
            w = _mm512_set1_epi64(a[(i+x) * k + p]); // 6 uint64_t blocks
            
            //00101000 -> 0x28
            sum1[i] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(w, alpha, mask, 0x28)), sum1[i]);

            //00010100 -> 14
            sum2[i] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(w, beta, mask, 0x14)), sum2[i]);

        }
    }

    __m512i scaling_factor = _mm512_set1_epi64((int)L * k / 8 );
    __m512i current_mask_counter = _mm512_load_si512((const void *)&(mask_counter[y])); //these are 8 mask counters, from y to y+8
    __m512i shift = _mm512_set1_epi64((uint64_t)1);
    scaling_factor = _mm512_add_epi64(scaling_factor, _mm512_sllv_epi64(current_mask_counter, shift));
    __m512i shift2 = _mm512_set1_epi64((uint64_t)2);
    //__m512i six = _mm512_set1_epi64((uint64_t)6);

    //todo pass the scaling factor 

    for (size_t i = 0; i < 8; i++){

        sum1[i] =  _mm512_sub_epi64(scaling_factor, _mm512_sllv_epi64(sum1[i], shift) + _mm512_sllv_epi64(sum1[i], shift2) + _mm512_sllv_epi64(sum2[i], shift));
        //t[i] = _mm512_add_epi64(scaling_factor, -2*t[i]);
        _mm512_store_epi64((void *) &(c[(x + i) * n + y]), sum1[i]);
    }
}



