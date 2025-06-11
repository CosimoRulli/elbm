#pragma once
#include <cstring>
//#include "simd.hpp"
#include <immintrin.h>
#include <iostream>
#include <iomanip>    

template<class T> inline void Log(const __m512i & value)
{
    const size_t n = sizeof(__m512i) / sizeof(T);
    T buffer[n];
    _mm512_storeu_si512((__m512i*)buffer, value);
    for (int i = 0; i < n; i++)
        std::cout << buffer[i] << " ";
}

inline void mult_2bits_vs_2bits_avx512_transposed(
        const uint64_t *alpha_a, const uint64_t *beta_a, const uint64_t *ma,
        const uint64_t *alpha_b, const uint64_t *beta_b, const uint64_t *mb,
        float *c,
        //const vec512_64 * a, const vec512_64 *b, int64_t * c,
        size_t m,
        size_t k,
        size_t n, 
        float s
) {

    float L = 512;
    k = k /64;

    __m512i sum1{}, sum2{}, sum3{}, sum4{};
    
    __m512i current_alpha_a, current_alpha_b, current_beta_a, current_beta_b, mask_a, mask_b;
    
    __m512i mask_counter1{}, mask_counter2{}, mask_counter3{}, mask_counter4{};
    
    float alpha_scale = 1.5*s;
    float beta_scale = 0.5*s;
    float mask_counter_sum1, mask_counter_sum2, mask_counter_sum3, mask_counter_sum4;

    //We have to compute 4 masks

    // m1: ma and mb -> alpha * alpha
    // m2: ma and not mb -> alpha * bets
    // m3: not ma and mb -> beta * alpha
    // m4: not ma and not mb -> beta * beta

    __m512i m1, m2, m3, m4;
    

    for(size_t i = 0; i < m; ++i) {

        for (size_t j=0; j< n; ++j){
            sum1 = _mm512_setzero_si512();
            sum2 = _mm512_setzero_si512();
            sum3 = _mm512_setzero_si512();
            sum4 = _mm512_setzero_si512();

            mask_counter1 = _mm512_setzero_si512();
            mask_counter2 = _mm512_setzero_si512();
            mask_counter3 = _mm512_setzero_si512();
            mask_counter4 = _mm512_setzero_si512();
        
            for (size_t p =0; p < k; p+=8){
                
                current_alpha_a = _mm512_load_si512((const void *) &(alpha_a[i*k + p]));
                current_alpha_b = _mm512_load_si512((const void *) &(alpha_b[j*k + p]));
                
                current_beta_a = _mm512_load_si512((const void *) &(beta_a[i*k + p]));
                current_beta_b = _mm512_load_si512((const void *) &(beta_b[j*k + p]));

                mask_a = _mm512_load_si512((const void *) &(ma[i*k + p]));
                mask_b = _mm512_load_si512((const void *) &(mb[j*k + p]));

                m1 = _mm512_and_si512(mask_a, mask_b); // ma and mb
                m2 = _mm512_andnot_si512(mask_b, mask_a); // not mb and ma
                m3 = _mm512_andnot_si512(mask_a, mask_b); // not ma and mb
                m4 = _mm512_or_si512(mask_a, mask_b); // not ma and not mb =>  not (ma or mb) (de Morgan)


                //00101000 -> 0x28
                sum1 = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_alpha_a, current_alpha_b, m1, 0x28)), sum1);
                
                sum2 = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_alpha_a, current_beta_b, m2, 0x28)), sum2);

                sum3 = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_beta_a, current_alpha_b, m3, 0x28)), sum3);
                
                sum4 = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_beta_a, current_beta_b, m4, 0x14)), sum4);

                //00010100 -> 14
                //sum2 = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(w, beta, mask, 0x14)), sum2);

                mask_counter1 = _mm512_add_epi64(_mm512_popcnt_epi64(m1), mask_counter1);
                mask_counter2 = _mm512_add_epi64(_mm512_popcnt_epi64(m2), mask_counter2);
                mask_counter3 = _mm512_add_epi64(_mm512_popcnt_epi64(m3), mask_counter3);
                mask_counter4 = _mm512_add_epi64(_mm512_popcnt_epi64(m4), mask_counter4);

            }
            mask_counter_sum1 = (float) (hsum(mask_counter1));
            mask_counter_sum2 = (float) (hsum(mask_counter2));
            mask_counter_sum3 = (float) (hsum(mask_counter3));
            mask_counter_sum4 = L - (float) (hsum(mask_counter4));


            c[i*n + j]+= alpha_scale * alpha_scale * (float) (mask_counter_sum1 - 2* hsum(sum1)) +
                        alpha_scale * beta_scale * (float) ( mask_counter_sum2 - 2* hsum(sum2)) +
                        beta_scale * alpha_scale * (float) ( mask_counter_sum3 - 2* hsum(sum3)) +
                        beta_scale * beta_scale * (float) ( mask_counter_sum4 - 2* hsum(sum4));
                        
        }

    }


}


inline void kernel_2_2_8x8(
       const uint64_t *alpha_a, const uint64_t *beta_a, const uint64_t *ma,
        const uint64_t *alpha_b, const uint64_t *beta_b, const uint64_t *mb,
        int64_t  *c,
        size_t m,
        size_t k,
        size_t n, 
        size_t x, 
        size_t y
        
) {
    // m true numbers of row of A
    // k numbers of cols of A divided by 64 due to bit packing == numbers of row of B (due to reshaping)
    // n true numbers of cols of B

    int64_t L = 512;
    __m512i sum1[8] {}; 
    __m512i sum2[8] {};
    __m512i sum4[8] {};

    k = k/64;

    __m512i current_alpha_a, current_alpha_b, current_beta_a, current_beta_b, mask_a, mask_b, m1, m2, m3, m4;
    __m512i mask_counter1[8]{}, mask_counter2[8]{}, mask_counter4[8]{};


    for (size_t p=0; p<k; p++){ 
        current_alpha_b = _mm512_load_si512((const void *)&(alpha_b[(p * n) + y])); 
        current_beta_b = _mm512_load_si512((const void *)&(beta_b[(p * n) + y])); 
        mask_b = _mm512_load_si512((const void *)&(mb[(p * n) + y]));

        for (size_t i = 0; i < 8; i++) {
            //Broadcast a vectors
            //cout<<"p, i "<<p<<","<<i<<"\n";
            current_alpha_a = _mm512_set1_epi64(alpha_a[(i+x) * k + p]); // 8 uint64_t blocks
            current_beta_a = _mm512_set1_epi64(beta_a[(i+x) * k + p]);
            mask_a = _mm512_set1_epi64(ma[(i+x) * k + p]);


            m1 = _mm512_and_si512(mask_a, mask_b); // ma and mb
            m2 = _mm512_andnot_si512(mask_b, mask_a); // not mb and ma
            m3 = _mm512_andnot_si512(mask_a, mask_b); // not ma and mb
            m4 = _mm512_or_si512(mask_a, mask_b); // not ma and not mb =>  not (ma or mb) (de Morgan)

            
            //00101000 -> 0x28
            sum1[i] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_alpha_a, current_alpha_b, m1, 0x28)), sum1[i]);
            
            sum2[i] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_alpha_a, current_beta_b, m2, 0x28)), sum2[i]);

            //sum3[i] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_beta_a, current_alpha_b, m3, 0x28)), sum3[i]);
            sum2[i] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_beta_a, current_alpha_b, m3, 0x28)), sum2[i]);
            
            sum4[i] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(current_beta_a, current_beta_b, m4, 0x14)), sum4[i]);

            mask_counter1[i] = _mm512_add_epi64(_mm512_popcnt_epi64(m1), mask_counter1[i] );
            mask_counter2[i] = _mm512_add_epi64(_mm512_popcnt_epi64(m2), mask_counter2[i] );
            mask_counter2[i] = _mm512_add_epi64(_mm512_popcnt_epi64(m3), mask_counter2[i] );
            //mask_counter3[i]  = _mm512_add_epi64(_mm512_popcnt_epi64(m3), mask_counter3[i] );
            mask_counter4[i] = _mm512_add_epi64(_mm512_popcnt_epi64(m4), mask_counter4[i] );

        }
    }
    //cout<<"Scaling factor "<<(int)L * k / 8<<"\n";
    __m512i scaling_factor = _mm512_set1_epi64((int)L * k / 8 );

    __m512i shift = _mm512_set1_epi64((uint64_t)1); // multiply by 2
    __m512i shift2 = _mm512_set1_epi64((uint64_t)2); // multiply by 4
    __m512i shift3 = _mm512_set1_epi64((uint64_t)3); //multiply by 8

    for (size_t i = 0; i < 8; i++){
        // ( mask_counter1 - 2* sum1 ) * 9
        sum1[i] = _mm512_sub_epi64(mask_counter1[i], _mm512_sllv_epi64(sum1[i], shift)); // sum1 =  mask_counter - 2 sum1
        sum1[i] = _mm512_add_epi64(sum1[i], _mm512_sllv_epi64(sum1[i], shift3) ); // sum1 = sum1 + 8sum1 = 9sum1

        // 3 * (mask_counter3 + mask_counter2 - 2sum2 - 2sum3)
        sum2[i] = _mm512_sllv_epi64(sum2[i], shift) ; //sum2 = 2sum2 + 2sum3
        //mask_counter2[i] = _mm512_add_epi64(mask_counter2[i], mask_counter3[i]); // mask_counter2 = mask_counter3 + mask_counter2
        sum2[i] = _mm512_sub_epi64(mask_counter2[i], sum2[i] ); // sum2 =  mask_counter2 - sum2
        sum2[i] = _mm512_add_epi64(_mm512_sllv_epi64(sum2[i], shift), sum2[i]); // sum2= 3sum2

        // L - mask_counter4 - 2*sum4
        sum4[i] = _mm512_add_epi64(_mm512_sllv_epi64(sum4[i], shift), mask_counter4[i]  ); //sum4 = 2sum4 +mask_counter4
        sum4[i] =  _mm512_sub_epi64(scaling_factor,sum4[i]); // sum4 = L - sum4

        // Total sum
        sum1[i] = _mm512_add_epi64(sum1[i], sum4[i]); //sum1  = sum1 + sum4
        sum1[i] = _mm512_add_epi64(sum1[i], sum2[i]); //sum1  = sum1 + sum2


        //_mm512_i32scatter_epi64((void *)&(c[((x+ i) / 64) * n *64 + (x+i) % 64 + (y / 8) *512 ]), vindex, sum1[i], scale );
        _mm512_store_epi64((void *) &(c[(x + i) * n + y]), sum1[i]);


    }

}