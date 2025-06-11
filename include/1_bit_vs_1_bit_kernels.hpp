#pragma once
#include <cstring>
#include <immintrin.h>

#include <cstdint>

typedef uint64_t vec512_64 __attribute__ (( vector_size(64) ));
typedef int64_t vec512_64_uns __attribute__ (( vector_size(64) ));


vec512_64 * alloc512_64i(size_t n){
    vec512_64 *ptr = (vec512_64*) std::aligned_alloc(64, 8*n);
    memset(ptr, 0, 8 * n);
    return ptr;
}

vec512_64_uns * alloc512_64i_uns(size_t n){
    vec512_64_uns *ptr = (vec512_64_uns*) std::aligned_alloc(64, 8*n);
    memset(ptr, 0, 8 * n);
    return ptr;
}


inline float hsum(__m128 x) {
    x = _mm_hadd_ps(x, x);
    auto t = (float*) &x;
    return t[0]+t[1]; // FIXME: use extract should be faster but doesn't do what I expect.
    //return _mm_extract_ps(l, 0) + _mm_extract_ps(l, 1);
}

inline float hsum(__m256 x) {
    __m128 l = _mm256_extractf128_ps(x, 0);
    __m128 h = _mm256_extractf128_ps(x, 1);
    l = _mm_add_ps(l, h);
    return hsum(l);
}

inline float hsum(__m512 x) {
    __m256 l = _mm512_extractf32x8_ps(x, 0);
    __m256 h = _mm512_extractf32x8_ps(x, 1);
    l = _mm256_add_ps(l, h);
    return hsum(l);
}

inline int64_t hsum(__m128i x) {
    //x = _mm_hadd_epi64(x, x);
    auto t = (int64_t*) &x;
    return t[0]+t[1]; // FIXME: use extract should be faster but doesn't do what I expect.
    //return _mm_extract_ps(l, 0) + _mm_extract_ps(l, 1);
}

inline int64_t hsum(__m256i x) {
    __m128i l = _mm256_extracti64x2_epi64(x, 0);
    __m128i h = _mm256_extracti64x2_epi64(x, 1);
    l = _mm_add_epi64(l, h);
    return hsum(l);
}

inline int64_t hsum(__m512i x) {
    __m256i l = _mm512_extracti64x4_epi64(x, 0);
    __m256i h = _mm512_extracti64x4_epi64(x, 1);
    l = _mm256_add_epi64(l, h);
    return hsum(l);
}




inline void mult_bin_avx512_transposed(
        const uint64_t *a, const uint64_t *b, int64_t *c,
        //const vec512_64 * a, const vec512_64 *b, int64_t * c,
        size_t m,
        size_t k,
        size_t n
) {
    int64_t L = 512;
    int scaling_factor = (int) (L * k /8);
    __m512i sum{};
    __m512i alpha, beta;
    for(size_t i = 0; i < m; ++i) {
        for (size_t j=0; j< n; ++j){
            sum = _mm512_setzero_si512();
            for (size_t p =0; p < k; p+=8){
                alpha = _mm512_load_si512((const void *) &(a[i*k + p]));
                beta = _mm512_load_si512((const void *) &(b[j*k + p]));
                sum = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(alpha, beta)), sum);
            }
         c[i*n + j]+= (int64_t) (scaling_factor - 2*hsum(sum));

        }
    }
}


inline void kernel_bin_8x8(
        const uint64_t *a, const uint64_t *b, 
        int64_t * c,
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
    __m512i t[8] {}; // accumulators for C
    __m512i alpha, beta;

    for (size_t p=0; p<k; p++){ 
        beta = _mm512_load_si512((const void *)&(b[(p * n) + y])); // 512 values of B are packed here
        for (size_t i = 0; i < 8; i++) {
            //Broadcast a
            alpha = _mm512_set1_epi64(a[(i+x) * k + p]);
            t[i] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(alpha, beta)), t[i]); //accumulate in c
        }
    }

    //todo this (scaling factor) holds as long as k >= 512 and k % 512 ==0
    __m512i scaling_factor = _mm512_set1_epi64((int)L * k / 8 );
    
    /* there are two ways to apply shifting:
    _mm_set_epi64 + _mm512_sll_epi64 
    _mm512_set1_epi64 + _mm512_sllv_epi64 

    _mm_set_epi64 : ?
    _mm512_sll_epi64 : latency: unknown, throughput: 1 (Icelake)

    _mm512_set1_epi64 : latency: 3, throughput: 1 (Skylake)
    _mm512_sllv_epi64 : latency: 1, throughput: 1 (Icelake)

    */  
    __m512i shift = _mm512_set1_epi64((uint64_t)1);

    /*
    _mm512_sub_epi64 : latency 1, throughput: 0.5 (Icelake)

    */

    for (size_t i = 0; i < 8; i++){
        t[i] =  _mm512_sub_epi64(scaling_factor,_mm512_sllv_epi64(t[i], shift));
        //t[i] = _mm512_add_epi64(scaling_factor, -2*t[i]);
        _mm512_store_epi64((void *) &(c[(x + i) * n + y]), t[i]);
    
    }

}









