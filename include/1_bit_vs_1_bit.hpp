#pragma once
#include <cstring>
#include <immintrin.h>
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



//
//uint64_t lower_qword(const __m128i v) {
//
//    return _mm_cvtsi128_si64(v);
//}
//
//
//uint64_t higher_qword(const __m128i v) {
//
//    return lower_qword(_mm_srli_si128(v, 8));
//}
//
//
////todo maybe c can be store using less bits? i.e. 32.
//inline void mult_bin_sse128(
//        //const __m512i *a, const __m512i *b, int64_t *c,
//        const vec512_64 * a, const vec512_64 *b, int64_t * c,
//                            size_t m,
//                            size_t k,
//                            size_t n
////                            size_t x, size_t y,
////                            size_t l, // not used
////                            size_t r, // not used
////                            size_t n, size_t k_pad
//                            ) {
//    uint64_t L = 512;
//
//    //__m256i sum{};
//    __m128i sum{};
//    sum = _mm_setzero_si128();
//    __m512i x{};
//    size_t k_lim = k / L;
//    for(size_t i = 0; i < m; ++i) {
//
//        for (size_t j=0; j< n; ++j){
//            for (size_t p =0; p < k_lim; ++p){
//                __m512i at = _mm512_load_si512((__m512i *) &(a[i*k_lim + p]));
//                __m512i bt = _mm512_load_si512((__m512i *) &(b[j*k_lim + p]));
//                //auto x = _mm512_xor_si512(a[i*k_lim + p], b[j*k_lim + p]);
//                x = _mm512_xor_si512(at, bt);
////
////                __m256i l = _mm256_popcnt_epi16(_mm512_extracti64x4_epi64(x, 0));
////                __m256i h = _mm256_popcnt_epi16(_mm512_extracti64x4_epi64(x, 1));
//                //sum = _mm256_add_epi64(l, h);
//                __m128i l0 = _mm512_extracti64x2_epi64(x, 0);
//                sum+= _popcnt64(lower_qword(l0));
//                sum+= _popcnt64(higher_qword(l0));
//                __m128i l1 = _mm512_extracti64x2_epi64(x, 1);
//                sum+= _popcnt64(lower_qword(l1));
//                sum+= _popcnt64(higher_qword(l1));
//                __m128i l2 = _mm512_extracti64x2_epi64(x, 2);
//                sum+= _popcnt64(lower_qword(l2));
//                sum+= _popcnt64(higher_qword(l2));
//                __m128i l3 = _mm512_extracti64x2_epi64(x, 3);
//                sum+= _popcnt64(lower_qword(l3));
//                sum+= _popcnt64(higher_qword(l3));
//
//            }
//            cout<<"Sum "<< hsum(sum)<< "\n";
//            c[i*n + j]+= (int64_t) (L - 2*hsum(sum));
//            cout<<"pippo "<<(int64_t) (L - 2*hsum(sum))<<"\n";
//            sum = _mm_setzero_si128();
//
//        }
//    }
//}



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

/*

Instruction: 
Name of instruction. Multiple names mean that these instructions have the same data. 
Instructions with or without V name prefix behave the same unless otherwise noted.


Operands:
i = immediate data, r = register, mm = 64 bit mmx register, x = 128 bit xmm register, y = 256 bit ymm 
register, z = 512 bit zmm register, xy = xmm or ymm register, v = any vector register (mmx, xmm, ymm, zmm).
m = memory operand, m32 = 32-bit memory operand, etc.


μops each port:
The number of μops for each execution port. p0 means a μop to execution port 0. p01means a μop that can go to either port 0 or port 1. p0 p1 means two μops going to port 0 and 1, respectively.
Port 0: Integer, f.p. and 256 bit vector ALU, mul, div, branch
Port 1: Integer, f.p. and 256 bit vector ALU (re-routed to port 0 for 512-bit vectors)
Port 23: Load
Port 49: Store
Port 78: Store address
Port 5: Integer and 512 bit vector ALU
Port 6: Integer ALU, branch


*/


/* _mm512_load_si512 Instruction: vmovdqa32 zmm, m512
   _mm512_set1_epi64 Instruction vpbroadcastq 
   _mm512_xor_si512 Instruction: vpxord zmm, zmm, zmm
   _mm512_popcnt_epi64 Instruction: vpopcntq zmm, zmm
   _mm512_add_epi64 Instruction: vpaddq zmm, zmm, zmm -> ATTENZIONE, vpaddq non trovata, trovato PADD/SUB(S,US) B/W/D/Q
   _mm512_sllv_epi64 Instruction: vpsllvq zmm, zmm, zmm
   _mm512_store_epi64 Instruction: vmovdqa64 m512, zmm



   Instruction      operands        port        latency (fog/intel)         tp (fog/intel)       
   VMOVDQA/U/32/64      z,m         2,3          4                           0.5 
VPBROADCAST B/W/D/Q     v,m         2,3          ? / 3 Skylake               0.5 / 1 Skylake        
VPAND/ANDN/OR/ XOR D/Q  z,z,z       0,5         1 / 1                       0.5 / 0.5
    VPOPCNTB/W/D/Q      v,v         5           3 / ?                       1 / ?
PADD/SUB(S,US)B/W/D/Q   mm,mm       0,5         1 / 1                       0.5 / 0.5
VPSLLW/D/Q              ?           ?           ? / 1                       ? / 1
VMOVDQA/U/32/64         z,m         2,3          4 / 5                       0.5 / 1
VMOVDQA/U/32/64         m,z         49,78       3 / 5                       0.5 / 1




*/
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

inline void kernel_bin_2x8(
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
    __m512i t[2] {}; // accumulators for C
    __m512i alpha, beta;

    for (size_t p=0; p<k; p++){ 
        beta = _mm512_load_si512((const void *)&(b[(p * n) + y])); // 512 values of B are packed here
        for (size_t i = 0; i < 2; i++) {
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

    for (size_t i = 0; i < 2; i++){
        t[i] =  _mm512_sub_epi64(scaling_factor,_mm512_sllv_epi64(t[i], shift));
        //t[i] = _mm512_add_epi64(scaling_factor, -2*t[i]);
        _mm512_store_epi64((void *) &(c[(x + i) * n + y]), t[i]);
    
    }

}



inline void kernel_bin_8x16(
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
    __m512i t[8][2] {}; // accumulators for C
    __m512i alpha, beta, beta1;
    for (size_t p=0; p<k; p++){ 
        beta = _mm512_load_si512((const void *)&(b[(p * n) + y])); // 512 values of B are packed here
        beta1 = _mm512_load_si512((const void *)&(b[(p * n) + y + 8])); // 512 values of B are packed here
        for (size_t i = 0; i < 8; i++) {
            //Broadcast a
            alpha = _mm512_set1_epi64(a[(i+x) * k + p]);
            t[i][0] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(alpha, beta)), t[i][0]); //accumulate in c
            t[i][1] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(alpha, beta1)), t[i][1]); //accumulate in c

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
        t[i][0] =  _mm512_sub_epi64(scaling_factor,_mm512_sllv_epi64(t[i][0], shift));
        t[i][1] =  _mm512_sub_epi64(scaling_factor,_mm512_sllv_epi64(t[i][1], shift));

        //t[i] = _mm512_add_epi64(scaling_factor, -2*t[i]);
        _mm512_store_epi64((void *) &(c[(x + i) * n + y]), t[i][0]);
        _mm512_store_epi64((void *) &(c[(x + i) * n + y + 8]), t[i][1]);
 
    
    }

}

inline void kernel_bin_6x16(
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
    __m512i t[6][2] {}; // accumulators for C
    __m512i alpha, beta, beta1;
    for (size_t p=0; p<k; p++){ 
        beta = _mm512_load_si512((const void *)&(b[(p * n) + y])); // 512 values of B are packed here
        beta1 = _mm512_load_si512((const void *)&(b[(p * n) + y + 8])); // 512 values of B are packed here
        for (size_t i = 0; i < 6; i++) {
            //Broadcast a
            alpha = _mm512_set1_epi64(a[(i+x) * k + p]);
            t[i][0] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(alpha, beta)), t[i][0]); //accumulate in c
            t[i][1] = _mm512_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(alpha, beta1)), t[i][1]); //accumulate in c

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

    for (size_t i = 0; i < 6; i++){
        t[i][0] =  _mm512_sub_epi64(scaling_factor,_mm512_sllv_epi64(t[i][0], shift));
        t[i][1] =  _mm512_sub_epi64(scaling_factor,_mm512_sllv_epi64(t[i][1], shift));

        //t[i] = _mm512_add_epi64(scaling_factor, -2*t[i]);
        _mm512_store_epi64((void *) &(c[(x + i) * n + y]), t[i][0]);
        _mm512_store_epi64((void *) &(c[(x + i) * n + y + 8]), t[i][1]);
 
    
    }

}

inline void kernel_bin_8x8_16bit(
        const uint16_t *a, const uint16_t *b, 
        int16_t * c,
        size_t m,
        size_t k,
        size_t n, 
        size_t x, 
        size_t y
) {
    // m true numbers of row of A
    // k numbers of cols of A divided by 16 due to bit packing == numbers of row of B (due to reshaping)
    // n true numbers of cols of B

    int16_t L = 512;
    __m512i t[8] {}; // accumulators for C
    __m512i alpha, beta;

    for (size_t p=0; p<k; p++){ 
        beta = _mm512_load_si512((const void *)&(b[(p * n) + y])); // 512 values of B are packed here
        for (size_t i = 0; i < 8; i++) {
            //Broadcast a
            alpha = _mm512_set1_epi16(a[(i+x) * k + p]);
            t[i] = _mm512_add_epi16(_mm512_popcnt_epi16(_mm512_xor_si512(alpha, beta)), t[i]); //accumulate in c
        }
    }


    //todo this (scaling factor) holds as long as k >= 512 and k % 512 ==0
    __m512i scaling_factor = _mm512_set1_epi16(L * k / 32 );
    __m512i shift = _mm512_set1_epi16(1);

    /*
    _mm512_sub_epi64 : latency 1, throughput: 0.5 (Icelake)

    */

    for (size_t i = 0; i < 8; i++){
        t[i] =  _mm512_sub_epi16(scaling_factor,_mm512_sllv_epi16(t[i], shift));
        //t[i] = _mm512_add_epi64(scaling_factor, -2*t[i]);
        _mm512_storeu_epi16((void *) &(c[(x + i) * n + y]), t[i]);
    
    }

}




inline void kernel_bin_8x16_16bit(
        const uint16_t *a, const uint16_t *b, 
        int16_t * c,
        size_t m,
        size_t k,
        size_t n, 
        size_t x, 
        size_t y
) {
    // m true numbers of row of A
    // k numbers of cols of A divided by 16 due to bit packing == numbers of row of B (due to reshaping)
    // n true numbers of cols of B

    int16_t L = 512;
    __m512i t[8][2] {}; // accumulators for C
    __m512i alpha, beta, beta1;

    for (size_t p=0; p<k; p++){ 
        beta = _mm512_load_si512((const void *)&(b[(p * n) + y])); // 512 values of B are packed here
        beta1 = _mm512_load_si512((const void *)&(b[(p * n) + y + 32])); // 512 values of B are packed here

        for (size_t i = 0; i < 8; i++) {
            //Broadcast a
            alpha = _mm512_set1_epi16(a[(i+x) * k + p]);
            t[i][0] = _mm512_add_epi16(_mm512_popcnt_epi16(_mm512_xor_si512(alpha, beta)), t[i][0]); //accumulate in c
            t[i][1] = _mm512_add_epi16(_mm512_popcnt_epi16(_mm512_xor_si512(alpha, beta1)), t[i][1]); //accumulate in c

        }

    }


    //todo this (scaling factor) holds as long as k >= 512 and k % 512 ==0
    __m512i scaling_factor = _mm512_set1_epi16(L * k / 32 );
    __m512i shift = _mm512_set1_epi16(1);

    /*
    _mm512_sub_epi64 : latency 1, throughput: 0.5 (Icelake)

    */

    for (size_t i = 0; i < 8; i++){
        t[i][0] =  _mm512_sub_epi16(scaling_factor,_mm512_sllv_epi16(t[i][0], shift));
        t[i][1] = _mm512_sub_epi16(scaling_factor,_mm512_sllv_epi16(t[i][1], shift));
        //t[i] = _mm512_add_epi64(scaling_factor, -2*t[i]);
        _mm512_storeu_epi16((void *) &(c[(x + i) * n + y]), t[i][0]);
        _mm512_storeu_epi16((void *) &(c[(x + i) * n + y + 32]), t[i][1]);

    
    }

}

