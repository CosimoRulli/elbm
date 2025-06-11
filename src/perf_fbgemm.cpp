#include <iostream>
#include <vector>

#include "parser.hpp"
#include "tuple"
#include "utils.hpp"
#include "fbgemm/Fbgemm.h"


using namespace fbgemm;
// Partially taken from https://gist.github.com/ppetrushkov/694bc7ec0f7663c63e067e9ecfdc7d99

enum class Method {
    standard, 
    packed  
};

// vector<uint64_t> measure_time_dense_fbgemm_int8(const vector<uint8_t> &A, const vector<int8_t> &B, const size_t M, const size_t K, const size_t N, vector<int32_t> & C, const size_t n_run){


//     vector<int32_t> col_offsets(N, 0);

//     //auto packedBN = fbgemm::PackBMatrix<int8_t, int32_t>(fbgemm::matrix_op_t::NoTranspose, K, N, B.data(), N);

//     float input_scale = 1.0f/255;
//     int32_t input_zeropoint = 0;

//     vector<float> weight_scales(N, 0);

//     for (int i=0; i<weight_scales.size(); ++i)
//         weight_scales[i] = 1.0f/255;
    
//     vector<int32_t> weight_zeropoints(N, 0);


//    vector<uint64_t> elapsed_times;

//    //warmup 
    
//     fbgemm::DoNothing<float, float> doNothingObj{};

 
//    for (size_t run=0; run < n_run; run ++){
//        auto start = std::chrono::high_resolution_clock::now();
//         fbgemm::matmul_u8i8acc32_ref(
//         M, N, K, K, N, N, A.data(), B.data(), C.data());
//     //      fbgemm::PackAWithRowOffset<uint8_t,int32_t> packAN(
//     //                   fbgemm::matrix_op_t::NoTranspose,
//     //                   M,
//     //                   N,
//     //                   A.data(),
//     //                   K, //lda==k for no-transpose
//     //                   nullptr, //buffer for packed matrix
//     //                   1, //groups
//     //                   nullptr); //buffer for packed data
//     //     fbgemm::ReQuantizeForFloat<false, fbgemm::QuantizationGranularity::OUT_CHANNEL> outputProcObj(
//     //                   doNothingObj,
//     //                   input_scale,
//     //                   weight_scales.data(),
//     //                   input_zeropoint,
//     //                   weight_zeropoints.data(),
//     //                   packAN.getRowOffsetBuffer(), //row offsets
//     //                   col_offsets.data(), //column offsets
//     //                   nullptr,
//     //                   N);
//     //     fbgemm::fbgemmPacked(
//     //         packAN,
//     //         packedBN,
//     //         C.data(),
//     //         (int32_t*)C.data(),
//     //         (int32_t)N, //ldc==n
//     //         outputProcObj,
//     //         0, //thread id
//     //         1); //num threads
//         auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
//        elapsed_times.push_back(elapsed);
//    }


//    return elapsed_times;

// }


vector<uint64_t> measure_time_dense_fbgemm_int8_packed(const vector<uint8_t> &A, const vector<int8_t> &B, const size_t M, const size_t K, const size_t N, vector<int32_t> & C, const size_t n_run){


    vector<int32_t> col_offsets(N, 0);

    //auto packedBN = fbgemm::PackBMatrix<int8_t, int32_t>(fbgemm::matrix_op_t::NoTranspose, K, N, B.data(), N);

    float input_scale = 1.0f/255;
    int32_t input_zeropoint = 0;

    vector<float> weight_scales(N, 0);

    for (int i=0; i<weight_scales.size(); ++i)
        weight_scales[i] = 1.0f/255;
    
    vector<int32_t> weight_zeropoints(N, 0);


   vector<uint64_t> elapsed_times;

   //warmup 
    
    
    fbgemm::DoNothing<int32_t, int32_t> doNothing32BitObj;
    fbgemm::memCopy<> memcopyObj(doNothing32BitObj);
    PackAMatrix<uint8_t> packA_int32(
            matrix_op_t::NoTranspose, M, K, A.data(), K, nullptr, 1);

    int num_threads = 1;
    int tid = 0;
   for (size_t run=0; run < n_run; run ++){
       auto start = std::chrono::high_resolution_clock::now();
        PackBMatrix<int8_t> packedB_int32(
        fbgemm::matrix_op_t::NoTranspose, K, N, B.data(), N, nullptr, 1);
        
        // printf ( "tid: %d, num_threads: %d\n", tid, num_threads );
        fbgemmPacked(
            packA_int32,
            packedB_int32,
            C.data(),
            C.data(),
            N,
            memcopyObj,
            tid,
            num_threads);
      

        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
       elapsed_times.push_back(elapsed);
   }


   return elapsed_times;

}




template<typename T>
void init_random(int size, T* ptr) {
    for (int i=0; i<size; ++i)
        ptr[i] = static_cast<T>(127.0f*static_cast<float>(rand()) / static_cast <float> (RAND_MAX /2));
}

template<typename T>
void init_zero(int size, T* ptr) {
    for (int i=0; i<size; ++i)
        ptr[i] = static_cast<T>(0);
}




void argparse(cmd_line_parser::parser& parser) {
    // here, we specify some shorthand for *optional* arguments
    parser.add("M",       // name
               "Number of rows of matrix A.",  // description
               "-m",                    // shorthand
               true, // required
               false                    // not boolean option: expected a value after the shorthand
    );

    parser.add("K",       // name
               "Number of cols of matrix A.",  // description
               "-k",                    // shorthand
               true, // required

               false                    // not boolean option: expected a value after the shorthand
    );
    parser.add("N",       // name
               "Number of cols of matrix B.",  // description
               "-n",                    // shorthand
                true, // required

               false                    // not boolean option: expected a value after the shorthand
    );

    parser.add("Check",       // name
               "Check the result",  // description
               "-c",                    // shorthand
               false,               // boolean option: expected no value after the shorthand
               true   
    );

}



void timing_print(std::string name, std::vector<uint64_t> timing, size_t m, size_t k, size_t n) {
    auto sep = '\t';
    
    std::cerr << name << sep;
    std::cerr << m << sep << n << sep << k << sep;
    std::cerr << *std::min_element(timing.begin(), timing.end()) << sep;
    auto sum = std::accumulate(timing.begin(), timing.end(), 0.0);
    std::cerr << sum / timing.size() << sep;
    std::cerr << endl;
}



int main(int argc, char**argv) {

    size_t n_run = 1000;
    const float toll = 0.01;
    
    cmd_line_parser::parser parser(argc, argv);

    argparse(parser);
    if (!parser.parse()) return 1;
    auto M = parser.get<size_t>("M");;
    auto K = parser.get<size_t>("K");;
    auto N = parser.get<size_t>("N");;

    cout<< "M " << M << "\n";
    cout<< "K " << K << "\n";
    cout<< "N " << N << "\n";
  
    bool check = parser.get<bool>("Check");
    if (check) n_run=1;

    vector<uint8_t> A(M*K);
    init_random(M*K, A.data());


    vector<int8_t> B(K*N);
    init_random(K*N, B.data());

    vector<int32_t> C(M*N, 0.);
    
    vector<uint64_t> elapsed_times;
    std::string method_name = "";

    bool output_packed = false;

    size_t mr, nr;



    method_name = std::string("FBGEMM-8bits");
    elapsed_times = measure_time_dense_fbgemm_int8_packed(A, B, M, K, N, C, n_run);



    timing_print(method_name, elapsed_times, M, N, K);

    if(check) {
        std::cout << "Testing correctness...\n";
        vector<float> C_truth(M*N);
        
        vector<float> float_A (A.size());
        std::transform(A.begin(), A.end(), float_A.begin(), [](uint8_t x) { return (float)x;});

        vector<float> float_B(B.size());
        std::transform(B.begin(), B.end(), float_B.begin(), [](int8_t x) { return (float)x;});

        trivial_dense_multiplication(float_A, float_B, M, K, N, C_truth);
        //print_mat(C_truth, M, N);
        // if (output_packed){
        //     cout<<"Packing C_truth matrix for comparison\n";
        //     C_truth = pack_matrix_for_binary_mul(C_truth, M, N);
        // }

        auto count = check_equality(C_truth, C, M, N, toll);
        // print_mat(C_truth, M, N);
        // std::cout<<"\n";
        // print_mat(C, M, N);
        if(count != 0) {
            std::cout << "ERROR: Wrong result!";
            std::cout << "Number of errors " << count << " \n";
            return 1;
        }


        std::cout << "Done!\n";
    }

    return 0;


}
