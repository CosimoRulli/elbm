
#include <iostream>
#include <vector>

#include "parser.hpp"
#include "utils.hpp"
#include "1_bit_vs_1_bit.hpp"
#include "tuple"


enum class Method {
    no_kernel, // 0
    kernel_bin_8x8, // 1
    kernel_bin_8x16, // 2
    kernel_bin_6x16, // 3
    kernel_bin_2x8, // 4
    kernel_bin_8x8_16bit, // 5
    kernel_bin_8x16_16bit, // 6
    };

void argparse(cmd_line_parser::parser& parser) {
    // here, we specify some shorthand for *optional* arguments
    parser.add("M",       // name
               "Number of rows of matrix A.",  // description
               "-m",                    // shorthand
               false                    // not boolean option: expected a value after the shorthand
    );

    parser.add("K",       // name
               "Number of cols of matrix A.",  // description
               "-k",                    // shorthand
               false                    // not boolean option: expected a value after the shorthand
    );
    parser.add("N",       // name
               "Number of cols of matrix B.",  // description
               "-n",                    // shorthand
               false                    // not boolean option: expected a value after the shorthand
    );

    parser.add("Method",       // name
               "Id of the method to test.",  // description
               "-t",                    // shorthand
               false                    // not boolean option: expected a value after the shorthand
    );

    // parser.add("Check",       // name
    //            "Check the result",  // description
    //            "-c",                    // shorthand
    //            true                    // boolean option: expected no value after the shorthand
    // );

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
    size_t n_run = 10000;
    const float toll = 0.0001;

    cmd_line_parser::parser parser(argc, argv);

    argparse(parser);
    if (!parser.parse()) return 1;

    size_t M = parser.get<size_t>("M"); 
    size_t K = parser.get<size_t>("K"); 
    size_t N = parser.get<size_t>("N"); 

    auto method_id = parser.get<uint32_t>("Method");
    Method method {method_id};
    //bool check = parser.get<bool>("Check");
    bool check = true;
    if (check) n_run=1;

    cout<< "M " << M << "\n";
    cout<< "K " << K << "\n";
    cout<< "N " << N << "\n";
    int pad = 16; 
    size_t original_M = M;
    size_t original_N = N;
    M = (M + (pad-1))/pad * pad;
    N = (N + (pad-1))/pad * pad;

    if (K < 64 || K % 64 !=0){
        std::cout<<"K must be divisible by 64\n";
        return 1;
    }

    vector<float> A = generate_bin_mat(M, K);
    auto B = generate_bin_mat(K, N);
    vector<float> C(M*N, 0.);

    vector<uint64_t> elapsed_times;
    std::string method_name = "";

    size_t mr, nr;
    switch (method) {
        case Method::no_kernel:
            method_name = std::string("no_kernel");
            elapsed_times = measure_time_bin_bin_transposed(A, B, C, M, K, N, n_run);
            break;
        case Method::kernel_bin_8x8:
            method_name = std::string("Bin Kernel 512 8x8");
            mr = 8;
            nr = 8;
            elapsed_times = measure_time_bin_bin_kernel<kernel_bin_8x8>(A, B, C, M, K, N, mr, nr, n_run);
            break;

        case Method::kernel_bin_8x16:
            method_name = std::string("Bin Kernel 512 8x16");
            mr = 8;
            nr = 16;
            elapsed_times = measure_time_bin_bin_kernel<kernel_bin_8x16>(A, B, C, M, K, N, mr, nr, n_run);
            break;
        case Method::kernel_bin_6x16:
            method_name = std::string("Bin kernel 512 6x16");
            mr = 6;
            nr = 16;
            elapsed_times = measure_time_bin_bin_kernel<kernel_bin_6x16>(A, B, C, M, K, N, mr, nr, n_run);
            break;

        case Method::kernel_bin_2x8:
            method_name = std::string("Bin kernel 512 2x8");
            mr = 2;
            nr = 8;
            elapsed_times = measure_time_bin_bin_kernel<kernel_bin_2x8>(A, B, C, M, K, N, mr, nr, n_run);
            break;

        case Method::kernel_bin_8x8_16bit:
            method_name = std::string("Bin kernel 512 8x8 with 16 bit");
            mr = 8;
            nr = 32;
            elapsed_times = measure_time_bin_bin_kernel_16bit<kernel_bin_8x8_16bit>(A, B, C, M, K, N, mr, nr, n_run);
            break;
            
        case Method::kernel_bin_8x16_16bit:
            method_name = std::string("Bin kernel 512 8x16 with 16 bit");
            mr = 8;
            nr = 64;
            elapsed_times = measure_time_bin_bin_kernel_16bit<kernel_bin_8x16_16bit >(A, B, C, M, K, N, mr, nr, n_run);
            break;
        

    }

    timing_print(method_name, elapsed_times, original_M, original_N, K);

    if(check) {
        std::cout << "Testing correctness...\n";
        vector<float> C_truth(M*N);
        trivial_dense_multiplication(A, B, M, K, N, C_truth);
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