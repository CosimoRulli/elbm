#include <iostream>
#include <vector>

#include "parser.hpp"
#include "utils.hpp"
#include "tuple"
#include "2_bit_vs_2_bits_kernels.hpp"
#include "utils.hpp"


enum class Method {
    kernel8x8
};


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

    int pad = 16; 
    size_t original_M = M;
    size_t original_N = N;
    M = (M + (pad-1))/pad * pad;
    N = (N + (pad-1))/pad * pad;


    if (K < 64 || K % 64 !=0){
        std::cout<<"K must be divisible by 512\n";
        return 1;
    }


   // auto method_id = parser.get<uint32_t>("Method");
    auto method_id = 0;
    Method method {method_id};
    bool check = parser.get<bool>("Check");
    if (check) n_run=1;

    float s = 0.3; //scaler for 2bits quantization


    // auto [A, true_m, true_k] = npy_from_path("/home/crulli/BenchmarkCosimo/A_2bits.npy");
    // auto [B, second_k, true_n] = npy_from_path("/home/crulli/BenchmarkCosimo/B_2bits.npy");

    auto A = generate_2bits_mat(M, K, s);
    auto B = generate_2bits_mat(K, N, s);
    
    vector<float> C(M*N, 0.);
    
    vector<uint64_t> elapsed_times;
    std::string method_name = "";

    bool output_packed = false;

    size_t mr, nr;
    switch (method) {
        case Method::kernel8x8:
            method_name = std::string("Matmul_2_2 Kernel 8x8");
            mr = 8;
            nr = 8;
            elapsed_times = measure_time_2_2_kernel_64_bit<kernel_2_2_8x8>(A, B, C, M, K, N, s, mr, nr, n_run );
            break;
    }

    timing_print(method_name, elapsed_times, original_M, original_N, K);

    if(check) {
        std::cout << "Testing correctness...\n";
        vector<float> C_truth(M*N);
        trivial_dense_multiplication(A, B, M, K, N, C_truth);
        //print_mat(C_truth, M, N);
        if (output_packed){
            cout<<"Packing C_truth matrix for comparison\n";
            C_truth = pack_matrix_for_binary_mul(C_truth, M, N);
        }

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