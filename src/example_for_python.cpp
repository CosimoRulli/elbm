
#include <iostream>
#include <vector>

#include "parser.hpp"
#include "for_python_interface.hpp"
#include "1_bit_vs_1_bit_kernels.hpp"
#include "tuple"

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

    // parser.add("Check",       // name
    //            "Check the result",  // description
    //            "-c",                    // shorthand
    //            true                    // boolean option: expected no value after the shorthand
    // );

}

template <typename  T1, typename  T2>
size_t check_equality(const vector<T1> &A, const T2 * B, const size_t M, const size_t N, const float toll){
    size_t count=0;
    for (size_t i=0; i< M; i++){
        for (size_t j=0; j< N; j++){
            //if (abs(A[i*N +j] - (T1) B[i*N +j]) >= toll) {count++; cout<<A[i*N +j]<< " "<< (T1) B[i*N +j]<<"\n";};
            if (abs(A[i*N +j] - (T1) B[i*N +j]) >= toll) {count++;};

        }
    }
    return count;
}

inline void trivial_dense_multiplication(const vector<int64_t> &A, const vector<int64_t> &B, const size_t M, const size_t K, const size_t N, vector<int64_t> &C) {
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

vector<int64_t> generate_binary_matrix(const size_t M, const size_t N){

    random_device rd;
    mt19937 e(rd());
    vector<int64_t> integer_values (M*N);
    uniform_real_distribution<> dist_float(-5, 5);

    float p;
    for (size_t i=0; i <M*N; i++){
        p = dist_float(e);
        integer_values[i] = (uint64_t)(p >= 0 ? 1 : -1);
    }

    return integer_values;
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

    const float toll = 0.0001;

    cmd_line_parser::parser parser(argc, argv);

    argparse(parser);
    if (!parser.parse()) return 1;

    size_t M = parser.get<size_t>("M"); 
    size_t K = parser.get<size_t>("K"); 
    size_t N = parser.get<size_t>("N"); 

    //bool check = parser.get<bool>("Check");
    bool check = false;

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

    auto A = generate_binary_matrix(M, K);
    auto B = generate_binary_matrix(K, N);

    auto C = binary_matrix_multiplication(A, B, M, K, N);

    
    // vector<float> A = generate_bin_mat(M, K);
    // auto B = generate_bin_mat(K, N);
    // vector<float> C(M*N, 0.);

    size_t mr = 8;
    size_t nr = 16;

    //auto elapsed_times = binary_multiplication(A, B, C, M, K, N, mr, nr);


    std::cout << "Testing correctness...\n";

    //vector<float> C_truth(M*N);
    vector<int64_t> C_truth(M*N);
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

    return 0;
}