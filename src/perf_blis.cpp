#include <iostream>
#include <vector>
#include <blis.h>
#include "parser.hpp"
#include "utils.hpp"
#include "tuple"


vector<uint64_t> measure_time_dense_blis(const vector<float> &A, const vector<float> &B, const size_t M, const size_t K, const size_t N, vector<float> & C, const size_t n_run){
    // size_t rsa, csa;
    // size_t rsb, csb;
    // size_t rsc, csc;

    float alpha = 1.0;
    float beta = .0;
    // rsa = 1;
    // rsb = 1;
    // rsc = 1;
    // csc = M;
    // csa = M;
    // csb = K;

   vector<uint64_t> elapsed_times;

   //warmup
   
   bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &alpha, A.data(), K, 1, B.data(), N,
              1, &beta, C.data(), N, 1);
 
   for (size_t run=0; run < n_run; run ++){
       auto start = std::chrono::high_resolution_clock::now();
        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, &alpha, A.data(), K, 1, B.data(), N,
              1, &beta, C.data(), N, 1);
       auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
       elapsed_times.push_back(elapsed);
   }


   return elapsed_times;

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
    const float toll = 0.0001;

    cmd_line_parser::parser parser(argc, argv);

    argparse(parser);
    if (!parser.parse()) return 1;

    size_t M = parser.get<size_t>("M"); ;
    size_t K = parser.get<size_t>("K"); ;
    size_t N = parser.get<size_t>("N"); ;

    //auto method_id = 2;
    bool check = parser.get<bool>("Check");
    if (check) n_run = 1;

    cout<< "M " << M << "\n";
    cout<< "K " << K << "\n";
    cout<< "N " << N << "\n";

    vector<float> A = generate_bin_mat(M, K);
    auto B = generate_bin_mat(K, N);
    vector<float> C(M*N, 0.);

    vector<uint64_t> elapsed_times = measure_time_dense_blis(A, B, M, K, N, C, n_run);
    std::string method_name = "BLIS";

    timing_print(method_name, elapsed_times, M, N, K);

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