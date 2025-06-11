
file_bin=$1
echo -e "method\tM\tK\tN\tmin_elapsed\tmean_elapsed" > $file_bin


for size in {64..2560..64}; do

    if ./build/perf_1bit_vs_1bit -m $size -k $size -n $size -t 1 -c; then 
    ./build/perf_1bit_vs_1bit -m $size -k $size -n $size -t 1 2>>$file_bin
    fi

    if ./build/perf_1bit_vs_2bits -m $size -k $size -n $size -t 1 -c; then 
    ./build/perf_1bit_vs_2bits -m $size -k $size -n $size -t 1 2>>$file_bin
    fi

    if ./build/perf_2bit_vs_2bits -m $size -k $size -n $size -c; then 
    ./build/perf_2bit_vs_2bits -m $size -k $size -n $size 2>>$file_bin
    fi

    if ./build/perf_onednn -m $size -k $size -n $size -c; then 
    ./build/perf_onednn -m $size -k $size -n $size 2>>$file_bin
    fi

    if ./build/perf_blis -m $size -k $size -n $size -c; then 
    ./build/perf_blis -m $size -k $size -n $size 2>>$file_bin
    fi

    # if ./build/perf_2bits_multiplication -m $size -k $size -n $size -t 3 -c; then 
    # ./build/perf_2bits_multiplication -m $size -k $size -n $size -t 3 2>>$file_bin
    # fi

    # if ./build/perf2bits_vs_2bits -m $size -k $size -n $size -t 1 -c; then 
    # ./build/perf2bits_vs_2bits -m $size -k $size -n $size -t 1 2>>$file_bin
    # fi



    # if ./build/perf_2bits_multiplication -m $size -k $size -n $size -t 4 -c; then 
    # ./build/perf_2bits_multiplication -m $size -k $size -n $size -t 4 2>>$file_bin
    # fi  
done
