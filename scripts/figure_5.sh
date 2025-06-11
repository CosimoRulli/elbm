
file=$1
echo -e "lib\tM\tK\tN\tmin_elapsed\tmean_elapsed" > $file


declare -a sizes=(
    "64 576 2916"
    "128 576 2916"
    "128 1152 676"
    "256 1152 676"
    "256 2304 144"
    "512 2304 144"
    "512 4608 25"
)

for size in "${sizes[@]}";
do read -a c_size  <<< "$size"

    if ./build/perf_1bit_vs_1bit -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} -t 1 -c; then 
    ./build/perf_1bit_vs_1bit -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} -t 1 2>>$file
    fi

    if ./build/perf_1bit_vs_2bits -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} -t 1 -c; then 
    ./build/perf_1bit_vs_2bits -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} -t 1 2>>$file
    fi

    if ./build/perf_2bit_vs_2bits -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} -c; then 
    ./build/perf_2bit_vs_2bits -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} 2>>$file
    fi

    if ./build/perf_onednn -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} -c; then 
    ./build/perf_onednn -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} 2>>$file
    fi

    if ./build/perf_blis -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} -c; then 
    ./build/perf_blis -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} 2>>$file
    fi


    if ./build/perf_fbgemm -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} -c; then 
    ./build/perf_fbgemm -m ${c_size[0]} -k ${c_size[1]} -n ${c_size[2]} 2>>$file
    fi


done