# Elbm
This repository contains the code for implementing efficient binary multiplication algorithms on CPU. 
Binary multiplication is an essential operation in computer arithmetic, 
and efficient algorithms can significantly improve the performance of multiplication operations in various applications.

## Example

Clone the directory using the ```recursive``` flag. Then, switch to the correct commit for each submodule.

```
cd external/oneDNN
git checkout 7981216b8341b8603e54472f5a0dd7a12ef9cf67
cd - 

cd external/FBGEMM
git checkout d0eb1847bd3705246ed1697b7d47eb7d9e00ba46
git submodule update --init --recursive 
cd -

cd ~./blis
git checkout 56772892450cc92b3fbd6a9d0460153a43fc47ab
cd - 
```

Install `blis` following the instruction provided in the original repo.


```
mkdir build && cd build 
cmake ..
make -j
```

Run ```build/perf_1bit_vs_1bit -h``` to see the execution options. 

