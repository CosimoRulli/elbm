cmake_minimum_required(VERSION 3.10)

project(encoding)

enable_testing()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()

MESSAGE(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

include_directories(include)

set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

#find_package(Boost COMPONENTS iostreams filesystem system REQUIRED)

# Catch2
#find_package(Catch2 2 REQUIRED)
include_directories(test) # use header only catch2

# cmd_line_parser
include_directories(external/cmd_line_parser/include)

#LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/external/sdsl-lite/lib)
#INCLUDE_DIRECTORIES(${CMAKE_HOME_DIRECTORY}/include
#                    ${CMAKE_BINARY_DIR}/external/sdsl-lite/include
#                    ${CMAKE_BINARY_DIR}/external/sdsl-lite/external/libdivsufsort-2.0.1/include
#                    )

# Set compiler options
if(NOT MSVC)
  set(CMAKE_C_FLAGS "-O3 -ffast-math -march=native -mavx512vl -funroll-loops ${CMAKE_C_FLAGS}")
  
else()
  include("${CMAKE_CURRENT_LIST_DIR}/msvc_simd_isa.cmake")
  if(SUPPORT_MSVC_AVX512)
    set(MSVC_SIMD_FLAGS "/arch:AVX512")
  elseif(SUPPORT_MSVC_AVX2)
    set(MSVC_SIMD_FLAGS "/arch:AVX2")
  elseif(SUPPORT_MSVC_AVX)
    set(MSVC_SIMD_FLAGS "/arch:AVX")
  else()
    set(MSVC_SIMD_FLAGS "")
  endif()
  set(CMAKE_C_FLAGS "/O2 /fp:fast ${MSVC_SIMD_FLAGS} ${CMAKE_C_FLAGS}")
endif()
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -Wall -Wextra -Wno-missing-braces -std=c++17 -Wno-unused-variable -Wno-sign-compare -fopenmp -lpthread")

# Set Windows stack size as on Linux: 2MB on 32-bit, 8MB on 64-bit
if (WIN32)
  math(EXPR stack_size "${CMAKE_SIZEOF_VOID_P}*${CMAKE_SIZEOF_VOID_P}*128*1024")
  if (MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:${stack_size}")
  else()
    # compiling with clang + lld
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xlinker /stack:${stack_size}")
  endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS}")

## Project-wide setup
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS YES)

add_subdirectory(external/cmd_line_parser)
include_directories(${PROJECT_SOURCE_DIR}/external/cmd_line_parser)

add_subdirectory(external/oneDNN)
include_directories(${PROJECT_SOURCE_DIR}/external/oneDNN)


add_subdirectory(external/FBGEMM)
#include_directories(${PROJECT_SOURCE_DIR}/external/FBGEMM/include/fbgemm)
#include_directories(${PROJECT_SOURCE_DIR}/external/FBGEMM/include/)
#include_directories(${PROJECT_SOURCE_DIR}/external/FBGEMM/)


set(DNNL_LIBRARY_TYPE "STATIC")
set(DNNL_CPU_RUNTIME "OMP")
set(DNNL_ARCH_OPT_FLAGS "${CMAKE_CXX_FLAGS}")
set(DNNL_ENABLE_PRIMITIVE_CACHE "ON")
set(ONEDNN_ENABLE_MAX_CPU_ISA "ON")
set(DNNL_BUILD_TESTS "OFF")
set(DNNL_BUILD_EXAMPLES "OFF")

set(BLIS_DIR /home/crulli/crulli/blis)
include_directories(${BLIS_DIR}/include/blis)
link_directories(${BLIS_DIR}/lib)
find_library(BLIS_LIB NAMES blis PATHS ${BLIS_DIR}/lib)


add_executable(perf_1bit_vs_1bit src/perf_1_bit_vs_1_bit.cpp)
target_link_libraries(perf_1bit_vs_1bit)

add_executable(example_for_python src/example_for_python.cpp)
target_link_libraries(example_for_python)

add_executable(perf_1bit_vs_2bits src/perf_1_bit_vs_2_bits.cpp)
target_link_libraries(perf_1bit_vs_2bits)

add_executable(perf_2bit_vs_2bits src/perf_2_bit_vs_2_bits.cpp)
target_link_libraries(perf_2bit_vs_2bits)

add_executable(perf_onednn src/perf_onednn.cpp)
target_link_libraries(perf_onednn dnnl)

add_executable(perf_blis src/perf_blis.cpp)
target_link_libraries(perf_blis ${BLIS_LIB})

add_executable(perf_fbgemm src/perf_fbgemm.cpp)
target_link_libraries(perf_fbgemm fbgemm)
