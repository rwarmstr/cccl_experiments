cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")  # enable cuda-gdb
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

project(signals LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(signals
    signals.cu
)

target_link_libraries(signals
    CUDA::cufft
)

set_target_properties(cub_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
