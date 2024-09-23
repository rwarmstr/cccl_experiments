cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CUDA_ARCHITECTURES "native")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

project(cub_fractal LANGUAGES CXX CUDA)

add_executable(cub_fractal
    cub_fractal.cu
)

target_link_libraries(cub_fractal
    ${OpenCV_LIBS}
)

set_target_properties(cub_fractal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
