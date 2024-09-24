find_package(CUDAToolkit REQUIRED)

add_executable(signals
    signals.cu
)

target_link_libraries(signals
    CUDA::cufft
)

set_target_properties(signals PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
