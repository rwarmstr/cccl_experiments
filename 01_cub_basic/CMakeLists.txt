add_executable(cub_test
    cub_basic.cu
)

set_target_properties(cub_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
