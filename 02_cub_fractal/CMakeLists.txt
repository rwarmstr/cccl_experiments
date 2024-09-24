find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cub_fractal
    cub_fractal.cu
)

target_link_libraries(cub_fractal
    ${OpenCV_LIBS}
)

set_target_properties(cub_fractal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
