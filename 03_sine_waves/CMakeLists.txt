find_package(CUDAToolkit REQUIRED)

add_executable(signals
    signals.cu
)

target_link_libraries(signals
    CUDA::cufft
)

set_target_properties(signals PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVUTIL REQUIRED libavutil)

link_directories(
    ${AVFORMAT_LIBRARY_DIRS}
    ${AVCODEC_LIBRARY_DIRS}
    ${AVUTIL_LIBRARY_DIRS}
)

# Collect all FFmpeg libraries
set(FFMPEG_LIBS
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${AVUTIL_LIBRARIES}
)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
find_package( OpenCV REQUIRED )
find_package(OpenMP REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(audio_fft
    audio_fft.cu
)

target_include_directories(audio_fft PUBLIC
    /usr/include/qwt
    ${OpenCV_INCLUDE_DIRS}
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
)

target_link_libraries(audio_fft
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    /usr/lib/libqwt-qt5.so
    CUDA::cufft
    ${FFMPEG_LIBS}
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

# Include definitions and compiler options for FFMPeg libraries as needed
target_compile_definitions(audio_fft PRIVATE ${AVFORMAT_CFLAGS_OTHER} ${AVCODEC_CFLAGS_OTHER} ${AVUTIL_CFLAGS_OTHER})
target_compile_options(audio_fft PRIVATE ${AVFORMAT_CFLAGS_OTHER} ${AVCODEC_CFLAGS_OTHER} ${AVUTIL_CFLAGS_OTHER})
