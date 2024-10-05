cmake_minimum_required(VERSION 3.20)

project(qwt_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
find_package( OpenCV REQUIRED )

add_executable(qwt_test 
    qwt_test.cpp
)

target_include_directories(qwt_test PRIVATE
    /usr/include/qwt
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(qwt_test PRIVATE
    Qt5::Core 
    Qt5::Gui 
    Qt5::Widgets 
    /usr/lib/libqwt-qt5.so
    ${OpenCV_LIBS}
)