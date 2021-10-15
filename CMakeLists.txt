cmake_minimum_required(VERSION 3.0.0)
project(knnframework VERSION 0.1.0 LANGUAGES CXX)

add_library(network STATIC
    network.cpp
    network.h
    neuron.cpp
    neuron.h
    mathhelper.cpp
    mathhelper.h
)

# comment this to disable example
add_executable(knnframework main.cpp)

target_link_libraries(knnframework PRIVATE network)

configure_file(t10k-images.idx3-ubyte t10k-images.idx3-ubyte COPYONLY)
configure_file(t10k-labels.idx1-ubyte t10k-labels.idx1-ubyte COPYONLY)
configure_file(train-images.idx3-ubyte train-images.idx3-ubyte COPYONLY)
configure_file(train-labels.idx1-ubyte train-labels.idx1-ubyte COPYONLY)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
