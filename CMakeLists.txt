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

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
