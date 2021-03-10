cmake_minimum_required(VERSION 3.0.0)
project(knnframework VERSION 0.1.0 LANGUAGES CXX CUDA)

add_library(network STATIC
    network.cu
    network.h
    neuron.cu
    neuron.h
    mathhelper.cpp
    mathhelper.h
)

target_compile_features(network PUBLIC cxx_std_11)
set_target_properties(network PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION true)

add_executable(knnframework main.cpp)
set_property(TARGET knnframework PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(knnframework PRIVATE network)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
