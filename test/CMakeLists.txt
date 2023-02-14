project(Google_tests)
add_subdirectory(lib/googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# adding the Google_Tests_run target
add_executable(Google_Tests_run NeuronTest.cpp NetworkTest.cpp ProjectTest.cpp NetworkTest.cpp ../lib/network.cpp)

configure_file(../t10k-images.idx3-ubyte t10k-images.idx3-ubyte COPYONLY)
configure_file(../t10k-labels.idx1-ubyte t10k-labels.idx1-ubyte COPYONLY)
configure_file(../train-images.idx3-ubyte train-images.idx3-ubyte COPYONLY)
configure_file(../train-labels.idx1-ubyte train-labels.idx1-ubyte COPYONLY)

# linking Google_Tests_run with knnlibrary which will be tested
target_link_libraries(Google_Tests_run knnlibrary)
target_link_libraries(Google_Tests_run gtest gtest_main)