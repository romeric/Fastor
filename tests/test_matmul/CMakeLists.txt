cmake_minimum_required(VERSION 3.1)
project(test_matmul)

set(CMAKE_CXX_STANDARD 14)

if(MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options("$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

# matmul
add_executable(test_matmul test_matmul.cpp)
add_test(test_matmul test_matmul)

target_include_directories (test_matmul PUBLIC ${FASTOR_INCLUDE_DIR})
target_include_directories (test_matmul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# matmul small
add_executable(test_matmul_small test_matmul_small.cpp)
add_test(test_matmul_small test_matmul_small)

target_include_directories (test_matmul_small PUBLIC ${FASTOR_INCLUDE_DIR})
target_include_directories (test_matmul_small PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# lazy matmul
add_executable(test_lazy_matmul test_lazy_matmul.cpp)
add_test(test_lazy_matmul test_lazy_matmul)

target_include_directories (test_lazy_matmul PUBLIC ${FASTOR_INCLUDE_DIR})
target_include_directories (test_lazy_matmul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../)