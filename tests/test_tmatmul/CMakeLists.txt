cmake_minimum_required(VERSION 3.20)
project(test_tmatmul)

set(CMAKE_CXX_STANDARD 14)

if(MSVC)
    add_compile_options("/std:c++17" "/W2" "$<$<CONFIG:RELEASE>:/Od>")
else()
    add_compile_options("$<$<CONFIG:RELEASE>:-O0>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

# unmasked
add_executable(test_tmatmul_unmasked test_tmatmul_unmasked.cpp)
add_test(test_tmatmul_unmasked test_tmatmul_unmasked)

target_include_directories (test_tmatmul_unmasked PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories (test_tmatmul_unmasked PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# masked
add_executable(test_tmatmul_masked test_tmatmul_masked.cpp)
add_test(test_tmatmul_masked test_tmatmul_masked)

target_include_directories (test_tmatmul_masked PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories (test_tmatmul_masked PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)