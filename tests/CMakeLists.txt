cmake_minimum_required(VERSION 3.1)
project(test_fastor)

set(FASTOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

set(CMAKE_CXX_STANDARD 14)

enable_testing()

if(MSVC)
    add_compile_options("/std:c++17" "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options("-march=native" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "g++")
        add_compile_options("-fabi-version=6")
    endif()
endif()

add_subdirectory(test_auxiliary_funcs)

add_subdirectory(test_simd_vectors)

add_subdirectory(test_tensor_basics)

add_subdirectory(test_booleans)

add_subdirectory(test_matmul)
# This test checks over 30000 combination
# MSVC creates an excessively large obj for this
if(NOT MSVC)
add_subdirectory(test_tmatmul)
endif()

add_subdirectory(test_transpose)

add_subdirectory(test_permute)

add_subdirectory(test_einsum)
add_subdirectory(test_einsum_explicit)

add_subdirectory(test_linalg)
add_subdirectory(test_lu)
add_subdirectory(test_inverse)
add_subdirectory(test_solve)

add_subdirectory(test_fixed_views_1d)
add_subdirectory(test_fixed_views_2d)
add_subdirectory(test_fixed_views_nd)
add_subdirectory(test_views_1d)
add_subdirectory(test_views_2d)
add_subdirectory(test_views_nd)
add_subdirectory(test_random_views)

add_subdirectory(test_mixed_views)

add_subdirectory(test_complex_expressions)

add_subdirectory(test_tensormap)

add_subdirectory(test_unary_bool_ops)

add_subdirectory(test_binary_cmp_ops)