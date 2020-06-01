cmake_minimum_required(VERSION 3.1)
project(benchmark_finite_difference)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

if(MSVC)
    add_compile_options("/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options("$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "g++")
        add_compile_options("-finline-functions -finline-limit=1000000 -ffp-contract=fast")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang|AppleClang|c++")
        add_compile_options("-ftemplate-depth=10000000 -fconstexpr-depth=1000000000 -mllvm -inline-threshold=1000000 -foperator-arrow-depth=100000000 -ffp-contract=fast")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        add_compile_options("-inline-forceinline -fp-model fast=2")
    endif()
endif()

include_directories(${FASTOR_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../)


add_executable(out_loops_c benchmark_finite_difference_c.cpp)
add_test(out_loops_c out_loops_c)

add_executable(out_loops benchmark_finite_difference.cpp)
add_test(out_loops out_loops)
target_compile_definitions(out_loops PRIVATE FASTOR_DISPATCH_DIV_TO_MUL_EXPR USE_LOOPS)

add_executable(out_seq_alias benchmark_finite_difference.cpp)
add_test(out_seq_alias out_seq_alias)
target_compile_definitions(out_seq_alias PRIVATE FASTOR_DISPATCH_DIV_TO_MUL_EXPR USE_SEQ_ALIAS)

add_executable(out_seq_noalias benchmark_finite_difference.cpp)
add_test(out_seq_noalias out_seq_noalias)
target_compile_definitions(out_seq_noalias PRIVATE FASTOR_DISPATCH_DIV_TO_MUL_EXPR USE_SEQ_NOALIAS FASTOR_NO_ALIAS)

add_executable(out_iseq benchmark_finite_difference.cpp)
add_test(out_iseq out_iseq)
target_compile_definitions(out_iseq PRIVATE FASTOR_DISPATCH_DIV_TO_MUL_EXPR USE_ISEQ)

add_executable(out_fseq benchmark_finite_difference.cpp)
add_test(out_fseq out_fseq)
target_compile_definitions(out_fseq PRIVATE FASTOR_DISPATCH_DIV_TO_MUL_EXPR USE_FSEQ FASTOR_NO_ALIAS)
