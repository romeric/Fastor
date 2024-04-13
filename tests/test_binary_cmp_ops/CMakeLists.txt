cmake_minimum_required(VERSION 3.20)
project(test_binary_cmp_ops)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_binary_cmp_ops test_binary_cmp_ops.cpp)
add_test(test_binary_cmp_ops test_binary_cmp_ops)

if(MSVC)
    add_compile_options(test_binary_cmp_ops PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_binary_cmp_ops PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_binary_cmp_ops PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_binary_cmp_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)