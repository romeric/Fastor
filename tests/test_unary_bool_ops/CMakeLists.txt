cmake_minimum_required(VERSION 3.1)
project(test_unary_bool_ops)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_unary_bool_ops test_unary_bool_ops.cpp)
add_test(test_unary_bool_ops test_unary_bool_ops)

if(MSVC)
    add_compile_options(test_unary_bool_ops PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_unary_bool_ops PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_unary_bool_ops PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_unary_bool_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)