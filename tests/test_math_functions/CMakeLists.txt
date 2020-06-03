cmake_minimum_required(VERSION 3.1)
project(test_math_functions)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_math_functions test_math_functions.cpp)
add_test(test_math_functions test_math_functions)

if(MSVC)
    target_compile_options(test_math_functions PRIVATE "/W1" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_math_functions PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_math_functions PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_math_functions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
