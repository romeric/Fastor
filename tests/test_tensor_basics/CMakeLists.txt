cmake_minimum_required(VERSION 3.20)
project(test_tensor_basics)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_tensor_basics test_tensor_basics.cpp)
add_test(test_tensor_basics test_tensor_basics)

if(MSVC)
    add_compile_options(test_tensor_basics PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_tensor_basics PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_tensor_basics PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_tensor_basics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)