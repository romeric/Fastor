cmake_minimum_required(VERSION 3.1)
project(test_inverse)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_inverse test_inverse.cpp)
add_test(test_inverse test_inverse)

if(MSVC)
    add_compile_options(test_inverse PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_inverse PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_inverse PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_inverse PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)