cmake_minimum_required(VERSION 3.1)
project(test_permute)

set(CMAKE_CXX_STANDARD 14)

enable_testing()

add_executable(test_permute test_permute.cpp)
add_test(test_permute test_permute)

if(MSVC)
    target_compile_options(test_permute PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_permute PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_permute PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_permute PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
