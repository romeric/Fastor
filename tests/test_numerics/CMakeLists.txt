cmake_minimum_required(VERSION 3.1)
project(test_numerics)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_numerics test_numerics.cpp)
add_test(test_numerics test_numerics)

if(MSVC)
    add_compile_options(test_numerics PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_numerics PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_numerics PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_numerics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)