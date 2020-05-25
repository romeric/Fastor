cmake_minimum_required(VERSION 3.1)
project(test_auxiliary_funcs)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_auxiliary_funcs test_auxiliary_funcs.cpp)
add_test(test_auxiliary_funcs test_auxiliary_funcs)

if(MSVC)
    add_compile_options(test_auxiliary_funcs PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_auxiliary_funcs PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_auxiliary_funcs PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_auxiliary_funcs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)