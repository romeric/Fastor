cmake_minimum_required(VERSION 3.1)
project(test_lu)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_lu test_lu.cpp)
add_test(test_lu test_lu)

if(MSVC)
    add_compile_options(test_lu PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_lu PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_lu PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_lu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)