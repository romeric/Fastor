cmake_minimum_required(VERSION 3.1)
project(test_solve)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_solve test_solve.cpp)
add_test(test_solve test_solve)

if(MSVC)
    add_compile_options(test_solve PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_solve PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_solve PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_solve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)