cmake_minimum_required(VERSION 3.1)
project(test_transpose)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_transpose test_transpose.cpp)
add_test(test_transpose test_transpose)

if(MSVC)
    add_compile_options(test_transpose PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_transpose PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_transpose PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_transpose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(test_transpose PRIVATE Fastor)
