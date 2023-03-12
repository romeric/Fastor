cmake_minimum_required(VERSION 3.1)
project(test_linalg)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_linalg test_linalg.cpp)
add_test(test_linalg test_linalg)

if(MSVC)
    add_compile_options(test_linalg PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_linalg PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_linalg PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_linalg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(test_linalg PRIVATE Fastor)
