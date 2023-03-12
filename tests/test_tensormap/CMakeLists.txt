cmake_minimum_required(VERSION 3.1)
project(test_tensormap)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_tensormap test_tensormap.cpp)
add_test(test_tensormap test_tensormap)

if(MSVC)
    add_compile_options(test_tensormap PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_tensormap PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_tensormap PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_tensormap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(test_tensormap PRIVATE Fastor)
