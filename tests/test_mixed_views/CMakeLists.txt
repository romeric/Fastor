cmake_minimum_required(VERSION 3.1)
project(test_mixed_views)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_mixed_views test_mixed_views.cpp)
add_test(test_mixed_views test_mixed_views)

if(MSVC)
    target_compile_options(test_mixed_views PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_mixed_views PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_mixed_views PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_mixed_views PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)