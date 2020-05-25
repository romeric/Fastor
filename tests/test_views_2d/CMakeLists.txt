cmake_minimum_required(VERSION 3.1)
project(test_views_2d)

set(CMAKE_CXX_STANDARD 14)

# strided vectorisation off
add_executable(test_views_2d test_views_2d.cpp)
add_test(test_views_2d test_views_2d)

if(MSVC)
    target_compile_options(test_views_2d PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_views_2d PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_views_2d PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_views_2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# strided vectorisation on
add_executable(test_views_2d_vec test_views_2d.cpp)
add_test(test_views_2d_vec test_views_2d_vec)

if(MSVC)
    target_compile_options(test_views_2d_vec PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_views_2d_vec PRIVATE "-DFASTOR_USE_VECTORISED_EXPR_ASSIGN" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_views_2d_vec PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_views_2d_vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)