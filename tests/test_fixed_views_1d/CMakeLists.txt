cmake_minimum_required(VERSION 3.1)
project(test_fixed_views_1d)

set(CMAKE_CXX_STANDARD 14)

# strided vectorisation off
add_executable(test_fixed_views_1d test_fixed_views_1d.cpp)
add_test(test_fixed_views_1d test_fixed_views_1d)

if(MSVC)
    target_compile_options(test_fixed_views_1d PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_fixed_views_1d PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_fixed_views_1d PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_fixed_views_1d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# strided vectorisation on
add_executable(test_fixed_views_1d_vec test_fixed_views_1d.cpp)
add_test(test_fixed_views_1d_vec test_fixed_views_1d_vec)

if(MSVC)
    target_compile_options(test_fixed_views_1d_vec PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_fixed_views_1d_vec PRIVATE "-DFASTOR_USE_VECTORISED_EXPR_ASSIGN" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_fixed_views_1d_vec PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_fixed_views_1d_vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)