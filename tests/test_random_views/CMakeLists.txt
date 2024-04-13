cmake_minimum_required(VERSION 3.20)
project(test_random_views)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_random_views_1d test_random_views_1d.cpp)
add_test(test_random_views_1d test_random_views_1d)

if(MSVC)
    target_compile_options(test_random_views_1d PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_random_views_1d PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_random_views_1d PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_random_views_1d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


add_executable(test_random_views_nd test_random_views_nd.cpp)
add_test(test_random_views_nd test_random_views_nd)

if(MSVC)
    target_compile_options(test_random_views_nd PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_random_views_nd PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_random_views_nd PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_random_views_nd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)