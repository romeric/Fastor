cmake_minimum_required(VERSION 3.1)
project(test_views_nd)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_views_nd test_views_nd.cpp)
add_test(test_views_nd test_views_nd)

if(MSVC)
    target_compile_options(test_views_nd PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_views_nd PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_views_nd PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_views_nd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


add_executable(test_views_nd_2 test_views_nd_2.cpp)
add_test(test_views_nd_2 test_views_nd_2)

if(MSVC)
    target_compile_options(test_views_nd_2 PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_views_nd_2 PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_views_nd_2 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_views_nd_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)