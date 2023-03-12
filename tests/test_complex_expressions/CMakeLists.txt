cmake_minimum_required(VERSION 3.1)
project(test_complex_expressions)

set(CMAKE_CXX_STANDARD 14)

# strided vectorisation off
add_executable(test_complex_expressions_1 test_complex_expressions.cpp)
add_test(test_complex_expressions_1 test_complex_expressions_1)

if(MSVC)
    target_compile_options(test_complex_expressions_1 PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_complex_expressions_1 PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_complex_expressions_1 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_complex_expressions_1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# strided vectorisation on
add_executable(test_complex_expressions_2 test_complex_expressions.cpp)
add_test(test_complex_expressions_2 test_complex_expressions_2)

if(MSVC)
    target_compile_options(test_complex_expressions_2 PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_complex_expressions_2 PRIVATE "-DFASTOR_USE_VECTORISED_EXPR_ASSIGN" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_complex_expressions_2 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_complex_expressions_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(test_complex_expressions_1 PRIVATE Fastor)
target_link_libraries(test_complex_expressions_2 PRIVATE Fastor)
