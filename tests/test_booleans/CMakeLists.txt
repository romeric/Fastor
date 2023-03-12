cmake_minimum_required(VERSION 3.1)
project(test_booleans)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_booleans test_booleans.cpp)
add_test(test_booleans test_booleans)

if(MSVC)
    add_compile_options(test_booleans PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_booleans PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_booleans PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_booleans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(test_booleans PRIVATE Fastor)
