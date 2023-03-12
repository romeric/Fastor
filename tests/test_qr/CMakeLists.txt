cmake_minimum_required(VERSION 3.1)
project(test_qr)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_qr test_qr.cpp)
add_test(test_qr test_qr)

if(MSVC)
    add_compile_options(test_qr PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(test_qr PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_qr PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_qr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(test_qr PRIVATE Fastor)
