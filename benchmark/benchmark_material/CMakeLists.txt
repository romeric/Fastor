cmake_minimum_required(VERSION 3.1)
project(benchmark_material)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

add_executable(benchmark_material benchmark_materials.cpp)
add_test(benchmark_material benchmark_material)

if(MSVC)
    add_compile_options(benchmark_material PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(benchmark_material PRIVATE "-DNDEBUG -ffast-math -ffp-contract=fast" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(benchmark_material PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(benchmark_material PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../)