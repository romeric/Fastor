cmake_minimum_required(VERSION 3.1)
project(test_simd_vectors)

set(CMAKE_CXX_STANDARD 14)

enable_testing()


add_executable(test_simd_vectors test_simd_vectors.cpp)
add_test(test_simd_vectors test_simd_vectors)

if(MSVC)
    target_compile_options(test_simd_vectors PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_simd_vectors PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_simd_vectors PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_simd_vectors PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


# complex
add_executable(test_simd_vectors_complex test_simd_vectors_complex.cpp)
add_test(test_simd_vectors_complex test_simd_vectors_complex)

if(MSVC)
    target_compile_options(test_simd_vectors_complex PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_simd_vectors_complex PRIVATE "-DCONTRACT_OPT=1" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_simd_vectors_complex PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_simd_vectors_complex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_link_libraries(test_simd_vectors PRIVATE Fastor)
target_link_libraries(test_simd_vectors_complex PRIVATE Fastor)
