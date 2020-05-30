cmake_minimum_required(VERSION 3.1)
project(test_einsum)

set(CMAKE_CXX_STANDARD 14)

enable_testing()

# single einsum
add_executable(test_einsum_single test_einsum_single.cpp)
add_test(test_einsum_single test_einsum_single)

if(MSVC)
    target_compile_options(test_einsum_single PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_einsum_single PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_einsum_single PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_einsum_single PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# tensor contraction
# default
add_executable(test_contraction_0 test_contraction.cpp)
add_test(test_contraction_0 test_contraction_0)

if(MSVC)
    target_compile_options(test_contraction_0 PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_contraction_0 PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_contraction_0 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_contraction_0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


# -DCONTRACT_OPT=1
add_executable(test_contraction_1 test_contraction.cpp)
add_test(test_contraction_1 test_contraction_1)

if(MSVC)
    target_compile_options(test_contraction_1 PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_contraction_1 PRIVATE "-DCONTRACT_OPT=1" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_contraction_1 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_contraction_1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


# -DCONTRACT_OPT=-1
add_executable(test_contraction_2 test_contraction.cpp)
add_test(test_contraction_2 test_contraction_2)

if(MSVC)
    target_compile_options(test_contraction_2 PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_contraction_2 PRIVATE "-DCONTRACT_OPT=-1" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_contraction_2 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_contraction_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


# tensor einsum
# default
add_executable(test_einsum_0 test_einsum.cpp)
add_test(test_einsum_0 test_einsum_0)

if(MSVC)
    target_compile_options(test_einsum_0 PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_einsum_0 PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_einsum_0 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_einsum_0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


# -DCONTRACT_OPT=1
add_executable(test_einsum_1 test_einsum.cpp)
add_test(test_einsum_1 test_einsum_1)

if(MSVC)
    target_compile_options(test_einsum_1 PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_einsum_1 PRIVATE "-DCONTRACT_OPT=1" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_einsum_1 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_einsum_1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


# -DCONTRACT_OPT=-1
add_executable(test_einsum_2 test_einsum.cpp)
add_test(test_einsum_2 test_einsum_2)

if(MSVC)
    target_compile_options(test_einsum_2 PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_einsum_2 PRIVATE "-DCONTRACT_OPT=-1" "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_einsum_2 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_einsum_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
