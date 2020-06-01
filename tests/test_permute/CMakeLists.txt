cmake_minimum_required(VERSION 3.1)
project(test_permute)

set(CMAKE_CXX_STANDARD 14)

enable_testing()

# default recursive CXX14
add_executable(test_permute_1 test_permute.cpp)
add_test(test_permute_1 test_permute_1)

if(MSVC)
    target_compile_options(test_permute_1 PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_permute_1 PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_permute_1 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_permute_1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


# while loop variant CXX14
add_executable(test_permute_2 test_permute.cpp)
add_test(test_permute_2 test_permute_2)

if(MSVC)
    target_compile_options(test_permute_2 PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_permute_2 PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_permute_2 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_permute_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_compile_definitions(test_permute_2 PRIVATE CONTRACT_OPT=-1)


# default recursive CXX17
add_executable(test_permute_3 test_permute.cpp)
add_test(test_permute_3 test_permute_3)
set_property(TARGET test_permute_3 PROPERTY CXX_STANDARD 17)

if(MSVC)
    target_compile_options(test_permute_3 PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_permute_3 PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_permute_3 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_permute_3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)


# while loop variant CXX17
add_executable(test_permute_4 test_permute.cpp)
add_test(test_permute_4 test_permute_4)
set_property(TARGET test_permute_4 PROPERTY CXX_STANDARD 17)

if(MSVC)
    target_compile_options(test_permute_4 PRIVATE "/W2" "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(test_permute_4 PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:RELEASE>:-march=native>")
endif()

target_include_directories(test_permute_4 PRIVATE ${FASTOR_INCLUDE_DIR})
target_include_directories(test_permute_4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_compile_definitions(test_permute_4 PRIVATE CONTRACT_OPT=-1)


