cmake_minimum_required(VERSION 3.1)
project(fastor VERSION 0.6.4)

option(BUILD_TESTING "Build the testing tree." ON)

set(FASTOR_SOURCE_DIR "${fastor_SOURCE_DIR}")
set(FASTOR_BINARY_DIR "${fastor_BINARY_DIR}")
set(FASTOR_INCLUDE_DIR "${fastor_SOURCE_DIR}")

include(GNUInstallDirs)

set(FASTOR_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(FASTOR_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(FASTOR_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_DATADIR}/cmake/Fastor")
set(FASTOR_INSTALL_PKGCONFIG_DIR "${CMAKE_INSTALL_DATADIR}/pkgconfig")

add_library(Fastor INTERFACE)

target_include_directories(
    Fastor INTERFACE
        $<BUILD_INTERFACE:${FASTOR_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${FASTOR_INSTALL_INCLUDE_DIR}>)

if(BUILD_TESTING)
    enable_testing()

    add_subdirectory(tests)
endif()

install(
    DIRECTORY "${FASTOR_INCLUDE_DIR}/Fastor"
    DESTINATION "${FASTOR_INSTALL_INCLUDE_DIR}")
install(
    TARGETS Fastor
    EXPORT FastorTargets)

export(
    TARGETS Fastor
    NAMESPACE Fastor::
    FILE "${FASTOR_BINARY_DIR}/FastorTargets.cmake")
export(PACKAGE Fastor)
install(
    EXPORT FastorTargets
    NAMESPACE Fastor::
    DESTINATION "${FASTOR_INSTALL_CMAKE_DIR}")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${FASTOR_SOURCE_DIR}/data/FastorConfig.cmake.in"
    "${FASTOR_BINARY_DIR}/FastorConfig.cmake"
    INSTALL_DESTINATION "${FASTOR_INSTALL_CMAKE_DIR}"
    PATH_VARS FASTOR_INSTALL_INCLUDE_DIR)
write_basic_package_version_file(
    "${FASTOR_BINARY_DIR}/FastorConfigVersion.cmake"
    VERSION ${fastor_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)
install(
    FILES
        "${FASTOR_BINARY_DIR}/FastorConfigVersion.cmake"
        "${FASTOR_BINARY_DIR}/FastorConfig.cmake"
    DESTINATION "${FASTOR_INSTALL_CMAKE_DIR}")

configure_file(
    "${FASTOR_SOURCE_DIR}/data/fastor.pc.in"
    "${FASTOR_BINARY_DIR}/fastor.pc" @ONLY)
install(
    FILES "${FASTOR_BINARY_DIR}/fastor.pc"
    DESTINATION "${FASTOR_INSTALL_PKGCONFIG_DIR}")
