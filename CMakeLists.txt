cmake_minimum_required(VERSION 3.21)

project(edge_audio
    VERSION 1.0.0
    DESCRIPTION "Edge speech admission and voice-command runtime around whisper.cpp"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EDGE_AUDIO_BUILD_TESTS "Build the unit and integration tests" ${PROJECT_IS_TOP_LEVEL})
option(EDGE_AUDIO_NATIVE "Tune for the build host CPU (-march=native)" OFF)
option(EDGE_AUDIO_WERROR "Treat compiler warnings as errors" OFF)
option(GGML_CUDA "Build whisper.cpp with the CUDA backend" OFF)

include(cmake/CompilerOptions.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)
add_subdirectory(apps)

if(EDGE_AUDIO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "edge_audio ${PROJECT_VERSION}: build=${CMAKE_BUILD_TYPE} "
               "cuda=${GGML_CUDA} native=${EDGE_AUDIO_NATIVE} tests=${EDGE_AUDIO_BUILD_TESTS}")
