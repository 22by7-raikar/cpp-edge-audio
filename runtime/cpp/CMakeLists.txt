cmake_minimum_required(VERSION 3.16)
project(audio_ml_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_Debug "-O0 -g")

# whisper.cpp backend toggle
option(GGML_CUDA "Enable CUDA backend in whisper.cpp" OFF)

# -------------------------------------------------------------------
# M6 Optimization options  (all OFF by default; enabled via -D flags
# or scripts/build_optimized.sh)
# -------------------------------------------------------------------

# -DOPTIMIZE_NATIVE=ON  : -march=native -mtune=native
# Generates code tuned to the build host CPU. Not portable across machines.
option(OPTIMIZE_NATIVE "Compile with -march=native -mtune=native" OFF)

# -DOPTIMIZE_LTO=ON     : Link-time optimization (requires gold or lld)
option(OPTIMIZE_LTO    "Enable link-time optimization (LTO)"       OFF)

# -DOPTIMIZE_FAST_MATH=ON : -ffast-math
# Enables reciprocal estimates, reassociation, NaN/Inf elision.
# Gives ~10-20% speedup on FFT-heavy code; acceptable for audio ML inference.
# Disable if exact IEEE compliance is required.
option(OPTIMIZE_FAST_MATH "Enable -ffast-math"                     OFF)

# -DBUILD_PROFILING=ON  : compile with -pg for gprof profiling
option(BUILD_PROFILING "Compile with -pg for gprof profiling"       OFF)

if(OPTIMIZE_NATIVE)
    add_compile_options(-march=native -mtune=native)
    message(STATUS "Optimization : native arch (-march=native -mtune=native)")
endif()

if(OPTIMIZE_FAST_MATH)
    add_compile_options(-ffast-math)
    message(STATUS "Optimization : fast-math enabled")
endif()

if(OPTIMIZE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_why)
    if(_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "Optimization : LTO enabled")
    else()
        message(WARNING "LTO requested but not supported: ${_ipo_why}")
    endif()
endif()

if(BUILD_PROFILING)
    add_compile_options(-pg)
    add_link_options(-pg)
    message(STATUS "Profiling    : gprof instrumentation enabled (-pg)")
endif()

# -------------------------------------------------------------------
# whisper.cpp
# Expects the submodule at vendor/whisper.cpp (repo root).
# Build with: cmake -DWHISPER_ROOT=<path> .. to override.
# -------------------------------------------------------------------
if(NOT WHISPER_ROOT)
    set(WHISPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/whisper.cpp")
endif()

if(NOT EXISTS "${WHISPER_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "whisper.cpp not found at ${WHISPER_ROOT}.\n"
        "Run: git submodule update --init from repo root, or set -DWHISPER_ROOT=<path>.")
endif()

# Ensure the subproject sees this cache variable.
set(GGML_CUDA ${GGML_CUDA} CACHE BOOL "Enable CUDA backend in whisper.cpp" FORCE)

add_subdirectory(${WHISPER_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/whisper_build)

# -------------------------------------------------------------------
# Pipeline executable
# -------------------------------------------------------------------
add_executable(audio_pipeline
    src/main.cpp
    src/audio/audio_io.cpp
    src/chunker/chunker.cpp
    src/chunker/vad.cpp
    src/gate/features.cpp
    src/gate/gate.cpp
    src/asr/asr.cpp
    src/scene/scene.cpp
    src/scene/adaptive.cpp
    src/logging/logger.cpp
)

target_include_directories(audio_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # dr_wav.h is now vendored at src/audio/dr_wav.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${WHISPER_ROOT}
)

target_link_libraries(audio_pipeline PRIVATE whisper)

if(GGML_CUDA)
    target_compile_definitions(audio_pipeline PRIVATE GGML_CUDA=1)
    message(STATUS "Backend      : CUDA requested (GGML_CUDA=ON)")
else()
    message(STATUS "Backend      : CPU only (GGML_CUDA=OFF)")
endif()

# math library (needed on some Linux configurations)
find_library(MATH_LIB m)
if(MATH_LIB)
    target_link_libraries(audio_pipeline PRIVATE ${MATH_LIB})
endif()

# Optional: threads
find_package(Threads REQUIRED)
target_link_libraries(audio_pipeline PRIVATE Threads::Threads)

message(STATUS "Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "WHISPER_ROOT : ${WHISPER_ROOT}")
message(STATUS "Install to   : ${CMAKE_INSTALL_PREFIX}")

# -------------------------------------------------------------------
# Tests (no external framework; self-contained test runner)
# Enable with: cmake -DBUILD_TESTS=ON ..
# -------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()

    # pipeline_modules: an object library so test runner can link all modules
    # without pulling in main.cpp or whisper (ASR is not tested here).
    add_library(pipeline_modules OBJECT
        src/audio/audio_io.cpp
        src/chunker/chunker.cpp
        src/chunker/vad.cpp
        src/gate/features.cpp
        src/gate/gate.cpp
        src/scene/scene.cpp
        src/scene/adaptive.cpp
        src/logging/logger.cpp
    )
    target_include_directories(pipeline_modules PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
        ${WHISPER_ROOT}
    )
    # pipeline_modules does not call whisper APIs directly; asr.cpp is excluded.
    # If a future test needs ASR, link whisper there explicitly.
    find_library(MATH_LIB_TESTS m)
    if(MATH_LIB_TESTS)
        target_link_libraries(pipeline_modules PUBLIC ${MATH_LIB_TESTS})
    endif()
    find_package(Threads REQUIRED)
    target_link_libraries(pipeline_modules PUBLIC Threads::Threads)

    add_executable(test_pipeline tests/test_main.cpp $<TARGET_OBJECTS:pipeline_modules>)
    target_include_directories(test_pipeline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
        ${WHISPER_ROOT}
    )
    target_link_libraries(test_pipeline PRIVATE pipeline_modules)

    add_test(NAME pipeline_unit_tests COMMAND test_pipeline)
endif()
