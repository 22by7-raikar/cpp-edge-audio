# Third-party dependencies.
#
#   whisper.cpp  git submodule at vendor/whisper.cpp (pinned commit)
#   dr_wav       single header in third_party/dr_wav (public domain / MIT-0)
#   GoogleTest   system package if present, otherwise fetched at a pinned tag

set(WHISPER_ROOT "${PROJECT_SOURCE_DIR}/vendor/whisper.cpp" CACHE PATH "whisper.cpp source tree")
if(NOT EXISTS "${WHISPER_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "whisper.cpp not found at ${WHISPER_ROOT}. "
        "Run `git submodule update --init` or pass -DWHISPER_ROOT=<path>.")
endif()

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${WHISPER_ROOT}" "${CMAKE_BINARY_DIR}/_deps/whisper" EXCLUDE_FROM_ALL)
# Treat vendored headers as system headers so our warning flags do not apply to them.
foreach(_target whisper ggml ggml-base)
    if(TARGET ${_target})
        get_target_property(_includes ${_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(_includes)
            set_target_properties(${_target} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_includes}")
        endif()
    endif()
endforeach()

add_library(dr_wav INTERFACE)
target_include_directories(dr_wav SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/dr_wav")

if(EDGE_AUDIO_BUILD_TESTS)
    find_package(GTest 1.12 QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2
            GIT_SHALLOW    TRUE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    include(GoogleTest)
endif()
