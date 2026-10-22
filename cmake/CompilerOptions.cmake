# Compiler settings applied to first-party targets only (never to vendored code).

add_library(edge_audio_options INTERFACE)
add_library(edge_audio::options ALIAS edge_audio_options)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edge_audio_options INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow)
    if(EDGE_AUDIO_WERROR)
        target_compile_options(edge_audio_options INTERFACE -Werror)
    endif()
    if(EDGE_AUDIO_NATIVE)
        target_compile_options(edge_audio_options INTERFACE -march=native -mtune=native)
    endif()
elseif(MSVC)
    target_compile_options(edge_audio_options INTERFACE /W4 /permissive-)
    if(EDGE_AUDIO_WERROR)
        target_compile_options(edge_audio_options INTERFACE /WX)
    endif()
endif()

# -ffast-math is deliberately not offered: the learned quality model must match the
# Python reference bit-for-bit, which requires strict IEEE-754 semantics.
