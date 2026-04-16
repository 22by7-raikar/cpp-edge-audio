# Install script for directory: /home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so.0.9.8"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src/libggml.so.0.9.8"
    "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src/libggml.so.0"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so.0.9.8"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src:"
           NEW_RPATH "")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src/libggml.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so"
         OLD_RPATH "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-cpu.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-alloc.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-backend.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-blas.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-cann.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-cpp.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-cuda.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-opt.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-metal.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-rpc.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-virtgpu.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-sycl.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-vulkan.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-webgpu.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-zendnn.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/ggml-openvino.h"
    "/home/apr/Personal/cpp-edge-audio/vendor/whisper.cpp/ggml/include/gguf.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so.0.9.8"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src/libggml-base.so.0.9.8"
    "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src/libggml-base.so.0"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so.0.9.8"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/src/libggml-base.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/ggml-config.cmake"
    "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/whisper_build/ggml/ggml-version.cmake"
    )
endif()

