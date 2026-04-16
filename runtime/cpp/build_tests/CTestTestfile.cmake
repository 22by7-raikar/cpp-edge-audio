# CMake generated Testfile for 
# Source directory: /home/apr/Personal/cpp-edge-audio/runtime/cpp
# Build directory: /home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(pipeline_unit_tests "/home/apr/Personal/cpp-edge-audio/runtime/cpp/build_tests/test_pipeline")
set_tests_properties(pipeline_unit_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/apr/Personal/cpp-edge-audio/runtime/cpp/CMakeLists.txt;161;add_test;/home/apr/Personal/cpp-edge-audio/runtime/cpp/CMakeLists.txt;0;")
subdirs("whisper_build")
