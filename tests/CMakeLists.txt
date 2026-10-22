# Unit and integration tests. None of them needs whisper.cpp or model weights:
# ASR is replaced by an injected Transcriber.

add_executable(edge_audio_tests
    cpp/test_dsp_and_gate.cpp
    cpp/test_file_features_and_policy.cpp
    cpp/test_quality_model.cpp
    cpp/test_quality_model_parity.cpp
    cpp/test_raw_pcm_decoder.cpp
    cpp/test_utterance_segmenter.cpp
    cpp/test_utterance_analyzer.cpp
    cpp/test_voice_command_loop.cpp)
target_link_libraries(edge_audio_tests PRIVATE edge_audio::core GTest::gtest_main)
target_compile_definitions(edge_audio_tests PRIVATE
    EDGE_AUDIO_PARITY_CORPUS="${CMAKE_CURRENT_SOURCE_DIR}/data/quality_model_v1_parity.bin")
gtest_discover_tests(edge_audio_tests DISCOVERY_MODE PRE_TEST)

# End-to-end check of the streaming stdout protocol: the driver runs the real
# VoiceCommandLoop with a fake transcriber; the Python script validates the
# JSON Lines it prints and the CLI's argument validation.
add_executable(stream_protocol_driver cpp/stream_protocol_driver.cpp)
target_link_libraries(stream_protocol_driver PRIVATE edge_audio::core)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME stream_protocol
        COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_stream_protocol.py"
                $<TARGET_FILE:stream_protocol_driver> $<TARGET_FILE:voice_command_stream>)
endif()
