# edge_audio_core: DSP, segmentation, quality admission, command routing and
# reporting. Pure C++17 with no model runtime dependency, so it is fully testable
# without whisper.cpp or model weights.
add_library(edge_audio_core STATIC
    edge_audio/audio/audio_io.cpp
    edge_audio/audio/raw_pcm_decoder.cpp
    edge_audio/segmentation/fixed_chunker.cpp
    edge_audio/segmentation/vad.cpp
    edge_audio/segmentation/utterance_segmenter.cpp
    edge_audio/quality/spectral_features.cpp
    edge_audio/quality/rule_gate.cpp
    edge_audio/quality/scene_classifier.cpp
    edge_audio/quality/adaptive_controller.cpp
    edge_audio/quality/file_features.cpp
    edge_audio/quality/quality_model.cpp
    edge_audio/quality/admission_policy.cpp
    edge_audio/quality/utterance_analyzer.cpp
    edge_audio/command/command_router.cpp
    edge_audio/command/command_event.cpp
    edge_audio/command/voice_command_loop.cpp
    edge_audio/report/json.cpp
    edge_audio/report/run_logger.cpp)
add_library(edge_audio::core ALIAS edge_audio_core)
target_include_directories(edge_audio_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(edge_audio_core
    PUBLIC  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:m>
    PRIVATE edge_audio::options dr_wav)

# edge_audio_asr: the whisper.cpp adapter. Kept separate so the core library and
# its tests never link a model runtime.
add_library(edge_audio_asr STATIC edge_audio/asr/asr_engine.cpp)
add_library(edge_audio::asr ALIAS edge_audio_asr)
target_include_directories(edge_audio_asr PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(edge_audio_asr PUBLIC edge_audio_core PRIVATE edge_audio::options whisper)
if(GGML_CUDA)
    target_compile_definitions(edge_audio_asr PRIVATE EDGE_AUDIO_USE_GPU=1)
endif()
