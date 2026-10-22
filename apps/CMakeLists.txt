# audio_pipeline: offline admission + transcription of a complete WAV file.
add_executable(audio_pipeline audio_pipeline.cpp)
target_link_libraries(audio_pipeline PRIVATE edge_audio::asr edge_audio::options)

# voice_command_stream: streaming PCM on stdin -> utterances -> admission -> ASR
# -> command events (JSON Lines on stdout).
add_executable(voice_command_stream voice_command_stream.cpp)
target_link_libraries(voice_command_stream PRIVATE edge_audio::asr edge_audio::options)

# evaluate_policies: native rule/learned/hybrid evaluation over labelled WAVs.
add_executable(evaluate_policies evaluate_policies.cpp)
target_link_libraries(evaluate_policies PRIVATE edge_audio::core edge_audio::options)

# benchmark_quality: latency of feature aggregation and native model inference.
add_executable(benchmark_quality benchmark_quality.cpp)
target_link_libraries(benchmark_quality PRIVATE edge_audio::core edge_audio::options)
