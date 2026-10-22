/// @file
/// @brief Structured output of the offline file pipeline.
/// @ingroup report

#pragma once

#include <array>
#include <string>
#include <vector>

#include "edge_audio/asr/transcription.h"
#include "edge_audio/quality/quality_model.h"
#include "edge_audio/quality/rule_gate.h"
#include "edge_audio/quality/scene_classifier.h"
#include "edge_audio/segmentation/fixed_chunker.h"

namespace edge_audio {

/// @defgroup report Reporting
/// @brief Machine-readable output: per-chunk lines and an optional JSON report
/// for the file pipeline (RunLogger), JSON Lines events for the stream
/// (serialize_command_event()).
/// @{

/// @brief Run parameters echoed into the report.
struct RunConfig {
    std::string input_path;               ///< Input WAV.
    std::string model_path;               ///< Whisper model; empty in analysis-only runs.
    std::string segmentation = "fixed";   ///< `fixed`, `vad` or `vad-packed`.
    int chunk_ms = 5000;                  ///< Fixed window length.
    int n_threads = 4;                    ///< ASR threads.
    bool gate_enabled = true;             ///< Rule gate on.
    std::string quality_policy = "rule";  ///< Admission policy name.
    double quality_threshold = kQualityDefaultThreshold;  ///< Learned threshold.
};

/// @brief File-level admission outcome.
struct QualitySummaryRecord {
    std::string policy = "rule";                              ///< Policy name.
    int chunk_count = 0;                                      ///< Analysis windows.
    std::string schema_version = "quality-file-features-v1";  ///< Feature schema.
    bool features_available = false;                          ///< #features is populated.
    std::array<float, kQualityFeatureCount> features{};       ///< Learned-model input.
    bool debug_features = false;                              ///< Include #features in the output.
    bool learned_evaluated = false;                           ///< Learned model ran.
    double learned_raw_score = 0.0;                           ///< Ensemble log-odds.
    double learned_probability = 0.0;                         ///< Model probability.
    double learned_inference_us = 0.0;                        ///< Model evaluation time.
    double threshold = kQualityDefaultThreshold;              ///< Decision threshold.
    bool learned_decision = false;                            ///< Learned verdict.
    bool rule_summary = false;                                ///< Rule verdict.
    bool final_admission = false;                             ///< Policy verdict.
    bool asr_ran = false;                                     ///< ASR invoked.
    std::string rejection_reason;                             ///< Empty when admitted.
    double asr_inference_ms = 0.0;                            ///< Total ASR time.
    std::string transcript;                                   ///< File transcript (learned/hybrid).
    std::string asr_error;                                    ///< ASR failure, if any.
};

/// @brief Writes one tab-separated `key=value` line per event to stdout, and
/// optionally a single JSON report when the run ends.
///
/// Line events: `run_start`, `chunk` (one per window), `quality_summary`,
/// `run_end`. The JSON report has `config`, `chunks`, `quality` and `summary`
/// sections with the same fields.
class RunLogger {
public:
    /// @brief Also write a JSON report to @p path at log_run_end(). Empty disables it.
    void set_json_path(const std::string& path) { json_path_ = path; }

    void log_run_start(const RunConfig& cfg);  ///< Record the configuration.

    /// @brief Record one window. @p asr is default-constructed if ASR did not run.
    void log_chunk(const Chunk& chunk, const GateResult& gate, const AsrResult& asr,
                   const SceneResult& scene = SceneResult{});

    void log_quality_summary(const QualitySummaryRecord& record);  ///< Record admission.

    /// @brief Print totals and real-time factor; write the JSON report if enabled.
    /// @return False if the JSON report could not be written.
    bool log_run_end(int total_chunks, int passed, int failed, int borderline,
                     double total_audio_sec, double total_inference_ms);

private:
    struct ChunkRecord {
        int index = 0;
        double start_sec = 0.0;
        double end_sec = 0.0;
        GateResult gate;
        std::string scene;
        double infer_ms = 0.0;
        std::string text;
    };

    bool write_json(int total_chunks, int passed, int failed, int borderline, double audio_sec,
                    double infer_ms, double rtf, double accept_rate) const;

    std::string json_path_;
    RunConfig run_cfg_;
    std::vector<ChunkRecord> chunks_;
    QualitySummaryRecord quality_summary_;
    bool has_quality_summary_ = false;
};

/// @}

}  // namespace edge_audio
