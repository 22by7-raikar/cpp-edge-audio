#pragma once

#include <array>
#include <fstream>
#include <string>
#include <vector>
#include "gate/gate.h"
#include "asr/asr.h"
#include "chunker/chunker.h"
#include "scene/scene.h"

namespace pipeline {

struct RunConfig {
    std::string input_path;
    std::string model_path;
    int         chunk_ms    = 5000;
    int         hop_ms      = 0;
    int         n_threads   = 4;
    bool        gate_enabled = true;
    std::string quality_policy = "rule";
    double      quality_threshold = 0.3;
};

struct QualitySummaryRecord {
    std::string policy = "rule";
    int chunk_count = 0;
    std::string schema_version = "quality-file-features-v1";
    bool features_available = false;
    std::array<float, 27> features{};
    bool debug_features = false;
    bool learned_evaluated = false;
    double learned_raw_score = 0.0;
    double learned_probability = 0.0;
    double learned_inference_us = 0.0;
    double threshold = 0.3;
    bool learned_decision = false;
    bool rule_summary = false;
    bool final_admission = false;
    bool asr_ran = false;
    std::string rejection_reason;
    double asr_inference_ms = 0.0;
    std::string transcript;
    std::string asr_error;
};

// Writes one structured log record per chunk to stdout and optionally a TSV file.
// When set_json_path() is called, also writes a JSON benchmark file at run end.
// JSON schema: { config:{...}, chunks:[{...},...], summary:{...} }
class Logger {
public:
    Logger() = default;

    // Open a TSV log file in addition to stdout. Pass "" to skip.
    bool open(const std::string& path);

    // Set path for JSON benchmark output. Written atomically at log_run_end().
    void set_json_path(const std::string& path);

    void log_run_start(const RunConfig& cfg);

    void log_chunk(
        const Chunk&      chunk,
        const GateResult& gate,
        const AsrResult&  asr,       // may be empty if gate rejected
        const SceneResult& scene = SceneResult{}
    );

    void log_quality_summary(const QualitySummaryRecord& record);

    void log_run_end(
        int    total_chunks,
        int    passed,
        int    failed,
        int    borderline,
        double total_audio_sec,
        double total_inference_ms
    );

    void close();
    ~Logger() { close(); }

private:
    // TSV file
    std::ofstream file_;
    bool          file_open_ = false;

    // JSON accumulation
    std::string json_path_;
    RunConfig   run_cfg_;

    struct ChunkRecord {
        int         idx       = 0;
        double      start_sec = 0.0;
        double      end_sec   = 0.0;
        std::string decision;
        std::string reason;
        // time-domain
        double rms            = 0.0;
        double silence_ratio  = 0.0;
        double clipping_ratio = 0.0;
        double zcr            = 0.0;
        // spectral
        double flatness       = 0.0;
        double centroid       = 0.0;
        double rolloff        = 0.0;
        double flux           = 0.0;
        double band_low       = 0.0;
        double band_mid       = 0.0;
        double band_high      = 0.0;
        double active_frac    = 0.0;
        // Scene
        std::string scene_label;
        // ASR
        double      infer_ms  = 0.0;
        bool        asr_ok    = false;
        std::string text;
    };
    std::vector<ChunkRecord> chunk_records_;
    QualitySummaryRecord quality_summary_;
    bool has_quality_summary_ = false;

    void write(const std::string& line);
    void write_json(
        int total_chunks, int passed, int failed, int borderline,
        double audio_sec, double infer_ms, double rtf, double accept_rate);
};

}  // namespace pipeline
