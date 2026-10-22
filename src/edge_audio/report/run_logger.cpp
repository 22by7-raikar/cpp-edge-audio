#include "edge_audio/report/run_logger.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "edge_audio/report/json.h"

namespace edge_audio {
namespace {

// Tabs would break the key=value line format.
std::string single_field(std::string text) {
    for (char& c : text) {
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    }
    return text;
}

std::string fixed(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

const char* boolean(bool value) {
    return value ? "true" : "false";
}

}  // namespace

void RunLogger::log_run_start(const RunConfig& cfg) {
    run_cfg_ = cfg;
    chunks_.clear();
    has_quality_summary_ = false;

    std::cout << "event=run_start" << "\tinput=" << cfg.input_path
              << "\tmodel=" << (cfg.model_path.empty() ? "none" : cfg.model_path)
              << "\tsegmentation=" << cfg.segmentation << "\tchunk_ms=" << cfg.chunk_ms
              << "\tthreads=" << cfg.n_threads << "\tgate=" << (cfg.gate_enabled ? 1 : 0)
              << "\tquality_policy=" << cfg.quality_policy
              << "\tquality_threshold=" << cfg.quality_threshold << '\n';
}

void RunLogger::log_chunk(const Chunk& chunk, const GateResult& gate, const AsrResult& asr,
                          const SceneResult& scene) {
    const GateMetrics& m = gate.metrics;
    std::cout << std::fixed << std::setprecision(4) << "event=chunk" << "\tidx=" << chunk.index
              << "\tstart=" << chunk.start_sec << "\tend=" << chunk.end_sec
              << "\tdecision=" << gate_decision_str(gate.decision) << "\treason=" << gate.reason
              << "\trms=" << m.rms << "\tsilence=" << m.silence_ratio
              << "\tclip=" << m.clipping_ratio << "\tzcr=" << std::setprecision(1) << m.zcr
              << std::setprecision(4) << "\tflatness=" << m.spectral_flatness
              << "\tcentroid=" << std::setprecision(1) << m.spectral_centroid
              << "\trolloff=" << m.spectral_rolloff << std::setprecision(4)
              << "\tflux=" << m.spectral_flux << "\tbl=" << m.band_energy_low
              << "\tbm=" << m.band_energy_mid << "\tbh=" << m.band_energy_high
              << "\tactive=" << m.active_frame_frac << "\tscene=" << scene_label_str(scene.label)
              << "\tinfer_ms=" << std::setprecision(1) << asr.inference_ms;
    if (asr.ok && !asr.text.empty()) {
        std::cout << "\ttext=" << single_field(asr.text);
    } else if (!asr.ok && !asr.error.empty()) {
        std::cout << "\tasr_error=" << single_field(asr.error);
    }
    std::cout << std::defaultfloat << '\n';

    if (!json_path_.empty()) {
        chunks_.push_back(ChunkRecord{chunk.index, chunk.start_sec, chunk.end_sec, gate,
                                      scene_label_str(scene.label), asr.inference_ms,
                                      asr.ok ? asr.text : std::string{}});
    }
}

void RunLogger::log_quality_summary(const QualitySummaryRecord& record) {
    quality_summary_ = record;
    has_quality_summary_ = true;

    std::ostringstream out;
    out << std::setprecision(17) << "event=quality_summary" << "\tpolicy=" << record.policy
        << "\tchunk_count=" << record.chunk_count << "\tschema_version=" << record.schema_version;
    if (record.learned_evaluated) {
        out << "\tlearned_raw_score=" << record.learned_raw_score
            << "\tlearned_probability=" << record.learned_probability
            << "\tlearned_inference_us=" << record.learned_inference_us;
    }
    out << "\tthreshold=" << record.threshold;
    if (record.learned_evaluated) {
        out << "\tlearned_decision=" << (record.learned_decision ? "admit" : "reject");
    }
    out << "\trule_summary=" << (record.rule_summary ? "admit" : "reject")
        << "\tfinal_admission=" << (record.final_admission ? "admit" : "reject")
        << "\tasr_ran=" << (record.asr_ran ? 1 : 0) << "\trejection_reason="
        << (record.rejection_reason.empty() ? "none" : record.rejection_reason)
        << "\tasr_infer_ms=" << record.asr_inference_ms;
    if (!record.transcript.empty()) {
        out << "\ttext=" << single_field(record.transcript);
    } else if (!record.asr_error.empty()) {
        out << "\tasr_error=" << single_field(record.asr_error);
    }
    if (record.debug_features && record.features_available) {
        out << "\tquality_features=";
        for (std::size_t i = 0; i < record.features.size(); ++i) {
            out << (i > 0 ? "," : "") << record.features[i];
        }
    }
    std::cout << out.str() << '\n';
}

bool RunLogger::log_run_end(int total_chunks, int passed, int failed, int borderline,
                            double total_audio_sec, double total_inference_ms) {
    const double rtf =
        total_audio_sec > 0.0 ? (total_inference_ms / 1000.0) / total_audio_sec : 0.0;
    const double accept_rate =
        total_chunks > 0 ? static_cast<double>(passed + borderline) / total_chunks : 0.0;

    std::cout << std::fixed << std::setprecision(3) << "event=run_end"
              << "\ttotal_chunks=" << total_chunks << "\tpassed=" << passed
              << "\tborderline=" << borderline << "\tfailed=" << failed
              << "\taccept_rate=" << accept_rate << "\taudio_sec=" << total_audio_sec
              << "\tinfer_ms=" << total_inference_ms << "\trtf=" << rtf << std::defaultfloat
              << '\n';

    if (json_path_.empty()) return true;
    return write_json(total_chunks, passed, failed, borderline, total_audio_sec, total_inference_ms,
                      rtf, accept_rate);
}

bool RunLogger::write_json(int total_chunks, int passed, int failed, int borderline,
                           double audio_sec, double infer_ms, double rtf,
                           double accept_rate) const {
    std::ofstream f(json_path_, std::ios::out | std::ios::trunc);
    if (!f) {
        std::cerr << "error: cannot write JSON report: " << json_path_ << '\n';
        return false;
    }

    const RunConfig& c = run_cfg_;
    f << "{\n  \"config\": {\n"
      << "    \"input\": " << json_quote(c.input_path) << ",\n"
      << "    \"model\": " << json_quote(c.model_path) << ",\n"
      << "    \"segmentation\": " << json_quote(c.segmentation) << ",\n"
      << "    \"chunk_ms\": " << c.chunk_ms << ",\n"
      << "    \"n_threads\": " << c.n_threads << ",\n"
      << "    \"gate_enabled\": " << boolean(c.gate_enabled) << ",\n"
      << "    \"quality_policy\": " << json_quote(c.quality_policy) << ",\n"
      << "    \"quality_threshold\": " << fixed(c.quality_threshold, 5) << "\n  },\n";

    f << "  \"chunks\": [\n";
    for (std::size_t i = 0; i < chunks_.size(); ++i) {
        const ChunkRecord& r = chunks_[i];
        const GateMetrics& m = r.gate.metrics;
        f << "    {\"idx\": " << r.index << ", \"start_sec\": " << fixed(r.start_sec, 4)
          << ", \"end_sec\": " << fixed(r.end_sec, 4)
          << ", \"decision\": " << json_quote(gate_decision_str(r.gate.decision))
          << ", \"reason\": " << json_quote(r.gate.reason) << ", \"rms\": " << fixed(m.rms, 5)
          << ", \"silence_ratio\": " << fixed(m.silence_ratio, 5)
          << ", \"clipping_ratio\": " << fixed(m.clipping_ratio, 5)
          << ", \"zcr\": " << fixed(m.zcr, 2) << ", \"flatness\": " << fixed(m.spectral_flatness, 5)
          << ", \"centroid_hz\": " << fixed(m.spectral_centroid, 2)
          << ", \"rolloff_hz\": " << fixed(m.spectral_rolloff, 2)
          << ", \"flux\": " << fixed(m.spectral_flux, 5)
          << ", \"band_low\": " << fixed(m.band_energy_low, 5)
          << ", \"band_mid\": " << fixed(m.band_energy_mid, 5)
          << ", \"band_high\": " << fixed(m.band_energy_high, 5)
          << ", \"active_frac\": " << fixed(m.active_frame_frac, 5)
          << ", \"scene\": " << json_quote(r.scene) << ", \"infer_ms\": " << fixed(r.infer_ms, 2)
          << ", \"transcript\": " << json_quote(r.text) << '}'
          << (i + 1 < chunks_.size() ? "," : "") << '\n';
    }
    f << "  ],\n";

    if (has_quality_summary_) {
        const QualitySummaryRecord& q = quality_summary_;
        const auto learned = [&q](double value, int precision) {
            return q.learned_evaluated ? fixed(value, precision) : std::string("null");
        };
        f << "  \"quality\": {\n"
          << "    \"policy\": " << json_quote(q.policy) << ",\n"
          << "    \"chunk_count\": " << q.chunk_count << ",\n"
          << "    \"schema_version\": " << json_quote(q.schema_version) << ",\n"
          << "    \"learned_raw_score\": " << learned(q.learned_raw_score, 17) << ",\n"
          << "    \"learned_probability\": " << learned(q.learned_probability, 17) << ",\n"
          << "    \"learned_inference_us\": " << learned(q.learned_inference_us, 3) << ",\n"
          << "    \"threshold\": " << fixed(q.threshold, 5) << ",\n"
          << "    \"learned_decision\": "
          << (q.learned_evaluated ? boolean(q.learned_decision) : "null") << ",\n"
          << "    \"rule_summary\": " << boolean(q.rule_summary) << ",\n"
          << "    \"final_admission\": " << boolean(q.final_admission) << ",\n"
          << "    \"asr_ran\": " << boolean(q.asr_ran) << ",\n"
          << "    \"rejection_reason\": "
          << json_quote(q.rejection_reason.empty() ? "none" : q.rejection_reason) << ",\n"
          << "    \"asr_inference_ms\": " << fixed(q.asr_inference_ms, 3) << ",\n"
          << "    \"transcript\": " << json_quote(q.transcript) << ",\n"
          << "    \"asr_error\": " << json_quote(q.asr_error);
        if (q.debug_features && q.features_available) {
            f << ",\n    \"features\": [";
            for (std::size_t i = 0; i < q.features.size(); ++i) {
                f << (i > 0 ? ", " : "") << fixed(q.features[i], 9);
            }
            f << ']';
        }
        f << "\n  },\n";
    }

    f << "  \"summary\": {\n"
      << "    \"total_chunks\": " << total_chunks << ",\n"
      << "    \"passed\": " << passed << ",\n"
      << "    \"borderline\": " << borderline << ",\n"
      << "    \"failed\": " << failed << ",\n"
      << "    \"accept_rate\": " << fixed(accept_rate, 4) << ",\n"
      << "    \"audio_sec\": " << fixed(audio_sec, 3) << ",\n"
      << "    \"total_infer_ms\": " << fixed(infer_ms, 1) << ",\n"
      << "    \"rtf\": " << fixed(rtf, 5) << "\n  }\n}\n";
    return static_cast<bool>(f);
}

}  // namespace edge_audio
