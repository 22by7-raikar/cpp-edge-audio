// logger.cpp
// Structured per-chunk logging:
//   - TSV to stdout (always) and optionally a .log file
//   - JSON to an optional bench file (--bench-json), written atomically at run end
// The JSON file is the primary input for tools/python/eval/compare_bench.py.

#include "logger.h"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace pipeline {

// -------------------------------------------------------
// TSV helpers
// -------------------------------------------------------

bool Logger::open(const std::string& path) {
    if (path.empty()) return true;
    file_.open(path, std::ios::out | std::ios::trunc);
    file_open_ = file_.is_open();
    return file_open_;
}

void Logger::set_json_path(const std::string& path) {
    json_path_ = path;
}

void Logger::close() {
    if (file_open_) {
        file_.close();
        file_open_ = false;
    }
}

void Logger::write(const std::string& line) {
    std::cout << line << '\n';
    if (file_open_) file_ << line << '\n';
}

void Logger::log_run_start(const RunConfig& cfg) {
    run_cfg_ = cfg;
    chunk_records_.clear();
    has_quality_summary_ = false;

    std::ostringstream oss;
    oss << "event=run_start"
        << "\tinput="    << cfg.input_path
        << "\tmodel="    << cfg.model_path
        << "\tchunk_ms=" << cfg.chunk_ms
        << "\thop_ms="   << cfg.hop_ms
        << "\tthreads="  << cfg.n_threads
        << "\tgate="     << (cfg.gate_enabled ? "1" : "0")
        << "\tquality_policy=" << cfg.quality_policy
        << "\tquality_threshold=" << cfg.quality_threshold;
    write(oss.str());
}

void Logger::log_quality_summary(const QualitySummaryRecord& record) {
    quality_summary_ = record;
    has_quality_summary_ = true;

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "event=quality_summary"
        << "\tpolicy=" << record.policy
        << "\tchunk_count=" << record.chunk_count
        << "\tschema_version=" << record.schema_version
        << "\tlearned_raw_score=";
    if (record.learned_evaluated) {
        oss << record.learned_raw_score;
    } else {
        oss << "not_run";
    }
    oss << "\tlearned_probability=";
    if (record.learned_evaluated) {
        oss << record.learned_probability;
    } else {
        oss << "not_run";
    }
    oss << "\tlearned_inference_us=";
    if (record.learned_evaluated) {
        oss << record.learned_inference_us;
    } else {
        oss << "not_run";
    }
    oss << "\tthreshold=" << record.threshold
        << "\tlearned_decision=";
    if (record.learned_evaluated) {
        oss << (record.learned_decision ? "admit" : "reject");
    } else {
        oss << "not_run";
    }
    oss << "\trule_summary=" << (record.rule_summary ? "admit" : "reject")
        << "\tfinal_admission=" << (record.final_admission ? "admit" : "reject")
        << "\tasr_ran=" << (record.asr_ran ? "1" : "0")
        << "\trejection_reason="
        << (record.rejection_reason.empty() ? "none" : record.rejection_reason)
        << "\tasr_infer_ms=" << record.asr_inference_ms;
    if (!record.transcript.empty()) {
        std::string text = record.transcript;
        for (char& c : text) if (c == '\t') c = ' ';
        oss << "\ttext=" << text;
    } else if (!record.asr_error.empty()) {
        oss << "\tasr_error=" << record.asr_error;
    }
    if (record.debug_features && record.features_available) {
        oss << "\tquality_features=";
        for (std::size_t index = 0; index < record.features.size(); ++index) {
            if (index > 0) oss << ',';
            oss << record.features[index];
        }
    }
    write(oss.str());
}

void Logger::log_chunk(
    const Chunk&      chunk,
    const GateResult& gate,
    const AsrResult&  asr,
    const SceneResult& scene)
{
    const GateMetrics& m = gate.metrics;

    // --- TSV line ---
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "event=chunk"
        << "\tidx="      << chunk.index
        << "\tstart="    << chunk.start_sec
        << "\tend="      << chunk.end_sec
        << "\tdur="      << chunk.duration_sec()
        << "\tdecision=" << gate_decision_str(gate.decision)
        << "\treason="   << gate.reason
        << "\trms="      << m.rms
        << "\tsilence="  << m.silence_ratio
        << "\tclip="     << m.clipping_ratio
        << "\tzcr="      << std::setprecision(1) << m.zcr
        << std::setprecision(4)
        << "\tflatness=" << m.spectral_flatness
        << "\tcentroid=" << std::setprecision(1) << m.spectral_centroid
        << "\trolloff="  << m.spectral_rolloff
        << std::setprecision(4)
        << "\tflux="     << m.spectral_flux
        << "\tbl="       << m.band_energy_low
        << "\tbm="       << m.band_energy_mid
        << "\tbh="       << m.band_energy_high
        << "\tactive="   << m.active_frame_frac
        << "\tscene="    << scene_label_str(scene.label)
        << std::setprecision(1)
        << "\tinfer_ms=" << asr.inference_ms;

    if (asr.ok && !asr.text.empty()) {
        std::string txt = asr.text;
        for (char& c : txt) if (c == '\t') c = ' ';
        oss << "\ttext=" << txt;
    } else if (!asr.ok && !asr.error.empty()) {
        oss << "\tasr_error=" << asr.error;
    }
    write(oss.str());

    // --- Accumulate for JSON ---
    if (!json_path_.empty()) {
        ChunkRecord r;
        r.idx            = chunk.index;
        r.start_sec      = chunk.start_sec;
        r.end_sec        = chunk.end_sec;
        r.decision       = gate_decision_str(gate.decision);
        r.reason         = gate.reason;
        r.rms            = m.rms;
        r.silence_ratio  = m.silence_ratio;
        r.clipping_ratio = m.clipping_ratio;
        r.zcr            = m.zcr;
        r.flatness       = m.spectral_flatness;
        r.centroid       = m.spectral_centroid;
        r.rolloff        = m.spectral_rolloff;
        r.flux           = m.spectral_flux;
        r.band_low       = m.band_energy_low;
        r.band_mid       = m.band_energy_mid;
        r.band_high      = m.band_energy_high;
        r.active_frac    = m.active_frame_frac;
        r.scene_label    = scene_label_str(scene.label);
        r.infer_ms       = asr.inference_ms;
        r.asr_ok         = asr.ok;
        r.text           = asr.ok ? asr.text : "";
        chunk_records_.push_back(std::move(r));
    }
}

void Logger::log_run_end(
    int    total_chunks,
    int    passed,
    int    failed,
    int    borderline,
    double total_audio_sec,
    double total_inference_ms)
{
    const double rtf = (total_audio_sec > 0.0)
        ? (total_inference_ms / 1000.0) / total_audio_sec
        : 0.0;

    const double accept_rate = (total_chunks > 0)
        ? static_cast<double>(passed + borderline) / total_chunks
        : 0.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "event=run_end"
        << "\ttotal_chunks=" << total_chunks
        << "\tpassed="       << passed
        << "\tborderline="   << borderline
        << "\tfailed="       << failed
        << "\taccept_rate="  << accept_rate
        << "\taudio_sec="    << total_audio_sec
        << "\tinfer_ms="     << total_inference_ms
        << "\trtf="          << rtf;
    write(oss.str());

    if (!json_path_.empty()) {
        write_json(total_chunks, passed, failed, borderline,
                   total_audio_sec, total_inference_ms, rtf, accept_rate);
    }
}

// -------------------------------------------------------
// JSON serializer (no external library)
// -------------------------------------------------------

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static std::string jstr(const std::string& s) {
    return '"' + json_escape(s) + '"';
}

static std::string jbool(bool v) { return v ? "true" : "false"; }

static std::string jd(double v, int prec = 5) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

void Logger::write_json(
    int total_chunks, int passed, int failed, int borderline,
    double audio_sec, double infer_ms, double rtf, double accept_rate)
{
    std::ofstream f(json_path_, std::ios::out | std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "WARNING: could not write JSON to: " << json_path_ << "\n";
        return;
    }

    const RunConfig& c = run_cfg_;
    f << "{\n";

    // --- config ---
    f << "  \"config\": {\n";
    f << "    \"input\":       " << jstr(c.input_path)            << ",\n";
    f << "    \"model\":       " << jstr(c.model_path)            << ",\n";
    f << "    \"chunk_ms\":    " << c.chunk_ms                    << ",\n";
    f << "    \"hop_ms\":      " << c.hop_ms                      << ",\n";
    f << "    \"n_threads\":   " << c.n_threads                   << ",\n";
    f << "    \"gate_enabled\": " << jbool(c.gate_enabled)        << ",\n";
    f << "    \"quality_policy\": " << jstr(c.quality_policy)      << ",\n";
    f << "    \"quality_threshold\": " << jd(c.quality_threshold)  << "\n";
    f << "  },\n";

    // --- chunks ---
    f << "  \"chunks\": [\n";
    for (size_t i = 0; i < chunk_records_.size(); ++i) {
        const ChunkRecord& r = chunk_records_[i];
        const bool last = (i + 1 == chunk_records_.size());
        f << "    {\n";
        f << "      \"idx\":            " << r.idx                          << ",\n";
        f << "      \"start_sec\":      " << jd(r.start_sec, 4)             << ",\n";
        f << "      \"end_sec\":        " << jd(r.end_sec, 4)               << ",\n";
        f << "      \"dur_sec\":        " << jd(r.end_sec - r.start_sec, 4) << ",\n";
        f << "      \"decision\":       " << jstr(r.decision)               << ",\n";
        f << "      \"reason\":         " << jstr(r.reason)                 << ",\n";
        f << "      \"rms\":            " << jd(r.rms)                      << ",\n";
        f << "      \"silence_ratio\":  " << jd(r.silence_ratio)            << ",\n";
        f << "      \"clipping_ratio\": " << jd(r.clipping_ratio)           << ",\n";
        f << "      \"zcr\":            " << jd(r.zcr, 2)                   << ",\n";
        f << "      \"flatness\":       " << jd(r.flatness)                 << ",\n";
        f << "      \"centroid_hz\":    " << jd(r.centroid, 2)              << ",\n";
        f << "      \"rolloff_hz\":     " << jd(r.rolloff, 2)               << ",\n";
        f << "      \"flux\":           " << jd(r.flux)                     << ",\n";
        f << "      \"band_low\":       " << jd(r.band_low)                 << ",\n";
        f << "      \"band_mid\":       " << jd(r.band_mid)                 << ",\n";
        f << "      \"band_high\":      " << jd(r.band_high)                << ",\n";
        f << "      \"active_frac\":    " << jd(r.active_frac)              << ",\n";
        f << "      \"scene\":          " << jstr(r.scene_label)            << ",\n";
        f << "      \"infer_ms\":       " << jd(r.infer_ms, 2)              << ",\n";
        f << "      \"transcript\":     " << jstr(r.text)                   << "\n";
        f << "    }" << (last ? "" : ",") << "\n";
    }
    f << "  ],\n";

    // --- file-level quality admission ---
    if (has_quality_summary_) {
        const QualitySummaryRecord& q = quality_summary_;
        f << "  \"quality\": {\n";
        f << "    \"policy\": " << jstr(q.policy) << ",\n";
        f << "    \"chunk_count\": " << q.chunk_count << ",\n";
        f << "    \"schema_version\": " << jstr(q.schema_version) << ",\n";
        f << "    \"learned_raw_score\": ";
        f << (q.learned_evaluated ? jd(q.learned_raw_score, 17) : "null") << ",\n";
        f << "    \"learned_probability\": ";
        f << (q.learned_evaluated ? jd(q.learned_probability, 17) : "null") << ",\n";
        f << "    \"learned_inference_us\": ";
        f << (q.learned_evaluated ? jd(q.learned_inference_us, 3) : "null") << ",\n";
        f << "    \"threshold\": " << jd(q.threshold) << ",\n";
        f << "    \"learned_decision\": ";
        f << (q.learned_evaluated ? jbool(q.learned_decision) : "null") << ",\n";
        f << "    \"rule_summary\": " << jbool(q.rule_summary) << ",\n";
        f << "    \"final_admission\": " << jbool(q.final_admission) << ",\n";
        f << "    \"asr_ran\": " << jbool(q.asr_ran) << ",\n";
        f << "    \"rejection_reason\": "
          << jstr(q.rejection_reason.empty() ? "none" : q.rejection_reason) << ",\n";
        f << "    \"asr_inference_ms\": " << jd(q.asr_inference_ms, 3) << ",\n";
        f << "    \"transcript\": " << jstr(q.transcript) << ",\n";
        f << "    \"asr_error\": " << jstr(q.asr_error);
        if (q.debug_features && q.features_available) {
            f << ",\n    \"features\": [";
            for (std::size_t index = 0; index < q.features.size(); ++index) {
                if (index > 0) f << ", ";
                f << jd(q.features[index], 9);
            }
            f << "]\n";
        } else {
            f << "\n";
        }
        f << "  },\n";
    }

    // --- summary ---
    f << "  \"summary\": {\n";
    f << "    \"total_chunks\": " << total_chunks             << ",\n";
    f << "    \"passed\":       " << passed                  << ",\n";
    f << "    \"borderline\":   " << borderline              << ",\n";
    f << "    \"failed\":       " << failed                  << ",\n";
    f << "    \"accept_rate\":  " << jd(accept_rate, 4)      << ",\n";
    f << "    \"audio_sec\":    " << jd(audio_sec, 3)        << ",\n";
    f << "    \"total_infer_ms\": " << jd(infer_ms, 1)       << ",\n";
    f << "    \"rtf\":          " << jd(rtf, 5)              << "\n";
    f << "  }\n";

    f << "}\n";
    f.flush();
}

}  // namespace pipeline
