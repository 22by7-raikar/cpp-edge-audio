// voice_command_stream: raw S16LE PCM on stdin -> one JSON command event per utterance.
//
//   arecord -f S16_LE -r 16000 -c 1 -t raw -q |
//       voice_command_stream --rate 16000 --channels 1 --model ggml-base.en.bin
//
// stdout carries only JSON Lines (one VoiceCommandEvent per utterance);
// diagnostics go to stderr. Exit status is non-zero on malformed input or if
// any admitted utterance failed ASR.

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "edge_audio/asr/asr_engine.h"
#include "edge_audio/audio/raw_pcm_decoder.h"
#include "edge_audio/command/voice_command_loop.h"
#include "edge_audio/quality/admission_policy.h"

namespace ea = edge_audio;

namespace {

struct Options {
    int rate = 0;
    int channels = 0;
    std::string model_path;
    int threads = 4;
    ea::QualityPolicy policy = ea::QualityPolicy::RULE;
    double threshold = ea::kQualityDefaultThreshold;
    std::string capture_device = "unknown";
    std::string transport = "stdin_pcm";
};

void print_usage(const char* program) {
    std::cerr << "usage: " << program
              << " --rate <hz> --channels <1|2> --model <path> [options]\n\n"
              << "  --policy <name>          rule | learned | hybrid (default: rule)\n"
              << "  --threshold <p>          learned probability threshold (default: "
              << ea::kQualityDefaultThreshold << ")\n"
              << "  --threads <n>            ASR threads (default: 4)\n"
              << "  --capture-device <label> recorded in every event (default: unknown)\n"
              << "  --transport <label>      recorded in every event (default: stdin_pcm)\n";
}

int parse_int(const std::string& option, const std::string& value) {
    std::size_t used = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != value.size()) {
        throw std::invalid_argument(option + " expects an integer");
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(arg + " requires a value");
            return argv[++i];
        };
        if (arg == "--rate")
            o.rate = parse_int(arg, value());
        else if (arg == "--channels")
            o.channels = parse_int(arg, value());
        else if (arg == "--model")
            o.model_path = value();
        else if (arg == "--threads")
            o.threads = parse_int(arg, value());
        else if (arg == "--policy")
            o.policy = ea::parse_quality_policy(value());
        else if (arg == "--threshold")
            o.threshold = ea::parse_quality_threshold(value());
        else if (arg == "--capture-device")
            o.capture_device = value();
        else if (arg == "--transport")
            o.transport = value();
        else
            throw std::invalid_argument("unknown option: " + arg);
    }

    if (o.rate <= 0 || o.rate > ea::RawPcmDecoder::kMaxSampleRate) {
        throw std::invalid_argument("--rate must be in [1, 384000]");
    }
    if ((o.rate * 20) % 1000 != 0) {
        throw std::invalid_argument("--rate must give whole 20 ms frames");
    }
    if (o.channels != 1 && o.channels != 2)
        throw std::invalid_argument("--channels must be 1 or 2");
    if (o.model_path.empty()) throw std::invalid_argument("--model is required");
    if (o.threads <= 0) throw std::invalid_argument("--threads must be positive");
    if (o.capture_device.empty() || o.transport.empty()) {
        throw std::invalid_argument("--capture-device and --transport must not be empty");
    }
    return o;
}

bool write_events(ea::VoiceCommandLoop& loop) {
    for (const auto& line : loop.pop_json_events()) {
        std::cout << line << '\n';
    }
    std::cout.flush();
    return static_cast<bool>(std::cout);
}

int run(const Options& o) {
    ea::AsrConfig asr_cfg;
    asr_cfg.model_path = o.model_path;
    asr_cfg.n_threads = o.threads;
    ea::AsrEngine asr(asr_cfg);
    if (!asr.ready()) throw std::runtime_error(asr.load_error());
    std::cerr << "[asr] backend=" << asr.backend_mode()
              << " cpu_fallback=" << (asr.used_cpu_fallback() ? "yes" : "no") << '\n';

    ea::VoiceCommandConfig cfg;
    cfg.segmenter.sample_rate = o.rate;
    cfg.quality_policy = o.policy;
    cfg.quality_threshold = o.threshold;
    cfg.capture_device = o.capture_device;
    cfg.transport = o.transport;
    ea::VoiceCommandLoop loop(cfg, asr.as_transcriber());

    ea::RawPcmDecoder decoder(std::cin, {o.rate, o.channels});
    std::vector<float> block;
    std::string decode_error;
    while (decoder.read_block(block, decode_error)) {
        loop.push(block.data(), static_cast<int>(block.size()));
        if (!write_events(loop)) throw std::runtime_error("cannot write to stdout");
    }
    loop.flush();
    if (!write_events(loop)) throw std::runtime_error("cannot write to stdout");
    if (!decode_error.empty()) throw std::runtime_error(decode_error);
    return loop.had_asr_error() ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 2;
    }
    try {
        return run(options);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
