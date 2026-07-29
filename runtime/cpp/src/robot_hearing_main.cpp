#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "asr/asr.h"
#include "audio/raw_pcm_stream.h"
#include "gate/quality_model.h"
#include "gate/quality_policy.h"
#include "robot/robot_hearing_loop.h"

namespace {

struct CliArgs {
    std::string source;
    int input_rate = 0;
    int input_channels = 0;
    pipeline::QualityPolicy quality_policy = pipeline::QualityPolicy::RULE;
    double quality_threshold = pipeline::kQualityDefaultThreshold;
    std::string model_path;
    int threads = 4;
    std::string capture_device = "unknown";
    std::string transport = "stdin_pcm";
    bool have_input_rate = false;
    bool have_input_channels = false;
};

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " --source stdin --input-rate <hz> --input-channels <1|2>"
           " --quality-policy <rule|learned|hybrid>"
           " [--quality-threshold 0.3] --model <model.bin>"
           " [--threads 4] [--capture-device unknown]"
           " [--transport stdin_pcm]\n";
}

bool parse_int(const std::string& value, int& output) {
    std::size_t parsed = 0;
    try {
        output = std::stoi(value, &parsed);
    } catch (const std::exception&) {
        return false;
    }
    return parsed == value.size();
}

bool parse_args(int argc, char** argv, CliArgs& args) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto next = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    option + " requires an argument");
            }
            return argv[++index];
        };

        try {
            if (option == "--source") {
                args.source = next();
            } else if (option == "--input-rate") {
                args.have_input_rate = true;
                if (!parse_int(next(), args.input_rate)) {
                    throw std::invalid_argument(
                        "--input-rate must be an integer");
                }
            } else if (option == "--input-channels") {
                args.have_input_channels = true;
                if (!parse_int(next(), args.input_channels)) {
                    throw std::invalid_argument(
                        "--input-channels must be an integer");
                }
            } else if (option == "--quality-policy") {
                args.quality_policy =
                    pipeline::parse_quality_policy(next());
            } else if (option == "--quality-threshold") {
                args.quality_threshold =
                    pipeline::parse_quality_threshold(next());
            } else if (option == "--model") {
                args.model_path = next();
            } else if (option == "--threads") {
                if (!parse_int(next(), args.threads)) {
                    throw std::invalid_argument(
                        "--threads must be an integer");
                }
            } else if (option == "--capture-device") {
                args.capture_device = next();
            } else if (option == "--transport") {
                args.transport = next();
            } else {
                throw std::invalid_argument("unknown argument: " + option);
            }
        } catch (const std::exception& error) {
            std::cerr << "ERROR: " << error.what() << "\n";
            return false;
        }
    }

    if (args.source != "stdin") {
        std::cerr << "ERROR: --source stdin is required\n";
        return false;
    }
    if (!args.have_input_rate || args.input_rate <= 0 ||
        args.input_rate > pipeline::RawPcmDecoder::kMaxSampleRate) {
        std::cerr << "ERROR: --input-rate must be in [1, "
                  << pipeline::RawPcmDecoder::kMaxSampleRate << "]\n";
        return false;
    }
    if ((args.input_rate * 20) % 1000 != 0) {
        std::cerr
            << "ERROR: --input-rate must produce whole 20 ms frames\n";
        return false;
    }
    if (!args.have_input_channels ||
        (args.input_channels != 1 && args.input_channels != 2)) {
        std::cerr << "ERROR: --input-channels must be 1 or 2\n";
        return false;
    }
    if (args.quality_threshold != pipeline::kQualityDefaultThreshold) {
        std::cerr
            << "ERROR: robot hearing requires --quality-threshold 0.3\n";
        return false;
    }
    if (args.model_path.empty()) {
        std::cerr << "ERROR: --model is required\n";
        return false;
    }
    if (args.threads <= 0) {
        std::cerr << "ERROR: --threads must be > 0\n";
        return false;
    }
    if (args.capture_device.empty() || args.transport.empty()) {
        std::cerr
            << "ERROR: --capture-device and --transport must be non-empty\n";
        return false;
    }
    return true;
}

bool write_events(pipeline::RobotHearingLoop& loop) {
    for (const auto& line : loop.pop_json_events()) {
        std::cout << line << '\n';
        if (!std::cout) {
            std::cerr << "ERROR: failed to write robot JSON event\n";
            return false;
        }
    }
    std::cout.flush();
    return static_cast<bool>(std::cout);
}

}  // namespace

int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        pipeline::AsrConfig asr_config;
        asr_config.model_path = args.model_path;
        asr_config.n_threads = args.threads;
        pipeline::AsrEngine asr(asr_config);
        if (!asr.ready()) {
            std::cerr << "ERROR loading model: " << asr.load_error() << "\n";
            return 1;
        }
        std::cerr
            << "[asr] backend_requested="
            << (asr.gpu_requested() ? "cuda" : "cpu")
            << " backend_active=" << asr.backend_mode()
            << " cpu_fallback=" << (asr.used_cpu_fallback() ? "yes" : "no")
            << "\n";

        pipeline::RobotHearingConfig loop_config;
        loop_config.segmenter.sample_rate = args.input_rate;
        loop_config.quality_policy = args.quality_policy;
        loop_config.quality_threshold = args.quality_threshold;
        loop_config.capture_device = args.capture_device;
        loop_config.transport = args.transport;

        pipeline::RobotHearingLoop loop(
            loop_config,
            [&](const float* samples, int sample_count) {
                return asr.transcribe(samples, sample_count);
            });

        pipeline::RawPcmDecoder decoder(
            std::cin,
            {args.input_rate, args.input_channels});
        std::vector<float> block;
        std::string decode_error;

        while (decoder.read_block(block, decode_error)) {
            loop.push(block.data(), static_cast<int>(block.size()));
            if (!write_events(loop)) return 1;
        }
        if (!decode_error.empty()) {
            std::cerr << "ERROR: " << decode_error << "\n";
            return 1;
        }

        loop.flush();
        if (!write_events(loop)) return 1;
        return loop.had_asr_error() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
