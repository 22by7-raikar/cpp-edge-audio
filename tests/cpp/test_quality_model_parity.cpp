// Native vs scikit-learn parity on every validation and test example.
//
// The corpus (tests/data/quality_model_v1_parity.bin) is built by
// tools/python/export/export_quality_model.py --corpus-only from the committed
// artifact: each row is a 27-float feature vector and the probability that
// scikit-learn produced for it at training time.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "edge_audio/quality/quality_model.h"

namespace {

constexpr std::array<char, 8> kMagic{{'Q', 'M', 'P', 'A', 'R', '2', '\0', '\0'}};
constexpr std::uint32_t kVersion = 2;
constexpr double kProbabilityTolerance = 1e-12;

struct Example {
    std::string id;
    std::array<float, edge_audio::kQualityFeatureCount> features{};
    double probability = 0.0;
    bool decision = false;
};

struct Corpus {
    std::string schema;
    std::string model_sha256;
    double threshold = 0.0;
    std::uint32_t validation_count = 0;
    std::uint32_t test_count = 0;
    std::vector<Example> examples;
};

class Reader {
public:
    explicit Reader(const std::string& path) : in_(path, std::ios::binary) {}
    bool ok() const { return static_cast<bool>(in_); }
    bool at_end() { return in_.peek() == std::char_traits<char>::eof(); }

    template <typename T>
    T value() {  // little-endian host assumed (x86-64, AArch64)
        T out{};
        in_.read(reinterpret_cast<char*>(&out), sizeof(out));
        if (!in_) throw std::runtime_error("truncated parity corpus");
        return out;
    }
    std::string text(std::size_t size) {
        std::string out(size, '\0');
        in_.read(out.data(), static_cast<std::streamsize>(size));
        if (!in_) throw std::runtime_error("truncated parity corpus");
        return out.substr(0, out.find('\0'));
    }

private:
    std::ifstream in_;
};

Corpus load_corpus() {
    Reader in(EDGE_AUDIO_PARITY_CORPUS);
    if (!in.ok()) throw std::runtime_error("cannot open " EDGE_AUDIO_PARITY_CORPUS);

    std::array<char, 8> magic{};
    for (char& c : magic) c = in.value<char>();
    if (magic != kMagic || in.value<std::uint32_t>() != kVersion) {
        throw std::runtime_error("unexpected parity corpus format");
    }
    Corpus corpus;
    const auto feature_count = in.value<std::uint32_t>();
    corpus.validation_count = in.value<std::uint32_t>();
    corpus.test_count = in.value<std::uint32_t>();
    corpus.schema = in.text(32);
    corpus.model_sha256 = in.text(64);
    corpus.threshold = in.value<double>();
    if (feature_count != edge_audio::kQualityFeatureCount) {
        throw std::runtime_error("parity corpus feature count mismatch");
    }
    const std::size_t rows = std::size_t{corpus.validation_count} + corpus.test_count;
    corpus.examples.resize(rows);
    for (Example& example : corpus.examples) {
        example.id = in.text(24);
        for (float& f : example.features) f = in.value<float>();
        example.probability = in.value<double>();
        example.decision = in.value<std::uint8_t>() != 0;
    }
    if (!in.at_end()) throw std::runtime_error("parity corpus has trailing bytes");
    return corpus;
}

class QualityModelParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { corpus_ = new Corpus(load_corpus()); }
    static void TearDownTestSuite() {
        delete corpus_;
        corpus_ = nullptr;
    }
    static Corpus* corpus_;
};

Corpus* QualityModelParity::corpus_ = nullptr;

}  // namespace

TEST_F(QualityModelParity, CorpusMatchesCompiledModel) {
    EXPECT_EQ(corpus_->schema, edge_audio::quality_schema_version());
    EXPECT_EQ(corpus_->model_sha256, edge_audio::quality_model_sha256());
    EXPECT_EQ(corpus_->threshold, edge_audio::quality_default_threshold());
    EXPECT_EQ(corpus_->validation_count, 1700u);
    EXPECT_EQ(corpus_->test_count, 1700u);
}

TEST_F(QualityModelParity, EveryDecisionAndProbabilityMatchesScikitLearn) {
    std::size_t disagreements = 0;
    double max_difference = 0.0;
    std::string worst_id;
    for (const Example& example : corpus_->examples) {
        const auto prediction = edge_audio::predict_quality(example.features);
        const double difference = std::fabs(prediction.probability - example.probability);
        if (difference > max_difference) {
            max_difference = difference;
            worst_id = example.id;
        }
        disagreements += prediction.should_transcribe != example.decision ? 1 : 0;
    }
    std::ostringstream formatted;
    formatted << std::setprecision(17) << max_difference;
    RecordProperty("max_abs_probability_difference", formatted.str());
    EXPECT_EQ(disagreements, 0u);
    EXPECT_LE(max_difference, kProbabilityTolerance) << "worst example: " << worst_id;
}
