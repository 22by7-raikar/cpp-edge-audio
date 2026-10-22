/// @file
/// @brief Combining rule and learned decisions into one admit/reject verdict.
/// @ingroup quality

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "edge_audio/asr/transcription.h"
#include "edge_audio/quality/quality_model.h"

namespace edge_audio {

/// @addtogroup quality
/// @{

/// @brief How a file or utterance is admitted to ASR.
enum class QualityPolicy {
    RULE,     ///< Admit if any chunk passes the rule gate (and is not music/silence).
    LEARNED,  ///< Admit if the learned probability reaches the threshold.
    HYBRID,   ///< Admit only if both RULE and LEARNED admit.
};

/// @brief Lower-case policy name: `rule`, `learned` or `hybrid`.
const char* quality_policy_str(QualityPolicy policy) noexcept;

/// @brief Parse a policy name.
/// @throws std::invalid_argument for any other string.
QualityPolicy parse_quality_policy(std::string_view value);

/// @brief Parse a probability threshold.
/// @throws std::invalid_argument unless @p value is a finite number in [0, 1].
double parse_quality_threshold(const std::string& value);

/// @brief The admission verdict and the decisions it was derived from.
///
/// `rejection_reason` is empty when admitted, otherwise one of
/// `rule_rejected`, `learned_below_threshold`, `hybrid_rule_rejected` or
/// `hybrid_learned_rejected`.
struct QualityAdmission {
    bool rule_should_transcribe = false;     ///< Rule decision.
    bool learned_should_transcribe = false;  ///< Learned decision (false under RULE).
    bool final_should_transcribe = false;    ///< Verdict under the chosen policy.
    std::string rejection_reason;            ///< Why it was rejected.
};

/// @brief Apply @p policy.
/// @param policy             Policy to apply.
/// @param rule_should_transcribe Rule verdict (any window admitted by the gate).
/// @param learned_prediction Required for LEARNED and HYBRID; ignored for RULE.
/// @throws std::invalid_argument if a learned policy has no prediction.
QualityAdmission decide_quality_admission(QualityPolicy policy, bool rule_should_transcribe,
                                          const QualityPrediction* learned_prediction);

/// @brief Whether ASR ran and what it returned.
struct AsrRun {
    bool ran = false;  ///< True if the transcriber was invoked.
    AsrResult result;  ///< Its result (default-constructed if not run).
};

/// @brief Invoke @p transcribe exactly once on the whole buffer if admitted.
///
/// This is the single point where admission turns into ASR compute, which keeps
/// the "zero or one Whisper call per file/utterance" guarantee in one place.
///
/// @param admitted     Final admission verdict.
/// @param asr_enabled  False in analysis-only runs.
/// @param samples      Mono 16 kHz PCM of the whole file or utterance.
/// @param sample_count Number of samples.
/// @param transcribe   ASR backend.
/// @throws std::invalid_argument if admitted and enabled but the buffer is empty,
///         larger than `INT_MAX` samples, or @p transcribe is empty.
AsrRun transcribe_if_admitted(bool admitted, bool asr_enabled, const float* samples,
                              std::size_t sample_count, const Transcriber& transcribe);

/// @}

}  // namespace edge_audio
