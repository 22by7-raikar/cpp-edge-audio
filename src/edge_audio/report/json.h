/// @file
/// @brief Minimal JSON string escaping shared by the event and report writers.
/// @ingroup report

#pragma once

#include <string>
#include <string_view>

namespace edge_audio {

/// @addtogroup report
/// @{

/// @brief Escape @p value for use inside a JSON string literal (RFC 8259):
/// quote, backslash and all control characters below U+0020.
std::string json_escape(std::string_view value);

/// @brief `"` + json_escape(@p value) + `"`.
std::string json_quote(std::string_view value);

/// @}

}  // namespace edge_audio
