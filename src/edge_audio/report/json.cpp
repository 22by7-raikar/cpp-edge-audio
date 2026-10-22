#include "edge_audio/report/json.h"

namespace edge_audio {

std::string json_escape(std::string_view value) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const char ch : value) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\b':
                escaped += "\\b";
                break;
            case '\f':
                escaped += "\\f";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (c < 0x20) {
                    escaped += "\\u00";
                    escaped.push_back(kHex[(c >> 4) & 0x0f]);
                    escaped.push_back(kHex[c & 0x0f]);
                } else {
                    escaped.push_back(ch);
                }
        }
    }
    return escaped;
}

std::string json_quote(std::string_view value) {
    return '"' + json_escape(value) + '"';
}

}  // namespace edge_audio
