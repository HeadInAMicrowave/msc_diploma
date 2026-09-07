#pragma once

#include <cctype>
#include <string>

namespace dx {

inline bool is_sentence_end(char c) { return c == '.' || c == '!' || c == '?'; }
inline bool is_separator(char c)    { return c == ',' || c == ';' || c == ':'; }
inline bool is_space(char c)        { return std::isspace(static_cast<unsigned char>(c)) != 0; }

// True if byte `pos` begins a sentence: index 0, or preceded (ignoring
// whitespace) by sentence-ending punctuation.
inline bool at_sentence_start(const std::string& s, size_t pos) {
    size_t i = pos;
    while (i > 0 && is_space(s[i - 1])) --i;
    return i == 0 || is_sentence_end(s[i - 1]);
}

inline std::string collapse_whitespace(const std::string& s) {
    std::string out;
    bool prev_space = false;
    for (char c : s) {
        if (is_space(c)) {
            if (!prev_space) out.push_back(' ');
            prev_space = true;
        } else {
            out.push_back(c);
            prev_space = false;
        }
    }
    size_t a = 0;
    while (a < out.size() && out[a] == ' ') ++a;
    size_t b = out.size();
    while (b > a && out[b - 1] == ' ') --b;
    return out.substr(a, b - a);
}

// Drop separators that no longer attach to anything, then collapse whitespace.
inline std::string strip_orphan_separators(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (is_separator(c)) {
            size_t j = out.size();
            while (j > 0 && out[j - 1] == ' ') --j;
            bool orphan = (j == 0) || is_sentence_end(out[j - 1]) || is_separator(out[j - 1]);
            if (orphan) continue;
        }
        out.push_back(c);
    }
    return collapse_whitespace(out);
}

// Remove [start, end) from `text` and repair the seam. Spans are expected to be
// word-aligned (the tagger merges whole basic tokens), so no boundary snapping
// is attempted here.
inline std::string apply_removal(const std::string& text, size_t start, size_t end) {
    if (start >= end || end > text.size()) return text;   // defensive: nothing to do
    const bool sentence_start = at_sentence_start(text, start);
    const bool removed_upper  = std::isupper(static_cast<unsigned char>(text[start])) != 0;

    std::string out = text.substr(0, start) + text.substr(end);

    if (sentence_start && removed_upper) {
        // The next word now begins this sentence: skip whitespace and any
        // separator that is about to be stripped, then capitalise.
        size_t i = start;
        while (i < out.size() && (is_space(out[i]) || is_separator(out[i]))) ++i;
        if (i < out.size() && std::islower(static_cast<unsigned char>(out[i])))
            out[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[i])));
    }
    return strip_orphan_separators(out);
}

}
