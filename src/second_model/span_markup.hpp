#pragma once

#include "wordpiece.hpp"

#include <string>
#include <vector>

namespace sm {

// A deletion span as byte offsets [start, end) into the ORIGINAL message.
struct Span {
    size_t start;
    size_t end;
};

inline std::string trim(std::string s) {
    auto ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (!s.empty() && ws(static_cast<unsigned char>(s.back()))) s.pop_back();
    size_t i = 0;
    while (i < s.size() && ws(static_cast<unsigned char>(s[i]))) ++i;
    return s.substr(i);
}

// Tolerate an accidental code fence around the answer, then trim. Anything
// else unexpected (quotes, prose) is deliberately left in place so the
// invariant check fails and the row is dropped: drop rate is a signal of
// prompt quality, and repairing answers would hide it.
inline std::string clean_response(std::string s) {
    auto p = s.find("```");
    if (p != std::string::npos) {
        s = s.substr(p + 3);
        auto q = s.find("```");
        if (q != std::string::npos) s = s.substr(0, q);
    }
    return trim(s);
}

inline bool has_reserved_marker(const std::string& original) {
    return original.find("<del>") != std::string::npos ||
           original.find("</del>") != std::string::npos;
}

// Single pass over the teacher's output against the original. Verifies that
// the untagged text equals the original byte-for-byte, that tags are balanced
// and non-nested, that no span is empty, and records each span's offsets into
// the original. Returns false with a reason on any failure.
inline bool parse_markup(const std::string& original, const std::string& marked,
                         std::vector<Span>& spans, std::string& why) {
    static const std::string open = "<del>", close = "</del>";
    spans.clear();
    size_t i = 0, o = 0;                 // i indexes `marked`, o indexes `original`
    bool in_del = false;
    size_t span_start = 0;
    while (i < marked.size()) {
        if (marked.compare(i, open.size(), open) == 0) {
            if (in_del) { why = "nested <del>"; return false; }
            in_del = true;
            span_start = o;
            i += open.size();
            continue;
        }
        if (marked.compare(i, close.size(), close) == 0) {
            if (!in_del) { why = "stray </del>"; return false; }
            if (o == span_start) { why = "empty <del></del>"; return false; }
            spans.push_back({span_start, o});
            in_del = false;
            i += close.size();
            continue;
        }
        if (o >= original.size() || marked[i] != original[o]) {
            why = "untagged text differs from original";
            return false;
        }
        ++i;
        ++o;
    }
    if (in_del) { why = "unclosed <del>"; return false; }
    if (o != original.size()) { why = "output shorter than original"; return false; }
    return true;
}

// Policy section 6 makes every span word-aligned (profanity is word-tight,
// clauses are whole). A span boundary that falls strictly inside a basic token
// indicates a sloppy answer: reject rather than snap, to keep the corpus pure.
// Boundaries on whitespace or exactly at a token edge are fine.
inline bool spans_word_aligned(const std::vector<Span>& spans,
                               const std::vector<wp::BasicToken>& words) {
    for (const auto& s : spans)
        for (const auto& w : words) {
            if (w.start < s.start && s.start < w.end) return false;
            if (w.start < s.end   && s.end   < w.end) return false;
        }
    return true;
}

// A word is TOXIC if its byte range overlaps any deletion span, else OUT.
// A fully wrapped message yields all TOXIC; an untagged one yields all OUT.
inline std::vector<std::string> tag_words(const std::vector<Span>& spans,
                                          const std::vector<wp::BasicToken>& words) {
    std::vector<std::string> tags(words.size(), "OUT");
    for (size_t k = 0; k < words.size(); ++k)
        for (const auto& s : spans)
            if (words[k].start < s.end && s.start < words[k].end) {
                tags[k] = "TOXIC";
                break;
            }
    return tags;
}

// Seed-file category headers use an explicit marker: a '#' line whose first
// token is the category name in square brackets, e.g.
//     # [DIRECTED_HOSTILITY_SEPARABLE]  (expected: ...)  ~60
// The name is ALL_CAPS with underscores. Any other '#' line is free prose and
// is skipped. The brackets exist because a first-token-is-ALL-CAPS heuristic
// was tripped in testing by a prose comment that began with a category name
// ("# PLAY_CRITICISM lines (...)"), which silently misfiled a whole block.
inline bool parse_category_header(const std::string& line, std::string& cat) {
    if (line.empty() || line[0] != '#') return false;
    size_t i = 1;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size() || line[i] != '[') return false;
    size_t start = ++i;
    while (i < line.size() && ((line[i] >= 'A' && line[i] <= 'Z') || line[i] == '_')) ++i;
    if (i - start < 3) return false;
    if (i >= line.size() || line[i] != ']') return false;
    cat = line.substr(start, i - start);
    return true;
}

}
