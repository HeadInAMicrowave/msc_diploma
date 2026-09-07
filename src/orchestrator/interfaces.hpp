#pragma once

#include <cctype>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

namespace dx {

class ToxicityGate {
public:
    virtual ~ToxicityGate() = default;
    // Probability-like toxicity score for the whole message, in [0, 1].
    virtual float score(const std::string& text) = 0;
    virtual const char* name() const = 0;
};

// A candidate deletion: byte offsets [start, end) into the message it was
// tagged on, and the tagger's confidence that the span is toxic.
struct TaggedSpan {
    size_t start;
    size_t end;
    float  confidence;
};

class SpanTagger {
public:
    virtual ~SpanTagger() = default;
    // All toxic spans found in `text`, in document order. Empty if none.
    virtual std::vector<TaggedSpan> tag(const std::string& text) = 0;
    virtual const char* name() const = 0;
};


class FunctionGate : public ToxicityGate {
public:
    explicit FunctionGate(std::function<float(const std::string&)> f) : f_(std::move(f)) {}
    float score(const std::string& text) override { return f_(text); }
    const char* name() const override { return "function-gate"; }
private:
    std::function<float(const std::string&)> f_;
};

class FunctionTagger : public SpanTagger {
public:
    explicit FunctionTagger(std::function<std::vector<TaggedSpan>(const std::string&)> f) : f_(std::move(f)) {}
    std::vector<TaggedSpan> tag(const std::string& text) override { return f_(text); }
    const char* name() const override { return "function-tagger"; }
private:
    std::function<std::vector<TaggedSpan>(const std::string&)> f_;
};


// Scores 1.0 if any lowercased alphanumeric token is in the word list, else 0.0.
// Exists only so the loop + real tagger can be exercised end to end before the
// TextCNN adapter is wired.
class LexiconGate : public ToxicityGate {
public:
    explicit LexiconGate(std::unordered_set<std::string> words) : words_(std::move(words)) {}
    float score(const std::string& text) override {
        std::string tok;
        auto flush = [&]() { bool hit = !tok.empty() && words_.count(tok); tok.clear(); return hit; };
        for (char c : text) {
            unsigned char u = static_cast<unsigned char>(c);
            if (std::isalnum(u)) tok.push_back(static_cast<char>(std::tolower(u)));
            else if (flush()) return 1.0f;
        }
        return flush() ? 1.0f : 0.0f;
    }
    const char* name() const override { return "lexicon-gate (smoke test only)"; }
private:
    std::unordered_set<std::string> words_;
};

}
