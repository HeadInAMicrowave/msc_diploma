#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace wp {

// A basic token: a whitespace/punctuation-delimited chunk of the original
// string, with byte offsets [start, end) into that original string.
struct BasicToken {
    std::string text;   // normalized (lowercased-ASCII) surface form
    size_t start;       // byte offset in the ORIGINAL input
    size_t end;         // byte offset in the ORIGINAL input (exclusive)
};

// Full tokenization result aligned to model input.
struct Tokenized {
    std::vector<BasicToken> words;       // basic tokens, in order
    std::vector<int64_t>    input_ids;   // includes [CLS] ... [SEP]
    std::vector<int>        word_of_piece;  // per input_id: index into words, or -1 for special
    std::vector<bool>       first_piece;    // per input_id: true if first subword of its basic token
};

class WordPieceTokenizer {
public:
    // Loads a standard BERT vocab.txt (one token per line, line number == id).
    explicit WordPieceTokenizer(const std::string& vocab_path,
                                int max_length = 64,
                                bool do_lower_case = true) :
        max_length_(max_length), do_lower_case_(do_lower_case) {
        std::ifstream in(vocab_path);
        if (!in) throw std::runtime_error("cannot open vocab file: " + vocab_path);
        std::string line;
        int id = 0;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            vocab_[line] = id++;
        }
        auto need = [&](const std::string& t) -> int {
            auto it = vocab_.find(t);
            if (it == vocab_.end())
                throw std::runtime_error("vocab missing required token: " + t);
            return it->second;
        };
        cls_id_ = need("[CLS]");
        sep_id_ = need("[SEP]");
        unk_id_ = need("[UNK]");
    }

    int cls_id() const { return cls_id_; }
    int sep_id() const { return sep_id_; }
    int unk_id() const { return unk_id_; }

    std::vector<BasicToken> basic_tokenize(const std::string& original) const {
        std::vector<BasicToken> out;
        size_t i = 0, n = original.size();
        while (i < n) {
            unsigned char c = static_cast<unsigned char>(original[i]);
            if (is_ws(c) || is_control(c)) { ++i; continue; }
            if (c < 0x80 && is_punct(c)) {                 // punctuation = its own token
                out.push_back({ normalize_byte(c), i, i + 1 });
                ++i;
                continue;
            }
            // accumulate a word until whitespace or punctuation
            size_t start = i;
            std::string norm;
            while (i < n) {
                unsigned char d = static_cast<unsigned char>(original[i]);
                if (is_ws(d) || is_control(d)) break;
                if (d < 0x80 && is_punct(d)) break;
                norm.push_back(static_cast<char>(normalize_byte_raw(d)));
                ++i;
            }
            out.push_back({ norm, start, i });
        }
        return out;
    }

    // Full tokenization producing model-ready input_ids with alignment metadata.
    Tokenized encode(const std::string& original) const {
        Tokenized t;
        t.words = basic_tokenize(original);

        t.input_ids.push_back(cls_id_);
        t.word_of_piece.push_back(-1);
        t.first_piece.push_back(false);

        const size_t budget = (max_length_ > 2) ? static_cast<size_t>(max_length_ - 1) : 1; // leave room for [SEP]
        for (int w = 0; w < static_cast<int>(t.words.size()); ++w) {
            std::vector<int> pieces = wordpiece(t.words[w].text);
            bool first = true;
            for (int id : pieces) {
                if (t.input_ids.size() >= budget) break;  // truncate long inputs
                t.input_ids.push_back(id);
                t.word_of_piece.push_back(w);
                t.first_piece.push_back(first);
                first = false;
            }
            if (t.input_ids.size() >= budget) break;
        }

        t.input_ids.push_back(sep_id_);
        t.word_of_piece.push_back(-1);
        t.first_piece.push_back(false);
        return t;
    }

private:
    std::unordered_map<std::string, int> vocab_;
    int max_length_;
    bool do_lower_case_;
    int cls_id_ = -1, sep_id_ = -1, unk_id_ = -1;
    static constexpr int kMaxCharsPerWord = 100;

    static bool is_ws(unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }
    static bool is_control(unsigned char c) {
        return c != '\t' && c != '\n' && c != '\r' && c < 0x20;
    }
    static bool is_punct(unsigned char c) {
        return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
               (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
    }
    unsigned char normalize_byte_raw(unsigned char c) const {
        if (do_lower_case_ && c >= 'A' && c <= 'Z') return c + 32;
        return c;  // bytes >= 0x80 pass through unchanged -> offsets preserved
    }
    std::string normalize_byte(unsigned char c) const {
        return std::string(1, static_cast<char>(normalize_byte_raw(c)));
    }

    // Greedy longest-match-first WordPiece over a single normalized basic token.
    std::vector<int> wordpiece(const std::string& token) const {
        std::vector<int> ids;
        if (static_cast<int>(token.size()) > kMaxCharsPerWord) { ids.push_back(unk_id_); return ids; }
        size_t start = 0, len = token.size();
        std::vector<int> tmp;
        while (start < len) {
            size_t end = len;
            int cur = -1;
            while (start < end) {
                std::string sub = token.substr(start, end - start);
                if (start > 0) sub = "##" + sub;
                auto it = vocab_.find(sub);
                if (it != vocab_.end()) { cur = it->second; break; }
                --end;
            }
            if (cur == -1) { tmp.clear(); break; }  // any failure => whole token is [UNK]
            tmp.push_back(cur);
            start = end;
        }
        if (tmp.empty()) ids.push_back(unk_id_);
        else ids = std::move(tmp);
        return ids;
    }
};

}
