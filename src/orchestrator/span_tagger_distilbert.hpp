// span_tagger_distilbert.hpp
// Build deps: LibTorch, nlohmann/json, wordpiece.hpp.

#pragma once

#include "interfaces.hpp"
#include "wordpiece.hpp"

#include <nlohmann/json.hpp>
#include <torch/script.h>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dx {

class DistilBertSpanTagger : public SpanTagger {
public:
    // `dir` holds model.pt, vocab.txt, labels.json. A word is tagged TOXIC when
    // P(TOXIC) >= word_threshold.
    DistilBertSpanTagger(const std::string& dir, torch::Device device, float word_threshold = 0.5f)
        : device_(device), threshold_(word_threshold) {
        nlohmann::json labels = nlohmann::json::parse(std::ifstream(dir + "/labels.json"));
        max_length_ = labels.value("max_length", 64);
        const bool lower = labels.value("do_lower_case", true);

        toxic_id_ = -1;
        for (auto& [k, v] : labels.at("id2label").items())
            if (v.get<std::string>() == "TOXIC") toxic_id_ = std::stoi(k);
        if (toxic_id_ < 0)
            throw std::runtime_error("labels.json has no TOXIC label; was schema.json relabeled?");

        tok_ = std::make_unique<wp::WordPieceTokenizer>(dir + "/vocab.txt", max_length_, lower);
        model_ = torch::jit::load(dir + "/model.pt");
        model_.to(device_);
        model_.eval();
    }

    std::vector<TaggedSpan> tag(const std::string& text) override {
        wp::Tokenized t = tok_->encode(text);
        const int64_t T = static_cast<int64_t>(t.input_ids.size());

        auto ids  = torch::from_blob(t.input_ids.data(), {1, T}, torch::kInt64).clone().to(device_);
        auto mask = torch::ones({1, T}, torch::kInt64).to(device_);

        torch::NoGradGuard ng;
        std::vector<torch::jit::IValue> inputs{ids, mask};
        auto logits = model_.forward(inputs).toTensor();                        // [1, T, num_labels]
        auto probs  = torch::softmax(logits, -1).select(2, toxic_id_).squeeze(0).to(torch::kCPU); // [T]

        // Per-word P(TOXIC) from each word's first subword.
        std::vector<float> word_p(t.words.size(), 0.0f);
        for (size_t i = 0; i < t.input_ids.size(); ++i) {
            if (!t.first_piece[i]) continue;
            const int w = t.word_of_piece[i];
            if (w < 0) continue;
            word_p[static_cast<size_t>(w)] = probs[static_cast<int64_t>(i)].item<float>();
        }

        // Merge runs of TOXIC words into spans; confidence is the mean P(TOXIC).
        std::vector<TaggedSpan> spans;
        size_t w = 0;
        while (w < word_p.size()) {
            if (word_p[w] < threshold_) { ++w; continue; }
            const size_t first = w;
            float sum = 0.0f;
            int n = 0;
            while (w < word_p.size() && word_p[w] >= threshold_) { sum += word_p[w]; ++n; ++w; }
            spans.push_back({t.words[first].start, t.words[w - 1].end, sum / static_cast<float>(n)});
        }
        return spans;
    }

    const char* name() const override { return "distilbert-span-tagger"; }

private:
    torch::Device device_;
    float threshold_;
    int max_length_ = 64;
    int toxic_id_ = -1;
    std::unique_ptr<wp::WordPieceTokenizer> tok_;
    torch::jit::script::Module model_;
};

}
