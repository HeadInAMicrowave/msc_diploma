// textcnn_gate.hpp
// Model 1 as a ToxicityGate. Wraps the native LibTorch TextCNN from the
// detection track (trained in C++, saved with torch::save).
// Scoring: forward -> sigmoid -> logit[target_index]. The full label vector
// is kept in last_labels() for the evaluation harness (obscene, insult, ...).
//
// Build deps: LibTorch, nlohmann/json, textcnn.hpp.

#pragma once

#include "interfaces.hpp"
#include "textcnn_config.hpp"
#include "textcnn.hpp"

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace dx {

class TextCnnGate : public ToxicityGate {
public:
    TextCnnGate(const std::string& dir, torch::Device device) : device_(device) {
        cfg_ = load_config(dir + "/m1_config.json");
        vocab_ = loadVocab(dir + "/" + vocab_name_).tok2id;          // the trainer's own loader
        if (vocab_.empty()) throw std::runtime_error("empty or unreadable vocab: " + dir + "/" + vocab_name_);
        if (cfg_.vocab_size <= 0) {
            cfg_.vocab_size = static_cast<int64_t>(vocab_.size());  // dense ids 0..V-1
        } else if (static_cast<int64_t>(vocab_.size()) != cfg_.vocab_size) {
            std::cerr << "[warn] " << vocab_name_ << " has " << vocab_.size() << " entries but m1_config.json says "
                      << cfg_.vocab_size << "; torch::load will fail if the embedding shape differs\n";
        }

        model_ = make_textcnn(cfg_);
        torch::load(model_, dir + "/" + weights_name_);
        model_->to(device_);
        model_->eval();
        last_labels_.assign(static_cast<size_t>(cfg_.num_labels), 0.0f);
    }

    float score(const std::string& text) override {
        std::vector<int64_t> ids = encode(text);
        const int64_t T = static_cast<int64_t>(ids.size());
        auto x = torch::from_blob(ids.data(), {1, T}, torch::kInt64).clone().to(device_);

        torch::NoGradGuard ng;
        auto probs = torch::sigmoid(model_->forward(x)).to(torch::kCPU);   // [1, num_labels]
        for (int64_t i = 0; i < cfg_.num_labels; ++i)
            last_labels_[static_cast<size_t>(i)] = probs[0][i].item<float>();
        return last_labels_[static_cast<size_t>(cfg_.target_index)];
    }

    // All sigmoid outputs from the most recent score() call, in training label
    // order (target, severe_toxicity, obscene, threat, insult, identity_attack,
    // sexual_explicit).
    const std::vector<float>& last_labels() const { return last_labels_; }

    // Fraction of the message's tokens that fall outside the training vocabulary
    // and are fed to the model as [UNK]. A per-message measure of the domain gap
    // between the Jigsaw vocabulary and game chat.
    float unk_rate(const std::string& text) const {
        std::vector<std::string> toks = textcnn_tokenize(text);
        if (toks.empty()) return 0.0f;
        size_t unk = 0;
        for (const auto& t : toks) if (!vocab_.count(t)) ++unk;
        return static_cast<float>(unk) / static_cast<float>(toks.size());
    }

    const char* name() const override { return "textcnn-gate"; }

private:
    // Tokenize with the training tokenizer, map to ids (UNK for OOV), truncate
    // to max_seq_len, then pad: to fixed_len if set (training-faithful), else
    // to at least min_seq_len so every kernel fits (dynamic, lower latency).
    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<std::string> toks = textcnn_tokenize(text);
        std::vector<int64_t> ids;
        ids.reserve(toks.size());
        for (const auto& t : toks) {
            auto it = vocab_.find(t);
            ids.push_back(it == vocab_.end() ? cfg_.unk_index : it->second);
            if (static_cast<int64_t>(ids.size()) >= cfg_.max_seq_len) break;
        }
        const int64_t target_len = fixed_len_ > 0 ? fixed_len_ : cfg_.min_seq_len;
        if (fixed_len_ > 0 && static_cast<int64_t>(ids.size()) > fixed_len_) ids.resize(static_cast<size_t>(fixed_len_));
        while (static_cast<int64_t>(ids.size()) < target_len) ids.push_back(cfg_.pad_index);
        return ids;
    }

    m1::TextCnnConfig load_config(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open " + path);
        nlohmann::json j = nlohmann::json::parse(in);
        m1::TextCnnConfig c;
        c.vocab_size   = j.value("vocab_size", int64_t{0});      // 0 -> derived from the vocab file
        c.embed_dim    = j.value("embed_dim",    c.embed_dim);
        c.num_filters  = j.value("num_filters",  c.num_filters);
        c.hidden_dim   = j.value("hidden_dim",   c.hidden_dim);
        c.num_labels   = j.value("num_labels",   c.num_labels);
        c.dropout      = j.value("dropout",      c.dropout);
        c.pad_index    = j.value("pad_index",    c.pad_index);
        c.unk_index    = j.value("unk_index",    c.unk_index);
        c.max_seq_len  = j.value("max_seq_len",  c.max_seq_len);
        c.min_seq_len  = j.value("min_seq_len",  c.min_seq_len);
        c.target_index = j.value("target_index", c.target_index);
        if (j.contains("kernel_sizes")) c.kernel_sizes = j["kernel_sizes"].get<std::vector<int64_t>>();
        weights_name_ = j.value("weights", std::string("textcnn_toxicity.pt"));
        vocab_name_   = j.value("vocab",   std::string("vocab.tsv"));
        fixed_len_    = j.value("fixed_len", int64_t{0});

        int64_t largest = 0;
        for (auto k : c.kernel_sizes) if (k > largest) largest = k;
        if (c.min_seq_len < largest) {
            std::cerr << "[warn] min_seq_len " << c.min_seq_len << " < largest kernel " << largest
                      << "; raising to " << largest << "\n";
            c.min_seq_len = largest;
        }
        return c;
    }

    torch::Device device_;
    m1::TextCnnConfig cfg_;
    std::string weights_name_ = "textcnn_toxicity.pt";
    std::string vocab_name_   = "vocab.tsv";
    int64_t     fixed_len_    = 0;
    std::unordered_map<std::string, int64_t> vocab_;
    TextCNN model_{nullptr};
    std::vector<float> last_labels_;
};

}
