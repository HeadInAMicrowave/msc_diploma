#pragma once

#include "textcnn_config.hpp"

#include <torch/torch.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr int kNumLabels = 7;
static const std::array<std::string, kNumLabels> kLabelColumns = {
    "target", "severe_toxicity", "obscene", "threat",
    "insult", "identity_attack", "sexual_explicit"
};

static constexpr int64_t kPad = 0;   // [PAD] -> must match pad_index
static constexpr int64_t kUnk = 1;   // [UNK]

struct Vocab {
    std::unordered_map<std::string, int64_t> tok2id;
};

static std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    auto flush = [&]() { if (!cur.empty()) { tokens.push_back(cur); cur.clear(); } };
    for (unsigned char uc : text) {
        if (uc < 128 && std::isalnum(uc)) {
            cur.push_back(static_cast<char>(std::tolower(uc)));
        }
        else if (uc < 128 && std::ispunct(uc)) {
            flush();
            tokens.emplace_back(1, static_cast<char>(uc));
        }
        else {
            flush();
        }
    }
    flush();
    return tokens;
}

inline Vocab loadVocab(const std::string& path) {
    Vocab v;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        size_t tab = line.rfind('\t');
        if (tab == std::string::npos) continue;
        v.tok2id[line.substr(0, tab)] = std::stoll(line.substr(tab + 1));
    }
    return v;
}

struct TextCNNImpl : torch::nn::Module {
    torch::nn::Embedding embedding{ nullptr };
    torch::nn::ModuleList convs;
    torch::nn::Dropout   dropout{ nullptr };
    torch::nn::Linear    fc_hidden{ nullptr };
    torch::nn::Linear    fc_out{ nullptr };

    TextCNNImpl(int64_t vocab_size,
        int64_t embed_dim,
        int64_t num_filters,
        std::vector<int64_t> kernel_sizes,
        int64_t hidden_dim,
        int64_t num_labels,
        double  dropout_p,
        int64_t pad_index) {
        embedding = register_module(
            "embedding",
            torch::nn::Embedding(
                torch::nn::EmbeddingOptions(vocab_size, embed_dim)
                .padding_idx(pad_index)));

        convs = register_module("convs", torch::nn::ModuleList());
        for (auto k : kernel_sizes) {
            convs->push_back(
                torch::nn::Conv1d(
                    torch::nn::Conv1dOptions(embed_dim, num_filters, k)));
        }

        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));

        const int64_t concat_dim =
            num_filters * static_cast<int64_t>(kernel_sizes.size());

        fc_hidden = register_module("fc_hidden",
            torch::nn::Linear(concat_dim, hidden_dim));
        fc_out = register_module("fc_out",
            torch::nn::Linear(hidden_dim, num_labels));
    }

    // x: [batch, seq_len] int64  ->  [batch, num_labels] raw logits
    torch::Tensor forward(torch::Tensor x) {
        auto emb = embedding->forward(x);   // [B, T, D]
        emb = emb.transpose(1, 2);          // [B, D, T]

        std::vector<torch::Tensor> pooled;
        pooled.reserve(convs->size());

        for (const auto& module : *convs) {
            auto* conv = module->as<torch::nn::Conv1d>();
            auto feat = torch::relu(conv->forward(emb));         // [B, M, T-k+1]
            feat = std::get<0>(torch::max(feat, /*dim=*/2));     // [B, M]
            pooled.push_back(feat);
        }

        auto cat = torch::cat(pooled, /*dim=*/1);
        cat = dropout->forward(cat);

        auto h = torch::relu(fc_hidden->forward(cat));
        h = dropout->forward(h);

        return fc_out->forward(h);   // logits
    }
};
TORCH_MODULE(TextCNN);


inline std::vector<std::string> textcnn_tokenize(const std::string& text) {
    return tokenize(text);
}

inline TextCNN make_textcnn(const m1::TextCnnConfig& c) {
    return TextCNN(c.vocab_size, c.embed_dim, c.num_filters, c.kernel_sizes,
                   c.hidden_dim, c.num_labels, c.dropout, c.pad_index);
}
