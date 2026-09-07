#include <torch/torch.h>

#include <string>
#include <vector>
#include <array>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <iostream>

#include "textcnn.hpp"   // kNumLabels, kLabelColumns, kPad, kUnk, Vocab, tokenize, loadVocab, TextCNN

// Path to the folder holding train.csv / test.csv.
#ifndef DATASET_DIR
#define DATASET_DIR "dataset"
#endif

struct Record {
    std::string text;
    std::array<float, kNumLabels> labels{};
};


struct PreparedData {
    torch::Tensor tokens;   // [N, max_seq_len] int64
    torch::Tensor labels;   // [N, kNumLabels] float32
    Vocab   vocab;
    int64_t vocab_size = 0;
    int64_t max_seq_len = 0;
};

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static float parseFloat(const std::string& s) {
    if (s.empty()) return 0.0f;
    try { return std::stof(s); }
    catch (...) { return 0.0f; }
}


// Reads one logical CSV record, which may span several physical lines when a
// field is quoted.
static bool readCsvRecord(std::istream& in, std::vector<std::string>& fields) {
    fields.clear();
    std::string field;
    bool inQuotes = false;
    bool anyChar = false;
    int c;
    while ((c = in.get()) != EOF) {
        anyChar = true;
        char ch = static_cast<char>(c);
        if (inQuotes) {
            if (ch == '"') {
                if (in.peek() == '"') { in.get(); field.push_back('"'); }
                else { inQuotes = false; }
            }
            else {
                field.push_back(ch);
            }
        }
        else {
            if (ch == '"') {
                inQuotes = true;
            }
            else if (ch == ',') {
                fields.push_back(field); field.clear();
            }
            else if (ch == '\n') {
                fields.push_back(field); return true;
            }
            else if (ch == '\r') {
                if (in.peek() == '\n') in.get();
                fields.push_back(field); return true;
            }
            else {
                field.push_back(ch);
            }
        }
    }
    if (anyChar) { fields.push_back(field); return true; }
    return false;
}

struct ColumnIndex {
    int comment = -1;
    std::array<int, kNumLabels> labels{};
};

static ColumnIndex mapColumns(const std::vector<std::string>& header) {
    ColumnIndex col;
    col.labels.fill(-1);
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        std::string h = trim(header[i]);
        if (h == "comment_text") col.comment = i;
        for (int k = 0; k < kNumLabels; ++k)
            if (h == kLabelColumns[k]) col.labels[k] = i;
    }
    if (col.comment < 0)   throw std::runtime_error("comment_text column not found");
    if (col.labels[0] < 0) throw std::runtime_error("target column not found");
    return col;
}

static Record makeRecord(const std::vector<std::string>& fields, const ColumnIndex& col) {
    Record r;
    if (col.comment < static_cast<int>(fields.size())) r.text = fields[col.comment];
    for (int k = 0; k < kNumLabels; ++k) {
        int idx = col.labels[k];
        r.labels[k] = (idx >= 0 && idx < static_cast<int>(fields.size()))
            ? parseFloat(fields[idx]) : 0.0f;
    }
    return r;
}

// Uniform random sample of `n` data rows in a single streaming pass.
static std::vector<Record> reservoirSample(const std::string& csvPath,
    size_t n, uint64_t seed) {
    std::ifstream in(csvPath, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + csvPath);

    std::vector<std::string> fields;
    if (!readCsvRecord(in, fields)) throw std::runtime_error("empty csv: " + csvPath);
    ColumnIndex col = mapColumns(fields);

    std::vector<Record> reservoir;
    reservoir.reserve(n);
    std::mt19937_64 rng(seed);
    size_t seen = 0;

    while (readCsvRecord(in, fields)) {
        ++seen;
        if (reservoir.size() < n) {
            reservoir.push_back(makeRecord(fields, col));
        }
        else {
            std::uniform_int_distribution<size_t> dist(0, seen - 1);
            size_t j = dist(rng);
            if (j < n) reservoir[j] = makeRecord(fields, col);
        }
        if (seen % 200000 == 0)
            std::cout << "  scanned " << seen << " rows...\n";
    }
    std::cout << "  scanned " << seen << " rows total\n";
    return reservoir;
}

static Vocab buildVocabFromTokens(const std::vector<std::vector<std::string>>& toks,
    int minCount, int64_t maxVocab) {
    std::unordered_map<std::string, int64_t> freq;
    for (const auto& doc : toks)
        for (const auto& t : doc) ++freq[t];

    std::vector<std::pair<std::string, int64_t>> kept;
    kept.reserve(freq.size());
    for (const auto& kv : freq)
        if (kv.second >= minCount) kept.push_back(kv);

    std::sort(kept.begin(), kept.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
        });
    if (maxVocab > 2 && static_cast<int64_t>(kept.size()) > maxVocab - 2)
        kept.resize(static_cast<size_t>(maxVocab - 2));

    Vocab v;
    v.tok2id.reserve(kept.size() + 2);
    v.tok2id["[PAD]"] = kPad;
    v.tok2id["[UNK]"] = kUnk;
    int64_t id = 2;
    for (const auto& kv : kept) v.tok2id[kv.first] = id++;
    return v;
}

static std::vector<int64_t> encodeTokens(const std::vector<std::string>& toks,
    const Vocab& v, int64_t L) {
    std::vector<int64_t> ids;
    ids.reserve(static_cast<size_t>(L));
    for (const auto& t : toks) {
        if (static_cast<int64_t>(ids.size()) >= L) break;   // truncate
        auto it = v.tok2id.find(t);
        ids.push_back(it != v.tok2id.end() ? it->second : kUnk);
    }
    while (static_cast<int64_t>(ids.size()) < L) ids.push_back(kPad);  // pad
    return ids;
}

// Reuse an existing vocabulary later (ie. for the test set).
std::vector<int64_t> encode(const std::string& text, const Vocab& v, int64_t L) {
    return encodeTokens(tokenize(text), v, L);
}

void saveVocab(const Vocab& v, const std::string& path) {
    std::vector<std::pair<std::string, int64_t>> items(v.tok2id.begin(), v.tok2id.end());
    std::sort(items.begin(), items.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    std::ofstream out(path, std::ios::binary);
    for (const auto& kv : items) out << kv.first << '\t' << kv.second << '\n';
}


// maxSeqLen <= 0  -> derive from autoQuantile of the sampled length distribution
PreparedData buildTrainingSet(const std::string& csvPath,
    size_t   sampleN,
    int      minCount,
    int64_t  maxVocab,
    int64_t  maxSeqLen,
    double   autoQuantile,
    uint64_t seed) {
    std::vector<Record> records = reservoirSample(csvPath, sampleN, seed);
    const int64_t N = static_cast<int64_t>(records.size());
    if (N == 0) throw std::runtime_error("no records sampled");

    std::vector<std::vector<std::string>> toks(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i)
        toks[static_cast<size_t>(i)] = tokenize(records[static_cast<size_t>(i)].text);

    int64_t L = maxSeqLen;
    if (L <= 0) {
        std::vector<int64_t> lengths(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; ++i)
            lengths[static_cast<size_t>(i)] =
            static_cast<int64_t>(toks[static_cast<size_t>(i)].size());
        std::sort(lengths.begin(), lengths.end());
        size_t idx = static_cast<size_t>(std::ceil(autoQuantile * (N - 1)));
        L = std::max<int64_t>(1, lengths[idx]);
    }
    if (L < 7) L = 7;   // must be >= the widest convolution kernel

    Vocab vocab = buildVocabFromTokens(toks, minCount, maxVocab);
    const int64_t V = static_cast<int64_t>(vocab.tok2id.size());

    auto tokens = torch::empty({ N, L }, torch::kInt64);
    auto labels = torch::empty({ N, kNumLabels }, torch::kFloat32);
    auto tok_a = tokens.accessor<int64_t, 2>();
    auto lab_a = labels.accessor<float, 2>();

    for (int64_t i = 0; i < N; ++i) {
        std::vector<int64_t> ids = encodeTokens(toks[static_cast<size_t>(i)], vocab, L);
        for (int64_t j = 0; j < L; ++j) tok_a[i][j] = ids[static_cast<size_t>(j)];
        for (int k = 0; k < kNumLabels; ++k)
            lab_a[i][k] = records[static_cast<size_t>(i)].labels[k];
    }

    return PreparedData{ tokens, labels, std::move(vocab), V, L };
}

// Optional: copy pretrained vectors ([vocab_size, embed_dim], vocab order).
void load_pretrained_embeddings(TextCNN& model, const torch::Tensor& vectors) {
    torch::NoGradGuard no_grad;
    model->embedding->weight.copy_(vectors);
}

// Kim-style max-norm constraint (s = 3). Call AFTER optimizer.step().
void apply_max_norm(torch::Tensor weight, double max_val) {
    torch::NoGradGuard no_grad;
    auto norms = weight.pow(2).sum(/*dim=*/1, /*keepdim=*/true).sqrt();
    auto scale = (max_val / (norms + 1e-7)).clamp_max(1.0);
    weight.mul_(scale);
}

class JigsawDataset : public torch::data::datasets::Dataset<JigsawDataset> {
    torch::Tensor tokens_;
    torch::Tensor labels_;
public:
    JigsawDataset(torch::Tensor tokens, torch::Tensor labels)
        : tokens_(std::move(tokens)), labels_(std::move(labels)) {}

    torch::data::Example<> get(size_t index) override {
        return { tokens_[static_cast<int64_t>(index)],
                labels_[static_cast<int64_t>(index)] };
    }

    torch::optional<size_t> size() const override {
        return static_cast<size_t>(tokens_.size(0));
    }
};

torch::Tensor predict(TextCNN model, torch::Tensor tokens, torch::Device device) {
    torch::NoGradGuard no_grad;
    model->eval();
    auto logits = model->forward(tokens.to(device));
    return torch::sigmoid(logits);
}

int main() {
    try {
        const size_t   SAMPLE_N = 20000;   // rows to draw at random
        const int      MIN_COUNT = 3;       // drop tokens rarer than this
        const int64_t  MAX_VOCAB = 50000;   // cap incl. [PAD],[UNK]; <=0 = uncapped
        const int64_t  MAX_SEQLEN = 0;       // <=0 -> derive from 95th percentile
        const double   QUANTILE = 0.95;
        const uint64_t SEED = 42;

        const int64_t embed_dim = 300;
        const int64_t num_filters = 128;
        const std::vector<int64_t> kernel_sizes = { 2, 3, 4, 5, 7 };
        const int64_t hidden_dim = 256;
        const double  dropout_p = 0.5;

        const int64_t batch_size = 128;
        const int64_t num_epochs = 5;
        const double  learning_rate = 1e-3;
        const bool    use_max_norm = true;

        const std::string train_csv = std::string(DATASET_DIR) + "/train.csv";
        std::cout << "Reading " << train_csv << "\n";

        auto t0 = std::chrono::steady_clock::now();
        PreparedData ds = buildTrainingSet(train_csv, SAMPLE_N, MIN_COUNT,
            MAX_VOCAB, MAX_SEQLEN, QUANTILE, SEED);
        auto t1 = std::chrono::steady_clock::now();

        std::cout << "Prepared " << ds.tokens.size(0) << " examples in "
            << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()
            << " s\n"
            << "  vocab_size  = " << ds.vocab_size << "\n"
            << "  max_seq_len = " << ds.max_seq_len << "\n";

        saveVocab(ds.vocab, "vocab.tsv");   // reused at inference time

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

        TextCNN model(ds.vocab_size, embed_dim, num_filters, kernel_sizes,
            hidden_dim, kNumLabels, dropout_p, kPad);
        model->to(device);


        auto dataset = JigsawDataset(ds.tokens, ds.labels)
            .map(torch::data::transforms::Stack<>());
        auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

        torch::optim::Adam optimizer(
            model->parameters(),
            torch::optim::AdamOptions(learning_rate).weight_decay(1e-5));

        for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
            model->train();
            double running_loss = 0.0;
            int64_t batches = 0;
            auto e0 = std::chrono::steady_clock::now();

            for (auto& batch : *loader) {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);

                optimizer.zero_grad();
                auto logits = model->forward(data);
                auto loss = torch::binary_cross_entropy_with_logits(logits, target);
                loss.backward();
                optimizer.step();

                if (use_max_norm) apply_max_norm(model->fc_out->weight, 3.0);

                running_loss += loss.item<double>();
                ++batches;
            }

            auto e1 = std::chrono::steady_clock::now();
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                << "  loss " << (running_loss / std::max<int64_t>(batches, 1))
                << "  (" << std::chrono::duration_cast<std::chrono::seconds>(e1 - e0).count()
                << " s)\n";
        }

        torch::save(model, "textcnn_toxicity.pt");
        std::cout << "Saved textcnn_toxicity.pt and vocab.tsv\n";

        auto ids = encode("you are an idiot", ds.vocab, ds.max_seq_len);
        auto sample = torch::from_blob(ids.data(),
            { 1, ds.max_seq_len },
            torch::kInt64).clone();
        std::cout << "Sample probabilities:\n" << predict(model, sample, device) << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}