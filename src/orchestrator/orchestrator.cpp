// orchestrator.cpp
// Runtime entry point for the two-model system.
// Usage:
//   orchestrator --m1 <dir> --m2 <dir> [options] < messages.txt
//   orchestrator --gate lexicon [--lexicon words.txt] --m2 <dir> < messages.txt
//
// Options:
//   --gate textcnn|lexicon   gate implementation (default textcnn). lexicon is a
//                            wordlist stand-in for smoke-testing the loop with
//                            the real tagger before model 1 is wired; never
//                            report results produced with it.
//   --lexicon <file>         one word per line for --gate lexicon
//   --threshold <f>          gate score at/above which a message is toxic (0.5)
//   --word-threshold <f>     tagger P(TOXIC) at/above which a word is toxic (0.5)
//   --max-tries <n>          maximum removals (5)
//   --timeout-ms <f>         wall-clock budget per message (200)
//   --rank confidence|oracle span ranking (confidence)
//   --on-cap delete|keep     if still toxic at the cap (delete)
//   --device cpu|cuda        (cuda if available)
//   --json                   emit one JSON record per line instead of text
//   --tag-only               skip the loop and the gate: emit model 2's raw spans
//                            per line as JSON, for evaluating the tagger alone
//                            (--m1 not required)
//
// Build deps: LibTorch, nlohmann/json, wordpiece.hpp, and textcnn.hpp (yours)
// for the textcnn gate. loop_test needs none of these.

#include "detox_loop.hpp"
#include "interfaces.hpp"
#include "span_tagger_distilbert.hpp"
#include "textcnn_gate.hpp"

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

using json = nlohmann::json;

namespace {

struct Options {
    std::string m1_dir, m2_dir, lexicon_path;
    std::string gate = "textcnn";
    std::string device;                 // empty -> auto
    float word_threshold = 0.5f;
    bool  json_out = false;
    bool  tag_only = false;
    dx::DetoxPolicy policy;
};

[[noreturn]] void usage_and_exit(const char* argv0, const std::string& err = "") {
    if (!err.empty()) std::cerr << "[error] " << err << "\n";
    std::cerr << "usage: " << argv0
              << " --m2 <dir> (--m1 <dir> | --gate lexicon [--lexicon <file>])\n"
                 "       [--threshold f] [--word-threshold f] [--max-tries n] [--timeout-ms f]\n"
                 "       [--rank confidence|oracle] [--on-cap delete|keep] [--device cpu|cuda] [--json] [--tag-only]\n";
    std::exit(2);
}

Options parse_args(int argc, char** argv) {
    Options o;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) usage_and_exit(argv[0], std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--m1")             o.m1_dir = need(i);
        else if (a == "--m2")             o.m2_dir = need(i);
        else if (a == "--gate")           o.gate = need(i);
        else if (a == "--lexicon")        o.lexicon_path = need(i);
        else if (a == "--threshold")      o.policy.threshold = std::stof(need(i));
        else if (a == "--word-threshold") o.word_threshold = std::stof(need(i));
        else if (a == "--max-tries")      o.policy.max_tries = std::stoi(need(i));
        else if (a == "--timeout-ms")     o.policy.timeout_ms = std::stod(need(i));
        else if (a == "--device")         o.device = need(i);
        else if (a == "--json")           o.json_out = true;
        else if (a == "--tag-only")       o.tag_only = true;
        else if (a == "--rank") {
            std::string v = need(i);
            if (v == "confidence") o.policy.rank = dx::Rank::Confidence;
            else if (v == "oracle") o.policy.rank = dx::Rank::Oracle;
            else usage_and_exit(argv[0], "--rank must be confidence|oracle");
        } else if (a == "--on-cap") {
            std::string v = need(i);
            if (v == "delete") o.policy.on_cap = dx::OnCap::Delete;
            else if (v == "keep") o.policy.on_cap = dx::OnCap::Keep;
            else usage_and_exit(argv[0], "--on-cap must be delete|keep");
        } else if (a == "-h" || a == "--help") {
            usage_and_exit(argv[0]);
        } else {
            usage_and_exit(argv[0], "unknown argument " + a);
        }
    }
    if (o.m2_dir.empty()) usage_and_exit(argv[0], "--m2 is required");
    if (!o.tag_only && o.gate == "textcnn" && o.m1_dir.empty())
        usage_and_exit(argv[0], "--m1 is required with --gate textcnn (unless --tag-only)");
    if (o.gate != "textcnn" && o.gate != "lexicon") usage_and_exit(argv[0], "--gate must be textcnn|lexicon");
    return o;
}

std::unordered_set<std::string> load_lexicon(const std::string& path) {
    // A small default so the smoke test runs with no file at all.
    std::unordered_set<std::string> words{
        "fuck", "fucking", "fucked", "fucker", "fck", "shit", "shitty", "sh1t", "bullshit",
        "bitch", "b1tch", "asshole", "dick", "damn", "wtf", "stfu", "omfg", "ffs"};
    if (path.empty()) return words;
    std::ifstream in(path);
    if (!in) { std::cerr << "[error] cannot open lexicon " << path << "\n"; std::exit(1); }
    words.clear();
    std::string w;
    while (std::getline(in, w)) {
        if (!w.empty() && w.back() == '\r') w.pop_back();
        if (!w.empty()) words.insert(w);
    }
    return words;
}

json to_json(const std::string& input, const dx::DetoxResult& r) {
    json j;
    j["input"]         = input;
    j["output"]        = r.output;
    j["action"]        = dx::action_name(r.action);
    j["deleted"]       = dx::is_deleted(r.action);
    j["iterations"]    = r.iterations;
    j["initial_score"] = r.initial_score;
    j["final_score"]   = r.final_score;
    j["removed"]       = r.removed;
    j["elapsed_ms"]    = r.elapsed_ms;
    j["gate_calls"]    = r.gate_calls;
    j["tagger_calls"]  = r.tagger_calls;
    return j;
}

} // namespace

int main(int argc, char** argv) {
    Options o = parse_args(argc, argv);

    torch::Device device(torch::kCPU);
    if (o.device == "cuda" || (o.device.empty() && torch::cuda::is_available())) device = torch::Device(torch::kCUDA);
    else if (!o.device.empty() && o.device != "cpu") usage_and_exit(argv[0], "--device must be cpu|cuda");

    std::unique_ptr<dx::ToxicityGate> gate;
    std::unique_ptr<dx::SpanTagger>   tagger;
    try {
        if (o.tag_only) {
            // no gate needed
        } else if (o.gate == "lexicon") {
            gate = std::make_unique<dx::LexiconGate>(load_lexicon(o.lexicon_path));
            std::cerr << "[warn] using the lexicon stand-in gate; results are for smoke testing only\n";
        } else {
            gate = std::make_unique<dx::TextCnnGate>(o.m1_dir, device);
        }
        tagger = std::make_unique<dx::DistilBertSpanTagger>(o.m2_dir, device, o.word_threshold);
    } catch (const std::exception& e) {
        std::cerr << "[error] loading models: " << e.what() << "\n";
        return 1;
    }
    // Warm both models once before serving. The first CUDA call pays cuDNN/cuBLAS
    // initialisation and kernel loading (hundreds of ms), which would otherwise
    // land inside the first message's timed region and could trip the timeout.
    // Results are discarded; steady-state latency starts at the first real message.
    try {
        if (gate) gate->score("warm up");
        tagger->tag("warm up");
    } catch (const std::exception& e) {
        std::cerr << "[error] warm-up forward pass failed: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "[ready] gate=" << (gate ? gate->name() : "none (--tag-only)") << " tagger=" << tagger->name()
              << " device=" << (device.is_cuda() ? "cuda" : "cpu")
              << " threshold=" << o.policy.threshold << " max_tries=" << o.policy.max_tries
              << " timeout_ms=" << o.policy.timeout_ms
              << " rank=" << (o.policy.rank == dx::Rank::Oracle ? "oracle" : "confidence")
              << " on_cap=" << (o.policy.on_cap == dx::OnCap::Keep ? "keep" : "delete") << "\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (o.tag_only) {
            json rec;
            rec["input"] = line;
            rec["spans"] = json::array();
            for (const auto& s : tagger->tag(line))
                rec["spans"].push_back({{"start", s.start}, {"end", s.end}, {"confidence", s.confidence},
                                        {"text", line.substr(s.start, s.end - s.start)}});
            std::cout << rec.dump() << "\n";
            continue;
        }
        dx::DetoxResult r = dx::detox(line, *gate, *tagger, o.policy);
        if (o.json_out) {
            json rec = to_json(line, r);
            if (auto* g = dynamic_cast<dx::TextCnnGate*>(gate.get()))
                rec["gate_unk_rate"] = g->unk_rate(line);   // domain-gap evidence per message
            std::cout << rec.dump() << "\n";
        } else {
            std::cout << r.output << "\n";
        }
    }
    return 0;
}
