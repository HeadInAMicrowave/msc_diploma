// parity_test.cpp
// Usage: ./parity_test <artifacts_dir>

#include "wordpiece.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <artifacts_dir>\n"; return 2; }
    std::string dir = argv[1];

    json labels = json::parse(std::ifstream(dir + "/labels.json"));
    int max_length = labels.value("max_length", 64);
    bool lower = labels.value("do_lower_case", true);

    wp::WordPieceTokenizer tok(dir + "/vocab.txt", max_length, lower);
    json fixture = json::parse(std::ifstream(dir + "/parity_expected.json"));

    int failures = 0;
    for (const auto& probe : fixture) {
        std::string text = probe.at("text").get<std::string>();
        std::vector<int64_t> expected;
        for (const auto& id : probe.at("input_ids")) expected.push_back(id.get<int64_t>());

        auto got = tok.encode(text).input_ids;

        bool ok = (got.size() == expected.size());
        for (size_t i = 0; ok && i < got.size(); ++i) ok = (got[i] == expected[i]);

        std::cout << (ok ? "[pass] " : "[FAIL] ") << "\"" << text << "\"\n";
        if (!ok) {
            ++failures;
            std::cout << "   expected:";
            for (auto x : expected) std::cout << " " << x;
            std::cout << "\n   got     :";
            for (auto x : got) std::cout << " " << x;
            std::cout << "\n";
        }
    }
    if (failures) {
        std::cout << "\n" << failures << " mismatch(es).\n";
        return 1;
    }
    std::cout << "\nAll probes match. Tokenizer parity verified.\n";
    return 0;
}
