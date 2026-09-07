// Usage: basic_dump <vocab.txt> < lines.txt
// (any BERT vocab containing [CLS] [SEP] [UNK]; only basic_tokenize is used)

#include "wordpiece.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <vocab.txt> < lines\n"; return 2; }
    wp::WordPieceTokenizer tok(argv[1]);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto words = tok.basic_tokenize(line);
        for (size_t i = 0; i < words.size(); ++i) std::cout << (i ? " " : "") << words[i].text;
        std::cout << "\n";
    }
    return 0;
}
