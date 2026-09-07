#pragma once

#include <cstdint>
#include <vector>

namespace m1 {

struct TextCnnConfig {
    int64_t vocab_size  = 0;       // REQUIRED: size of the saved vocab (incl. PAD/UNK)
    int64_t embed_dim   = 300;
    std::vector<int64_t> kernel_sizes{2, 3, 4, 5, 7};
    int64_t num_filters = 128;
    int64_t hidden_dim  = 256;
    int64_t num_labels  = 7;
    double  dropout     = 0.5;     // irrelevant in eval mode, needed to construct
    int64_t pad_index   = 0;       // kPad in the training file
    int64_t unk_index   = 1;       // kUnk in the training file
    int64_t max_seq_len = 256;     // truncate longer inputs
    int64_t min_seq_len = 7;       // >= largest kernel, or Conv1d fails on short chat
    int     target_index = 0;      // which logit is the gate score: "target" is 0
};

}
