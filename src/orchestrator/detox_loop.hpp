#pragma once

#include "cleanup.hpp"
#include "interfaces.hpp"

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace dx {

enum class Rank  { Confidence, Oracle };
enum class OnCap { Delete, Keep };

struct DetoxPolicy {
    float  threshold  = 0.5f;   // gate score at or above this is "toxic"
    int    max_tries  = 5;      // maximum removals
    double timeout_ms = 200.0;  // wall-clock budget for the whole loop
    Rank   rank       = Rank::Confidence;
    OnCap  on_cap     = OnCap::Delete;
};

enum class Action {
    Pass,                 // gate cleared the original; untouched
    Rewritten,            // one or more removals; gate now clears it
    DeletedFused,         // gate toxic, tagger found nothing removable
    DeletedCapTries,      // still toxic after max_tries; on_cap = Delete
    DeletedCapTimeout,    // still toxic at timeout;      on_cap = Delete
    DeletedEroded,        // removals emptied the message
    KeptPartialTries,     // still toxic after max_tries; on_cap = Keep
    KeptPartialTimeout,   // still toxic at timeout;      on_cap = Keep
};

inline const char* action_name(Action a) {
    switch (a) {
        case Action::Pass:               return "pass";
        case Action::Rewritten:          return "rewritten";
        case Action::DeletedFused:       return "deleted_fused";
        case Action::DeletedCapTries:    return "deleted_cap_tries";
        case Action::DeletedCapTimeout:  return "deleted_cap_timeout";
        case Action::DeletedEroded:      return "deleted_eroded";
        case Action::KeptPartialTries:   return "kept_partial_tries";
        case Action::KeptPartialTimeout: return "kept_partial_timeout";
    }
    return "unknown";
}

inline bool is_deleted(Action a) {
    return a == Action::DeletedFused || a == Action::DeletedCapTries ||
           a == Action::DeletedCapTimeout || a == Action::DeletedEroded;
}

struct DetoxResult {
    std::string output;                 // final text; empty when deleted
    Action action = Action::Pass;
    int    iterations   = 0;            // removals performed
    float  initial_score = 0.0f;        // gate score of the input
    float  final_score   = 0.0f;        // last gate score observed
    std::vector<std::string> removed;   // removed span texts, in removal order
    double elapsed_ms   = 0.0;
    int    gate_calls   = 0;
    int    tagger_calls = 0;
};

// Millisecond clock, injectable so the timeout branch can be tested.
using Clock = std::function<double()>;

inline double steady_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline DetoxResult detox(const std::string& text, ToxicityGate& gate, SpanTagger& tagger,
                         const DetoxPolicy& p, const Clock& now = steady_now_ms) {
    DetoxResult r;
    const double t0 = now();
    auto finish = [&](Action a, std::string out) {
        r.action = a;
        r.output = std::move(out);
        r.elapsed_ms = now() - t0;
        return r;
    };

    r.initial_score = gate.score(text);
    ++r.gate_calls;
    r.final_score = r.initial_score;
    if (r.initial_score < p.threshold) return finish(Action::Pass, text);

    std::string cur = text;
    for (;;) {
        if (r.iterations >= p.max_tries)
            return finish(p.on_cap == OnCap::Delete ? Action::DeletedCapTries : Action::KeptPartialTries,
                          p.on_cap == OnCap::Delete ? std::string() : cur);
        if (now() - t0 >= p.timeout_ms)
            return finish(p.on_cap == OnCap::Delete ? Action::DeletedCapTimeout : Action::KeptPartialTimeout,
                          p.on_cap == OnCap::Delete ? std::string() : cur);

        std::vector<TaggedSpan> spans = tagger.tag(cur);
        ++r.tagger_calls;
        if (spans.empty()) return finish(Action::DeletedFused, std::string());

        std::string next;
        float next_score = 0.0f;
        bool  have_next_score = false;
        size_t pick = 0;

        if (p.rank == Rank::Confidence) {
            for (size_t i = 1; i < spans.size(); ++i)
                if (spans[i].confidence > spans[pick].confidence) pick = i;  // tie -> earliest
            next = apply_removal(cur, spans[pick].start, spans[pick].end);
        } else {
            // Oracle: score every candidate residual, keep the lowest.
            std::string best;
            float best_score = 0.0f;
            for (size_t i = 0; i < spans.size(); ++i) {
                std::string cand = apply_removal(cur, spans[i].start, spans[i].end);
                float sc = 0.0f;                     // an empty residual is trivially non-toxic
                if (!cand.empty()) { sc = gate.score(cand); ++r.gate_calls; }
                bool better = (i == 0) || sc < best_score ||
                              (sc == best_score && spans[i].confidence > spans[pick].confidence);
                if (better) { pick = i; best = cand; best_score = sc; }
            }
            next = best;
            next_score = best_score;
            have_next_score = true;
        }

        r.removed.push_back(cur.substr(spans[pick].start, spans[pick].end - spans[pick].start));
        cur = next;
        ++r.iterations;

        if (cur.empty()) return finish(Action::DeletedEroded, std::string());

        if (!have_next_score) { next_score = gate.score(cur); ++r.gate_calls; }
        r.final_score = next_score;
        if (next_score < p.threshold) return finish(Action::Rewritten, cur);
    }
}

}
