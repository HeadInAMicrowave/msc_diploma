// loop_test.cpp
// Usage: ./loop_test

#include "cleanup.hpp"
#include "detox_loop.hpp"
#include "interfaces.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dx;

static int failures = 0;

static void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::cout << (ok ? "[pass] " : "[FAIL] ") << what;
    if (!ok && !detail.empty()) std::cout << "\n       " << detail;
    std::cout << "\n";
    if (!ok) ++failures;
}

// Build a TaggedSpan for the first occurrence of `sub` in `text`.
static TaggedSpan span_of(const std::string& text, const std::string& sub, float conf) {
    size_t p = text.find(sub);
    if (p == std::string::npos) throw std::runtime_error("fixture bug: '" + sub + "' not in '" + text + "'");
    return {p, p + sub.size(), conf};
}

// Scripted gate: exact-text lookup with a default for anything unscripted.
struct ScriptedGate {
    std::map<std::string, float> scores;
    float fallback;
    FunctionGate gate;
    ScriptedGate(std::map<std::string, float> s, float fb)
        : scores(std::move(s)), fallback(fb),
          gate([this](const std::string& t) { auto it = scores.find(t); return it == scores.end() ? fallback : it->second; }) {}
};

// Scripted tagger: exact-text lookup; unscripted text yields no spans.
struct ScriptedTagger {
    std::map<std::string, std::vector<std::pair<std::string, float>>> spans;
    FunctionTagger tagger;
    explicit ScriptedTagger(std::map<std::string, std::vector<std::pair<std::string, float>>> s)
        : spans(std::move(s)),
          tagger([this](const std::string& t) {
              std::vector<TaggedSpan> out;
              auto it = spans.find(t);
              if (it != spans.end())
                  for (const auto& [sub, conf] : it->second) out.push_back(span_of(t, sub, conf));
              return out;
          }) {}
};

static std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? " | " : "") + v[i];
    return s;
}

// ---------------------------------------------------------------- cleanup ----

static void test_cleanup() {
    const std::string baron =
        "You should've listened to me and went top. You are such a trash player, when I call go baron you go baron.";
    auto sp = span_of(baron, "You are such a trash player,", 1.f);
    check(apply_removal(baron, sp.start, sp.end) ==
          "You should've listened to me and went top. When I call go baron you go baron.",
          "cleanup: clause removal inherits capital at sentence start");

    const std::string baron_lower =
        "you should've listened to me and went top. you are such a trash player, when i call go baron you go baron.";
    sp = span_of(baron_lower, "you are such a trash player,", 1.f);
    check(apply_removal(baron_lower, sp.start, sp.end) ==
          "you should've listened to me and went top. when i call go baron you go baron.",
          "cleanup: lowercase original stays lowercase");

    sp = span_of(baron, "You are such a trash player", 1.f);   // without its comma
    check(apply_removal(baron, sp.start, sp.end) ==
          "You should've listened to me and went top. When I call go baron you go baron.",
          "cleanup: orphaned comma is stripped and recasing still lands");

    check(apply_removal("fuck this lag", 0, 4) == "this lag",  "cleanup: word-tight, lowercase kept");
    check(apply_removal("Fuck this lag", 0, 4) == "This lag",  "cleanup: word-tight, capital inherited");
    check(apply_removal("go top you moron", 7, 16) == "go top", "cleanup: trailing removal trims");

    std::string s = "stop feeding you useless bot, group mid";
    sp = span_of(s, "stop feeding you useless bot,", 1.f);
    check(apply_removal(s, sp.start, sp.end) == "group mid", "cleanup: leading clause with comma");

    s = "nice shot idiot, now push mid you clown";
    sp = span_of(s, "idiot,", 1.f);
    std::string s2 = apply_removal(s, sp.start, sp.end);
    check(s2 == "nice shot now push mid you clown", "cleanup: mid-sentence word+comma");
    sp = span_of(s2, "you clown", 1.f);
    check(apply_removal(s2, sp.start, sp.end) == "nice shot now push mid", "cleanup: second removal on residual");

    check(apply_removal("you clown", 0, 9).empty(), "cleanup: whole message -> empty");
    check(apply_removal("abc", 2, 1) == "abc" && apply_removal("abc", 0, 9) == "abc", "cleanup: invalid span is a no-op");
}

// ------------------------------------------------------------------- loop ----

static void test_loop() {
    DetoxPolicy P;   // defaults: 0.5 / 5 tries / 200 ms / Confidence / Delete

    {
        ScriptedGate g({{"gg wp everyone", 0.05f}}, 0.9f);
        ScriptedTagger t({});
        auto r = detox("gg wp everyone", g.gate, t.tagger, P);
        check(r.action == Action::Pass && r.output == "gg wp everyone" && r.iterations == 0 && r.tagger_calls == 0,
              "loop: clean message passes untouched, tagger not called");
    }

    {
        ScriptedGate g({{"fuck this lag", 0.95f}, {"this lag", 0.05f}}, 0.9f);
        ScriptedTagger t({{"fuck this lag", {{"fuck", 0.97f}}}});
        auto r = detox("fuck this lag", g.gate, t.tagger, P);
        check(r.action == Action::Rewritten && r.output == "this lag" && r.iterations == 1 &&
              r.removed == std::vector<std::string>{"fuck"} && r.gate_calls == 2,
              "loop: one span removed, rewritten", r.output);
    }

    {
        const std::string m = "nice shot idiot, now push mid you clown";
        ScriptedGate g({{m, 0.9f}, {"nice shot idiot, now push mid", 0.7f}, {"nice shot now push mid", 0.1f}}, 0.9f);
        ScriptedTagger t({{m, {{"idiot,", 0.8f}, {"you clown", 0.9f}}},
                          {"nice shot idiot, now push mid", {{"idiot,", 0.8f}}}});
        auto r = detox(m, g.gate, t.tagger, P);
        check(r.action == Action::Rewritten && r.output == "nice shot now push mid" && r.iterations == 2 &&
              r.removed == std::vector<std::string>{"you clown", "idiot,"},
              "loop: greedy removes one span per iteration, highest confidence first",
              join(r.removed) + " -> " + r.output);
    }

    {
        ScriptedGate g({}, 0.95f);
        ScriptedTagger t({});
        auto r = detox("how are you this bad at this game", g.gate, t.tagger, P);
        check(r.action == Action::DeletedFused && r.output.empty() && r.iterations == 0,
              "loop: fused toxicity at entry -> deleted");
    }

    {
        ScriptedGate g({}, 0.9f);
        ScriptedTagger t({{"you clown, go top", {{"you clown,", 0.9f}}}});
        auto r = detox("you clown, go top", g.gate, t.tagger, P);
        check(r.action == Action::DeletedFused && r.output.empty() && r.iterations == 1 &&
              r.removed == std::vector<std::string>{"you clown,"},
              "loop: fused after a removal -> deleted, removal recorded");
    }

    {
        ScriptedGate g({}, 0.9f);
        FunctionTagger always_first([](const std::string& s) {
            size_t sp = s.find(' ');
            return std::vector<TaggedSpan>{{0, sp == std::string::npos ? s.size() : sp, 0.9f}};
        });
        DetoxPolicy p2 = P; p2.max_tries = 2;
        auto r = detox("a b c d e f", g.gate, always_first, p2);
        check(r.action == Action::DeletedCapTries && r.output.empty() && r.iterations == 2,
              "loop: max_tries reached while toxic -> deleted (on_cap=Delete)");
        p2.on_cap = OnCap::Keep;
        r = detox("a b c d e f", g.gate, always_first, p2);
        check(r.action == Action::KeptPartialTries && r.output == "c d e f" && r.iterations == 2,
              "loop: max_tries reached while toxic -> partial kept (on_cap=Keep)", r.output);
    }

    {
        ScriptedGate g({}, 0.9f);
        FunctionTagger always_first([](const std::string& s) {
            size_t sp = s.find(' ');
            return std::vector<TaggedSpan>{{0, sp == std::string::npos ? s.size() : sp, 0.9f}};
        });
        double fake = 0.0;
        Clock clock = [&]() { double v = fake; fake += 150.0; return v; };
        // t0=0; cap check now=150 (<200) -> iterate; next cap check now=300 -> timeout
        auto r = detox("a b c d e f", g.gate, always_first, P, clock);
        check(r.action == Action::DeletedCapTimeout && r.output.empty() && r.iterations == 1,
              "loop: timeout while toxic -> deleted after 1 removal", std::to_string(r.iterations));
        DetoxPolicy keep = P; keep.on_cap = OnCap::Keep; fake = 0.0;
        r = detox("a b c d e f", g.gate, always_first, keep, clock);
        check(r.action == Action::KeptPartialTimeout && r.output == "b c d e f",
              "loop: timeout while toxic -> partial kept (on_cap=Keep)");
    }

    {
        const std::string m = "alpha beta";
        ScriptedGate g({{m, 0.9f}, {"beta", 0.8f}, {"alpha", 0.1f}}, 0.9f);
        ScriptedTagger t({{m, {{"alpha", 0.9f}, {"beta", 0.6f}}}});
        DetoxPolicy conf = P;                       // confidence -> removes "alpha" -> "beta" (0.8, toxic)
        auto rc = detox(m, g.gate, t.tagger, conf);
        DetoxPolicy orc = P; orc.rank = Rank::Oracle;   // oracle -> removes "beta" -> "alpha" (0.1, clean)
        auto ro = detox(m, g.gate, t.tagger, orc);
        check(rc.removed.front() == "alpha" && ro.removed.front() == "beta" &&
              ro.action == Action::Rewritten && ro.output == "alpha" && ro.iterations == 1,
              "loop: oracle ranking chooses by residual score, confidence by tagger confidence",
              "conf removed " + rc.removed.front() + ", oracle removed " + ro.removed.front());
        check(ro.gate_calls == 3, "loop: oracle reuses candidate scores (1 entry + 2 candidates, no re-score)",
              std::to_string(ro.gate_calls));
    }

    {
        ScriptedGate g({}, 0.9f);
        ScriptedTagger t({{"you clown", {{"you clown", 0.95f}}}});
        auto r = detox("you clown", g.gate, t.tagger, P);
        check(r.action == Action::DeletedEroded && r.output.empty() && r.iterations == 1,
              "loop: removal that empties the message -> deleted_eroded");
    }

    {
        ScriptedGate g({{"x y", 0.5f}}, 0.0f);
        ScriptedTagger t({});
        auto r = detox("x y", g.gate, t.tagger, P);
        check(r.action == Action::DeletedFused, "loop: score == threshold is treated as toxic");
    }

    {
        const std::string m = "p q";
        ScriptedGate g({{m, 0.9f}, {"q", 0.1f}}, 0.9f);
        ScriptedTagger t({{m, {{"p", 0.7f}, {"q", 0.7f}}}});
        auto r = detox(m, g.gate, t.tagger, P);
        check(r.removed.front() == "p" && r.output == "q", "loop: confidence tie -> earliest span");
    }
}

int main() {
    test_cleanup();
    test_loop();
    if (failures) {
        std::cout << "\n" << failures << " failure(s).\n";
        return 1;
    }
    std::cout << "\nAll loop and cleanup fixtures pass.\n";
    return 0;
}
