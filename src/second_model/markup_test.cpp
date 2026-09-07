// markup_test.cpp
// No network, no model. Usage: ./markup_test <vocab.txt>
// (any BERT vocab containing [CLS] [SEP] [UNK]; only basic_tokenize is used)

#include "wordpiece.hpp"
#include "span_markup.hpp"

#include <iostream>
#include <string>
#include <vector>

// `expect` is one char per basic token: T = TOXIC, O = OUT.
struct Good { const char* original; const char* marked; const char* expect; };

static const Good kGood[] = {
    {"fuck this lag", "<del>fuck</del> this lag", "TOO"},
    {"gg wp everyone", "gg wp everyone", "OOO"},
    {"how are you this bad at this game", "<del>how are you this bad at this game</del>", "TTTTTTTT"},
    {"ez game get good scrub", "ez game get good <del>scrub</del>", "OOOOT"},
    {"stop feeding you useless bot, group mid", "<del>stop feeding you useless bot,</del> group mid", "TTTTTTOO"},
    {"f u c k this game", "<del>f u c k</del> this game", "TTTTOO"},
    {"nice shot idiot, now push mid you clown",
     "nice shot <del>idiot,</del> now push mid <del>you clown</del>", "OOTTOOOTT"},
    {"that bitch stole my kill", "that <del>bitch</del> stole my kill", "OTOOO"},
    {"you fucking noob", "<del>you fucking noob</del>", "TTT"},
    {"you fucking noob, group mid", "<del>you fucking noob,</del> group mid", "TTTTOO"},
    {"this team is garbage", "<del>this team is garbage</del>", "TTTT"},
    {"you're all useless, just group and def", "<del>you're all useless,</del> just group and def", "TTTTTTOOOO"},
    {"You should've listened to me and went top. You are such a trash player, when I call go baron you go baron.",
     "You should've listened to me and went top. <del>You are such a trash player,</del> when I call go baron you go baron.",
     "OOOOOOOOOOOTTTTTTTOOOOOOOOO"},
    {"fuck this lag", "```\n<del>fuck</del> this lag\n```", "TOO"},
};

struct Bad { const char* original; const char* marked; const char* what; };

static const Bad kBadParse[] = {
    {"a b c",         "<del>a <del>b</del> c</del>",       "nested tags"},
    {"fuck this lag", "<del>fuck</del> this lagg",          "altered text"},
    {"fuck this lag", "<del>fuck</del> this",               "truncated output"},
    {"fuck this lag", "<del></del>fuck this lag",           "empty span"},
    {"fuck this lag", "<del>fuck this lag",                 "unclosed tag"},
    {"fuck this lag", "fuck</del> this lag",                "stray close tag"},
    {"fuck this lag", "Fuck this lag",                      "casing changed"},
    {"fuck this lag", "\"<del>fuck</del> this lag\"",       "quoted answer"},
    {"fuck this lag", "Here it is: <del>fuck</del> this lag", "prose prefix"},
};

static const Bad kBadAlign[] = {
    {"assassin build is strong", "<del>ass</del>assin build is strong", "span cuts inside a word"},
    {"classic map",              "<del>cl</del>assic map",              "span cuts inside a word"},
};

struct Header { const char* line; bool matches; const char* cat; };

static const Header kHeaders[] = {
    {"# [DIRECTED_HOSTILITY_SEPARABLE]  (expected: separable)  ~60", true,  "DIRECTED_HOSTILITY_SEPARABLE"},
    {"# [CLEAN]",                                                    true,  "CLEAN"},
    {"#   [MIXED]  (profanity + hostility in one message)  ~21",     true,  "MIXED"},
    {"# [SARCASM_REVIEW]  -- FLAGGED, NOT FORCE-LABELLED  ~10",      true,  "SARCASM_REVIEW"},
    {"# PLAY_CRITICISM lines (\"our macro is trash\", \"sloppy play from all of us\"),", false, ""},
    {"# DIRECTED_HOSTILITY_SEPARABLE  (unbracketed old style)",     false, ""},
    {"# FORMAT",                                                    false, ""},
    {"# The first eight lines are fused",                          false, ""},
    {"# ============================================================================", false, ""},
    {"# \"you fucking noob\" -- RULED (stacked-attack rule)",     false, ""},
    {"# [lowercase]",                                              false, ""},
    {"# [AB]",                                                     false, ""},
    {"# [UNCLOSED  (missing bracket)",                             false, ""},
    {"not a comment",                                              false, ""},
};

static std::string render(const std::vector<std::string>& tags) {
    std::string s;
    for (const auto& t : tags) s.push_back(t == "TOXIC" ? 'T' : 'O');
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <vocab.txt>\n"; return 2; }
    wp::WordPieceTokenizer tok(argv[1]);
    int failures = 0;

    for (const auto& c : kGood) {
        std::vector<sm::Span> spans;
        std::string why;
        auto words  = tok.basic_tokenize(c.original);
        auto marked = sm::clean_response(c.marked);
        bool ok = sm::parse_markup(c.original, marked, spans, why)
               && sm::spans_word_aligned(spans, words);
        std::string got = ok ? render(sm::tag_words(spans, words)) : ("<rejected: " + why + ">");
        bool pass = ok && got == c.expect;
        std::cout << (pass ? "[pass] " : "[FAIL] ") << "\"" << c.original << "\"\n";
        if (!pass) { ++failures; std::cout << "   expected " << c.expect << "\n   got      " << got << "\n"; }
    }

    for (const auto& b : kBadParse) {
        std::vector<sm::Span> spans;
        std::string why;
        bool rejected = !sm::parse_markup(b.original, sm::clean_response(b.marked), spans, why);
        std::cout << (rejected ? "[pass] " : "[FAIL] ") << "reject " << b.what
                  << (rejected ? " (" + why + ")" : "") << "\n";
        if (!rejected) ++failures;
    }

    for (const auto& b : kBadAlign) {
        std::vector<sm::Span> spans;
        std::string why;
        auto words = tok.basic_tokenize(b.original);
        bool parsed   = sm::parse_markup(b.original, b.marked, spans, why);
        bool rejected = parsed && !sm::spans_word_aligned(spans, words);
        std::cout << (rejected ? "[pass] " : "[FAIL] ") << "reject " << b.what << "\n";
        if (!rejected) ++failures;
    }

    for (const auto& h : kHeaders) {
        std::string cat;
        bool m = sm::parse_category_header(h.line, cat);
        bool pass = (m == h.matches) && (!m || cat == h.cat);
        std::cout << (pass ? "[pass] " : "[FAIL] ") << "header \"" << h.line << "\""
                  << (m ? " -> " + cat : "") << "\n";
        if (!pass) ++failures;
    }

    if (failures) {
        std::cout << "\n" << failures << " failure(s).\n";
        return 1;
    }
    std::cout << "\nAll fixtures pass. Span recovery verified against the prompt's examples.\n";
    return 0;
}
