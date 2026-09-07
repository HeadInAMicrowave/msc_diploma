// teacher_client.cpp
// Generates the span-labeled training set for model 2 (the toxic-span tagger)
// by asking a teacher model to mark deletions inline.
// Env: ANTHROPIC_API_KEY (required), VOCAB_TXT (path to a BERT vocab).
// Usage: ./teacher_client <seed.txt> <out.jsonl> [--limit N]

#include "wordpiece.hpp"
#include "span_markup.hpp"

#include <nlohmann/json.hpp>
#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

static const char* kTeacherModel = "claude-sonnet-5";
static const char* kApiUrl       = "https://api.anthropic.com/v1/messages";
static const char* kApiVersion   = "2023-06-01";

static size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    static_cast<std::string*>(userdata)->append(ptr, size * nmemb);
    return size * nmemb;
}

// System prompt, verbatim from teacher_prompt.md section 2. The policy, the
// prompt, and the seed must stay in agreement: edit the prompt document first,
// then mirror here.
static const char* kSystemPrompt = R"PROMPT(You annotate video-game chat messages for a toxicity-removal system. You will receive one raw chat message. Mark the minimal text that must be deleted to make the message non-toxic, while keeping as much of the player's legitimate message as possible.

OUTPUT FORMAT (strict):
- Return the ORIGINAL message, unchanged, with each span to delete wrapped in <del> and </del>.
- Deletion only. Never paraphrase, insert, reorder, or change spelling, casing, or punctuation. If the tags are removed, the text must equal the input exactly.
- If nothing should be deleted, return the message unchanged with no tags.
- If the entire message is toxic with no keepable content, wrap the whole message in a single <del>...</del>.
- Output the marked-up text only. No explanations, no quotes, no code fences.

WHAT TO DELETE:
1. Profanity and slurs, including disguised forms: leetspeak (sh1t, b1tch), spaced letters (f u c k), substitutions (phuck, azzhole), and acronym profanity (wtf, stfu, omfg). Delete these even when not aimed at anyone, and even in a positive context ("holy sh1t nice" -> delete "sh1t"). The span depends on the word's role: an EXPLETIVE used for emphasis is deleted alone ("fuck this lag" -> "this lag"); a profane word used AS THE INSULT ABOUT A PERSON ("quit being a prick", "you're such a bitch") is hostility, and the whole clause goes.
2. Swear-free hostility that attacks a PERSON: their worth, their competence framed as a personal failing, or their identity. For example "you are such a trash player", "you're useless", "how are you this bad", "you're dead weight and nothing else", "you have no business being in this rank". Telling someone they should not be playing is hostility however mild the words: "uninstall", "go back to the tutorial", "go back to bot games". A sentence whose whole content is that a specific person is bad is an attack even with a mild word: "you are such a noob", "diff jungle". This includes attacks on the TEAM or group collectively: "this team is garbage", "you're all useless", "everyone here is braindead". A team is people. Delete the clause that carries the attack.

WHAT TO KEEP:
- Criticism of a PLAY, DECISION, or SITUATION rather than a person: "that was a trash play", "bad call", "that was a garbage call", "we should've grouped", "this lag is unplayable". Keep these even when blunt.
- Collective, self-inclusive criticism of the team's PLAY: "our macro is trash", "we all played that badly", "sloppy play from all of us". This owns a shared mistake and criticises the play, not the people.
- Ritual competitive taunts about the OUTCOME: "ez", "gg ez", "git gud", "get good", "skill issue", and "noob" used as a tag or address on such a taunt ("git gud noob", "noob team lol"). These do not assert a specific person's inadequacy. But "you are such a noob" does, and is deleted (rule 2). Stacking profanity or a demeaning word onto a taunt makes the whole phrase an attack ("you fucking noob", "you worthless noob").
- Self-directed remarks ("I'm trash") and reported speech ("he told me to uninstall").
- All neutral, strategic, and friendly chat.

THE TEST: Does the span strip a specific person's worth (delete), or challenge their skill, outcome, or decision (keep)?

HOW TO DELIMIT:
- Delete the smallest unit that removes the toxicity: a word before a phrase, a phrase before a clause, a clause before a sentence, a sentence before the whole message.
- When deleting a clause, include the punctuation that binds it to the sentence (such as its trailing comma) so the remainder reads cleanly. Check the remainder: a doubled comma means the comma belonged in the span. Do not include the surrounding spaces.
- Include a conjunction that would be left dangling: "push mid, you're dead weight but at least you can push" -> delete "you're dead weight but"; "stfu and play" -> delete "stfu and".
- Never leave a stub. If the smallest deletion leaves text that is not a message ("quit being a"), or that only continues the same attack on the same person ("never ganks" after "fuck this useless jungler,"), the whole message is toxic: wrap it all.
- Keep exasperation that belongs to the instruction: "buy a control ward for once you trash support" -> delete only "you trash support".
- Mark separate toxic parts as separate <del> spans. Do not merge them across clean text.
- When a clause mixes an attack with a needed instruction, delete only the attacking words and keep the instruction.
- When profanity is stacked onto a taunt aimed at a person ("you fucking noob", "phuck that noob"), the whole phrase is the attack, not just the profane word. Delete all of it. If nothing else remains, delete the whole message.

WHEN UNCERTAIN: prefer the smaller deletion, or no deletion. Over-deleting silences a real message; under-deleting can be caught later. Do not guess on sarcasm. If it is not clearly demeaning, keep it.

EXAMPLES:

Input: You should've listened to me and went top. You are such a trash player, when I call go baron you go baron.
Output: You should've listened to me and went top. <del>You are such a trash player,</del> when I call go baron you go baron.

Input: fuck this lag
Output: <del>fuck</del> this lag

Input: how are you this bad at this game
Output: <del>how are you this bad at this game</del>

Input: gg wp everyone
Output: gg wp everyone

Input: that was a garbage call, we should group
Output: that was a garbage call, we should group

Input: ez game get good scrub
Output: ez game get good <del>scrub</del>

Input: stop feeding you useless bot, group mid
Output: <del>stop feeding you useless bot,</del> group mid

Input: sh1t that hurt
Output: <del>sh1t</del> that hurt

Input: f u c k this game
Output: <del>f u c k</del> this game

Input: that bitch stole my kill
Output: that <del>bitch</del> stole my kill

Input: fuck you, you useless player
Output: <del>fuck you, you useless player</del>

Input: go top you moron
Output: go top <del>you moron</del>

Input: nice shot idiot, now push mid you clown
Output: nice shot <del>idiot,</del> now push mid <del>you clown</del>

Input: I'm trash at this hero honestly
Output: I'm trash at this hero honestly

Input: you're trash and honestly just uninstall
Output: <del>you're trash and honestly just uninstall</del>

Input: this team is garbage
Output: <del>this team is garbage</del>

Input: you're all useless, just group and def
Output: <del>you're all useless,</del> just group and def

Input: our macro is trash right now, let's regroup
Output: our macro is trash right now, let's regroup

Input: you fucking noob
Output: <del>you fucking noob</del>

Input: you fucking noob, group mid
Output: <del>you fucking noob,</del> group mid

Input: you're dead weight and nothing else
Output: <del>you're dead weight and nothing else</del>

Input: you have no business being in this rank
Output: <del>you have no business being in this rank</del>

Input: quit being a prick
Output: <del>quit being a prick</del>

Input: fuck this useless jungler, never ganks
Output: <del>fuck this useless jungler, never ganks</del>

Input: push mid, you're dead weight but at least you can push
Output: push mid, <del>you're dead weight but</del> at least you can push

Input: stfu and play
Output: <del>stfu and</del> play

Input: buy a control ward for once you trash support
Output: buy a control ward for once <del>you trash support</del>

Input: you are such a noob
Output: <del>you are such a noob</del>

Input: git gud noob
Output: git gud noob

Input: go back to the tutorial
Output: <del>go back to the tutorial</del>

Input: holy sh1t nice
Output: holy <del>sh1t</del> nice

Input: phuck that noob
Output: <del>phuck that noob</del>)PROMPT";

// Returns the teacher's raw text content, or an empty string on transport or
// response-parse failure. The caller treats empty as a drop.
static std::string call_teacher(CURL* curl, const std::string& api_key,
                                const std::string& message) {
    json body = {
        {"model", kTeacherModel},
        {"max_tokens", 1024},
        {"system", kSystemPrompt},
        {"messages", json::array({ json{{"role", "user"}, {"content", message}} })},
    };
    std::string payload = body.dump();
    std::string response;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "content-type: application/json");
    headers = curl_slist_append(headers, ("x-api-key: " + api_key).c_str());
    headers = curl_slist_append(headers, ("anthropic-version: " + std::string(kApiVersion)).c_str());

    curl_easy_setopt(curl, CURLOPT_URL, kApiUrl);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

    CURLcode rc = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);

    if (rc != CURLE_OK) {
        std::cerr << "[warn] curl error: " << curl_easy_strerror(rc) << "\n";
        return "";
    }
    if (status != 200) {
        std::cerr << "[warn] HTTP " << status << ": " << response.substr(0, 300) << "\n";
        return "";
    }
    try {
        json j = json::parse(response);
        for (const auto& block : j.at("content"))
            if (block.value("type", "") == "text") return block.value("text", "");
    } catch (const std::exception& e) {
        std::cerr << "[warn] response parse error: " << e.what() << "\n";
    }
    return "";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <seed.txt> <out.jsonl> [--limit N]\n";
        return 1;
    }
    std::string seed_path = argv[1], out_path = argv[2];
    long limit = -1;
    for (int i = 3; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--limit") limit = std::stol(argv[i + 1]);

    const char* key_env = std::getenv("ANTHROPIC_API_KEY");
    if (!key_env) { std::cerr << "[error] ANTHROPIC_API_KEY not set\n"; return 1; }
    std::string api_key = key_env;

    // Only basic_tokenize is needed here, but it lives on WordPieceTokenizer,
    // which requires a vocab containing the special tokens. Point VOCAB_TXT at
    // the exported vocab.txt (or any BERT vocab with [CLS] [SEP] [UNK]).
    const char* vocab_env = std::getenv("VOCAB_TXT");
    std::string vocab_path = vocab_env ? vocab_env : "artifacts/vocab.txt";
    std::unique_ptr<wp::WordPieceTokenizer> tok;
    try {
        tok = std::make_unique<wp::WordPieceTokenizer>(vocab_path);
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what()
                  << "\n(hint: set VOCAB_TXT to the exported vocab.txt path)\n";
        return 1;
    }

    std::ifstream in(seed_path);
    if (!in) { std::cerr << "[error] cannot open " << seed_path << "\n"; return 1; }
    std::ofstream fout(out_path);
    if (!fout) { std::cerr << "[error] cannot open " << out_path << "\n"; return 1; }

    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl = curl_easy_init();
    if (!curl) { std::cerr << "[error] curl init failed\n"; return 1; }

    std::string line;
    std::string category = "UNCATEGORIZED";
    long done = 0, kept = 0, dropped = 0;
    // Per category, how often the teacher chose each branch:
    // [0] out (no span), [1] partial (some spans), [2] fused (whole message).
    // Read against the category's author intent, this is the disagreement QC.
    std::map<std::string, std::array<long, 3>> by_cat;

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        line = sm::trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') {                              // structure, not data
            std::string cat;
            if (sm::parse_category_header(line, cat)) category = cat;
            continue;
        }
        if (limit >= 0 && done >= limit) break;
        ++done;

        auto drop = [&](const std::string& why) {
            ++dropped;
            std::cerr << "[drop] line " << done << " [" << category << "] " << why
                      << ": \"" << line << "\"\n";
        };

        // Pre-checks that cost no API call.
        if (sm::has_reserved_marker(line)) { drop("input contains reserved <del> marker"); continue; }
        auto words = tok->basic_tokenize(line);
        if (words.empty()) { drop("no tokens"); continue; }

        std::string raw = call_teacher(curl, api_key, line);
        std::string marked = sm::clean_response(raw);
        std::vector<sm::Span> spans;
        std::string why;

        if (raw.empty()) {
            drop("transport or response failure");
        } else if (!sm::parse_markup(line, marked, spans, why)) {
            drop(why);
        } else if (!sm::spans_word_aligned(spans, words)) {
            drop("span boundary falls inside a word");
        } else {
            auto tags = sm::tag_words(spans, words);
            size_t n_toxic = static_cast<size_t>(
                std::count(tags.begin(), tags.end(), std::string("TOXIC")));
            int branch = (n_toxic == 0) ? 0 : (n_toxic == tags.size() ? 2 : 1);
            by_cat[category][branch]++;

            json row;
            row["tokens"] = json::array();
            for (const auto& w : words) row["tokens"].push_back(w.text);
            row["tags"] = tags;
            row["category"] = category;
            fout << row.dump() << "\n";
            ++kept;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(120));  // be polite to the API
        if (done % 50 == 0)
            std::cerr << "[progress] " << done << " processed, " << kept
                      << " kept, " << dropped << " dropped\n";
    }

    curl_easy_cleanup(curl);
    curl_global_cleanup();

    std::cerr << "[done] " << kept << " kept, " << dropped << " dropped -> " << out_path << "\n";
    std::cerr << "[branches by category]  out = no span, partial = some spans, fused = whole message\n";
    for (const auto& [cat, c] : by_cat)
        std::cerr << "  " << cat << ": out=" << c[0] << " partial=" << c[1]
                  << " fused=" << c[2] << "\n";
    return 0;
}
