# Teacher Prompt and Span-Labeling Protocol (revision 2, aligned with annotation_policy.md rev. 2)

This document turns `annotation_policy.md` into the artifact the teacher client actually sends, plus the protocol that converts the teacher's answer into per-token `TOXIC`/`OUT` labels. It has three parts: the labeling protocol (what the teacher returns and how it becomes labels), the system prompt verbatim, and operational notes.

---

## 1. Protocol: inline deletion markup, not free-text diff

The earlier plan was "teacher returns the cleaned message; diff it against the original to recover spans." That is fragile. Any incidental change the teacher makes — a normalized space, a fixed capital, a dropped comma, a light paraphrase — produces a diff full of spurious insertions and misaligned deletions, and there is no clean way to tell a real toxic span from a cosmetic edit.

The robust alternative is to have the teacher return the **original message with deletions marked inline**:

```
You should've listened to me and went top. <del>You are such a trash player,</del> when I call go baron you go baron.
```

This gives you an exact, alignment-free recovery of spans and an invariant you can check mechanically:

> **Invariant.** Stripping every `<del>` and `</del>` tag from the teacher's output must reproduce the input **byte-for-byte**.

If the invariant holds, every bracketed range is a genuine deletion and nothing else was touched. If it fails, the teacher paraphrased, inserted, or reformatted, and the row is dropped. The "diff" becomes trivial: the spans *are* the tagged ranges. Both views from policy §1 — the deletion view (strip tagged content → cleaned message) and the labelling view (tagged ranges → labels) — fall out of the same string.

The fused case needs no special token: the teacher wraps the whole message in one `<del>…</del>` and the cleaned version is empty. The no-span case is the message returned untagged.

### From spans to token labels

The client sends the **raw message** (not the pre-tokenized word array the current client sends), because clause-level judgement needs the natural sentence, and the invariant makes raw text safe. On receipt:

1. **Strip tags and compare** to the original. Mismatch → drop the row.
2. **Check tag well-formedness**: balanced, non-nested, non-overlapping. Malformed → drop.
3. **Extract each tagged range** as a `[start, end)` byte offset into the original.
4. **Map ranges to words** using `basic_tokenize`, which already returns each word's `[start, end)` offsets. A word is `TOXIC` if its range overlaps any deletion range; otherwise `OUT`.
5. **Enforce word-boundary alignment.** Policy §6 makes every span word-aligned (profanity is word-tight; clauses are whole). A span that begins or ends *inside* a word indicates a sloppy answer → drop rather than snap, to keep the corpus pure.
6. Emit the row in the **same JSONL shape as today** — `{"tokens":[...],"tags":[...]}` — with `OUT`/`TOXIC` in place of `KEEP`/`PROF`.

Because the output shape is unchanged, `train_export.py` needs only the label schema swapped; its alignment logic (first subword carries the label, continuations ignored) is untouched. A fully tagged message yields all-`TOXIC`; an untagged message yields all-`OUT`.

---

## 2. The system prompt (paste verbatim)

Replace the body of `build_system_prompt()` with the text below. The few-shot examples are part of the prompt, as in the current client.

```
You annotate video-game chat messages for a toxicity-removal system. You will receive one raw chat message. Mark the minimal text that must be deleted to make the message non-toxic, while keeping as much of the player's legitimate message as possible.

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
Output: <del>phuck that noob</del>
```

### What the examples teach, and why each is there

The set is chosen so every rule in the policy has at least one instance the teacher can pattern-match against:

- **Separable clause removal with binding comma** — the baron example (§3, §6.2).
- **Word-tight profanity**, plain and obfuscated — `fuck`, `sh1t`, `f u c k` (§4a, §6.3). The spaced form shows a span covering several basic tokens.
- **Fused → whole message**, both without profanity (`how are you this bad`) and with (`you're trash... uninstall`) (§3). The second proves fused is about *keepable residue*, not about swear words.
- **Restraint** — `gg wp` and `garbage call` return untagged (§5). `garbage call` is the person-vs-play line in action: the same word that is toxic in `trash player` is fine aimed at a call.
- **The frontier** — `ez game get good scrub` keeps `ez` and `get good`, removes `scrub` (§7).
- **Word-tight even when directed at a person** — `that bitch stole my kill` keeps the event report; contrast with `fuck you, you useless player`, where the whole clause is the attack and there is no residue to keep. This pair teaches *why* profanity is word-tight while hostility is clause-level: it is about what survives.
- **Mixed attack + instruction** — `go top you moron` keeps the instruction (§6.5).
- **Separate spans, not merged**, plus **don't guess on sarcasm** — `nice shot idiot, now push mid you clown` keeps `nice shot` (§6.4, §8).
- **Self-directed stays** — `I'm trash at this hero` (§8).
- **Group-directed is directed** — `this team is garbage` goes (fused) and `you're all useless, just group and def` keeps the instruction (separable), while `our macro is trash right now` stays. The self-inclusive *play* criticism is the contrast: it owns a shared mistake instead of stripping anyone's worth (§4b, §7).
- **Lexically clean hostility** — `you're dead weight and nothing else`, `you have no business being in this rank`: no swear, no obvious insult word; the teacher missed exactly these on the first run, so they are now worked examples.
- **Profane predicate vs expletive** — `quit being a prick` goes whole (the word IS the insult; `quit being a` is a stub), while `that bitch stole my kill` stays word-tight (§4a).
- **Never leave a stub, never leave a continuation** — `fuck this useless jungler, never ganks` goes whole: `never ganks` only continues the attack (§3).
- **Dangling conjunction** — `you're dead weight but`, `stfu and` (§6.3). **Exasperation stays** — `for once` (§6.5).
- **Predicate vs tag** — `you are such a noob` goes, `git gud noob` stays (§7 decision). **Dismissal** — `go back to the tutorial` goes (§4b decision). **Celebratory expletive still removed** — `holy sh1t nice` → `holy nice` (§4a decision).
- **Stacked attack** — `you fucking noob` goes entirely (fused): profanity on a person-directed taunt makes the whole phrase the attack, so the bare-`noob`-stays default no longer applies. `you fucking noob, group mid` shows the instruction still survives (§7 stacked-attack rule).

---

## 3. Operational notes

**Corpus composition is now the bottleneck, and it is lopsided.** The current seed is profanity-heavy, and the only swear-free hostile examples in it (`you totally suck`, `get good scrub`, `absolute trash team`) were labelled `KEEP` under the old scope. Under the new policy some of those flip, but more importantly there are almost **no positive examples of the system's headline capability** — swear-free directed hostility. That is the hardest class and the one with the least data. Seed expansion should deliberately over-sample it: directed insults with no profanity, in the full range from mild (`you're kinda bad`) to fused (`how are you this bad`), alongside the *play*-directed near-misses (`that was a bad call`) that teach the boundary. Without that contrast the tagger will learn "any negative word = toxic," which is exactly the wordlist behaviour you are trying to escape.

**Include out-of-scope messages on purpose.** A model 2 that only ever sees toxic inputs learns that something must always be deleted. Feeding it untagged clean messages — even though model 1 gates most of them away at runtime — teaches restraint, which is what protects `that was a garbage call`.

**Send raw text, verify hard, drop on failure.** The invariant check is the whole guarantee. Do not repair a bad answer; drop it and count it, exactly as the current client counts dropped rows. Drop rate is a useful signal of prompt quality — if it is high, the prompt is unclear, not the teacher.

**The teacher model string in `teacher_client.cpp` is dated.** It is hardcoded to `claude-sonnet-4-5`; a current Sonnet-class model (`claude-sonnet-5`) is the natural strong-but-affordable teacher. This is a one-line constant change when you update the client.

**Downstream is nearly untouched.** The JSONL shape is preserved, so `train_export.py` changes only its label schema (`KEEP`/`PROF` → `OUT`/`TOXIC` in `schema.json`). `wordpiece.hpp`, `parity_test.cpp`, and the first-subword alignment all carry over unchanged. The client is the only component that needs new logic: raw-text request, invariant check, tag extraction, offset-to-word mapping.
