# Toxic-Span Annotation Policy

**Revision 2.** Amended after the blind human-labelling pass on the test set, which surfaced four
underspecified conventions (binding comma, dangling conjunction, profane predicate, non-message
residue) and six boundary decisions (§7, §4a, §4b). Decisions are marked **[DECISION]** and are
revisable; flipping one is a one-line edit here plus the matching edit in the teacher prompt.

**Purpose.** This document defines what the system treats as removable toxicity. It governs three things that must agree with one another: the teacher prompt that generates training data, the token-level `TOXIC`/`OUT` labels that train model 2, and the human judgement used to audit both. If these three drift apart, the model learns an incoherent target. This policy is the single source of truth for "what should be removed."

It is written to be encoded almost verbatim into the teacher prompt and to be lifted, with light edits, into the dataset/methodology chapter of the thesis.

---

## 1. What the annotation produces

For each message, annotation produces the **minimally deleted version**: the same message with the smallest amount of text removed such that (a) the residual is no longer toxic and (b) as much non-toxic content as possible survives. The removed character ranges are the **toxic spans**. Every token inside a removed span is labelled `TOXIC`; every other token is `OUT`.

Two equivalent views, both of which the annotator (or teacher) must satisfy simultaneously:

- **Deletion view:** produce the cleaned message by deleting spans.
- **Labelling view:** the diff between original and cleaned recovers the spans, which become per-token labels.

A message may contain **zero, one, or several** toxic spans. Model 2 is trained to tag all of them in a single pass; the runtime loop removes them one at a time. Annotation therefore marks *every* toxic span, each delimited as tightly as the rules below allow.

---

## 2. Governing principle: minimal removal, maximal retention

The system's job is to keep the player's legitimate message alive while removing the toxicity. Every rule in this document serves that principle. When two annotations both yield a non-toxic residual, **the one that deletes less is correct.**

Concretely: prefer removing a word over a phrase, a phrase over a clause, a clause over a sentence, a sentence over the whole message. Escalate only when the smaller removal leaves toxicity behind.

This mirrors the runtime's greedy minimal-removal loop. The annotation standard and the deletion policy are deliberately the same shape.

---

## 3. The separability test

Before annotating, decide which of two cases a toxic message is in. This decision maps directly onto the system's two branches.

**Separable.** The toxicity lives in an identifiable span, and deleting that span leaves a coherent, non-toxic residual that still carries real content. → Annotate the span(s). This is the common case and the one the system is built to handle well.

> *"You should've listened to me and went top. You are such a trash player, when I call go baron you go baron."*
> Separable. Remove *"You are such a trash player,"*. Residual: *"You should've listened to me and went top. When I call go baron you go baron."* — coherent and content-bearing.

**Fused.** The toxicity is the message. There is no span whose removal leaves meaningful non-toxic content, because the hostile proposition *is* the content. → Mark the **whole message** as a single toxic span (equivalently: the cleaned version is empty).

> *"how are you this bad at this game"*
> Fused. There is no clause to keep — the entire utterance is the insult. Cleaned version: empty.

The test in one question: **after removing the toxic part, is there a real message left?** If yes, separable. If no, fused.

Two clarifications of "real message", both from the blind pass:

- A residual that only **continues the attack on the same target** is not a real message. *"fuck this useless jungler, never ganks"* → *"never ganks"* is a fragment of the same complaint about the same person; the message is fused.
- A residual that is **not a message at all** is never acceptable. *"quit being a prick"* → removing only *prick* leaves *"quit being a"*, which says nothing; the whole clause is the attack (see §4a on profane predicates). If the smallest removal that satisfies the rules leaves no message, the message is fused. Never leave a stub.

---

## 4. What is toxic (in scope)

Two families are removable.

### 4a. Profanity and slurs (lexical and obfuscated)

Expletives, vulgar terms, and slurs, **including disguised forms**: leetspeak (`sh1t`, `b1tch`), spacing (`f u c k`), substitution (`phuck`, `azzhole`), and acronym profanity (`wtf`, `stfu`, `omfg`). These are toxic **regardless of whether they are directed at a person** — an undirected expletive is still removed.

Span extent depends on the word's **role**, not just its presence:

- An **expletive** — profanity used for emphasis or as an exclamation — is word-tight: remove the token(s) only, not the surrounding clause.

  > *"fuck this lag"* → remove *"fuck"* → *"this lag"*.  *"that fucking jungler never ganks"* → *"that jungler never ganks"*.

- A **profane predicate** — a profane noun or adjective used *as the insult about a person* (*prick*, *bitch*, *asshole* said of or to someone) — is directed hostility, and the span is the clause that carries it (§4b, §6.1). Word-tight removal would leave a stub or a still-hostile frame.

  > *"quit being a prick"* → whole message (fused). *"you're such a bitch, group mid"* → remove *"you're such a bitch,"*.

**[DECISION — celebratory expletives]** An expletive in a positive or celebratory context (*"holy sh1t nice"*, *"fuck yeah, nice shot"*) is **removed**. It is not hostile, but it is profanity, and profanity removal is within this system's scope by title. This is the decision most open to reversal: a system scoped to *toxicity alone* would keep it. If reversed, change this paragraph, the matching line in the teacher prompt, and expect the `PROFANITY_PLAIN` / `PROFANITY_OBFUSCATED` positives in the corpus to shrink.

### 4b. Swear-free directed hostility

Hostility that contains no profanity but attacks a person. This is the capability that distinguishes this system from a wordlist filter, and it is governed by a single criterion:

> **A swear-free span is toxic if it demeans the *person* — their worth, their competence framed as a personal failing, or their identity. "The person" includes a *group of persons*: the team, "you all," "everyone here." It is not toxic if it comments on the *play*, the *situation*, or the *outcome*.**

This "person vs. play" line is the core judgement. Learn it through the contrast:

| Toxic (attacks the person) | Out (comments on play/situation) |
|---|---|
| you are such a trash player | that was a trash play |
| you're garbage at this | that call was garbage |
| how are you this bad | that was a bad call |
| uninstall the game | we should uninstall that strategy |
| go back to the tutorial | let's go back to basics next game |
| stop feeding, you're useless | we're feeding mid, let's group |
| this team is garbage | our macro is trash right now |
| you're all useless | we all played that fight badly |

The left column removes worth from the person. The right column criticises a decision or an event. The system keeps the right column even when the tone is blunt, because blunt strategic criticism is legitimate chat.

**[DECISION — dismissal from play]** Telling a person they should not be playing — *uninstall*, *go back to the tutorial*, *go back to bot games*, *you should not be allowed to play ranked* — is directed hostility regardless of how mildly it is phrased. It attacks the person's standing to participate, not a play. Mild phrasing is not a mitigating factor.

**Group-directed hostility is directed hostility.** A team is people; demeaning them collectively (*this team is garbage*, *you're all useless*, *everyone on this team is braindead*) demeans the persons, and is removed under the same separable/fused test as an attack on one player. The contrast is self-inclusive collective criticism of the *play* — *our macro is trash*, *we all played badly*, *sloppy play from all of us* — which owns a shared mistake rather than stripping anyone's worth, and stays.

---

## 5. What stays (out of scope)

The following are **not** annotated and must survive:

- **Strategic and gameplay criticism** directed at decisions, not people: *"that was the wrong call," "we should've grouped," "bad rotation there."*
- **Situational frustration** not aimed at a person: *"this lag is unplayable," "these servers are awful."*
- **Neutral and pro-social chat:** callouts, encouragement, coordination, *"gg," "nice shot," "regroup at base."*
- **Ritual competitive banter** that does not demean the person's worth (see §7 for the exact boundary — this is the contestable frontier and is stated explicitly rather than left to intuition).

If a message is entirely out of scope, it has no spans. (Note: model 1 gates first, so most such messages never reach annotation at all. But the corpus deliberately includes some, so model 2 learns restraint — a message reaching model 2 is not automatically toxic.)

---

## 6. Span delimitation rules

Once a toxic part is identified, delimit it precisely:

1. **Include the whole hostile unit, and nothing else.** For directed hostility, the span is the clause that carries the attack (*"you are such a trash player"*), including a directly attached address term (*"you idiot"* → whole span). Do not swallow an adjacent neutral clause.
2. **Include trailing punctuation that binds the span to the sentence** (the comma in *"trash player,"*) so the residual reads cleanly. This is the convention human annotators most often skip. Check the residual: *"don't chase, you trash player, they have ult up"* with the comma left out yields *"don't chase, , they have ult up"* — a double comma is the sign the comma belonged to the span. Leading/trailing whitespace cleanup and sentence-start recasing are handled by the runtime, not the annotation — annotate on the raw text.
3. **Include a coordinating conjunction that would otherwise dangle** at the edge of the residual. *"push mid, you're dead weight but at least you can push"* → remove *"you're dead weight but"*, leaving *"push mid, at least you can push"*, not *"push mid, but at least you can push"*. *"stfu and play"* → remove *"stfu and"*, leaving *"play"*. The conjunction joined the attack to the rest; without the attack it joins nothing.
4. **Expletive spans are word-tight** (§4a): the token(s) only. **Profane predicates are clause-level** (§4a).
5. **Keep exasperation that belongs to the instruction.** *"buy a control ward for once you trash support"* → remove *"you trash support"* only. *"for once"* is blunt, but it modifies the instruction, not the person; taking it is over-removal (§2).
6. **Multiple independent toxic spans get separate annotations**, each minimally scoped. Do not merge two toxic clauses separated by clean text into one span.

   > *"you're trash and this team is garbage too"* → two spans if both are directed hostility, or one span if the second is situational — judge each independently.
7. **Overlap with neutral content:** if a single clause fuses toxic and necessary information inseparably, treat it as fused *for that clause* — but first try to split. Prefer *"go top ~~you moron~~"* (remove two words) over deleting the whole instruction.

---

## 7. Boundary cases — explicit rulings

These are the contestable calls. The thesis must state a position rather than pretend the line is objective; here is the position, and it is a **tunable frontier** the evaluation can move.

| Case | Ruling | Rationale |
|---|---|---|
| `you totally suck` | **Toxic** | Generalised demeaning of the person, not a specific play. |
| `you are such a trash player` | **Toxic** | Canonical directed insult. |
| `git gud` / `get good` | **Out (default)** | Imperative taunt; asserts a skill gap, does not strip personal worth. |
| `ez` / `gg ez` | **Out (default)** | Ritualised competitive taunt about the *outcome*. |
| `noob` as a **tag or vocative** (`git gud noob`, `noob team lol`) | **Out (default)** | Mild, ubiquitous label attached to a taunt; does not assert anything about the person's worth. |
| `noob` as a **predicate** (`you are such a noob`, `you're a noob`) | **Toxic** **[DECISION]** | A sentence whose whole content is an assertion of the person's inadequacy is directed hostility, whatever the word. The blind pass placed this toxic; the earlier lean-out did not distinguish tag from predicate. Also toxic when stacked with a demeaning modifier or profanity (*"you worthless noob"*, *"you fucking noob"*; see the stacked-attack rule). |
| `diff jungle` / `diff <role>` | **Toxic** **[DECISION]** | Names a specific player as the reason for the loss; it is about a person's inadequacy, not the outcome. Contrast `ez`, which is about the result. The most contestable ruling in this table. |
| `you fucking noob` (profanity stacked on a taunt) | **Toxic (whole phrase)** | Profanity signals hostile intent and escalates the taunt into an attack; with nothing else in the message, it is fused. |
| `uninstall` (directed) | **Toxic** | Tells the person they should not play — attacks their right to participate. |
| `scrub` / `bot` (as insult) | **Toxic** | Demeaning personal labels when directed at a player. |
| `this team is garbage` / `you're all useless` (group-directed) | **Toxic** | A team is people; collective demeaning is directed hostility. |
| `our macro is trash` / `we all played badly` (self-inclusive play) | **Out** | Owns a shared mistake; criticises the play, not the people. |

**Default lean:** ritual taunts about the *outcome* (`ez`, `gg ez`, `git gud`, `skill issue`) sit **out** of scope, so the system errs toward *retention* of banter rather than over-policing. The blind pass moved two items across the line — `noob` as a predicate and `diff <role>` — on the principle that a taunt which asserts a *specific person's* inadequacy is no longer about the outcome. That principle, not the word list, is the ruling. This is a deliberate, stated choice: over-removal silences legitimate speech (§10), and the culture of game chat tolerates ritual taunting. If the thesis wants a stricter system, this table is the knob to turn, and moving it is a clean ablation.

The rule that resolves new cases: **does it strip the person's worth (toxic), or challenge their skill/outcome (out)?**

**Stacked-attack rule.** Profanity stacked onto a frontier taunt aimed at a person converts the taunt into a directed attack: *you fucking noob* is not ritual banter with an expletive attached, it is hostility. Delete the whole phrase, not just the expletive — then apply the ordinary separable/fused test. Standing alone, *you fucking noob* has no residue and is fused (whole message deleted); *you fucking noob, group mid* is separable (the instruction survives). The override applies only to taunts directed at *another* player: *fuck, I'm such a noob* is self-directed, so only the expletive goes and *I'm such a noob* stays.

---

## 8. Further edge cases

- **Self-directed:** *"I'm trash," "I feed every game"* → **Out.** The person is not attacking another player; no target.
- **Sarcasm / backhanded:** *"wow, great play, genius"* → judge the intent. If it demeans the person, toxic; delimit the sarcastic span. This is genuinely hard and low-agreement — flag such items for review rather than guessing.
- **Quoted / reported toxicity:** *"he told me to uninstall"* → **Out** by default (reporting, not attacking), unless the message endorses the attack.
- **Conditional / hypothetical hostility:** *"if you do that again you're a clown"* → toxic if it lands as an insult; span is the insulting clause.
- **Mixed profanity + hostility:** apply both rules. *"fuck you, you useless player"* → the whole thing is directed hostility; remove the clause, not just the expletive. The same holds when profanity is stacked on a frontier taunt (*"you fucking noob"*, *"phuck that noob"* → whole phrase; see the stacked-attack rule in §7) and when the profane word is the predicate (*"quit being a prick"*; §4a).
- **Empty after cleaning:** if minimal removal empties the message, that is the fused branch (§3) — correct, not an error.

---

## 9. Worked examples

| Original | Cleaned (residual) | Span(s) removed | Branch |
|---|---|---|---|
| You should've listened and went top. You are such a trash player, when I call baron you go baron. | You should've listened and went top. When I call baron you go baron. | *You are such a trash player,* | Separable |
| fuck this lag | this lag | *fuck* | Separable |
| how are you this bad at this game | *(empty)* | whole message | Fused |
| gg wp everyone | gg wp everyone | — | Out (no span) |
| stop feeding you useless bot, group mid | group mid | *stop feeding you useless bot,* | Separable |
| that was a garbage call, we should group | that was a garbage call, we should group | — | Out (criticises the call, not the person) |
| this team is garbage | *(empty)* | whole message | Fused (group-directed) |
| you're all useless, just group and def | just group and def | *you're all useless,* | Separable (group-directed) |
| you fucking noob | *(empty)* | whole message | Fused (stacked attack: profanity on a taunt, no residue) |
| you fucking noob, group mid | group mid | *you fucking noob,* | Separable (stacked attack, instruction kept) |
| quit being a prick | *(empty)* | whole message | Fused (profane predicate; *"quit being a"* is a stub) |
| fuck this useless jungler, never ganks | *(empty)* | whole message | Fused (*"never ganks"* only continues the attack) |
| push mid, you're dead weight but at least you can push | push mid, at least you can push | *you're dead weight but* | Separable (dangling conjunction taken) |
| buy a control ward for once you trash support | buy a control ward for once | *you trash support* | Separable (*"for once"* belongs to the instruction) |
| you are such a noob | *(empty)* | whole message | Fused (predicate, §7 decision) |
| git gud noob | git gud noob | — | Out (tag/vocative, §7) |
| go back to the tutorial | *(empty)* | whole message | Fused (dismissal, §4b decision) |
| holy sh1t nice | holy nice | *sh1t* | Separable (celebratory expletive still removed, §4a decision) |
| ez game get good scrub | ez game get good | *scrub* (directed insult); `ez`/`get good` retained | Separable |
| you're trash and honestly just uninstall | *(empty)* | whole message | Fused (no neutral residue) |

The last two rows show the frontier at work: *scrub* is a directed personal label (removed) while *ez*/*get good* stay (§7); and a message that is *entirely* directed hostility with no keepable content collapses to fused even though it contains no profanity.

---

## 10. Consistency and the asymmetry of errors

Two annotation errors are not equal, and this ties directly to the thesis's asymmetric-cost argument:

- **Over-annotation** (marking non-toxic text) trains the model to delete legitimate speech — it silences the player. This is the false-positive analogue and is the more damaging error for a *rewriting* system, because the player's real message is destroyed.
- **Under-annotation** (missing toxicity) lets hostility through — but the runtime loop and model 1's re-check provide a second chance to catch it, and whole-message deletion is the backstop.

Because the loop backstops under-annotation but nothing backstops over-annotation, **when genuinely uncertain, prefer the smaller span or leave it out.** Retention is the safer default. This is why §7's frontier leans out.

**Annotator checklist (from the blind pass).** Thirteen of twenty-three human–teacher disagreements on the first test set were conventions, not judgements. Before marking a span, check: (1) the binding comma is inside it; (2) a conjunction that would dangle is inside it; (3) the residual is a real message — not a stub, not a continuation of the same attack; (4) a profane word *about a person* takes its clause, an expletive takes only itself; (5) exasperation that modifies the instruction stays.

**Known teacher failure modes (same pass).** The LLM teacher missed directed hostility that has *no lexical cue* (*"you're dead weight and nothing else"*, *"you have no business being in this rank"*, *"you're the worst mid I've seen"*), and applied word-tight removal to profane predicates, leaving stubs. Both are addressed in the prompt (rules and examples); the teacher's error rate on the human-verified test rows is the estimate of training-label noise and is reported as such.

Ambiguous items (sarcasm, novel slang, culturally specific taunts) should be flagged rather than force-labelled, so inter-annotator disagreement can be measured — that agreement rate is itself a reportable result and an honest acknowledgement that the toxic/non-toxic line is partly subjective.

---

## 11. Relationship to model 1

Model 1 (the TextCNN gate) and this policy are **aligned but trained independently**, so they will not agree perfectly. That gap is expected and is what the runtime loop exists to absorb: model 1 decides *whether* a residual is still toxic; this policy defines *what* model 2 removes. Annotators should label to this internal standard consistently and **not** try to reverse-engineer model 1's decision boundary. Residual disagreement is handled at runtime by iterating (or by whole-message deletion when the loop cannot satisfy the gate), not by bending the annotation.
