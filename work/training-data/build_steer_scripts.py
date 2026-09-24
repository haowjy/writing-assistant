#!/usr/bin/env python3
"""Build steer-scripts.json and steer-scripts.md.

Atoms = user steer messages. Each script is a spine of ordered steers that can be
attached to any world/brief. Placeholders: {protagonist}, {other}, {object}, {place}.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent

KINDS = [
    "ADD_CONSTRAINT",
    "CHANGE_DIRECTION",
    "PUSH_BACK",
    "CLARIFY_ANSWER",
    "REQUEST_REVISION",
    "NEW_MATERIAL",
    "SCOPE_CUT",
    "APPROVE",
    "INTERRUPT",
]
BANDS = ["2-3", "4-6", "8-12"]


def S(text, kind):
    return {"text": text, "kind": kind}


scripts = []


def add(sid, band, pressure, rationale, steers, *, group="standard", tests=None,
        escalates=False):
    assert band in BANDS, band
    assert pressure in {"REQUIRES_ASK", "PERMITS_DEFAULT", "MUST_NOT_ASK"}, pressure
    for st in steers:
        assert st["kind"] in KINDS, (sid, st)
    n = len(steers)
    lo, hi = (2, 3) if band == "2-3" else (4, 6) if band == "4-6" else (8, 12)
    assert lo <= n <= hi, (sid, band, n)
    entry = {
        "id": sid,
        "group": group,
        "depth_band": band,
        "n_steers": n,
        "response_pressure": pressure,
        "rationale": rationale,
        "steers": steers,
    }
    if tests:
        entry["tests"] = tests
    if escalates:
        entry["escalates_conflict"] = True
    scripts.append(entry)


# ============================================================ BAND 2-3 (A)

add("SS-A01", "2-3", "PERMITS_DEFAULT",
    "The correction is unambiguous, so a default is fine; the point is that the "
    "silent assumption gets surfaced and corrected rather than carried forward.",
    [S("They're not strangers. They've known each other for years.", "PUSH_BACK"),
     S("Keep the rest of it as-is - just fix how they know each other.", "APPROVE")],
    group="silent_assumption",
    tests="Model silently assumed {protagonist} and {other} meet for the first time; "
          "steer forces it to revise the relationship without asking what it should be.")

add("SS-A02", "2-3", "REQUIRES_ASK",
    "Whether the relationship is romantic is a real fork the model should not have "
    "taken unilaterally, so a good collaborator surfaces the choice.",
    [S("Wait - you wrote them as lovers. Did I say that?", "PUSH_BACK"),
     S("Leave it ambiguous for now, but ask me before you commit to anything.", "ADD_CONSTRAINT")],
    group="silent_assumption",
    tests="Model silently assumed a romance. Script checks that the agent stops, "
          "names the assumption, and treats the unresolved status as a decision to ask about.")

add("SS-A03", "2-3", "MUST_NOT_ASK",
    "The requested fix is concrete; asking which character should find the object "
    "would be hedging rather than collaborating.",
    [S("I never said {protagonist} was the one who found {object} - that came from nowhere. Fix that.", "PUSH_BACK"),
     S("Rewrite the paragraph so {other} finds it.", "REQUEST_REVISION"),
     S("Good.", "APPROVE")],
    group="silent_assumption",
    tests="Model silently assigned an action to the wrong character. A clear correction "
          "should be executed, not questioned.")

add("SS-A04", "2-3", "REQUIRES_ASK",
    "'Darker' has no single reading; the right move is to ask what kind of darker, "
    "not to pick a default and run.",
    [S("Make it darker.", "ADD_CONSTRAINT"),
     S("I mean less comfort and more dread - not gore or shock, just quieter menace.", "CLARIFY_ANSWER")])

add("SS-A05", "2-3", "PERMITS_DEFAULT",
    "The skip is explicit and the assumption is stated in the steer itself, so the "
    "agent may proceed on that stated assumption.",
    [S("Cut to the morning after, and assume the night passed quietly - no need to show it.", "SCOPE_CUT"),
     S("Actually keep {place} in view the whole time.", "CHANGE_DIRECTION")])

add("SS-A06", "2-3", "MUST_NOT_ASK",
    "The instruction is complete and narrow; asking for elaboration would be a "
    "failure of nerve.",
    [S("End it on {object} on the table. Nothing after that, no reaction.", "ADD_CONSTRAINT"),
     S("And don't explain why.", "ADD_CONSTRAINT")])

add("SS-A07", "2-3", "REQUIRES_ASK",
    "'More likeable' could mean sympathetic, competent, or harmless; the agent should "
    "ask which sense the writer wants before rewriting.",
    [S("Can you make {other} more likeable?", "ADD_CONSTRAINT"),
     S("I mean the reader should root for them, not that they should be nice about it.",
       "CLARIFY_ANSWER"),
     S("Yes, that.", "APPROVE")])

add("SS-A08", "2-3", "PERMITS_DEFAULT",
    "The writer has given a clear direction and reason; the agent can proceed on the "
    "stated reading while staying inside the character's established harshness.",
    [S("No. That's not {protagonist} at all.", "PUSH_BACK"),
     S("You keep smoothing them into someone reasonable. Stop.", "PUSH_BACK"),
     S("Try again. Don't make them apologise.", "ADD_CONSTRAINT")],
    escalates=True)

add("SS-A09", "2-3", "MUST_NOT_ASK",
    "Dropping and re-adding material are both explicit; there is nothing to clarify.",
    [S("Forget the second half. Drop it.", "SCOPE_CUT"),
     S("Put {place} back in though.", "CHANGE_DIRECTION")])

add("SS-A10", "2-3", "PERMITS_DEFAULT",
    "The aside is self-contained and the return is explicit, so the agent can answer "
    "briefly and resume without asking.",
    [S("Different question - what's {protagonist}'s surname again?", "INTERRUPT"),
     S("Right. Now keep going from the doorway.", "CHANGE_DIRECTION")])

add("SS-A11", "2-3", "MUST_NOT_ASK",
    "A new fact plus its consequence is stated plainly; the agent should apply it.",
    [S("New fact: {other} can't read.", "NEW_MATERIAL"),
     S("So the note has to be read aloud - by {protagonist}.", "ADD_CONSTRAINT")])

add("SS-A12", "2-3", "PERMITS_DEFAULT",
    "The preference and the salvage instruction are clear enough to act on without "
    "an interrogative detour.",
    [S("I liked the first version better.", "PUSH_BACK"),
     S("Go back to the first version, but keep the new ending - I liked that part.", "REQUEST_REVISION")])


# ============================================================ BAND 4-6 (B)

add("SS-B01", "4-6", "REQUIRES_ASK",
    "Whether a narrator is unreliable is a global craft decision the writer never "
    "authorized; after fixing it, the agent should ask before reaching for that device.",
    [S("Hold on. Why is {protagonist} lying to the reader now? I didn't ask for that.",
       "PUSH_BACK"),
     S("Take it out. They can be wrong, not deceptive.", "REQUEST_REVISION"),
     S("That's better.", "APPROVE"),
     S("Now push the scene toward the confrontation, but keep it understated.", "ADD_CONSTRAINT")],
    group="silent_assumption",
    tests="Model silently made {protagonist} an unreliable narrator. Script tests "
          "detecting an unauthorized craft device, reverting it, and asking next time.")

add("SS-B02", "4-6", "PERMITS_DEFAULT",
    "The writer supplies the missing fact (daytime) themselves, so the agent may "
    "proceed on that default and revise locally.",
    [S("It's daytime in this scene, just so you know.", "ADD_CONSTRAINT"),
     S("Redo the opening with light coming in.", "REQUEST_REVISION"),
     S("Good. Keep the argument though.", "APPROVE"),
     S("Actually, move the whole argument to {place} and see if it changes.", "CHANGE_DIRECTION")],
    group="silent_assumption",
    tests="Model silently set the scene at night. Script tests correction of an "
          "ambient default the writer never specified.")

add("SS-B03", "4-6", "MUST_NOT_ASK",
    "Killing a character off-page is a major event with a clear correction; the "
    "agent should undo it without asking permission to do so.",
    [S("You can't kill {other} off-screen. Nobody said that.", "PUSH_BACK"),
     S("Put them back. Injured is fine.", "ADD_CONSTRAINT"),
     S("Ok. Now what does {protagonist} actually lose here?", "ADD_CONSTRAINT"),
     S("Take the second option.", "CLARIFY_ANSWER")],
    group="silent_assumption",
    tests="Model silently killed {other} off-screen. Script tests reversing an "
          "unapproved plot event and re-deriving the actual stakes.")

add("SS-B04", "4-6", "PERMITS_DEFAULT",
    "The writer's intent (no reconciliation) is stated repeatedly and clearly; the "
    "agent should execute it rather than negotiate.",
    [S("No - that ending is too neat. They don't make up.", "PUSH_BACK"),
     S("I said don't resolve it and you resolved it anyway.", "PUSH_BACK"),
     S("Strip the apology and the hug. Leave them in the same room, cold.", "REQUEST_REVISION"),
     S("Stop trying to make me like them.", "PUSH_BACK"),
     S("Fine. Read me the version where nothing is fixed.", "REQUEST_REVISION")],
    escalates=True)

add("SS-B05", "4-6", "MUST_NOT_ASK",
    "The constraint is explicit and its consequence is left open for the agent to "
    "work out; asking for the plot would be abdicating the work.",
    [S("Add a constraint: {protagonist} physically can't leave {place}, not just doesn't want to.", "ADD_CONSTRAINT"),
     S("So the whole plan has to change.", "NEW_MATERIAL"),
     S("Right. But keep {object} central.", "ADD_CONSTRAINT"),
     S("Drop the subplot with {other}'s brother.", "SCOPE_CUT")])

add("SS-B06", "4-6", "PERMITS_DEFAULT",
    "The interrupt is explicitly flagged and the return point is given, so the agent "
    "can answer and resume on a stated default.",
    [S("Unrelated - can you draft a two-line note in {protagonist}'s voice?", "INTERRUPT"),
     S("Thanks. Back to the scene.", "CHANGE_DIRECTION"),
     S("Now rewrite the ending so {other} leaves first and {protagonist} stays put.", "REQUEST_REVISION"),
     S("And make {protagonist} not react.", "ADD_CONSTRAINT")])

add("SS-B07", "4-6", "PERMITS_DEFAULT",
    "The reversal is explicit and the replacement is supplied, so the agent can take "
    "the new direction without asking.",
    [S("New character: {protagonist}'s sister shows up.", "NEW_MATERIAL"),
     S("No, not a sister. A former partner.", "CHANGE_DIRECTION"),
     S("They're here for {object}.", "NEW_MATERIAL"),
     S("Keep it brief, one scene.", "ADD_CONSTRAINT"),
     S("Good.", "APPROVE")])

add("SS-B08", "4-6", "REQUIRES_ASK",
    "A two-word tone note is a classic underdetermined request; the agent should ask "
    "what register is wanted rather than guess.",
    [S("Make it funny.", "ADD_CONSTRAINT"),
     S("Not that funny. Dry.", "CLARIFY_ANSWER"),
     S("Better, but the last line is too much.", "PUSH_BACK"),
     S("Cut it.", "SCOPE_CUT"),
     S("Ok, done.", "APPROVE")])

add("SS-B09", "4-6", "MUST_NOT_ASK",
    "The answer resolves the ambiguity the agent raised; the follow-on facts and "
    "rewrite are explicit.",
    [S("Yes, {other} asked first.", "CLARIFY_ANSWER"),
     S("And {protagonist} lied about the answer.", "NEW_MATERIAL"),
     S("That changes the whole scene. Rewrite it.", "CHANGE_DIRECTION"),
     S("Keep the first paragraph.", "ADD_CONSTRAINT")])

add("SS-B10", "4-6", "MUST_NOT_ASK",
    "Each instruction is concrete and self-contained; the agent should just do it.",
    [S("Strip the dialogue tags. All of them.", "ADD_CONSTRAINT"),
     S("Hmm, now the middle is confusing - I can't tell who's speaking in two places.", "PUSH_BACK"),
     S("Add back just 'she said' twice.", "REQUEST_REVISION"),
     S("Fine.", "APPROVE")])

add("SS-B11", "4-6", "PERMITS_DEFAULT",
    "The writer supplies the missing answer and then an explicit reversal, so the "
    "agent can proceed on the latest stated decision.",
    [S("Can we end the chapter earlier?", "SCOPE_CUT"),
     S("Yes, right after {object} appears.", "CLARIFY_ANSWER"),
     S("Actually no - end it before {object}.", "CHANGE_DIRECTION"),
     S("And start next chapter in {place}.", "ADD_CONSTRAINT")])

add("SS-B12", "4-6", "REQUIRES_ASK",
    "The writer rejects every option and asks for 'messier', which is not actionable "
    "until the agent asks what mess would serve the scene.",
    [S("Give me three options for what {other} does next.", "NEW_MATERIAL"),
     S("None of these. Too tidy.", "PUSH_BACK"),
     S("Try again, messier.", "REQUEST_REVISION"),
     S("The third one. Go.", "APPROVE"),
     S("Wait, what does {protagonist} want in that version?", "INTERRUPT")],
    escalates=True)


# ============================================================ BAND 8-12 (C)

add("SS-C01", "8-12", "MUST_NOT_ASK",
    "The factual correction is unambiguous; every later steer is a concrete craft "
    "instruction, so asking would stall a clear loop.",
    [S("This place isn't abandoned - people live here.", "PUSH_BACK"),
     S("Show a few of them in the background - enough that I believe people live here.", "ADD_CONSTRAINT"),
     S("Now take the chase into the market.", "NEW_MATERIAL"),
     S("Slow it down there.", "ADD_CONSTRAINT"),
     S("Hmm, the middle drags again.", "PUSH_BACK"),
     S("Cut the second stall.", "SCOPE_CUT"),
     S("Keep {other} out of it for now.", "ADD_CONSTRAINT"),
     S("Better.", "APPROVE"),
     S("One more thing - what happened to {object}?", "INTERRUPT"),
     S("Right, leave that open.", "APPROVE")],
    group="silent_assumption",
    tests="Model silently treated {place} as abandoned, contradicting the world. Script "
          "tests a long run where the correction must persist across many steers.")

add("SS-C02", "8-12", "REQUIRES_ASK",
    "Character knowledge is exactly the kind of decision the writer must own; the "
    "agent should adopt the stated default now and ask before deciding it again.",
    [S("Wait, does {protagonist} know? I never decided that.", "PUSH_BACK"),
     S("Assume they don't know, but flag it if it changes what they can do.", "ADD_CONSTRAINT"),
     S("Actually, ask me next time before you pick.", "ADD_CONSTRAINT"),
     S("Ok, continue.", "APPROVE"),
     S("Now bring {other} to {place}.", "NEW_MATERIAL"),
     S("They arrive early.", "ADD_CONSTRAINT"),
     S("No - late.", "CHANGE_DIRECTION"),
     S("Late works. Keep it.", "APPROVE"),
     S("What does {other} want from this?", "INTERRUPT"),
     S("Don't answer that yet. Draft the arrival.", "REQUEST_REVISION"),
     S("Good.", "APPROVE")],
    group="silent_assumption",
    tests="Model silently granted {protagonist} knowledge of a secret. Script tests "
          "knowledge-state correction plus an explicit instruction to ask in future.")

add("SS-C03", "8-12", "PERMITS_DEFAULT",
    "The writer states the seasonal assumption, which licenses the agent to proceed; "
    "later steers are concrete revisions.",
    [S("It's winter, if that matters. Assume it does.", "ADD_CONSTRAINT"),
     S("Go back and add breath and cold hands.", "REQUEST_REVISION"),
     S("Both.", "CLARIFY_ANSWER"),
     S("Now the same scene, but the heating's broken.", "CHANGE_DIRECTION"),
     S("Yes, that makes it worse for everyone.", "APPROVE"),
     S("Add that {object} can't freeze.", "NEW_MATERIAL"),
     S("So why is it here then? Figure it out.", "ADD_CONSTRAINT"),
     S("Keep the answer off-page.", "ADD_CONSTRAINT"),
     S("Cut the neighbour scene.", "SCOPE_CUT"),
     S("Fine. Show me the whole thing.", "REQUEST_REVISION")],
    group="silent_assumption",
    tests="Model silently chose a season. Script tests adopting a stated default and "
          "then tracking it as a hard constraint across the revision.")

add("SS-C04", "8-12", "MUST_NOT_ASK",
    "A dead parent is a large unauthorized canon change; the correction and following "
    "revisions are explicit enough that asking would be timid.",
    [S("I didn't say {protagonist}'s parent is dead.", "PUSH_BACK"),
     S("Unkill them and deal with the mess.", "REQUEST_REVISION"),
     S("Now the letter has to mean something else.", "NEW_MATERIAL"),
     S("Yes - keep the letter, change the reason.", "APPROVE"),
     S("Drop the funeral entirely.", "SCOPE_CUT"),
     S("Replace it with a phone call.", "CHANGE_DIRECTION"),
     S("Who's on the other end?", "INTERRUPT"),
     S("The neighbour. And {protagonist} says nothing.", "CLARIFY_ANSWER"),
     S("That's the chapter. Stop there.", "ADD_CONSTRAINT"),
     S("Actually add two lines of {place} at dusk.", "CHANGE_DIRECTION"),
     S("Now it works.", "APPROVE")],
    group="silent_assumption",
    tests="Model silently invented a major backstory death. Script tests reverting "
          "canon, re-deriving dependent objects, and holding the new state.")

add("SS-C05", "8-12", "PERMITS_DEFAULT",
    "The writer is specific about what they reject and what they want instead; the "
    "agent can follow the stated direction without an interrogative pause.",
    [S("The reconciliation is too easy. They haven't earned it.", "PUSH_BACK"),
     S("You keep fixing this and I keep telling you not to.", "PUSH_BACK"),
     S("Take out the apology and the hug. I don't want them touching at all.", "REQUEST_REVISION"),
     S("Now it's just cold. Good.", "APPROVE"),
     S("But you've made {other} cruel. That's not right either.", "PUSH_BACK"),
     S("Reread their last argument. {other} is allowed to be tired here, not cruel.",
       "ADD_CONSTRAINT"),
     S("Yes. Now let them fail to talk.", "ADD_CONSTRAINT"),
     S("End there. Don't resolve it next scene either.", "ADD_CONSTRAINT"),
     S("I mean it. No closure for three chapters.", "ADD_CONSTRAINT"),
     S("Ok.", "APPROVE")],
    escalates=True)

add("SS-C06", "8-12", "REQUIRES_ASK",
    "The writer rejects a coincidence-driven solution but the replacement cause is "
    "never fixed; a good collaborator would ask what should drive the turn.",
    [S("No. That's a completely different story.", "PUSH_BACK"),
     S("You solved it with a coincidence. I hate that.", "PUSH_BACK"),
     S("Rewrite without the letter arriving.", "REQUEST_REVISION"),
     S("Now {protagonist} has nothing to act on. Good.", "APPROVE"),
     S("Give {other} a reason to lie, not a secret.", "NEW_MATERIAL"),
     S("No - make {other} tell the truth badly.", "CHANGE_DIRECTION"),
     S("That's worse. Keep it.", "APPROVE"),
     S("Add that {object} was never missing.", "NEW_MATERIAL"),
     S("Wait, that changes chapter one.", "PUSH_BACK"),
     S("Go back and check we didn't contradict ourselves.", "REQUEST_REVISION"),
     S("Fix only what breaks. Leave the rest.", "ADD_CONSTRAINT"),
     S("And don't explain away the coincidence.", "ADD_CONSTRAINT")],
    escalates=True)

add("SS-C07", "8-12", "PERMITS_DEFAULT",
    "Arc instructions are explicit and the one open question is answered by the "
    "writer, so the agent can proceed on stated decisions.",
    [S("Let's plan the whole arc before drafting.", "ADD_CONSTRAINT"),
     S("Three acts, five beats each.", "ADD_CONSTRAINT"),
     S("Too many. Three each.", "CHANGE_DIRECTION"),
     S("The turn is {other} refusing.", "CLARIFY_ANSWER"),
     S("Put a false victory before it.", "NEW_MATERIAL"),
     S("No, after.", "CHANGE_DIRECTION"),
     S("Draft act one now.", "REQUEST_REVISION"),
     S("Keep it under 800 words.", "ADD_CONSTRAINT"),
     S("Slightly long. Trim {place}.", "REQUEST_REVISION"),
     S("Good enough.", "APPROVE")])

add("SS-C08", "8-12", "MUST_NOT_ASK",
    "Each steer supplies both the new material and the constraint on it; the agent "
    "should draft rather than seek permission.",
    [S("New brief: {place}, one room, two people.", "NEW_MATERIAL"),
     S("{protagonist} is waiting. {other} never shows.", "NEW_MATERIAL"),
     S("Actually {other} shows up late.", "CHANGE_DIRECTION"),
     S("And they're angry about something we don't explain.", "ADD_CONSTRAINT"),
     S("Drop the explanation from the last paragraph.", "SCOPE_CUT"),
     S("Cut the whole conversation short.", "SCOPE_CUT"),
     S("Now write the silence.", "REQUEST_REVISION"),
     S("Too poetic. Plain.", "PUSH_BACK"),
     S("Better.", "APPROVE"),
     S("What's {protagonist} holding?", "INTERRUPT"),
     S("{object}. And they put it down.", "CLARIFY_ANSWER"),
     S("End on that. Done.", "APPROVE")])

add("SS-C09", "8-12", "REQUIRES_ASK",
    "The writer keeps redirecting the reunion's cause without settling it, so the "
    "agent should surface the unresolved motive rather than keep guessing.",
    [S("Bring {other} back into the plot.", "CHANGE_DIRECTION"),
     S("They've been gone four chapters, so it should feel strange.", "ADD_CONSTRAINT"),
     S("Give them a lame excuse.", "NEW_MATERIAL"),
     S("No, make it a real one that {protagonist} doesn't accept.", "CHANGE_DIRECTION"),
     S("Make {protagonist} push them on it.", "REQUEST_REVISION"),
     S("And {other} deflects.", "NEW_MATERIAL"),
     S("Keep the scene long and let it breathe - I want the silence to get uncomfortable.", "ADD_CONSTRAINT"),
     S("Actually, cut it in half.", "SCOPE_CUT"),
     S("The half where they argue.", "CLARIFY_ANSWER"),
     S("Ok. Now what does {protagonist} do after?", "INTERRUPT"),
     S("Leave it. Next chapter.", "APPROVE")],
    escalates=True)

add("SS-C10", "8-12", "MUST_NOT_ASK",
    "Style constraints are precise and reversible on the page; the agent should apply "
    "them directly and accept the writer's verdicts.",
    [S("Rewrite the draft without adverbs.", "ADD_CONSTRAINT"),
     S("That's better but bloodless.", "PUSH_BACK"),
     S("Put back three strong verbs and one adverb.", "REQUEST_REVISION"),
     S("Keep 'quietly'.", "CLARIFY_ANSWER"),
     S("Now tighten every sentence over fifteen words.", "ADD_CONSTRAINT"),
     S("You split some and it reads choppy.", "PUSH_BACK"),
     S("Rejoin the two worst ones.", "REQUEST_REVISION"),
     S("Good.", "APPROVE"),
     S("Different thing - does {protagonist} have a scar?", "INTERRUPT"),
     S("Ok, doesn't matter. Continue.", "CHANGE_DIRECTION"),
     S("Actually it matters. Say where.", "CHANGE_DIRECTION"),
     S("Fine, leave it unmentioned for now.", "SCOPE_CUT")])

add("SS-C11", "8-12", "PERMITS_DEFAULT",
    "The stakes are specified concretely and the resulting plothole is named by the "
    "writer, so the agent can act on the stated fixes.",
    [S("We need to raise the stakes.", "ADD_CONSTRAINT"),
     S("Make {object} the only copy.", "NEW_MATERIAL"),
     S("So if it's lost, it's gone.", "ADD_CONSTRAINT"),
     S("Now {other} wants to destroy it.", "NEW_MATERIAL"),
     S("That's a jump. Motivate it first.", "PUSH_BACK"),
     S("You're overexplaining again. One line.", "REQUEST_REVISION"),
     S("Better. Keep.", "APPROVE"),
     S("Cut the scene where they hide it.", "SCOPE_CUT"),
     S("Wait, then chapter four makes no sense.", "PUSH_BACK"),
     S("Reinsert half of it.", "REQUEST_REVISION"),
     S("Which half? The part in {place}.", "CLARIFY_ANSWER"),
     S("Done.", "APPROVE")])

add("SS-C12", "8-12", "PERMITS_DEFAULT",
    "The side task and the return path are both explicit, so the agent can service "
    "the interrupt and resume on stated instructions.",
    [S("Different project for a sec - outline a short story idea.", "INTERRUPT"),
     S("No, shorter. Four sentences.", "REQUEST_REVISION"),
     S("Hmm, I don't love it. Park it.", "SCOPE_CUT"),
     S("Back to the chapter though.", "CHANGE_DIRECTION"),
     S("Where were we?", "INTERRUPT"),
     S("Right. Continue from the argument.", "REQUEST_REVISION"),
     S("Add {protagonist}'s hesitation before the door.", "ADD_CONSTRAINT"),
     S("Now cut the hesitation.", "CHANGE_DIRECTION"),
     S("No, keep a small version.", "CLARIFY_ANSWER"),
     S("Good.", "APPROVE")])


# ============================================================ build artifact

doc = {"schema_version": 1, "scripts": scripts}

# ---- verification ----
errors = []
for sc in scripts:
    if not sc["steers"]:
        errors.append(f"{sc['id']}: no steers")
    for st in sc["steers"]:
        if "kind" not in st or st["kind"] not in KINDS:
            errors.append(f"{sc['id']}: bad kind {st}")
    for k in ("id", "depth_band", "n_steers", "response_pressure", "rationale"):
        if k not in sc:
            errors.append(f"{sc['id']}: missing {k}")
    if sc["n_steers"] != len(sc["steers"]):
        errors.append(f"{sc['id']}: n_steers mismatch")
    if sc["response_pressure"] not in {"REQUIRES_ASK", "PERMITS_DEFAULT", "MUST_NOT_ASK"}:
        errors.append(f"{sc['id']}: bad pressure")
    if not sc["rationale"].strip():
        errors.append(f"{sc['id']}: empty rationale")

if errors:
    raise SystemExit("VALIDATION ERRORS:\n" + "\n".join(errors))

(OUT / "steer-scripts.json").write_text(json.dumps(doc, indent=2) + "\n")

# ---- coverage ----
band_kind = defaultdict(Counter)
kind_tot = Counter()
pressure_tot = Counter()
group_tot = Counter()
total_steers = 0
terse_total = 0
terse_by_band = Counter()
escalating = []
assumption = []

for sc in scripts:
    band = sc["depth_band"]
    pressure_tot[sc["response_pressure"]] += 1
    group_tot[sc["group"]] += 1
    if sc.get("escalates_conflict"):
        escalating.append(sc["id"])
    if sc["group"] == "silent_assumption":
        assumption.append(sc)
    for st in sc["steers"]:
        total_steers += 1
        band_kind[band][st["kind"]] += 1
        kind_tot[st["kind"]] += 1
        if len(st["text"].split()) < 12:
            terse_total += 1
            terse_by_band[band] += 1

# coverage assertions
for band in BANDS:
    missing = [k for k in KINDS if band_kind[band][k] == 0]
    if missing:
        raise SystemExit(f"band {band} missing kinds: {missing}")
for k in KINDS:
    if kind_tot[k] == 0:
        raise SystemExit(f"kind {k} never used")
if terse_total * 4 < total_steers:
    raise SystemExit("less than a quarter terse")
if len(escalating) < 6:
    raise SystemExit("fewer than 6 escalating scripts")
if len(assumption) != 10:
    raise SystemExit(f"expected 10 assumption scripts, got {len(assumption)}")

# ---- markdown ----
lines = []
lines.append("# Steer Scripts - Coverage Companion")
lines.append("")
lines.append("Reusable steer scripts for the conversational writing collaborator. Each "
             "steer is a user message that continues a session after an opening brief. "
             "Placeholders (`{protagonist}`, `{other}`, `{object}`, `{place}`) keep a "
             "script composable with any world.")
lines.append("")
lines.append(f"Artifact: `steer-scripts.json` (schema_version 1). Scripts: "
             f"**{len(scripts)}**. Steers: **{total_steers}**.")
lines.append("")
lines.append("## Depth x kind coverage")
lines.append("")
header = "| band | " + " | ".join(KINDS) + " | total |"
lines.append(header)
lines.append("|" + "---|" * (len(KINDS) + 2))
for band in BANDS:
    row = [band] + [str(band_kind[band][k]) for k in KINDS]
    row.append(str(sum(band_kind[band].values())))
    lines.append("| " + " | ".join(row) + " |")
tot_row = ["ALL"] + [str(kind_tot[k]) for k in KINDS] + [str(total_steers)]
lines.append("| " + " | ".join(tot_row) + " |")
lines.append("")
lines.append("## Response pressure")
lines.append("")
lines.append("| pressure | scripts |")
lines.append("|---|---|")
for p in ["REQUIRES_ASK", "PERMITS_DEFAULT", "MUST_NOT_ASK"]:
    lines.append(f"| {p} | {pressure_tot[p]} |")
lines.append("")
lines.append("## Other checks")
lines.append("")
lines.append(f"- Terse steers (under 12 words): **{terse_total}/{total_steers}** "
             f"({100.0 * terse_total / total_steers:.0f}%), required >= 25%.")
lines.append(f"- Escalating-conflict scripts (no fast resolution): **{len(escalating)}** "
             f"({', '.join(escalating)}), required >= 6.")
lines.append(f"- Scripts by group: standard={group_tot['standard']}, "
             f"silent_assumption={group_tot['silent_assumption']}.")
lines.append("")
lines.append("## The 10 noticed-a-silent-assumption scripts")
lines.append("")
lines.append("These are the exact behaviour the project has no data for: the model "
             "silently assumed something, the writer notices, and the first steer is the "
             "correction. Each row names what the script tests.")
lines.append("")
lines.append("| id | band | pressure | what the model assumed / what is tested |")
lines.append("|---|---|---|---|")
for sc in assumption:
    lines.append(f"| {sc['id']} | {sc['depth_band']} | {sc['response_pressure']} | "
                 f"{sc['tests']} |")
lines.append("")

(OUT / "steer-scripts.md").write_text("\n".join(lines) + "\n")

print("OK")
print("scripts:", len(scripts), "steers:", total_steers)
print("terse:", terse_total, "escalating:", len(escalating))
print("kinds:", dict(kind_tot))
print("pressures:", dict(pressure_tot))
