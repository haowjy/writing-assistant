#!/usr/bin/env python3
"""Build the proposed vocabulary extension by appending new atoms to the
existing variation-catalog-v1.json. Existing items are preserved exactly;
new items are appended in list order. Fails loudly on any duplicate.

This is a PROPOSAL artifact. It does not modify data/.
"""

import json
from pathlib import Path

ROOT = Path("/home/jimyao/gitrepos/research/writing-assistant")
SRC = ROOT / "data/training/variation-catalog-v1.json"
OUT = Path(
    "/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/vocabulary-extension.json"
)

NEW_GENRES = [
    "police procedural",
    "medical drama",
    "campus novel",
    "epistolary fiction",
    "sports fiction",
    "military science fiction",
    "dystopian fiction",
    "weird fiction",
    "picaresque",
    "war fiction",
    "Nordic saga",
    "Russian psychological realism",
    "French naturalism",
    "Japanese I-novel (shishōsetsu)",
    "Chinese wuxia",
    "Latin American boom novel",
    "West African oral epic",
    "Italian neorealism",
    "German Bildungsroman",
    "Japanese light novel",
]

NEW_STYLES = [
    "brisk and plot-forward",
    "workmanlike and transparent",
    "plain declarative",
    "procedural and step-by-step",
    "crisp newsroom clarity",
    "terse hard-boiled",
    "warm and reassuring",
    "chatty first-person",
    "briskly comic",
    "clinical and detached",
    "urgent present-tense",
    "pulpy and propulsive",
    "reader-friendly and formula-aware",
    "sensory but unfussy",
    "precise scientific register",
    "measured expository clarity",
]

NEW_TROPES = [
    "a secret kept to protect someone else's reputation",
    "a mentor's advice that misfires",
    "a favor that must be repaid at the worst time",
    "a shared joke that becomes a private language",
    "a rival who is right about the facts",
    "a parent living through a child's success",
    "a sibling who kept the family business afloat",
    "a marriage negotiated like a contract",
    "an office alliance that outlives its purpose",
    "a manager promoted past their competence",
    "a well-meaning lie told to spare a friend",
    "a whistleblower with mixed motives",
    "a patient who knows more than the doctor",
    "a customer who becomes a confidant",
    "a committee that cannot decide",
    "a town's unofficial historian",
    "a retirement nobody has planned for",
    "an inheritance that splits siblings",
    "a neighborhood watch with divided loyalties",
    "a colleague who takes credit by accident",
    "a contractor who finds the previous worker's shortcut",
    "a shift worker covering for an absent colleague",
    "a teacher's pet who resents the role",
    "a coach who benches the star",
    "a nurse who breaks a small rule for a patient",
    "a chef whose signature dish is not their own",
    "a landlord and tenant who need each other",
    "an estranged cousin who arrives with a claim",
    "a wedding guest with an uninvited truth",
    "a funeral that reveals a hidden debt",
    "a charity gala with a hidden agenda",
    "a local election decided by one vote",
    "a union rep caught between members and management",
    "a neighbor who hears everything through the wall",
    "a best friend tired of being the sidekick",
    "a couple who cannot agree on a name",
    "a family recipe treated as a trade secret",
    "a group project where one person does the work",
    "a public apology that makes things worse",
    "a regular customer who notices a change",
]

NEW_SITUATIONS = [
    "a price sign is wrong and the stallholder must choose which price to honor",
    "two customers claim the last crate before the market opens",
    "the night market loses power during its busiest hour",
    "a courier must decide whether to open an undeliverable package",
    "a shop owner recognizes a returned item as stolen",
    "a factory line halts because one worker refuses to sign off",
    "a foreman must choose between a deadline and an unlogged safety check",
    "a construction crew uncovers an unmarked pipe",
    "a night shift finds the previous shift's log incomplete",
    "a mine elevator stalls with visitors still below",
    "a warehouse inventory counts one pallet too many",
    "a triage nurse must assign the last free bed",
    "a doctor's order contradicts a patient's stated wish",
    "an ambulance arrives with no record of who called",
    "a cleaner finds a discarded chart in the stairwell",
    "a clinic runs out of a routine medicine during a rush",
    "a patient's relatives disagree about how much to tell them",
    "a witness is asked a question nobody anticipated",
    "a jury is told to ignore testimony it has already heard",
    "a public defender meets a client minutes before the hearing",
    "a clerk finds a filing error that could void a decision",
    "a council vote ties and the chair must break it",
    "a permit is revoked while the work is half done",
    "a ferry manifest lists one passenger too many",
    "a cargo log disagrees with the port records",
    "a train is held at a signal with no explanation",
    "a pilot must choose between the schedule and a passenger's need",
    "a border crossing closes while a line of travelers waits",
    "a driver is offered a load with vague papers",
    "irrigation from one farm floods the neighbor's field",
    "a harvest crew finds a crop that is not theirs",
    "a kitchen runs out of an ingredient mid-service",
    "a restaurant critic arrives unannounced on the busiest night",
    "canning-line labels are switched before shipping",
    "a sample is mislabeled on the last day of a study",
    "a result cannot be reproduced before a deadline",
    "a code deploy fails minutes before a demo",
    "a server shows an unauthorized access nobody reported",
    "a student's data does not match the lab notebook",
    "a prison visit is denied for a reason not stated",
    "a guard must decide whether to report small contraband",
    "a stadium announcer is handed a last-minute substitution",
    "a festival stage loses power mid-performance",
    "a teacher finds a note that names a student",
    "a substitute is handed a lesson plan for the wrong class",
    "a well runs dry while a caravan waits",
    "a wildfire forces a choice about which animals to release",
    "a hiker's only map is contradicted by a marked trail",
]

NEW_CONTINUITY = [
    "delay a revelation until its physical evidence exists on the page",
    "let a character learn a fact only after acting on the wrong one",
    "plant a detail that must stay dormant until its later payoff",
    "pay off a consequence established earlier in the same scene",
    "reveal information to the reader before the POV character can act on it",
    "withhold a name until every speaker has used it wrongly at least once",
    "stage a discovery at the moment it changes the character's next choice",
    "return an earlier promise as an obligation now due",
    "let an earlier lie constrain what a character can say now",
    "reveal the cost of a choice before its benefit",
    "keep a secondary character ignorant of a fact the reader already knows",
    "introduce a detail that must not matter until a later chapter",
    "force a choice between revealing a secret and protecting a relationship",
    "let an object's history surface only when it is handled again",
    "withhold the reason for an action until its consequence is visible",
    "make a character notice a change they cannot yet interpret",
    "delay an explanation until the person who needs it has stopped asking",
    "let a consequence arrive from a decision made before the scene begins",
]

NEW_COMPOSITION_RULES = [
    "When a brief leaves a material choice open, the task must ask the choice or state the assumed default explicitly; it must never resolve the ambiguity silently.",
    "When a length budget is specified, treat it as a hard ceiling and verify the draft against it before submission.",
    "Across a batch, vary the opening construction, dominant image, and physical setting; repeated stems or clustered settings count as a coverage failure.",
    "A sampled trope must press on a concrete decision in the scene rather than decorate the setting.",
    "A sampled continuity challenge that creates positive pressure must name the detail, the moment of disclosure or payoff, and the consequence of mistiming it.",
    "In a genre blend, each contributing genre's conventions must remain load-bearing in the scene; a blend that only relabels a single genre is not admissible.",
]


def check_unique(existing, new, key):
    overlap = sorted(set(existing) & set(new))
    if overlap:
        raise SystemExit(f"Duplicate {key}: {overlap}")
    if len(new) != len(set(new)):
        dupes = sorted({x for x in new if new.count(x) > 1})
        raise SystemExit(f"Internal duplicate {key}: {dupes}")


def main():
    catalog = json.loads(SRC.read_text())
    for key, new in [
        ("genres", NEW_GENRES),
        ("styles", NEW_STYLES),
        ("tropes", NEW_TROPES),
        ("situations", NEW_SITUATIONS),
        ("continuity_challenges", NEW_CONTINUITY),
    ]:
        check_unique(catalog[key], new, key)
        catalog[key] = catalog[key] + new

    check_unique(catalog["composition_rules"], NEW_COMPOSITION_RULES, "composition_rules")
    catalog["composition_rules"] = catalog["composition_rules"] + NEW_COMPOSITION_RULES

    # Preserve the exact top-level key order of the source file plus no extras.
    ordered = {
        "schema_version": catalog["schema_version"],
        "description": catalog["description"],
        "genres": catalog["genres"],
        "styles": catalog["styles"],
        "tropes": catalog["tropes"],
        "situations": catalog["situations"],
        "continuity_challenges": catalog["continuity_challenges"],
        "book_candidates": catalog["book_candidates"],
        "composition_rules": catalog["composition_rules"],
    }
    OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n")

    print("wrote", OUT)
    for key in [
        "genres",
        "styles",
        "tropes",
        "situations",
        "continuity_challenges",
        "book_candidates",
        "composition_rules",
    ]:
        print(f"  {key}: {len(ordered[key])}")


if __name__ == "__main__":
    main()
