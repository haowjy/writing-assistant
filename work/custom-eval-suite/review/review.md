# Scenario review

All labels are model-authored drafts. Review importance, interpretation, task
feasibility, scoring scope, and source cutoff before accepting a case.

Edit scenario-review.json with reviewer identity, status, and corrections.
Judgment review retains scores, evidence, explanations, and uncertainty.

## F1-01 — F1

Genre: literary_fiction

Instruction specificity: explicit

Mara has been considering leaving a tidal archive. Ilan has noticed the distance between them, but they have not talked about what leaving would mean. Their everyday work carries the weight of that unspoken choice.

Mara keeps the harbor archive. Ilan repairs its flooded shelves. At noon Mara accuses Ilan of selling a brass ledger, but she has no evidence. Ilan says he moved it above the tide line. Mara cannot open the upper room because only the retired keeper has its key. A red scarf hangs on a chair; nobody refers to it again. The chapter ends before anyone enters the upper room.

Write 120–220 words about Mara and Ilan in a tidal archive, in restrained, concrete prose. Use close third person centered on Mara. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Stage a disagreement over the object. Return only the prose, without a title or commentary.

This is for a literary fiction story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve Only the retired keeper has the upper-room key. Keep Mara suspects Ilan sold the ledger; this is an unproven accusation. Do not assert: The ledger is destroyed."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Mara; restrained, concrete prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested literary fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-01 — F2

Genre: fantasy

Instruction specificity: explicit

Read notes/source.md and drafts/scene.md. Expand only the first sentence into 120–220 words in restrained, concrete prose. Keep the final sentence exactly unchanged. Put only the replacement prose between <prose> and </prose> in the file, then retain the final sentence after </prose>. Use close third person on Mara. Preserve facts and unresolved beliefs. Reply with a short completion note; do not repeat the scene.

This is for a fantasy story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "preserve-ending",
      "metric": "Q1",
      "kind": "protected",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md",
      "text": "The harbor bell rang twice."
    },
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve Only the retired keeper has the upper-room key. Keep Mara suspects Ilan sold the ledger; this is an unproven accusation. Do not assert: The ledger is destroyed."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Mara; restrained, concrete prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested fantasy genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-01 — F3

Genre: romance

Instruction specificity: explicit

Mara has received a letter from an old love who wants to meet again. Ilan knows about the invitation but has not read the letter. A possible reunion complicates Mara's feelings about remaining at a tidal archive; no decision has been made.

Mara keeps the harbor archive. Ilan repairs its flooded shelves. At noon Mara accuses Ilan of selling a brass ledger, but she has no evidence. Ilan says he moved it above the tide line. Mara cannot open the upper room because only the retired keeper has its key. A red scarf hangs on a chair; nobody refers to it again. The chapter ends before anyone enters the upper room.

Give three numbered, meaningfully different next-scene directions. For each, explain the dramatic choice, its consequence, and why it fits known facts. Do not draft the scene or resolve the central uncertainty.

This is for a romance story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "three-directions",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Exactly three developed viable directions."
    }
  ],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Among all three requested pairs, count pairs whose two alternatives are viable and meaningfully different in dramatic choice or consequence. Missing alternatives earn no diverse pairs. Return diverse viable pairs / 3.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-01 — F4

Genre: historical_fiction

Instruction specificity: explicit

Read source/chapter.md. Create a navigable Markdown continuity wiki under kb/, starting at kb/index.md. Its audience is a writer planning the next chapter. Prioritize consequential facts, uncertainty, and character knowledge. At least one linked topic page is needed. Omit incidental details where appropriate; cite the source path. Keep source files unchanged. Use no more than 500 words across KB pages.

This is for a historical fiction story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "wiki-words",
      "metric": "Q1",
      "kind": "kb_word_budget",
      "method": "deterministic",
      "required": false,
      "max": 500
    },
    {
      "id": "wiki-links",
      "metric": "Q1",
      "kind": "wiki_links",
      "method": "deterministic",
      "required": true
    },
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "Only the retired keeper has the upper-room key."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "Mara suspects Ilan sold the ledger; this is an unproven accusation."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "Ilan repairs the flooded shelves."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "Only the retired keeper has the upper-room key.",
      "importance": "required",
      "weight": 3,
      "evidence": "The year is 1936. News reaches a tidal archive through letters, newspapers, and radio broadcasts. A regional strike has delayed ordinary deliveries, and Mara and Ilan are working with the supplies already on hand. Any unusual phenomena in the supplied premise remain part of this story's world.\n\nMara keeps the harbor archive. Ilan repairs its flooded shelves. At noon Mara accuses Ilan of selling a brass ledger, but she has no evidence. Ilan says he moved it above the tide line. Mara cannot open the upper room because only the retired keeper has its key. A red scarf hangs on a chair; nobody refers to it again. The chapter ends before anyone enters the upper room.",
      "after_update": null
    },
    {
      "id": "belief",
      "statement": "Mara suspects Ilan sold the ledger; this is an unproven accusation.",
      "importance": "required",
      "weight": 3,
      "evidence": "The year is 1936. News reaches a tidal archive through letters, newspapers, and radio broadcasts. A regional strike has delayed ordinary deliveries, and Mara and Ilan are working with the supplies already on hand. Any unusual phenomena in the supplied premise remain part of this story's world.\n\nMara keeps the harbor archive. Ilan repairs its flooded shelves. At noon Mara accuses Ilan of selling a brass ledger, but she has no evidence. Ilan says he moved it above the tide line. Mara cannot open the upper room because only the retired keeper has its key. A red scarf hangs on a chair; nobody refers to it again. The chapter ends before anyone enters the upper room.",
      "placement": "Any findable page; keep attribution and uncertainty."
    },
    {
      "id": "context",
      "statement": "Ilan repairs the flooded shelves.",
      "importance": "optional",
      "weight": 1,
      "evidence": "The year is 1936. News reaches a tidal archive through letters, newspapers, and radio broadcasts. A regional strike has delayed ordinary deliveries, and Mara and Ilan are working with the supplies already on hand. Any unusual phenomena in the supplied premise remain part of this story's world.\n\nMara keeps the harbor archive. Ilan repairs its flooded shelves. At noon Mara accuses Ilan of selling a brass ledger, but she has no evidence. Ilan says he moved it above the tide line. Mara cannot open the upper room because only the retired keeper has its key. A red scarf hangs on a chair; nobody refers to it again. The chapter ends before anyone enters the upper room."
    },
    {
      "id": "incidental",
      "statement": "The scarf is red.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "The year is 1936. News reaches a tidal archive through letters, newspapers, and radio broadcasts. A regional strike has delayed ordinary deliveries, and Mara and Ilan are working with the supplies already on hand. Any unusual phenomena in the supplied premise remain part of this story's world.\n\nMara keeps the harbor archive. Ilan repairs its flooded shelves. At noon Mara accuses Ilan of selling a brass ledger, but she has no evidence. Ilan says he moved it above the tide line. Mara cannot open the upper room because only the retired keeper has its key. A red scarf hangs on a chair; nobody refers to it again. The chapter ends before anyone enters the upper room.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "Only the retired keeper has the upper-room key."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Mara suspects Ilan sold the ledger; this is an unproven accusation."
    }
  ]
}
```

## F5-01 — F5

Genre: adventure

Instruction specificity: explicit

Use the continuity wiki starting at kb/index.md. Write 120–220 words about Mara and Ilan in a tidal archive, in restrained, concrete prose. Use close third person centered on Mara. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Save prose in drafts/scene.md; keep the KB unchanged.

This is for a adventure story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "retrieve-state",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "Only the retired keeper has the upper-room key."
    },
    {
      "id": "retrieve-belief",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "Mara suspects Ilan sold the ledger; this is an unproven accusation."
    },
    {
      "id": "preserve-kb",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve Only the retired keeper has the upper-room key. Keep Mara suspects Ilan sold the ledger; this is an unproven accusation. Do not assert: The ledger is destroyed."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Mara; restrained, concrete prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested adventure genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-02 — F1

Genre: mystery

Instruction specificity: explicit

An anonymous note has arrived at a glass orchard, referring to an earlier disappearance. Suri and Aven have different recollections of that day. Neither the note's author nor its connection to the present problem has been established.

Suri tends glass trees that grow only under recorded birdsong. Aven maintains the recorder and cannot hear high notes after an accident. When one tree cracks, Suri wonders whether Aven damaged its song. Aven has not confessed and no test has been run. The original recording is sealed in the municipal vault. Three empty cups sit beside the workbench. The scene ends with Suri asking permission to inspect the vault.

Write 120–220 words about Suri and Aven in a glass orchard, in spare prose. Use close third person centered on Suri. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Continue immediately after the supplied passage. Return only the prose, without a title or commentary.

This is for a mystery story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The original recording is sealed in the municipal vault. Keep Suri suspects a damaged song; Aven's responsibility is unknown. Do not assert: Aven deliberately broke the tree."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Suri; spare prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested mystery genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-02 — F2

Genre: science_fiction

Instruction specificity: explicit

Read notes/source.md. Write 120–220 words about Suri and Aven in a glass orchard, in spare prose. Use close third person centered on Suri. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Save only prose in drafts/scene.md. Reply briefly afterward.

This is for a science fiction story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The original recording is sealed in the municipal vault. Keep Suri suspects a damaged song; Aven's responsibility is unknown. Do not assert: Aven deliberately broke the tree."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Suri; spare prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested science fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-02 — F3

Genre: horror

Instruction specificity: explicit

For three nights, people near a glass orchard have heard their own voices answering from empty rooms. Suri heard it once and has told Aven, who has not witnessed it. No cause has been established, and nobody agrees on whether entering alone is safe.

Suri tends glass trees that grow only under recorded birdsong. Aven maintains the recorder and cannot hear high notes after an accident. When one tree cracks, Suri wonders whether Aven damaged its song. Aven has not confessed and no test has been run. The original recording is sealed in the municipal vault. Three empty cups sit beside the workbench. The scene ends with Suri asking permission to inspect the vault.

Give three numbered, meaningfully different next-scene directions. For each, explain the dramatic choice, its consequence, and why it fits known facts. Do not draft the scene or resolve the central uncertainty.

This is for a horror story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "three-directions",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Exactly three developed viable directions."
    }
  ],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Among all three requested pairs, count pairs whose two alternatives are viable and meaningfully different in dramatic choice or consequence. Missing alternatives earn no diverse pairs. Return diverse viable pairs / 3.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-02 — F4

Genre: comedy

Instruction specificity: explicit

Read source/chapter.md. Create a navigable Markdown continuity wiki under kb/, starting at kb/index.md. Its audience is a writer planning the next chapter. Prioritize consequential facts, uncertainty, and character knowledge. At least one linked topic page is needed. Omit incidental details where appropriate; cite the source path. Keep source files unchanged. Use no more than 500 words across KB pages.

This is for a comedy story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "wiki-words",
      "metric": "Q1",
      "kind": "kb_word_budget",
      "method": "deterministic",
      "required": false,
      "max": 500
    },
    {
      "id": "wiki-links",
      "metric": "Q1",
      "kind": "wiki_links",
      "method": "deterministic",
      "required": true
    },
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "Suri may inspect the vault tomorrow; the cause of the crack remains unknown."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "Suri suspects a damaged song; Aven's responsibility is unknown."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "Aven cannot hear high notes after an accident."
    },
    {
      "id": "accepted-update",
      "metric": "Q11",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Suri may inspect the vault tomorrow; the cause of the crack remains unknown. Preserve relevant history and do not adopt the draft-only suggestion."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "The original recording is sealed in the municipal vault.",
      "importance": "required",
      "weight": 3,
      "evidence": "A visiting committee has mistaken Suri for the person in charge of a glass orchard. Aven knows the mistake, but an elaborate welcome has already begun. Their attempt to get through the visit without embarrassing anyone keeps producing smaller misunderstandings.\n\nSuri tends glass trees that grow only under recorded birdsong. Aven maintains the recorder and cannot hear high notes after an accident. When one tree cracks, Suri wonders whether Aven damaged its song. Aven has not confessed and no test has been run. The original recording is sealed in the municipal vault. Three empty cups sit beside the workbench. The scene ends with Suri asking permission to inspect the vault.",
      "after_update": "Suri may inspect the vault tomorrow; the cause of the crack remains unknown."
    },
    {
      "id": "belief",
      "statement": "Suri suspects a damaged song; Aven's responsibility is unknown.",
      "importance": "required",
      "weight": 3,
      "evidence": "A visiting committee has mistaken Suri for the person in charge of a glass orchard. Aven knows the mistake, but an elaborate welcome has already begun. Their attempt to get through the visit without embarrassing anyone keeps producing smaller misunderstandings.\n\nSuri tends glass trees that grow only under recorded birdsong. Aven maintains the recorder and cannot hear high notes after an accident. When one tree cracks, Suri wonders whether Aven damaged its song. Aven has not confessed and no test has been run. The original recording is sealed in the municipal vault. Three empty cups sit beside the workbench. The scene ends with Suri asking permission to inspect the vault.",
      "placement": "Any findable page; keep attribution and uncertainty."
    },
    {
      "id": "context",
      "statement": "Aven cannot hear high notes after an accident.",
      "importance": "optional",
      "weight": 1,
      "evidence": "A visiting committee has mistaken Suri for the person in charge of a glass orchard. Aven knows the mistake, but an elaborate welcome has already begun. Their attempt to get through the visit without embarrassing anyone keeps producing smaller misunderstandings.\n\nSuri tends glass trees that grow only under recorded birdsong. Aven maintains the recorder and cannot hear high notes after an accident. When one tree cracks, Suri wonders whether Aven damaged its song. Aven has not confessed and no test has been run. The original recording is sealed in the municipal vault. Three empty cups sit beside the workbench. The scene ends with Suri asking permission to inspect the vault."
    },
    {
      "id": "incidental",
      "statement": "There are three empty cups.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "A visiting committee has mistaken Suri for the person in charge of a glass orchard. Aven knows the mistake, but an elaborate welcome has already begun. Their attempt to get through the visit without embarrassing anyone keeps producing smaller misunderstandings.\n\nSuri tends glass trees that grow only under recorded birdsong. Aven maintains the recorder and cannot hear high notes after an accident. When one tree cracks, Suri wonders whether Aven damaged its song. Aven has not confessed and no test has been run. The original recording is sealed in the municipal vault. Three empty cups sit beside the workbench. The scene ends with Suri asking permission to inspect the vault.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "Suri may inspect the vault tomorrow; the cause of the crack remains unknown."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Suri suspects a damaged song; Aven's responsibility is unknown."
    }
  ]
}
```

## F5-02 — F5

Genre: thriller

Instruction specificity: explicit

The original recording is sealed in the municipal vault.

Aven cannot hear high notes after an accident.

Suri suspects a damaged song; Aven's responsibility is unknown.

Someone has warned Suri that a stranger will arrive at a glass orchard before morning asking for information. Aven has seen signs that someone has been watching the entrance. The warning's source and the stranger's intentions remain unknown.

Write 120–220 words about Suri and Aven in a glass orchard, in spare prose. Use close third person centered on Suri. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Return a short note, then prose inside <prose></prose>.

This is for a thriller story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The original recording is sealed in the municipal vault. Keep Suri suspects a damaged song; Aven's responsibility is unknown. Do not assert: Aven deliberately broke the tree."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Suri; spare prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested thriller genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-03 — F1

Genre: fantasy

Instruction specificity: explicit

In this region, places can remember promises spoken inside them. a night ferry has begun repeating a fragment of an old vow after dark. Oren and Leda do not know whose promise it was or whether it concerns them.

Oren is a ferry clerk. Leda boards with an unsigned ticket and asks him not to announce her destination. Oren thinks she is fleeing a creditor, but she has not told him why she is traveling. The ferry will stop at Bell Quay before continuing to the island. Only the captain knows the revised sailing time. A chipped blue plate rolls under a bench. Leda waits by the gangway while Oren decides whether to call the captain.

Write 120–220 words about Oren and Leda in a night ferry, in measured pacing with sustained tension. Use close third person centered on Oren. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Let most of the scene unfold through dialogue. Return only the prose, without a title or commentary.

This is for a fantasy story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The captain alone knows the revised sailing time. Keep Oren's creditor explanation is a guess, not Leda's stated motive. Do not assert: Leda admits she is fleeing a creditor."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Oren; measured pacing with sustained tension."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested fantasy genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-03 — F2

Genre: romance

Instruction specificity: explicit

Read notes/source.md and drafts/scene.md. Expand only the first sentence into 120–220 words in measured pacing with sustained tension. Keep the final sentence exactly unchanged. Put only the replacement prose between <prose> and </prose> in the file, then retain the final sentence after </prose>. Use close third person on Oren. Preserve facts and unresolved beliefs. Reply with a short completion note; do not repeat the scene.

This is for a romance story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "preserve-ending",
      "metric": "Q1",
      "kind": "protected",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md",
      "text": "The rope scraped against the rail."
    },
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The captain alone knows the revised sailing time. Keep Oren's creditor explanation is a guess, not Leda's stated motive. Do not assert: Leda admits she is fleeing a creditor."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Oren; measured pacing with sustained tension."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested romance genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-03 — F3

Genre: historical_fiction

Instruction specificity: explicit

The year is 1936. News reaches a night ferry through letters, newspapers, and radio broadcasts. A regional strike has delayed ordinary deliveries, and Oren and Leda are working with the supplies already on hand. Any unusual phenomena in the supplied premise remain part of this story's world.

Oren is a ferry clerk. Leda boards with an unsigned ticket and asks him not to announce her destination. Oren thinks she is fleeing a creditor, but she has not told him why she is traveling. The ferry will stop at Bell Quay before continuing to the island. Only the captain knows the revised sailing time. A chipped blue plate rolls under a bench. Leda waits by the gangway while Oren decides whether to call the captain.

Give three numbered, meaningfully different next-scene directions. For each, explain the dramatic choice, its consequence, and why it fits known facts. Do not draft the scene or resolve the central uncertainty.

This is for a historical fiction story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "three-directions",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Exactly three developed viable directions."
    }
  ],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Among all three requested pairs, count pairs whose two alternatives are viable and meaningfully different in dramatic choice or consequence. Missing alternatives earn no diverse pairs. Return diverse viable pairs / 3.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-03 — F4

Genre: adventure

Instruction specificity: explicit

Read source/chapter.md. Create a navigable Markdown continuity wiki under kb/, starting at kb/index.md. Its audience is a writer planning the next chapter. Prioritize consequential facts, uncertainty, and character knowledge. At least one linked topic page is needed. Omit incidental details where appropriate; cite the source path. Keep source files unchanged. Use no more than 500 words across KB pages.

This is for a adventure story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "wiki-words",
      "metric": "Q1",
      "kind": "kb_word_budget",
      "method": "deterministic",
      "required": false,
      "max": 500
    },
    {
      "id": "wiki-links",
      "metric": "Q1",
      "kind": "wiki_links",
      "method": "deterministic",
      "required": true
    },
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "The captain alone knows the revised sailing time."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "Oren's creditor explanation is a guess, not Leda's stated motive."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "Bell Quay is the first stop."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "The captain alone knows the revised sailing time.",
      "importance": "required",
      "weight": 3,
      "evidence": "A route beyond a night ferry is opening for the first time in years. Oren and Leda have been invited to join an expedition, but preparations are incomplete. The present problem must be considered alongside the chance to leave; they have not departed.\n\nOren is a ferry clerk. Leda boards with an unsigned ticket and asks him not to announce her destination. Oren thinks she is fleeing a creditor, but she has not told him why she is traveling. The ferry will stop at Bell Quay before continuing to the island. Only the captain knows the revised sailing time. A chipped blue plate rolls under a bench. Leda waits by the gangway while Oren decides whether to call the captain.",
      "after_update": null
    },
    {
      "id": "belief",
      "statement": "Oren's creditor explanation is a guess, not Leda's stated motive.",
      "importance": "required",
      "weight": 3,
      "evidence": "A route beyond a night ferry is opening for the first time in years. Oren and Leda have been invited to join an expedition, but preparations are incomplete. The present problem must be considered alongside the chance to leave; they have not departed.\n\nOren is a ferry clerk. Leda boards with an unsigned ticket and asks him not to announce her destination. Oren thinks she is fleeing a creditor, but she has not told him why she is traveling. The ferry will stop at Bell Quay before continuing to the island. Only the captain knows the revised sailing time. A chipped blue plate rolls under a bench. Leda waits by the gangway while Oren decides whether to call the captain.",
      "placement": "Any findable page; keep attribution and uncertainty."
    },
    {
      "id": "context",
      "statement": "Bell Quay is the first stop.",
      "importance": "optional",
      "weight": 1,
      "evidence": "A route beyond a night ferry is opening for the first time in years. Oren and Leda have been invited to join an expedition, but preparations are incomplete. The present problem must be considered alongside the chance to leave; they have not departed.\n\nOren is a ferry clerk. Leda boards with an unsigned ticket and asks him not to announce her destination. Oren thinks she is fleeing a creditor, but she has not told him why she is traveling. The ferry will stop at Bell Quay before continuing to the island. Only the captain knows the revised sailing time. A chipped blue plate rolls under a bench. Leda waits by the gangway while Oren decides whether to call the captain."
    },
    {
      "id": "incidental",
      "statement": "The plate is chipped and blue.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "A route beyond a night ferry is opening for the first time in years. Oren and Leda have been invited to join an expedition, but preparations are incomplete. The present problem must be considered alongside the chance to leave; they have not departed.\n\nOren is a ferry clerk. Leda boards with an unsigned ticket and asks him not to announce her destination. Oren thinks she is fleeing a creditor, but she has not told him why she is traveling. The ferry will stop at Bell Quay before continuing to the island. Only the captain knows the revised sailing time. A chipped blue plate rolls under a bench. Leda waits by the gangway while Oren decides whether to call the captain.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "The captain alone knows the revised sailing time."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Oren's creditor explanation is a guess, not Leda's stated motive."
    }
  ]
}
```

## F5-03 — F5

Genre: literary_fiction

Instruction specificity: explicit

Use the continuity wiki starting at kb/index.md. Write 120–220 words about Oren and Leda in a night ferry, in measured pacing with sustained tension. Use close third person centered on Oren. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Save prose in drafts/scene.md; keep the KB unchanged.

This is for a literary fiction story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "retrieve-state",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "The captain alone knows the revised sailing time."
    },
    {
      "id": "retrieve-belief",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "Oren's creditor explanation is a guess, not Leda's stated motive."
    },
    {
      "id": "preserve-kb",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The captain alone knows the revised sailing time. Keep Oren's creditor explanation is a guess, not Leda's stated motive. Do not assert: Leda admits she is fleeing a creditor."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Oren; measured pacing with sustained tension."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested literary fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-04 — F1

Genre: science_fiction

Instruction specificity: explicit

an isolated radio station is part of a settlement on an artificial island maintained by autonomous systems. A recent system update has changed the way public records are synchronized. Nessa and Tomas have not established whether this affects the events described below.

Nessa runs a mountain radio station with her brother Tomas. A rescue signal arrives on a frequency missing from their log. Tomas says their mother once used that channel, but he is remembering a childhood visit and may be mistaken. Nessa has never heard the voice before. The antenna cannot be turned until the ice thaws. There are five pencils in a mug. They agree to record the next transmission rather than answer it immediately.

Write 120–220 words about Nessa and Tomas in an isolated radio station, in dialogue-led prose with understated tension. Use close third person centered on Nessa. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Center a private hesitation before an outward decision. Return only the prose, without a title or commentary.

This is for a science fiction story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The antenna cannot be turned until the ice thaws. Keep Tomas's memory of their mother's channel is uncertain. Do not assert: Nessa recognizes her mother's voice."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Nessa; dialogue-led prose with understated tension."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested science fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-04 — F2

Genre: horror

Instruction specificity: explicit

Read notes/source.md. Write 120–220 words about Nessa and Tomas in an isolated radio station, in dialogue-led prose with understated tension. Use close third person centered on Nessa. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Save only prose in drafts/scene.md. Reply briefly afterward.

This is for a horror story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The antenna cannot be turned until the ice thaws. Keep Tomas's memory of their mother's channel is uncertain. Do not assert: Nessa recognizes her mother's voice."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Nessa; dialogue-led prose with understated tension."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested horror genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-04 — F3

Genre: comedy

Instruction specificity: explicit

A visiting committee has mistaken Nessa for the person in charge of an isolated radio station. Tomas knows the mistake, but an elaborate welcome has already begun. Their attempt to get through the visit without embarrassing anyone keeps producing smaller misunderstandings.

Nessa runs a mountain radio station with her brother Tomas. A rescue signal arrives on a frequency missing from their log. Tomas says their mother once used that channel, but he is remembering a childhood visit and may be mistaken. Nessa has never heard the voice before. The antenna cannot be turned until the ice thaws. There are five pencils in a mug. They agree to record the next transmission rather than answer it immediately.

Give three numbered, meaningfully different next-scene directions. For each, explain the dramatic choice, its consequence, and why it fits known facts. Do not draft the scene or resolve the central uncertainty.

This is for a comedy story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "three-directions",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Exactly three developed viable directions."
    }
  ],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Among all three requested pairs, count pairs whose two alternatives are viable and meaningfully different in dramatic choice or consequence. Missing alternatives earn no diverse pairs. Return diverse viable pairs / 3.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-04 — F4

Genre: thriller

Instruction specificity: explicit

Read source/chapter.md. Create a navigable Markdown continuity wiki under kb/, starting at kb/index.md. Its audience is a writer planning the next chapter. Prioritize consequential facts, uncertainty, and character knowledge. At least one linked topic page is needed. Omit incidental details where appropriate; cite the source path. Keep source files unchanged. Use no more than 500 words across KB pages.

This is for a thriller story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "wiki-words",
      "metric": "Q1",
      "kind": "kb_word_budget",
      "method": "deterministic",
      "required": false,
      "max": 500
    },
    {
      "id": "wiki-links",
      "metric": "Q1",
      "kind": "wiki_links",
      "method": "deterministic",
      "required": true
    },
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "The antenna points east; the voice is still unidentified."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "Tomas's memory of their mother's channel is uncertain."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "They agreed to record the next transmission before replying."
    },
    {
      "id": "accepted-update",
      "metric": "Q11",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "The antenna points east; the voice is still unidentified. Preserve relevant history and do not adopt the draft-only suggestion."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "The antenna cannot be turned until the ice thaws.",
      "importance": "required",
      "weight": 3,
      "evidence": "Someone has warned Nessa that a stranger will arrive at an isolated radio station before morning asking for information. Tomas has seen signs that someone has been watching the entrance. The warning's source and the stranger's intentions remain unknown.\n\nNessa runs a mountain radio station with her brother Tomas. A rescue signal arrives on a frequency missing from their log. Tomas says their mother once used that channel, but he is remembering a childhood visit and may be mistaken. Nessa has never heard the voice before. The antenna cannot be turned until the ice thaws. There are five pencils in a mug. They agree to record the next transmission rather than answer it immediately.",
      "after_update": "The antenna points east; the voice is still unidentified."
    },
    {
      "id": "belief",
      "statement": "Tomas's memory of their mother's channel is uncertain.",
      "importance": "required",
      "weight": 3,
      "evidence": "Someone has warned Nessa that a stranger will arrive at an isolated radio station before morning asking for information. Tomas has seen signs that someone has been watching the entrance. The warning's source and the stranger's intentions remain unknown.\n\nNessa runs a mountain radio station with her brother Tomas. A rescue signal arrives on a frequency missing from their log. Tomas says their mother once used that channel, but he is remembering a childhood visit and may be mistaken. Nessa has never heard the voice before. The antenna cannot be turned until the ice thaws. There are five pencils in a mug. They agree to record the next transmission rather than answer it immediately.",
      "placement": "Any findable page; keep attribution and uncertainty."
    },
    {
      "id": "context",
      "statement": "They agreed to record the next transmission before replying.",
      "importance": "optional",
      "weight": 1,
      "evidence": "Someone has warned Nessa that a stranger will arrive at an isolated radio station before morning asking for information. Tomas has seen signs that someone has been watching the entrance. The warning's source and the stranger's intentions remain unknown.\n\nNessa runs a mountain radio station with her brother Tomas. A rescue signal arrives on a frequency missing from their log. Tomas says their mother once used that channel, but he is remembering a childhood visit and may be mistaken. Nessa has never heard the voice before. The antenna cannot be turned until the ice thaws. There are five pencils in a mug. They agree to record the next transmission rather than answer it immediately."
    },
    {
      "id": "incidental",
      "statement": "Five pencils are in the mug.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "Someone has warned Nessa that a stranger will arrive at an isolated radio station before morning asking for information. Tomas has seen signs that someone has been watching the entrance. The warning's source and the stranger's intentions remain unknown.\n\nNessa runs a mountain radio station with her brother Tomas. A rescue signal arrives on a frequency missing from their log. Tomas says their mother once used that channel, but he is remembering a childhood visit and may be mistaken. Nessa has never heard the voice before. The antenna cannot be turned until the ice thaws. There are five pencils in a mug. They agree to record the next transmission rather than answer it immediately.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "The antenna points east; the voice is still unidentified."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Tomas's memory of their mother's channel is uncertain."
    }
  ]
}
```

## F5-04 — F5

Genre: mystery

Instruction specificity: explicit

The antenna cannot be turned until the ice thaws.

They agreed to record the next transmission before replying.

Tomas's memory of their mother's channel is uncertain.

An anonymous note has arrived at an isolated radio station, referring to an earlier disappearance. Nessa and Tomas have different recollections of that day. Neither the note's author nor its connection to the present problem has been established.

Write 120–220 words about Nessa and Tomas in an isolated radio station, in dialogue-led prose with understated tension. Use close third person centered on Nessa. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Return a short note, then prose inside <prose></prose>.

This is for a mystery story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The antenna cannot be turned until the ice thaws. Keep Tomas's memory of their mother's channel is uncertain. Do not assert: Nessa recognizes her mother's voice."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Nessa; dialogue-led prose with understated tension."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested mystery genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-05 — F1

Genre: romance

Instruction specificity: explicit

Dara has received a letter from an old love who wants to meet again. Sen knows about the invitation but has not read the letter. A possible reunion complicates Dara's feelings about remaining at a provincial courthouse; no decision has been made.

Dara records testimony at a provincial court. Sen claims a copper seal was copied, but the expert has examined only a drawing. Dara knows the original seal remains in a locked evidence chest. The judge has prohibited discussion of the witness's family outside court. A clerk spills sand on the back steps. The hearing pauses before the chest can be opened. No verdict has been given.

Write 120–220 words about Dara and Sen in a provincial courthouse, in formal, precise prose. Use close third person centered on Dara. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Show an interrupted practical task rather than summarizing. Return only the prose, without a title or commentary.

This is for a romance story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The original seal remains in a locked evidence chest. Keep Sen alleges copying; the expert has seen only a drawing. Do not assert: The court finds the seal counterfeit."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Dara; formal, precise prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested romance genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-05 — F2

Genre: historical_fiction

Instruction specificity: explicit

Read notes/source.md and drafts/scene.md. Expand only the first sentence into 120–220 words in formal, precise prose. Keep the final sentence exactly unchanged. Put only the replacement prose between <prose> and </prose> in the file, then retain the final sentence after </prose>. Use close third person on Dara. Preserve facts and unresolved beliefs. Reply with a short completion note; do not repeat the scene.

This is for a historical fiction story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "preserve-ending",
      "metric": "Q1",
      "kind": "protected",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md",
      "text": "The judge called for silence."
    },
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The original seal remains in a locked evidence chest. Keep Sen alleges copying; the expert has seen only a drawing. Do not assert: The court finds the seal counterfeit."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Dara; formal, precise prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested historical fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-05 — F3

Genre: adventure

Instruction specificity: explicit

A route beyond a provincial courthouse is opening for the first time in years. Dara and Sen have been invited to join an expedition, but preparations are incomplete. The present problem must be considered alongside the chance to leave; they have not departed.

Dara records testimony at a provincial court. Sen claims a copper seal was copied, but the expert has examined only a drawing. Dara knows the original seal remains in a locked evidence chest. The judge has prohibited discussion of the witness's family outside court. A clerk spills sand on the back steps. The hearing pauses before the chest can be opened. No verdict has been given.

Give three numbered, meaningfully different next-scene directions. For each, explain the dramatic choice, its consequence, and why it fits known facts. Do not draft the scene or resolve the central uncertainty.

This is for a adventure story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "three-directions",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Exactly three developed viable directions."
    }
  ],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Among all three requested pairs, count pairs whose two alternatives are viable and meaningfully different in dramatic choice or consequence. Missing alternatives earn no diverse pairs. Return diverse viable pairs / 3.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-05 — F4

Genre: literary_fiction

Instruction specificity: explicit

Read source/chapter.md. Create a navigable Markdown continuity wiki under kb/, starting at kb/index.md. Its audience is a writer planning the next chapter. Prioritize consequential facts, uncertainty, and character knowledge. At least one linked topic page is needed. Omit incidental details where appropriate; cite the source path. Keep source files unchanged. Use no more than 500 words across KB pages.

This is for a literary fiction story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "wiki-words",
      "metric": "Q1",
      "kind": "kb_word_budget",
      "method": "deterministic",
      "required": false,
      "max": 500
    },
    {
      "id": "wiki-links",
      "metric": "Q1",
      "kind": "wiki_links",
      "method": "deterministic",
      "required": true
    },
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "The original seal remains in a locked evidence chest."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "weight": 3,
      "text": "Sen alleges copying; the expert has seen only a drawing."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "The witness's family must not be discussed outside court."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "The original seal remains in a locked evidence chest.",
      "importance": "required",
      "weight": 3,
      "evidence": "Dara has been considering leaving a provincial courthouse. Sen has noticed the distance between them, but they have not talked about what leaving would mean. Their everyday work carries the weight of that unspoken choice.\n\nDara records testimony at a provincial court. Sen claims a copper seal was copied, but the expert has examined only a drawing. Dara knows the original seal remains in a locked evidence chest. The judge has prohibited discussion of the witness's family outside court. A clerk spills sand on the back steps. The hearing pauses before the chest can be opened. No verdict has been given.",
      "after_update": null
    },
    {
      "id": "belief",
      "statement": "Sen alleges copying; the expert has seen only a drawing.",
      "importance": "required",
      "weight": 3,
      "evidence": "Dara has been considering leaving a provincial courthouse. Sen has noticed the distance between them, but they have not talked about what leaving would mean. Their everyday work carries the weight of that unspoken choice.\n\nDara records testimony at a provincial court. Sen claims a copper seal was copied, but the expert has examined only a drawing. Dara knows the original seal remains in a locked evidence chest. The judge has prohibited discussion of the witness's family outside court. A clerk spills sand on the back steps. The hearing pauses before the chest can be opened. No verdict has been given.",
      "placement": "Any findable page; keep attribution and uncertainty."
    },
    {
      "id": "context",
      "statement": "The witness's family must not be discussed outside court.",
      "importance": "optional",
      "weight": 1,
      "evidence": "Dara has been considering leaving a provincial courthouse. Sen has noticed the distance between them, but they have not talked about what leaving would mean. Their everyday work carries the weight of that unspoken choice.\n\nDara records testimony at a provincial court. Sen claims a copper seal was copied, but the expert has examined only a drawing. Dara knows the original seal remains in a locked evidence chest. The judge has prohibited discussion of the witness's family outside court. A clerk spills sand on the back steps. The hearing pauses before the chest can be opened. No verdict has been given."
    },
    {
      "id": "incidental",
      "statement": "A clerk spilled sand on the steps.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "Dara has been considering leaving a provincial courthouse. Sen has noticed the distance between them, but they have not talked about what leaving would mean. Their everyday work carries the weight of that unspoken choice.\n\nDara records testimony at a provincial court. Sen claims a copper seal was copied, but the expert has examined only a drawing. Dara knows the original seal remains in a locked evidence chest. The judge has prohibited discussion of the witness's family outside court. A clerk spills sand on the back steps. The hearing pauses before the chest can be opened. No verdict has been given.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "The original seal remains in a locked evidence chest."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Sen alleges copying; the expert has seen only a drawing."
    }
  ]
}
```

## F5-05 — F5

Genre: fantasy

Instruction specificity: explicit

Use the continuity wiki starting at kb/index.md. Write 120–220 words about Dara and Sen in a provincial courthouse, in formal, precise prose. Use close third person centered on Dara. End before the central uncertainty is resolved. Invent dialogue and sensory details, but respect established facts and character knowledge. Save prose in drafts/scene.md; keep the KB unchanged.

This is for a fantasy story.

**Private review labels**

```json
{
  "rubric_version": 1,
  "checks": [
    {
      "id": "retrieve-state",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "The original seal remains in a locked evidence chest."
    },
    {
      "id": "retrieve-belief",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "Sen alleges copying; the expert has seen only a drawing."
    },
    {
      "id": "preserve-kb",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "word-budget",
      "metric": "Q1",
      "kind": "word_range",
      "method": "deterministic",
      "required": false,
      "artifact": "scene",
      "min": 120,
      "max": 220
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Preserve The original seal remains in a locked evidence chest. Keep Sen alleges copying; the expert has seen only a drawing. Do not assert: The court finds the seal counterfeit."
    },
    {
      "id": "pov-style",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Close third person on Dara; formal, precise prose."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested fantasy genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-06 — F1

Genre: horror

Instruction specificity: loose

For three nights, people near a closing neighborhood cinema have heard their own voices answering from empty rooms. Emil heard it once and has told Ruth, who has not witnessed it. No cause has been established, and nobody agrees on whether entering alone is safe.

Emil cleans a cinema scheduled to close on Friday. Ruth finds an unlabeled reel and says it may show the old neighborhood festival. She has seen only its first damaged frame. Emil promised the owner that nothing would be screened without checking the projector. A green umbrella lies in the aisle. The reel is stored in the projection booth overnight. Nobody has watched the full film.

Could you turn this into a scene? I'd like it to feel more alive. Just the story, please.

This is for a horror story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested horror genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-06 — F2

Genre: comedy

Instruction specificity: loose

Could you make a scene out of the notes and save it in drafts/scene.md? The background is in notes/source.md. Leave the notes alone.

This is for a comedy story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested comedy genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-06 — F3

Genre: thriller

Instruction specificity: loose

Someone has warned Emil that a stranger will arrive at a closing neighborhood cinema before morning asking for information. Ruth has seen signs that someone has been watching the entrance. The warning's source and the stranger's intentions remain unknown.

Emil cleans a cinema scheduled to close on Friday. Ruth finds an unlabeled reel and says it may show the old neighborhood festival. She has seen only its first damaged frame. Emil promised the owner that nothing would be screened without checking the projector. A green umbrella lies in the aisle. The reel is stored in the projection booth overnight. Nobody has watched the full film.

I'm not sure where to take this. What are some directions I could try?

This is for a thriller story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Assess meaningful variety among the viable directions actually offered. Do not require a fixed number, numbered formatting, or a prescribed explanation. Count meaningfully different viable pairs divided by all pairs of offered directions. Fewer than two directions yields zero; superficial rephrasings are not diverse. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-06 — F4

Genre: mystery

Instruction specificity: loose

Can you turn source/chapter.md into a small Markdown wiki I can use while writing the next chapter? Put the starting page at kb/index.md and make things easy to find. Leave the chapter itself alone.

This is for a mystery story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: The projector is safe; the reel's contents remain unknown."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: Ruth's festival identification is tentative."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: The cinema is scheduled to close on Friday."
    },
    {
      "id": "accepted-update",
      "metric": "Q11",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "The projector is safe; the reel's contents remain unknown. Preserve relevant history and do not adopt the draft-only suggestion."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "The projector must be checked before anything is screened.",
      "importance": "optional",
      "weight": 3,
      "evidence": "An anonymous note has arrived at a closing neighborhood cinema, referring to an earlier disappearance. Emil and Ruth have different recollections of that day. Neither the note's author nor its connection to the present problem has been established.\n\nEmil cleans a cinema scheduled to close on Friday. Ruth finds an unlabeled reel and says it may show the old neighborhood festival. She has seen only its first damaged frame. Emil promised the owner that nothing would be screened without checking the projector. A green umbrella lies in the aisle. The reel is stored in the projection booth overnight. Nobody has watched the full film.",
      "after_update": "The projector is safe; the reel's contents remain unknown.",
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "belief",
      "statement": "Ruth's festival identification is tentative.",
      "importance": "optional",
      "weight": 3,
      "evidence": "An anonymous note has arrived at a closing neighborhood cinema, referring to an earlier disappearance. Emil and Ruth have different recollections of that day. Neither the note's author nor its connection to the present problem has been established.\n\nEmil cleans a cinema scheduled to close on Friday. Ruth finds an unlabeled reel and says it may show the old neighborhood festival. She has seen only its first damaged frame. Emil promised the owner that nothing would be screened without checking the projector. A green umbrella lies in the aisle. The reel is stored in the projection booth overnight. Nobody has watched the full film.",
      "placement": "Any findable page; keep attribution and uncertainty.",
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "context",
      "statement": "The cinema is scheduled to close on Friday.",
      "importance": "optional",
      "weight": 1,
      "evidence": "An anonymous note has arrived at a closing neighborhood cinema, referring to an earlier disappearance. Emil and Ruth have different recollections of that day. Neither the note's author nor its connection to the present problem has been established.\n\nEmil cleans a cinema scheduled to close on Friday. Ruth finds an unlabeled reel and says it may show the old neighborhood festival. She has seen only its first damaged frame. Emil promised the owner that nothing would be screened without checking the projector. A green umbrella lies in the aisle. The reel is stored in the projection booth overnight. Nobody has watched the full film."
    },
    {
      "id": "incidental",
      "statement": "The umbrella is green.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "An anonymous note has arrived at a closing neighborhood cinema, referring to an earlier disappearance. Emil and Ruth have different recollections of that day. Neither the note's author nor its connection to the present problem has been established.\n\nEmil cleans a cinema scheduled to close on Friday. Ruth finds an unlabeled reel and says it may show the old neighborhood festival. She has seen only its first damaged frame. Emil promised the owner that nothing would be screened without checking the projector. A green umbrella lies in the aisle. The reel is stored in the projection booth overnight. Nobody has watched the full film.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "The projector is safe; the reel's contents remain unknown."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Ruth's festival identification is tentative."
    }
  ]
}
```

## F5-06 — F5

Genre: science_fiction

Instruction specificity: loose

a closing neighborhood cinema is part of a settlement on an artificial island maintained by autonomous systems. A recent system update has changed the way public records are synchronized. Emil and Ruth have not established whether this affects the events described below.

The projector must be checked before anything is screened.

The cinema is scheduled to close on Friday.

Ruth's festival identification is tentative.

Can you write a scene from these notes? Just the story.

This is for a science fiction story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested science fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-07 — F1

Genre: historical_fiction

Instruction specificity: loose

The year is 1936. News reaches a drought-struck communal garden through letters, newspapers, and radio broadcasts. A regional strike has delayed ordinary deliveries, and Yara and Belen are working with the supplies already on hand. Any unusual phenomena in the supplied premise remain part of this story's world.

Yara allocates water in a communal garden. Belen finds a buried clay jar and believes it marks an old well. No map or excavation confirms this. Yara must reserve two barrels for seedlings under an agreement with the growers. A dog sleeps against the west wall. The council will discuss digging after the morning inspection. Until then, the jar stays where it was found.

Could you turn this into a scene? I'd like it to feel more alive. Just the story, please.

This is for a historical fiction story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested historical fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-07 — F2

Genre: adventure

Instruction specificity: loose

The scene in drafts/scene.md feels flat. Can you rework it? The background is in notes/source.md. Leave the notes alone.

This is for a adventure story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested adventure genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-07 — F3

Genre: literary_fiction

Instruction specificity: loose

Yara has been considering leaving a drought-struck communal garden. Belen has noticed the distance between them, but they have not talked about what leaving would mean. Their everyday work carries the weight of that unspoken choice.

Yara allocates water in a communal garden. Belen finds a buried clay jar and believes it marks an old well. No map or excavation confirms this. Yara must reserve two barrels for seedlings under an agreement with the growers. A dog sleeps against the west wall. The council will discuss digging after the morning inspection. Until then, the jar stays where it was found.

I'm not sure where to take this. What are some directions I could try?

This is for a literary fiction story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Assess meaningful variety among the viable directions actually offered. Do not require a fixed number, numbered formatting, or a prescribed explanation. Count meaningfully different viable pairs divided by all pairs of offered directions. Fewer than two directions yields zero; superficial rephrasings are not diverse. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-07 — F4

Genre: fantasy

Instruction specificity: loose

Can you turn source/chapter.md into a small Markdown wiki I can use while writing the next chapter? Put the starting page at kb/index.md and make things easy to find. Leave the chapter itself alone.

This is for a fantasy story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: Two water barrels are reserved for seedlings."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: Belen's proposed well is unconfirmed."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: The council discusses digging after the morning inspection."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "Two water barrels are reserved for seedlings.",
      "importance": "optional",
      "weight": 3,
      "evidence": "In this region, places can remember promises spoken inside them. a drought-struck communal garden has begun repeating a fragment of an old vow after dark. Yara and Belen do not know whose promise it was or whether it concerns them.\n\nYara allocates water in a communal garden. Belen finds a buried clay jar and believes it marks an old well. No map or excavation confirms this. Yara must reserve two barrels for seedlings under an agreement with the growers. A dog sleeps against the west wall. The council will discuss digging after the morning inspection. Until then, the jar stays where it was found.",
      "after_update": null,
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "belief",
      "statement": "Belen's proposed well is unconfirmed.",
      "importance": "optional",
      "weight": 3,
      "evidence": "In this region, places can remember promises spoken inside them. a drought-struck communal garden has begun repeating a fragment of an old vow after dark. Yara and Belen do not know whose promise it was or whether it concerns them.\n\nYara allocates water in a communal garden. Belen finds a buried clay jar and believes it marks an old well. No map or excavation confirms this. Yara must reserve two barrels for seedlings under an agreement with the growers. A dog sleeps against the west wall. The council will discuss digging after the morning inspection. Until then, the jar stays where it was found.",
      "placement": "Any findable page; keep attribution and uncertainty.",
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "context",
      "statement": "The council discusses digging after the morning inspection.",
      "importance": "optional",
      "weight": 1,
      "evidence": "In this region, places can remember promises spoken inside them. a drought-struck communal garden has begun repeating a fragment of an old vow after dark. Yara and Belen do not know whose promise it was or whether it concerns them.\n\nYara allocates water in a communal garden. Belen finds a buried clay jar and believes it marks an old well. No map or excavation confirms this. Yara must reserve two barrels for seedlings under an agreement with the growers. A dog sleeps against the west wall. The council will discuss digging after the morning inspection. Until then, the jar stays where it was found."
    },
    {
      "id": "incidental",
      "statement": "A dog sleeps against the west wall.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "In this region, places can remember promises spoken inside them. a drought-struck communal garden has begun repeating a fragment of an old vow after dark. Yara and Belen do not know whose promise it was or whether it concerns them.\n\nYara allocates water in a communal garden. Belen finds a buried clay jar and believes it marks an old well. No map or excavation confirms this. Yara must reserve two barrels for seedlings under an agreement with the growers. A dog sleeps against the west wall. The council will discuss digging after the morning inspection. Until then, the jar stays where it was found.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "Two water barrels are reserved for seedlings."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Belen's proposed well is unconfirmed."
    }
  ]
}
```

## F5-07 — F5

Genre: romance

Instruction specificity: loose

Could you write a scene using the notes in kb/? Save it in drafts/scene.md. Keep the notes as they are for now.

This is for a romance story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "retrieve-state",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "Two water barrels are reserved for seedlings."
    },
    {
      "id": "retrieve-belief",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "Belen's proposed well is unconfirmed."
    },
    {
      "id": "preserve-kb",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested romance genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-08 — F1

Genre: comedy

Instruction specificity: loose

A visiting committee has mistaken Kavi for the person in charge of a border map office. Noor knows the mistake, but an elaborate welcome has already begun. Their attempt to get through the visit without embarrassing anyone keeps producing smaller misunderstandings.

Kavi copies boundary maps. Noor brings a folded survey that places her house on both sides of the border. The surveyor's signature is missing. Kavi thinks a copying error caused the overlap, but the archive has not been checked. Only the supervisor can authorize a corrected official copy. A yellow stamp pad has dried out. Noor leaves her map for inspection and keeps her existing residence papers.

Could you turn this into a scene? I'd like it to feel more alive. Just the story, please.

This is for a comedy story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested comedy genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-08 — F2

Genre: thriller

Instruction specificity: loose

Could you make a scene out of the notes and save it in drafts/scene.md? The background is in notes/source.md. Leave the notes alone.

This is for a thriller story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested thriller genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-08 — F3

Genre: mystery

Instruction specificity: loose

An anonymous note has arrived at a border map office, referring to an earlier disappearance. Kavi and Noor have different recollections of that day. Neither the note's author nor its connection to the present problem has been established.

Kavi copies boundary maps. Noor brings a folded survey that places her house on both sides of the border. The surveyor's signature is missing. Kavi thinks a copying error caused the overlap, but the archive has not been checked. Only the supervisor can authorize a corrected official copy. A yellow stamp pad has dried out. Noor leaves her map for inspection and keeps her existing residence papers.

I'm not sure where to take this. What are some directions I could try?

This is for a mystery story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Assess meaningful variety among the viable directions actually offered. Do not require a fixed number, numbered formatting, or a prescribed explanation. Count meaningfully different viable pairs divided by all pairs of offered directions. Fewer than two directions yields zero; superficial rephrasings are not diverse. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-08 — F4

Genre: science_fiction

Instruction specificity: loose

Can you turn source/chapter.md into a small Markdown wiki I can use while writing the next chapter? Put the starting page at kb/index.md and make things easy to find. Leave the chapter itself alone.

This is for a science fiction story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: A provisional correction is authorized; the archive check is still pending."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: Kavi suspects a copying error; the cause has not been established."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: Noor retains her existing residence papers."
    },
    {
      "id": "accepted-update",
      "metric": "Q11",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "A provisional correction is authorized; the archive check is still pending. Preserve relevant history and do not adopt the draft-only suggestion."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "Only the supervisor can authorize a corrected official copy.",
      "importance": "optional",
      "weight": 3,
      "evidence": "a border map office is part of a settlement on an artificial island maintained by autonomous systems. A recent system update has changed the way public records are synchronized. Kavi and Noor have not established whether this affects the events described below.\n\nKavi copies boundary maps. Noor brings a folded survey that places her house on both sides of the border. The surveyor's signature is missing. Kavi thinks a copying error caused the overlap, but the archive has not been checked. Only the supervisor can authorize a corrected official copy. A yellow stamp pad has dried out. Noor leaves her map for inspection and keeps her existing residence papers.",
      "after_update": "A provisional correction is authorized; the archive check is still pending.",
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "belief",
      "statement": "Kavi suspects a copying error; the cause has not been established.",
      "importance": "optional",
      "weight": 3,
      "evidence": "a border map office is part of a settlement on an artificial island maintained by autonomous systems. A recent system update has changed the way public records are synchronized. Kavi and Noor have not established whether this affects the events described below.\n\nKavi copies boundary maps. Noor brings a folded survey that places her house on both sides of the border. The surveyor's signature is missing. Kavi thinks a copying error caused the overlap, but the archive has not been checked. Only the supervisor can authorize a corrected official copy. A yellow stamp pad has dried out. Noor leaves her map for inspection and keeps her existing residence papers.",
      "placement": "Any findable page; keep attribution and uncertainty.",
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "context",
      "statement": "Noor retains her existing residence papers.",
      "importance": "optional",
      "weight": 1,
      "evidence": "a border map office is part of a settlement on an artificial island maintained by autonomous systems. A recent system update has changed the way public records are synchronized. Kavi and Noor have not established whether this affects the events described below.\n\nKavi copies boundary maps. Noor brings a folded survey that places her house on both sides of the border. The surveyor's signature is missing. Kavi thinks a copying error caused the overlap, but the archive has not been checked. Only the supervisor can authorize a corrected official copy. A yellow stamp pad has dried out. Noor leaves her map for inspection and keeps her existing residence papers."
    },
    {
      "id": "incidental",
      "statement": "The stamp pad is yellow and dry.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "a border map office is part of a settlement on an artificial island maintained by autonomous systems. A recent system update has changed the way public records are synchronized. Kavi and Noor have not established whether this affects the events described below.\n\nKavi copies boundary maps. Noor brings a folded survey that places her house on both sides of the border. The surveyor's signature is missing. Kavi thinks a copying error caused the overlap, but the archive has not been checked. Only the supervisor can authorize a corrected official copy. A yellow stamp pad has dried out. Noor leaves her map for inspection and keeps her existing residence papers.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "A provisional correction is authorized; the archive check is still pending."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Kavi suspects a copying error; the cause has not been established."
    }
  ]
}
```

## F5-08 — F5

Genre: horror

Instruction specificity: loose

For three nights, people near a border map office have heard their own voices answering from empty rooms. Kavi heard it once and has told Noor, who has not witnessed it. No cause has been established, and nobody agrees on whether entering alone is safe.

Only the supervisor can authorize a corrected official copy.

Noor retains her existing residence papers.

Kavi suspects a copying error; the cause has not been established.

Can you write a scene from these notes? Just the story.

This is for a horror story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested horror genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-09 — F1

Genre: adventure

Instruction specificity: loose

A route beyond a clockmaker's workshop is opening for the first time in years. Pia and Ren have been invited to join an expedition, but preparations are incomplete. The present problem must be considered alongside the chance to leave; they have not departed.

Pia inherits a watch that stops whenever Ren enters the workshop. Ren says the spring was replaced last winter, though the repair book has a missing page. Pia suspects a trick and has no proof. The watch cannot be opened without its uncommon square key, held by a collector. A white moth rests on the window. Pia agrees to write down the stopping times before drawing conclusions.

Could you turn this into a scene? I'd like it to feel more alive. Just the story, please.

This is for a adventure story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested adventure genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-09 — F2

Genre: literary_fiction

Instruction specificity: loose

The scene in drafts/scene.md feels flat. Can you rework it? The background is in notes/source.md. Leave the notes alone.

This is for a literary fiction story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested literary fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-09 — F3

Genre: fantasy

Instruction specificity: loose

In this region, places can remember promises spoken inside them. a clockmaker's workshop has begun repeating a fragment of an old vow after dark. Pia and Ren do not know whose promise it was or whether it concerns them.

Pia inherits a watch that stops whenever Ren enters the workshop. Ren says the spring was replaced last winter, though the repair book has a missing page. Pia suspects a trick and has no proof. The watch cannot be opened without its uncommon square key, held by a collector. A white moth rests on the window. Pia agrees to write down the stopping times before drawing conclusions.

I'm not sure where to take this. What are some directions I could try?

This is for a fantasy story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Assess meaningful variety among the viable directions actually offered. Do not require a fixed number, numbered formatting, or a prescribed explanation. Count meaningfully different viable pairs divided by all pairs of offered directions. Fewer than two directions yields zero; superficial rephrasings are not diverse. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-09 — F4

Genre: romance

Instruction specificity: loose

Can you turn source/chapter.md into a small Markdown wiki I can use while writing the next chapter? Put the starting page at kb/index.md and make things easy to find. Leave the chapter itself alone.

This is for a romance story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: A collector holds the square key needed to open the watch."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: Pia suspects a trick; neither mechanism nor intent is established."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: Pia agreed to record stopping times before drawing conclusions."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "A collector holds the square key needed to open the watch.",
      "importance": "optional",
      "weight": 3,
      "evidence": "Pia has received a letter from an old love who wants to meet again. Ren knows about the invitation but has not read the letter. A possible reunion complicates Pia's feelings about remaining at a clockmaker's workshop; no decision has been made.\n\nPia inherits a watch that stops whenever Ren enters the workshop. Ren says the spring was replaced last winter, though the repair book has a missing page. Pia suspects a trick and has no proof. The watch cannot be opened without its uncommon square key, held by a collector. A white moth rests on the window. Pia agrees to write down the stopping times before drawing conclusions.",
      "after_update": null,
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "belief",
      "statement": "Pia suspects a trick; neither mechanism nor intent is established.",
      "importance": "optional",
      "weight": 3,
      "evidence": "Pia has received a letter from an old love who wants to meet again. Ren knows about the invitation but has not read the letter. A possible reunion complicates Pia's feelings about remaining at a clockmaker's workshop; no decision has been made.\n\nPia inherits a watch that stops whenever Ren enters the workshop. Ren says the spring was replaced last winter, though the repair book has a missing page. Pia suspects a trick and has no proof. The watch cannot be opened without its uncommon square key, held by a collector. A white moth rests on the window. Pia agrees to write down the stopping times before drawing conclusions.",
      "placement": "Any findable page; keep attribution and uncertainty.",
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "context",
      "statement": "Pia agreed to record stopping times before drawing conclusions.",
      "importance": "optional",
      "weight": 1,
      "evidence": "Pia has received a letter from an old love who wants to meet again. Ren knows about the invitation but has not read the letter. A possible reunion complicates Pia's feelings about remaining at a clockmaker's workshop; no decision has been made.\n\nPia inherits a watch that stops whenever Ren enters the workshop. Ren says the spring was replaced last winter, though the repair book has a missing page. Pia suspects a trick and has no proof. The watch cannot be opened without its uncommon square key, held by a collector. A white moth rests on the window. Pia agrees to write down the stopping times before drawing conclusions."
    },
    {
      "id": "incidental",
      "statement": "A white moth rests on the window.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "Pia has received a letter from an old love who wants to meet again. Ren knows about the invitation but has not read the letter. A possible reunion complicates Pia's feelings about remaining at a clockmaker's workshop; no decision has been made.\n\nPia inherits a watch that stops whenever Ren enters the workshop. Ren says the spring was replaced last winter, though the repair book has a missing page. Pia suspects a trick and has no proof. The watch cannot be opened without its uncommon square key, held by a collector. A white moth rests on the window. Pia agrees to write down the stopping times before drawing conclusions.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "A collector holds the square key needed to open the watch."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Pia suspects a trick; neither mechanism nor intent is established."
    }
  ]
}
```

## F5-09 — F5

Genre: historical_fiction

Instruction specificity: loose

Could you write a scene using the notes in kb/? Save it in drafts/scene.md. Keep the notes as they are for now.

This is for a historical fiction story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "retrieve-state",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "A collector holds the square key needed to open the watch."
    },
    {
      "id": "retrieve-belief",
      "metric": "Q12",
      "kind": "evidence_exposed",
      "method": "deterministic",
      "required": false,
      "text": "Pia suspects a trick; neither mechanism nor intent is established."
    },
    {
      "id": "preserve-kb",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested historical fiction genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F1-10 — F1

Genre: thriller

Instruction specificity: loose

Someone has warned Ada that a stranger will arrive at a family restaurant after closing before morning asking for information. Milo has seen signs that someone has been watching the entrance. The warning's source and the stranger's intentions remain unknown.

Ada and Milo run their aunt's restaurant for a week. Milo finds a recipe card and thinks the crossed-out ingredient is the secret to her soup. Their aunt has not confirmed this and is away until Sunday. Ada promised not to serve untested dishes to paying guests. A spoon lies under the freezer. They agree to make a small test batch after closing and keep the regular menu tomorrow.

Could you turn this into a scene? I'd like it to feel more alive. Just the story, please.

This is for a thriller story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested thriller genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F2-10 — F2

Genre: mystery

Instruction specificity: loose

Could you make a scene out of the notes and save it in drafts/scene.md? The background is in notes/source.md. Leave the notes alone.

This is for a mystery story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "saved-scene",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "drafts/scene.md"
    },
    {
      "id": "edit-scope",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "drafts/scene.md"
      ]
    },
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested mystery genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F3-10 — F3

Genre: science_fiction

Instruction specificity: loose

a family restaurant after closing is part of a settlement on an artificial island maintained by autonomous systems. A recent system update has changed the way public records are synchronized. Ada and Milo have not established whether this affects the events described below.

Ada and Milo run their aunt's restaurant for a week. Milo finds a recipe card and thinks the crossed-out ingredient is the secret to her soup. Their aunt has not confirmed this and is away until Sunday. Ada promised not to serve untested dishes to paying guests. A spoon lies under the freezer. They agree to make a small test batch after closing and keep the regular menu tomorrow.

I'm not sure where to take this. What are some directions I could try?

This is for a science fiction story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [],
  "rubrics": {
    "Q5": {
      "description": "Assess usefulness, feasibility within canon, and actionable development; honor the latest feedback. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    },
    "Q6": {
      "description": "Assess meaningful variety among the viable directions actually offered. Do not require a fixed number, numbered formatting, or a prescribed explanation. Count meaningfully different viable pairs divided by all pairs of offered directions. Fewer than two directions yields zero; superficial rephrasings are not diverse. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```

## F4-10 — F4

Genre: horror

Instruction specificity: loose

Can you turn source/chapter.md into a small Markdown wiki I can use while writing the next chapter? Put the starting page at kb/index.md and make things easy to find. Leave the chapter itself alone.

This is for a horror story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "wiki-index",
      "metric": "Q3",
      "kind": "nonempty",
      "method": "deterministic",
      "required": true,
      "path": "kb/index.md"
    },
    {
      "id": "wiki-only",
      "metric": "Q1",
      "kind": "allowed_changes",
      "method": "deterministic",
      "required": true,
      "paths": [
        "kb/*.md"
      ]
    },
    {
      "id": "consequential-state",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: The batch has been tasted privately; tomorrow's regular menu is unchanged."
    },
    {
      "id": "qualified-belief",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 3,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: Milo's interpretation of the crossed-out ingredient is speculative."
    },
    {
      "id": "optional-context",
      "metric": "Q8",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "weight": 1,
      "text": "Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item: They will keep the regular menu tomorrow."
    },
    {
      "id": "accepted-update",
      "metric": "Q11",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "The batch has been tasted privately; tomorrow's regular menu is unchanged. Preserve relevant history and do not adopt the draft-only suggestion."
    }
  ],
  "rubrics": {
    "Q7": {
      "description": "Identify factual KB claims and count source-supported claims / checked claims. Distinguish uncertainty and qualified interpretation; an empty KB cannot pass. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        0,
        1
      ],
      "anchors": "0: none satisfied; 1: all satisfied. State numerator and denominator."
    },
    "Q9": {
      "description": "Assess evidence, entity and temporal understanding, selection for the stated purpose, and uncertainty. Accept defensible alternative interpretations and placement. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence."
    }
  },
  "knowledge": [
    {
      "id": "essential",
      "statement": "Untested dishes must not be served to paying guests.",
      "importance": "optional",
      "weight": 3,
      "evidence": "For three nights, people near a family restaurant after closing have heard their own voices answering from empty rooms. Ada heard it once and has told Milo, who has not witnessed it. No cause has been established, and nobody agrees on whether entering alone is safe.\n\nAda and Milo run their aunt's restaurant for a week. Milo finds a recipe card and thinks the crossed-out ingredient is the secret to her soup. Their aunt has not confirmed this and is away until Sunday. Ada promised not to serve untested dishes to paying guests. A spoon lies under the freezer. They agree to make a small test batch after closing and keep the regular menu tomorrow.",
      "after_update": "The batch has been tasted privately; tomorrow's regular menu is unchanged.",
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "belief",
      "statement": "Milo's interpretation of the crossed-out ingredient is speculative.",
      "importance": "optional",
      "weight": 3,
      "evidence": "For three nights, people near a family restaurant after closing have heard their own voices answering from empty rooms. Ada heard it once and has told Milo, who has not witnessed it. No cause has been established, and nobody agrees on whether entering alone is safe.\n\nAda and Milo run their aunt's restaurant for a week. Milo finds a recipe card and thinks the crossed-out ingredient is the secret to her soup. Their aunt has not confirmed this and is away until Sunday. Ada promised not to serve untested dishes to paying guests. A spoon lies under the freezer. They agree to make a small test batch after closing and keep the regular menu tomorrow.",
      "placement": "Any findable page; keep attribution and uncertainty.",
      "omission": "Accept if defensible for next-chapter planning."
    },
    {
      "id": "context",
      "statement": "They will keep the regular menu tomorrow.",
      "importance": "optional",
      "weight": 1,
      "evidence": "For three nights, people near a family restaurant after closing have heard their own voices answering from empty rooms. Ada heard it once and has told Milo, who has not witnessed it. No cause has been established, and nobody agrees on whether entering alone is safe.\n\nAda and Milo run their aunt's restaurant for a week. Milo finds a recipe card and thinks the crossed-out ingredient is the secret to her soup. Their aunt has not confirmed this and is away until Sunday. Ada promised not to serve untested dishes to paying guests. A spoon lies under the freezer. They agree to make a small test batch after closing and keep the regular menu tomorrow."
    },
    {
      "id": "incidental",
      "statement": "A spoon lies under the freezer.",
      "importance": "incidental",
      "weight": 0,
      "evidence": "For three nights, people near a family restaurant after closing have heard their own voices answering from empty rooms. Ada heard it once and has told Milo, who has not witnessed it. No cause has been established, and nobody agrees on whether entering alone is safe.\n\nAda and Milo run their aunt's restaurant for a week. Milo finds a recipe card and thinks the crossed-out ingredient is the secret to her soup. Their aunt has not confirmed this and is away until Sunday. Ada promised not to serve untested dishes to paying guests. A spoon lies under the freezer. They agree to make a small test batch after closing and keep the regular menu tomorrow.",
      "omission": "allowed"
    }
  ],
  "source_cutoff": "supplied passage only",
  "entrypoints": [
    "kb/index.md"
  ],
  "probes": [
    {
      "id": "state",
      "question": "What practical restriction or permission governs the next action? Answer with supporting page paths.",
      "expected": "The batch has been tasted privately; tomorrow's regular menu is unchanged."
    },
    {
      "id": "belief",
      "question": "Which central belief is still uncertain? Name its owner and supporting page paths.",
      "expected": "Milo's interpretation of the crossed-out ingredient is speculative."
    }
  ]
}
```

## F5-10 — F5

Genre: comedy

Instruction specificity: loose

A visiting committee has mistaken Ada for the person in charge of a family restaurant after closing. Milo knows the mistake, but an elaborate welcome has already begun. Their attempt to get through the visit without embarrassing anyone keeps producing smaller misunderstandings.

Untested dishes must not be served to paying guests.

They will keep the regular menu tomorrow.

Milo's interpretation of the crossed-out ingredient is speculative.

Can you write a scene from these notes? Just the story.

This is for a comedy story.

**Private review labels**

```json
{
  "rubric_version": 2,
  "checks": [
    {
      "id": "continuity",
      "metric": "Q13",
      "kind": "semantic",
      "method": "llm_judge",
      "required": true,
      "text": "Respect the supplied starting facts and distinguish character beliefs from established knowledge. Accept plausible new events, revelations, dialogue, viewpoint, and endings; no particular continuation is required."
    },
    {
      "id": "genre-fit",
      "metric": "Q1",
      "kind": "semantic",
      "method": "llm_judge",
      "required": false,
      "text": "Fit the requested comedy genre in a defensible way; do not require a fixed trope checklist. Genre blends are allowed where the source premise supports them."
    }
  ],
  "rubrics": {
    "Q2": {
      "description": "Rate selected prose holistically for coherence, characterization, pacing, language, and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. Do not penalize an allowed alternative plot for differing from a reference. The request leaves creative choices open. Do not invent unstated requirements for length, viewpoint, style, structure, ending, or number of ideas.",
      "range": [
        1,
        5
      ],
      "anchors": "1: fails the brief; 2: substantial weaknesses; 3: usable with revision; 4: effective with minor weaknesses; 5: consistently effective. Cite concrete evidence.",
      "dimensions": [
        "coherence",
        "characterization",
        "pacing",
        "language",
        "redundancy",
        "style_adherence"
      ]
    }
  },
  "knowledge": [],
  "source_cutoff": "supplied passage only"
}
```
