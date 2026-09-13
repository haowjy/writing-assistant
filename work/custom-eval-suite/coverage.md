# Current scenario coverage

Snapshot: 2026-09-12. The collection contains **50 synthetic development scenarios
across ten worlds**. This table records the inputs we have, not the variety a model
might produce from them. It is an audit aid for final data generation; coverage
quotas and acceptance remain undecided.

Sources: [world records](../../data/scenarios/worlds.json),
[scenario definitions](../../data/scenarios/development.json),
[genre contexts](../../data/scenarios/genres.json), and
[review packet](review/review.md). Counts below refer to scenario definitions,
not completed evaluations. The current revision includes the requested genre mix:
ten genre contexts distributed across the existing 50 tasks. It does not integrate
external datasets or claim final genre quality.

## Coverage by dimension

| Dimension | Present in the current collection | Gap to revisit during final data generation |
|---|---|---|
| Task family | Five families, ten cases each: prose, file authoring, brainstorming, KB construction, KB use | Check balance within each family as external-source cases are added. |
| Genre | Ten explicit labels, five cases per genre and one per family; each genre has two or three loose requests and the remainder explicit | Contexts are short authored frames, not full independent genre narratives. Blends are allowed (for example, the glass-tree premise in a historical setting). Review whether each case actually exercises its intended genre. |
| Tone/style | Separate `genre` and `prose_style` fields. Ten prose-style descriptions; loose scenarios keep `style=unspecified` | Not a balanced style experiment. Explicit F3/F4 metadata does not imply the prompt requests that prose style. |
| Viewpoint | Close third person requested in the 15 explicit prose-producing cases (F1/F2/F5, worlds 01–05) | No designated first-person, omniscient, second-person, multiple-viewpoint, or unreliable-narrator conditions. Loose requests leave viewpoint open. |
| Narrative form | Short scenes, local revisions, brainstorming replies; five stored F1 subtype labels | No designated letters, journals, scripts, verse, or nonlinear narratives. The loose F1 cases do not actually request their stored subtypes. |
| Character names | Twenty distinct focal names; each world names a pair. Names are short, single-word forms | No declared naming traditions, aliases, nicknames, full names, similar-name disambiguation, or renaming conditions. Do not infer culture or identity from a name. |
| Cast and relationships | Two named focal characters per world, sometimes unnamed supporting roles. Siblings, coworkers, family businesses, and institutional relationships appear | No designed ensemble-cast or multi-generation condition; relationship types are not systematically labeled. |
| Setting | Ten local settings: archive, orchard, ferry, radio station, court, cinema, garden, map office, workshop, restaurant | No explicit era, geographic, cultural, or multi-location balance. |
| Story situation | Genre frames add departure, disappearance, magical vows, infrastructure changes, reunion, unexplained voices, supply delays, mistaken identity, expedition, and surveillance | Underlying passages still repeat uncertain belief plus practical prerequisite. Each genre reuses its frame across five worlds; no claim of independent plot diversity. |
| Conflict and stakes | Suspicion, access, permission, evidence, commitments, and resource constraints | Intensity and consequences are not labeled or balanced. Different objects often occupy the same narrative role. |
| Knowledge interpretation | Each world supplies a salient state, uncertain belief, optional context, incidental detail, and accepted update | Very small fact sets. No substantial conflicting documents, long chronology, large relationship graph, or deliberately difficult entity resolution. |
| Character knowledge | Attributed uncertain beliefs and a few explicit differences in access to information | No systematic secret-sharing or multi-character knowledge-state matrix. |
| Instruction specificity | 25 explicit and 25 loose initial requests; five each per family | All explicit cases use worlds 01–05 and all loose cases use 06–10. Specificity is confounded with world; these are not matched wording comparisons. |
| Conversation | 40 one-turn cases, five two-turn brainstorming cases, five three-turn KB cases; follow-ups are fixed | No responsive clarification, negotiation, or branching user policy. Update feedback remains more explicit even in loose-initial-request cases. |
| Source/context length | Base passages: 64–77 whitespace-separated words. Genre-expanded source passages: 100–124 words. Revision seeds still have two sentences | No long chapters, long-context retrieval, or substantial multi-document input conditions. |
| Requested prose length | Explicit prose cases request 120–220 words; loose cases omit a numerical target | No designated long-form condition. Output length is not known before execution. |
| Output destination | F1: reply; F2: manuscript file; F3: reply; F4: KB files; F5: five replies and five files | No matched reply-versus-file comparison with all other variables fixed. |
| KB representation | Five F5 file fixtures: three linked Markdown KBs, two flat Markdown KBs. Five other F5 cases supply notes in the prompt | No structured-record or hybrid KB fixtures. Flat and linked layouts occur in different worlds, not matched pairs. |
| KB size/navigation | Linked fixtures contain an index plus canon, interpretations, and story-context pages; flat fixtures contain one page. File fixtures total 73–91 whitespace-separated words including Markdown | Still tiny and templated; genre context is included, but no deep hierarchy or large wiki. |
| KB maintenance | Five F4 cases build a KB, then receive a draft-only suggestion and an accepted update | These do not start with a substantial existing KB. No long edit history or cumulative maintenance workload. |
| Provenance and source diversity | All 50 scenarios and ten worlds are synthetic project-authored fixtures | No human, half-synthetic, synthetic-fanfic, or external-dataset-derived cases in this collection yet. Acquired datasets remain separate. |

## World inventory

Every world supplies one case in each family. The suffix `01`, for example,
identifies F1-01 through F5-01. Prose-style strings below are copied from the world records. They describe prose
technique separately from genre and are not claims about generated outputs.

| Suffix | World | Named pair | Setting | Prose-style field | Source words |
|---|---|---|---|---|---:|
| 01 | The Tidal Archive | Mara, Ilan | a tidal archive | restrained, concrete prose | 69 |
| 02 | The Glass Orchard | Suri, Aven | a glass orchard | spare prose | 70 |
| 03 | The Night Ferry | Oren, Leda | a night ferry | measured pacing with sustained tension | 77 |
| 04 | Winter Radio | Nessa, Tomas | an isolated radio station | dialogue-led prose with understated tension | 75 |
| 05 | The Copper Trial | Dara, Sen | a provincial courthouse | formal, precise prose | 66 |
| 06 | The Last Cinema | Emil, Ruth | a closing neighborhood cinema | warm prose with comic melancholy | 66 |
| 07 | The Salt Garden | Yara, Belen | a drought-struck communal garden | sensory description grounded in physical detail | 64 |
| 08 | The Paper Border | Kavi, Noor | a border map office | dry, bureaucratic language | 67 |
| 09 | The Hollow Clock | Pia, Ren | a clockmaker's workshop | intimate, restrained prose | 67 |
| 10 | The Blue Kitchen | Ada, Milo | a family restaurant after closing | understated, conversational prose | 69 |

## Scenario matrix

Each row accounts for five scenarios, one per family. These are stored condition
labels. An asterisk means the loose F1 prompt is now a general scene request and
does not enforce the named subtype. `Update` means build first, then update.

| Suffix | Initial request | F1: prose | F2: files | F3: planning | F4: KB creation | F5: KB use |
|---|---|---|---|---|---|---|
| 01 | explicit | scene | local revision | alternatives | construction | flat Markdown file |
| 02 | explicit | continuation | explore then write | feedback | update | notes in prompt |
| 03 | explicit | dialogue | local revision | alternatives | construction | linked Markdown files |
| 04 | explicit | reflection | explore then write | feedback | update | notes in prompt |
| 05 | explicit | action | local revision | alternatives | construction | linked Markdown files |
| 06 | loose | scene* | explore then write | feedback | update | notes in prompt |
| 07 | loose | continuation* | local revision | alternatives | construction | flat Markdown file |
| 08 | loose | dialogue* | explore then write | feedback | update | notes in prompt |
| 09 | loose | reflection* | local revision | alternatives | construction | linked Markdown files |
| 10 | loose | action* | explore then write | feedback | update | notes in prompt |

## Genre assignment

Each cell identifies the requested genre for that family/suffix. Genre context is
present in the supplied passage or KB notes, not only in metadata. Related genre
versions retain their parent world in source lineage. Genres are balanced across
families and represented in both specificity groups; other dimensions are not
fully balanced or independently manipulated.

| Suffix | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| 01 | literary fiction | fantasy | romance | historical fiction | adventure |
| 02 | mystery | science fiction | horror | comedy | thriller |
| 03 | fantasy | romance | historical fiction | adventure | literary fiction |
| 04 | science fiction | horror | comedy | thriller | mystery |
| 05 | romance | historical fiction | adventure | literary fiction | fantasy |
| 06 | horror | comedy | thriller | mystery | science fiction |
| 07 | historical fiction | adventure | literary fiction | fantasy | romance |
| 08 | comedy | thriller | mystery | science fiction | horror |
| 09 | adventure | literary fiction | fantasy | romance | historical fiction |
| 10 | thriller | mystery | science fiction | horror | comedy |

## Final generation review

All items remain pending. Use this inventory when designing and accepting the
final collection; it does not authorize generation or larger evaluation runs.

- [ ] Agree on genre, tone, style, form, naming, cast, setting, and knowledge-complexity labels separately. Preserve unknowns rather than guess.
- [ ] Choose coverage targets and useful intersections, such as genre × task or naming complexity × KB use. A full Cartesian product is not required.
- [ ] Review actual briefs and source content against their labels; resolve the loose F1 subtype mismatch before treating subtype counts as coverage.
- [ ] Add narrative structures beyond the repeated uncertain-belief/practical-restriction pattern.
- [ ] Design larger and more varied KBs, with reviewed factual content and plausible organization. Distinguish representation from content and retrieval difficulty.
- [ ] Decide which comparisons need matched variants; keep their shared source lineage visible and count them as related cases.
- [ ] Review instruction/label alignment: open choices must not become hidden grading requirements. Clarification-dependent cases need a responsive user policy.
- [ ] Assign source groups to development, grader calibration, and final evaluation before deriving variants. Audit external-benchmark overlap separately.
- [ ] Recount the final generated collection and record human acceptance plus any remaining gaps.
