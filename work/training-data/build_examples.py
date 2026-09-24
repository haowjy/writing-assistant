"""Authored worked examples; no model calls. Rebuilds only work artifacts."""
import json
from pathlib import Path

ROOT = Path('/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation')
SCENARIOS = {s['id']: s for s in json.load(open('data/scenarios/development.json'))}
examples = []
notes = {}

def add(case_id, response, files, evidence, why, note=''):
    s = SCENARIOS[case_id]
    required = [c for c in s['labels']['checks'] if c.get('required')]
    assert set(evidence) == {c['id'] for c in required}, case_id
    satisfies = []
    for c in required:
        value = evidence[c['id']]
        met, text = value if isinstance(value, tuple) else (True, value)
        satisfies.append({'check_id': c['id'], 'met': met, 'evidence': text})
    examples.append({'case_id': case_id, 'family': s['family'],
                     'channel': 'file' if files else 'reply', 'brief': s['visible']['brief'],
                     'response': response, 'files': files, 'satisfies': satisfies,
                     'why_it_is_good': why})
    notes[case_id] = note

add('F1-01', '''“Then tell me which shelf.”

Mara stood between Ilan and the stairs. He had brought a plane down with him; curls of wet wood clung to its blade.

“Above the tide line. I told you.”

“That isn't a shelf.”

He put the plane on the table. “You said I sold it.”

She had said it at noon, loudly enough for the harbor office to hear. Now she wanted him to offer something she could check. The upper room was locked. Only the retired keeper had the key.

“If the ledger's there—”

“When you find it,” he said, “will you tell them?”

Mara picked a shaving off the blade. She had been sorting her own papers into a box beneath this table, taking a few home each night.

“I haven't found it.”

“No.” He held out his hand for the plane.

She passed it over, then moved aside. He stopped at the foot of the stairs.

“Will you?”

She set the shaving on top of her box.''', {}, {
'continuity': '“Only the retired keeper had the key.” preserves access; “If the ledger\'s there—” and “I haven\'t found it.” leave Mara\'s accusation unproved and the ledger unlocated.'},
'The demand for a shelf turns a vague accusation into something Mara could verify. Ilan asks for a public correction rather than reassurance; the shaving placed on her packed box connects her intended departure to the work she is leaving him.')

add('F1-02', '''No answer came from the vault office. Suri kept the receiver against her ear until Aven pointed to its dark indicator.

“They've closed.”

She set it down. The original birdsong was still sealed in the municipal vault. Asking to inspect it had not put it within reach.

“Tell me about the day again.”

“The disappearance?”

“You said you heard the gate.”

“I said I saw it open.”

Suri unfolded the anonymous note on the workbench. She had remembered him saying heard. She could almost supply his voice. But since the accident there were high sounds he could not hear, and she had begun filling those gaps for him without asking.

“Write it down,” she said.

“What you remember?”

“What you do.”

He took a pencil. She turned over a clean sheet for herself. Beyond the bench, birdsong played among the glass trees. Perhaps the song had been damaged. She had no test to set beside that thought.

When Aven reached for her sheet, she covered it with her hand.

“Separately first.”''', {}, {
'continuity': '“The original birdsong was still sealed in the municipal vault.” preserves the source; “Perhaps the song had been damaged. She had no test to set beside that thought.” leaves cause and Aven\'s responsibility unresolved.'},
'The unanswered call continues directly from Suri\'s request for access. The correction from “heard” to “saw” gives the mystery a small, usable discrepancy without making Aven\'s hearing loss proof of guilt; separate written accounts create the next action.')

add('F1-03', '''“If I ask the captain, what may I tell him?” Oren said.

“That I need the sailing time.”

“He'll ask where you're going.”

“Will he? Or will you?”

Oren held the unsigned ticket against the rail. A creditor, he had thought when she boarded. He had already supplied a man on the quay, an unpaid sum. Leda had supplied neither.

“Bell Quay first,” he said. “Then the island.”

“I know the stops.”

“The time has changed. The captain hasn't told anyone.”

From below the deck came three words: “Until you return.” The ferry had been repeating that much of a vow after dark. Oren waited for more. Nothing followed.

“Was that someone?” Leda asked.

“The boat.”

“Whose words?”

“I don't know.”

“Then please don't put mine beside them.”

He lifted the ticket away from the rail.

“I can ask about the time without showing him this.”

“Can you ask without pointing?”

Oren looked toward the wheelhouse, then back at his own hand. He lowered it.''', {}, {
'continuity': '“The captain hasn\'t told anyone.” keeps the revised time private; “A creditor, he had thought” and “Leda had supplied neither.” explicitly distinguish Oren\'s conjecture from her motive.'},
'The dialogue makes privacy a sequence of concrete negotiations: naming a destination, showing a ticket, pointing. The ferry\'s incomplete vow interrupts that negotiation without explaining Leda, and Oren\'s lowered hand supplies a small outward change.')

add('F1-04', '''To answer would take one press of Nessa's thumb.

“Record,” Tomas said.

“I remember.”

The transmit button sat beneath her hand. They had agreed to wait for another transmission, but somewhere a person had asked for rescue. She moved her thumb to the desk.

“Mother used that frequency,” he said.

“You were a child.”

“I could still remember.”

“Yes. You could.”

She opened a blank entry in the log. The island's autonomous systems had synchronized the public records again overnight. Was the missing frequency a consequence of the update? She had no way to tell from the empty line.

“Turn the antenna?” Tomas asked.

“Not until the ice thaws.”

“Then what do we write?”

Nessa nearly typed their mother's name. It would give the entry a heading, something less bare than a number. But the voice had been a stranger's to her, and Tomas had remembered a channel, not identified a speaker.

“Frequency. Time. Exact words.”

She armed the recorder and turned the screen so he could see.

“You take the time.”''', {}, {
'continuity': '“Not until the ice thaws.” preserves the antenna restriction; “Yes. You could.” qualifies Tomas\'s recollection; “the voice had been a stranger\'s to her” avoids identifying the caller.'},
'The thumb moving off the transmit button stages hesitation before the decision to record. Nessa almost gives an uncertain entry a familiar name, then chooses three observable fields; the last instruction gives Tomas a role without accepting his memory as fact.')

add('F1-05', '''Halfway through tying the deposition bundle, Dara caught her letter in the cord.

Sen put a finger on the knot.

“That isn't testimony.”

“I know.”

She loosened the loop. The envelope came free, creased across the address. Sen knew that its writer had asked to see her again; he had not read the invitation. She turned it face down beneath her blotter.

“Will you need someone to take your place?” he asked.

“For an afternoon?”

“You haven't said how long.”

Dara pulled the cord too tightly. The top sheet buckled. Before she could smooth it, Sen offered his hand to hold the bundle square.

Across the room, the evidence chest remained locked. The copper seal was inside; the expert had examined only a drawing. Sen's claim that it had been copied still awaited that examination.

“I haven't agreed to go,” she said.

His hand stayed under the papers.

“Shall I let go?”

She looked at his fingers, clear of the knot now.

“Not yet.”

Dara began the loop again.''', {}, {
'continuity': '“The evidence chest remained locked. The copper seal was inside; the expert had examined only a drawing.” and “Sen\'s claim ... still awaited that examination.” preserve the unresolved allegation and access restriction.'},
'The private letter physically interrupts an official task. “Shall I let go?” remains a practical question while allowing Dara\'s “Not yet” to carry the attraction; neither the proposed reunion nor the seal\'s authenticity is settled.')

add('F2-02', 'The orchard scene is saved in drafts/scene.md.', {'drafts/scene.md': '''Under Suri's fingertip, the crack divided a glass leaf.

“Don't lift it,” Aven said.

She withdrew her hand. The recorder played on. A maintenance cart passed between the rows, following the island's painted service line.

“Did the update touch this machine?”

“Public records changed. I don't know about this.”

“You maintain it.”

“That isn't a test.”

Suri looked at him. She had been ready to make his answer stand for one. He could no longer hear the highest notes, but that did not tell her whether the recording had lost them.

“The original's in the municipal vault,” she said. “Still sealed.”

He nodded.

“Then we compare them.”

“When we're allowed.”

She fetched the request form from the workbench. In the box for purpose she wrote DAMAGE, stopped, and added INVESTIGATION. There was no box for suspect.

Aven held the paper flat while she signed it. She left the cause blank and addressed the form to the vault office.

“May I inspect the original recording?” she read aloud.'''+ '\n'}, {
'saved-scene': 'drafts/scene.md contains “Under Suri\'s fingertip, the crack divided a glass leaf.”',
'edit-scope': 'Only “drafts/scene.md” is written; “notes/source.md” remains byte-for-byte unchanged in the replay.',
'continuity': '“The original\'s in the municipal vault ... Still sealed.” preserves access; “that did not tell her whether the recording had lost them” keeps Aven\'s hearing loss distinct from evidence of damage.'},
'Aven\'s “That isn\'t a test” stops Suri from treating his job as evidence against him. The form converts suspicion into a request she is authorized to make, and its blank cause field holds the uncertainty without a closing explanation.')

add('F2-04', 'Written to drafts/scene.md; the caller remains unidentified.', {'drafts/scene.md': '''“Don't use my name on the radio.”

Tomas looked up from the log. “I wasn't going to answer.”

“I know.”

Nessa pulled the microphone nearer, then pushed it back to its place. The last time she had heard her own voice, it had come from the empty room behind them. Tomas had only her account of it.

“You said it sounded exactly like you.”

“It did.”

“Say something now.”

“No.”

He shut the log. She opened it again to the missing frequency.

“Mother used that channel,” he said.

“You think she did.”

“I remember the visit.”

“That isn't the same thing.”

The rescue voice had been unfamiliar. Nessa would write that down, too.

“Could we turn the antenna away?”

“It's iced fast. We wait for a thaw.”

“Or we leave this room.”

She checked the recorder. Ready. They had agreed to record the next transmission before replying.

“Bring your chair here.”

He dragged it beside hers. Its legs squealed across the floor. Both of them stopped to listen.'''+ '\n'}, {
'saved-scene': 'drafts/scene.md begins “Don\'t use my name on the radio.”',
'edit-scope': 'The sole written path is “drafts/scene.md”; the replay retains “notes/source.md” unchanged.',
'continuity': '“It\'s iced fast. We wait for a thaw.” preserves the antenna constraint; “You think she did.” qualifies Tomas\'s memory; “The rescue voice had been unfamiliar.” preserves Nessa\'s lack of recognition.'},
'The ban on using Nessa\'s name makes the reported horror affect a present decision. Small reversals with the microphone and log keep the exchange physical, while the chair\'s ordinary squeal becomes a sound both siblings must assess.')

add('F2-05', 'Expanded the first sentence in drafts/scene.md and retained the final sentence verbatim.', {'drafts/scene.md': '''<prose>
Dara set her pen across the inkwell before the last word could dry. She had written copied without an attribution.

“Your statement,” she said to Sen, “is that the seal was copied.”

“Yes.”

“On what examination?”

“The expert saw it.”

“The drawing.”

He looked toward the evidence chest. It remained locked; the original copper seal was inside. Dara waited with her hand beside the pen.

“The drawing,” Sen said.

She inserted alleges above the line. Paper deliveries had stopped with the strike, and this sheet would have to serve. The small correction left scarcely room for a date.

From the public benches came a question about the witness's family. Another voice answered before Dara could make out the words. The judge had forbidden that discussion outside court; inside, everyone appeared determined to have it at once.

She laid a clean scrap over the wet correction and raised her eyes to the bench.
</prose>
The judge called for silence.
'''}, {
'preserve-ending': '“The judge called for silence.” survives verbatim, after </prose>.',
'saved-scene': 'drafts/scene.md contains “Dara set her pen across the inkwell before the last word could dry.” within the requested delimiters.',
'edit-scope': 'Only “drafts/scene.md” changes; “notes/source.md” is preserved.',
'continuity': '“It remained locked; the original copper seal was inside.” preserves the chest state; “The drawing,” Sen said. and “She inserted alleges above the line.” keep copying an allegation.'},
'The expansion gives setting down a pen a reason: Dara must qualify the record. Scarce paper makes that correction materially awkward, and the rising voices lead directly into the protected final sentence.')

add('F2-07', 'For this revision, I assumed the scene should stop before the inspection and make the expedition a competing demand. Saved it in drafts/scene.md.', {'drafts/scene.md': '''Beyond the garden wall lay a road Yara had never walked. Belen had brought the expedition list into the beds and was trying to flatten it against a dry water barrel.

“We could take this one.”

“We couldn't.”

“It's empty.”

Yara pointed to the two full barrels by the seedlings. “And those stay here.”

“I wasn't asking for those.”

She knelt beside the buried jar. Dust settled on the leaves. Its rim gave her nothing to measure a well by, only a circle of clay with earth packed inside.

“If there's water below it,” Belen said, “you could leave.”

“If.”

Belen folded the list. “You could ask someone else to mind the beds.”

That had not appeared among the equipment or provisions. Yara brushed dust from a seedling with the side of her finger.

“The council discusses digging after the inspection. We don't move the jar before then.”

“I know.”

“Put another name beside mine.”

“For the expedition?”

“For the watering.”

Belen opened the list again and turned it over.'''+ '\n'}, {
'saved-scene': 'drafts/scene.md contains “Beyond the garden wall lay a road Yara had never walked.”',
'edit-scope': 'The only write targets “drafts/scene.md”; “notes/source.md” remains unchanged.',
'continuity': '“And those stay here.” protects the two full seedling barrels; “If.” qualifies the well; “We don\'t move the jar before then.” keeps it in place until the council discussion. The expedition remains in preparation.'},
'The empty barrel creates an argument without spending the protected water. Belen\'s list changes use at the end, letting Yara take a practical step toward leaving without finding a convenient well or abandoning her obligations.')

add('F2-10', 'I treated the note and the recipe as separate uncertainties, and used Ada’s viewpoint. The scene is saved in drafts/scene.md.', {'drafts/scene.md': '''Two saucers for tasting; a third for the anonymous note.

Ada moved the third away from the stove.

“You'll get soup on it.”

“I'm trying to remember,” Milo said.

“With a ladle?”

He set it down. “You were here when they disappeared.”

“I came back here afterward.”

“You made soup.”

“I washed bowls.”

The little test pan began to spit. Ada lowered the flame. Their aunt would be away until Sunday; there was nobody to settle even the question of the crossed-out ingredient. Milo thought it was the secret. Ada could think of other reasons to cross something out.

“Does the note say soup?” she asked.

“No.”

“Then leave the card out of it.”

He pushed the recipe toward her and drew the note closer. She put a spoonful into each tasting saucer. Neither reached for one.

“What did you wash first?” he asked.

“The bowls.”

“Before the plates?”

Ada looked at the rack. That was how she did it now. She could not remember whether it had always been so.

She turned off the burner. Tomorrow's regular menu was still on the board.'''+ '\n'}, {
'saved-scene': 'drafts/scene.md contains “Two saucers for tasting; a third for the anonymous note.”',
'edit-scope': 'Only “drafts/scene.md” is written; “notes/source.md” is unchanged.',
'continuity': '“Their aunt would be away until Sunday”; “Milo thought it was the secret”; and “Tomorrow\'s regular menu was still on the board.” preserve the absence, uncertain recipe, and menu agreement. The small batch is private; neither the note\'s author nor the disappearance is explained.'},
'The third saucer gives the anonymous note a physical place away from the cooking. Ada challenges Milo\'s memory before discovering that her own answer may come from habit, so the uncertainty grows through a specific question about washing order.')

add('F3-02', '''1. **Test the boundary of a report.** Suri asks Aven to wait outside an empty room while she enters and speaks. His inability to hear high notes makes “I heard nothing” an inadequate result, so they must agree beforehand on what he can reliably report. Stop at the threshold, with the procedure disputed. This puts Suri's willingness to risk herself against Aven's willingness to endorse the test; neither the voices nor the cracked tree is explained.

2. **Put the accusation in writing.** While requesting access to the sealed original recording, Suri must decide whether to name Aven as a possible cause of the damaged song. Aven asks to see the request before she submits it. A refusal would threaten their working relationship; agreement forces her to distinguish observation from suspicion. The vault stays closed, and the argument exposes how little evidence either has.

3. **Arrange a night watch.** Suri proposes that they stay together in the orchard until morning. Aven will agree only if she stops asking him to imitate the voice she heard. The bargain leaves them physically close but unable to use the reassurance Suri wants. Recorded birdsong can continue while they prepare two places to sit; no new voice needs to speak, and Aven still has not witnessed the phenomenon.''', {}, {
'three-directions': '“Test the boundary of a report,” “Put the accusation in writing,” and “Arrange a night watch” offer exactly three choices: disputed testing, accountable accusation, and conditional companionship, each with a consequence and an unresolved cause.'},
'The options differ in what Suri risks: her safety, her written credibility, or a condition of companionship. Each direction names a playable decision and a stopping point, so the writer can begin a scene without first inventing a solution to the mystery.',
'FIRST BRIEF ONLY. The followup asks for all three directions to reduce physical action and increase interpersonal tension, remain distinct, and explain the changes. This response does not perform that revision in advance.')

add('F3-05', '''1. **A place held open.** Dara proposes asking the expedition to hold their places until the hearing can resume. Sen wants to finish preparing their equipment instead of making a request that advertises their uncertainty. Their choice is whether to disclose the delay and risk losing the invitation. The seal remains locked away, the expert still has only a drawing, and departure becomes a practical negotiation rather than a solved case.

2. **Prepare to travel separately.** Dara offers to remain for the examination while Sen takes responsibility for their expedition preparations. Sen must decide whether accepting that division makes his copying allegation look convenient. Let them negotiate who carries which supplies and messages, then stop before either leaves. This tests their willingness to separate while keeping both the verdict and the expedition's readiness unsettled.

3. **Draw a boundary around useful information.** During preparations, another prospective traveler asks about the witness's family, claiming the answer would help choose the route. Dara must refuse under the judge's prohibition, even if that costs them assistance. Sen can support her refusal or try to redirect the question. The scene would test what they will sacrifice to join the journey; the new request is a proposed complication, not proof of anything about the seal.''', {}, {
'three-directions': '“A place held open,” “Prepare to travel separately,” and “Draw a boundary around useful information” give exactly three developed directions: negotiate delay, divide responsibilities, or refuse prohibited disclosure; each preserves the locked seal and unresolved allegation.'},
'The adventure invitation creates three different costs: losing a place, separating, or refusing help. The third option uses the existing privacy prohibition as an obstacle, while explicitly keeping the proposed traveler\'s question from becoming new canon.')

add('F3-07', '''For these options, I'm assuming you want the pressure to come from Yara's possible departure while the well remains unconfirmed.

1. **Ask for a replacement.** Before the morning inspection, Yara asks Belen who could take over the watering rota. Belen answers as though the question were temporary, forcing Yara to decide whether to correct that assumption. The consequence is a first, partial admission that she may leave. The jar stays untouched, and the two seedling barrels remain reserved.

2. **Refuse a useful hope.** Belen asks Yara to speak for the proposed well at the council discussion after inspection. Yara will report the jar but will not call it evidence of water. Her restraint can sound to Belen like an excuse to leave. This gives them a dispute about what they owe the garden without confirming the well or beginning an unauthorized excavation.

3. **Choose a small job together.** Yara asks Belen to help repair a leaking watering can before the inspection. Belen wants to discuss the jar; Yara keeps returning to the seam. Finishing the repair gives the garden something it needs while allowing the question of her departure to remain unspoken. The scene can turn on whether she puts the repaired can on her own hook or leaves it for whoever comes next.''', {}, {},
'The opening states the main unstated choice: relationship pressure rather than a discovery plot. The three directions respectively invite an admission, a disagreement over evidence, and an evasive shared task; the hook for the watering can gives the quietest option a visible ending.',
'No required checks are declared for this case. The empty satisfies list reflects the source contract, not missing coverage.')

add('F4-01', 'Created kb/index.md and kb/ledger.md with linked continuity notes and source citations.', {
'kb/index.md': '''# The Tidal Archive

Next-chapter reference: [Ledger, access, and character knowledge](ledger.md).

## Working conditions

The year is 1936. Letters, newspapers, and radio carry news to the archive. A regional strike has delayed ordinary deliveries; Mara and Ilan must use existing supplies.

Source: `source/chapter.md`.
''',
'kb/ledger.md': '''# Ledger and access

[Archive index](index.md)

## Current restriction

Only the retired keeper has the upper-room key. Mara cannot open that room. The chapter ends before anyone enters it; the ledger's location has not been verified.

## Accounts and knowledge

Mara keeps the harbor archive. At noon she accuses Ilan of selling the brass ledger, without evidence. Treat the accusation as Mara's suspicion, not a sale established by the narrator.

Ilan repairs the flooded shelves. He says he moved the ledger above the tide line; his statement has not been independently confirmed.

## Continuing the chapter

Access requires the keeper's key. Any discovery of the ledger would be a new event, not an existing fact. Neither account currently settles its whereabouts.

Source for the restrictions and accounts: `source/chapter.md`.
'''}, {
'wiki-links': '“[Ledger, access, and character knowledge](ledger.md)” links the index to the topic; “[Archive index](index.md)” returns to it. Both local destinations exist.',
'wiki-index': 'kb/index.md begins “# The Tidal Archive” and provides the topic link.',
'wiki-only': 'Only “kb/index.md” and “kb/ledger.md” are written; “source/chapter.md” is unchanged.',
'consequential-state': '“Only the retired keeper has the upper-room key. Mara cannot open that room.”',
'qualified-belief': '“At noon she accuses Ilan of selling the brass ledger, without evidence. Treat the accusation as Mara\'s suspicion, not a sale established by the narrator.”'},
'The topic page separates the locked-room restriction from two competing accounts. The index preserves the supply shortage because it constrains the next chapter, while omitting the unused scarf; reciprocal links keep the small wiki navigable.')

add('F4-04', 'The radio-station wiki starts at kb/index.md; its topic page separates present restrictions from uncertain reports.', {
'kb/index.md': '''# Winter Radio

For the next chapter, see [Station state and unresolved reports](station.md).

Nessa runs the mountain station with her brother Tomas. A rescue signal has arrived on a frequency absent from their log.

Source: `source/chapter.md`.
''',
'kb/station.md': '''# Station state and unresolved reports

[Winter Radio index](index.md)

## Equipment and agreed action

The antenna cannot be turned until the ice thaws. Nessa and Tomas have agreed to record the next transmission before replying.

## Voice and memory

Nessa has never heard the rescue voice before. Tomas remembers their mother using this channel during a childhood visit, but his memory may be mistaken. Neither the caller's identity nor the channel's history is established by his recollection.

## Warning and observation

Someone has warned Nessa that a stranger will arrive before morning asking for information. The warning's source and the stranger's intentions are unknown.

Tomas has seen signs of someone watching the entrance. The source does not establish a connection between that observation, the warning, and the rescue signal.

Source for all recorded facts and attributed reports: `source/chapter.md`.
'''}, {
'wiki-links': '“[Station state and unresolved reports](station.md)” and “[Winter Radio index](index.md)” connect both existing pages in both directions.',
'wiki-index': 'kb/index.md contains “# Winter Radio” and the linked next-chapter reference.',
'wiki-only': 'Only “kb/index.md” and “kb/station.md” change; “source/chapter.md” remains untouched.',
'consequential-state': (False, 'UNMET ON FIRST TURN: the label requires “The antenna points east; the voice is still unidentified.” That state is authorized only in followup 2. The correct current wiki says “The antenna cannot be turned until the ice thaws.”'),
'qualified-belief': '“Tomas remembers their mother using this channel during a childhood visit, but his memory may be mistaken.”',
'accepted-update': (False, 'UNMET ON FIRST TURN: the accepted eastward turn occurs only in followup 2; applying it now would leak later instructions. “The antenna cannot be turned until the ice thaws.” is the first-brief state.')},
'The wiki gives equipment, memory, and warning separate sections so the writer can find the constraint without treating every report as fact. “The source does not establish a connection” blocks a tempting invented conspiracy; the first-turn pages do not anticipate the later thaw.',
'FIRST BRIEF ONLY. Followup 1 tests rejection of the draft-only proposal that Nessa recognizes her mother. Followup 2 authorizes thawing and the eastward antenna turn, and tests updating current state while retaining relevant history and an unidentified voice. Two final-state required labels are intentionally unmet at turn 0; they are temporally inapplicable, not broken for the full conversation.')

add('F4-09', 'For the wiki’s structure, I assumed you want separate pages for the watch problem and the reunion decision. Both are linked from kb/index.md.', {
'kb/index.md': '''# The Hollow Clock

- [Watch, access, and evidence](watch.md)
- [Invitation and character knowledge](invitation.md)

These pages record the chapter's current state for continuation. Source: `source/chapter.md`.
''',
'kb/watch.md': '''# Watch, access, and evidence

[Index](index.md) · [Invitation](invitation.md)

Pia inherited a watch that stops whenever Ren enters the workshop. A collector holds the uncommon square key required to open it.

Ren says the spring was replaced last winter. The repair book has a missing page; the chapter does not verify his account or explain the omission.

Pia suspects a trick, without proof. Neither the mechanism nor anyone's intent has been established. She has agreed to record stopping times before drawing conclusions.

Source: `source/chapter.md`.
''',
'kb/invitation.md': '''# Invitation and character knowledge

[Index](index.md) · [Watch](watch.md)

Pia has received a letter from an old love who wants to meet again. Ren knows about the invitation but has not read the letter.

A possible reunion complicates Pia's feelings about remaining at the workshop. She has made no decision. Do not give Ren knowledge of the letter's contents beyond the invitation he already knows about.

Source: `source/chapter.md`.
'''}, {
'wiki-index': 'kb/index.md contains “# The Hollow Clock” and links to “watch.md” and “invitation.md”.',
'wiki-only': 'Only “kb/index.md”, “kb/watch.md”, and “kb/invitation.md” are written; “source/chapter.md” is unchanged.'},
'The watch page distinguishes a repeated event from Pia\'s unproved explanation. The invitation page records exactly what Ren has and has not seen, and its link to the watch page lets a writer move between the emotional decision and the practical obstacle.')

add('F5-02', '''The scene stops before the visitor is identified.

<prose>
Whoever stood beyond the orchard gate had kept out of the light.

Suri caught Aven's sleeve before he stepped outside.

“Did you see someone?”

“Something moved.”

“You said they'd been watching.”

“I said there were marks by the entrance.”

She released him. The warning had said before morning. It had not given a name, or a reason she should trust it.

A high chime sounded from the gate. Aven did not turn. Since the accident he could not hear that pitch; Suri touched his wrist and pointed.

“Stay where I can see you,” she said.

“What do we tell them?”

“Nothing about the recording.”

“The original's sealed in the municipal vault.”

“I know.”

She still suspected a damaged song, and still had no evidence that made Aven responsible. Sending him to the gate would settle neither question.

Suri moved beside him, leaving the gate shut.

“Who are you here to see?”

She waited with her hand on the latch.
</prose>''', {}, {
'continuity': '“The original\'s sealed in the municipal vault.” preserves location and access; “still had no evidence that made Aven responsible” keeps the suspected damaged song from becoming an accusation proved.'},
'The correction from a watcher to marks at the entrance narrows the evidence under pressure. Aven\'s missed chime makes his hearing limitation matter to the action, and Suri chooses to stand beside him rather than use him to test the stranger.')

add('F5-03', 'Saved the ferry scene in drafts/scene.md.', {'drafts/scene.md': '''One more crossing, Oren had told himself yesterday. Now Leda was holding out the departure slate for him to change.

“Have you asked?”

“Not yet.”

“The old time's still up.”

He took the chalk. Only the captain knew the revised sailing time. Oren rubbed out the old figures with his sleeve and wrote PLEASE ASK instead.

“Ask whom?” Leda said.

He put the chalk down.

“You're quiet lately.”

“There's work.”

She looked at the blank space beneath his notice. “There's this.”

He had decided, without being told, that a creditor might explain her journey. It had been easier to imagine what she was leaving than to say what he wanted to leave.

“Bell Quay first,” he said. “That's unchanged.”

“I didn't ask about the stops.”

“No.”

Leda propped the slate against the wall. Oren took it down again. He could at least spare the next person a useless question.

He added THE CAPTAIN below PLEASE ASK, then carried the slate to its hook.'''+ '\n'}, {
'preserve-kb': 'Only “drafts/scene.md” is written; all four “kb/” files remain byte-for-byte unchanged.',
'continuity': '“Only the captain knew the revised sailing time.” and “Bell Quay first” preserve the practical facts; “He had decided, without being told, that a creditor might explain her journey.” explicitly marks his guess.'},
'The incomplete sign exposes Oren\'s withdrawal through a job he can actually do. His correction at the end helps one future passenger without pretending he has resolved either Leda\'s motive or his own decision to leave.')

add('F5-05', 'The courthouse scene is in drafts/scene.md, with the wiki preserved.', {'drafts/scene.md': '''It was from beneath the witness bench that the courthouse spoke.

“By my own hand.”

Dara laid a sheet on the bench and wrote the words. After dark, the building repeated fragments of an old vow. It had never supplied her with a name.

“An admission,” Sen said.

“Whose?”

He looked toward the locked evidence chest.

“The seal was copied.”

“You allege that. The expert has seen only a drawing.”

The original copper seal remained in the chest. Dara drew a line beneath the phrase she had recorded. Sen leaned closer, and she covered the empty space where a speaker's name would ordinarily go.

“Then why write it?” he asked.

“Because we heard it.”

“And if it speaks again?”

“I'll write that, too.”

He lowered his voice. “The witness's family—”

“Here, in court. That stays here.”

The bench gave a small wooden click. Dara lifted her pen, keeping its point above the paper.

This time she left room for another line.'''+ '\n'}, {
'preserve-kb': 'The replay writes only “drafts/scene.md”; “kb/index.md”, “kb/canon.md”, “kb/interpretations.md”, and “kb/story-context.md” are unchanged.',
'continuity': '“The original copper seal remained in the chest.” follows the locked-chest reference; “You allege that. The expert has seen only a drawing.” prevents a verdict from an unidentified vow.'},
'The building\'s words sound like testimony, giving Dara a professional reason to resist interpreting them. Her hand covers the name field while her pen remains ready to record more; those two actions separate evidence collection from attribution.')

add('F5-07', 'I chose Yara’s viewpoint and assumed the reunion should remain undecided. Saved the scene in drafts/scene.md.', {'drafts/scene.md': '''By the time Belen asked about the letter, Yara had watered the same seedling twice.

“Is there an answer to send?”

Yara tipped the can upright. “Not yet.”

Belen offered her a rag for the spill. She took it by the dry corner; for a moment they were both holding it.

“What did the letter say?”

“That we might meet.”

“I knew that much.”

“Yes.”

The folded sheet pressed against her ribs inside her pocket. She could read out the proposed day and place. Instead, she tucked the corner of the letter down.

Belen let go of the rag.

“If there's a well, things could be different here.”

“We don't know there's a well.”

“The council will discuss digging after the inspection.”

“And these two barrels stay for the seedlings.”

“I wasn't asking you to spend them.”

Yara knelt to wipe the water from the rim of the tray. Belen crouched opposite her and lifted one corner so she could reach beneath.

“Thank you,” she said.

She put the rag between them instead of in Belen's hand.'''+ '\n'}, {
'preserve-kb': 'Only “drafts/scene.md” is written; the single-page “kb/index.md” remains unchanged.',
'continuity': '“We don\'t know there\'s a well,” “these two barrels stay for the seedlings,” and “Not yet.” preserve the unconfirmed well, seedling reserve, and undecided reunion. Belen asks “What did the letter say?” rather than knowing its text.'},
'The shared grip on the rag lets physical proximity complicate a conversation about someone else\'s invitation. Yara answers practical questions more readily than personal ones, and setting the rag between them changes the earlier contact without naming an emotion.')

examples.sort(key=lambda e: e['case_id'])
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'worked-examples').mkdir(exist_ok=True)
(ROOT / 'worked-examples.json').write_text(json.dumps({'schema_version': 1, 'examples': examples}, ensure_ascii=False, indent=2) + '\n')
(ROOT / 'example-notes.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2) + '\n')
for e in examples:
    lines = [f"# {e['case_id']} — {e['family']} / {e['channel']}", '',
             '## Brief', '', e['brief'], '', '## Response', '', e['response']]
    for path, content in e['files'].items():
        lines += ['', f'## Written file: `{path}`', '', '```markdown', content.rstrip('\n'), '```']
    lines += ['', '## Required checks', '']
    if not e['satisfies']:
        lines += ['This case declares no required checks.']
    for c in e['satisfies']:
        lines += [f"- **{'MET (author assessment)' if c['met'] else 'UNMET'} — {c['check_id']}**: {c['evidence']}"]
    lines += ['', '## Craft choices', '', e['why_it_is_good']]
    if notes[e['case_id']]:
        lines += ['', '## Turn scope', '', notes[e['case_id']]]
    (ROOT / 'worked-examples' / f"{e['case_id']}.md").write_text('\n'.join(lines) + '\n')
print(f'Wrote {len(examples)} examples to JSON and {len(examples)} individual Markdown files.')
