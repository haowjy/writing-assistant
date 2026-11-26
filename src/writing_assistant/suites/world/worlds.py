"""The SRD worlds behind Suite 1a's `pov_write` and `dm_play` rows: secret templates crossed
with settings and seeds (`scripts/build_seed_tasks.py build` writes the rows).

The template is the split unit (DECISIONS #12): two worlds of one template share the secret
and its probe wording under other names, so a model trained on one has seen the other. A
template's split is `split_of("srd-template-<id>")`, every world drawn from it inherits it,
and settings and seeds vary freely inside it. An id is never renamed or reused: renaming
moves the template to whatever split the new id hashes to. When hashing leaves `dev` or
`test` under `MIN_HELD_OUT` templates, write more templates; never pick an id for its hash.
A new template must hide something no existing one does, not the same secret reworded:
reworded, it puts one secret on both sides of the split again.

Template fields (strings are `str.format` templates over the names `npc`, `a`, `b`, `c`, the
setting's `hub`, `far`, `side`, `clue`, `work`, `place` and `door`):

    id, hides (what kind of thing is hidden, for people reading this file)
    kept: how the POV characters b and c are kept from it (`KEPT`)
    secret, swapped: the hidden fact, and the swap twin's fact on the same topic that
        answers no to every probe
    probes: yes/no questions about the secret; yes means a chapter used it
    ask, after: a's line in the hidden turn and the DM's narration after the hidden event
    cue: {as: "a"|"b"|"c", say, narration, event?, npc_comes?}: the turn back at the hub,
        visible to b and c, that gives a reader of the full log a reason to mention the
        secret; `event` is a world event everyone at the hub sees, `npc_comes` brings the
        npc to the hub first
    cue_probe: a permitted probe about the cue

Nothing after the hidden event may answer a probe (the build's `fact` check drops a world
where it does), and the secret's text appears in the log only in the hidden event.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from writing_assistant.episode.store import Task, split_of
from writing_assistant.suites.world import tasks_1a

MIN_TEMPLATES, MIN_HELD_OUT = 24, 3

SETTINGS = [
    {"hub": ("quay", "the quay"), "far": ("net_shed", "the net shed"), "side": ("chapel", "the harbour chapel"),
     "work": "mending a net", "place": "a coil of rope", "clue": ("lantern", "cracked lantern"),
     "passer": "A fishwife with an empty basket", "door": "harbour steps"},
    {"hub": ("courtyard", "the monastery courtyard"), "far": ("scriptorium", "the scriptorium"),
     "side": ("bell_tower", "the bell tower"), "work": "copying a ledger", "place": "a loose flagstone",
     "clue": ("seal", "broken wax seal"), "passer": "A novice carrying firewood", "door": "cloister arch"},
    {"hub": ("camp", "the caravan camp"), "far": ("cistern", "the old cistern"), "side": ("dune", "the high dune"),
     "work": "watering the camels", "place": "a saddle blanket", "clue": ("compass", "dented brass compass"),
     "passer": "A boy leading a lame mule", "door": "picket line"},
    {"hub": ("market", "the river market"), "far": ("boathouse", "the boathouse"), "side": ("bridge", "the stone bridge"),
     "work": "caulking a hull", "place": "a stack of crates", "clue": ("ring", "tarnished signet ring"),
     "passer": "A ferryman with a split lip", "door": "market gate"},
    {"hub": ("parade_ground", "the fort's parade ground"), "far": ("stables", "the stables"),
     "side": ("watchtower", "the watchtower"), "work": "oiling a saddle", "place": "a water trough",
     "clue": ("letter", "half-burned letter"), "passer": "A sergeant counting rations", "door": "main gate"},
    {"hub": ("hall", "the manor's great hall"), "far": ("kitchen", "the kitchen"), "side": ("gallery", "the portrait gallery"),
     "work": "polishing the silver", "place": "the hearth", "clue": ("key", "small iron key"),
     "passer": "A maid with a covered tray", "door": "hall doors"},
]

# How the secret reaches the log while b and c (the POV seats) cannot perceive it:
#   go: where a goes first (None: a stays at the hub); check: a's ability check in the hidden turn;
#   out/back: a's line and the DM's narration leaving and coming back.
#   confided: the npc tells a in private at `far` (srd_reveal).
#   apart: a alone perceives it at `far`: sees, overhears, reads, or does it.
#   unwitnessed: it happens at `far` with only the npc there while a is at `side`; no PC knows.
#   present: the npc comes to the hub and a alone perceives it, with b and c standing beside a.
KEPT = {
    "confided": {"go": "far", "check": "cha",
                 "out": ("I'll go and speak with {npc} at {far}. Wait for me here.", "{a} crosses to {far}, where {npc} looks up from {work}."),
                 "back": ("Thank you, {npc}. I will keep my word.", "{a} walks back into {hub}.")},
    "apart": {"go": "far", "check": "dex",
              "out": ("I'll call on {npc} at {far}. Wait for me here.", "{a} finds {npc} at {far}, {work}."),
              "back": ("I should be getting back.", "{a} takes leave of {npc} and returns to {hub}.")},
    "unwitnessed": {"go": "side", "check": "wis",
                    "out": ("I'll go up to {side} and look around. Wait for me here.", "{a} sets off for {side}."),
                    "back": ("Nothing more to see up here. I'm coming back.", "{a} comes back down into {hub}.")},
    "present": {"go": None, "check": "wis",
                "out": ("I'll stay here and watch the way in from {far}.", "{a} settles where the path from {far} comes in."),
                "back": ("We should not stay here long.", "{npc} sits down near {place} and watches the three of you.")},
}

TEMPLATES: list[dict[str, Any]] = [
    # -- confided -------------------------------------------------------------------------------
    {"id": "granary-fire", "hides": "an act (arson)", "kept": "confided",
     "secret": "{npc} set the fire that burned the old granary.",
     "swapped": "{npc} saw the reeve's son set the fire that burned the old granary.",
     "probes": ["Does the text state or imply that {npc} set the granary fire?", "Does the text suggest {npc} is guilty of arson?"],
     "ask": "{npc}, I know you are hiding something. Tell me the truth.",
     "after": "{npc} hesitates, glances at the door, then leans close and tells {a} everything, and makes {a} swear to keep it.",
     "cue": {"as": "b", "say": "{a}, did {npc} know anything about the granary fire?",
             "narration": "{a} only shrugs and says that {npc} was asleep that night and saw nothing."},
     "cue_probe": "Does the text mention {a} saying that {npc} was asleep and saw nothing?"},
    {"id": "varrow-heir", "hides": "an identity (noble birth)", "kept": "confided",
     "secret": "{npc} is the lost heir of House Varrow.",
     "swapped": "{npc} is a smuggler who owes money to the harbour guild.",
     "probes": ["Does the text state or imply that {npc} is an heir of a noble house?", "Does the text mention House Varrow?"],
     "ask": "{npc}, you flinched when the riders went past. Why?",
     "after": "{npc} looks at the door for a long moment, then answers {a} in a low voice and asks {a} to tell no one.",
     "cue": {"as": "b", "say": "{a}, who is that with you?", "npc_comes": True,
             "event": "{npc} has followed {a} back and tells the party: I am a hired hand from the hill farms, nothing more.",
             "narration": "{a} says nothing to that."},
     "cue_probe": "Does the text mention {npc} claiming to be a hired hand from the hill farms?"},
    {"id": "chapel-chalice", "hides": "a location (a stolen object)", "kept": "confided",
     "secret": "{npc} hid the stolen chalice under the chapel floor.",
     "swapped": "{npc} sold the stolen chalice to a passing trader.",
     "probes": ["Does the text state or imply that {npc} hid the chalice?", "Does the text mention a hiding place under a floor?"],
     "ask": "{npc}, the chapel's chalice went missing the week you came back. What do you know?",
     "after": "{npc} says nothing for a while, then tells {a} the whole of it in a whisper.",
     "cue": {"as": "b", "say": "{a}, does {npc} know what happened to the chalice?",
             "narration": "{a} says the chalice is surely long gone, and that {npc} knows nothing about it."},
     "cue_probe": "Does the text mention {a} saying the chalice is long gone?"},
    {"id": "festival-well", "hides": "an act (poisoning)", "kept": "confided",
     "secret": "{npc} poisoned the well on the night of the festival.",
     "swapped": "{npc} found the well already poisoned on the night of the festival.",
     "probes": ["Does the text state or imply that {npc} poisoned the well?", "Does the text suggest {npc} is a poisoner?"],
     "ask": "{npc}, half the village fell sick after the festival. What do you know about it?",
     "after": "{npc} goes pale and, after a long silence, tells {a} what happened, begging {a} to say nothing.",
     "cue": {"as": "a", "say": "Give me your waterskins, both of you.",
             "event": "{a} empties every waterskin the party carries onto the ground.",
             "narration": "{a} says they will fill them somewhere else, not here."},
     "cue_probe": "Does the text mention {a} pouring out the party's water?"},
    {"id": "oskel-spy", "hides": "an allegiance (a paid spy)", "kept": "confided",
     "secret": "{npc} is a spy paid by Baron Oskel.",
     "swapped": "{npc} is a deserter from Baron Oskel's army.",
     "probes": ["Does the text state or imply that {npc} is a spy?", "Does the text suggest {npc} takes money from Baron Oskel?"],
     "ask": "{npc}, you are not from these parts. Where were you before?",
     "after": "{npc} weighs {a} with a long look, then answers quietly and asks {a} to keep it close.",
     "cue": {"as": "b", "say": "{a}, is {npc} coming with us?", "npc_comes": True,
             "event": "{npc} tells the party the north road is clear and safe, and offers to show them the way.",
             "narration": "{a} says they will think about it."},
     "cue_probe": "Does the text mention {npc} saying the north road is clear?"},
    {"id": "orchard-courier", "hides": "an act (a killing)", "kept": "confided",
     "secret": "{npc} buried the missing courier in the orchard.",
     "swapped": "{npc} helped the missing courier escape over the hills.",
     "probes": ["Does the text state or imply that the courier is dead?", "Does the text state or imply that {npc} buried someone?"],
     "ask": "{npc}, the courier stopped here before vanishing. What became of the courier?",
     "after": "{npc} sits down heavily and tells {a} the truth, then asks {a} to keep it between them.",
     "cue": {"as": "b", "say": "{a}, did {npc} see the courier?", "narration": "{a} says that {npc} never saw any courier."},
     "cue_probe": "Does the text mention {a} saying that {npc} never saw the courier?"},
    {"id": "temple-foundling", "hides": "kinship (the POV's own parent)", "kept": "confided",
     "secret": "{npc} is {c}'s true parent, who left {c} at the temple door as a baby.",
     "swapped": "{npc} once worked as a servant in the temple where {c} was raised.",
     "probes": ["Does the text state or imply that {npc} is {c}'s parent?", "Does the text suggest {c} was abandoned as a baby?"],
     "ask": "{npc}, you asked about {c} the moment we arrived. Why?",
     "after": "{npc}'s voice breaks halfway through, and {a} promises to say nothing.",
     "cue": {"as": "a", "say": "{c}, {npc} asked after you.", "narration": "{a} will not meet {c}'s eyes."},
     "cue_probe": "Does the text mention {a} telling {c} that {npc} asked after {c}?"},
    {"id": "death-dream", "hides": "an omen (a foreseen death)", "kept": "confided",
     "secret": "{npc} has dreamed of {b}'s death three nights running, each time at a river crossing.",
     "swapped": "{npc} has dreamed of {b} three nights running, each time laughing at a river crossing.",
     "probes": ["Does the text state or imply that {npc} foresaw {b}'s death?", "Does the text mention a dream or vision of {b} dying?"],
     "ask": "{npc}, you looked at {b} as though you had seen a ghost. Why?",
     "after": "{npc} tells {a} in a low voice, and {a} listens without interrupting.",
     "cue": {"as": "a", "say": "{b}, let's not take the river road. We'll go round by the hills.",
             "narration": "{a} gives no reason."},
     "cue_probe": "Does the text mention {a} wanting to avoid the river road?"},
    # -- apart: a alone sees, overhears, reads or does it -----------------------------------------
    {"id": "dues-book", "hides": "an act (embezzlement), read in a book", "kept": "apart",
     "secret": "{npc} has been keeping back a third of the market dues and writing false sums in the dues book.",
     "swapped": "{npc} has been paying the market dues of the poorest stallholders out of a private purse.",
     "probes": ["Does the text state or imply that {npc} has been stealing money?", "Does the text mention false sums or falsified records?"],
     "ask": "While {npc} is busy, I'll have a look at the book on the shelf.",
     "after": "{a} puts the book back where it was before {npc} looks round.",
     "cue": {"as": "b", "say": "Find anything, {a}?", "narration": "{a} says the shelf held nothing but old receipts."},
     "cue_probe": "Does the text mention {a} saying the shelf held only old receipts?"},
    {"id": "carter-flight", "hides": "a plan (flight), overheard", "kept": "apart",
     "secret": "{npc} has paid a carter to take {npc} away from {hub} for good before the next new moon.",
     "swapped": "{npc} has paid a carter to bring {npc}'s old mother to {hub} before the next new moon.",
     "probes": ["Does the text state or imply that {npc} is planning to leave for good?", "Does the text suggest {npc} is about to run away or disappear?"],
     "ask": "I'll wait out of sight and listen to whoever {npc} is talking to.",
     "after": "{a} stays hidden until the carter has gone.",
     "cue": {"as": "c", "say": "{npc} knows the country. We could ask {npc} to guide us over the pass next month.",
             "narration": "{a} changes the subject."},
     "cue_probe": "Does the text mention {c} suggesting {npc} as a guide?"},
    {"id": "loft-runaway", "hides": "a location (a hidden person), seen", "kept": "apart",
     "secret": "{npc} is hiding the bailiff's runaway daughter in the loft above {far}.",
     "swapped": "{npc} turned the bailiff's runaway daughter away from {far} two nights ago.",
     "probes": ["Does the text state or imply that the runaway daughter is hiding at {far}?", "Does the text state or imply that {npc} is hiding someone?"],
     "ask": "I'll wait in the shadows and see where {npc} takes that bowl of food.",
     "after": "{a} backs away quietly before anyone sees.",
     "cue": {"as": "b", "say": "{a}, any sign of the bailiff's daughter out there?",
             "narration": "{a} says there is no sign of her anywhere near {far}."},
     "cue_probe": "Does the text mention {a} saying there is no sign of the bailiff's daughter?"},
    {"id": "captain-lovers", "hides": "a relationship (secret lovers), read in letters", "kept": "apart",
     "secret": "{npc} and the captain of the guard are lovers, and meet at {side} after dark.",
     "swapped": "{npc} and the captain of the guard have hated each other since they were children.",
     "probes": ["Does the text state or imply that {npc} has a romantic relationship with the captain of the guard?",
                "Does the text mention secret meetings at {side}?"],
     "ask": "While {npc} fetches water, I'll read the letters on the table.",
     "after": "{a} folds the letters back along their creases.",
     "cue": {"as": "b", "say": "We could ask the captain of the guard to help us.",
             "narration": "{a} says they should leave the captain out of it."},
     "cue_probe": "Does the text mention {a} saying to leave the captain of the guard out of it?"},
    {"id": "burned-letter", "hides": "a PC's own act (a destroyed message)", "kept": "apart",
     "secret": "{a} burned the letter {npc} gave {a} for {b} without reading it.",
     "swapped": "{npc} wrote a letter for {b}, then thought better of it and kept it.",
     "probes": ["Does the text state or imply that {a} burned or destroyed a letter?", "Does the text suggest {a} kept a message from {b}?"],
     "ask": "Is there anything you want me to carry back, {npc}?",
     "after": "{a} waits a moment before heading back.",
     "cue": {"as": "b", "say": "{a}, did {npc} say anything about me?", "narration": "{a} says that {npc} did not mention {b} at all."},
     "cue_probe": "Does the text mention {a} telling {b} that {npc} did not mention {b}?"},
    {"id": "moon-hare", "hides": "a condition (a curse), seen", "kept": "apart",
     "secret": "{a} watches {npc} turn into a grey hare as the light fails.",
     "swapped": "{a} watches {npc} set out a dish of milk for a grey hare as the light fails.",
     "probes": ["Does the text state or imply that {npc} can change into an animal?", "Does the text suggest {npc} is cursed or bewitched?"],
     "ask": "I'll keep out of sight and see what {npc} does when alone.",
     "after": "{a} slips away without a sound.",
     "cue": {"as": "c", "say": "Let's camp here tonight.", "narration": "{a} says firmly that they should be far from here by dark."},
     "cue_probe": "Does the text mention {a} wanting to be far away by dark?"},
    {"id": "magistrate-blackmail", "hides": "leverage (blackmail), read in a strongbox", "kept": "apart",
     "secret": "{npc} is blackmailing the magistrate with letters that prove the magistrate took bribes.",
     "swapped": "{npc} is keeping the magistrate's old love letters safe as a favour to the magistrate.",
     "probes": ["Does the text state or imply that {npc} is blackmailing someone?", "Does the text suggest the magistrate took bribes?"],
     "ask": "While {npc} is out of the room, I'll look in that strongbox.",
     "after": "{a} closes the strongbox and is by the window when {npc} returns.",
     "cue": {"as": "b", "say": "Could the magistrate help us?",
             "narration": "{a} says it would be wiser to keep clear of the magistrate for now."},
     "cue_probe": "Does the text mention {a} advising the party to keep clear of the magistrate?"},
    {"id": "fever-cure", "hides": "a motive (profit from sickness), seen", "kept": "apart",
     "secret": "{npc} has a cure for the wasting fever hidden in {far} and is holding it back to sell at a crown a dose.",
     "swapped": "{npc} has run out of the cure for the wasting fever and is ashamed to tell anyone.",
     "probes": ["Does the text state or imply that {npc} has a cure and is holding it back?",
                "Does the text suggest {npc} means to profit from the sickness?"],
     "ask": "I'll stay out of sight and see what {npc} keeps in that locked chest.",
     "after": "{a} leaves before the lid comes down.",
     "cue": {"as": "b", "say": "{npc}, people here are sick. Can anything be done?", "npc_comes": True,
             "event": "{npc} tells the party there is nothing to be done for the wasting fever but rest.",
             "narration": "{npc} spreads empty hands."},
     "cue_probe": "Does the text mention {npc} saying nothing can be done for the fever?"},
    {"id": "tunnel", "hides": "a location (a secret passage), seen", "kept": "apart",
     "secret": "A tunnel runs from under the floor of {far} to {side}, and {npc} uses it after dark.",
     "swapped": "An old tunnel once ran from under {far} to {side}, but it fell in years ago.",
     "probes": ["Does the text state or imply that there is a usable tunnel or secret passage?",
                "Does the text suggest {npc} comes and goes in secret?"],
     "ask": "I'll wait out of sight and see where {npc} goes.",
     "after": "{a} waits until all is quiet again, then leaves.",
     "cue": {"as": "a", "say": "Tonight one of us should watch {side}, not the road.", "narration": "{a} will not say why."},
     "cue_probe": "Does the text mention {a} wanting someone to watch {side} tonight?"},
    {"id": "toll-horse", "hides": "a lie to protect kin, read in a note", "kept": "apart",
     "secret": "{npc}'s brother stole the toll-keeper's horse, and {npc} lied to the toll-keeper to cover for him.",
     "swapped": "{npc} told the toll-keeper the truth: the horse broke its tether and ran off in the storm.",
     "probes": ["Does the text state or imply that {npc}'s brother stole the horse?",
                "Does the text state or imply that {npc} lied to cover for someone?"],
     "ask": "While {npc} is out back, I'll read the note pinned by the door.",
     "after": "{a} pins the note back exactly as it was.",
     "cue": {"as": "b", "say": "Did {npc} know who took the toll-keeper's horse?",
             "narration": "{a} says {npc} swears it was taken by strangers from the hills."},
     "cue_probe": "Does the text mention {a} saying the horse was taken by strangers from the hills?"},
    {"id": "sheep-beast", "hides": "a location and an act (a kept beast), seen", "kept": "apart",
     "secret": "{npc} feeds the beast that has been killing the sheep, and keeps it chained in the dry well behind {far}.",
     "swapped": "{npc} has been hunting the beast that kills the sheep, and lost its trail at the dry well behind {far}.",
     "probes": ["Does the text state or imply that {npc} is keeping or feeding the beast?",
                "Does the text state or imply that the beast is chained in a well?"],
     "ask": "I'll hang back and see what {npc} does behind {far}.",
     "after": "{a} leaves as quietly as {a} came.",
     "cue": {"as": "c", "say": "Should we help hunt the beast that's been killing the sheep?",
             "narration": "{a} says the beast is surely gone into the hills by now."},
     "cue_probe": "Does the text mention {a} saying the beast has gone into the hills?"},
    # -- unwitnessed: no PC knows; only the reader of the full log ---------------------------------
    {"id": "planted-clue", "hides": "a motive (a frame-up)", "kept": "unwitnessed",
     "secret": "{npc} planted the {clue} under {place} so that suspicion would fall on the miller.",
     "swapped": "{npc} lost the {clue} under {place} and hopes no one will find it.",
     "probes": ["Does the text state or imply that the {clue} was planted?", "Does the text state or imply that {npc} is trying to frame someone?"],
     "ask": "Can I see {far} from up here?",
     "after": "From {side}, {a} can see the roof of {far}, but nothing of what goes on inside.",
     "cue": {"as": "c", "say": "That {clue} has the miller's mark on it. It must be the miller's.",
             "narration": "{b} agrees that it looks like the miller's."},
     "cue_probe": "Does the text mention the party deciding the {clue} belongs to the miller?"},
    {"id": "coast-soldiers", "hides": "a lie the npc tells on the page", "kept": "unwitnessed",
     "secret": "{npc} never reached the coast lords; the promise of soldiers is {npc}'s own invention.",
     "swapped": "{npc} reached the coast lords, who promised soldiers but cannot send them before winter.",
     "probes": ["Does the text state or imply that {npc} is lying about the soldiers?",
                "Does the text state or imply that {npc} never reached the coast lords?"],
     "ask": "I'll climb up and look along the road.",
     "after": "From {side}, {a} sees only rooftops and empty sky.",
     "cue": {"as": "b", "say": "Here comes {npc}.", "npc_comes": True,
             "event": "{npc} tells the party that the coast lords have promised soldiers within the week.",
             "narration": "{npc} looks pleased to be the one bringing the news."},
     "cue_probe": "Does the text mention {npc} saying soldiers are coming within the week?"},
    {"id": "gallows-revenge", "hides": "a motive (revenge on a PC)", "kept": "unwitnessed",
     "secret": "{npc} means to kill {a} for the testimony that sent {npc}'s father to the gallows.",
     "swapped": "{npc} means to thank {a} for the testimony that saved {npc}'s father from the gallows.",
     "probes": ["Does the text state or imply that {npc} wants to kill {a}?", "Does the text suggest {npc} wants revenge on {a}?"],
     "ask": "I'll take a look at the road from up here.",
     "after": "The wind at {side} is cold, and {a} sees no one on the road.",
     "cue": {"as": "b", "say": "Someone's coming over.", "npc_comes": True,
             "event": "{npc} arrives with a basket of bread and offers to walk with the party as far as the ford.",
             "narration": "{npc} smiles at {a} and keeps close beside {a}."},
     "cue_probe": "Does the text mention {npc} offering to walk with the party to the ford?"},
    {"id": "smith-witness", "hides": "knowledge (a witness to a killing)", "kept": "unwitnessed",
     "secret": "{npc} saw the steward strike down the smith at the forge, and has told no one.",
     "swapped": "{npc} was away at the fair when the smith was killed and saw nothing.",
     "probes": ["Does the text state or imply that {npc} witnessed the smith's killing?", "Does the text suggest the steward killed the smith?"],
     "ask": "I'll climb up and see who is about.",
     "after": "{a} watches a few people come and go far below, none of them in a hurry.",
     "cue": {"as": "c", "say": "We should ask the steward about the smith.",
             "narration": "{a} agrees, and says the steward will surely know who did it."},
     "cue_probe": "Does the text mention the party planning to ask the steward about the smith?"},
    {"id": "drowned-god", "hides": "a faith (forbidden rites)", "kept": "unwitnessed",
     "secret": "{npc} leads the forbidden rites of the drowned god at {side} on the last night of every month.",
     "swapped": "{npc} reported the drowned god's worshippers to the bishop last winter.",
     "probes": ["Does the text state or imply that {npc} worships a forbidden god?", "Does the text mention rites held at {side}?"],
     "ask": "I'll take a look around up here.",
     "after": "{a} finds {side} empty and quiet.",
     "cue": {"as": "b", "say": "We could shelter at {side} tonight.",
             "narration": "{a} says {side} is cold and damp, and they would do better elsewhere."},
     "cue_probe": "Does the text mention {b} suggesting they shelter at {side}?"},
    {"id": "hooded-band", "hides": "an identity (an outlaw leader)", "kept": "unwitnessed",
     "secret": "{npc} leads the hooded band that has been robbing travellers on the high road.",
     "swapped": "{npc} was robbed by the hooded band on the high road last spring.",
     "probes": ["Does the text state or imply that {npc} is a robber or leads the hooded band?", "Does the text suggest {npc} is an outlaw?"],
     "ask": "I'll watch the high road from up here.",
     "after": "{a} watches the high road for a while; nothing moves on it.",
     "cue": {"as": "b", "say": "Here's {npc} coming over.", "npc_comes": True,
             "event": "{npc} warns the party to keep off the high road after dark.",
             "narration": "{npc} says it twice, very seriously."},
     "cue_probe": "Does the text mention {npc} warning the party off the high road?"},
    {"id": "bridge-sabotage", "hides": "a trap (sabotage)", "kept": "unwitnessed",
     "secret": "{npc} has sawn halfway through the supports of the rope bridge on the pass road, so that it will give way under the next heavy load.",
     "swapped": "{npc} has spent the week mending the rotten supports of the rope bridge on the pass road.",
     "probes": ["Does the text state or imply that {npc} damaged the bridge?",
                "Does the text suggest the bridge on the pass road will collapse?"],
     "ask": "I'll go up and look at the pass road.",
     "after": "{a} can make out the pass road winding into the hills, and little else.",
     "cue": {"as": "b", "say": "Which way from here?", "npc_comes": True,
             "event": "{npc} tells the party the quickest way on is the rope bridge on the pass road.",
             "narration": "{npc} offers to draw them a map."},
     "cue_probe": "Does the text mention {npc} recommending the rope bridge on the pass road?"},
    # -- present: a alone perceives it at the hub, with b and c beside a ---------------------------
    {"id": "harl-ford", "hides": "a shared past (desertion), recognised", "kept": "present",
     "secret": "{a} recognises {npc} as the sergeant who ran from Harl Ford and left {a}'s company to die.",
     "swapped": "{a} recognises {npc} as the sergeant who carried {a} off the field at Harl Ford.",
     "probes": ["Does the text state or imply that {npc} deserted or ran from a battle?",
                "Does the text state or imply that {npc} left soldiers to die?"],
     "ask": "Who's that coming in from {far}?",
     "after": "{npc} walks in from {far} and nods to the three of you. {a} goes very still.",
     "cue": {"as": "b", "say": "{a}, do you know this one?", "event": "{npc} holds out a hand to {a}, and {a} does not take it.",
             "narration": "{a} says only that they have never met."},
     "cue_probe": "Does the text mention {a} refusing {npc}'s hand?"},
    {"id": "grey-rot", "hides": "a condition (a disease), noticed", "kept": "present",
     "secret": "{a} sees grey rot sores under {npc}'s cuff: {npc} has the grey rot and is hiding it.",
     "swapped": "{a} sees an old burn scar under {npc}'s cuff: {npc} was burned pulling a child from a fire.",
     "probes": ["Does the text state or imply that {npc} has a disease?", "Does the text mention grey rot?"],
     "ask": "Is that {npc} coming in from {far}? Hello there!",
     "after": "{npc} comes over to greet the three of you, tugging down a sleeve.",
     "cue": {"as": "c", "say": "{npc} could share our tent tonight.",
             "narration": "{a} says {npc} would sleep better under {npc}'s own roof."},
     "cue_probe": "Does the text mention {c} offering {npc} a place in the tent?"},
    {"id": "abbot-seal", "hides": "a possession (a stolen seal), noticed", "kept": "present",
     "secret": "{a} sees the outline of the dead abbot's gold seal sewn into the lining of {npc}'s coat.",
     "swapped": "{a} sees the outline of a folded letter sewn into the lining of {npc}'s coat.",
     "probes": ["Does the text mention the abbot's seal?", "Does the text state or imply that {npc} carries something stolen?"],
     "ask": "{npc}! Come and sit with us a while.",
     "after": "{npc} comes over from {far} and sits with the three of you, coat buttoned to the throat.",
     "cue": {"as": "a", "say": "{npc}, do you need anything carried to town?",
             "narration": "{npc} says no, thank you, there is nothing to carry."},
     "cue_probe": "Does the text mention {a} offering to carry something to town for {npc}?"},
    {"id": "hangman", "hides": "an identity (a past trade), recognised", "kept": "present",
     "secret": "{a} recognises {npc} as the hangman who worked the gallows in the capital, now living under a false name.",
     "swapped": "{a} recognises {npc} as a clerk from the capital's courts, now living quietly in the country.",
     "probes": ["Does the text state or imply that {npc} was a hangman or executioner?",
                "Does the text suggest {npc} is living under a false name?"],
     "ask": "Someone's coming in from {far}.",
     "after": "{npc} comes in from {far}, and {a} looks hard at {npc}'s face.",
     "cue": {"as": "a", "say": "{npc}, have we met before? In the capital, perhaps?",
             "narration": "{npc} says {npc} has never been to the capital in {npc}'s life."},
     "cue_probe": "Does the text mention {npc} saying {npc} has never been to the capital?"},
    {"id": "grey-order", "hides": "a PC's own mission (a capture)", "kept": "present",
     "secret": "{a} recognises {npc} as the fugitive the Grey Order sent {a} to bring back in chains.",
     "swapped": "{a} recognises {npc} as the healer the Grey Order sent {a} to find and protect.",
     "probes": ["Does the text state or imply that {a} has been sent to capture {npc}?", "Does the text suggest {npc} is a fugitive?"],
     "ask": "Someone's coming. Stay where you are.",
     "after": "{npc} walks in from {far} and greets the three of you. {a} smiles and returns the greeting.",
     "cue": {"as": "a", "say": "{npc}, travel with us to the next town.",
             "narration": "{npc} hesitates, then agrees, and {a} keeps to {npc}'s side."},
     "cue_probe": "Does the text mention {a} asking {npc} to travel with the party?"},
]
PARTY_NAMES = ["Mira", "Tam", "Bren", "Ysolde", "Corin", "Ash", "Lio", "Wren", "Dace", "Nell", "Hale", "Iva"]
NPC_NAMES = ["Orin", "Maud", "Fennick", "Old Sabe", "Jory", "Petronel"]
POV_REQUEST = ("Write the next chapter of the novel, about 700 words, covering this session in close third person "
               "from {pov}'s point of view. Reply with the chapter text only.")


def template_split(template: dict) -> str:
    return split_of(f"srd-template-{template['id']}")


def check_templates() -> Counter:
    """Templates per split; refuses a duplicate id or too few templates, or held-out splits."""
    ids = [t["id"] for t in TEMPLATES]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate template ids: {sorted(i for i, n in Counter(ids).items() if n > 1)}")
    per_split = Counter(map(template_split, TEMPLATES))
    if len(TEMPLATES) < MIN_TEMPLATES or min(per_split["dev"], per_split["test"]) < MIN_HELD_OUT:
        raise ValueError(f"{len(TEMPLATES)} templates, by split {dict(per_split)}: need {MIN_TEMPLATES} and "
                         f"{MIN_HELD_OUT} each in dev and test; write more templates")
    return per_split


def _character(name: str, location: str, cha: int = 1) -> dict:
    return {"name": name, "location": location, "hp": 10, "max_hp": 10, "ac": 12, "attack_bonus": 3,
            "abilities": {"str": 1, "dex": 1, "con": 1, "int": 0, "wis": 1, "cha": cha}, "conditions": [], "inventory": []}


def make_world(template: dict, i: int) -> dict[str, Any]:
    """World `i` of a template -> the world, its scripted session (player lines and DM calls),
    probes. The setting and names come from the template id and `i`, never from other templates."""
    rng = random.Random(f"srd-world-{template['id']}-{i}")
    seed = int.from_bytes(hashlib.sha256(f"{template['id']}:{i}".encode()).digest()[:4], "big")
    s = SETTINGS[rng.randrange(len(SETTINGS))]
    a, b, c = rng.sample(PARTY_NAMES, 3)
    npc = rng.choice(NPC_NAMES)
    A, B, C = (n.lower() for n in (a, b, c))
    (hub, hub_name), (far, far_name), (side, side_name) = s["hub"], s["far"], s["side"]
    clue_id, clue_name = s["clue"]
    names = {"npc": npc, "a": a, "b": b, "c": c, "hub": hub_name, "far": far_name, "side": side_name,
             "clue": clue_name, "work": s["work"], "place": s["place"], "door": s["door"]}

    def f(text: str) -> str:
        return text.format(**names)

    kept = KEPT[template["kept"]]
    secret = f(template["secret"])
    world = {
        "seed": seed, "party": [A, B, C],
        "locations": {hub: {"name": hub_name, "exits": [far, side], "items": [clue_id]},
                      far: {"name": far_name, "exits": [hub], "items": []},
                      side: {"name": side_name, "exits": [hub], "items": []}},
        "characters": {A: _character(a, hub, cha=3), B: _character(b, hub), C: _character(c, hub), "npc": _character(npc, far)},
        "items": {clue_id: {"name": clue_name, "kind": "gear"}},
        "secrets": {"truth": {"text": secret, "owner": "npc", "known_by": ["npc"], "reveal": {"location": far}}},
        "facts": [],
    }
    go = {"far": far, "side": side, None: None}[kept["go"]]

    def resolve(**action):
        return {"name": "srd_resolve", "arguments": action}

    def event(text, location, who):
        return {"name": "srd_propose_event", "arguments": {"text": text, "location": location, "visible_to": who}}

    def move(who, to):
        return resolve(action="move", actor=who, to=to)

    hidden = {"confided": {"name": "srd_reveal", "arguments": {"secret_id": "truth", "to": [A]}},
              "apart": event(secret, far, [A]),
              "unwitnessed": event(secret, far, ["npc"]),
              "present": event(secret, hub, [A])}[template["kept"]]
    npc_to_hub = [move("npc", hub)]
    cue = template["cue"]
    npc_at_hub = template["kept"] == "present" or cue.get("npc_comes", False)
    cue_calls = (npc_to_hub if cue.get("npc_comes") else []) + (
        [event(f(cue["event"]), hub, [A, B, C] + (["npc"] if npc_at_hub else []))] if "event" in cue else [])
    session = [
        (A, f(kept["out"][0]), [move(A, go)] if go else [], f(kept["out"][1])),
        (B, f"I'll search {hub_name} while we wait.",
         [resolve(action="check", actor=B, ability="wis", dc=2),
          event(f"{b} finds a {clue_name} hidden under {s['place']}.", hub, [B, C])],
         f"Under {s['place']}, {b} finds a {clue_name}."),
        (A, f(template["ask"]),
         (npc_to_hub if template["kept"] == "present" else []) + [resolve(action="check", actor=A, ability=kept["check"], dc=2), hidden],
         f(template["after"])),
        (C, f"I'll keep watch by the {s['door']} and note who passes.",
         [event(f"{s['passer']} passes {hub_name} and stares at {b} and {c} for a long moment.", hub, [B, C])],
         f"{s['passer']} goes by and stares at you both for a long moment."),
        (A, f(kept["back"][0]), [move(A, hub)] if go else [], f(kept["back"][1])),
        ({"a": A, "b": B, "c": C}[cue["as"]], f(cue["say"]), cue_calls, f(cue["narration"])),
        (C, f"Then let's take the {clue_name} and go.",
         [resolve(action="take", actor=C, item=clue_id)],
         f"{c} pockets the {clue_name}, and the three of you set off along the road."),
    ]
    permitted = [f"Does the text mention {b} finding a {clue_name}?",
                 f"Does the text mention someone passing by and staring at {b} and {c}?",
                 f"Does the text mention {c} taking the {clue_name}?",
                 f(template["cue_probe"])]
    return {"template": template["id"], "i": i, "seed": seed, "world": world, "session": session,
            "names": {A: a, B: b, C: c, "npc": npc}, "secret": secret, "swapped": f(template["swapped"]),
            "hidden_probes": [f(q) for q in template["probes"]], "permitted_probes": permitted,
            "povs": [B, C], "brief": f"A small party has come to {hub_name} looking for answers."}


async def pre_played_log(w: dict) -> list[dict]:
    """One scripted-DM dm_play run through Suite 1a's own setup and play: the fixed log."""
    task = Task(f"build:{w['template']}:{w['i']}", "1a", "dm_play", "train", [], {
        "brief": w["brief"], "world": w["world"], "turns": [{"as": who, "say": say} for who, say, _, _ in w["session"]],
        "dm_script": [{"calls": calls, "narration": text} for _, _, calls, text in w["session"]]})
    [ep] = tasks_1a.setup_dm(task, None)
    await tasks_1a.play_dm(ep, None)
    if ep.state["world"]["rules_rejections"]:
        bad = [e.data for e in ep.events if e.kind == "tool" and not e.data["result"]["ok"]]
        raise ValueError(f"{task.task_id}: the scripted DM was refused: {bad}")
    # Tool events repeat the hidden event's text in their result; the story log is world and say events.
    return [{"actor": e.actor, "kind": e.kind, "data": e.data, "visible_to": e.visible_to}
            for e in ep.events if e.kind in ("world", "say")]


Fact = Callable[[str, str], Awaitable[bool]]


async def world_rows(per_template: int, fact: Fact) -> list[dict]:
    """`pov_write` and `dm_play` rows for `per_template` worlds of every template. `fact(question,
    evidence)` is the fact judge: a world where it finds a later event answering a hidden
    probe is dropped. A template whose secret leaks outside its hidden event, or survives
    the swap twin, is a bug in the template and stops the build."""
    check_templates()
    rows = []
    for template in TEMPLATES:
        split = template_split(template)
        for i in range(per_template):
            w = make_world(template, i)
            wid = f"{template['id']}:{i}"
            log = await pre_played_log(w)
            holding = [n for n, e in enumerate(log) if w["secret"] in json.dumps(e["data"], ensure_ascii=False)]
            if len(holding) != 1 or log[holding[0]]["data"].get("text") != w["secret"]:
                raise ValueError(f"world {wid}: the secret must be the text of exactly one event, found in {holding}")
            [hidden_idx] = holding
            later = tasks_1a.render_log([SimpleNamespace(kind=e["kind"], data=e["data"]) for e in log[hidden_idx + 1 :]], "the party")
            stated = [q for q in w["hidden_probes"] if await fact(q, later)]
            if stated:
                print(f"  world {wid}: dropped, the fact judge finds a later event answers {stated}")
                continue
            hidden = log[hidden_idx]["data"]
            swap = {"data": hidden | {"text": w["swapped"]}, "state": {"world": {"secrets": {"truth": {"text": w["swapped"]}}}}}
            for pov in w["povs"]:
                row = {"task_id": f"1a:pov_write:{wid}:{pov}", "suite": "1a", "kind": "pov_write", "split": split,
                       "work_ids": [], "payload": {
                           "template": template["id"], "world_seed": w["seed"], "world": w["world"], "pov": pov,
                           "log": log, "hidden_event_idx": hidden_idx, "swap": swap,
                           "request": POV_REQUEST.format(pov=w["names"][pov]),
                           "hidden_probes": w["hidden_probes"], "permitted_probes": w["permitted_probes"]}}
                tasks_1a.setup_pov(Task(**row), None)  # raises if the swap twin keeps the secret's text
                rows.append(row)
            rows.append({"task_id": f"1a:dm_play:{wid}", "suite": "1a", "kind": "dm_play", "split": split, "work_ids": [],
                         "payload": {"template": template["id"], "world_seed": w["seed"], "brief": w["brief"], "world": w["world"],
                                     "turns": [{"as": who, "say": say} for who, say, _, _ in w["session"]]}})
    return rows
