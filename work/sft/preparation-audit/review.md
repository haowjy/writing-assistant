# Native SFT mask audit

Synthetic format probes only; no training executed.

## sft-mask-F1

51 input tokens; 23 loss tokens.

Rendered conversation:

```text
<bos><|turn>system
<|think|>
You help writers.<turn|>
<|turn>user
Write two sentences about repairing a clock.<turn|>
<|turn>model
Nell laid the stopped clock on a folded towel. One tooth from its brass wheel caught in the cloth.<turn|>

```

Supervised text (all other tokens have label -100):

```text
Nell laid the stopped clock on a folded towel. One tooth from its brass wheel caught in the cloth.<turn|>

```

## sft-mask-F2

335 input tokens; 79 loss tokens.

Rendered conversation:

```text
<bos><|turn>system
<|think|>
<|tool>declaration:write_file{description:<|"|>Create or replace a file. Commit canon only when authorized.<|"|>,parameters:{properties:{content:{description:<|"|>Complete text<|"|>,type:<|"|>STRING<|"|>},path:{description:<|"|>File<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>,<|"|>content<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><|tool>declaration:patch_file{description:<|"|>Replace one exact text span; fails on ambiguous matches.<|"|>,parameters:{properties:{new:{description:<|"|>Replacement<|"|>,type:<|"|>STRING<|"|>},old:{description:<|"|>Unique old text<|"|>,type:<|"|>STRING<|"|>},path:{description:<|"|>File<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>,<|"|>old<|"|>,<|"|>new<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
Save this opening in drafts/opening.md: The shop bell rang, although the door stayed shut.<turn|>
<|turn>model
<|tool_call>call:write_file{content:<|"|>The shop bell rang, although the door stayed shut.<|"|>,path:<|"|>drafts/opening.md<|"|>}<tool_call|><|tool_response>response:write_file{ok:true,result:<|"|>written<|"|>}<tool_response|>Saved the opening.<turn|>
<|turn>user
Change shop to bakery, preserving the rest.<turn|>
<|turn>model
<|tool_call>call:patch_file{new:<|"|>bakery<|"|>,old:<|"|>shop<|"|>,path:<|"|>drafts/opening.md<|"|>}<tool_call|><|tool_response>response:patch_file{ok:true,result:<|"|>written<|"|>}<tool_response|>Updated the location.<turn|>

```

Supervised text (all other tokens have label -100):

```text
<|tool_call>call:write_file{content:<|"|>The shop bell rang, although the door stayed shut.<|"|>,path:<|"|>drafts/opening.md<|"|>}<tool_call|>Saved the opening.<turn|>
<|tool_call>call:patch_file{new:<|"|>bakery<|"|>,old:<|"|>shop<|"|>,path:<|"|>drafts/opening.md<|"|>}<tool_call|>Updated the location.<turn|>

```

## sft-mask-F3

68 input tokens; 37 loss tokens.

Rendered conversation:

```text
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
Suggest two reasons a gardener would hide a parcel. Leave the choice open.<turn|>
<|turn>model
1. It contains bulbs for a surprise garden; hiding it protects a kind intention.
2. It contains a disputed deed; hiding it postpones a confrontation over ownership.<turn|>

```

Supervised text (all other tokens have label -100):

```text
1. It contains bulbs for a surprise garden; hiding it protects a kind intention.
2. It contains a disputed deed; hiding it postpones a confrontation over ownership.<turn|>

```

## sft-mask-F4

333 input tokens; 107 loss tokens.

Rendered conversation:

```text
<bos><|turn>system
<|think|>
<|tool>declaration:read_file{description:<|"|>Read a UTF-8 workspace file.<|"|>,parameters:{properties:{path:{description:<|"|>File<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><|tool>declaration:write_file{description:<|"|>Create or replace a file. Commit canon only when authorized.<|"|>,parameters:{properties:{content:{description:<|"|>Complete text<|"|>,type:<|"|>STRING<|"|>},path:{description:<|"|>File<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>,<|"|>content<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
Make a small linked wiki from source/note.md. Preserve uncertainty.<turn|>
<|turn>model
<|tool_call>call:read_file{path:<|"|>source/note.md<|"|>}<tool_call|><|tool_response>response:read_file{ok:true,result:<|"|># Canon

Eda suspects the gate was moved. Nobody has measured its position.<|"|>}<tool_response|><|tool_call>call:write_file{content:<|"|># Index

- [Gate](gate.md)<|"|>,path:<|"|>kb/index.md<|"|>}<tool_call|><|tool_response>response:write_file{ok:true,result:<|"|>written<|"|>}<tool_response|><|tool_call>call:write_file{content:<|"|># Canon

Eda suspects the gate was moved. Nobody has measured its position.<|"|>,path:<|"|>kb/gate.md<|"|>}<tool_call|><|tool_response>response:write_file{ok:true,result:<|"|>written<|"|>}<tool_response|>Created the wiki; the gate movement remains an unverified suspicion.<turn|>

```

Supervised text (all other tokens have label -100):

```text
<|tool_call>call:read_file{path:<|"|>source/note.md<|"|>}<tool_call|><|tool_call>call:write_file{content:<|"|># Index

- [Gate](gate.md)<|"|>,path:<|"|>kb/index.md<|"|>}<tool_call|><|tool_call>call:write_file{content:<|"|># Canon

Eda suspects the gate was moved. Nobody has measured its position.<|"|>,path:<|"|>kb/gate.md<|"|>}<tool_call|>Created the wiki; the gate movement remains an unverified suspicion.<turn|>

```

## sft-mask-F5

167 input tokens; 48 loss tokens.

Rendered conversation:

```text
<bos><|turn>system
<|think|>
<|tool>declaration:read_file{description:<|"|>Read a UTF-8 workspace file.<|"|>,parameters:{properties:{path:{description:<|"|>File<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>path<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
Read kb/canon.md and write a short opening in chat without resolving the question.<turn|>
<|turn>model
<|tool_call>call:read_file{path:<|"|>kb/canon.md<|"|>}<tool_call|><|tool_response>response:read_file{ok:true,result:<|"|># Canon

Eda suspects the gate was moved. Nobody has measured its position.<|"|>}<tool_response|>Eda set her basket beside the gatepost. The latch met her hand at an unfamiliar height; she went back inside for a ruler.<turn|>

```

Supervised text (all other tokens have label -100):

```text
<|tool_call>call:read_file{path:<|"|>kb/canon.md<|"|>}<tool_call|>Eda set her basket beside the gatepost. The latch met her hand at an unfamiliar height; she went back inside for a ruler.<turn|>

```
