# Native Gemma tool integration

The local instruction-tuned backend now passes tool schemas to the checkpoint's
chat template, renders tool results in its native response structure, and parses
generated calls with its response grammar. Ordinary final replies remain text.
The implementation follows [Google's Gemma function-calling documentation](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4).

A single E2B-IT smoke on the RTX 3090 executed `read_file` and `write_file`, then
returned a normal reply. Both tool calls succeeded, with no tool errors. The model
omitted the source file's trailing newline in its raw write arguments, so the
exact-copy instruction failed. Parsing did not remove the newline.

The run used NF4 weights, BF16 compute, temperature 0, a 4,096-token context and
512 output tokens per call. Three generations took about 5.1 seconds, excluding
model loading. This checks the transport and workspace loop; it does not establish
performance on the five task families.

- [Recorded configuration and result summary](native-inference-validation.json).
- [Local review with rendered prompts and raw outputs](../../runs/validation/gemma-native-v1/review.md).
- [Original five-case pilot](pilot-e2b.md), preserved with its earlier protocol.
- [Runtime contract](../../docs/local-inference.md).

All 43 unit tests passed, including cached checkpoint template/parser checks,
multiline tool arguments, tool-result rendering, and a scripted file-writing loop.
The original five cases have not been rerun. No Astra grading was invoked.
Native tool use on the other checkpoints is unverified; base-model transcript
runs reject tools until an appropriate template/protocol is established. The
200-attempt manifest remains a proposal, not a claim that every condition is ready.
