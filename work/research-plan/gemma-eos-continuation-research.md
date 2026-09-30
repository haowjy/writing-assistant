# EOS after a Gemma answer with a pending user turn

**An EOS-ended answer can be followed by another user message.** `<eos>` stops the
current generation; it is not a rule forbidding a later model call. However, the
pinned Gemma 4 template normally uses `<turn|>` to separate assistant and user
turns. A continuation retaining the *sampled* EOS is not byte-identical to a
conversation reconstructed through `apply_chat_template`. Treat that difference
as an explicit training policy, not as something the template guarantees.

## Evidence and implications

1. A [Transformers Gemma discussion](https://github.com/huggingface/transformers/issues/32110)
   records Gemma 1.1 models emitting `<eos>` without the template's end-of-turn
   marker. A Transformers maintainer states that models sometimes stop with EOS
   despite being trained on end-of-turn tokens, and advises formatting the message
   with the template when building the next chat turn; they did not observe a
   long-conversation performance regression. **This is Gemma 1.1/2 evidence,
   not a Gemma 4 token-ledger prescription.** Re-rendering the assistant would
   replace/normalize sampled tokens and is disallowed for our GRPO action ledger.
2. [Google's Gemma prompt guide](https://ai.google.dev/gemma/docs/core/prompt-structure)
   distinguishes role and turn delimiters and illustrates adjacent user/model
   turns separated by end-of-turn tokens. Its example is older Gemma syntax.
   [Google's Gemma 4 tool guide](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
   shows the newer `<|turn>user`, `<|turn>model`, `<turn|>` and
   `<|tool_response>` framing. It demonstrates parsing output and then rendering
   subsequent messages with the processor; it does not specify how to train a
   loss mask across a sampled EOS and a later user turn.
3. The pinned tokenizer/chat template used by production v2 emits
   `<turn|>\n<|turn>user\n…<turn|>\n<|turn>model\n` for an ordinary
   assistant-to-user transition. In a read-only audit, deriving the *external
   suffix after* the canonical `<turn|>` yields only
   `\n<|turn>user\n…<turn|>\n<|turn>model\n`. Appending those 25 masked
   tokens after the **actual** fifth sampled `<eos>` in v2 slot 002 preserves
   action IDs, passes token-ledger verification and stays well within the 24,576
   context envelope. It does **not** invent a missing next model response.
4. [Unsloth's Gemma 4 training guide](https://unsloth.ai/docs/models/gemma-4/train)
   warns that inference must use the same chat template and EOS configuration as
   training. That caution makes an actual-model continuation probe important;
   it does not by itself choose whether to add a second turn delimiter after EOS.

Reddit search returned generic “continue generation” and EOS-in-prompt discussions,
not a substantiated Gemma 4 multi-turn case. Attempts to fetch the relevant
[r/LocalLLaMA continuation thread](https://www.reddit.com/r/LocalLLaMA/comments/1deug0t/how_do_you_implement_a_continue_feature/)
and its old.reddit mirror were blocked/thin (403 and a network-policy page).
Do **not** attribute a recommendation to those posts from search snippets.
The quoted Gemini 3 “EOS fails to fire” loop concerns missing EOS; here EOS was
actually sampled and generation stopped as configured. These are different cases.

## Policy now in the branch

`native_suffix()` accepts sampled EOS for a **final answer followed by a user**;
it appends only the user-turn suffix as loss-masked environment tokens. It does not
replace EOS, insert `<turn|>` before the user, or relax tool-call boundaries.
Focused pinned-tokenizer tests verify that the next generation sees the EOS plus
user message and that all sampled action IDs and masks remain exact. This proves
protocol accounting, not that a Gemma 4 model will produce a useful next response
or that the noncanonical EOS/user adjacency is optimal. Require a separate
actual-model behavior probe and controlled GPU/source qualification before training
under this changed policy. The stopped v2 run remains immutable; no continuation
response exists in its evidence.
