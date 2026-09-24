# What would change if candidates ran inside pi

This report assesses replacing our five file tools with pi, a general coding-agent harness, and explains what would stop working in evaluation and training. The source recommends keeping our harness and adding one contained execution tool if running code is the goal: pi introduces shell access and different records, so adopting it requires both operating-system isolation and explicit conversion of its outputs.

A harness is the program that gives a model tools, runs its actions and records the results. The candidate is the model being tested, rather than the separate model that judges its output. Our candidate currently receives `list_dir`, `read_file`, `search`, `write_file` and `patch_file`: listing, reading, searching, writing and editing within a project workspace. The knowledge base, abbreviated KB, is a set of Markdown reference files under `kb/`.

This is a rewrite of a read-only assessment of pi version 0.87.0, not a fresh test of current pi. Its sources are the locally installed README and `docs/security.md`, `docs/containerization.md`, `docs/settings.md`, `docs/extensions.md`, `docs/session-format.md`, `docs/json.md` and `docs/sdk.md` under `/home/jimyao/.local/share/pnpm/global/5/.pnpm/@earendil-works+pi-coding-agent@0.87.0_.../`, plus this repository. That installation path is abbreviated in the source. References below retain the original file names and line numbers. The assessment made no model, network or container calls and modified no source, tests or runs.

## Pi can execute commands with the permissions of its process

Pi starts with four tools enabled: `read`, `write`, `edit` and `bash` (`README.md:91`). Its full built-in set is `read`, `bash`, `powershell` on Windows, `edit`, `write`, `grep`, `find` and `ls` (`README.md:626`; `docs/settings.md:287`). The first group reads and changes files; the shell tools execute commands; the remaining tools search or list files.

The enabled set is configurable. `defaultTools` selects tools (`docs/settings.md:285–303`); `--tools` is a strict list of permitted names; `--no-builtin-tools` removes built-ins while retaining extensions; and `--no-tools` disables everything (`docs/settings.md:303`; `docs/usage.md:213–215`). Extensions can replace built-ins, and the underlying operations can be redirected to a remote machine over SSH, a container or a small virtual machine (`docs/extensions.md:2183–2196`, `2218–2260`). A software development kit, or SDK, also selects tools programmatically through `tools` and `customTools` (`docs/sdk.md:1031–1032`).

Those selection mechanisms do not themselves confine a command's effects. Pi documents no built-in sandbox: its tools can read, write, edit and execute with the pi process's permissions (`docs/security.md:29–35`). Built-in paths resolve against the current working directory, called `cwd`, or use absolute paths. The reviewed documentation makes no claim that all paths are restricted to a workspace. A custom override can implement access control; the `tool-override.ts` example is described that way (`docs/extensions.md:2218`).

There is no built-in `web_fetch` or `web_search`, but shell commands such as `curl`, `pip` and `git` can reach the network when the process can. This is why the documentation tells operators to restrict unnecessary network access (`docs/security.md:52`). Pi itself also checks `pi.dev` for updates unless disabled (`README.md:325`; `docs/settings.md:107`). Tool output is truncated at 50 KB or 2,000 lines per result (`docs/extensions.md`, Output Truncation); that limits text entering context, not file or network access.

The practical consequence is that pi's process permissions become the candidate's reach. Confinement must come from our tool implementation or the operating system, rather than an assumption that the working directory forms a security boundary (`docs/security.md:33`).

## Pi records a different kind of transcript

Pi sessions are trees of JSONL entries: JSONL stores one JSON record per line (`docs/session-format.md:1–7`). An assistant message has a `content` array containing typed text, thinking or tool-call blocks. A tool response is a separate `ToolResultMessage` with `role: "toolResult"`, `toolCallId`, `toolName` and content blocks (`docs/session-format.md:96`, `113–124`). Shell commands entered with `!` produce `BashExecutionMessage` records (`docs/session-format.md:147`).

Noninteractive `pi --mode json "..."` produces a JSONL event stream (`docs/json.md:1–10`). Remote procedure call mode, `--mode rpc`, and the SDK expose `AgentSessionEvent` records (`docs/json.md:29–45`; `docs/sdk.md:280–325`). These are usable records, but our code expects a Chat-Completions-style message list with tool observations labelled `role: "tool"`. Renaming a runner does not reconcile those structures.

## Our scorer would miss actions outside its workspace snapshot

Our result contains `before` and `after` dictionaries mapping workspace paths to file contents. `Workspace.snapshot()` recursively scans only `self.root`, the designated workspace root (`workspace.py:71–78`); `suite.py` takes those snapshots around `run_agent` (`suite.py:212`, `223`, `226`). The scorer depends on this inventory:

- `mechanical_score` derives changed paths by comparing the dictionaries (`scoring.py:126–129`).
- Path checks inspect the declared `check["path"]` in `after` (`scoring.py:139–141`).
- Knowledge-base checks select paths beginning `kb/` (`scoring.py:160–168`, `229–231`).
- `evidence_exposed`, which checks whether a required fact appeared in a tool observation, reads events with `type == "tool"` and `call.function.name` (`scoring.py:200–208`).
- Q4, the internal tool-correctness measure, counts those same events and their `observation.valid` status (`scoring.py:221–226`).

If pi writes outside the scanned root, the output is invisible to `after`. An otherwise delivered artifact at an unexpected location can be scored absent. Conversely, `allowed_changes`, the rule restricting modified paths, can pass because the snapshot never saw an out-of-workspace change (`scoring.py:196–199`). Reading an outside file also introduces evidence beyond the declared workspace without making it part of the file inventory.

The `_PASSES_ON_ABSENT` guard handles missing declared paths (`scoring.py:119`, `139–146`), but cannot discover another path the candidate chose. Missing declared prose becomes `unscored` or `missing_prose` during scoring and aggregation (`scoring.py:139–146`, `59–116`), potentially failing completion because of delivery layout rather than writing quality.

Prose extraction has the same dependency. `artifacts.extract_prose` reads the selected path from `after` or a per-turn snapshot (`artifacts.py:28–40`). Duplicate suppression and span hashing—the comparison of selected text and its fingerprints—assume those strings exist. Supporting pi therefore requires preserving the observed file inventory and adapting its events, not merely accepting its final chat response.

## Training needs a converter to preserve what is and is not learned

The loss mask determines which transcript tokens contribute to training. Our intended boundary trains on assistant output while excluding observations returned by tools. This prevents the training process from treating file contents or tool results as text the assistant should have generated.

Our `encode_trajectory` function rejects messages carrying `thinking`, `reasoning` or `reasoning_content` fields (`training.py:74–76`). It serializes `role == "tool"` content as JSON before calling `render_messages` (`training.py:82–88`). That renderer requires a tool message, a `tool_call_id` and a preceding assistant `tool_calls` entry (`inference.py:82–95`). `data.validate_records` enforces the same role vocabulary and requires a final assistant answer without pending tool calls (`data.py:69–76`, `101`).

The training template marks assistant spans with `{% generation %}`, including explicit boundaries for calls and turn endings (`training.py:40–70`). The tokenizer returns those boundaries as `assistant_masks` (`training.py:102–115`). Pi's content blocks and `toolResult` role do not automatically produce that native Gemma structure. A converter must rebuild it, and an incorrect conversion could accidentally train on tool observations.

The source says feeding pi output unchanged would trigger the thinking/reasoning rejection. There is an unresolved detail: pi's thinking can be nested in typed content blocks, whereas the described guard checks message fields. The two schemas are incompatible, but the exact first rejection path is not proven by this description. It needs a concrete conversion test rather than an assumption.

Pi retains the conceptual assistant-versus-tool distinction, so the boundary can be reconstructed. It does not preserve our exact `message["tool_calls"]` structure, `role == "tool"` content or Gemma template annotations (`training.py:42–43`, `59–60`; `inference.py:82–95`; `docs/session-format.md:96`, `113–124`, `240`). Until a converter exists and is checked, the source recommends keeping pi transcripts out of `training.encode_trajectory`.

## Attempt identity and saved results would need to include pi

`run_selected` calculates an identity from visible task inputs, observations, model metadata and a harness hash (`suite.py:154–172`). A hash is a fingerprint used here to tell whether the producing configuration has changed. The current harness fingerprint covers only this package's Python source files:

```python
fingerprint({p.name: fingerprint(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))})
```

If a Node/TypeScript pi process actually runs the candidate, that fingerprint no longer describes the whole producer. Provenance—being able to identify how a result was made—and resume decisions would silently rely on incomplete identity information (`suite.py:154–159`, `237–249`). Pi's version and configuration must be included as a separate harness identity.

Our attempt layout is `destination / identity / attempt-NNNN-xxxx`, containing `started.json`, `visible.json`, `trace.jsonl` and `result.json` (`suite.py:173–196`, `235–253`). Pi normally saves sessions under `~/.pi/agent/sessions/<cwd-slug>/...`, where the slug identifies the working directory (`docs/session-format.md:9–13`; `docs/settings.md`, `sessionDir`). Its session must be redirected or copied into the attempt directory for `saved_results` and `_read_attempt` to recover a self-contained result (`suite.py:372–382`, `337–367`).

Our result also requires `status`, `output`, `before`, `after`, `turns`, `trace`, `usage` and `latency_seconds` (`suite.py:214–249`). Pi's `turn_end` contains `message` and `toolResults`; it has no equivalent workspace snapshots, per-turn file snapshots or `read_tokens` counter (`docs/json.md:40–45`). The per-turn snapshot matters because prose extraction can select a file as it existed on a particular turn (`artifacts.py:31–34`).

Pi runs its own model loop instead of calling our `Backend` interface (`agent.py:79`). Its transcript cannot be passed directly to `TransformersBackend.complete`, and our `inference.render_messages` is specific to the Gemma route (`inference.py:75–95`). Those interfaces would need deliberate bridging.

## Reward would inherit missing evidence and changed denominators

`reward.mechanics_score` uses checks with `id`, `applicable` and `passed` fields (`reward.py:74–99`). `declared_check_set` fixes the expected check identifiers before sampling (`reward.py:95–99`). The outcomes come from workspace snapshots and known trace shapes (`scoring.py:122–216`), so adapting the runner without adapting observation changes the reward calculation.

A file written at an unobserved path can leave its intended check unscored (`scoring.py:139–146`). Depending on the applicable-check filter, it then drops out or fails, changing the denominator (`reward.py:83–91`). `critical_failures` can penalize only declared identifiers (`reward.py:101–113`); an outside write with no corresponding declared failure cannot trigger that cap.

The source also warns that an arbitrary file could satisfy a `contains` or `nonempty` check when its path is not pinned. That mechanism is underexplained in the original: its earlier account says path checks read only declared paths, while outside files are absent from snapshots. The warning is retained as an unresolved risk claim, not a demonstrated exploit. A concrete check and event sequence are needed to establish when it applies.

## Pi's tools cannot be declared under the current scenario contract

`compile_scenarios` rejects names absent from `TOOL_SCHEMAS`, our five-tool schema (`suite.py:66–68`; `workspace.py:109–136`). `run_agent` builds its permitted list from the same schema (`agent.py:37–41`). Pi names such as `bash`, `grep`, `find`, `ls` and `edit` therefore cannot be declared or validated as our existing tools.

Read budgets would also change. `max_read_tokens`, the cap on tokens exposed by reads, currently applies only to `read_file`, `search` and `list_dir` (`agent.py:144–157`); a shell could display a file through another command. The source includes `toolResult` in its list of incompatible tool names, although `toolResult` is a message role rather than a tool. Both the tool-name mismatch and the message-role mismatch are real, but they require different conversions.

## Unattended shell access needs an operating-system boundary

The source's answer is yes: granting a real shell to an untrusted or unattended candidate requires a container, virtual machine or equivalent operating-system sandbox. Pi's project-trust setting governs input loading, not process capabilities (`docs/security.md:25`). Its documentation recommends isolation with only required files and credentials for generated code that will not be closely monitored and for unattended automation (`docs/security.md:41–53`). Candidate rollouts fit that use.

Our current design explicitly says the path-restricted file tools are not an operating-system sandbox and expose no arbitrary shell (`src/.context/CONTEXT.md:44–45`; `workspace.py:19–27`). The separate Codex grader uses `--sandbox read-only` with shell, execution and web search disabled (`grading.py:210–227`). Adding candidate shell execution would deliberately change the current containment contract.

The source considers ordinary Docker sufficient for this design; it does not require `nvidia-container-toolkit`. That toolkit makes a GPU available inside a container. The assessment found neither `nvidia-ctk` nor `nvidia-container-toolkit` on the executable search path, but the proposed pi container needs no GPU: the model can remain loaded on the host or be served over HTTP (`backends.ChatServerBackend`, `backends.py:43–89`; `inference.load_checkpoint`, `inference.py:232–334`). Only pi runs in the CPU-only container, communicating with the model service. The pi Plain Docker example similarly runs the process in a container and passes provider credentials in (`docs/containerization.md:14`, `46–75`). Gondolin/QEMU and OpenShell are heavier alternatives, not requirements of that Docker route (`docs/containerization.md:13–45`, `110+`).

A container with a host directory mounted read/write can still modify that host directory (`docs/security.md:53`). The source therefore requires read-only mounts or copying inputs into an isolated workspace and copying approved outputs back out. “Docker is sufficient” is an architecture recommendation, not evidence from a tested configuration in this assessment.

## The proposed smaller change is one contained execution tool

The desired benefit is for a candidate to run code or tests against its own artifacts and inspect failures. The source recommends adding one operation to `Workspace` and one entry to `TOOL_SCHEMAS` instead of replacing the entire runner.

The operation would accept a command or a script already inside `self.root`, launch it through `subprocess` with that working directory, impose a wall-clock timeout and output-size cap comparable to the existing `max_bytes`, and use a fixed interpreter or permitted-command list. It would apply existing path resolution (`workspace.py:19–27`) and return the current observation shape: `{"ok", "valid", "result"|"error"}` (`workspace.py:137–157`). Execution would be allowed only in scenarios listing the tool (`suite.py:66–68`).

These choices preserve dispatch (`agent.py:143`), before/after snapshots (`suite.py:212–226`), scoring's trace and file contracts (`scoring.py:126–146`, `200–226`), and the native Gemma training mask (`training.py:40–115`; `inference.py:75–95`). The existing isolated grader subprocess supplies a related pattern (`grading.py:210–299`).

The runner still needs the operating-system containment described above. A working directory, timeout, output cap and fixed interpreter do not by themselves stop executed code accessing other paths. The source's final recommendation includes a container, but its initial minimal-tool description does not specify how network reach, writable scratch space, returned artifacts and the model-service connection would be configured. Those implementation choices remain open.

Running pi with fewer tools is also possible—through SDK `tools: ["read", "bash"]`, `--tools`, or built-in overrides (`docs/sdk.md:1031`; `docs/settings.md:303`; `docs/extensions.md:2183`). It still requires the transcript converter, revised identity and layout, and container boundary. That is why the source judges it more work for the same code-execution benefit.

If pi is adopted later, the source calls for a `pi` harness entry in `candidate_metadata` (`scoring.py:37–51`), a fingerprint of pi's version/configuration rather than only our Python files (`suite.py:154–159`), and a converter test proving that `toolResult` content is excluded from `assistant_masks` (`training.py:102–115`).

## The environment-interface argument applies, but the paper was not checked

The source considers an AgentGym design described in its prompt: a uniform HTTP interface with `/createEnv`, `/observation`, `/available_actions`, `/step` and `/reset`. These mean creating an environment, reading its state, listing permitted actions, taking an action and resetting state. It identifies four correspondences with our system:

1. A declared action list resembles `TOOL_SCHEMAS` and the agent's permitted tools (`workspace.py:109–136`; `agent.py:37–41`). The scorer recognizes known events; arbitrary shell commands introduce effects its declared checks may not cover (`scoring.py:200–226`; `reward.py:95–99`).
2. Step/reset resembles one isolated workspace per attempt and a clean retry (`suite.py:173–196`). Shell side effects outside that workspace survive its reset unless separately contained.
3. Uniform observations make it possible to preserve the assistant/tool training boundary (`training.py:82–115`; `inference.py:82–95`). Pi's typed-block format needs conversion before that boundary reaches our template (`docs/session-format.md:96–124`).
4. Reproducible comparisons require declaring effective inputs, capabilities and state. Our identity and reward denominator assume that declaration (`suite.py:154–172`; `reward.py:95–99`); an unrestricted shell changes the environment being measured without automatically recording it.

The author did not fetch arXiv 2406.04151 because paper/network access was outside that assessment's scope. These points assess the argument as supplied in the prompt, not a verified reading of AgentGym. Additional claims in the paper about sandboxing or evaluation are not covered.

The original assessment is [pi-scope.md](../../transcript-analysis/pi-scope.md). Its unresolved converter, reward-path and containment details are recorded in [the audit](../lane-a-audit.md); resolving them requires implementation evidence beyond a writing rewrite.
