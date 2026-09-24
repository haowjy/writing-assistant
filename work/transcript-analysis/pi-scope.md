# Scoping `pi` as a candidate harness

Read-only scope. Sources: pi v0.87.0 docs under
`/home/jimyao/.local/share/pnpm/global/5/.pnpm/@earendil-works+pi-coding-agent@0.87.0_.../`
(README.md, `docs/security.md`, `docs/containerization.md`, `docs/settings.md`,
`docs/extensions.md`, `docs/session-format.md`, `docs/json.md`, `docs/sdk.md`) and our
repo under `/home/jimyao/gitrepos/research/writing-assistant`. No source, tests, or runs
were modified. No model, network, or container calls were made.

---

## 1. What `pi` actually exposes

### Default tool surface

- Default startup tools are four: `read`, `write`, `edit`, `bash` (`README.md:91`).
- Full built-in set: `read`, `bash`, `powershell` (Windows), `edit`, `write`, `grep`,
  `find`, `ls` (`README.md:626`; `docs/settings.md:287`).
- These are selectable: `defaultTools` picks the enabled subset
  (`docs/settings.md:285-303`); `--tools` is a strict allowlist, `--no-builtin-tools`
  drops all built-ins while keeping extensions, `--no-tools` disables everything
  (`docs/settings.md:303`; `docs/usage.md:213-215`).
- Extensions can override or replace any built-in tool (`docs/extensions.md:2183-2196`),
  and built-in tool *operations* are pluggable — `read`/`bash`/`edit`/etc. can be routed
  to SSH, containers, or a micro-VM (`docs/extensions.md:2218-2260`).

### Shell, arbitrary paths, network

- `bash` is a real shell tool. "Built-in tools can read files, write files, edit files,
  and run shell commands with the permissions of the pi process"
  (`docs/security.md:31`). There is **no built-in sandbox** (`docs/security.md:29-35`).
- Paths are not confined to a workspace. Tools resolve against `cwd` plus absolute paths;
  nothing in the docs claims a path-allowlist boundary. A custom tool *can* implement
  access control (`docs/extensions.md:2218`; the `tool-override.ts` example is described
  as "logging and access control").
- Network reach is not a first-class built-in tool in the default set (no `web_fetch` /
  `web_search` among built-ins). But `bash` runs arbitrary processes, so `curl`, `pip`,
  `git`, etc. are available by default; the docs tell operators to "restrict network
  access when the task does not need it" precisely because the shell can reach it
  (`docs/security.md:52`). Pi itself also contacts `pi.dev` for update checks unless
  disabled (`README.md:325`, `docs/settings.md:107`).
- Output is truncated to 50 KB / 2000 lines per tool result (`docs/extensions.md`, Output
  Truncation section) — a context guard, not a security control.

### Transcript / event format

- Sessions are JSONL trees (`docs/session-format.md:1-7`). Messages are typed:
  `AssistantMessage.content` is an array of `TextContent | ThinkingContent | ToolCall`;
  tool output is a separate `ToolResultMessage` with `role: "toolResult"`,
  `toolCallId`, `toolName`, and `content` blocks (`docs/session-format.md:96,113-124`);
  shell `!` commands become `BashExecutionMessage` (`docs/session-format.md:147`).
- Non-interactive runs expose the same stream: `pi --mode json "..."` emits JSONL events
  (`docs/json.md:1-10`), and `--mode rpc` / the SDK expose `AgentSessionEvent`s
  (`docs/json.md:29-45`; `docs/sdk.md:280-325`). The SDK can constrain the tool set
  programmatically: `tools: ["read", "bash", ...]` plus `customTools`
  (`docs/sdk.md:1031-1032`).

Net: pi is a coding agent with a genuine shell, unconfined paths, and network reach, and
it says so explicitly. Any confinement would have to come from us (tool override) or from
the OS/container (`docs/security.md:33`).

---

## 2. What breaks in OUR pipeline

Our pipeline assumes (a) a workspace-scoped filesystem with a fixed five-tool schema,
(b) `before`/`after` whole-workspace snapshots as the observable artifact set, (c) a
Chat-Completions-shaped message list with `role: "tool"` observations, and (d) identity
and provenance hashed from our own harness. A pi candidate violates each.

### 2.1 The artifact set and `scoring.py`

`mechanical_score` reconstructs state entirely from `result["before"]` /
`result["after"]` and from declared `check["path"]` keys:

- `changed` is the diff of `before` vs `after` dict keys (`scoring.py:126-129`).
- Path checks read `check["path"] in result.get("after", {})` and pull text from
  `result["after"]` (`scoring.py:139-141`).
- KB checks filter `result["after"]` to `p.startswith("kb/")` (`scoring.py:160-168`,
  `229-231`).
- `evidence_exposed` reads trace events shaped as
  `e["type"] == "tool"` / `e["call"]["function"]["name"]` (`scoring.py:200-208`).
- Q4 counts trace events with `e["type"] == "tool"` and
  `e["observation"].get("valid", ...)` (`scoring.py:221-226`).

`before`/`after` are produced only by `Workspace.snapshot()` — a recursive scan of
`self.root` (`workspace.py:71-78`) — which is taken around `run_agent` in
`suite.py:212`, `223`, `226`. If a pi candidate writes to any path outside that root
(or reads a file it did not place there), it never enters `after`, so:

- a delivered artifact at an unexpected path is scored as absent;
- `allowed_changes` sees a smaller `changed` set and can pass a change that actually
  touched a path we never observed (`scoring.py:196-199`);
- the `_PASSES_ON_ABSENT` guard (`scoring.py:119`, `139-146`) still protects declared
  paths, but it cannot see a *different* path the agent used instead.

`aggregate_checks` marks a missing declared path as `unscored`/`missing_prose`
(`scoring.py:139-146`, `59-116`), so a pi run that writes a file anywhere the harness
did not snapshot fails completion for reasons that are about layout, not writing.
`artifacts.extract_prose` has the same dependency: file selectors index
`result["after"][selector["path"]]` or a turn `snapshot` (`artifacts.py:28-40`), and
duplicate suppression / span hashing assume those strings exist.

### 2.2 The loss-mask boundary and `training.py`

The claim "observations are masked out of the loss" is implemented in two places, both
specific to our message schema:

- `encode_trajectory` rejects any message carrying `thinking` / `reasoning` /
  `reasoning_content` (`training.py:74-76`) and normalizes `role == "tool"` message
  content to JSON (`training.py:82-87`) before calling `render_messages`
  (`training.py:88`). `render_messages` itself requires `role == "tool"`, a
  `tool_call_id`, and a preceding assistant `tool_calls` entry
  (`inference.py:82-95`).
- The mask is produced by rewriting the chat template so that only assistant spans get
  `{% generation %}` annotations, with tool-call spans and turn endings explicitly
  bracketed (`training.py:40-70`), then read back as `assistant_masks`
  (`training.py:102-115`). `data.validate_records` enforces the same role vocabulary
  (`data.py:69-76`) and that the trajectory ends on an assistant answer without pending
  tool calls (`data.py:101`).

A pi transcript is a **different schema**: assistant content is an array of typed blocks
including `ThinkingContent`, and tool output is `role: "toolResult"` with `content`
blocks and `details` (`docs/session-format.md:96,113-124,240`). Consequences if we feed
pi output to `encode_trajectory` unchanged:

- The `thinking`/`reasoning` guard at `training.py:74-76` rejects the run (pi records
  reasoning as a reasoning/thinking field/block), so preparation fails rather than
  silently supervising it.
- Even after conversion, the boundary is *reconstructable but not preserved by pi*. Pi
  keeps assistant vs toolResult roles, so the conceptual boundary survives; what does
  not survive is our native Gemma template annotation, because it keys on our exact
  `message["tool_calls"]` shape and our `role == "tool"` content
  (`training.py:42-43,59-60`; `inference.py:82-95`). A converter would have to rebuild
  that shape, and any field it misses becomes a supervision-leak bug. The masking is not
  automatic.

### 2.3 Identity, layout, and resume (`suite.py`, `inference.py`)

- `run_selected` computes `identity` from `visible`, `observation`, `model`, and a
  `harness` hash (`suite.py:154-172`). That `harness` hash is
  `fingerprint({p.name: fingerprint(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))})`
  — our Python source files only (`suite.py:154-159`). Running the candidate inside pi
  means the actual harness is Node/TypeScript, so `harness_hash` would no longer identify
  what produced the result; provenance and resume identity silently lose their meaning
  (`suite.py:237-249`).
- Layout is `destination / identity / attempt-NNNN-xxxx` with `started.json`,
  `visible.json`, `trace.jsonl`, `result.json` (`suite.py:173-196`, `235-253`). Pi writes
  its sessions to `~/.pi/agent/sessions/<cwd-slug>/...` by default
  (`docs/session-format.md:9-13`; `docs/settings.md` `sessionDir`), not into the attempt
  directory. Any pi run would require copying/repointing its session file to make the
  attempt self-contained for `saved_results` (`suite.py:372-382`) and `_read_attempt`
  (`suite.py:337-367`).
- `result` is expected to carry `status`, `output`, `before`, `after`, `turns`, `trace`,
  `usage`, `latency_seconds` (`suite.py:214-249`). Pi's `turn_end` carries
  `message` + `toolResults` (`docs/json.md:40-45`); there is no `before`/`after`
  snapshot, no per-turn workspace snapshot, and no `read_tokens` counter. The turn
  snapshot is load-bearing: `artifacts.extract_prose` selects from
  `result["turns"][...]["snapshot"]` (`artifacts.py:31-34`).
- `inference.render_messages` is only used for our Gemma path (`inference.py:75-95`);
  the pi transcript cannot be fed into `TransformersBackend.complete` without a converter,
  and pi drives its own model loop rather than calling our `Backend`
  (`agent.py:79`).

### 2.4 Declared checks and `reward.py`

`reward.mechanics_score` is keyed on check dicts with `id`, `applicable`, and `passed`
(`reward.py:74-99`), and `declared_check_set` freezes those IDs before sampling
(`reward.py:95-99`). The check outcomes are the ones `scoring.mechanical_score` produced
from the `before`/`after` snapshots (`scoring.py:122-216`). So reward inherits every
breakage above:

- a check whose declared path the pi agent wrote elsewhere becomes `unscored`
  (`scoring.py:139-146`) and, under `mechanics_score`'s "applicable" filter, either drops
  out or fails, changing the reward denominator (`reward.py:83-91`);
- because reward is computed from the *task's* declared check set rather than from a
  known artifact inventory, an agent that can create arbitrary files can satisfy a
  `contains`/`nonempty` check from a file the evaluator never designated — reward would
  not notice unless the check pinned the path (and path checks are only as good as the
  snapshot, §2.1).
- `critical_failures` can only fire on IDs that were declared (`reward.py:101-113`); a
  pi agent that, say, writes outside the workspace to accomplish a task produces no
  declared failure, so the cap never triggers.

### 2.5 Tool-availability validation

`compile_scenarios` rejects any tool name not in `TOOL_SCHEMAS` (`suite.py:66-68`), and
`run_agent` builds its allowlist from the same five schemas (`agent.py:37-41`). Pi's tool
names (`bash`, `grep`, `find`, `ls`, `edit`, `toolResult`, ...) do not exist in
`TOOL_SCHEMAS` (`workspace.py:109-136`), so pi's surface cannot be declared, validated, or
budgeted by the current scenario contract (including `max_read_tokens`, applied only to
`read_file`/`search`/`list_dir`, `agent.py:144-157`).

---

## 3. Does adding execution force containerization?

**Yes — for any untrusted/unattended candidate, granting a real shell forces an
OS-level container or VM boundary; Docker alone is sufficient and
`nvidia-container-toolkit` is not required.**

Reasoning, grounded in the docs:

- Pi states its built-in boundary is not a sandbox: no built-in sandbox
  (`docs/security.md:29`), tools run with the pi process's full permissions
  (`docs/security.md:31`), and project trust is only an input-loading guard, not a
  capability limit (`docs/security.md:25`).
- The docs' own guidance for "generated code you do not intend to monitor closely, or
  unattended automation" is to run inside a container/VM/sandbox with only the required
  files and credentials (`docs/security.md:41-53`). Candidate rollouts are exactly that.
- Our own design already treats this as the standard: the Codex grader runs
  `--sandbox read-only` with shell/exec/web-search disabled
  (`grading.py:210-227`), and the project context says "these path-constrained file tools
  are not an OS sandbox; no arbitrary shell is exposed to candidates"
  (`src/.context/CONTEXT.md:44-45`). If we expose a shell to the *candidate*, the
  existing containment story no longer applies.

Why Docker is enough and GPU toolkit is not needed here: `nvidia-container-toolkit` (not
installed; `nvidia-ctk`/`nvidia-container-toolkit` absent from PATH) only matters if the
GPU/model must be visible *inside* the container. The Plain Docker pattern runs only the
pi process in the container and passes provider keys in
(`docs/containerization.md:14,46-75`). In our architecture the candidate model is loaded
on the host or served over HTTP (`backends.ChatServerBackend`, `backends.py:43-89`;
`inference.load_checkpoint`, `inference.py:232-334`), so pi can run CPU-only in a container
while inference stays on the host. The container needs no GPU; hence no
`nvidia-container-toolkit`. The Gondolin/QEMU and OpenShell routes
(`docs/containerization.md:13-45,110+`) are heavier alternatives and are not required by
the Docker path.

Caveat: bind-mounting the host workspace read/write still lets the container modify host
files (`docs/security.md:53`); a candidate that can execute code can write through that
mount. Read-only mounts or copy-in/copy-out are required to bound writes.

---

## 4. The minimal path that gets the benefit without the blast radius

**Add one bounded execution tool to `Workspace`; do not migrate to pi wholesale.**

The benefit worth having is "the candidate can run code/tests against its own artifacts
and observe failures." That does not require a general shell, pi's tool surface, or a new
transcript format. It requires one new entry in `TOOL_SCHEMAS` plus a `Workspace` method
that:

1. takes a command (or a script path already inside `self.root`),
2. runs it via `subprocess` with `cwd=self.root`, a wall-clock timeout, output-size cap
   (mirroring the existing `max_bytes`), and a fixed interpreter/allowlist,
3. applies the same path resolution as the other tools (`workspace.py:19-27`) and returns
   the observation in the existing `{"ok", "valid", "result"|"error"}` shape
   (`workspace.py:137-157`).

This slots into `agent.py`'s existing dispatch (`agent.py:143`), keeps the
`before`/`after` snapshot model intact (`suite.py:212-226`), keeps `scoring.py`'s
snapshot/trace contract (`scoring.py:126-146,200-226`), and keeps the native Gemma loss
mask and `role: "tool"` boundary unchanged (`training.py:40-115`;
`inference.py:75-95`). It is also consistent with the project's stated separation of
network execution: the Codex grader is already run as an isolated subprocess with an
explicit sandbox and disabled capabilities (`grading.py:210-299`), so a bounded runner is
an existing pattern, not a new architecture.

The intermediate option — run the candidate under pi but constrain it — is possible
(`tools: ["read", "bash"]` via the SDK, `docs/sdk.md:1031`; `--tools` allowlist,
`docs/settings.md:303`; built-in override, `docs/extensions.md:2183`), but it still leaves
the transcript, identity, and layout mismatches of §2.2–§2.4 to bridge, and it still needs
a container. It is more work than one tool for the same "can execute code" benefit.

---

## 5. Does AgentGym's uniform HTTP environment API argument apply?

The AgentGym design described in the prompt (uniform `/createEnv`, `/observation`,
`/available_actions`, `/step`, `/reset` endpoints instead of a general shell) applies to
us, for four concrete reasons already visible in our codebase:

1. **Bounded action space → bounded observability.** AgentGym's `/available_actions` is
   the analogue of our `TOOL_SCHEMAS` allowlist (`workspace.py:109-136`;
   `agent.py:37-41`). `scoring.py` can only reason about tool events it knows
   (`scoring.py:200-226`); a general shell produces an open-ended event stream that no
   declared-check set can enumerate (`reward.py:95-99`).
2. **Step/reset maps to isolated workspaces and resume.** AgentGym's `/step` + `/reset`
   is our per-attempt `Workspace` and clean retry (`suite.py:173-196`). A shell agent's
   side effects are not confined to that reset boundary (§2.1).
3. **Uniform observation shape preserves the supervision boundary.** Our loss mask
   depends on a regular `role: "tool"` observation (`training.py:82-115`;
   `inference.py:82-95`). A uniform environment response is exactly what keeps that
   regular. Pi's typed-block transcript does not (`docs/session-format.md:96-124`).
4. **Reproducibility/comparability.** Our identity hash and reward denominators assume
   declared, frozen inputs (`suite.py:154-172`; `reward.py:95-99`). A shell makes the
   effective action set and environment state part of the measurement without being
   declared.

Named uncertainty: I did not fetch arXiv 2406.04151 (network/paper access was out of
scope and the constraint forbids unnecessary external calls), so this answers the
*argument as stated in the prompt*, not a verified reading of the paper. If the paper
makes additional claims about sandboxing or evaluation harnesses, they are not reflected
here.

---

## Recommendation

1. **Do not adopt pi as the candidate harness.** It would invalidate our snapshot-based
   scoring (`scoring.py:126-146`), our native Gemma loss masking (`training.py:40-115`;
   `inference.py:75-95`), our attempt identity/provenance (`suite.py:154-172,237-249`),
   and our declared-check reward (`reward.py:74-99`) — each fix is a bespoke converter,
   and each converter is a place for the tool-observation boundary to leak.
2. **If execution capability is the goal, add a single bounded, subprocess-based
   execution tool to `Workspace`** (section 4), gated behind the existing scenario
   `tools` allowlist (`suite.py:66-68`) and run in an OS container with the workspace
   mounted read-only or copy-in/copy-out (`docs/security.md:53`). Docker is sufficient;
   no `nvidia-container-toolkit` is needed because inference stays on the host
   (`docs/containerization.md:46`).
3. **If pi is ever used, treat it as a separate harness identity**, not as a drop-in
   backend: extend `candidate_metadata` with a `pi` harness (`scoring.py:37-51`), hash
   the pi version/config instead of our Python source (`suite.py:154-159`), and write a
   converter plus a test that asserts pi `toolResult` content is masked out of
   `assistant_masks` (`training.py:102-115`). Until that converter exists, pi transcripts
   should not enter `training.encode_trajectory`.

### Contradiction / gap flagged

- `src/.context/CONTEXT.md:44-45` states the file tools are not an OS sandbox and no
  arbitrary shell is exposed. That is consistent with `workspace.py:19-27` and remains
  true today; this note records that adding a pi-style shell would be a deliberate
  departure from that contract, not an extension of it.
- The pi docs are explicit that there is no built-in sandbox (`docs/security.md:29-35`)
  and silent on any in-process path allowlist for built-in tools. No built-in tool is
  documented to confine reads/writes to `cwd`. I found no doc claim to the contrary.
