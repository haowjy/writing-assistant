---
name: subagent
description: Spawn for a bounded task no specialist agent fits. Pass the full task and the context it needs.
mode: subagent
model: luna
effort: high
model-policies:
- match:
    alias: luna
  override:
    effort: high
- match:
    alias: deepseek
  override:
    effort: high
- match:
    alias: sol
  override:
    effort: high
- match:
    alias: hydrogen
  override:
    effort: medium
- match:
    alias: sonnet
  override:
    effort: medium
sandbox: danger-full-access
approval: auto
tools:
  bash(killall *): deny
  bash(pkill -f *): deny
---
