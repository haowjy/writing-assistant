"""Creative-writing benchmark and RL environment.

Layer order (enforced by tests/test_layering.py): episode -> llm, search -> judges ->
corpus -> suites, drivers -> run -> cli. Nothing imports upward.
"""
