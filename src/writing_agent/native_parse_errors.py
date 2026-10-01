"""Pinned malformed-output messages shared by native inference and protocol parsing.

This leaf module stays independent of the writing-agent runtime so the task-graph
experiment identity can bind the exact classifier behavior through ``native_*.py``.
"""

NATIVE_TOOL_CALL_OUTPUT_INCOMPLETE = "Native tool-call output was not completely parsed"
NATIVE_TOOL_ARGUMENTS_NOT_OBJECT = "Native tool arguments must be an object"

_NATIVE_OUTPUT_PARSE_ERROR_PREFIXES = (
    "json: could not parse after dialect transforms",
    "json parser could not parse region as JSON",
    "json: input contains reserved sentinel characters",
    "Required response_template fields missing from parsed output:",
    NATIVE_TOOL_CALL_OUTPUT_INCOMPLETE,
    NATIVE_TOOL_ARGUMENTS_NOT_OBJECT,
)


def is_native_output_parse_error(error: ValueError) -> bool:
    """Recognize only pinned parser errors attributable to model-output text."""
    return str(error).startswith(_NATIVE_OUTPUT_PARSE_ERROR_PREFIXES)
