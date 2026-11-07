"""The little-endian u32 codec for persisted task-graph token ledgers."""

from __future__ import annotations


def encode_u32_token_ids(tokens: tuple[int, ...]) -> bytes:
    """Encode token IDs as consecutive little-endian unsigned 32-bit integers."""
    if not isinstance(tokens, tuple) or any(type(token) is not int for token in tokens):
        raise TypeError("token IDs must be a tuple of integers")
    if any(token < 0 or token >= 2**32 for token in tokens):
        raise OverflowError("token ID exceeds the u32 ledger codec")
    return b"".join(token.to_bytes(4, "little") for token in tokens)


def decode_u32_token_ids(data: bytes, count: int | None = None) -> tuple[int, ...]:
    """Decode little-endian u32 IDs, optionally requiring an exact committed count."""
    if not isinstance(data, bytes):
        raise ValueError("token IDs must be bytes")
    if count is None:
        if len(data) % 4:
            raise ValueError("token bytes are not aligned to the u32 ledger codec")
        count = len(data) // 4
    if type(count) is not int or count < 0:
        raise ValueError("token count must be a nonnegative integer")
    if len(data) != 4 * count:
        raise ValueError("token bytes do not match their committed count")
    return tuple(
        int.from_bytes(data[offset : offset + 4], "little") for offset in range(0, len(data), 4)
    )


__all__ = ["decode_u32_token_ids", "encode_u32_token_ids"]
