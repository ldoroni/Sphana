"""Length-prefixed msgpack framing for TCP RPC communication.

Wire format:
    [4 bytes big-endian uint32: payload length][msgpack payload]

All messages are serialized/deserialized as Python dicts using msgpack.
"""

from __future__ import annotations

import struct
from typing import Any

import msgpack

# 4-byte big-endian unsigned int header
_HEADER_FORMAT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

# Maximum message size: 64 MiB (safety limit)
MAX_MESSAGE_SIZE = 64 * 1024 * 1024


def encode_message(data: dict[str, Any]) -> bytes:
    """Serialize a dict to a length-prefixed msgpack frame.

    Args:
        data: The message payload as a Python dict.

    Returns:
        Bytes containing the 4-byte length header followed by the
        msgpack-encoded payload.

    Raises:
        ValueError: If the serialized payload exceeds MAX_MESSAGE_SIZE.
    """
    payload = msgpack.packb(data, use_bin_type=True)
    length = len(payload)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Message size {length} exceeds maximum {MAX_MESSAGE_SIZE}"
        )
    header = struct.pack(_HEADER_FORMAT, length)
    return header + payload


def decode_header(header_bytes: bytes) -> int:
    """Extract the payload length from a 4-byte header.

    Args:
        header_bytes: Exactly 4 bytes read from the socket.

    Returns:
        The payload length as an integer.

    Raises:
        ValueError: If header_bytes is not exactly 4 bytes or the
            decoded length exceeds MAX_MESSAGE_SIZE.
    """
    if len(header_bytes) != _HEADER_SIZE:
        raise ValueError(
            f"Header must be {_HEADER_SIZE} bytes, got {len(header_bytes)}"
        )
    (length,) = struct.unpack(_HEADER_FORMAT, header_bytes)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Payload length {length} exceeds maximum {MAX_MESSAGE_SIZE}"
        )
    return length


def decode_payload(payload_bytes: bytes) -> dict[str, Any]:
    """Deserialize a msgpack payload into a Python dict.

    Args:
        payload_bytes: The raw msgpack bytes.

    Returns:
        The deserialized Python dict.
    """
    return msgpack.unpackb(payload_bytes, raw=False)


def header_size() -> int:
    """Return the size of the length-prefix header in bytes."""
    return _HEADER_SIZE