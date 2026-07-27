# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Low-level encoding primitives for the loom bytecode format.

Provides LEB128 varint encoding, little-endian fixed-width integers,
and a ByteBuffer class for building binary data with position tracking.

LEB128 (Little-Endian Base 128) is the same encoding used by DWARF,
protobuf, and MLIR bytecode. Each byte uses 7 bits for data and 1 bit
(high bit) as a continuation flag.

Signed varints use zigzag encoding before LEB128:
  encoded = (value << 1) ^ (value >> 63)
This maps small-magnitude values (0, -1, 1, -2, 2, ...) to small
unsigned values (0, 1, 2, 3, 4, ...) for compact encoding.
"""

from __future__ import annotations

import struct

__all__ = [
    "ByteBuffer",
    "encode_varint",
    "decode_varint",
    "encode_signed_varint",
    "decode_signed_varint",
]


# ============================================================================
# LEB128 varint encoding
# ============================================================================


def encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as LEB128."""
    if value < 0:
        raise ValueError(f"encode_varint requires non-negative value, got {value}")
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)


def decode_varint(data: bytes | bytearray, offset: int = 0) -> tuple[int, int]:
    """Decode an unsigned LEB128 varint.

    Returns (value, new_offset) where new_offset is the position after
    the varint's last byte.
    """
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        payload = byte & 0x7F
        if shift == 63 and payload > 1:
            raise ValueError("varint too large (> 64 bits)")
        value |= payload << shift
        if (byte & 0x80) == 0:
            if shift > 0 and payload == 0:
                raise ValueError("non-canonical varint encoding")
            return value, offset
        shift += 7
        if shift >= 64:
            raise ValueError("varint too large (> 64 bits)")
    raise ValueError("unterminated varint at end of data")


def encode_signed_varint(value: int) -> bytes:
    """Encode a signed integer as zigzag + LEB128."""
    zigzag = (value << 1) ^ (value >> 63)
    return encode_varint(zigzag)


def decode_signed_varint(data: bytes | bytearray, offset: int = 0) -> tuple[int, int]:
    """Decode a signed zigzag + LEB128 varint.

    Returns (value, new_offset).
    """
    zigzag, offset = decode_varint(data, offset)
    value = (zigzag >> 1) ^ -(zigzag & 1)
    return value, offset


# ============================================================================
# ByteBuffer
# ============================================================================


class ByteBuffer:
    """Growable byte buffer for building binary data.

    Provides typed write methods for little-endian integers, varints,
    strings, and raw bytes. Tracks position for offset calculations.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data = bytearray()

    @property
    def position(self) -> int:
        """Current write position (byte offset from start)."""
        return len(self._data)

    def get_bytes(self) -> bytes:
        """Return the buffer contents as immutable bytes."""
        return bytes(self._data)

    # --- Fixed-width integers (little-endian) ---

    def write_u8(self, value: int) -> None:
        """Write an unsigned 8-bit integer."""
        self._data.append(value & 0xFF)

    def write_u16_le(self, value: int) -> None:
        """Write an unsigned 16-bit integer, little-endian."""
        self._data.extend(struct.pack("<H", value))

    def write_u32_le(self, value: int) -> None:
        """Write an unsigned 32-bit integer, little-endian."""
        self._data.extend(struct.pack("<I", value))

    def write_u64_le(self, value: int) -> None:
        """Write an unsigned 64-bit integer, little-endian."""
        self._data.extend(struct.pack("<Q", value))

    # --- Variable-length integers ---

    def write_varint(self, value: int) -> None:
        """Write an unsigned LEB128 varint."""
        self._data.extend(encode_varint(value))

    def write_signed_varint(self, value: int) -> None:
        """Write a signed zigzag + LEB128 varint."""
        self._data.extend(encode_signed_varint(value))

    # --- Bytes and strings ---

    def write_bytes(self, data: bytes | bytearray) -> None:
        """Write raw bytes."""
        self._data.extend(data)

    def write_string(self, text: str) -> None:
        """Write a length-prefixed UTF-8 string (varint length + data)."""
        encoded = text.encode("utf-8")
        self.write_varint(len(encoded))
        self._data.extend(encoded)

    def write_null_terminated_string(self, text: str) -> None:
        """Write a null-terminated UTF-8 string."""
        self._data.extend(text.encode("utf-8"))
        self._data.append(0)

    # --- Alignment ---

    def pad_to_alignment(self, alignment: int) -> None:
        """Pad with zero bytes to the next multiple of alignment."""
        remainder = self.position % alignment
        if remainder != 0:
            padding = alignment - remainder
            self._data.extend(b"\x00" * padding)

    # --- Patching ---

    def patch_u32_le(self, offset: int, value: int) -> None:
        """Overwrite 4 bytes at the given offset with a u32 value.

        Used for back-patching sizes and offsets after the data they
        describe has been written.
        """
        struct.pack_into("<I", self._data, offset, value)

    def patch_u64_le(self, offset: int, value: int) -> None:
        """Overwrite 8 bytes at the given offset with a u64 value."""
        struct.pack_into("<Q", self._data, offset, value)
