# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for bytecode encoding primitives."""

import pytest

from loom.format.bytecode.encoding import (
    ByteBuffer,
    decode_signed_varint,
    decode_varint,
    encode_signed_varint,
    encode_varint,
)

# ============================================================================
# LEB128 varint
# ============================================================================


class TestVarint:
    def test_zero(self) -> None:
        assert encode_varint(0) == b"\x00"

    def test_small(self) -> None:
        assert encode_varint(1) == b"\x01"
        assert encode_varint(127) == b"\x7f"

    def test_two_byte(self) -> None:
        assert encode_varint(128) == b"\x80\x01"
        assert encode_varint(16383) == b"\xff\x7f"

    def test_three_byte(self) -> None:
        assert encode_varint(16384) == b"\x80\x80\x01"

    def test_large(self) -> None:
        # 2^32 - 1
        encoded = encode_varint(0xFFFFFFFF)
        value, _ = decode_varint(encoded)
        assert value == 0xFFFFFFFF

    def test_very_large(self) -> None:
        # 2^64 - 1
        encoded = encode_varint((1 << 64) - 1)
        value, _ = decode_varint(encoded)
        assert value == (1 << 64) - 1

    def test_negative_fails(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            encode_varint(-1)

    def test_round_trip(self) -> None:
        for value in [
            0,
            1,
            127,
            128,
            255,
            256,
            16383,
            16384,
            65535,
            1 << 20,
            1 << 32,
            1 << 48,
        ]:
            encoded = encode_varint(value)
            decoded, offset = decode_varint(encoded)
            assert decoded == value
            assert offset == len(encoded)

    def test_decode_with_offset(self) -> None:
        data = b"\x42" + encode_varint(300) + b"\x99"
        value, offset = decode_varint(data, 1)
        assert value == 300
        assert offset == 1 + len(encode_varint(300))

    def test_unterminated(self) -> None:
        with pytest.raises(ValueError, match="unterminated"):
            decode_varint(b"\x80\x80")  # No terminating byte.

    def test_non_canonical_zero_fails(self) -> None:
        with pytest.raises(ValueError, match="non-canonical"):
            decode_varint(b"\x80\x00")

    def test_non_canonical_small_value_fails(self) -> None:
        with pytest.raises(ValueError, match="non-canonical"):
            decode_varint(b"\x81\x00")

    def test_too_large_fails(self) -> None:
        with pytest.raises(ValueError, match="too large"):
            decode_varint(b"\x80\x80\x80\x80\x80\x80\x80\x80\x80\x02")


class TestSignedVarint:
    def test_zero(self) -> None:
        assert decode_signed_varint(encode_signed_varint(0))[0] == 0

    def test_positive(self) -> None:
        for value in [1, 2, 63, 64, 127, 128, 1000, 1 << 30]:
            encoded = encode_signed_varint(value)
            decoded, _ = decode_signed_varint(encoded)
            assert decoded == value

    def test_negative(self) -> None:
        for value in [-1, -2, -63, -64, -128, -1000, -(1 << 30)]:
            encoded = encode_signed_varint(value)
            decoded, _ = decode_signed_varint(encoded)
            assert decoded == value

    def test_zigzag_ordering(self) -> None:
        """Zigzag maps small magnitudes to small unsigned values."""
        # 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4
        assert encode_signed_varint(0) == encode_varint(0)
        assert encode_signed_varint(-1) == encode_varint(1)
        assert encode_signed_varint(1) == encode_varint(2)
        assert encode_signed_varint(-2) == encode_varint(3)
        assert encode_signed_varint(2) == encode_varint(4)

    def test_non_canonical_fails(self) -> None:
        with pytest.raises(ValueError, match="non-canonical"):
            decode_signed_varint(b"\x80\x00")


# ============================================================================
# ByteBuffer
# ============================================================================


class TestByteBuffer:
    def test_empty(self) -> None:
        buf = ByteBuffer()
        assert buf.position == 0
        assert buf.get_bytes() == b""

    def test_write_u8(self) -> None:
        buf = ByteBuffer()
        buf.write_u8(0x42)
        assert buf.get_bytes() == b"\x42"
        assert buf.position == 1

    def test_write_u16_le(self) -> None:
        buf = ByteBuffer()
        buf.write_u16_le(0x0102)
        assert buf.get_bytes() == b"\x02\x01"

    def test_write_u32_le(self) -> None:
        buf = ByteBuffer()
        buf.write_u32_le(0x01020304)
        assert buf.get_bytes() == b"\x04\x03\x02\x01"

    def test_write_u64_le(self) -> None:
        buf = ByteBuffer()
        buf.write_u64_le(0x0102030405060708)
        assert buf.get_bytes() == b"\x08\x07\x06\x05\x04\x03\x02\x01"

    def test_write_varint(self) -> None:
        buf = ByteBuffer()
        buf.write_varint(300)
        assert buf.get_bytes() == encode_varint(300)

    def test_write_string(self) -> None:
        buf = ByteBuffer()
        buf.write_string("hello")
        data = buf.get_bytes()
        length, offset = decode_varint(data)
        assert length == 5
        assert data[offset : offset + 5] == b"hello"

    def test_write_null_terminated(self) -> None:
        buf = ByteBuffer()
        buf.write_null_terminated_string("test")
        assert buf.get_bytes() == b"test\x00"

    def test_pad_to_alignment(self) -> None:
        buf = ByteBuffer()
        buf.write_u8(0x42)
        buf.pad_to_alignment(8)
        assert buf.position == 8
        data = buf.get_bytes()
        assert data[0] == 0x42
        assert data[1:8] == b"\x00" * 7

    def test_pad_already_aligned(self) -> None:
        buf = ByteBuffer()
        buf.write_bytes(b"\x00" * 8)
        buf.pad_to_alignment(8)
        assert buf.position == 8  # No padding added.

    def test_patch_u32(self) -> None:
        buf = ByteBuffer()
        buf.write_u32_le(0)  # Placeholder.
        buf.write_u8(0xFF)
        buf.patch_u32_le(0, 42)
        data = buf.get_bytes()
        assert data[0:4] == b"\x2a\x00\x00\x00"
        assert data[4] == 0xFF

    def test_patch_u64(self) -> None:
        buf = ByteBuffer()
        buf.write_u64_le(0)
        buf.patch_u64_le(0, 0x0102030405060708)
        assert buf.get_bytes() == b"\x08\x07\x06\x05\x04\x03\x02\x01"

    def test_sequential_writes(self) -> None:
        buf = ByteBuffer()
        buf.write_u8(1)
        buf.write_u16_le(2)
        buf.write_u32_le(3)
        assert buf.position == 7

    def test_write_bytes(self) -> None:
        buf = ByteBuffer()
        buf.write_bytes(b"\x01\x02\x03")
        assert buf.get_bytes() == b"\x01\x02\x03"


# ============================================================================
# File header construction (integration)
# ============================================================================


class TestFileHeader:
    """Test building a loombc file header with ByteBuffer."""

    def test_header_layout(self) -> None:
        buf = ByteBuffer()
        # Header field: magic bytes.
        buf.write_bytes(b"LOOM")
        # Header field: format version.
        buf.write_u8(0)
        # Header field: location mode.
        buf.write_u8(0)
        # Header field: module count.
        buf.write_u16_le(1)
        # Header field: file string pool length.
        buf.write_u32_le(0)
        # Header field: reserved.
        buf.write_u32_le(0)
        # Header field: producer string.
        buf.write_null_terminated_string("loom-py")
        # Pad to 8-byte alignment.
        buf.pad_to_alignment(8)

        data = buf.get_bytes()
        assert data[0:4] == b"LOOM"
        assert data[4] == 0  # version
        assert data[5] == 0  # location_mode
        assert data[6:8] == b"\x01\x00"  # module_count = 1 (LE)
        assert len(data) % 8 == 0  # Aligned.
