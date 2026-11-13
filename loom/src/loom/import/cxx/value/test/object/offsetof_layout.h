// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct [[gnu::packed]] OffsetBlock {
  unsigned char tag;
  unsigned payload[3];
};

struct [[gnu::packed]] OffsetPacket {
  unsigned char tag;
  OffsetBlock blocks[3];
  unsigned matrix[2][3];
};

static_assert(__builtin_offsetof(OffsetPacket, blocks[2].payload[1]) == 32);
static_assert(__builtin_offsetof(OffsetPacket, matrix[1][2]) == 60);
static_assert(__builtin_offsetof(const OffsetPacket, blocks[0].tag) == 1);

template <typename T, unsigned Index>
constexpr auto selected_offset() {
  return __builtin_offsetof(T, blocks[Index].payload[1]);
}

template <unsigned Index>
constexpr auto fixed_type_offset() {
  return __builtin_offsetof(OffsetPacket, blocks[Index].payload[1]);
}

static_assert(selected_offset<OffsetPacket, 0>() == 6);
static_assert(selected_offset<OffsetPacket, 2>() == 32);
static_assert(fixed_type_offset<1>() == 19);
static_assert(fixed_type_offset<2>() == 32);

union OffsetUnion {
  unsigned scalar;
  OffsetBlock blocks[3];
};

struct OffsetEnvelope {
  unsigned prefix;
  OffsetUnion value;
};

static_assert(__builtin_offsetof(OffsetEnvelope, value.blocks[2].payload[1]) ==
              35);
