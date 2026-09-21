// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

template <class Wide>
struct [[gnu::packed]] Packet {
  // Eight bytes transported without changing their bit representation.
  Wide wide;
  // Floating-point field with an independently typed memory access.
  float real;
  // Narrow field whose access must leave adjacent bytes untouched.
  unsigned short narrow;
  // Final byte makes consecutive records visit every alignment residue.
  unsigned char byte;
};
using IntegerPacket = Packet<unsigned long long>;
static_assert(sizeof(IntegerPacket) == 15 && alignof(IntegerPacket) == 1);
static_assert(__builtin_offsetof(IntegerPacket, real) == 8);
static_assert(__builtin_offsetof(IntegerPacket, narrow) == 12);
static_assert(__builtin_offsetof(IntegerPacket, byte) == 14);

// The source and destination visit all residues modulo eight in different
// orders. The surrounding bytes expose stores wider than the source fields.
void packed_copy(const unsigned char* input, unsigned char* output) {
  for (unsigned i = 0; i < 256; ++i) {
    output[i] = 37;
  }
  auto* source = reinterpret_cast<const IntegerPacket*>(input + 1);
  auto* destination = reinterpret_cast<IntegerPacket*>(output + 5);
  for (unsigned i = 0; i < 8; ++i) {
    destination[i].wide = source[i].wide;
    destination[i].real = source[i].real;
    destination[i].narrow = source[i].narrow;
    destination[i].byte = source[i].byte;
  }
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void packed_copy_kernel(const unsigned char* input, unsigned char* output) {
  packed_copy(input, output);
}

struct MemberPacked {
  // The following packed member begins immediately after this byte.
  unsigned char tag;
  // Packing belongs to this member, rather than its enclosing record.
  unsigned value [[gnu::packed]];
};
static_assert(sizeof(MemberPacked) == 5 && alignof(MemberPacked) == 1);

#pragma pack(push, 2)
struct Capped {
  // The pack cap still permits one byte of padding before the float.
  unsigned char tag;
  // Two-byte aligned field in the nested record.
  float real;
  // Nested member retains its own packed layout.
  MemberPacked member;
};
#pragma pack(pop)
static_assert(sizeof(Capped) == 12 && alignof(Capped) == 2);
static_assert(__builtin_offsetof(Capped, real) == 2);
static_assert(__builtin_offsetof(Capped, member) == 6);

struct __attribute__((packed)) Envelope {
  // Moves the nested record to an odd byte origin.
  unsigned char prefix;
  // Packing the parent retains the child's internal padding and tail byte.
  Capped data;
};
static_assert(sizeof(Envelope) == 13 && alignof(Envelope) == 1);

// Volatile qualifiers propagate through both nested records. Unsigned updates
// retain their arithmetic conversions even at an unaligned byte origin.
void packed_update(volatile Envelope* output) {
  output[2].data.real = 1.25f;
  output[2].data.real *= 2.0f;
  output[2].data.member.value = 0x80000000u;
  output[2].data.member.value /= 2u;
  output[2].data.member.value |= 1u;
  output[2].data.member.tag = 255;
  output[2].data.member.tag++;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void packed_update_kernel(volatile Envelope* output) {
  packed_update(output);
}

typedef unsigned U4 __attribute__((vector_size(16)));
struct [[gnu::packed]] VectorPacket {
  // The vector begins at an odd byte offset.
  unsigned char tag;
  // The source access spans all four lanes without touching the tag.
  U4 value;
};
static_assert(sizeof(VectorPacket) == 17 && alignof(VectorPacket) == 1);

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void packed_vector_kernel(const VectorPacket* input, VectorPacket* output) {
  output->value = input->value;
}
