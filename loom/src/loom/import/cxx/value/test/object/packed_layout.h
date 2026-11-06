// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

struct __attribute__((packed)) PackedRecord {
  // Byte establishing a misaligned word origin.
  unsigned char tag;
  // Naturally four-byte word stored at byte one.
  unsigned value;
};
static_assert(sizeof(PackedRecord) == 5 && alignof(PackedRecord) == 1);
static_assert(__builtin_offsetof(PackedRecord, value) == 1);

struct PackedField {
  // Byte preceding the packed field.
  unsigned char tag;
  // Only this member has reduced alignment.
  unsigned value __attribute__((__packed__));
  // The following ordinary word still requires its natural alignment.
  unsigned tail;
};
static_assert(sizeof(PackedField) == 12 && alignof(PackedField) == 4);
static_assert(__builtin_offsetof(PackedField, value) == 1);
static_assert(__builtin_offsetof(PackedField, tail) == 8);

struct [[using __gnu__: __packed__]] PackedAlignedField {
  // Byte preceding an explicitly aligned member.
  unsigned char tag;
  // An explicit minimum overrides GNU packing.
  alignas(16) unsigned value;
};
static_assert(sizeof(PackedAlignedField) == 32 &&
              alignof(PackedAlignedField) == 16);
static_assert(__builtin_offsetof(PackedAlignedField, value) == 16);

struct NaturalInner {
  // The inner record keeps its own natural padding.
  unsigned char tag;
  // Word aligned within the inner record.
  unsigned value;
};
struct PackedOuter {
  // Byte preceding a complete naturally padded record.
  unsigned char tag;
  // Packing changes placement, not the nested record's representation.
  NaturalInner inner;
  // Arrays retain element size and stride.
  unsigned short tail[2];
} __attribute__((packed));
static_assert(sizeof(PackedOuter) == 13 && alignof(PackedOuter) == 1);
static_assert(__builtin_offsetof(PackedOuter, inner) == 1);
static_assert(__builtin_offsetof(PackedOuter, tail) == 9);

union [[gnu::packed]] PackedUnion {
  // Alternative byte-sized payload.
  unsigned char tag;
  // The union is large enough for its widest member but byte aligned.
  unsigned value;
};
static_assert(sizeof(PackedUnion) == 4 && alignof(PackedUnion) == 1);

template <class Word>
struct Block {
  // Two-byte scale preceding the packed payload.
  unsigned short scale;
  // The element width is determined by specialization.
  Word words[4];
} __attribute__((packed));
static_assert(sizeof(Block<unsigned>) == 18 && alignof(Block<unsigned>) == 1);
static_assert(__builtin_offsetof(Block<unsigned>, words) == 2);
static_assert(sizeof(Block<unsigned short>) == 10);
static_assert(sizeof(Block<unsigned>) == 18);

template <class T>
struct MemberTemplate {
  // Byte preceding the dependent member.
  unsigned char tag;
  // The request survives substitution of a concrete member type.
  [[gnu::packed]] T value;
};
static_assert(sizeof(MemberTemplate<unsigned>) == 5);
static_assert(__builtin_offsetof(MemberTemplate<unsigned>, value) == 1);

struct [[gnu::packed]] PackedBits {
  // Ends one bit short of a natural word boundary.
  unsigned first : 31;
  // Crosses the natural word boundary.
  unsigned second : 2;
};
static_assert(sizeof(PackedBits) == 5 && alignof(PackedBits) == 1);

#pragma pack(push, 1)
struct [[gnu::packed]] PragmaCap {
  // Byte preceding a member whose explicit request is capped by the pragma.
  unsigned char tag;
  // Pragma caps differ from GNU packing and cap explicit member alignment.
  alignas(16) unsigned value;
};
#pragma pack(pop)
static_assert(sizeof(PragmaCap) == 5 && alignof(PragmaCap) == 1);
static_assert(__builtin_offsetof(PragmaCap, value) == 1);

#define PACKED __attribute__((__packed__))
struct PACKED MacroRecord {
  // Byte establishing the following word's displacement.
  unsigned char tag;
  // Attribute spelling is retained through preprocessing.
  unsigned value;
};
#undef PACKED
static_assert(sizeof(MacroRecord) == 5);

struct CommaMembers {
  // Leading byte before both words.
  unsigned char tag;
  // The declaration attribute applies to both declarators.
  [[gnu::packed]] unsigned first, second;
};
static_assert(sizeof(CommaMembers) == 9 && alignof(CommaMembers) == 1);
static_assert(__builtin_offsetof(CommaMembers, second) == 5);

struct IndividualMembers {
  // Leading byte before a selectively packed word.
  unsigned char tag;
  // Declarator attributes only apply to the declarator that carries them.
  unsigned first __attribute__((packed)), second;
};
static_assert(__builtin_offsetof(IndividualMembers, first) == 1);
static_assert(__builtin_offsetof(IndividualMembers, second) == 8);

struct __attribute__((packed)) Forward;
struct Forward {
  // Leading byte before a word governed by the forward declaration.
  unsigned char tag;
  // The complete definition retains its previously declared packing request.
  unsigned value;
};
static_assert(sizeof(Forward) == 5 && alignof(Forward) == 1);

struct PackedNamedBits {
  unsigned char tag;
  unsigned bits [[gnu::packed]] : 3;
  unsigned char tail;
};
static_assert(sizeof(PackedNamedBits) == 3 && alignof(PackedNamedBits) == 1);
static_assert(__builtin_offsetof(PackedNamedBits, tail) == 2);

struct PackedTrailingBits {
  unsigned char tag;
  unsigned bits : 3 __attribute__((__packed__));
  unsigned char tail;
};
static_assert(sizeof(PackedTrailingBits) == 3 &&
              alignof(PackedTrailingBits) == 1);
static_assert(__builtin_offsetof(PackedTrailingBits, tail) == 2);

struct PackedAlignedBits {
  unsigned char tag;
  unsigned bits : 3 __attribute__((packed, aligned(2)));
  unsigned char tail;
};
static_assert(sizeof(PackedAlignedBits) == 4 &&
              alignof(PackedAlignedBits) == 2);
static_assert(__builtin_offsetof(PackedAlignedBits, tail) == 3);

struct PackedZeroWidth {
  unsigned char tag;
  unsigned : 0 __attribute__((packed, aligned(8)));
  unsigned char tail;
};
static_assert(sizeof(PackedZeroWidth) == 9 && alignof(PackedZeroWidth) == 1);
static_assert(__builtin_offsetof(PackedZeroWidth, tail) == 8);

template <unsigned Alignment>
struct PackedDependentBits {
  unsigned char tag;
  unsigned bits [[gnu::packed]] : 3 __attribute__((aligned(Alignment)));
  unsigned char tail;
};
static_assert(sizeof(PackedDependentBits<2>) == 4);
static_assert(sizeof(PackedDependentBits<8>) == 16);
static_assert(__builtin_offsetof(PackedDependentBits<8>, tail) == 9);
static_assert(sizeof(PackedDependentBits<2>) == 4);
