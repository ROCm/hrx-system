// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

struct Padded {
  // Byte preceding natural or explicit padding.
  unsigned char first;
  // Naturally aligned word after the first byte.
  unsigned count;
  // Trailing halfword followed by record padding.
  unsigned short last;
};
static_assert(sizeof(Padded) == 12 && alignof(Padded) == 4);
static_assert(__builtin_offsetof(Padded, count) == 4);
static_assert(__builtin_offsetof(Padded, last) == 8);

struct alignas(32) Aligned {
  // Byte preceding natural or explicit padding.
  unsigned char first;
  // Word with an explicit field alignment.
  alignas(16) unsigned count;
  // Trailing halfword followed by record padding.
  unsigned short last;
};
static_assert(sizeof(Aligned) == 32 && alignof(Aligned) == 32);
static_assert(__builtin_offsetof(Aligned, count) == 16);
static_assert(__builtin_offsetof(Aligned, last) == 20);

struct Nested {
  // Leading byte establishing the following field displacement.
  unsigned char prefix;
  // Nested record retaining its own padding.
  Padded value;
  // Halfword after the complete nested footprint.
  unsigned short suffix;
};
static_assert(sizeof(Nested) == 20 && alignof(Nested) == 4);
static_assert(__builtin_offsetof(Nested, value) == 4);
static_assert(__builtin_offsetof(Nested, suffix) == 16);

template <class T>
struct Typed {
  // Leading byte establishing the following field displacement.
  unsigned char prefix;
  // Specialization determines this field layout.
  T value;
};
static_assert(sizeof(Typed<unsigned>) == 8);
static_assert(__builtin_offsetof(Typed<unsigned>, value) == 4);

#pragma pack(push, 1)
struct Packed {
  // Leading byte establishing the following field displacement.
  unsigned char prefix;
  // Word governed by the enclosing packing state.
  unsigned value;
};
#pragma pack(pop)
static_assert(sizeof(Packed) == 5 && alignof(Packed) == 1);
static_assert(__builtin_offsetof(Packed, value) == 1);

struct Natural {
  // Leading byte establishing the following field displacement.
  unsigned char prefix;
  // Word governed by the enclosing packing state.
  unsigned value;
};
static_assert(sizeof(Natural) == 8 && alignof(Natural) == 4);
static_assert(__builtin_offsetof(Natural, value) == 4);
