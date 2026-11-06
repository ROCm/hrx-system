// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// Enum representations and constant expressions are checked under every
// supported source data model.

namespace inferred_storage {
enum Empty {};
enum Small { zero, one };
enum Signed { negative = -1, positive = 2 };
enum Unsigned { high = 0x80000000u };
enum Wide { wide = 1ULL << 40 };
enum Mixed { minimum = -1, maximum = 0xffffffffu };
enum Full { full = 0xffffffffffffffffULL };
static_assert(__is_same(__underlying_type(Empty), int));
static_assert(__is_same(__underlying_type(Small), unsigned int));
static_assert(__is_same(decltype(+one), int));
static_assert(sizeof(Signed) == 4 && negative < 0);
static_assert(sizeof(Unsigned) == 4 && high > 0);
static_assert(sizeof(Wide) == 8 && wide == (1ULL << 40));
static_assert(sizeof(Mixed) == 8 && minimum < 0);
static_assert(maximum > 0 && maximum == 0xffffffffULL);
static_assert(sizeof(Full) == 8 && full > 0);
static_assert(full == 0xffffffffffffffffULL);
#if __SIZEOF_LONG__ == 8
static_assert(__is_same(__underlying_type(Wide), unsigned long));
static_assert(__is_same(decltype(+wide), long));
static_assert(__is_same(__underlying_type(Full), unsigned long));
static_assert(__is_same(decltype(+full), unsigned long));
#else
static_assert(__is_same(__underlying_type(Wide), unsigned long long));
static_assert(__is_same(decltype(+wide), long long));
static_assert(__is_same(__underlying_type(Full), unsigned long long));
static_assert(__is_same(decltype(+full), unsigned long long));
#endif
}  // namespace inferred_storage

namespace preliminary_values {
enum UnsignedStep { last = 0xffffffffu, next, check = next == 0x100000000ULL };
enum SignedStep {
  signed_last = 9223372036854775807LL,
  signed_next,
  signed_check = signed_next == 0x8000000000000000ULL
};
enum Negative { first = -2, second, third, fourth };
enum NegativeWide { lowest = -9223372036854775807LL - 1, following_lowest };
enum Temporary {
  initial = 0u,
  unsigned_type = __is_same(decltype(initial), unsigned),
  following = initial - 1u
};
enum WideTemporary {
  large = 1ULL << 40,
  following_large = large + 1,
  large_type = __is_same(decltype(large), unsigned long long)
};
static_assert(check == 1 && signed_check == 1);
static_assert(second == -1 && third == 0 && fourth == 1);
static_assert(following_lowest == -9223372036854775807LL);
static_assert(unsigned_type == 1 && following == 0xffffffffu);
static_assert(large_type == 1);
static_assert(following_large == (1ULL << 40) + 1);
}  // namespace preliminary_values

namespace fixed_storage {
enum class Byte : unsigned char { first = 254, last };
enum class SignedByte : signed char { low = -128, high = 127 };
enum class Short : unsigned short { high = 65535 };
enum class Word : unsigned { high = 0xffffffffu };
enum class SignedWord : int { low = -2147483647 - 1 };
enum class Long : unsigned long long { high = 0xffffffffffffffffULL };
enum class SignedLong : long long { low = -9223372036854775807LL - 1 };
enum class Flag : bool { no, yes };
enum PlainByte : unsigned char { byte = 255 };
enum Copy : unsigned { copied = byte };
static_assert(sizeof(Byte) == 1 && int(Byte::last) == 255);
static_assert(sizeof(SignedByte) == 1 && int(SignedByte::low) == -128);
static_assert(sizeof(Short) == 2 && unsigned(Short::high) == 65535);
static_assert(sizeof(Word) == 4 && unsigned(Word::high) == 0xffffffffu);
static_assert(int(SignedWord::low) == -2147483647 - 1);
static_assert(sizeof(Long) == 8 &&
              (unsigned long long)Long::high == 0xffffffffffffffffULL);
static_assert((long long)SignedLong::low == -9223372036854775807LL - 1);
static_assert(!bool(Flag::no) && bool(Flag::yes));
static_assert(__is_same(decltype(+byte), int));
static_assert(copied == 255);
}  // namespace fixed_storage

namespace dependent_values {
template <unsigned long long Value>
constexpr auto value() {
  enum Kind { first = Value, next, last = next + 1 };
  return last;
}
template <class T, T Value>
constexpr T fixed() {
  enum class Kind : T { first = Value, next };
  return T(Kind::next);
}
static_assert(value<1>() == 3);
static_assert(value<(1ULL << 40)>() == (1ULL << 40) + 2);
static_assert(value<0xffffffffu>() == 0x100000001ULL);
static_assert(fixed<unsigned char, 254>() == 255);
static_assert(fixed<long long, -2>() == -1);
}  // namespace dependent_values
