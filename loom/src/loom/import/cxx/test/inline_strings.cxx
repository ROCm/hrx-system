// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/buffer.h>
#include <loomcxx/check.h>

// The returned pointer outlives the call that materializes its source literal.
static const char* text() { return "a\0b"; }

static unsigned code_units(unsigned position) {
  loom::assume(position < 3u);
  return static_cast<unsigned char>(text()[position]) + u8"é"[position] +
         u"\U0001F9F5"[position];
}

static unsigned writable(unsigned position, unsigned replacement) {
  loom::assume(position < 8u);
  alignas(16) char first[8] = "abc";
  const char second[8] = {"abc"};
  first[position] = replacement;
  return static_cast<unsigned char>(first[position]) +
         256u * static_cast<unsigned char>(second[position]);
}

static unsigned wide_units(unsigned position) {
  loom::assume(position < 4u);
  char16_t utf16[4] = u"\U0001F9F5";
  char32_t utf32[4] = U"\U0001F9F5";
  wchar_t wide[4] = L"\U0001F9F5";
  return utf16[position] + utf32[position] + wide[position];
}

static int narrow_units(unsigned position) {
  loom::assume(position < 2u);
  signed char signed_bytes[] = "\xff";
  unsigned char unsigned_bytes[] = {"\xff"};
  char8_t utf8[4] = u8"é";
  return signed_bytes[position] + unsigned_bytes[position] + utf8[position];
}

static unsigned nested(unsigned row, unsigned position) {
  loom::assume(row < 2u && position < 4u);
  struct Record {
    // String-initialized member retaining its declared array extent.
    char label[4];
    // Non-array member adjacent to the initialized bytes.
    unsigned value;
  };
  Record record = {"abc", 7};
  char table[2][4] = {"def", "ghi"};
  char single[1][4] = {"jkl"};
  return record.label[position] + table[row][position] + single[0][position] +
         record.value;
}

static unsigned volatile_units(unsigned position) {
  loom::assume(position < 4u);
  volatile char16_t observed[4] = u"\U0001F9F5";
  volatile char table[1][4] = {"abc"};
  return observed[position] + table[0][position];
}

static unsigned literal_aliases() {
  const auto* escaped = text();
  const auto& reference = "xyz";
  auto* whole = &"xyz";
  return escaped[2] + reference[1] + (*whole)[3] + ""[0];
}

static unsigned payloads(unsigned position, unsigned replacement) {
  loom::assume(position < 16u);
  auto* original = loom::buffer::constant<16>("a\0\xffZ");
  auto* first =
      loom::buffer::alloca_init<loom::memory_space::private_, 16>("a\0\xffZ");
  auto* second =
      loom::buffer::alloca_init<loom::memory_space::private_, 16>("a\0\xffZ");
  first[position] = replacement;
  return first[position] + 256u * second[position] +
         65536u * original[position];
}

LOOM_CHECK_CASE(exact_buffer_payloads) {
  const auto first = payloads(0, 3);
  const auto zero = payloads(1, 5);
  const auto binary = payloads(2, 7);
  const auto repeated = payloads(0, 11);
  loom::check::expect_equal(first, 3u + 65792u * 97u);
  loom::check::expect_equal(zero, 5u);
  loom::check::expect_equal(binary, 7u + 65792u * 255u);
  loom::check::expect_equal(repeated, 11u + 65792u * 97u);
}

LOOM_CHECK_CASE(string_objects) {
  const auto aliases = literal_aliases();
  const auto first = code_units(0);
  const auto second = code_units(1);
  const auto last = code_units(2);
  loom::check::expect_equal(aliases, 219u);
  loom::check::expect_equal(first, 97u + 195u + 0xd83eu);
  loom::check::expect_equal(second, 169u + 0xddf5u);
  loom::check::expect_equal(last, 98u);
}

LOOM_CHECK_CASE(fresh_string_arrays) {
  const auto written_first = writable(0, 7);
  const auto written_again = writable(0, 9);
  const auto written_tail = writable(7, 11);
  const auto narrow_first = narrow_units(0);
  const auto narrow_second = narrow_units(1);
  const auto wide_first = wide_units(0);
  const auto wide_second = wide_units(1);
  const auto wide_tail = wide_units(3);
  const auto nested_first = nested(0, 0);
  const auto nested_last = nested(1, 2);
  const auto nested_tail = nested(1, 3);
  const auto volatile_first = volatile_units(0);
  const auto volatile_second = volatile_units(1);
  const auto volatile_tail = volatile_units(3);
  loom::check::expect_equal(written_first, 7u + 256u * 97u);
  loom::check::expect_equal(written_again, 9u + 256u * 97u);
  loom::check::expect_equal(written_tail, 11u);
  loom::check::expect_equal(narrow_first, 449);
  loom::check::expect_equal(narrow_second, 169);
  loom::check::expect_equal(wide_first, 0xd83eu + 2u * 0x1f9f5u);
  loom::check::expect_equal(wide_second, 0xddf5u);
  loom::check::expect_equal(wide_tail, 0u);
  loom::check::expect_equal(nested_first, 97u + 100u + 106u + 7u);
  loom::check::expect_equal(nested_last, 99u + 105u + 108u + 7u);
  loom::check::expect_equal(nested_tail, 7u);
  loom::check::expect_equal(volatile_first, 0xd83eu + 97u);
  loom::check::expect_equal(volatile_second, 0xddf5u + 98u);
  loom::check::expect_equal(volatile_tail, 0u);
}
