// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/view.h>

namespace {

template <class Word>
unsigned rotation(unsigned round) = delete;
template <>
unsigned rotation<unsigned>(unsigned round) {
  return round + 3u;
}
template <>
unsigned rotation<unsigned long long>(unsigned round) {
  return round + 7u;
}

template <class Word>
unsigned rotation(unsigned round, unsigned lane) = delete;
template <>
unsigned rotation<unsigned>(unsigned round, unsigned lane) {
  return round + lane + 11u;
}
template <>
unsigned rotation<unsigned long long>(unsigned round, unsigned lane) {
  return round + lane + 19u;
}

template <class Type>
unsigned select(Type value) = delete;
template <class Type>
unsigned select(const Type* value) = delete;
template <>
unsigned select<unsigned>(const unsigned* value) {
  return *value + 17u;
}
template <>
unsigned select(unsigned value) {
  return value + 5u;
}

namespace algorithm {
template <class Type>
unsigned combine(Type value) = delete;
template <class Type>
unsigned combine(Type value, unsigned bias) = delete;
template <>
unsigned combine<unsigned>(unsigned value);
template <>
unsigned combine<unsigned>(unsigned value, unsigned bias);
}  // namespace algorithm

template <>
unsigned algorithm::combine<unsigned>(unsigned value) {
  return value + 29u;
}
template <>
unsigned algorithm::combine<unsigned>(unsigned value, unsigned bias) {
  return value + bias + 31u;
}

static void specialized_values(unsigned value,
                               loom::type::buffer<unsigned> output) {
  const unsigned selected = value;
  auto values =
      loom::buffer::view<10>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(rotation<unsigned>(value), values, 0);
  loom::view::store(rotation<unsigned long long>(value), values, 1);
  loom::view::store(rotation<unsigned>(value, 2u), values, 2);
  loom::view::store(rotation<unsigned long long>(value, 2u), values, 3);
  loom::view::store(select(value), values, 4);
  loom::view::store(select<unsigned>(value), values, 5);
  loom::view::store(select(&selected), values, 6);
  loom::view::store(select<unsigned>(&selected), values, 7);
  loom::view::store(algorithm::combine(value), values, 8);
  loom::view::store(algorithm::combine(value, 2u), values, 9);
}

static void specialization_oracle(unsigned value,
                                  loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<10>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(value + 3u, values, 0);
  loom::view::store(value + 7u, values, 1);
  loom::view::store(value + 13u, values, 2);
  loom::view::store(value + 21u, values, 3);
  loom::view::store(value + 5u, values, 4);
  loom::view::store(value + 5u, values, 5);
  loom::view::store(value + 17u, values, 6);
  loom::view::store(value + 17u, values, 7);
  loom::view::store(value + 29u, values, 8);
  loom::view::store(value + 33u, values, 9);
}

static unsigned specialization_input(loom::check::ordinal trial) {
  unsigned value = trial;
  return value < 4u ? value : ~0u - (value - 4u);
}

}  // namespace

LOOM_CHECK_SCENARIO(function_specializations) {
  loom::check::trial<8>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto value = loom::check::generate<specialization_input>(trial);
    const auto storage = loom::check::fill<18, unsigned>(0x5a5a5a5au);
    const auto output = loom::check::slice<10>(storage, 4);
    loom::check::compare<specialized_values, specialization_oracle>(
        value, output, [&] { loom::check::expect_bitwise(storage, storage); });
  });
}
