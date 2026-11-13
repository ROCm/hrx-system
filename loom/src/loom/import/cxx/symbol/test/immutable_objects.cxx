// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/view.h>

namespace detail {
struct Pair {
  // First value in each immutable pair.
  unsigned first;
  // Second value, selected by the lookup.
  unsigned second;
};

static constexpr unsigned table[4] = {11, 22, 33, 44};
static constexpr unsigned nested[2][3] = {{1, 2, 3}, {4, 5, 6}};
static constexpr Pair pairs[2] = {{10, 20}, {30, 40}};
static constexpr unsigned scalar = 7;
}  // namespace detail

static unsigned lookup(unsigned index) {
  return detail::table[index & 3u] + detail::nested[index & 1u][index % 3u] +
         detail::pairs[index & 1u].second;
}

static unsigned addressed_scalar() {
  const unsigned* pointer = &detail::scalar;
  return *pointer;
}

static unsigned local_lookup(unsigned index) {
  alignas(64) static constexpr unsigned table[4] = {11, 22, 33, 44};
  static const unsigned nested[2][3] = {{1, 2, 3}, {4, 5, 6}};
  static constexpr detail::Pair pairs[2] = {{10, 20}, {30, 40}};
  return table[index & 3u] + nested[index & 1u][index % 3u] +
         pairs[index & 1u].second;
}

template <unsigned Bias>
static unsigned local_template_lookup(unsigned index) {
  static constexpr unsigned table[2] = {Bias, Bias + 1};
  return table[index & 1u];
}

static void local_observations(unsigned index,
                               loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<4>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(local_lookup(index), values, 0);
  loom::view::store(local_lookup(index + 1u), values, 1);
  loom::view::store(local_template_lookup<3>(index), values, 2);
  loom::view::store(local_template_lookup<7>(index), values, 3);
}

static unsigned local_reference(unsigned index) {
  return 11u * (1u + (index & 3u)) + 1u + 3u * (index & 1u) + index % 3u +
         20u * (1u + (index & 1u));
}

static void local_oracle(unsigned index, loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<4>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(local_reference(index), values, 0);
  loom::view::store(local_reference(index + 1u), values, 1);
  loom::view::store(3u + (index & 1u), values, 2);
  loom::view::store(7u + (index & 1u), values, 3);
}

static unsigned local_index(loom::check::ordinal trial) {
  unsigned value = trial;
  return value < 6u ? value : ~0u - (value - 6u);
}
