// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/vector.h>

using Flat = unsigned __attribute__((ext_vector_type(8)));
using Fixed = loom::type::vector<loom::type::shape<2, 4>, unsigned>;
using Rows =
    loom::type::vector<loom::type::shape<loom::type::dynamic, 4>, unsigned>;
using Line =
    loom::type::vector<loom::type::shape<loom::type::dynamic>, unsigned>;
using Volume = loom::type::vector<
    loom::type::shape<loom::type::dynamic, 2, loom::type::dynamic>, unsigned>;
using FixedVolume = loom::type::vector<loom::type::shape<2, 2, 2>, unsigned>;

// Exact ordinary helpers accept an explicit shape assertion at the callsite.
[[loom::inline]] static unsigned first_and_last(Fixed value) {
  auto flat = loom::vector::bitcast<Flat>(value);
  return flat[0] + flat[7];
}

[[loom::inline]] static Rows preserve_rows(Rows value) { return value; }

[[loom::inline]] static unsigned read_rows(Rows value) {
  return first_and_last(loom::vector::refine<Fixed>(value));
}

unsigned staging(unsigned seed) {
  Flat flat = {seed,     seed + 1, seed + 2, seed + 3,
               seed + 4, seed + 5, seed + 6, seed + 7};
  auto matrix = loom::vector::bitcast<Fixed>(flat);
  auto rows = loom::vector::refine<Rows>(matrix, {loom::type::index{2}});
  return read_rows(preserve_rows(rows));
}

unsigned native_carrier(unsigned seed) {
  Flat flat = {seed, 1, 2, 3, 4, 5, 6, seed + 7};
  auto dynamic = loom::vector::refine<Line>(flat, {loom::type::index{8}});
  auto restored = loom::vector::refine<Flat>(dynamic);
  return restored[0] + restored[7];
}

unsigned overflow_rank(unsigned seed) {
  Flat flat = {seed, 1, 2, 3, 4, 5, 6, seed + 7};
  auto volume = loom::vector::bitcast<FixedVolume>(flat);
  auto dynamic = loom::vector::refine<Volume>(
      volume, {loom::type::index{2}, loom::type::index{2}});
  auto fixed = loom::vector::refine<FixedVolume>(dynamic);
  auto restored = loom::vector::bitcast<Flat>(fixed);
  return restored[0] + restored[7];
}

unsigned loop_transport(unsigned iterations, unsigned seed) {
  auto first = loom::vector::splat<Rows>({loom::type::index{2}}, seed);
  auto second = loom::vector::splat<Rows>({loom::type::index{2}}, seed + 3);
  for (unsigned i = 0; i < iterations; ++i) {
    auto old_first = first;
    first = second;
    second = old_first;
  }
  return read_rows(first);
}

unsigned slice_rows(unsigned seed) {
  using Input = loom::type::vector<loom::type::shape<4, 4>, unsigned>;
  auto input = loom::vector::iota<Input>(seed, 1u);
  auto rows = loom::vector::slice<Rows>(input, {1, 0}, {loom::type::index{2}});
  return read_rows(rows);
}
