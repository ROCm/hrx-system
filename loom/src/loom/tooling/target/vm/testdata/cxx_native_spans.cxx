// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/view.h>

namespace loomt = loom::type;
using Keys = loomt::view<loomt::shape<loomt::dynamic>, unsigned>;
using Readings = loomt::view<loomt::shape<loomt::dynamic>, unsigned long long>;

struct Samples {
  Keys keys;
  Readings readings;
};

[[loom::import("samples", "sum")]]
unsigned long long sum(Samples samples);

unsigned long long run(unsigned count, unsigned long long base) {
  if (count > 257) {
    return 0;
  }
  auto keys = loom::view::alloca<loomt::view<loomt::shape<259>, unsigned>>();
  auto readings =
      loom::view::alloca<loomt::view<loomt::shape<259>, unsigned long long>>();
  loom::view::store(777u, keys, 0);
  loom::view::store(999ull, readings, 0);
  loom::view::store(555u, keys, 258);
  loom::view::store(333ull, readings, 258);
  for (unsigned i = 0; i < count; ++i) {
    loom::view::store(3u + i * 5u, keys, i + 1);
    loom::view::store(base + i * 7ull, readings, i + 1);
  }
  Samples samples{loom::view::subview<loomt::dynamic>(keys, {1}, {count}),
                  loom::view::subview<loomt::dynamic>(readings, {1}, {count})};
  auto first = sum(samples);
  if (base != 0 && count != 0) {
    auto value = loom::view::load(samples.readings, 0);
    loom::view::store(value + 11ull, samples.readings, 0);
  }
  auto second = sum(samples);
  return first + 3ull * second + loom::view::load(keys, 0) +
         loom::view::load(readings, 0) + loom::view::load(keys, 258) +
         loom::view::load(readings, 258);
}
