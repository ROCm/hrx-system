// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/view.h>

namespace loomt = loom::type;
struct Sample {
  // Weight applied by the native consumer.
  unsigned key;
  // Value updated between calls through a sliced collection.
  unsigned long long reading;
};
using Samples = loomt::view<loomt::shape<loomt::dynamic>, Sample>;

[[loom::import("samples", "sum")]]
unsigned long long sum(Samples samples);

unsigned long long run(unsigned count, unsigned long long base) {
  if (count > 257) {
    return 0;
  }
  auto storage = loom::view::alloca<loomt::view<loomt::shape<259>, Sample>>();
  loom::view::store(Sample{777u, 999ull}, storage, 0);
  loom::view::store(Sample{555u, 333ull}, storage, 258);
  for (unsigned i = 0; i < count; ++i) {
    loom::view::store(Sample{3u + i * 5u, base + i * 7ull}, storage, i + 1);
  }
  auto samples = loom::view::subview<loomt::dynamic>(storage, {1}, {count});
  auto first = sum(samples);
  if (base != 0 && count != 0) {
    auto sample = loom::view::load(samples, 0);
    sample.reading += 11ull;
    loom::view::store(sample, samples, 0);
  }
  auto second = sum(samples);
  auto before = loom::view::load(storage, 0);
  auto after = loom::view::load(storage, 258);
  return first + 3ull * second + before.key + before.reading + after.key +
         after.reading;
}
