// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <initializer_list>

static auto distance(const int* first, const int* last) { return last - first; }

// Runtime positions keep signed element arithmetic observable after export.
[[loom::noinline]] long long endpoint_distance(unsigned first, unsigned last) {
  int values[8];
  return distance(values + first, values + last);
}

[[loom::noinline]] long long walked_distance(unsigned count) {
  int values[8];
  const int* cursor = values;
  for (unsigned i = 0; i < count; ++i) {
    ++cursor;
  }
  return values - cursor;
}

long long list_distance(int value) {
  std::initializer_list<int> values{value, value + 1, value + 2};
  return distance(values.end(), values.begin());
}

long long empty_distance() {
  const int* first = nullptr;
  const int* last = nullptr;
  return last - first;
}
