// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>

#include "loom/binding/c/example/cxx/native_abi.h"

static_assert(sizeof(Pair) == 8);
static_assert(sizeof(Mixed) == 8);
static_assert(sizeof(Large) == 24);
static_assert(sizeof(Words4) == 16);

extern "C" Pair native_pair(Pair value) {
  return {value.first ^ 0x55aa55aau, value.second + 41u};
}

extern "C" Mixed native_mixed(Mixed value) {
  return {value.value + 0.5f, value.tag ^ 0xa55aa55au};
}

extern "C" Large native_large(Large value) {
  return {value.second * 3u, value.third + 5u,
          value.first ^ 0x123456789abcdef0ull};
}

extern "C" Words4 native_words(Words4 value) {
  return value * Words4{2u, 3u, 4u, 5u};
}

extern "C" const unsigned* native_advance(const unsigned* base) {
  return base + 4;
}

static bool require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "failed: %s\n", message);
  }
  return condition;
}

int main() {
  bool ok = true;

  Pair pair = loom_pair({3u, 11u});
  ok &= require(pair.first == 18u && pair.second == 9u,
                "native caller to Loom Pair definition");

  Mixed mixed = loom_mixed({1.5f, 23u});
  ok &= require(mixed.value == 3.25f && mixed.tag == 32u,
                "native caller to Loom Mixed definition");

  Large large = loom_large({2u, 3u, 5u});
  ok &= require(large.first == 16u && large.second == 15u && large.third == 20u,
                "native caller to Loom Large definition");

  Words4 words = loom_words(Words4{1u, 2u, 3u, 4u});
  ok &= require(
      words[0] == 2u && words[1] == 4u && words[2] == 6u && words[3] == 8u,
      "native caller to Loom SIMD definition");

  unsigned values[8] = {};
  ok &= require(loom_advance(values) == values + 3,
                "native caller to Loom pointer definition");

  pair = loom_call_native_pair({0x12345678u, 17u});
  ok &= require(pair.first == 0x479e03d2u && pair.second == 58u,
                "Loom caller to native Pair definition");

  mixed = loom_call_native_mixed({2.25f, 0x0f0f0f0fu});
  ok &= require(mixed.value == 2.75f && mixed.tag == 0xaa55aa55u,
                "Loom caller to native Mixed definition");

  large = loom_call_native_large({7u, 11u, 13u});
  ok &= require(large.first == 33u && large.second == 18u &&
                    large.third == 0x123456789abcdef7ull,
                "Loom caller to native Large definition");

  words = loom_call_native_words(Words4{1u, 2u, 3u, 4u});
  ok &= require(words[0] == 22u && words[1] == 66u && words[2] == 132u &&
                    words[3] == 220u,
                "Loom caller to native SIMD definition");

  ok &= require(loom_call_native_advance(values) == values + 5,
                "Loom caller to native pointer definition");

  if (ok) {
    std::puts("Loom and native C++ exchanged every value correctly.");
  }
  return ok ? 0 : 1;
}
