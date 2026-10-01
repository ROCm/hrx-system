// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>

#include "iree/testing/gtest.h"

// These definitions are linked directly from the object emitted by Loom in a
// separate build action. The compiler process exits before the link begins.
extern "C" uint64_t mix(uint64_t input, uint64_t delta);
extern "C" uint64_t choose_mix(uint64_t input, uint64_t delta, uint64_t limit);
extern "C" uint64_t preserve(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                             uint64_t e, uint64_t f);
extern "C" uint64_t shift_mix(uint64_t input, uint64_t count);
extern "C" uint64_t load_word(uint64_t unused, const uint64_t* input,
                              uint64_t index);
extern "C" uint64_t add_word(uint32_t word, uint64_t bias);
extern "C" uint32_t divide_mix(uint32_t word);
extern "C" uint32_t replace_narrow(uint8_t* bytes, uint16_t* words,
                                   uint64_t index, uint32_t replacement);
extern "C" uint64_t recurrence(uint64_t first, uint64_t second,
                               uint64_t iterations);

namespace {

TEST(NativeCallableTest, OrdinaryCLinkage) {
  uint64_t state = UINT64_C(0x243f6a8885a308d3);
  for (unsigned i = 0; i < 1000; ++i) {
    uint64_t words[6];
    for (uint64_t& word : words) {
      state = state * UINT64_C(6364136223846793005) + 1;
      word = state;
    }
    const uint64_t a = words[0], b = words[1], c = words[2];
    const uint64_t d = words[3], e = words[4], f = words[5];
    ASSERT_EQ(mix(a, b), (a + b) ^ a);
    ASSERT_EQ(choose_mix(a, b, c), a < c ? ((a + b) ^ a) : ((a * b) ^ c));
    ASSERT_EQ(preserve(a, b, c, d, e, f),
              ((a + b) * (c + d) + (e + f) * a) ^ (b ^ c));
    ASSERT_EQ(shift_mix(a, b), (a >> (b & 63)) ^ a);
    ASSERT_EQ(load_word(a, words, i % 6), words[i % 6]);
    ASSERT_EQ(add_word(static_cast<uint32_t>(a), b),
              static_cast<uint64_t>(static_cast<uint32_t>(a)) + b);
    const uint32_t word = static_cast<uint32_t>(a);
    ASSERT_EQ(divide_mix(word), ((word / 7) + (word % 7)) ^ word);
    uint64_t first = a, second = b;
    for (unsigned step = 0; step < i % 23; ++step) {
      const uint64_t sum = first + second;
      first = second;
      second = sum;
    }
    ASSERT_EQ(recurrence(a, b, i % 23), first);
  }
}

TEST(NativeCallableTest, NarrowMemoryPreservesNeighbors) {
  const std::array<uint8_t, 6> initial_bytes = {0, 127, 128, 255, 17, 201};
  const std::array<uint16_t, 6> initial_words = {0,     32767, 32768,
                                                 65535, 513,   54321};
  for (size_t index = 0; index < initial_bytes.size(); ++index) {
    auto bytes = initial_bytes;
    auto words = initial_words;
    const uint32_t previous = bytes[index] | (uint32_t{words[index]} << 8);
    const uint32_t replacement = 0xabcdef42u + static_cast<uint32_t>(index);
    auto expected_bytes = initial_bytes;
    auto expected_words = initial_words;
    expected_bytes[index] = static_cast<uint8_t>(replacement);
    expected_words[index] = static_cast<uint16_t>(replacement);
    ASSERT_EQ(replace_narrow(bytes.data(), words.data(), index, replacement),
              previous);
    EXPECT_EQ(bytes, expected_bytes);
    EXPECT_EQ(words, expected_words);
  }
}

}  // namespace
