// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/ring.h"

#include <array>
#include <cstdint>
#include <vector>

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/testing/gtest.h"

namespace {

using Ring = iree_hal_amd_gpu_pm4_ring_t;
using Reservation = iree_hal_amd_gpu_pm4_ring_reservation_t;
constexpr uint32_t kUnwritten = 0xa5a5a5a5;

void ExpectNop(const uint32_t* words, uint32_t count) {
  ASSERT_GE(count, 2u);
  ASSERT_LE(count, 16385u);
  EXPECT_EQ(words[0], 0xc0001000u | ((count - 2) << 16));
  for (uint32_t i = 1; i < count; ++i) {
    EXPECT_EQ(words[i], 0u);
  }
}

TEST(RingTest, InitializationDoesNotTouchNativeStorage) {
  std::array<uint32_t, 64> words;
  words.fill(kUnwritten);
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  EXPECT_EQ(ring.words, words.data());
  EXPECT_EQ(ring.capacity, 64u);
  EXPECT_EQ(ring.reserved_words, 0u);
  for (uint32_t word : words) {
    EXPECT_EQ(word, kUnwritten);
  }
}

TEST(RingTest, AllPublicationPaddingResidues) {
  constexpr uint32_t kExtents[] = {16, 24, 24, 24, 24, 24, 24, 32};
  for (uint32_t residue = 0; residue < 8; ++residue) {
    SCOPED_TRACE(residue);
    std::array<uint32_t, 128> words;
    words.fill(kUnwritten);
    Ring ring;
    iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
    Reservation reservation;
    const uint32_t body = 16 + residue;
    ASSERT_TRUE(
        iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, body, &reservation));
    EXPECT_EQ(reservation.words, words.data());
    EXPECT_EQ(reservation.position, 0u);
    EXPECT_EQ(ring.reserved_words, kExtents[residue]);
    for (uint32_t i = 0; i < body; ++i) {
      EXPECT_EQ(words[i], kUnwritten);
    }
    if (residue) {
      ExpectNop(words.data() + body, kExtents[residue] - body);
    }
    EXPECT_EQ(words[kExtents[residue]], kUnwritten);
  }
}

TEST(RingTest, InsufficientCreditLeavesEverythingUnchanged) {
  std::array<uint32_t, 64> words;
  words.fill(kUnwritten);
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  Reservation reservation;
  ASSERT_TRUE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 24, &reservation));
  ASSERT_TRUE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 24, &reservation));
  const Reservation previous = reservation;
  const auto previous_words = words;
  // Filling every DWORD would alias the native empty-ring position.
  EXPECT_FALSE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 16, &reservation));
  EXPECT_EQ(ring.reserved_words, 48u);
  EXPECT_EQ(reservation.words, previous.words);
  EXPECT_EQ(reservation.position, previous.position);
  EXPECT_EQ(words, previous_words);
  ASSERT_TRUE(
      iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 24, 16, &reservation));
  EXPECT_EQ(reservation.position, 48u);
  EXPECT_EQ(ring.reserved_words, 64u);
}

TEST(RingTest, WrapPreservesTheUnconsumedNotificationTail) {
  std::array<uint32_t, 64> words;
  words.fill(kUnwritten);
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  Reservation first;
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 24, &first));
  iree_hal_amd_gpu_pm4_emit_nop(first.words, 24);
  Reservation second;
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 16, 24, &second));
  iree_hal_amd_gpu_pm4_emit_nop(second.words, 24);
  // The completion marker in the second publication ends at 40; its trailing
  // event at [40, 48) remains unconsumed even after that completion is
  // observed.
  const auto before_wrap = words;
  Reservation third;
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 16, 24, &third));
  EXPECT_EQ(words, before_wrap);
  ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 40, 24, &third));
  EXPECT_EQ(third.position, 64u);
  EXPECT_EQ(third.words, words.data());
  EXPECT_EQ(ring.reserved_words, 88u);
  ExpectNop(words.data() + 48, 16);
  for (uint32_t i = 40; i < 48; ++i) {
    EXPECT_EQ(words[i], before_wrap[i]);
  }
}

TEST(RingTest, CompletionCreditProgressesAcrossEveryBodyResidueAndWrap) {
  for (uint32_t capacity : {64u, 128u, 1024u, 65536u}) {
    SCOPED_TRACE(capacity);
    std::vector<uint32_t> words(capacity, kUnwritten);
    Ring ring;
    iree_hal_amd_gpu_pm4_ring_initialize(words.data(), capacity, &ring);
    uint64_t consumed = 0;
    // Completed release packets return credit immediately. The final eight
    // DWORDs model the independent event tail, followed by native padding.
    for (uint32_t iteration = 0; iteration < 1024; ++iteration) {
      const uint32_t body = 16 + iteration % (capacity == 64 ? 7 : 17);
      const uint64_t previous_end = ring.reserved_words;
      Reservation reservation;
      ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, consumed, body,
                                                        &reservation));
      EXPECT_EQ(reservation.words,
                words.data() + (reservation.position & (capacity - 1)));
      EXPECT_EQ(ring.reserved_words & 7, 0u);
      EXPECT_LT(ring.reserved_words - consumed, capacity);
      if (reservation.position != previous_end) {
        ExpectNop(words.data() + (previous_end & (capacity - 1)),
                  static_cast<uint32_t>(reservation.position - previous_end));
      }
      const uint32_t padding =
          static_cast<uint32_t>(ring.reserved_words - reservation.position) -
          body;
      if (padding) {
        ExpectNop(reservation.words + body, padding);
      }
      iree_hal_amd_gpu_pm4_emit_nop(reservation.words, body);
      consumed = reservation.position + body - 8;
    }
  }
}

TEST(RingTest, LargestAdmittedSpanUsesValidWrapPackets) {
  std::vector<uint32_t> words(65536, kUnwritten);
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  uint64_t consumed = 0;
  for (uint32_t iteration = 0; iteration < 12; ++iteration) {
    const uint32_t body = iteration % 2 ? 16382 : 8199;
    const uint64_t previous_end = ring.reserved_words;
    Reservation reservation;
    ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, consumed, body,
                                                      &reservation));
    if (reservation.position != previous_end) {
      ExpectNop(words.data() + (previous_end & 65535),
                static_cast<uint32_t>(reservation.position - previous_end));
    }
    iree_hal_amd_gpu_pm4_emit_nop(reservation.words, body);
    consumed = reservation.position + body - 8;
  }
  EXPECT_GT(ring.reserved_words, words.size());
}

TEST(RingTest, CompletedMaximumTailsCannotStrandTheNextPublication) {
  std::array<uint32_t, 1024> words;
  Ring ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  uint64_t consumed = 0;
  for (uint32_t iteration = 0; iteration < 1024; ++iteration) {
    // A 495-DWORD body needs nine DWORDs of padding. Eight notification
    // DWORDs and that padding remain live after the release marker completes.
    // Alternating a shorter body also varies the amount of wrap padding.
    const uint32_t body = iteration % 2 ? 495 : 247;
    Reservation reservation;
    ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, consumed, body,
                                                      &reservation));
    EXPECT_LT(ring.reserved_words - consumed, words.size());
    consumed = reservation.position + body - 8;
    EXPECT_EQ(ring.reserved_words - consumed, 17u);
  }
}

}  // namespace
