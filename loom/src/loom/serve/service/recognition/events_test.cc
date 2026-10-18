// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/recognition/events.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// One observation followed by two revisions of the same segment. The later
// document value must not overwrite an earlier event still awaiting delivery.
std::vector<uint8_t> Packet() {
  std::vector<uint8_t> bytes(146);
  iree_unaligned_store_le_u32(bytes.data(), 3);
  iree_unaligned_store_le_u32(bytes.data() + 4,
                              LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
  iree_unaligned_store_le_u32(bytes.data() + 8, 20);
  iree_unaligned_store_le_u64(bytes.data() + 12, 160);
  iree_unaligned_store_le_u64(bytes.data() + 20, 480);
  iree_unaligned_store_le_f32(bytes.data() + 28, 0.75f);
  for (size_t i = 0; i < 2; ++i) {
    uint8_t* record = bytes.data() + 32 + 57 * i;
    iree_unaligned_store_le_u32(record,
                                LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT);
    iree_unaligned_store_le_u32(record + 4, 49);
    iree_unaligned_store_le_u64(record + 8, 7);
    iree_unaligned_store_le_u64(record + 16, i + 1);
    iree_unaligned_store_le_u64(record + 32, 480);
    iree_unaligned_store_le_u32(
        record + 40, i ? LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL : 0);
    iree_unaligned_store_le_u32(record + 44, 1);
    record[56] = i ? 'B' : 'a';
  }
  return bytes;
}

class RecognitionEventsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_recognition_events_t* events = nullptr;
    IREE_ASSERT_OK(loom_serve_recognition_events_allocate(
        3, 146, iree_allocator_system(), &events));
    events_.reset(events);
  }

  iree_status_t Parse(const std::vector<uint8_t>& bytes,
                      loom_serve_recognition_output_flags_t outputs =
                          LOOM_SERVE_RECOGNITION_OUTPUT_ALL,
                      uint64_t input_begin = 160, uint64_t input_end = 480) {
    return loom_serve_recognition_events_parse(
        events_.get(), iree_make_const_byte_span(bytes.data(), bytes.size()),
        input_begin, input_end, outputs);
  }

  // Native caller owns one reusable packet independently of source storage.
  std::unique_ptr<loom_serve_recognition_events_t,
                  decltype(&loom_serve_recognition_events_free)>
      events_{nullptr, loom_serve_recognition_events_free};
};

TEST_F(RecognitionEventsTest, OwnsOrderedActivityAndIndependentRevisions) {
  auto bytes = Packet();
  IREE_ASSERT_OK(Parse(bytes));
  std::fill(bytes.begin(), bytes.end(), 0);
  ASSERT_EQ(loom_serve_recognition_events_count(events_.get()), 3u);
  const auto* activity = loom_serve_recognition_events_at(events_.get(), 0);
  ASSERT_EQ(activity->kind, LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
  EXPECT_EQ(activity->activity.begin, 160u);
  EXPECT_EQ(activity->activity.end, 480u);
  EXPECT_EQ(activity->activity.speech_probability, 0.75f);
  for (size_t i = 0; i < 2; ++i) {
    const auto* event = loom_serve_recognition_events_at(events_.get(), i + 1);
    ASSERT_EQ(event->kind, LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT);
    EXPECT_EQ(event->transcript.segment, 7u);
    EXPECT_EQ(event->transcript.revision, i + 1);
    EXPECT_EQ(event->transcript.end, 480u);
    EXPECT_EQ(
        std::string(event->transcript.text.data, event->transcript.text.size),
        i ? "B" : "a");
    EXPECT_EQ(event->transcript.flags,
              i ? LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL : 0u);
  }
  IREE_ASSERT_OK(Parse({}, LOOM_SERVE_RECOGNITION_OUTPUT_ALL, 480, 480));
  EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);
}

TEST_F(RecognitionEventsTest, ActivityPartitionsExactlyTheAcknowledgedPrefix) {
  auto bytes = Packet();
  bytes.resize(60);
  iree_unaligned_store_le_u32(bytes.data(), 2);
  std::copy_n(bytes.data() + 4, 28, bytes.data() + 32);
  iree_unaligned_store_le_u64(bytes.data() + 20, 320);
  iree_unaligned_store_le_u64(bytes.data() + 40, 320);
  IREE_ASSERT_OK(Parse(bytes));
  ASSERT_EQ(loom_serve_recognition_events_count(events_.get()), 2u);
  EXPECT_EQ(loom_serve_recognition_events_at(events_.get(), 0)->activity.end,
            320u);
  EXPECT_EQ(loom_serve_recognition_events_at(events_.get(), 1)->activity.begin,
            320u);

  const std::pair<size_t, uint64_t> discontinuities[] = {
      {12, 159}, {12, 161}, {40, 319}, {40, 321}, {48, 479}};
  for (auto [offset, position] : discontinuities) {
    SCOPED_TRACE(offset);
    SCOPED_TRACE(position);
    auto invalid = bytes;
    iree_unaligned_store_le_u64(invalid.data() + offset, position);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(invalid));
    EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);
  }
  std::swap_ranges(bytes.begin() + 4, bytes.begin() + 32, bytes.begin() + 32);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(bytes));
  EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);
}

TEST_F(RecognitionEventsTest, OnlyRequestedActivityRequiresCoverage) {
  auto bytes = Packet();
  bytes.erase(bytes.begin() + 4, bytes.begin() + 32);
  iree_unaligned_store_le_u32(bytes.data(), 2);
  IREE_ASSERT_OK(Parse(bytes, LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT));
  EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 2u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(bytes));
  EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);

  IREE_ASSERT_OK(Parse({}, LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse({}));
  EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);
}

TEST_F(RecognitionEventsTest, RejectsMalformedPacketsWithoutPublishingPrefix) {
  const std::vector<std::function<void(std::vector<uint8_t>&)>> corruptions = {
      [](auto& bytes) { bytes.resize(3); },
      [](auto& bytes) { bytes.resize(9); },
      [](auto& bytes) { bytes.pop_back(); },
      [](auto& bytes) { bytes.push_back(0); },
      [](auto& bytes) { iree_unaligned_store_le_u32(bytes.data(), 0); },
      [](auto& bytes) { iree_unaligned_store_le_u32(bytes.data(), 4); },
      [](auto& bytes) { iree_unaligned_store_le_u32(bytes.data() + 4, 32); },
      [](auto& bytes) {
        iree_unaligned_store_le_u32(bytes.data() + 8, UINT32_MAX);
      },
      [](auto& bytes) { iree_unaligned_store_le_u32(bytes.data() + 8, 19); },
      // Count excludes the last record; its bytes cannot be silently ignored.
      [](auto& bytes) { iree_unaligned_store_le_u32(bytes.data(), 2); },
      // An invalid last transcript cannot expose the valid preceding events.
      [](auto& bytes) { bytes.back() = 0xff; },
  };
  for (size_t i = 0; i < corruptions.size(); ++i) {
    SCOPED_TRACE(i);
    IREE_ASSERT_OK(Parse(Packet()));
    auto bytes = Packet();
    corruptions[i](bytes);
    IREE_EXPECT_NOT_OK(Parse(bytes));
    EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(Packet(), LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT));
  EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(Packet(), LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);
}

TEST_F(RecognitionEventsTest, ActivityRequiresActualInputAndFiniteProbability) {
  auto bytes = Packet();
  bytes.resize(32);
  iree_unaligned_store_le_u32(bytes.data(), 1);
  for (float probability :
       {0.0f, 1.0f, -0.1f, 1.1f, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    iree_unaligned_store_le_f32(bytes.data() + 28, probability);
    if (probability == 0.0f || probability == 1.0f) {
      IREE_ASSERT_OK(Parse(bytes, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Parse(bytes));
      EXPECT_EQ(loom_serve_recognition_events_count(events_.get()), 0u);
    }
  }
  iree_unaligned_store_le_f32(bytes.data() + 28, 0.5f);
  for (uint64_t end : {159u, 160u, 481u}) {
    iree_unaligned_store_le_u64(bytes.data() + 20, end);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Parse(bytes));
  }
}

TEST(RecognitionEventsAllocationTest, BoundsCannotOverflowOwnedStorage) {
  for (const auto& bounds : {std::pair<size_t, size_t>{0, 4},
                             {1, 3},
                             {SIZE_MAX, 4},
                             {1, SIZE_MAX}}) {
    loom_serve_recognition_events_t* events = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        loom_serve_recognition_events_allocate(
            bounds.first, bounds.second, iree_allocator_system(), &events));
    EXPECT_EQ(events, nullptr);
  }
}

}  // namespace
