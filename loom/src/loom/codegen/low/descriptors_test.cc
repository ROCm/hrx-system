// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include <algorithm>
#include <array>

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(LowDescriptorsTest, LookupRetainsIdentityAcrossSharedViews) {
  const char strings[] =
      "test.alpha"
      "test.beta"
      "test.gamma";
  std::array<loom_low_descriptor_t, 3> descriptors = {};
  descriptors[0].key_string_ref = LOOM_STRING_REF(0, 0, 10);
  descriptors[1].key_string_ref = LOOM_STRING_REF(0, 10, 9);
  descriptors[2].key_string_ref = LOOM_STRING_REF(0, 19, 10);
  std::array<uint16_t, 3> references = {0, 1, 2};
  loom_low_descriptor_set_t set = {
      .descriptors = descriptors.data(),
      .descriptor_count = 3,
      .descriptor_ordinal_count = 3,
      .descriptor_refs = references.data(),
      .descriptor_ref_count = 3,
      .string_pool = {{strings}, {sizeof(strings) - 1}},
  };
  for (auto& descriptor : descriptors) {
    descriptor.stable_id = loom_low_descriptor_stable_id_from_key(
        loom_low_descriptor_set_string(&set, descriptor.key_string_ref));
  }
  std::sort(references.begin(), references.end(),
            [&](uint32_t lhs, uint32_t rhs) {
              return descriptors[lhs].stable_id < descriptors[rhs].stable_id;
            });

  // A full view, prefix view and sparse view share the same immutable lookup
  // index. Hidden ordinals must not be returned by either lookup entry point.
  const uint64_t sparse_membership = 5;
  const uint32_t visible_counts[] = {3, 1, 2};
  for (uint32_t view = 0; view < 3; ++view) {
    set.descriptor_ordinal_count = view == 1 ? 1 : 3;
    set.descriptor_count = visible_counts[view];
    set.descriptor_membership_words = view == 2 ? &sparse_membership : nullptr;
    for (uint32_t ordinal = 0; ordinal < descriptors.size(); ++ordinal) {
      const auto& descriptor = descriptors[ordinal];
      const auto key =
          loom_low_descriptor_set_string(&set, descriptor.key_string_ref);
      const bool visible =
          view == 0 || ordinal == 0 || (view == 2 && ordinal == 2);
      const uint32_t expected =
          visible ? ordinal : LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
      EXPECT_EQ(loom_low_descriptor_set_lookup_descriptor_by_id(
                    &set, descriptor.stable_id),
                expected);
      EXPECT_EQ(loom_low_descriptor_set_lookup_descriptor(&set, key), expected);
    }
    for (auto key : {IREE_SV(""), IREE_SV("test"), IREE_SV("test.alph"),
                     IREE_SV("test.alpha.extra"), IREE_SV("test.delta")}) {
      EXPECT_EQ(loom_low_descriptor_set_lookup_descriptor(&set, key),
                LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
      EXPECT_EQ(loom_low_descriptor_set_lookup_descriptor_by_id(
                    &set, loom_low_descriptor_stable_id_from_key(key)),
                LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
    }
    EXPECT_EQ(loom_low_descriptor_set_lookup_descriptor_by_id(
                  &set, LOOM_LOW_STABLE_ID_NONE),
              LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  }
}

TEST(LowDescriptorsTest, EmptyDescriptorLookup) {
  const loom_low_descriptor_set_t set = {};
  EXPECT_EQ(
      loom_low_descriptor_set_lookup_descriptor(&set, IREE_SV("test.add")),
      LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  EXPECT_EQ(loom_low_descriptor_set_lookup_descriptor_by_id(&set, 1),
            LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
}

TEST(LowDescriptorsTest, EventSeparationLookupUsesProducerRange) {
  constexpr uint16_t kEventCount = 4;
  const std::array<loom_low_event_separation_t, 5> separations = {{
      {.minimum_issue_separation_cycles = -2, .consumer_event_id = 1},
      {.minimum_issue_separation_cycles = 4, .consumer_event_id = 3},
      {.minimum_issue_separation_cycles = 7, .consumer_event_id = 0},
      {.minimum_issue_separation_cycles = 8, .consumer_event_id = 1},
      {.minimum_issue_separation_cycles = 9, .consumer_event_id = 3},
  }};
  const std::array<loom_low_timing_event_t, kEventCount> events = {{
      {.separation_start = 0, .separation_count = 2},
      {},
      {.separation_start = 2, .separation_count = 3},
      {},
  }};
  const loom_low_schedule_model_t model = {
      .timing_events = events.data(),
      .event_separations = separations.data(),
      .timing_event_count = kEventCount,
      .event_separation_count = IREE_ARRAYSIZE(separations),
  };

  const std::array<std::array<int, kEventCount>, kEventCount> expected_indices =
      {{
          {{-1, 0, -1, 1}},
          {{-1, -1, -1, -1}},
          {{2, 3, -1, 4}},
          {{-1, -1, -1, -1}},
      }};

  for (uint16_t producer = 0; producer < kEventCount; ++producer) {
    for (uint16_t consumer = 0; consumer < kEventCount; ++consumer) {
      const auto* actual = loom_low_schedule_model_lookup_event_separation(
          &model, producer, consumer);
      const int expected_index = expected_indices[producer][consumer];
      if (expected_index < 0) {
        EXPECT_EQ(actual, nullptr);
      } else {
        EXPECT_EQ(actual, &separations[expected_index]);
      }
    }
  }
}

TEST(LowDescriptorsTest, SummarizesOnlyMemoryAttachedEffects) {
  loom_low_effect_t effects[6] = {};
  effects[0].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[1].kind = LOOM_LOW_EFFECT_KIND_WRITE;
  effects[1].width_bits = 64;
  effects[2].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[2].memory_space = LOOM_LOW_MEMORY_SPACE_GENERIC;
  effects[3].kind = LOOM_LOW_EFFECT_KIND_WRITE;
  effects[3].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  effects[3].width_bits = 7;
  effects[4].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[4].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  effects[4].width_bits = 32;
  effects[5].kind = LOOM_LOW_EFFECT_KIND_WRITE;
  effects[5].memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
  effects[5].width_bits = 64;
  loom_low_descriptor_t descriptor = {};
  const loom_low_descriptor_layout_t layout = {
      .effect_count = IREE_ARRAYSIZE(effects),
  };
  loom_low_descriptor_set_t descriptor_set = {
      .descriptor_layouts = &layout,
      .descriptor_layout_count = 1,
      .effects = effects,
      .effect_count = IREE_ARRAYSIZE(effects),
  };

  const loom_low_descriptor_memory_effect_summary_t summary =
      loom_low_descriptor_memory_effect_summary(&descriptor_set, &descriptor);

  EXPECT_EQ(summary.read_unknown_width_count, 1u);
  EXPECT_EQ(summary.write_unknown_width_count, 1u);
  EXPECT_EQ(summary.read_byte_count, 4u);
  EXPECT_EQ(summary.write_byte_count, 8u);
}

}  // namespace
}  // namespace loom
