// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/service_schedule.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ServiceSchedule, RetainsAppliedPrefixAndIndependentOutcomes) {
  alignas(8) std::array<uint8_t, 280> bytes;
  loom_runtime_service_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_runtime_service_plan_initialize(
      3, iree_make_byte_span(bytes.data(), bytes.size()), &plan));
  auto layout = loom_runtime_service_plan_layout(plan);
  EXPECT_EQ(layout.byte_length, 280u);
  EXPECT_EQ(layout.decisions_offset, 64u);
  EXPECT_EQ(layout.results_offset, 184u);
  EXPECT_EQ(layout.reserved, 0u);

  // More outputs than the one named input. Repeated keys are representable;
  // their independent acceptance belongs to the source, not this buffer.
  const loom_runtime_service_decision_t decision = {
      .service_ordinal = 7,
      .service_incarnation = 11,
      .offer = {2, 17},
      .choice_ordinal = 3,
  };
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_TRUE(loom_runtime_service_plan_append(plan, &decision));
  }
  EXPECT_FALSE(loom_runtime_service_plan_append(plan, &decision));
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).selection_flags,
            LOOM_RUNTIME_SERVICE_SELECTION_CAPACITY);
  const auto copy = loom_runtime_service_plan_decision(plan, 2);
  EXPECT_EQ(copy.service_ordinal, 7u);
  EXPECT_EQ(copy.service_incarnation, 11u);
  EXPECT_EQ(copy.offer.ordinal, 2u);
  EXPECT_EQ(copy.offer.revision, 17u);
  EXPECT_EQ(copy.choice_ordinal, 3u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data() + 64), 7u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data() + 96), 3u);

  loom_runtime_service_plan_complete(plan, IREE_STATUS_UNAVAILABLE, {4, 23});
  layout = loom_runtime_service_plan_layout(plan);
  EXPECT_EQ(layout.decision_count, 3u);
  EXPECT_EQ(layout.applied_count, 1u);
  const auto accepted = loom_runtime_service_plan_outcome(plan, 0);
  EXPECT_TRUE(accepted.accepted);
  EXPECT_EQ(accepted.status_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(accepted.operation.ordinal, 4u);
  EXPECT_EQ(accepted.operation.serial, 23u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data() + 184), 1u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data() + 208), 23u);

  // A stale attempt advances the same prefix without replaying acceptance.
  loom_runtime_service_plan_complete(plan, IREE_STATUS_OK, {});
  const auto stale = loom_runtime_service_plan_outcome(plan, 1);
  EXPECT_FALSE(stale.accepted);
  EXPECT_EQ(stale.status_code, IREE_STATUS_OK);
  EXPECT_EQ(stale.operation.serial, 0u);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 2u);
  EXPECT_EQ(loom_runtime_service_plan_outcome(plan, 0).operation.serial, 23u);

  // Explicitly discard the last unattempted decision before fresh selection.
  loom_runtime_service_plan_reset(plan);
  layout = loom_runtime_service_plan_layout(plan);
  EXPECT_EQ(layout.decision_count, 0u);
  EXPECT_EQ(layout.applied_count, 0u);
  EXPECT_EQ(layout.selection_flags, 0u);
  EXPECT_EQ(layout.decision_capacity, 3u);
  EXPECT_EQ(layout.byte_length, 280u);
}

TEST(ServiceSchedule, ColdStorageBoundsAndEmptyPlan) {
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(loom_runtime_service_plan_size(0, &length));
  EXPECT_EQ(length, 64u);
  alignas(8) std::array<uint8_t, 72> bytes;
  loom_runtime_service_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_runtime_service_plan_initialize(
      0, iree_make_byte_span(bytes.data(), 64), &plan));
  const loom_runtime_service_decision_t decision = {};
  EXPECT_FALSE(loom_runtime_service_plan_append(plan, &decision));
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).decision_count, 0u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_service_plan_initialize(
          0, iree_make_byte_span(bytes.data() + 1, 64), &plan));
  EXPECT_EQ(plan, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_runtime_service_plan_initialize(
                            0, iree_make_byte_span(bytes.data(), 63), &plan));
  EXPECT_EQ(plan, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_runtime_service_plan_size(IREE_HOST_SIZE_MAX, &length));
  EXPECT_EQ(length, 0u);
}

}  // namespace
