// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/service_control.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ServiceControlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_host_size_t use_capacities[] = {2, 1};
    const loom_runtime_service_offer_capacity_t offers[] = {
        {2, use_capacities}, {1, use_capacities}};
    const loom_runtime_service_control_capacity_t capacity = {2, offers, 2,
                                                              use_capacities};
    iree_host_size_t length = 0;
    IREE_ASSERT_OK(loom_runtime_service_control_size(&capacity, &length));
    storage_.resize(length / sizeof(uint64_t));
    IREE_ASSERT_OK(loom_runtime_service_control_initialize(
        &capacity, 19, iree_make_byte_span(storage_.data(), length),
        &control_));
  }

  void PublishOffer(uint64_t duration) {
    const loom_runtime_service_use_t uses[] = {
        {
            .resource_ordinal = 7,
            .estimates = LOOM_RUNTIME_SERVICE_ESTIMATE_DURATION,
            .duration_ns = duration,
        },
        {.resource_ordinal = 11}};
    const loom_runtime_service_choice_publication_t choices[] = {
        {{
             .instance_ordinal = 2,
             .request_count = 8,
             .devices = 4,
             .use_count = 2,
         },
         uses},
        {{
             .instance_ordinal = 3,
             .request_count = 8,
             .devices = 8,
             .use_count = 1,
         },
         uses + 1}};
    const loom_runtime_service_offer_t offer = {
        .key = {0, 23},
        .work = LOOM_RUNTIME_SERVICE_WORK_EXECUTION,
        .choice_count = 2,
    };
    loom_runtime_service_publish_offer(control_, &offer, choices);
  }

  // One retained region; copied values never borrow an individual row.
  std::vector<uint64_t> storage_;
  // Typed handle into storage_, invalidated only after all readers join.
  loom_runtime_service_control_t* control_ = nullptr;
};

TEST_F(ServiceControlTest, LayoutAndInitiallyEmptyStableSlots) {
  const auto layout = loom_runtime_service_control_layout(control_);
  EXPECT_EQ(layout.byte_length, storage_.size() * 8);
  EXPECT_EQ(layout.incarnation, 19u);
  EXPECT_EQ(layout.observation_offset, 64u);
  EXPECT_EQ(layout.offers_offset, 128u);
  EXPECT_EQ(layout.operations_offset, 256u);
  EXPECT_EQ(layout.offer_capacity, 2u);
  EXPECT_EQ(layout.operation_capacity, 2u);
  EXPECT_EQ(layout.reserved, 0u);
  // The external representation is little endian, independent of host ABI.
  EXPECT_EQ(iree_unaligned_load_le_u64(storage_.data() + 1), 19u);
  loom_runtime_service_observation_t observation = {};
  EXPECT_EQ(loom_runtime_service_read_observation(control_, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.offer_count, 0u);
  for (uint64_t i = 0; i < 2; ++i) {
    loom_runtime_service_offer_t offer = {};
    loom_runtime_service_operation_t operation = {};
    EXPECT_EQ(loom_runtime_service_read_offer(control_, i, &offer),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(loom_runtime_service_read_operation(control_, i, &operation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(offer.key.ordinal, i);
    EXPECT_EQ(offer.choice_count, 0u);
    EXPECT_EQ(operation.key.ordinal, i);
    EXPECT_EQ(operation.key.serial, 0u);
  }
}

TEST_F(ServiceControlTest, EstimatePublicationDoesNotChangeLegalRevision) {
  PublishOffer(100);
  loom_runtime_service_choice_t choice = {};
  ASSERT_EQ(loom_runtime_service_read_choice(control_, {0, 23}, 0, &choice),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(choice.devices, 4u);
  EXPECT_EQ(choice.estimates, 0u);  // Device time is not whole-unit time.
  loom_runtime_service_use_t use = {};
  ASSERT_EQ(loom_runtime_service_read_choice_use(control_, {0, 23}, 0,
                                                 choice.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(use.duration_ns, 100u);
  PublishOffer(0);
  use.duration_ns = 93;
  EXPECT_EQ(loom_runtime_service_read_choice_use(control_, {0, 23}, 0,
                                                 choice.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_CHANGED);
  EXPECT_EQ(use.duration_ns, 93u);  // Rejected copies leave outputs intact.
  ASSERT_EQ(loom_runtime_service_read_choice(control_, {0, 23}, 0, &choice),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  ASSERT_EQ(loom_runtime_service_read_choice_use(control_, {0, 23}, 0,
                                                 choice.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(use.estimates, LOOM_RUNTIME_SERVICE_ESTIMATE_DURATION);
  EXPECT_EQ(use.duration_ns, 0u);  // Explicitly known zero is valid.
}

TEST_F(ServiceControlTest, AcceptedUsesSurviveOfferWithdrawalAndDetectReuse) {
  PublishOffer(100);
  const loom_runtime_service_use_t uses[] = {{.resource_ordinal = 11}};
  const loom_runtime_service_operation_t accepted = {
      .key = {1, 7},
      .origin = {0, 23},
      .choice_ordinal = 1,
      .work = LOOM_RUNTIME_SERVICE_WORK_EXECUTION,
      .waits = LOOM_RUNTIME_SERVICE_WAIT_CAPACITY,
      .devices = 8,
      .use_count = 1,
  };
  loom_runtime_service_publish_operation(control_, &accepted, uses);
  const loom_runtime_service_offer_t withdrawn = {.key = {0, 24}};
  loom_runtime_service_publish_offer(control_, &withdrawn, nullptr);
  loom_runtime_service_operation_t operation = {};
  ASSERT_EQ(loom_runtime_service_read_operation(control_, 1, &operation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(operation.origin.revision, 23u);
  EXPECT_EQ(operation.work, LOOM_RUNTIME_SERVICE_WORK_EXECUTION);
  loom_runtime_service_use_t use = {};
  ASSERT_EQ(loom_runtime_service_read_operation_use(
                control_, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(use.resource_ordinal, 11u);
  auto next = accepted;
  next.key.serial = 8;
  loom_runtime_service_publish_operation(control_, &next, uses);
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control_, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_CHANGED);
  EXPECT_EQ(use.resource_ordinal, 11u);
}

TEST_F(ServiceControlTest, ConcurrentCopiesJoinExactEstimatePublication) {
  PublishOffer(1);
  std::atomic<bool> started{false};
  std::thread writer([&] {
    while (!started.load()) {
      std::this_thread::yield();
    }
    for (uint64_t i = 2; i < 4096; ++i) {
      PublishOffer(i);
    }
  });
  started.store(true);
  for (uint64_t i = 0; i < 4096; ++i) {
    loom_runtime_service_choice_t choice = {};
    if (loom_runtime_service_read_choice(control_, {0, 23}, 0, &choice) !=
        LOOM_RUNTIME_SERVICE_READ_COMPLETE) {
      continue;
    }
    loom_runtime_service_use_t use = {.duration_ns = UINT64_MAX};
    const auto read = loom_runtime_service_read_choice_use(
        control_, {0, 23}, 0, choice.publication, 0, &use);
    if (read == LOOM_RUNTIME_SERVICE_READ_COMPLETE) {
      EXPECT_EQ(use.resource_ordinal, 7u);
      // Each publication starts at 2 and corresponds exactly to one duration.
      EXPECT_EQ(use.duration_ns, choice.publication / 2);
    } else {
      EXPECT_EQ(use.duration_ns, UINT64_MAX);
    }
  }
  writer.join();
}

TEST(ServiceControl, RejectsIncompleteOrUnrepresentableStorage) {
  loom_runtime_service_control_capacity_t capacity = {};
  alignas(8) uint8_t bytes[129] = {};
  loom_runtime_service_control_t* control = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_service_control_initialize(
          &capacity, 0, iree_make_byte_span(bytes, 128), &control));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_service_control_initialize(
          &capacity, 1, iree_make_byte_span(bytes + 1, 128), &control));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_service_control_initialize(
          &capacity, 1, iree_make_byte_span(bytes, 127), &control));
  capacity.offer_capacity = 1;
  iree_host_size_t length = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_runtime_service_control_size(&capacity, &length));
  EXPECT_EQ(length, 0u);
  const loom_runtime_service_offer_capacity_t offer = {0, nullptr};
  capacity.offers = &offer;
  capacity.offer_capacity = SIZE_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_runtime_service_control_size(&capacity, &length));
  EXPECT_EQ(length, 0u);
  EXPECT_EQ(control, nullptr);
}

}  // namespace
