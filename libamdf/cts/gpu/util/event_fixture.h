// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_UTIL_EVENT_FIXTURE_H_
#define AMDF_CTS_GPU_UTIL_EVENT_FIXTURE_H_

#include "amdf/gpu.h"
#include "gtest/gtest.h"
#include "libamdf/cts/gpu/util/user_queue.h"
#include "libamdf/cts/util/wait.h"

// Shares native wake ownership across firmware and shader producers. Queue
// fixtures retain their own publication and execution-retirement protocols.
template <typename QueueFixture>
class GpuEventTest : public QueueFixture {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(QueueFixture::SetUp());
    if (this->IsSkipped()) {
      return;
    }
    if ((this->features_ & AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS) == 0) {
      GTEST_SKIP() << "device has no native event producer";
    }
    const amdf_gpu_event_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
        .structure_size = sizeof(create_info),
    };
    ASSERT_EQ(
        this->gpu_api_->event_create(this->device_, &create_info, &event_),
        AMDF_STATUS_OK);
    ASSERT_EQ(this->gpu_api_->event_query_info(event_, &event_info_),
              AMDF_STATUS_OK);
    if (event_info_.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
        event_info_.format_version != 1 ||
        (event_info_.publication_modes & this->family_.publication_modes) ==
            0) {
      GTEST_SKIP() << "producer has no recipe for this native event format";
    }
  }

  void TearDown() override {
    // Queue destruction joins all interrupt tails before releasing their ID.
    ASSERT_NO_FATAL_FAILURE(QueueFixture::TearDown());
    if (event_) {
      ASSERT_EQ(this->gpu_api_->event_destroy(event_), AMDF_STATUS_OK);
    }
  }

  // Only the GPU can signal this event. The host acquires the authoritative
  // generation after each hint; spurious wakes cannot establish completion.
  ::testing::AssertionResult WaitForRecord(uint64_t generation_address,
                                           uint64_t generation) {
    const amdf::cts::WaitDeadline deadline;
    uint64_t remaining_ns = deadline.remaining_ns();
    while (remaining_ns != 0) {
      const amdf_status_t status =
          this->gpu_api_->event_wait(event_, remaining_ns);
      if (status != AMDF_STATUS_OK) {
        return ::testing::AssertionFailure()
               << "native event wait failed: domain="
               << amdf_status_domain(status)
               << " code=" << amdf_status_code(status);
      }
      if (GpuLoadAcquire<uint64_t>(generation_address) == generation) {
        return ::testing::AssertionSuccess();
      }
      remaining_ns = deadline.remaining_ns();
    }
    return ::testing::AssertionFailure()
           << "native event did not publish value " << generation;
  }

  // Native event retained through successful queue-first teardown.
  amdf_gpu_event_t* event_ = nullptr;
  // Immutable producer operands; mailbox and interrupt values stay separate.
  amdf_gpu_event_info_t event_info_ = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(event_info_),
  };
};

#endif  // AMDF_CTS_GPU_UTIL_EVENT_FIXTURE_H_
