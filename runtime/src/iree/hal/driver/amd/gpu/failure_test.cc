// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/failure.h"

#include <string>
#include <thread>
#include <vector>

#include "iree/hal/driver/amd/gpu/publication.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class FailureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&reported_);
    sink_ = {
        .fn =
            [](void* data, const iree_hal_device_event_t* event) {
              auto* self = static_cast<FailureTest*>(data);
              EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
              const auto* payload = reinterpret_cast<
                  const iree_hal_device_driver_failure_event_t*>(
                  event->payload.data);
              EXPECT_EQ(payload->status_code, IREE_STATUS_DEADLINE_EXCEEDED);
              EXPECT_EQ(payload->backend_result_code, self->native_failure_);
              self->messages_.emplace_back(payload->message.data,
                                           payload->message.size);
              iree_atomic_store(&self->report_count_, self->messages_.size(),
                                iree_memory_order_release);
              iree_notification_post(&self->reported_, IREE_ALL_WAITERS);
            },
        .user_data = this,
    };
    iree_hal_amd_gpu_failure_initialize(&endpoint_, nullptr, sink_, 5,
                                        &failure_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    iree_hal_amd_gpu_failure_deinitialize(&failure_);
    iree_notification_deinitialize(&reported_);
  }
  void AwaitReport() {
    iree_notification_await(
        &reported_,
        [](void* data) {
          return iree_atomic_load(
                     &static_cast<FailureTest*>(data)->report_count_,
                     iree_memory_order_acquire) != 0;
        },
        this, iree_infinite_timeout());
  }

  // Immutable identity whose capture requires no native device query.
  amdf_endpoint_info_t endpoint_ = {.id = {{0x1234, 0x5678}}, .name = "GPU"};
  // Native error preserved even when cleanup also exceeds its deadline.
  amdf_status_t native_failure_ =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  // Existing public diagnostic destination.
  iree_hal_device_event_sink_t sink_ = {};
  // Actual terminal record and its host ownership notification.
  iree_hal_amd_gpu_failure_t failure_ = {};
  // Existing worker, used only by the independent-monitor integration case.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Receipt for the sink's final copied message.
  iree_notification_t reported_;
  // Release-published event count; no sleep is used to infer readiness.
  iree_atomic_int32_t report_count_ = IREE_ATOMIC_VAR_INIT(0);
  // Copies of borrowed event data, inspected after each reporting thread joins.
  std::vector<std::string> messages_;
};

TEST_F(FailureTest, PreservesFirstErrorAndReportsCachedProgressOnce) {
  ASSERT_TRUE(iree_hal_amd_gpu_failure_begin(&failure_, native_failure_,
                                             "gpu.event_wait", 1000));
  EXPECT_FALSE(iree_hal_amd_gpu_failure_begin(
      &failure_, amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL), "later",
      2000));
  iree_hal_amd_gpu_failure_set_phase(
      &failure_, IREE_HAL_AMD_GPU_FAILURE_INACTIVATING, 3, 2, 4);
  EXPECT_EQ(iree_hal_amd_gpu_failure_poll(&failure_, 5000999), 5001000);
  EXPECT_TRUE(messages_.empty());
  EXPECT_EQ(iree_hal_amd_gpu_failure_poll(&failure_, 5001000),
            IREE_TIME_INFINITE_FUTURE);
  EXPECT_EQ(iree_hal_amd_gpu_failure_poll(&failure_, 6001000),
            IREE_TIME_INFINITE_FUTURE);
  ASSERT_EQ(messages_.size(), 1u);
  EXPECT_NE(messages_[0].find("phase=inactivating queue=3 progress=2/4"),
            std::string::npos);
  EXPECT_NE(messages_[0].find("original=gpu.event_wait"), std::string::npos);
  EXPECT_NE(messages_[0].find("retained"), std::string::npos);
}

TEST_F(FailureTest, JoinedAttemptDisarmsDeadlineWithoutReporting) {
  ASSERT_TRUE(
      iree_hal_amd_gpu_failure_begin(&failure_, native_failure_, "wait", 0));
  iree_hal_amd_gpu_failure_finish(&failure_);
  EXPECT_EQ(iree_hal_amd_gpu_failure_poll(&failure_, 6000000),
            IREE_TIME_INFINITE_FUTURE);
  EXPECT_TRUE(messages_.empty());
}

TEST_F(FailureTest, ConcurrentDeadlineObserversPublishOneDiagnostic) {
  ASSERT_TRUE(
      iree_hal_amd_gpu_failure_begin(&failure_, native_failure_, "wait", 0));
  std::thread other([&] { iree_hal_amd_gpu_failure_poll(&failure_, 6000000); });
  iree_hal_amd_gpu_failure_poll(&failure_, 6000000);
  other.join();
  EXPECT_EQ(messages_.size(), 1u);
}

TEST_F(FailureTest, ExistingPublisherReportsWithoutRetirementWorkerProgress) {
  failure_.timeout_ms = 0;
  ASSERT_TRUE(iree_hal_amd_gpu_failure_begin(
      &failure_, native_failure_, "native removal", iree_time_now()));
  iree_hal_amd_gpu_failure_set_phase(
      &failure_, IREE_HAL_AMD_GPU_FAILURE_INACTIVATING, 1, 0, 4);
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      0, nullptr, {}, iree_allocator_system(), &publication_));
  iree_hal_amd_gpu_publication_watch_failure(&publication_, &failure_);
  AwaitReport();
  iree_hal_amd_gpu_publication_deinitialize(&publication_);
  ASSERT_EQ(messages_.size(), 1u);
  EXPECT_NE(messages_[0].find("original=native removal"), std::string::npos);
}

}  // namespace
