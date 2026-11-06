// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/file_completion.h"

#include <semaphore>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class FileCompletionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    iree_hal_amd_gpu_file_completion_initialize(
        proactor_,
        [](void* user_data, iree_status_t status, bool retired) {
          auto* self = static_cast<FileCompletionTest*>(user_data);
          IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, status);
          self->outcomes_.push_back(retired);
          if (!retired && self->entered_) {
            self->entered_->release();
            self->resume_->acquire();
          }
          if (!retired && self->retire_in_callback_) {
            iree_hal_amd_gpu_file_completion_notify(&self->completion_,
                                                    self->failure_, true);
          }
          if (retired && self->free_in_callback_) {
            delete self->free_in_callback_;
            self->free_in_callback_ = nullptr;
          }
        },
        this, &completion_);
  }
  void TearDown() override { iree_async_proactor_release(proactor_); }
  void Poll() {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }

  // Actual poll owner and NOP admission, without a mock event queue.
  iree_async_proactor_t* proactor_ = nullptr;
  // Embedded result storage reused by the ordinary file slot.
  iree_hal_amd_gpu_file_completion_t completion_ = {};
  // Native device fault, separate from the host NOP's successful delivery.
  amdf_status_t failure_ = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  // Poll-thread outcomes; true is the one final lifetime handoff.
  std::vector<bool> outcomes_;
  // Models final native stop racing an active logical-failure callback.
  bool retire_in_callback_ = false;
  // Optional owner reclaimed directly from its final callback.
  iree_hal_amd_gpu_file_completion_t* free_in_callback_ = nullptr;
  // Optional handshake proving the first proactor callback is active.
  std::binary_semaphore* entered_ = nullptr;
  // Releases that callback only after another thread publishes retirement.
  std::binary_semaphore* resume_ = nullptr;
};

TEST_F(FileCompletionTest, CoalescesFailureAndRetirementBeforePoll) {
  iree_hal_amd_gpu_file_completion_notify(&completion_, failure_, false);
  iree_hal_amd_gpu_file_completion_notify(&completion_, failure_, true);
  EXPECT_TRUE(outcomes_.empty());
  Poll();
  EXPECT_EQ(outcomes_, (std::vector<bool>{true}));
}

TEST_F(FileCompletionTest, RetainsNativeOwnershipAfterEarlyError) {
  iree_hal_amd_gpu_file_completion_notify(&completion_, failure_, false);
  Poll();
  EXPECT_EQ(outcomes_, (std::vector<bool>{false}));
  iree_hal_amd_gpu_file_completion_notify(&completion_, failure_, true);
  Poll();
  EXPECT_EQ(outcomes_, (std::vector<bool>{false, true}));
}

TEST_F(FileCompletionTest, RetirementDuringCallbackCannotResubmitItsLiveNop) {
  retire_in_callback_ = true;
  iree_hal_amd_gpu_file_completion_notify(&completion_, failure_, false);
  Poll();
  EXPECT_EQ(outcomes_, (std::vector<bool>{false, true}));
}

TEST_F(FileCompletionTest, FinalCallbackCanFreeItsOwnStorage) {
  free_in_callback_ = new iree_hal_amd_gpu_file_completion_t;
  iree_hal_amd_gpu_file_completion_initialize(proactor_, completion_.notify,
                                              this, free_in_callback_);
  iree_hal_amd_gpu_file_completion_notify(free_in_callback_, failure_, true);
  Poll();
  EXPECT_EQ(free_in_callback_, nullptr);
  EXPECT_EQ(outcomes_, (std::vector<bool>{true}));
}

TEST_F(FileCompletionTest, ConcurrentRetirementJoinsActiveLogicalCallback) {
  std::binary_semaphore entered(0), resume(0);
  entered_ = &entered;
  resume_ = &resume;
  iree_hal_amd_gpu_file_completion_notify(&completion_, failure_, false);
  std::thread native([&] {
    entered.acquire();
    iree_hal_amd_gpu_file_completion_notify(&completion_, failure_, true);
    resume.release();
  });
  Poll();
  native.join();
  EXPECT_EQ(outcomes_, (std::vector<bool>{false, true}));
}

}  // namespace
