// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/device_status.h"

#include <sys/ioctl.h>

#include "gtest/gtest.h"

namespace {

constexpr NTSTATUS kStatusNoMemory = static_cast<NTSTATUS>(0xC0000017u);
constexpr NTSTATUS kStatusDeviceRemoved = static_cast<NTSTATUS>(0xC00002B6u);

struct FakeDeviceState {
  // Result from the execution-state query.
  NTSTATUS query_status = 0;
  // Execution state returned by a successful query.
  D3DKMT_DEVICEEXECUTION_STATE execution_state = D3DKMT_DEVICEEXECUTION_ACTIVE;
  // Result from the blocking native paging wait.
  NTSTATUS wait_status = 0;
  // Native diagnostics issued after ambiguous errors or paging completion.
  uint32_t query_count = 0;
  // Native waits issued before the independent device-state observation.
  uint32_t wait_count = 0;
  // Mapped paging progress supplied to the production helper.
  volatile uint64_t progress = 0;
  // Progress published by a successful native wait.
  uint64_t progress_after_wait = 7;
};

FakeDeviceState* current_state = nullptr;
amdf_linux_dxg_connection_t* current_connection = nullptr;

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  EXPECT_EQ(context, current_connection);
  switch (_IOC_NR(request)) {
    case 0x0E: {
      ++current_state->query_count;
      auto* query = static_cast<D3DKMT_GETDEVICESTATE*>(arguments);
      EXPECT_EQ(query->hDevice, 0x10u);
      EXPECT_EQ(query->StateType, D3DKMT_DEVICESTATE_EXECUTION);
      query->ExecutionState = current_state->execution_state;
      return static_cast<uint32_t>(current_state->query_status);
    }
    case 0x3A: {
      ++current_state->wait_count;
      const auto* wait =
          static_cast<D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU*>(arguments);
      EXPECT_EQ(wait->hDevice, 0x10u);
      EXPECT_EQ(wait->ObjectCount, 1u);
      EXPECT_EQ(wait->ObjectHandleArray[0], 0x20u);
      EXPECT_EQ(wait->FenceValueArray[0], 7u);
      if (current_state->wait_status == 0) {
        current_state->progress = current_state->progress_after_wait;
      }
      return static_cast<uint32_t>(current_state->wait_status);
    }
    default:
      ADD_FAILURE() << "unexpected DXG operation " << _IOC_NR(request);
      return static_cast<uint32_t>(kStatusNoMemory);
  }
}

class LinuxDxgDeviceStatusTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    current_connection = reinterpret_cast<amdf_linux_dxg_connection_t*>(this);
    kmt_.connection = current_connection;
    amdf_gpu_linux_dxg_device_status_initialize(&status_);
  }

  void TearDown() override {
    current_connection = nullptr;
    current_state = nullptr;
  }

  amdf_status_t Observe(NTSTATUS status) {
    return amdf_gpu_linux_dxg_device_status_observe_error(
        &status_, &kmt_, 0x10, amdf_linux_dxg_kmt_make_status(status));
  }

  FakeDeviceState state_;
  amdf_linux_dxg_kmt_t kmt_ = {};
  amdf_gpu_linux_dxg_device_status_t status_ = {};
};

TEST_F(LinuxDxgDeviceStatusTest, RejectedOperationDoesNotFailActiveDevice) {
  EXPECT_EQ(Observe(kStatusNoMemory),
            amdf_linux_dxg_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_query(&status_), AMDF_STATUS_OK);
  EXPECT_EQ(state_.query_count, 1u);
}

TEST_F(LinuxDxgDeviceStatusTest, ExplicitRemovalIsStickyWithoutDiagnostics) {
  const amdf_status_t removed =
      amdf_linux_dxg_kmt_make_status(kStatusDeviceRemoved);
  EXPECT_EQ(Observe(kStatusDeviceRemoved), removed);
  EXPECT_EQ(Observe(kStatusNoMemory), removed);
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_query(&status_), removed);
  EXPECT_EQ(state_.query_count, 0u);
}

TEST_F(LinuxDxgDeviceStatusTest, TerminalExecutionStateIsSticky) {
  state_.execution_state = D3DKMT_DEVICEEXECUTION_STOPPED;
  const amdf_status_t lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(Observe(kStatusNoMemory), lost);
  state_.execution_state = D3DKMT_DEVICEEXECUTION_ACTIVE;
  EXPECT_EQ(Observe(kStatusNoMemory), lost);
  EXPECT_EQ(state_.query_count, 1u);
}

TEST_F(LinuxDxgDeviceStatusTest, FailedDiagnosticDoesNotInventDeviceLoss) {
  state_.query_status = kStatusNoMemory;
  EXPECT_EQ(Observe(kStatusNoMemory),
            amdf_linux_dxg_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_query(&status_), AMDF_STATUS_OK);
}

TEST_F(LinuxDxgDeviceStatusTest, ResetFenceIsFailureInsteadOfProgress) {
  state_.progress = UINT64_MAX;
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_query_fence_progress(
                &status_, &state_.progress),
            0u);
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_query(&status_),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  EXPECT_EQ(state_.query_count, 0u);
}

TEST_F(LinuxDxgDeviceStatusTest, PagingWaitConfirmsActiveDevice) {
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_wait_for_paging(
                &status_, &kmt_, 0x10, 0x20, &state_.progress, 7),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.progress, 7u);
  EXPECT_EQ(state_.wait_count, 1u);
  EXPECT_EQ(state_.query_count, 1u);
}

TEST_F(LinuxDxgDeviceStatusTest, PagingWaitCannotPublishStoppedDevice) {
  state_.execution_state = D3DKMT_DEVICEEXECUTION_HUNG;
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_wait_for_paging(
                &status_, &kmt_, 0x10, 0x20, &state_.progress, 7),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  EXPECT_EQ(state_.progress, 7u);
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_query(&status_),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
}

TEST_F(LinuxDxgDeviceStatusTest, FailedPagingWaitPreservesRejection) {
  state_.wait_status = kStatusNoMemory;
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_wait_for_paging(
                &status_, &kmt_, 0x10, 0x20, &state_.progress, 7),
            amdf_linux_dxg_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_device_status_query(&status_), AMDF_STATUS_OK);
  EXPECT_EQ(state_.query_count, 1u);
}

}  // namespace
