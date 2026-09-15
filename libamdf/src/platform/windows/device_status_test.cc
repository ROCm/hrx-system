// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/device_status.h"

#include <ntstatus.h>

#include <array>
#include <thread>

#include "gtest/gtest.h"

namespace {

struct FakeDeviceState {
  // Result of the diagnostic KMT call, separate from execution state.
  NTSTATUS query_status = STATUS_SUCCESS;
  // Execution state returned only by a successful query.
  D3DKMT_DEVICEEXECUTION_STATE execution_state = D3DKMT_DEVICEEXECUTION_ACTIVE;
  // Number of native diagnostics issued by the implementation.
  uint32_t query_count = 0;
};

FakeDeviceState* current_state = nullptr;

NTSTATUS APIENTRY FakeGetDeviceState(D3DKMT_GETDEVICESTATE* query) {
  EXPECT_EQ(query->hDevice, 0x10u);
  EXPECT_EQ(query->StateType, D3DKMT_DEVICESTATE_EXECUTION);
  ++current_state->query_count;
  query->ExecutionState = current_state->execution_state;
  return current_state->query_status;
}

class DeviceStatusTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    api_.get_device_state = FakeGetDeviceState;
    amdf_kmt_device_status_initialize(&status_);
  }

  void TearDown() override { current_state = nullptr; }

  amdf_status_t Observe(NTSTATUS status) {
    return amdf_kmt_device_status_observe_error(&status_, &api_, 0x10,
                                                amdf_kmt_make_status(status));
  }

  // Native diagnostic responses controlled by this test.
  FakeDeviceState state_;
  // Real KMT dependency seam with only the diagnostic entry point supplied.
  amdf_kmt_api_t api_ = {};
  // Production device-owned failure latch.
  amdf_kmt_device_status_t status_;
};

TEST_F(DeviceStatusTest, RejectedOperationDoesNotFailAnActiveDevice) {
  for (NTSTATUS status :
       {STATUS_NO_MEMORY, STATUS_INSUFFICIENT_RESOURCES,
        STATUS_INVALID_PARAMETER, STATUS_GRAPHICS_ALLOCATION_BUSY}) {
    EXPECT_EQ(Observe(status), amdf_kmt_make_status(status));
    EXPECT_EQ(amdf_kmt_device_status_query(&status_), AMDF_STATUS_OK);
  }
  EXPECT_EQ(state_.query_count, 4u);
  EXPECT_EQ(Observe(STATUS_SUCCESS), AMDF_STATUS_OK);
  EXPECT_EQ(state_.query_count, 4u);
}

TEST_F(DeviceStatusTest, TimeoutAndHostErrorsDoNotQueryOrPoisonDevice) {
  const amdf_status_t errors[] = {
      amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED),
      amdf_make_api_status(AMDF_STATUS_CODE_BUSY),
      amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_NOT_ENOUGH_MEMORY),
  };
  for (amdf_status_t error : errors) {
    EXPECT_EQ(
        amdf_kmt_device_status_observe_error(&status_, &api_, 0x10, error),
        error);
    EXPECT_EQ(amdf_kmt_device_status_query(&status_), AMDF_STATUS_OK);
  }
  EXPECT_EQ(state_.query_count, 0u);
}

TEST_F(DeviceStatusTest, ExplicitDeviceLossRemainsStickyWithoutDiagnostics) {
  const amdf_status_t removed = amdf_kmt_make_status(STATUS_DEVICE_REMOVED);
  EXPECT_EQ(Observe(STATUS_DEVICE_REMOVED), removed);
  EXPECT_EQ(Observe(STATUS_DEVICE_HUNG), removed);
  EXPECT_EQ(Observe(STATUS_NO_MEMORY), removed);
  EXPECT_EQ(Observe(STATUS_SUCCESS), removed);
  EXPECT_EQ(amdf_kmt_device_status_query(&status_), removed);
  EXPECT_EQ(state_.query_count, 0u);
}

TEST_F(DeviceStatusTest, DeviceExecutionOutOfMemoryIsNotOrdinaryRejection) {
  state_.execution_state = D3DKMT_DEVICEEXECUTION_ERROR_OUTOFMEMORY;
  const amdf_status_t lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(Observe(STATUS_NO_MEMORY), lost);
  state_.execution_state = D3DKMT_DEVICEEXECUTION_ACTIVE;
  EXPECT_EQ(Observe(STATUS_SUCCESS), lost);
  EXPECT_EQ(state_.query_count, 1u);
}

TEST_F(DeviceStatusTest, TerminalExecutionStatesAreLatched) {
  const D3DKMT_DEVICEEXECUTION_STATE states[] = {
      D3DKMT_DEVICEEXECUTION_RESET,
      D3DKMT_DEVICEEXECUTION_HUNG,
      D3DKMT_DEVICEEXECUTION_STOPPED,
      D3DKMT_DEVICEEXECUTION_ERROR_DMAFAULT,
      D3DKMT_DEVICEEXECUTION_ERROR_DMAPAGEFAULT,
  };
  for (D3DKMT_DEVICEEXECUTION_STATE state : states) {
    amdf_kmt_device_status_t status;
    amdf_kmt_device_status_initialize(&status);
    state_.execution_state = state;
    const amdf_status_t lost =
        amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
    EXPECT_EQ(
        amdf_kmt_device_status_observe_error(
            &status, &api_, 0x10, amdf_kmt_make_status(STATUS_UNSUCCESSFUL)),
        lost);
    EXPECT_EQ(amdf_kmt_device_status_query(&status), lost);
  }
}

TEST_F(DeviceStatusTest, FailedDiagnosticIsReportedWithoutInventingDeviceLoss) {
  state_.query_status = STATUS_ACCESS_DENIED;
  EXPECT_EQ(Observe(STATUS_UNSUCCESSFUL),
            amdf_kmt_make_status(STATUS_ACCESS_DENIED));
  EXPECT_EQ(amdf_kmt_device_status_query(&status_), AMDF_STATUS_OK);
  state_.query_status = STATUS_SUCCESS;
  EXPECT_EQ(Observe(STATUS_NO_MEMORY), amdf_kmt_make_status(STATUS_NO_MEMORY));
  EXPECT_EQ(amdf_kmt_device_status_query(&status_), AMDF_STATUS_OK);
}

TEST_F(DeviceStatusTest, DiagnosticRemovalPreservesNativeCause) {
  state_.query_status = STATUS_DEVICE_REMOVED;
  EXPECT_EQ(Observe(STATUS_UNSUCCESSFUL),
            amdf_kmt_make_status(STATUS_DEVICE_REMOVED));
  EXPECT_EQ(amdf_kmt_device_status_query(&status_),
            amdf_kmt_make_status(STATUS_DEVICE_REMOVED));
}

TEST_F(DeviceStatusTest, UnknownExecutionStateDoesNotInventTerminality) {
  state_.execution_state = static_cast<D3DKMT_DEVICEEXECUTION_STATE>(0x1000);
  EXPECT_EQ(Observe(STATUS_UNSUCCESSFUL),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(amdf_kmt_device_status_query(&status_), AMDF_STATUS_OK);
}

TEST_F(DeviceStatusTest, ConcurrentFatalErrorsPreserveOneCompleteCause) {
  const NTSTATUS errors[] = {STATUS_DEVICE_REMOVED, STATUS_DEVICE_HUNG,
                             STATUS_GRAPHICS_GPU_EXCEPTION_ON_DEVICE};
  std::array<amdf_status_t, 3> results;
  std::array<std::thread, 3> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] { results[i] = Observe(errors[i]); });
  }
  for (auto& thread : threads) thread.join();
  const amdf_status_t terminal = amdf_kmt_device_status_query(&status_);
  EXPECT_TRUE(terminal == amdf_kmt_make_status(errors[0]) ||
              terminal == amdf_kmt_make_status(errors[1]) ||
              terminal == amdf_kmt_make_status(errors[2]));
  for (amdf_status_t result : results) EXPECT_EQ(result, terminal);
  EXPECT_EQ(state_.query_count, 0u);
}

}  // namespace
