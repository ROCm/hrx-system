// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/device.h"

#include <malloc.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/windows/endpoint.h"
#include "libamdf/src/xdna/endpoint_profile.h"

namespace {

constexpr NTSTATUS kSuccess = 0;
constexpr NTSTATUS kFailure = static_cast<NTSTATUS>(0xC0000001u);

enum class Operation {
  kCreateDevice,
  kCreatePagingQueue,
  kDestroyPagingQueue,
  kDestroyDevice,
  kCloseAdapter,
};

struct FakeKmtState {
  // Number of paging-queue releases rejected before native consumption.
  uint32_t paging_queue_destroy_failures_remaining = 2;
  // Sync-object handle returned with the paging queue, or zero for malformed
  // output.
  D3DKMT_HANDLE paging_sync_object = 0;
  // Ordered native operations used to verify local and published ownership.
  std::vector<Operation> operations;
  // Number of host allocations retained by endpoint/device bookkeeping.
  uint32_t live_allocation_count = 0;
  // Number of successfully released paging queues.
  uint32_t paging_queue_destroy_success_count = 0;
  // Number of successfully released logical devices.
  uint32_t device_destroy_success_count = 0;
  // Number of successfully closed endpoint adapters.
  uint32_t adapter_close_success_count = 0;
  // Mapped paging progress returned with the malformed queue result.
  volatile uint64_t paging_progress = 0;
};

FakeKmtState* current_state = nullptr;

void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                         uint64_t minimum_alignment) {
  auto* state = static_cast<FakeKmtState*>(user_data);
  void* pointer = _aligned_malloc(static_cast<size_t>(byte_length),
                                  static_cast<size_t>(minimum_alignment));
  if (pointer != nullptr) ++state->live_allocation_count;
  return pointer;
}

void AMDF_CALL Free(void* user_data, void* allocation) {
  if (allocation == nullptr) return;
  auto* state = static_cast<FakeKmtState*>(user_data);
  EXPECT_NE(state->live_allocation_count, 0u);
  --state->live_allocation_count;
  _aligned_free(allocation);
}

NTSTATUS APIENTRY FakeQueryAdapterInfo(const D3DKMT_QUERYADAPTERINFO* query) {
  (void)query;
  ADD_FAILURE() << "ordinary device creation needs no private context ABI";
  return kFailure;
}

NTSTATUS APIENTRY FakeCreateDevice(D3DKMT_CREATEDEVICE* create) {
  current_state->operations.push_back(Operation::kCreateDevice);
  EXPECT_EQ(create->hAdapter, 0x08u);
  create->hDevice = 0x10;
  return kSuccess;
}

NTSTATUS APIENTRY FakeDestroyDevice(const D3DKMT_DESTROYDEVICE* destroy) {
  current_state->operations.push_back(Operation::kDestroyDevice);
  EXPECT_EQ(destroy->hDevice, 0x10u);
  ++current_state->device_destroy_success_count;
  return kSuccess;
}

NTSTATUS APIENTRY FakeGetDeviceState(D3DKMT_GETDEVICESTATE*) {
  ADD_FAILURE() << "device-state query is not part of this construction path";
  return kFailure;
}

NTSTATUS APIENTRY FakeCreatePagingQueue(D3DKMT_CREATEPAGINGQUEUE* create) {
  current_state->operations.push_back(Operation::kCreatePagingQueue);
  EXPECT_EQ(create->hDevice, 0x10u);
  create->hPagingQueue = 0x20;
  create->hSyncObject = current_state->paging_sync_object;
  create->FenceValueCPUVirtualAddress =
      const_cast<uint64_t*>(&current_state->paging_progress);
  return kSuccess;
}

NTSTATUS APIENTRY FakeDestroyPagingQueue(D3DDDI_DESTROYPAGINGQUEUE* destroy) {
  current_state->operations.push_back(Operation::kDestroyPagingQueue);
  EXPECT_EQ(destroy->hPagingQueue, 0x20u);
  if (current_state->paging_queue_destroy_failures_remaining != 0) {
    --current_state->paging_queue_destroy_failures_remaining;
    return kFailure;
  }
  ++current_state->paging_queue_destroy_success_count;
  return kSuccess;
}

NTSTATUS APIENTRY FakeCreateContextVirtual(D3DKMT_CREATECONTEXTVIRTUAL*) {
  ADD_FAILURE() << "malformed paging state must stop context construction";
  return kFailure;
}

NTSTATUS APIENTRY FakeDestroyContext(const D3DKMT_DESTROYCONTEXT*) {
  ADD_FAILURE() << "no context was acquired by this construction path";
  return kFailure;
}

NTSTATUS APIENTRY FakeCloseAdapter(const D3DKMT_CLOSEADAPTER* close) {
  current_state->operations.push_back(Operation::kCloseAdapter);
  EXPECT_EQ(close->hAdapter, 0x08u);
  ++current_state->adapter_close_success_count;
  return kSuccess;
}

class WindowsXdnaDeviceRollbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    instance_.host_allocator = {
        .user_data = &state_,
        .allocate = Allocate,
        .free = Free,
    };
    instance_.kmt.query_adapter_info = FakeQueryAdapterInfo;
    instance_.kmt.create_device = FakeCreateDevice;
    instance_.kmt.destroy_device = FakeDestroyDevice;
    instance_.kmt.get_device_state = FakeGetDeviceState;
    instance_.kmt.create_paging_queue = FakeCreatePagingQueue;
    instance_.kmt.destroy_paging_queue = FakeDestroyPagingQueue;
    instance_.kmt.create_context_virtual = FakeCreateContextVirtual;
    instance_.kmt.destroy_context = FakeDestroyContext;
    instance_.kmt.close_adapter = FakeCloseAdapter;

    ASSERT_EQ(amdf_calloc(instance_.host_allocator, sizeof(*endpoint_),
                          amdf_alignof(amdf_platform_endpoint_t),
                          reinterpret_cast<void**>(&endpoint_)),
              AMDF_STATUS_OK);
    endpoint_->instance = &instance_;
    endpoint_->adapter = 0x08;
    endpoint_->physical_adapter_index = 0;

    endpoint_info_.array.column_origin = 0;
    endpoint_info_.array.column_count = 8;
    profile_.execution_capabilities =
        AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1;
    profile_.info = &endpoint_info_;
  }

  void TearDown() override {
    state_.paging_queue_destroy_failures_remaining = 0;
    if (endpoint_ != nullptr) {
      EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
      endpoint_ = nullptr;
    }
    EXPECT_EQ(state_.live_allocation_count, 0u);
    current_state = nullptr;
  }

  FakeKmtState state_;
  amdf_platform_instance_t instance_ = {};
  amdf_platform_endpoint_t* endpoint_ = nullptr;
  amdf_xdna_endpoint_info_t endpoint_info_ = {};
  amdf_xdna_endpoint_profile_t profile_ = {};
};

TEST_F(WindowsXdnaDeviceRollbackTest,
       CreatesPagingDeviceWithoutInterpreterOrContextProcedures) {
  profile_.execution_capabilities = 0;
  const auto capabilities = amdf_xdna_umd_query_context_capabilities(&profile_);
  EXPECT_EQ(capabilities.scheduling_modes, 0u);
  EXPECT_EQ(capabilities.placement_modes, 0u);
  instance_.kmt.create_context_virtual = nullptr;
  instance_.kmt.destroy_context = nullptr;
  state_.paging_sync_object = 0x21;
  state_.paging_queue_destroy_failures_remaining = 0;
  amdf_xdna_umd_device_t* device = nullptr;
  amdf_xdna_umd_device_result_t result = {};
  ASSERT_EQ(
      amdf_xdna_umd_device_create(endpoint_, &profile_,
                                  instance_.host_allocator, &device, &result),
      AMDF_STATUS_OK);
  ASSERT_NE(device, nullptr);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreateDevice,
                                    Operation::kCreatePagingQueue}));
  EXPECT_EQ(amdf_xdna_umd_device_destroy(device), AMDF_STATUS_OK);
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 1u);
  EXPECT_EQ(state_.device_destroy_success_count, 1u);
}

TEST_F(WindowsXdnaDeviceRollbackTest,
       ReportsNoFixedPlacementBeforeAndAfterDeviceCreation) {
  const auto capabilities = amdf_xdna_umd_query_context_capabilities(&profile_);
  EXPECT_EQ(capabilities.scheduling_modes,
            AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED);
  EXPECT_EQ(capabilities.placement_modes, 0u);
  EXPECT_TRUE(state_.operations.empty());
  state_.paging_sync_object = 0x21;
  state_.paging_queue_destroy_failures_remaining = 0;
  amdf_xdna_umd_device_t* device = nullptr;
  amdf_xdna_umd_device_result_t result = {};
  ASSERT_EQ(
      amdf_xdna_umd_device_create(endpoint_, &profile_,
                                  instance_.host_allocator, &device, &result),
      AMDF_STATUS_OK);
  EXPECT_EQ(result.placement_modes, 0u);
  EXPECT_EQ(amdf_xdna_umd_device_destroy(device), AMDF_STATUS_OK);
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 1u);
  EXPECT_EQ(state_.device_destroy_success_count, 1u);
}

TEST_F(WindowsXdnaDeviceRollbackTest,
       ReportsFailedRollbackWithoutRetainingDevice) {
  amdf_xdna_umd_device_t* device =
      reinterpret_cast<amdf_xdna_umd_device_t*>(uintptr_t{1});
  amdf_xdna_umd_device_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original_result = result;

  const amdf_status_t status = amdf_xdna_umd_device_create(
      endpoint_, &profile_, instance_.host_allocator, &device, &result);

  EXPECT_EQ(status, amdf_kmt_make_status(kFailure));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(device), uintptr_t{1});
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(state_.live_allocation_count, 1u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreateDevice,
                                    Operation::kCreatePagingQueue,
                                    Operation::kDestroyPagingQueue}));
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 0u);
  EXPECT_EQ(state_.device_destroy_success_count, 0u);
  EXPECT_EQ(state_.adapter_close_success_count, 0u);

  EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
  endpoint_ = nullptr;
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kCreateDevice, Operation::kCreatePagingQueue,
                Operation::kDestroyPagingQueue, Operation::kCloseAdapter}));
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 0u);
  EXPECT_EQ(state_.device_destroy_success_count, 0u);
  EXPECT_EQ(state_.adapter_close_success_count, 1u);
  EXPECT_EQ(state_.live_allocation_count, 0u);
}

TEST_F(WindowsXdnaDeviceRollbackTest,
       ExplicitDestroyFailureRetainsPublishedDevice) {
  state_.paging_sync_object = 0x21;
  state_.paging_queue_destroy_failures_remaining = 1;
  amdf_xdna_umd_device_t* device = nullptr;
  amdf_xdna_umd_device_result_t result = {};
  ASSERT_EQ(
      amdf_xdna_umd_device_create(endpoint_, &profile_,
                                  instance_.host_allocator, &device, &result),
      AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_device_destroy(device),
            amdf_kmt_make_status(kFailure));
  EXPECT_EQ(state_.live_allocation_count, 2u);
  EXPECT_EQ(state_.device_destroy_success_count, 0u);

  EXPECT_EQ(amdf_xdna_umd_device_destroy(device), AMDF_STATUS_OK);
  EXPECT_EQ(state_.live_allocation_count, 1u);
  EXPECT_EQ(state_.paging_queue_destroy_success_count, 1u);
  EXPECT_EQ(state_.device_destroy_success_count, 1u);
}

}  // namespace
