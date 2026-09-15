// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/private_allocation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103u);
constexpr NTSTATUS kStatusUnsuccessful = static_cast<NTSTATUS>(0xC0000001u);

enum class Operation {
  kCreate,
  kMap,
  kWait,
  kResident,
  kLock,
  kInvalidate,
  kUnlock,
  kDestroy,
};

struct FakeKmtState {
  // Host lock storage supplied by the native dependency.
  std::array<uint8_t, 8192> host_bytes = {};
  // Exact allocation-private request before the driver writes its reply.
  std::array<uint8_t, 56> private_data = {};
  // Firmware aperture base returned independently of the GPU VA mapping.
  uint64_t firmware_address = 0;
  // Residency policy captured from the native request.
  D3DDDI_MAKERESIDENT_FLAGS resident_flags = {};
  // Allocation-relative cache publication offset.
  uint64_t invalidated_offset = 0;
  // Length of the cache publication request.
  uint64_t invalidated_length = 0;
  // One-based native paging wait that fails, or zero for success.
  uint32_t fail_wait_call = 0;
  // Number of paging waits observed.
  uint32_t wait_count = 0;
  // Requested paging fence values in call order.
  std::vector<uint64_t> wait_targets;
  // Native operation order observed during realization and teardown.
  std::vector<Operation> operations;
};

FakeKmtState* g_fake_state = nullptr;

uint32_t ReadU32(const void* bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, static_cast<const uint8_t*>(bytes) + offset,
              sizeof(value));
  return value;
}

uint64_t ReadU64(const void* bytes, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, static_cast<const uint8_t*>(bytes) + offset,
              sizeof(value));
  return value;
}

NTSTATUS APIENTRY FakeCreateAllocation(D3DKMT_CREATEALLOCATION* create) {
  g_fake_state->operations.push_back(Operation::kCreate);
  EXPECT_EQ(create->NumAllocations, 1u);
  EXPECT_EQ(create->pAllocationInfo2[0].PrivateDriverDataSize,
            g_fake_state->private_data.size());
  std::memcpy(g_fake_state->private_data.data(),
              create->pAllocationInfo2[0].pPrivateDriverData,
              g_fake_state->private_data.size());
  create->hResource = create->Flags.CreateResource ? 0x21 : 0;
  create->pAllocationInfo2[0].hAllocation = 0x20;
  std::memcpy(
      static_cast<uint8_t*>(create->pAllocationInfo2[0].pPrivateDriverData) +
          0x30,
      &g_fake_state->firmware_address, sizeof(uint64_t));
  return 0;
}

NTSTATUS APIENTRY
FakeDestroyAllocation(const D3DKMT_DESTROYALLOCATION2* destroy) {
  g_fake_state->operations.push_back(Operation::kDestroy);
  EXPECT_EQ(destroy->hDevice, 0x10u);
  EXPECT_EQ(destroy->Flags.AssumeNotInUse, 1u);
  return 0;
}

NTSTATUS APIENTRY FakeMapGpuVirtualAddress(D3DDDI_MAPGPUVIRTUALADDRESS* map) {
  g_fake_state->operations.push_back(Operation::kMap);
  EXPECT_EQ(map->hPagingQueue, 0x30u);
  EXPECT_EQ(map->hAllocation, 0x20u);
  EXPECT_EQ(map->SizeInPages, 2u);
  EXPECT_EQ(map->Protection.Write, 1u);
  map->VirtualAddress = UINT64_C(0x12340000);
  map->PagingFenceValue = 1;
  return kStatusPending;
}

NTSTATUS APIENTRY FakeMakeResident(D3DDDI_MAKERESIDENT* resident) {
  g_fake_state->operations.push_back(Operation::kResident);
  g_fake_state->resident_flags = resident->Flags;
  EXPECT_EQ(resident->hPagingQueue, 0x30u);
  EXPECT_EQ(resident->NumAllocations, 1u);
  EXPECT_EQ(resident->AllocationList[0], 0x20u);
  resident->PagingFenceValue = 2;
  return kStatusPending;
}

NTSTATUS APIENTRY
FakeWaitFromCpu(const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) {
  g_fake_state->operations.push_back(Operation::kWait);
  ++g_fake_state->wait_count;
  EXPECT_EQ(wait->hDevice, 0x10u);
  EXPECT_EQ(wait->ObjectCount, 1u);
  EXPECT_EQ(wait->ObjectHandleArray[0], 0x40u);
  g_fake_state->wait_targets.push_back(wait->FenceValueArray[0]);
  return g_fake_state->fail_wait_call == g_fake_state->wait_count
             ? kStatusUnsuccessful
             : 0;
}

NTSTATUS APIENTRY FakeLock(D3DKMT_LOCK2* lock) {
  g_fake_state->operations.push_back(Operation::kLock);
  EXPECT_EQ(lock->hDevice, 0x10u);
  EXPECT_EQ(lock->hAllocation, 0x20u);
  lock->pData = g_fake_state->host_bytes.data();
  return 0;
}

NTSTATUS APIENTRY FakeUnlock(const D3DKMT_UNLOCK2* unlock) {
  g_fake_state->operations.push_back(Operation::kUnlock);
  EXPECT_EQ(unlock->hDevice, 0x10u);
  EXPECT_EQ(unlock->hAllocation, 0x20u);
  return 0;
}

NTSTATUS APIENTRY FakeInvalidate(const D3DKMT_INVALIDATECACHE* invalidate) {
  g_fake_state->operations.push_back(Operation::kInvalidate);
  EXPECT_EQ(invalidate->hDevice, 0x10u);
  EXPECT_EQ(invalidate->hAllocation, 0x20u);
  g_fake_state->invalidated_offset = invalidate->Offset;
  g_fake_state->invalidated_length = invalidate->Length;
  return 0;
}

class WindowsXdnaPrivateAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_fake_state = &state_;
    kmt_.create_allocation = FakeCreateAllocation;
    kmt_.destroy_allocation = FakeDestroyAllocation;
    kmt_.map_gpu_virtual_address = FakeMapGpuVirtualAddress;
    kmt_.make_resident = FakeMakeResident;
    kmt_.wait_from_cpu = FakeWaitFromCpu;
    kmt_.lock = FakeLock;
    kmt_.unlock = FakeUnlock;
    kmt_.invalidate_cache = FakeInvalidate;
    device_.kmt = &kmt_;
    device_.device = 0x10;
    device_.paging_queue = 0x30;
    device_.paging_sync_object = 0x40;
    device_.paging_fence = &paging_fence_;
  }

  void TearDown() override { g_fake_state = nullptr; }

  FakeKmtState state_;
  amdf_kmt_api_t kmt_ = {};
  amdf_xdna_umd_device_t device_ = {};
  volatile uint64_t paging_fence_ = 0;
};

TEST_F(WindowsXdnaPrivateAllocationTest,
       RealizesMapsPublishesAndReleasesPrivateAllocation) {
  const amdf_windows_xdna_private_allocation_descriptor_t descriptor = {
      .requested_byte_length = 4200,
      .allocation_byte_length = 8192,
      .type = 0x3328,
      .policy = 2,
      .xcl_flags = 0x80000000u,
      .selector = 7,
      .flags = AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS,
  };
  amdf_windows_xdna_private_allocation_t allocation = {};
  amdf_windows_xdna_private_allocation_initialize(&device_, &descriptor,
                                                  &allocation);
  EXPECT_TRUE(state_.operations.empty());
  EXPECT_EQ(allocation.device, &device_);
  EXPECT_EQ(allocation.descriptor.requested_byte_length,
            descriptor.requested_byte_length);
  EXPECT_EQ(allocation.realization_phase, 0u);
  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_realize(&allocation)));

  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kResident,
                                    Operation::kWait}));
  EXPECT_EQ(ReadU64(state_.private_data.data(), 0x08), 4200u);
  EXPECT_EQ(ReadU64(state_.private_data.data(), 0x10), 8192u);
  EXPECT_EQ(ReadU32(state_.private_data.data(), 0x18), 7u);
  EXPECT_EQ(ReadU32(state_.private_data.data(), 0x1C), 0x3328u);
  EXPECT_EQ(ReadU32(state_.private_data.data(), 0x20), 2u);
  EXPECT_EQ(ReadU32(state_.private_data.data(), 0x28), 0x80000000u);
  EXPECT_EQ(state_.resident_flags.CantTrimFurther, 0u);
  EXPECT_EQ(state_.resident_flags.MustSucceed, 0u);
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(allocation.device_address, UINT64_C(0x12340000));

  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_lock(&allocation)));
  ASSERT_EQ(allocation.host_pointer, state_.host_bytes.data());
  std::memset(static_cast<uint8_t*>(allocation.host_pointer) + 64, 0xA5, 128);
  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_publish(&allocation, 64, 128)));
  EXPECT_EQ(state_.invalidated_offset, 64u);
  EXPECT_EQ(state_.invalidated_length, 128u);

  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_destroy(&allocation)));
  EXPECT_EQ(state_.operations[state_.operations.size() - 2],
            Operation::kUnlock);
  EXPECT_EQ(state_.operations.back(), Operation::kDestroy);
  EXPECT_EQ(allocation.device, nullptr);
}

TEST_F(WindowsXdnaPrivateAllocationTest,
       PreservesFirmwareAddressAcrossPagingWaits) {
  const amdf_windows_xdna_private_allocation_descriptor_t descriptor = {
      .requested_byte_length = 8192,
      .allocation_byte_length = 8192,
      .type = 0x3323,
      .policy = 2,
      .xcl_flags = 0x010e0001,
      .selector = 1,
      .flags = AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS,
  };
  for (uint32_t fail_wait : {0u, 1u, 2u}) {
    state_ = {};
    state_.firmware_address = UINT64_C(0x8000000);
    state_.fail_wait_call = fail_wait;
    amdf_windows_xdna_private_allocation_t allocation = {};
    amdf_windows_xdna_private_allocation_initialize(&device_, &descriptor,
                                                    &allocation);
    EXPECT_EQ(allocation.firmware_address, 0u);
    const amdf_status_t status =
        amdf_windows_xdna_private_allocation_realize(&allocation);
    EXPECT_EQ(amdf_status_is_ok(status), fail_wait == 0);
    EXPECT_EQ(allocation.firmware_address, state_.firmware_address);
    EXPECT_EQ(ReadU64(state_.private_data.data(), 0x30), 0u);
    if (fail_wait != 0) {
      // The resumed paging operation has no allocation reply to reread.
      ASSERT_EQ(amdf_windows_xdna_private_allocation_realize(&allocation),
                AMDF_STATUS_OK);
    }
    EXPECT_EQ(allocation.firmware_address, state_.firmware_address);
    EXPECT_EQ(allocation.device_address, UINT64_C(0x12340000));
    EXPECT_NE(allocation.firmware_address, allocation.device_address);
    ASSERT_EQ(amdf_windows_xdna_private_allocation_destroy(&allocation),
              AMDF_STATUS_OK);
    EXPECT_EQ(allocation.firmware_address, 0u);
  }
}

TEST_F(WindowsXdnaPrivateAllocationTest,
       RealizesSharedHostLockedAllocationWithoutDeviceMapping) {
  const amdf_windows_xdna_private_allocation_descriptor_t descriptor = {
      .requested_byte_length = 4096,
      .allocation_byte_length = 4096,
      .type = 0x332B,
      .flags = AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_SHARED_RESOURCE,
  };
  amdf_windows_xdna_private_allocation_t allocation = {};
  amdf_windows_xdna_private_allocation_initialize(&device_, &descriptor,
                                                  &allocation);
  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_realize(&allocation)));
  EXPECT_EQ(state_.operations, (std::vector<Operation>{Operation::kCreate}));
  EXPECT_EQ(allocation.resource, 0x21u);
  EXPECT_EQ(allocation.device_address, 0u);
  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_lock(&allocation)));
  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_destroy(&allocation)));
}

TEST_F(WindowsXdnaPrivateAllocationTest,
       RetriesMapWaitWithoutSubmittingMapTwice) {
  state_.fail_wait_call = 1;
  const amdf_windows_xdna_private_allocation_descriptor_t descriptor = {
      .requested_byte_length = 4200,
      .allocation_byte_length = 8192,
      .type = 0x3328,
      .policy = 2,
      .xcl_flags = 0x80000000u,
      .flags = AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS,
  };
  amdf_windows_xdna_private_allocation_t allocation = {};
  amdf_windows_xdna_private_allocation_initialize(&device_, &descriptor,
                                                  &allocation);
  EXPECT_FALSE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_realize(&allocation)));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait}));

  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_realize(&allocation)));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kWait,
                                    Operation::kResident, Operation::kWait}));
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1, 1, 2}));
  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_destroy(&allocation)));
}

TEST_F(WindowsXdnaPrivateAllocationTest,
       RetriesResidencyWaitWithoutSubmittingResidencyTwice) {
  state_.fail_wait_call = 2;
  const amdf_windows_xdna_private_allocation_descriptor_t descriptor = {
      .requested_byte_length = 4200,
      .allocation_byte_length = 8192,
      .type = 0x3328,
      .policy = 2,
      .xcl_flags = 0x80000000u,
      .flags = AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS,
  };
  amdf_windows_xdna_private_allocation_t allocation = {};
  amdf_windows_xdna_private_allocation_initialize(&device_, &descriptor,
                                                  &allocation);
  EXPECT_FALSE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_realize(&allocation)));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kResident,
                                    Operation::kWait}));

  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_realize(&allocation)));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kResident,
                                    Operation::kWait, Operation::kWait}));
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1, 2, 2}));
  ASSERT_TRUE(amdf_status_is_ok(
      amdf_windows_xdna_private_allocation_destroy(&allocation)));
}

}  // namespace
