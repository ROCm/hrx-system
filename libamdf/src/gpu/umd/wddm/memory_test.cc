// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/wddm/device.h"

namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103u);
constexpr NTSTATUS kStatusNoMemory = static_cast<NTSTATUS>(0xC0000017u);

enum class Operation {
  kQueryLayout,
  kReserveAddress,
  kCreateAllocation,
  kMap,
  kWait,
  kMakeResident,
  kUnmap,
  kEvict,
  kDestroyAllocation,
  kFreeAddress,
};

struct FakeMemoryState {
  // Native allocation domain selected by the caller's profile.
  amdf_wkmi_bridge_gpu_allocation_domain_t allocation_domain =
      AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL;
  // Allocation handle returned by native success, including malformed zero.
  D3DKMT_HANDLE allocation_handle = 0x20;
  // Resource handle returned by native success, or zero when ungrouped.
  D3DKMT_HANDLE resource_handle = 0;
  // Number of allocation slots in the native result.
  uint32_t allocation_count = 1;
  // Status returned by allocation destruction.
  NTSTATUS destroy_status = 0;
  // Host backing borrowed by native allocation creation, if any.
  void* host_pointer = nullptr;
  // Number of memory headers returned to the host allocator.
  uint32_t metadata_free_count = 0;
  // Write protection expected on an ordinary mapping request.
  uint32_t expected_write = 1;
  // Execute protection expected on an ordinary mapping request.
  uint32_t expected_execute = 0;
  // Address returned by the native mapping request, including unexpected
  // output.
  uint64_t mapped_device_address = UINT64_C(0x100000);
  // Next fence value assigned to an accepted paging operation.
  uint64_t next_paging_fence = 1;
  // Fence whose CPU wait should fail, or zero for none.
  uint64_t failing_wait_target = 0;
  // Number of matching CPU waits rejected before observation succeeds.
  uint32_t wait_failures_remaining = 0;
  // Number of page-table removal requests accepted by the fake.
  uint32_t unmap_count = 0;
  // Monitored paging progress exposed to production code.
  volatile uint64_t paging_fence = 0;
  // Native and bridge operations in call order.
  std::vector<Operation> operations;
  // Paging fence targets observed by CPU waits.
  std::vector<uint64_t> wait_targets;
};

FakeMemoryState* current_state = nullptr;

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL
FakeQueryAllocationLayout(amdf_wkmi_bridge_gpu_adapter_t* adapter,
                          uint64_t byte_length, uint32_t* out_allocation_count,
                          uint64_t* out_maximum_allocation_byte_length) {
  auto* state = reinterpret_cast<FakeMemoryState*>(adapter);
  state->operations.push_back(Operation::kQueryLayout);
  EXPECT_EQ(byte_length, UINT64_C(65536));
  *out_allocation_count = state->allocation_count;
  *out_maximum_allocation_byte_length = 65536 / state->allocation_count;
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

amdf_wkmi_bridge_result_t AMDF_WKMI_BRIDGE_CALL FakeCreateAllocation(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeMemoryState*>(adapter);
  state->operations.push_back(Operation::kCreateAllocation);
  EXPECT_EQ(create_info->device_handle, 0x10u);
  EXPECT_EQ(create_info->domain, state->allocation_domain);
  EXPECT_EQ(create_info->byte_length, UINT64_C(65536));
  if (state->allocation_domain ==
      AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL) {
    EXPECT_EQ(create_info->placement_device_address, UINT64_C(0x100000));
    EXPECT_EQ(create_info->host_pointer, nullptr);
  } else {
    EXPECT_EQ(create_info->placement_device_address, 0u);
    EXPECT_NE(create_info->host_pointer, nullptr);
  }
  state->host_pointer = create_info->host_pointer;
  EXPECT_EQ(allocation_handle_capacity, state->allocation_count);
  out_allocation_handles[0] = state->allocation_handle;
  for (uint32_t i = 1; i < state->allocation_count; ++i) {
    out_allocation_handles[i] = 0x20;
  }
  *out_resource_handle = state->resource_handle;
  *out_allocation_count = state->allocation_count;
  *out_native_status = 0;
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

NTSTATUS APIENTRY FakeUnexpectedCreateAllocation(D3DKMT_CREATEALLOCATION*) {
  ADD_FAILURE() << "WKMI owns physical allocation construction";
  return kStatusNoMemory;
}

NTSTATUS APIENTRY
FakeReserveGpuVirtualAddress(D3DDDI_RESERVEGPUVIRTUALADDRESS* reserve) {
  current_state->operations.push_back(Operation::kReserveAddress);
  EXPECT_EQ(reserve->hAdapter, 0x08u);
  EXPECT_EQ(reserve->Size, UINT64_C(65536));
  reserve->VirtualAddress = 0x100000;
  return 0;
}

NTSTATUS APIENTRY FakeMapGpuVirtualAddress(D3DDDI_MAPGPUVIRTUALADDRESS* map) {
  if (map->Protection.NoAccess != 0) {
    current_state->operations.push_back(Operation::kUnmap);
    ++current_state->unmap_count;
    EXPECT_EQ(map->BaseAddress, current_state->mapped_device_address);
    EXPECT_EQ(map->hAllocation, 0u);
  } else {
    current_state->operations.push_back(Operation::kMap);
    EXPECT_EQ(map->BaseAddress, UINT64_C(0x100000));
    EXPECT_EQ(map->hAllocation, 0x20u);
    EXPECT_EQ(map->Protection.Write, current_state->expected_write);
    EXPECT_EQ(map->Protection.Execute, current_state->expected_execute);
    map->VirtualAddress = current_state->mapped_device_address;
  }
  EXPECT_EQ(map->hPagingQueue, 0x30u);
  EXPECT_EQ(map->SizeInPages, 16u);
  map->PagingFenceValue = current_state->next_paging_fence++;
  return kStatusPending;
}

NTSTATUS APIENTRY FakeMakeResident(D3DDDI_MAKERESIDENT* resident) {
  current_state->operations.push_back(Operation::kMakeResident);
  EXPECT_EQ(resident->hPagingQueue, 0x30u);
  EXPECT_EQ(resident->NumAllocations, 1u);
  EXPECT_EQ(resident->AllocationList[0], 0x20u);
  EXPECT_EQ(resident->Flags.CantTrimFurther, 1u);
  resident->PagingFenceValue = current_state->next_paging_fence++;
  return kStatusPending;
}

NTSTATUS APIENTRY
FakeWaitFromCpu(const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) {
  current_state->operations.push_back(Operation::kWait);
  EXPECT_EQ(wait->hDevice, 0x10u);
  EXPECT_EQ(wait->ObjectCount, 1u);
  EXPECT_EQ(wait->ObjectHandleArray[0], 0x40u);
  const uint64_t target = wait->FenceValueArray[0];
  current_state->wait_targets.push_back(target);
  if (target == current_state->failing_wait_target &&
      current_state->wait_failures_remaining != 0) {
    --current_state->wait_failures_remaining;
    return kStatusNoMemory;
  }
  current_state->paging_fence = target;
  return 0;
}

NTSTATUS APIENTRY FakeEvict(D3DKMT_EVICT* evict) {
  current_state->operations.push_back(Operation::kEvict);
  EXPECT_EQ(evict->hDevice, 0x10u);
  EXPECT_EQ(evict->NumAllocations, 1u);
  EXPECT_EQ(evict->AllocationList[0], 0x20u);
  return 0;
}

NTSTATUS APIENTRY
FakeDestroyAllocation(const D3DKMT_DESTROYALLOCATION2* destroy) {
  current_state->operations.push_back(Operation::kDestroyAllocation);
  EXPECT_EQ(destroy->hDevice, 0x10u);
  EXPECT_EQ(destroy->hResource, current_state->resource_handle);
  if (current_state->resource_handle != 0) {
    EXPECT_EQ(destroy->AllocationCount, 0u);
    EXPECT_EQ(destroy->phAllocationList, nullptr);
  } else {
    EXPECT_EQ(destroy->AllocationCount, 1u);
    EXPECT_EQ(destroy->phAllocationList[0], 0x20u);
  }
  EXPECT_EQ(destroy->Flags.AssumeNotInUse, 1u);
  return current_state->destroy_status;
}

NTSTATUS APIENTRY
FakeFreeGpuVirtualAddress(const D3DKMT_FREEGPUVIRTUALADDRESS* free_address) {
  current_state->operations.push_back(Operation::kFreeAddress);
  EXPECT_EQ(free_address->hAdapter, 0x08u);
  EXPECT_EQ(free_address->BaseAddress, UINT64_C(0x100000));
  EXPECT_EQ(free_address->Size, UINT64_C(65536));
  return 0;
}

NTSTATUS APIENTRY FakeUnexpectedInvalidateCache(const D3DKMT_INVALIDATECACHE*) {
  ADD_FAILURE() << "this test creates device-local memory without host access";
  return kStatusNoMemory;
}

class WindowsGpuMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    bridge_.gpu_allocation_query_layout = FakeQueryAllocationLayout;
    bridge_.gpu_allocation_create = FakeCreateAllocation;
    kmt_.create_allocation = FakeUnexpectedCreateAllocation;
    kmt_.destroy_allocation = FakeDestroyAllocation;
    kmt_.reserve_gpu_virtual_address = FakeReserveGpuVirtualAddress;
    kmt_.free_gpu_virtual_address = FakeFreeGpuVirtualAddress;
    kmt_.map_gpu_virtual_address = FakeMapGpuVirtualAddress;
    kmt_.make_resident = FakeMakeResident;
    kmt_.evict = FakeEvict;
    kmt_.invalidate_cache = FakeUnexpectedInvalidateCache;
    kmt_.wait_from_cpu = FakeWaitFromCpu;
    device_.host_allocator = amdf_allocator_system();
    device_.host_allocator.user_data = &state_;
    device_.host_allocator.free = [](void* user_data, void* allocation) {
      ++static_cast<FakeMemoryState*>(user_data)->metadata_free_count;
      amdf_free(amdf_allocator_system(), allocation);
    };
    device_.kmt = &kmt_;
    device_.adapter = 0x08;
    device_.device = 0x10;
    device_.paging_queue = 0x30;
    device_.paging_sync_object = 0x40;
    device_.paging_fence = &state_.paging_fence;
    device_.memory_capabilities.virtual_address_bit_count = 48;
    device_.memory_capabilities.read_only_memory_supported = 1;
    device_.memory_capabilities.no_execute_memory_supported = 1;
    device_.memory_capabilities.cache_coherent_memory_supported = 1;
    device_.wkmi_adapter.api = &bridge_;
    device_.wkmi_adapter.native =
        reinterpret_cast<amdf_wkmi_bridge_gpu_adapter_t*>(&state_);

    create_info_.device_access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    create_info_.required_flags =
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    create_info_.byte_length = 65536;
    create_info_.minimum_alignment = 65536;
    ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 1, &profile_),
              AMDF_STATUS_OK);
  }

  void TearDown() override { current_state = nullptr; }

  // Native dependency responses and ordered observations.
  FakeMemoryState state_;
  // Existing WKMI adapter seam used by production allocation code.
  amdf_wkmi_bridge_api_t bridge_ = {};
  // Native KMT procedures used by production attachment and release code.
  amdf_kmt_api_t kmt_ = {};
  // Live device state borrowed by the memory attachment under test.
  amdf_gpu_umd_device_t device_ = {};
  // Device-local allocation request used by the lifecycle witness.
  amdf_memory_native_create_info_t create_info_ = {};
  // Device-local profile passed through the trusted UMD boundary.
  amdf_memory_native_profile_t profile_ = {};
};

TEST_F(WindowsGpuMemoryTest,
       DestroyRetriesAcceptedUnmapWithoutSubmittingItAgain) {
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(result.device_address, UINT64_C(0x100000));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryLayout, Operation::kReserveAddress,
                Operation::kCreateAllocation, Operation::kMap, Operation::kWait,
                Operation::kMakeResident, Operation::kWait}));

  state_.failing_wait_target = 3;
  state_.wait_failures_remaining = 1;
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(state_.unmap_count, 1u);
  ASSERT_GE(state_.operations.size(), 2u);
  EXPECT_EQ(state_.operations[state_.operations.size() - 2], Operation::kUnmap);
  EXPECT_EQ(state_.operations.back(), Operation::kWait);

  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.unmap_count, 1u);
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1, 2, 3, 3}));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryLayout, Operation::kReserveAddress,
                Operation::kCreateAllocation, Operation::kMap, Operation::kWait,
                Operation::kMakeResident, Operation::kWait, Operation::kUnmap,
                Operation::kWait, Operation::kWait, Operation::kEvict,
                Operation::kDestroyAllocation, Operation::kFreeAddress}));
}

TEST_F(WindowsGpuMemoryTest, MapsExactReadExecuteAccessWithoutWrite) {
  state_.expected_write = 0;
  state_.expected_execute = 1;
  create_info_.device_access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};

  ASSERT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
}

TEST_F(WindowsGpuMemoryTest,
       MalformedNativeAllocationKeepsBackingOnFailedFree) {
  state_.allocation_domain = AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
  state_.allocation_handle = 0;
  state_.resource_handle = 0x21;
  state_.destroy_status = kStatusNoMemory;
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0, &profile_),
            AMDF_STATUS_OK);
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_memory_result_t original_result = result;
  EXPECT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            amdf_kmt_make_status(STATUS_INVALID_HANDLE));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(state_.metadata_free_count, 0u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kQueryLayout,
                                    Operation::kReserveAddress,
                                    Operation::kCreateAllocation}));
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(state_.metadata_free_count, 0u);
  amdf_gpu_umd_memory_abandon(memory);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryLayout, Operation::kReserveAddress,
                Operation::kCreateAllocation, Operation::kDestroyAllocation}));
  MEMORY_BASIC_INFORMATION information = {};
  ASSERT_NE(
      VirtualQuery(state_.host_pointer, &information, sizeof(information)), 0u);
  EXPECT_EQ(information.State, MEM_COMMIT);
  // Only fake native handles remain; reclaim the real test backing directly.
  EXPECT_TRUE(VirtualFree(information.AllocationBase, 0, MEM_RELEASE));
}

TEST_F(WindowsGpuMemoryTest, MalformedUngroupedAllocationReleasesValidHandles) {
  state_.allocation_handle = 0;
  state_.allocation_count = 2;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_memory_result_t original_result = result;
  EXPECT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            amdf_kmt_make_status(STATUS_INVALID_HANDLE));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(state_.metadata_free_count, 0u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kQueryLayout,
                                    Operation::kReserveAddress,
                                    Operation::kCreateAllocation}));
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryLayout, Operation::kReserveAddress,
                Operation::kCreateAllocation, Operation::kDestroyAllocation,
                Operation::kFreeAddress}));
}

TEST_F(WindowsGpuMemoryTest, UnexpectedMappingRemainsWithConstructingOwner) {
  state_.mapped_device_address = UINT64_C(0x200000);
  state_.failing_wait_target = 1;
  state_.wait_failures_remaining = 1;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_memory_result_t original_result = result;
  EXPECT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryLayout, Operation::kReserveAddress,
                Operation::kCreateAllocation, Operation::kMap}));
  EXPECT_EQ(state_.metadata_free_count, 0u);
  const amdf_status_t release_status = amdf_gpu_umd_memory_destroy(memory);
  EXPECT_EQ(release_status, amdf_kmt_make_status(kStatusNoMemory));
  if (!amdf_status_is_ok(release_status)) {
    amdf_gpu_umd_memory_abandon(memory);
  }
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_EQ(
      state_.operations,
      (std::vector<Operation>{
          Operation::kQueryLayout, Operation::kReserveAddress,
          Operation::kCreateAllocation, Operation::kMap, Operation::kWait}));
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1}));
  EXPECT_EQ(state_.unmap_count, 0u);
}

TEST_F(WindowsGpuMemoryTest, ProfileUsesCapturedGpuMmuCapabilities) {
  EXPECT_EQ(profile_.device_address.address_bit_count, 48u);
  EXPECT_EQ(profile_.device_address.minimum_address, 0u);
  EXPECT_EQ(profile_.device_address.maximum_address, (UINT64_C(1) << 48) - 1);
  EXPECT_EQ(profile_.allocation.maximum_byte_length, UINT64_C(1) << 47);
  EXPECT_EQ(profile_.allocation.maximum_alignment, UINT64_C(1) << 47);
  EXPECT_EQ(profile_.supported_flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
  amdf_memory_native_profile_t system_profile = {};
  ASSERT_EQ(
      amdf_gpu_umd_device_query_memory_profile(&device_, 0, &system_profile),
      AMDF_STATUS_OK);
  EXPECT_NE(system_profile.supported_flags & AMDF_MEMORY_FLAG_HOST_COHERENT,
            0u);

  device_.memory_capabilities.read_only_memory_supported = 0;
  device_.memory_capabilities.no_execute_memory_supported = 0;
  device_.memory_capabilities.cache_coherent_memory_supported = 0;
  amdf_memory_native_profile_t restricted_profile = {};
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0,
                                                     &restricted_profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(restricted_profile.guaranteed_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                AMDF_MEMORY_ACCESS_EXECUTE);
  EXPECT_EQ(restricted_profile.supported_flags & AMDF_MEMORY_FLAG_HOST_COHERENT,
            0u);
}

TEST_F(WindowsGpuMemoryTest, MissingMemoryApiDoesNotPublishProfile) {
  kmt_.reserve_gpu_virtual_address = nullptr;
  amdf_memory_native_profile_t output = {};
  output.ordinal = 73;

  EXPECT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0, &output),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  EXPECT_EQ(output.ordinal, 73u);
}

}  // namespace
