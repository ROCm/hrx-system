// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/memory_reservation.h"
#include "libamdf/src/gpu/umd/wddm/device.h"

namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103u);
constexpr NTSTATUS kStatusNoMemory = static_cast<NTSTATUS>(0xC0000017u);

enum class Operation {
  kImport,
  kQueryLayout,
  kReserveAddress,
  kCreateAllocation,
  kMap,
  kWait,
  kMakeResident,
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
  // OS reclamation required before returning independently owned host backing.
  uint32_t expected_synchronous_destroy = 0;
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
  // Number of native mapping requests issued for allocation chunks.
  uint32_t map_request_count = 0;
  // Reserved device VA extent, including any caller-required alignment slack.
  uint64_t reservation_byte_length = 65536;
  // Mapping request rejected before acceptance, or UINT32_MAX for none.
  uint32_t failing_map_ordinal = UINT32_MAX;
  // Next fence value assigned to an accepted paging operation.
  uint64_t next_paging_fence = 1;
  // Fence whose CPU wait should fail, or zero for none.
  uint64_t failing_wait_target = 0;
  // Number of matching CPU waits rejected before observation succeeds.
  uint32_t wait_failures_remaining = 0;
  // Monitored paging progress exposed to production code.
  volatile uint64_t paging_fence = 0;
  // Explicit successful-wait result, or zero to publish the requested point.
  uint64_t progress_on_wait_success = 0;
  // Native result injected after shared-resource acquisition.
  amdf_status_t import_status = AMDF_STATUS_OK;
  // Independent NT reference captured by the native import dependency.
  HANDLE imported_handle = nullptr;
  // Native and bridge operations in call order.
  std::vector<Operation> operations;
  // Paging fence targets observed by CPU waits.
  std::vector<uint64_t> wait_targets;
  // Native cache ranges observed when publishing noncoherent host writes.
  std::vector<D3DKMT_INVALIDATECACHE> invalidations;
  // Native range updates observed while exercising reservation replacement.
  std::vector<D3DDDI_MAPGPUVIRTUALADDRESS> mappings;
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
    out_allocation_handles[i] = 0x20 + i;
  }
  *out_resource_handle = state->resource_handle;
  *out_allocation_count = state->allocation_count;
  *out_native_status = 0;
  return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
}

amdf_status_t AMDF_WKMI_BRIDGE_CALL FakePrepareBufferImport(
    amdf_wkmi_bridge_gpu_adapter_t* adapter,
    const amdf_wkmi_bridge_gpu_buffer_import_info_t* info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length, uint64_t* out_buffer_byte_length) {
  auto* state = reinterpret_cast<FakeMemoryState*>(adapter);
  state->operations.push_back(Operation::kImport);
  EXPECT_EQ(info->device_handle, 0x10u);
  EXPECT_EQ(*resource_handle, 0u);
  EXPECT_EQ(*allocation_handle, 0u);
  state->imported_handle = info->shared_handle;
  *resource_handle = state->resource_handle;
  *allocation_handle = 0x20;
  if (!amdf_status_is_ok(state->import_status)) {
    return state->import_status;
  }
  *out_native_byte_length = 65536;
  *out_buffer_byte_length = 16384;
  return AMDF_STATUS_OK;
}

NTSTATUS APIENTRY FakeUnexpectedCreateAllocation(D3DKMT_CREATEALLOCATION*) {
  ADD_FAILURE() << "WKMI owns physical allocation construction";
  return kStatusNoMemory;
}

NTSTATUS APIENTRY
FakeReserveGpuVirtualAddress(D3DDDI_RESERVEGPUVIRTUALADDRESS* reserve) {
  current_state->operations.push_back(Operation::kReserveAddress);
  EXPECT_EQ(reserve->hAdapter, 0x08u);
  EXPECT_EQ(reserve->Size, current_state->reservation_byte_length);
  reserve->VirtualAddress = 0x100000;
  return 0;
}

NTSTATUS APIENTRY FakeMapGpuVirtualAddress(D3DDDI_MAPGPUVIRTUALADDRESS* map) {
  current_state->operations.push_back(Operation::kMap);
  const uint32_t ordinal = current_state->map_request_count++;
  const uint64_t chunk_byte_length = 65536 / current_state->allocation_count;
  const uint64_t byte_offset = ordinal * chunk_byte_length;
  EXPECT_EQ(map->Protection.NoAccess, 0u);
  EXPECT_EQ(map->BaseAddress, UINT64_C(0x100000) + byte_offset);
  EXPECT_EQ(map->hAllocation, 0x20u + ordinal);
  EXPECT_EQ(map->Protection.Write, current_state->expected_write);
  EXPECT_EQ(map->Protection.Execute, current_state->expected_execute);
  EXPECT_EQ(map->hPagingQueue, 0x30u);
  EXPECT_EQ(map->SizeInPages, chunk_byte_length / 4096);
  if (ordinal == current_state->failing_map_ordinal) {
    return kStatusNoMemory;
  }
  map->VirtualAddress = current_state->mapped_device_address + byte_offset;
  map->PagingFenceValue = current_state->next_paging_fence++;
  return kStatusPending;
}

NTSTATUS APIENTRY FakeMakeResident(D3DDDI_MAKERESIDENT* resident) {
  current_state->operations.push_back(Operation::kMakeResident);
  EXPECT_EQ(resident->hPagingQueue, 0x30u);
  EXPECT_EQ(resident->NumAllocations, current_state->allocation_count);
  for (uint32_t i = 0; i < resident->NumAllocations; ++i) {
    EXPECT_EQ(resident->AllocationList[i], 0x20u + i);
  }
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
  current_state->paging_fence = current_state->progress_on_wait_success != 0
                                    ? current_state->progress_on_wait_success
                                    : target;
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
    const uint32_t first_valid = current_state->allocation_handle == 0 ? 1 : 0;
    EXPECT_EQ(destroy->AllocationCount,
              current_state->allocation_count - first_valid);
    for (uint32_t i = 0; i < destroy->AllocationCount; ++i) {
      EXPECT_EQ(destroy->phAllocationList[i], 0x20u + first_valid + i);
    }
  }
  EXPECT_EQ(destroy->Flags.AssumeNotInUse, 1u);
  EXPECT_EQ(destroy->Flags.SynchronousDestroy,
            current_state->expected_synchronous_destroy);
  return current_state->destroy_status;
}

NTSTATUS APIENTRY
FakeFreeGpuVirtualAddress(const D3DKMT_FREEGPUVIRTUALADDRESS* free_address) {
  current_state->operations.push_back(Operation::kFreeAddress);
  EXPECT_EQ(free_address->hAdapter, 0x08u);
  EXPECT_EQ(free_address->BaseAddress, UINT64_C(0x100000));
  EXPECT_EQ(free_address->Size, current_state->reservation_byte_length);
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
    bridge_.gpu_buffer_prepare_import = FakePrepareBufferImport;
    kmt_.create_allocation = FakeUnexpectedCreateAllocation;
    kmt_.destroy_allocation = FakeDestroyAllocation;
    kmt_.reserve_gpu_virtual_address = FakeReserveGpuVirtualAddress;
    kmt_.free_gpu_virtual_address = FakeFreeGpuVirtualAddress;
    kmt_.map_gpu_virtual_address = FakeMapGpuVirtualAddress;
    kmt_.make_resident = FakeMakeResident;
    kmt_.invalidate_cache = FakeUnexpectedInvalidateCache;
    kmt_.wait_from_cpu = FakeWaitFromCpu;
    kmt_.get_device_state = [](D3DKMT_GETDEVICESTATE* query) -> NTSTATUS {
      query->ExecutionState = D3DKMT_DEVICEEXECUTION_ACTIVE;
      return 0;
    };
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

TEST_F(WindowsGpuMemoryTest, DestroyReclaimsMappingAndResidencyDirectly) {
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(result.device_address, UINT64_C(0x100000));
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryLayout, Operation::kReserveAddress,
                Operation::kCreateAllocation, Operation::kMap, Operation::kWait,
                Operation::kMakeResident, Operation::kWait,
                Operation::kDestroyAllocation, Operation::kFreeAddress}));
}

TEST_F(WindowsGpuMemoryTest, OwnsNativeAndOveralignedHostStorage) {
  for (uint64_t alignment : {UINT64_C(65536), UINT64_C(131072)}) {
    SCOPED_TRACE(alignment);
    state_ = {};
    state_.allocation_domain = AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
    state_.expected_synchronous_destroy = 1;
    state_.reservation_byte_length = alignment;
    create_info_.minimum_alignment = alignment;
    create_info_.required_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0, &profile_),
              AMDF_STATUS_OK);
    amdf_gpu_umd_memory_t* memory = nullptr;
    amdf_gpu_umd_memory_result_t result = {};
    ASSERT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                          &create_info_, &memory, &result),
              AMDF_STATUS_OK);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(state_.host_pointer) % alignment, 0u);
    EXPECT_EQ(result.alignment, alignment);
    auto* bytes = static_cast<uint8_t*>(state_.host_pointer);
    bytes[0] = 0xA5;
    bytes[65535] = 0x5A;
    MEMORY_BASIC_INFORMATION information = {};
    ASSERT_NE(VirtualQuery(bytes, &information, sizeof(information)), 0u);
    EXPECT_EQ(information.State, MEM_COMMIT);
    EXPECT_GE(information.RegionSize, 65536u);
    void* reservation = information.AllocationBase;
    EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
    ASSERT_NE(VirtualQuery(reservation, &information, sizeof(information)), 0u);
    EXPECT_EQ(information.State, MEM_FREE);
    EXPECT_EQ(state_.metadata_free_count, 1u);
  }
}

TEST_F(WindowsGpuMemoryTest, BorrowsHostViewsAcrossNativeAllocationChunks) {
  state_.allocation_domain = AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
  state_.allocation_count = 2;
  state_.expected_synchronous_destroy = 1;
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0, &profile_),
            AMDF_STATUS_OK);
  kmt_.invalidate_cache = [](const D3DKMT_INVALIDATECACHE* invalidate) {
    current_state->invalidations.push_back(*invalidate);
    return static_cast<NTSTATUS>(0);
  };
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  device_.host_allocator.allocate = [](void*, uint64_t, uint64_t) -> void* {
    ADD_FAILURE() << "a persistent native host view requires no allocation";
    return nullptr;
  };
  const size_t native_operation_count = state_.operations.size();
  amdf_memory_map_info_t request = {};
  request.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  request.byte_length = 128;
  const uint64_t offsets[] = {128, 32736};
  amdf_gpu_umd_host_mapping_t* mappings[2] = {};
  amdf_gpu_umd_host_mapping_result_t views[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    request.byte_offset = offsets[i];
    ASSERT_EQ(amdf_gpu_umd_memory_map(memory, &profile_.host_mapping, &request,
                                      &mappings[i], &views[i]),
              AMDF_STATUS_OK);
    EXPECT_EQ(views[i].pointer,
              static_cast<uint8_t*>(state_.host_pointer) + offsets[i]);
    EXPECT_EQ(views[i].byte_length, request.byte_length);
    std::memset(views[i].pointer, 0xA0 + i, request.byte_length);
  }
  amdf_gpu_umd_host_mapping_destroy(mappings[0]);
  // The second view straddles two native allocations, neither of which is
  // owned by the view. Cache publication still uses memory-relative offsets.
  EXPECT_EQ(amdf_gpu_umd_host_mapping_cache_control(
                mappings[1], AMDF_HOST_CACHE_OPERATION_FLUSH, offsets[1], 128),
            AMDF_STATUS_OK);
  ASSERT_EQ(state_.invalidations.size(), 2u);
  EXPECT_EQ(state_.invalidations[0].hDevice, device_.device);
  EXPECT_EQ(state_.invalidations[0].hAllocation, 0x20u);
  EXPECT_EQ(state_.invalidations[0].Offset, offsets[1]);
  EXPECT_EQ(state_.invalidations[0].Length, 32u);
  EXPECT_EQ(state_.invalidations[1].hDevice, device_.device);
  EXPECT_EQ(state_.invalidations[1].hAllocation, 0x21u);
  EXPECT_EQ(state_.invalidations[1].Offset, 0u);
  EXPECT_EQ(state_.invalidations[1].Length, 96u);
  EXPECT_EQ(static_cast<uint8_t*>(views[1].pointer)[127], 0xA1);
  amdf_gpu_umd_host_mapping_destroy(mappings[1]);
  EXPECT_EQ(static_cast<uint8_t*>(state_.host_pointer)[128], 0xA0);
  EXPECT_EQ(static_cast<uint8_t*>(state_.host_pointer)[32736], 0xA1);
  EXPECT_EQ(state_.operations.size(), native_operation_count);
  EXPECT_EQ(state_.metadata_free_count, 0u);
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.metadata_free_count, 1u);
}

TEST_F(WindowsGpuMemoryTest,
       ReservationMapsRangesAcrossPreparedAllocationChunks) {
  state_.allocation_count = 2;
  create_info_.required_flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  kmt_.reserve_gpu_virtual_address =
      [](D3DDDI_RESERVEGPUVIRTUALADDRESS* reserve) {
        reserve->VirtualAddress = 0x200000;
        return static_cast<NTSTATUS>(0);
      };
  kmt_.map_gpu_virtual_address = [](D3DDDI_MAPGPUVIRTUALADDRESS* map) {
    current_state->operations.push_back(Operation::kMap);
    current_state->mappings.push_back(*map);
    map->VirtualAddress = map->BaseAddress;
    map->PagingFenceValue = current_state->next_paging_fence++;
    return kStatusPending;
  };
  kmt_.free_gpu_virtual_address =
      [](const D3DKMT_FREEGPUVIRTUALADDRESS* release) {
        current_state->operations.push_back(Operation::kFreeAddress);
        EXPECT_EQ(release->BaseAddress, UINT64_C(0x200000));
        EXPECT_EQ(release->Size, 65536u);
        EXPECT_EQ(current_state->paging_fence,
                  current_state->next_paging_fence - 1);
        return static_cast<NTSTATUS>(0);
      };
  amdf_gpu_umd_device_t* devices[] = {&device_};
  amdf_gpu_umd_memory_reservation_t* reservation = nullptr;
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_create(
                device_.host_allocator, 1, devices, 65536, 0, &reservation),
            AMDF_STATUS_OK);
  state_.operations.clear();
  const amdf_gpu_umd_mapping_update_t update = {
      .target_byte_offset = 8192,
      .byte_length = 8192,
      .memory = memory,
      .source_byte_offset = 28672,
      .access = AMDF_MEMORY_ACCESS_READ,
  };
  bool mutated = false;
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_update(reservation, 0, 1, &update,
                                                   &mutated),
            AMDF_STATUS_OK);
  ASSERT_TRUE(mutated);
  ASSERT_EQ(state_.mappings.size(), 2u);
  EXPECT_EQ(state_.mappings[0].hAllocation, 0x20u);
  EXPECT_EQ(state_.mappings[0].BaseAddress, UINT64_C(0x202000));
  EXPECT_EQ(state_.mappings[0].OffsetInPages, 7u);
  EXPECT_EQ(state_.mappings[0].SizeInPages, 1u);
  EXPECT_EQ(state_.mappings[1].hAllocation, 0x21u);
  EXPECT_EQ(state_.mappings[1].BaseAddress, UINT64_C(0x203000));
  EXPECT_EQ(state_.mappings[1].OffsetInPages, 0u);
  EXPECT_EQ(state_.mappings[1].SizeInPages, 1u);
  EXPECT_EQ(state_.mappings[0].Protection.Write, 0u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kMap, Operation::kMap,
                                    Operation::kWait}));
  // The queued form preserves the same chunk boundaries and permissions in a
  // single vector. Only native dependencies are substituted, not translation.
  bridge_.gpu_kernel_queue_create =
      [](amdf_wkmi_bridge_gpu_adapter_t*,
         const amdf_wkmi_bridge_gpu_kernel_queue_create_info_t*,
         amdf_wkmi_bridge_gpu_kernel_queue_t** queue,
         amdf_wkmi_bridge_gpu_kernel_queue_info_t* info,
         uint32_t*) -> amdf_wkmi_bridge_result_t {
    *queue =
        reinterpret_cast<amdf_wkmi_bridge_gpu_kernel_queue_t*>(current_state);
    *info = {};
    info->progress_fence_pointer = &current_state->paging_fence;
    info->progress_fence_handle = 0x50;
    info->context_handle = 0x60;
    info->queue_handle = 0x70;
    return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
  };
  bridge_.gpu_kernel_queue_destroy =
      [](amdf_wkmi_bridge_gpu_kernel_queue_t*,
         uint32_t*) -> amdf_wkmi_bridge_result_t {
    return AMDF_WKMI_BRIDGE_RESULT_SUCCESS;
  };
  kmt_.update_gpu_virtual_address =
      [](const D3DKMT_UPDATEGPUVIRTUALADDRESS* native) {
        EXPECT_EQ(native->NumOperations, 2u);
        EXPECT_EQ(native->FenceValue, 9u);
        for (uint32_t i = 0; i < native->NumOperations; ++i) {
          const auto& operation = native->Operations[i];
          EXPECT_EQ(operation.OperationType,
                    D3DDDI_UPDATEGPUVIRTUALADDRESS_MAP_PROTECT);
          EXPECT_EQ(operation.MapProtect.hAllocation, 0x20u + i);
          EXPECT_EQ(operation.MapProtect.BaseAddress,
                    UINT64_C(0x202000) + i * 4096);
          EXPECT_EQ(operation.MapProtect.SizeInBytes, 4096u);
          EXPECT_EQ(operation.MapProtect.AllocationOffsetInBytes,
                    i == 0 ? 28672u : 0u);
          EXPECT_EQ(operation.MapProtect.Protection.Write, 0u);
          EXPECT_EQ(operation.MapProtect.Protection.Execute, 0u);
        }
        return static_cast<NTSTATUS>(0);
      };
  kmt_.wait_on_hardware_queue =
      [](const D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE* wait) {
        EXPECT_EQ(wait->hHwQueue, 0x70u);
        EXPECT_EQ(wait->FenceValueArray[0], 10u);
        return static_cast<NTSTATUS>(0);
      };
  amdf_gpu_umd_kernel_queue_t* queue = nullptr;
  ASSERT_EQ(amdf_gpu_umd_kernel_queue_create(
                &device_, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, &queue),
            AMDF_STATUS_OK);
  bool accepted = false;
  mutated = false;
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_update_on_queue(
                reservation, queue, 0, 1, &update, 10, &mutated, &accepted),
            AMDF_STATUS_OK);
  EXPECT_TRUE(mutated);
  EXPECT_TRUE(accepted);
  ASSERT_EQ(amdf_gpu_umd_kernel_queue_destroy(queue), AMDF_STATUS_OK);
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
  ASSERT_EQ(state_.mappings.size(), 3u);
  EXPECT_EQ(state_.mappings.back().Protection.NoAccess, 1u);
  EXPECT_EQ(state_.mappings.back().hAllocation, 0u);
  EXPECT_EQ(state_.mappings.back().BaseAddress, UINT64_C(0x200000));
  EXPECT_EQ(state_.mappings.back().SizeInPages, 16u);
  EXPECT_EQ(state_.operations.back(), Operation::kFreeAddress);
  kmt_.free_gpu_virtual_address = FakeFreeGpuVirtualAddress;
  ASSERT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
}

TEST_F(WindowsGpuMemoryTest,
       DetachThroughAnotherAliasFollowsPartialUpdatesOnTheirPagingQueue) {
  auto second = device_;
  second.device = 0x11;
  second.paging_queue = 0x31;
  amdf_gpu_umd_device_t* devices[] = {&device_, &second};
  amdf_gpu_umd_memory_reservation_t* reservation = nullptr;
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_create(
                device_.host_allocator, 2, devices, 65536, 0, &reservation),
            AMDF_STATUS_OK);
  kmt_.map_gpu_virtual_address = [](D3DDDI_MAPGPUVIRTUALADDRESS* map) {
    current_state->operations.push_back(Operation::kMap);
    EXPECT_EQ(map->hPagingQueue, 0x31u);
    current_state->mappings.push_back(*map);
    if (current_state->mappings.size() == 2) {
      return kStatusNoMemory;
    }
    map->VirtualAddress = map->BaseAddress;
    map->PagingFenceValue = current_state->next_paging_fence++;
    return kStatusPending;
  };
  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) {
        current_state->operations.push_back(Operation::kWait);
        EXPECT_EQ(wait->hDevice, 0x11u);
        EXPECT_EQ(wait->FenceValueArray[0],
                  current_state->next_paging_fence - 1);
        current_state->paging_fence = wait->FenceValueArray[0];
        return static_cast<NTSTATUS>(0);
      };
  const amdf_gpu_umd_mapping_update_t updates[] = {
      {.target_byte_offset = 0, .byte_length = 4096},
      {.target_byte_offset = 8192, .byte_length = 4096},
  };
  bool mutated = false;
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_update(reservation, 1, 2, updates,
                                                   &mutated),
            amdf_kmt_make_status(kStatusNoMemory));
  ASSERT_TRUE(mutated);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kFreeAddress),
            0);
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_detach(reservation, 0),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.mappings.back().Protection.NoAccess, 1u);
  EXPECT_EQ(state_.mappings.back().SizeInPages, 16u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kReserveAddress, Operation::kMap,
                                    Operation::kMap, Operation::kMap,
                                    Operation::kWait}));
  EXPECT_EQ(state_.metadata_free_count, 0u);
  uint64_t address = 0;
  uint32_t mapping_ordinal = UINT32_MAX;
  uint64_t granularity = 0;
  amdf_memory_access_t required_access = 0;
  amdf_gpu_umd_memory_reservation_query_access(reservation, 1, &address,
                                               &mapping_ordinal, &granularity,
                                               &required_access);
  EXPECT_EQ(address, 0x100000u);
  EXPECT_EQ(mapping_ordinal, 0u);
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kReserveAddress, Operation::kMap, Operation::kMap,
                Operation::kMap, Operation::kWait, Operation::kMap,
                Operation::kWait, Operation::kFreeAddress}));
  EXPECT_EQ(state_.metadata_free_count, 1u);
}

TEST_F(WindowsGpuMemoryTest,
       FailedFinalClearConsumesOwnerWithoutReleasingAddress) {
  amdf_gpu_umd_device_t* devices[] = {&device_};
  amdf_gpu_umd_memory_reservation_t* reservation = nullptr;
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_create(
                device_.host_allocator, 1, devices, 65536, 0, &reservation),
            AMDF_STATUS_OK);
  kmt_.map_gpu_virtual_address = [](D3DDDI_MAPGPUVIRTUALADDRESS*) {
    current_state->operations.push_back(Operation::kMap);
    return kStatusNoMemory;
  };
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_detach(reservation, 0),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(state_.metadata_free_count, 0u);
  EXPECT_EQ(
      state_.operations,
      (std::vector<Operation>{Operation::kReserveAddress, Operation::kMap}));
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kReserveAddress, Operation::kMap,
                                    Operation::kMap}));
  EXPECT_EQ(state_.metadata_free_count, 1u);
}

TEST_F(WindowsGpuMemoryTest, FailedDetachReadinessPreservesOwnerAndAddress) {
  amdf_gpu_umd_device_t* devices[] = {&device_};
  amdf_gpu_umd_memory_reservation_t* reservation = nullptr;
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_create(
                device_.host_allocator, 1, devices, 65536, 0, &reservation),
            AMDF_STATUS_OK);
  kmt_.map_gpu_virtual_address = [](D3DDDI_MAPGPUVIRTUALADDRESS* map) {
    current_state->operations.push_back(Operation::kMap);
    EXPECT_EQ(map->Protection.NoAccess, 1u);
    EXPECT_EQ(map->hAllocation, 0u);
    EXPECT_EQ(map->BaseAddress, 0x100000u);
    EXPECT_EQ(map->SizeInPages, 16u);
    map->VirtualAddress = map->BaseAddress;
    map->PagingFenceValue = current_state->next_paging_fence++;
    return kStatusPending;
  };
  kmt_.wait_from_cpu = [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU*) {
    current_state->operations.push_back(Operation::kWait);
    return kStatusNoMemory;
  };
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_detach(reservation, 0),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(state_.metadata_free_count, 0u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kReserveAddress, Operation::kMap,
                                    Operation::kWait}));
  // Final destruction still consumes metadata but cannot release unproven VA.
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kFreeAddress),
            0);
  EXPECT_EQ(state_.metadata_free_count, 1u);
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

class WindowsGpuResetFenceTest : public WindowsGpuMemoryTest,
                                 public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(DuringWait, WindowsGpuResetFenceTest,
                         ::testing::Bool());

TEST_P(WindowsGpuResetFenceTest,
       PreventsMemoryPublicationBeforeIndependentNativeReclamation) {
  if (GetParam()) {
    state_.progress_on_wait_success = UINT64_MAX;
  } else {
    state_.paging_fence = UINT64_MAX;
  }
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_memory_result_t original = result;
  const auto lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            lost);
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(state_.wait_targets.size(), GetParam() ? 1u : 0u);
  ASSERT_NE(memory, nullptr);
  // The reset sentinel did not complete the map. Native allocation destruction
  // independently reclaims the failed construction's mapping and residency;
  // cleanup needs neither another paging wait nor a manufactured fence value.
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_EQ(state_.wait_targets.size(), GetParam() ? 1u : 0u);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kDestroyAllocation),
            1);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kFreeAddress),
            1);
}

TEST_F(WindowsGpuMemoryTest,
       MalformedNativeAllocationKeepsBackingOnFailedFree) {
  state_.allocation_domain = AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
  state_.expected_synchronous_destroy = 1;
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
  EXPECT_EQ(amdf_gpu_umd_memory_remaining_backing_byte_length(memory), 65536u);
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

TEST_F(WindowsGpuMemoryTest,
       RegisteredPagesRemainBorrowedAcrossReleaseAndRollback) {
  auto* pages = static_cast<uint8_t*>(
      VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  ASSERT_NE(pages, nullptr);
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 2, &profile_),
            AMDF_STATUS_OK);
  create_info_.registered_host_pointer = pages;
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  for (uint64_t failing_wait_target : {0u, 1u, 2u}) {
    SCOPED_TRACE(failing_wait_target);
    state_ = {};
    state_.allocation_domain =
        AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST;
    state_.expected_synchronous_destroy = 1;
    state_.failing_wait_target = failing_wait_target;
    state_.wait_failures_remaining = 1;
    std::memset(pages, 0xA5, 65536);
    amdf_gpu_umd_memory_t* memory = nullptr;
    amdf_gpu_umd_memory_result_t result;
    std::memset(&result, 0x5A, sizeof(result));
    const auto original_result = result;
    const auto status = amdf_gpu_umd_memory_prepare(
        &device_, 0, nullptr, &profile_, &create_info_, &memory, &result);
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(state_.host_pointer, pages);
    if (failing_wait_target == 0) {
      EXPECT_EQ(status, AMDF_STATUS_OK);
    } else {
      EXPECT_EQ(status, amdf_kmt_make_status(kStatusNoMemory));
      EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    }
    ASSERT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
    EXPECT_EQ(state_.metadata_free_count, 1u);
    EXPECT_EQ(state_.operations.back(), Operation::kFreeAddress);
    MEMORY_BASIC_INFORMATION information = {};
    ASSERT_NE(VirtualQuery(pages, &information, sizeof(information)), 0u);
    ASSERT_EQ(information.State, MEM_COMMIT);
    EXPECT_EQ(pages[0], 0xA5);
    EXPECT_EQ(pages[65535], 0xA5);
    std::memset(pages, 0x3C, 65536);
  }
  EXPECT_TRUE(VirtualFree(pages, 0, MEM_RELEASE));
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

TEST_F(WindowsGpuMemoryTest, EmptyGpuVaFailureDoesNotClaimLostPayload) {
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  kmt_.free_gpu_virtual_address =
      [](const D3DKMT_FREEGPUVIRTUALADDRESS* free_address) -> NTSTATUS {
    EXPECT_EQ(FakeFreeGpuVirtualAddress(free_address), 0);
    return kStatusNoMemory;
  };
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(state_.operations.back(), Operation::kFreeAddress);
  EXPECT_EQ(amdf_gpu_umd_memory_remaining_backing_byte_length(memory), 0u);
  amdf_gpu_umd_memory_abandon(memory);
  EXPECT_EQ(state_.metadata_free_count, 1u);
}

TEST_F(WindowsGpuMemoryTest, FailedGpuVaReleaseRetainsOwnedHostBacking) {
  state_.allocation_domain = AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
  state_.expected_synchronous_destroy = 1;
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 0, &profile_),
            AMDF_STATUS_OK);
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  kmt_.free_gpu_virtual_address =
      [](const D3DKMT_FREEGPUVIRTUALADDRESS* free_address) -> NTSTATUS {
    EXPECT_EQ(FakeFreeGpuVirtualAddress(free_address), 0);
    return kStatusNoMemory;
  };
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(state_.operations.back(), Operation::kFreeAddress);
  EXPECT_EQ(amdf_gpu_umd_memory_remaining_backing_byte_length(memory), 65536u);
  amdf_gpu_umd_memory_abandon(memory);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  // The fake driver no longer retains this real host allocation.
  MEMORY_BASIC_INFORMATION information = {};
  ASSERT_NE(
      VirtualQuery(state_.host_pointer, &information, sizeof(information)), 0u);
  EXPECT_EQ(information.State, MEM_COMMIT);
  EXPECT_TRUE(VirtualFree(information.AllocationBase, 0, MEM_RELEASE));
}

TEST_F(WindowsGpuMemoryTest, DestroyReclaimsUnexpectedMapping) {
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
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryLayout, Operation::kReserveAddress,
                Operation::kCreateAllocation, Operation::kMap,
                Operation::kDestroyAllocation, Operation::kFreeAddress}));
  EXPECT_TRUE(state_.wait_targets.empty());
}

TEST_F(WindowsGpuMemoryTest, DestroyReclaimsEveryChunkAfterPartialMapping) {
  state_.allocation_count = 2;
  state_.failing_map_ordinal = 1;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original_result = result;
  EXPECT_EQ(amdf_gpu_umd_memory_prepare(&device_, 0, nullptr, &profile_,
                                        &create_info_, &memory, &result),
            amdf_kmt_make_status(kStatusNoMemory));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kQueryLayout, Operation::kReserveAddress,
                Operation::kCreateAllocation, Operation::kMap, Operation::kMap,
                Operation::kDestroyAllocation, Operation::kFreeAddress}));
  EXPECT_TRUE(state_.wait_targets.empty());
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
  ASSERT_NE(system_profile.visibility.describe_host, nullptr);
  const auto noncoherent = system_profile.visibility.describe_host(
      system_profile.visibility.data, AMDF_EXTERNAL_MEMORY_TYPE_NONE, 0);
  EXPECT_EQ(noncoherent.flush.executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API);
  const auto coherent = system_profile.visibility.describe_host(
      system_profile.visibility.data, AMDF_EXTERNAL_MEMORY_TYPE_NONE,
      AMDF_MEMORY_FLAG_HOST_COHERENT);
  EXPECT_EQ(coherent.flush.executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
  EXPECT_TRUE(state_.operations.empty());

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

TEST_F(WindowsGpuMemoryTest, ImportedRangeAndIndependentExportOwnership) {
  state_.resource_handle = 0x21;
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 3, &profile_),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile_.roles,
            AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_EXPORT);
  EXPECT_EQ(profile_.external_memory_support_count, 1u);
  EXPECT_EQ(profile_.external_memory_support[0].type,
            AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE);
  EXPECT_EQ(profile_.external_memory_support[0].source_offset_alignment, 1u);
  const amdf_memory_native_import_info_t info = {
      .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .required_flags = AMDF_MEMORY_FLAG_SHAREABLE,
  };
  HANDLE source_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  ASSERT_NE(source_handle, nullptr);
  amdf_external_memory_t source = {};
  source.type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE;
  source.payload.native_handle = source_handle;
  source.source_byte_offset = 13;
  source.byte_length = 117;
  const amdf_external_memory_t original = source;
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_umd_memory_prepare_import(&device_, &profile_, &info,
                                               &source, &memory, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(&source, &original, sizeof(source)), 0);
  EXPECT_NE(state_.imported_handle, source_handle);
  EXPECT_EQ(result.device_address, UINT64_C(0x10000d));
  EXPECT_EQ(result.source_byte_offset, 13u);
  EXPECT_EQ(result.byte_length, 117u);
  EXPECT_EQ(result.native_allocation_byte_length, 65536u);
  EXPECT_EQ(result.alignment, 1u);
  EXPECT_FALSE(amdf_physical_memory_id_is_valid(&result.physical_backing_id));
  EXPECT_TRUE(CloseHandle(source_handle));
  amdf_memory_export_info_t export_info = {};
  export_info.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE;
  amdf_external_memory_t exported = {};
  ASSERT_EQ(amdf_gpu_umd_memory_export(memory, &export_info, &exported),
            AMDF_STATUS_OK);
  EXPECT_NE(exported.payload.native_handle, state_.imported_handle);
  ASSERT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  DWORD flags = 0;
  EXPECT_FALSE(GetHandleInformation(state_.imported_handle, &flags));
  EXPECT_EQ(GetLastError(), ERROR_INVALID_HANDLE);
  EXPECT_TRUE(GetHandleInformation(exported.payload.native_handle, &flags));
  exported.release(exported.release_user_data,
                   AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE, exported.payload);
}

class WindowsGpuImportFailureTest
    : public WindowsGpuMemoryTest,
      public ::testing::WithParamInterface<uint32_t> {};

INSTANTIATE_TEST_SUITE_P(NativeProgress, WindowsGpuImportFailureTest,
                         ::testing::Values(0u, 1u, 2u, 3u));

TEST_P(WindowsGpuImportFailureTest, RetainsProgressForOrderedRollback) {
  state_.resource_handle = 0x21;
  ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, 3, &profile_),
            AMDF_STATUS_OK);
  const amdf_memory_native_import_info_t info = {
      .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .required_flags = AMDF_MEMORY_FLAG_SHAREABLE,
  };
  HANDLE source_handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  ASSERT_NE(source_handle, nullptr);
  amdf_external_memory_t source = {};
  source.type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE;
  source.payload.native_handle = source_handle;
  source.byte_length = GetParam() == 1 ? 16385 : 4096;
  if (GetParam() == 0) {
    state_.import_status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  } else if (GetParam() >= 2) {
    state_.failing_wait_target = GetParam() - 1;
    state_.wait_failures_remaining = 1;
  }
  const amdf_external_memory_t original = source;
  amdf_gpu_umd_memory_result_t result;
  std::memset(&result, 0xa5, sizeof(result));
  const auto original_result = result;
  amdf_gpu_umd_memory_t* memory = nullptr;
  const amdf_status_t status = amdf_gpu_umd_memory_prepare_import(
      &device_, &profile_, &info, &source, &memory, &result);
  EXPECT_FALSE(amdf_status_is_ok(status));
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(std::memcmp(&source, &original, sizeof(source)), 0);
  ASSERT_NE(memory, nullptr);
  DWORD flags = 0;
  EXPECT_TRUE(GetHandleInformation(source_handle, &flags));
  EXPECT_TRUE(GetHandleInformation(state_.imported_handle, &flags));
  ASSERT_EQ(amdf_gpu_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_FALSE(GetHandleInformation(state_.imported_handle, &flags));
  EXPECT_EQ(GetLastError(), ERROR_INVALID_HANDLE);
  EXPECT_TRUE(GetHandleInformation(source_handle, &flags));
  EXPECT_TRUE(CloseHandle(source_handle));
  const auto destroy =
      std::find(state_.operations.begin(), state_.operations.end(),
                Operation::kDestroyAllocation);
  ASSERT_NE(destroy, state_.operations.end());
  if (GetParam() <= 1) {
    EXPECT_EQ(state_.operations,
              (std::vector<Operation>{Operation::kImport,
                                      Operation::kDestroyAllocation}));
  } else {
    EXPECT_EQ(state_.operations[state_.operations.size() - 2],
              Operation::kDestroyAllocation);
    EXPECT_EQ(state_.operations.back(), Operation::kFreeAddress);
  }
}

TEST_F(WindowsGpuMemoryTest,
       SystemSignalsUseCoherenceAndCpuVisibleUpdateWidth) {
  for (uint32_t major : {9u, 11u, 12u}) {
    device_.properties.gfx_ip_major = major;
    for (uint32_t mask = 0; mask < 16; ++mask) {
      const bool coherent = (mask & 1u) != 0;
      const bool atomic_update_64 = (mask & 2u) != 0;
      device_.memory_capabilities.cache_coherent_memory_supported = coherent;
      device_.memory_capabilities.atomic_update_64_supported = atomic_update_64;
      device_.properties.is_discrete = (mask & 4u) != 0;
      device_.properties.supports_platform_atomics = (mask & 8u) != 0;
      for (uint32_t ordinal = 0; ordinal < 4; ++ordinal) {
        SCOPED_TRACE(::testing::Message()
                     << "major=" << major << " capability_mask=" << mask
                     << " profile=" << ordinal);
        amdf_memory_native_profile_t profile = {};
        ASSERT_EQ(amdf_gpu_umd_device_query_memory_profile(&device_, ordinal,
                                                           &profile),
                  AMDF_STATUS_OK);
        const amdf_atomic_operations_t operations =
            ordinal == 0 && coherent
                ? AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE
                : 0u;
        EXPECT_EQ(profile.atomic_operations_32, operations);
        EXPECT_EQ(profile.atomic_operations_64,
                  atomic_update_64 ? operations : 0u);
        EXPECT_NE(profile.visibility.describe_site, nullptr);
      }
    }
  }
  EXPECT_TRUE(state_.operations.empty());
}

}  // namespace
