// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/memory.h"

#include <array>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/memory_profile.h"

namespace {

enum class Operation {
  kQueryLayout,
  kRetainReference,
  kImport,
  kReserveAddress,
  kCreateAllocation,
  kMap,
  kWait,
  kMakeResident,
  kInvalidate,
  kDestroyAllocation,
  kFreeAddress,
  kReleaseReference,
};

struct Invalidation {
  uint32_t allocation_handle;
  uint64_t byte_offset;
  uint64_t byte_length;

  bool operator==(const Invalidation& other) const {
    return allocation_handle == other.allocation_handle &&
           byte_offset == other.byte_offset && byte_length == other.byte_length;
  }
};

struct FakeMemoryState {
  uint32_t allocation_count = 1;
  uint64_t maximum_allocation_byte_length = 65536;
  uint64_t expected_byte_length = 65536;
  uint64_t next_paging_fence = 1;
  uint32_t failing_map_ordinal = UINT32_MAX;
  amdf_status_t destroy_status = AMDF_STATUS_OK;
  amdf_gpu_dxg_memory_ownership_t destroy_ownership =
      AMDF_GPU_DXG_MEMORY_OWNERSHIP_CONSUMED;
  uint32_t destroy_count = 0;
  uint64_t reserved_byte_length = 0;
  void* host_pointer = nullptr;
  void* expected_host_pointer = nullptr;
  amdf_dxg_bridge_gpu_allocation_domain_t allocation_domain =
      AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL;
  // Exact private allocation-policy bits expected at the bridge boundary.
  amdf_dxg_bridge_gpu_allocation_flags_t allocation_flags = 0;
  std::vector<Operation> operations;
  std::vector<uint64_t> wait_targets;
  std::vector<Invalidation> invalidations;
  std::vector<uintptr_t> retained_references;
  std::vector<uintptr_t> released_references;
};

thread_local FakeMemoryState* current_state = nullptr;

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
FakeQueryAllocationLayout(amdf_dxg_bridge_gpu_adapter_t* adapter,
                          uint64_t byte_length, uint32_t* out_allocation_count,
                          uint64_t* out_maximum_allocation_byte_length) {
  auto* state = reinterpret_cast<FakeMemoryState*>(adapter);
  state->operations.push_back(Operation::kQueryLayout);
  EXPECT_EQ(byte_length, state->expected_byte_length);
  *out_allocation_count = state->allocation_count;
  *out_maximum_allocation_byte_length = state->maximum_allocation_byte_length;
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL FakeCreateAllocations(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource, uint32_t* out_allocation_count,
    uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeMemoryState*>(adapter);
  state->operations.push_back(Operation::kCreateAllocation);
  EXPECT_EQ(create_info->device_handle, 0x10u);
  EXPECT_EQ(create_info->domain, state->allocation_domain);
  EXPECT_EQ(create_info->flags, state->allocation_flags);
  EXPECT_EQ(create_info->byte_length, state->expected_byte_length);
  if (state->allocation_domain == AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL) {
    EXPECT_EQ(create_info->placement_device_address, 0x200000u);
    EXPECT_EQ(create_info->host_pointer, nullptr);
  } else {
    EXPECT_EQ(create_info->placement_device_address, 0u);
    EXPECT_NE(create_info->host_pointer, nullptr);
    if (state->expected_host_pointer != nullptr) {
      EXPECT_EQ(create_info->host_pointer, state->expected_host_pointer);
    }
    state->host_pointer = create_info->host_pointer;
  }
  EXPECT_EQ(allocation_handle_capacity, state->allocation_count);
  for (uint32_t i = 0; i < state->allocation_count; ++i) {
    out_allocation_handles[i] = 0x20 + i;
  }
  *out_resource = 0;
  *out_allocation_count = state->allocation_count;
  *out_native_status = 0;
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

amdf_status_t AMDF_DXG_BRIDGE_CALL FakePrepareBufferImport(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_buffer_import_info_t* import_info,
    uint32_t* resource_handle, uint32_t* allocation_handle,
    uint64_t* out_native_byte_length, uint64_t* out_buffer_byte_length) {
  auto* state = reinterpret_cast<FakeMemoryState*>(adapter);
  state->operations.push_back(Operation::kImport);
  EXPECT_EQ(import_info->device_handle, 0x10u);
  EXPECT_EQ(import_info->adapter_luid, 0x12345678u);
  EXPECT_EQ(import_info->shared_handle, reinterpret_cast<void*>(0x101u));
  *resource_handle = 0x30;
  *allocation_handle = 0x20;
  *out_native_byte_length = 65536;
  *out_buffer_byte_length = 16384;
  return AMDF_STATUS_OK;
}

amdf_status_t FakeReserveAddress(void* context, uint32_t adapter_handle,
                                 uint64_t base_address,
                                 uint64_t minimum_address, uint64_t byte_length,
                                 uint64_t* out_address) {
  auto* state = static_cast<FakeMemoryState*>(context);
  state->operations.push_back(Operation::kReserveAddress);
  EXPECT_EQ(adapter_handle, 0x08u);
  EXPECT_EQ(base_address, 0u);
  EXPECT_EQ(minimum_address, 0u);
  state->reserved_byte_length = byte_length;
  *out_address = 0x200000;
  return AMDF_STATUS_OK;
}

amdf_status_t FakeMapAllocation(void* context, uint32_t paging_queue_handle,
                                uint64_t address, uint32_t allocation_handle,
                                uint64_t allocation_byte_offset,
                                uint64_t byte_length,
                                amdf_memory_access_t access,
                                uint64_t* out_address,
                                uint64_t* out_paging_fence) {
  auto* state = static_cast<FakeMemoryState*>(context);
  state->operations.push_back(Operation::kMap);
  const uint32_t ordinal = allocation_handle - 0x20;
  EXPECT_EQ(paging_queue_handle, 0x40u);
  EXPECT_EQ(allocation_byte_offset, 0u);
  EXPECT_EQ(ordinal < state->allocation_count, true);
  EXPECT_EQ(address,
            0x200000u + ordinal * state->maximum_allocation_byte_length);
  EXPECT_EQ(byte_length, state->maximum_allocation_byte_length);
  EXPECT_EQ(access, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  if (ordinal == state->failing_map_ordinal) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  *out_address = address;
  *out_paging_fence = state->next_paging_fence++;
  return AMDF_STATUS_OK;
}

amdf_status_t FakeMakeResident(void* context, uint32_t paging_queue_handle,
                               uint32_t allocation_count,
                               const uint32_t* allocation_handles,
                               uint32_t* out_resident_allocation_count,
                               uint64_t* out_paging_fence) {
  auto* state = static_cast<FakeMemoryState*>(context);
  state->operations.push_back(Operation::kMakeResident);
  EXPECT_EQ(paging_queue_handle, 0x40u);
  EXPECT_EQ(allocation_count, state->allocation_count);
  for (uint32_t i = 0; i < allocation_count; ++i) {
    EXPECT_EQ(allocation_handles[i], 0x20u + i);
  }
  *out_resident_allocation_count = allocation_count;
  *out_paging_fence = state->next_paging_fence++;
  return AMDF_STATUS_OK;
}

amdf_status_t FakeWaitForPaging(void* context, uint32_t device_handle,
                                uint32_t synchronization_object_handle,
                                const volatile uint64_t* progress_fence,
                                uint64_t target_value) {
  auto* state = static_cast<FakeMemoryState*>(context);
  state->operations.push_back(Operation::kWait);
  EXPECT_EQ(device_handle, 0x10u);
  EXPECT_EQ(synchronization_object_handle, 0x50u);
  EXPECT_NE(progress_fence, nullptr);
  state->wait_targets.push_back(target_value);
  return AMDF_STATUS_OK;
}

amdf_gpu_dxg_memory_release_result_t FakeDestroyAllocations(
    void* context, uint32_t device_handle, uint32_t resource_handle,
    uint32_t allocation_count, const uint32_t* allocation_handles,
    bool synchronous) {
  auto* state = static_cast<FakeMemoryState*>(context);
  state->operations.push_back(Operation::kDestroyAllocation);
  ++state->destroy_count;
  EXPECT_EQ(device_handle, 0x10u);
  if (resource_handle == 0) {
    EXPECT_EQ(allocation_count, state->allocation_count);
    for (uint32_t i = 0; i < allocation_count; ++i) {
      EXPECT_EQ(allocation_handles[i], 0x20u + i);
    }
  } else {
    EXPECT_EQ(resource_handle, 0x30u);
  }
  EXPECT_EQ(synchronous, state->host_pointer != nullptr);
  return {
      .status = state->destroy_status,
      .ownership = state->destroy_ownership,
  };
}

amdf_status_t FakeFreeAddress(void* context, uint32_t adapter_handle,
                              uint64_t address, uint64_t byte_length) {
  auto* state = static_cast<FakeMemoryState*>(context);
  state->operations.push_back(Operation::kFreeAddress);
  EXPECT_EQ(adapter_handle, 0x08u);
  EXPECT_EQ(address, 0x200000u);
  EXPECT_EQ(byte_length, state->reserved_byte_length);
  return AMDF_STATUS_OK;
}

amdf_status_t FakeInvalidateAllocation(void* context, uint32_t device_handle,
                                       uint32_t allocation_handle,
                                       uint64_t byte_offset,
                                       uint64_t byte_length) {
  auto* state = static_cast<FakeMemoryState*>(context);
  state->operations.push_back(Operation::kInvalidate);
  EXPECT_EQ(device_handle, 0x10u);
  state->invalidations.push_back({allocation_handle, byte_offset, byte_length});
  return AMDF_STATUS_OK;
}

amdf_status_t FakeRetainExternalReference(uintptr_t reference,
                                          uintptr_t* out_reference) {
  current_state->operations.push_back(Operation::kRetainReference);
  current_state->retained_references.push_back(reference);
  *out_reference = reference + 1;
  return AMDF_STATUS_OK;
}

amdf_status_t FakeReleaseExternalReference(uintptr_t reference) {
  current_state->operations.push_back(Operation::kReleaseReference);
  current_state->released_references.push_back(reference);
  return AMDF_STATUS_OK;
}

class DxgMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    bridge_.gpu_allocation_query_layout = FakeQueryAllocationLayout;
    bridge_.gpu_allocation_create = FakeCreateAllocations;
    bridge_.gpu_buffer_prepare_import = FakePrepareBufferImport;
    adapter_.api = &bridge_;
    adapter_.native = reinterpret_cast<amdf_dxg_bridge_gpu_adapter_t*>(&state_);
    native_api_ = {
        .reserve_address = FakeReserveAddress,
        .map_allocation = FakeMapAllocation,
        .make_resident = FakeMakeResident,
        .wait_for_paging = FakeWaitForPaging,
        .destroy_allocations = FakeDestroyAllocations,
        .free_address = FakeFreeAddress,
        .invalidate_allocation = FakeInvalidateAllocation,
        .retain_external_reference = FakeRetainExternalReference,
        .release_external_reference = FakeReleaseExternalReference,
    };
    device_ = {
        .host_allocator = amdf_allocator_system(),
        .native_api = &native_api_,
        .native_context = &state_,
        .bridge_adapter = &adapter_,
        .adapter_handle = 0x08,
        .adapter_luid = 0x12345678,
        .device_handle = 0x10,
        .paging_queue_handle = 0x40,
        .paging_synchronization_object_handle = 0x50,
        .paging_progress_fence = &paging_progress_,
    };
    capabilities_ = {
        .virtual_address_bit_count = 48,
        .read_only_memory_supported = 1,
        .no_execute_memory_supported = 1,
        .cache_coherent_memory_supported = 1,
        .atomic_update_64_supported = 1,
        .d3d12_resource_supported = 1,
        .program_gfx_major = 11,
        .host_cache_transition =
            {
                .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
                .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
                .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
                .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
                .host_fence_before = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
                .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
                .range_granularity = 64,
            },
    };
    create_info_ = {
        .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .required_flags =
            AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        .byte_length = 65536,
        .minimum_alignment = 65536,
    };
  }

  void TearDown() override { current_state = nullptr; }

  amdf_memory_native_profile_t Profile(uint32_t ordinal) {
    amdf_memory_native_profile_t profile = {};
    EXPECT_EQ(
        amdf_gpu_dxg_query_memory_profile(&capabilities_, ordinal, &profile),
        AMDF_STATUS_OK);
    return profile;
  }

  FakeMemoryState state_;
  volatile uint64_t paging_progress_ = 0;
  amdf_dxg_bridge_api_t bridge_ = {};
  amdf_gpu_dxg_adapter_t adapter_ = {};
  amdf_gpu_dxg_memory_native_api_t native_api_ = {};
  amdf_gpu_dxg_memory_device_t device_ = {};
  amdf_gpu_dxg_memory_capabilities_t capabilities_ = {};
  amdf_memory_native_create_info_t create_info_ = {};
};

TEST_F(DxgMemoryTest, OwnsMultiChunkLocalMemoryInLifecycleOrder) {
  state_.allocation_count = 2;
  state_.maximum_allocation_byte_length = 32768;
  const amdf_memory_native_profile_t profile = Profile(1);
  amdf_gpu_dxg_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_dxg_memory_prepare(&device_, &profile, &create_info_,
                                        &memory, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.device_address, 0x200000u);
  EXPECT_EQ(result.physical_backing_id.words[1], (UINT64_C(2) << 32) | 0x21u);
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{2, 3}));
  EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.operations, (std::vector<Operation>{
                                   Operation::kQueryLayout,
                                   Operation::kReserveAddress,
                                   Operation::kCreateAllocation,
                                   Operation::kMap,
                                   Operation::kMap,
                                   Operation::kWait,
                                   Operation::kMakeResident,
                                   Operation::kWait,
                                   Operation::kDestroyAllocation,
                                   Operation::kFreeAddress,
                               }));
}

TEST_F(DxgMemoryTest, AlignsSystemStorageAndPublishesEachNativeSpan) {
  state_.allocation_count = 2;
  state_.maximum_allocation_byte_length = 32768;
  state_.allocation_domain = AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  create_info_.minimum_alignment = UINT64_C(2) << 20;
  const amdf_memory_native_profile_t profile = Profile(0);
  amdf_gpu_dxg_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_dxg_memory_prepare(&device_, &profile, &create_info_,
                                        &memory, &result),
            AMDF_STATUS_OK);
  ASSERT_NE(state_.host_pointer, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(state_.host_pointer) %
                create_info_.minimum_alignment,
            0u);
  EXPECT_EQ(result.alignment, create_info_.minimum_alignment);

  amdf_gpu_dxg_memory_t* mapping = nullptr;
  amdf_gpu_umd_host_mapping_result_t mapping_result = {};
  const amdf_memory_map_info_t map_info = {
      .byte_offset = 32760,
      .byte_length = 16,
  };
  ASSERT_EQ(amdf_gpu_dxg_memory_map(memory, &profile.host_mapping, &map_info,
                                    &mapping, &mapping_result),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_result.visibility.flush.executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API);
  ASSERT_EQ(amdf_gpu_dxg_memory_cache_control(
                mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 32760, 16),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.invalidations,
            (std::vector<Invalidation>{{0x20, 32760, 8}, {0x21, 0, 8}}));
  EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), AMDF_STATUS_OK);
}

TEST_F(DxgMemoryTest, RegistersExactSubpageRangesOverSharedBacking) {
  alignas(AMDF_GPU_DXG_PAGE_SIZE)
      std::array<uint8_t, AMDF_GPU_DXG_PAGE_SIZE * 6>
          caller_pages = {};
  constexpr uint64_t kLogicalByteLength = AMDF_GPU_DXG_PAGE_SIZE * 4 + 17;
  constexpr std::array<uint64_t, 2> kHostByteOffsets = {3, 19};
  constexpr uint64_t kNativeByteLength = AMDF_GPU_DXG_PAGE_SIZE * 5;
  state_.allocation_domain =
      AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST;
  state_.expected_byte_length = kNativeByteLength;
  state_.maximum_allocation_byte_length = kNativeByteLength;
  state_.expected_host_pointer = caller_pages.data();
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  create_info_.byte_length = kLogicalByteLength;
  create_info_.minimum_alignment = 1;
  const amdf_memory_native_profile_t profile = Profile(2);
  amdf_physical_memory_id_t first_backing_id = {};

  for (uint64_t host_byte_offset : kHostByteOffsets) {
    create_info_.registered_host_pointer =
        caller_pages.data() + host_byte_offset;
    amdf_gpu_dxg_memory_t* memory = nullptr;
    amdf_gpu_umd_memory_result_t result = {};
    ASSERT_EQ(amdf_gpu_dxg_memory_prepare(&device_, &profile, &create_info_,
                                          &memory, &result),
              AMDF_STATUS_OK);
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(result.source_byte_offset, host_byte_offset);
    EXPECT_EQ(result.byte_length, kLogicalByteLength);
    EXPECT_EQ(result.alignment, 1u);
    EXPECT_EQ(result.native_allocation_byte_length, kNativeByteLength);
    EXPECT_EQ(result.device_address, 0x200000u + host_byte_offset);
    EXPECT_EQ(result.physical_backing_id.words[0],
              reinterpret_cast<uintptr_t>(caller_pages.data()));
    EXPECT_EQ(result.physical_backing_id.words[1], kNativeByteLength);
    if (first_backing_id.words[0] == 0) {
      first_backing_id = result.physical_backing_id;
    } else {
      EXPECT_TRUE(amdf_physical_memory_id_is_equal(
          &first_backing_id, &result.physical_backing_id));
    }

    amdf_gpu_dxg_memory_t* mapping = nullptr;
    amdf_gpu_umd_host_mapping_result_t mapping_result = {};
    const amdf_memory_map_info_t map_info = {
        .byte_length = kLogicalByteLength,
    };
    ASSERT_EQ(amdf_gpu_dxg_memory_map(memory, &profile.host_mapping, &map_info,
                                      &mapping, &mapping_result),
              AMDF_STATUS_OK);
    EXPECT_EQ(mapping_result.pointer, create_info_.registered_host_pointer);

    state_.invalidations.clear();
    ASSERT_EQ(
        amdf_gpu_dxg_memory_cache_control(
            mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kLogicalByteLength),
        AMDF_STATUS_OK);
    EXPECT_EQ(state_.invalidations,
              (std::vector<Invalidation>{
                  {0x20, host_byte_offset, kLogicalByteLength}}));
    EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  }
}

TEST_F(DxgMemoryTest, PreservesIndependentNativeAllocationPolicies) {
  state_.allocation_domain = AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
  const amdf_memory_native_profile_t profile = Profile(0);
  struct TestCase {
    // One independent public policy requested by the allocation.
    amdf_memory_flags_t memory_flag;
    // Corresponding private bridge policy.
    amdf_dxg_bridge_gpu_allocation_flags_t allocation_flag;
  };
  const TestCase test_cases[] = {
      {AMDF_MEMORY_FLAG_HOST_COHERENT,
       AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN},
      {AMDF_MEMORY_FLAG_DEVICE_UNCACHED,
       AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED},
      {AMDF_MEMORY_FLAG_QUEUE_STORAGE,
       AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_QUEUE_STORAGE},
  };
  for (const TestCase& test_case : test_cases) {
    create_info_.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                                  AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                                  test_case.memory_flag;
    state_.allocation_flags = test_case.allocation_flag;
    amdf_gpu_dxg_memory_t* memory = nullptr;
    amdf_gpu_umd_memory_result_t result = {};
    ASSERT_EQ(amdf_gpu_dxg_memory_prepare(&device_, &profile, &create_info_,
                                          &memory, &result),
              AMDF_STATUS_OK);
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  }
}

TEST_F(DxgMemoryTest, ConsumedFailureAdvancesBeforeRetry) {
  const amdf_memory_native_profile_t profile = Profile(1);
  amdf_gpu_dxg_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_dxg_memory_prepare(&device_, &profile, &create_info_,
                                        &memory, &result),
            AMDF_STATUS_OK);
  state_.destroy_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  state_.destroy_ownership = AMDF_GPU_DXG_MEMORY_OWNERSHIP_CONSUMED;
  EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), state_.destroy_status);
  EXPECT_EQ(state_.destroy_count, 1u);
  state_.destroy_status = AMDF_STATUS_OK;
  EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroy_count, 1u);
  EXPECT_EQ(state_.operations.back(), Operation::kFreeAddress);
}

TEST_F(DxgMemoryTest, RetainedFailureRetriesTheNativeAllocation) {
  const amdf_memory_native_profile_t profile = Profile(1);
  amdf_gpu_dxg_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_dxg_memory_prepare(&device_, &profile, &create_info_,
                                        &memory, &result),
            AMDF_STATUS_OK);
  state_.destroy_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  state_.destroy_ownership = AMDF_GPU_DXG_MEMORY_OWNERSHIP_RETAINED;
  EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), state_.destroy_status);
  state_.destroy_status = AMDF_STATUS_OK;
  state_.destroy_ownership = AMDF_GPU_DXG_MEMORY_OWNERSHIP_CONSUMED;
  EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroy_count, 2u);
}

TEST_F(DxgMemoryTest, PartialMappingFailureRetainsCompleteRollbackOwner) {
  state_.allocation_count = 2;
  state_.maximum_allocation_byte_length = 32768;
  state_.failing_map_ordinal = 1;
  const amdf_memory_native_profile_t profile = Profile(1);
  amdf_gpu_dxg_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  EXPECT_EQ(amdf_gpu_dxg_memory_prepare(&device_, &profile, &create_info_,
                                        &memory, &result),
            amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroy_count, 1u);
  EXPECT_EQ(state_.operations.back(), Operation::kFreeAddress);
}

TEST_F(DxgMemoryTest, ImportOwnsIndependentReferencesAndLogicalRange) {
  const amdf_memory_native_profile_t profile = Profile(3);
  const amdf_memory_native_import_info_t import_info = {
      .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .required_flags = profile.guaranteed_flags,
      .minimum_alignment = 65536,
  };
  const amdf_external_memory_payload_t external_payload = {
      .native_handle = reinterpret_cast<void*>(0x100u),
  };
  const amdf_external_memory_t external = {
      .payload = external_payload,
      .source_byte_offset = 4096,
      .byte_length = 8192,
  };
  amdf_gpu_dxg_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_dxg_memory_prepare_import(&device_, &profile, &import_info,
                                               &external, &memory, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.source_byte_offset, 4096u);
  EXPECT_EQ(result.byte_length, 8192u);
  EXPECT_EQ(result.device_address, 0x201000u);

  amdf_external_memory_t exported = {};
  const amdf_memory_export_info_t export_info = {};
  ASSERT_EQ(amdf_gpu_dxg_memory_export(memory, &export_info, &exported),
            AMDF_STATUS_OK);
  EXPECT_EQ(exported.payload.native_handle, reinterpret_cast<void*>(0x102u));
  exported.release(exported.release_user_data, exported.type, exported.payload);
  EXPECT_EQ(amdf_gpu_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.retained_references,
            (std::vector<uintptr_t>{0x100u, 0x101u}));
  EXPECT_EQ(state_.released_references,
            (std::vector<uintptr_t>{0x102u, 0x101u}));
}

}  // namespace
