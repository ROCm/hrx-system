// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/ioctl.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/dxg/linux/device.h"

namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103u);
constexpr NTSTATUS kStatusNoMemory = static_cast<NTSTATUS>(0xC0000017u);

enum class Operation {
  kQueryLayout,
  kReserveAddress,
  kCreateAllocation,
  kMap,
  kWait,
  kQueryDeviceState,
  kMakeResident,
  kInvalidate,
  kDestroyAllocation,
  kFreeAddress,
};

enum class FailureStage {
  kNone,
  kReserveAddress,
  kCreateAllocation,
  kMap,
  kMapWait,
  kMakeResident,
  kResidentWait,
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
  FailureStage failure_stage = FailureStage::kNone;
  amdf_dxg_provider_gpu_allocation_domain_t allocation_domain =
      AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_LOCAL;
  D3DKMT_DEVICEEXECUTION_STATE execution_state = D3DKMT_DEVICEEXECUTION_ACTIVE;
  uint32_t destroy_failures_remaining = 0;
  uint32_t free_failures_remaining = 0;
  uint32_t invalidate_failures_remaining = 0;
  volatile uint64_t paging_progress = 0;
  void* host_pointer = nullptr;
  bool synchronous_destroy = false;
  std::vector<Operation> operations;
  std::vector<Invalidation> invalidations;
};

FakeMemoryState* current_state = nullptr;
amdf_linux_dxg_connection_t* current_connection = nullptr;

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  EXPECT_EQ(context, current_connection);
  switch (_IOC_NR(request)) {
    case 0x08: {
      current_state->operations.push_back(Operation::kReserveAddress);
      auto* reserve = static_cast<D3DDDI_RESERVEGPUVIRTUALADDRESS*>(arguments);
      EXPECT_EQ(reserve->hAdapter, 0x08u);
      EXPECT_EQ(reserve->Size, 65536u);
      if (current_state->failure_stage == FailureStage::kReserveAddress) {
        return static_cast<uint32_t>(kStatusNoMemory);
      }
      reserve->VirtualAddress = 0x100000;
      return 0;
    }
    case 0x0B: {
      current_state->operations.push_back(Operation::kMakeResident);
      auto* resident = static_cast<D3DDDI_MAKERESIDENT*>(arguments);
      EXPECT_EQ(resident->hPagingQueue, 0x40u);
      EXPECT_EQ(resident->NumAllocations, 1u);
      EXPECT_EQ(resident->AllocationList[0], 0x20u);
      EXPECT_EQ(resident->Flags.CantTrimFurther, 1u);
      if (current_state->failure_stage == FailureStage::kMakeResident) {
        return static_cast<uint32_t>(kStatusNoMemory);
      }
      resident->PagingFenceValue = 2;
      return static_cast<uint32_t>(kStatusPending);
    }
    case 0x0C: {
      current_state->operations.push_back(Operation::kMap);
      auto* map = static_cast<D3DDDI_MAPGPUVIRTUALADDRESS*>(arguments);
      EXPECT_EQ(map->hPagingQueue, 0x40u);
      EXPECT_EQ(map->BaseAddress, 0x100000u);
      EXPECT_EQ(map->hAllocation, 0x20u);
      EXPECT_EQ(map->SizeInPages, 16u);
      EXPECT_EQ(map->Protection.Write, 1u);
      EXPECT_EQ(map->Protection.Execute, 0u);
      if (current_state->failure_stage == FailureStage::kMap) {
        return static_cast<uint32_t>(kStatusNoMemory);
      }
      map->VirtualAddress = map->BaseAddress;
      map->PagingFenceValue = 1;
      return static_cast<uint32_t>(kStatusPending);
    }
    case 0x0E: {
      current_state->operations.push_back(Operation::kQueryDeviceState);
      auto* query = static_cast<D3DKMT_GETDEVICESTATE*>(arguments);
      EXPECT_EQ(query->hDevice, 0x10u);
      EXPECT_EQ(query->StateType, D3DKMT_DEVICESTATE_EXECUTION);
      query->ExecutionState = current_state->execution_state;
      return 0;
    }
    case 0x13: {
      current_state->operations.push_back(Operation::kDestroyAllocation);
      const auto* destroy = static_cast<D3DKMT_DESTROYALLOCATION2*>(arguments);
      EXPECT_EQ(destroy->hDevice, 0x10u);
      EXPECT_EQ(destroy->hResource, 0u);
      EXPECT_EQ(destroy->AllocationCount, 1u);
      EXPECT_EQ(destroy->phAllocationList[0], 0x20u);
      EXPECT_EQ(destroy->Flags.AssumeNotInUse, 1u);
      EXPECT_EQ(destroy->Flags.SynchronousDestroy,
                current_state->synchronous_destroy);
      if (current_state->destroy_failures_remaining != 0) {
        --current_state->destroy_failures_remaining;
        return static_cast<uint32_t>(kStatusNoMemory);
      }
      return 0;
    }
    case 0x20: {
      current_state->operations.push_back(Operation::kFreeAddress);
      const auto* free_address =
          static_cast<D3DKMT_FREEGPUVIRTUALADDRESS*>(arguments);
      EXPECT_EQ(free_address->hAdapter, 0x08u);
      EXPECT_EQ(free_address->BaseAddress, 0x100000u);
      EXPECT_EQ(free_address->Size, 65536u);
      if (current_state->free_failures_remaining != 0) {
        --current_state->free_failures_remaining;
        return static_cast<uint32_t>(kStatusNoMemory);
      }
      return 0;
    }
    case 0x24: {
      current_state->operations.push_back(Operation::kInvalidate);
      const auto* invalidate = static_cast<D3DKMT_INVALIDATECACHE*>(arguments);
      EXPECT_EQ(invalidate->hDevice, 0x10u);
      current_state->invalidations.push_back(
          {invalidate->hAllocation, invalidate->Offset, invalidate->Length});
      if (current_state->invalidate_failures_remaining != 0) {
        --current_state->invalidate_failures_remaining;
        return static_cast<uint32_t>(kStatusNoMemory);
      }
      return 0;
    }
    case 0x3A: {
      current_state->operations.push_back(Operation::kWait);
      const auto* wait =
          static_cast<D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU*>(arguments);
      EXPECT_EQ(wait->hDevice, 0x10u);
      EXPECT_EQ(wait->ObjectCount, 1u);
      EXPECT_EQ(wait->ObjectHandleArray[0], 0x50u);
      const uint64_t target = wait->FenceValueArray[0];
      EXPECT_TRUE(target == 1 || target == 2);
      if ((target == 1 &&
           current_state->failure_stage == FailureStage::kMapWait) ||
          (target == 2 &&
           current_state->failure_stage == FailureStage::kResidentWait)) {
        return static_cast<uint32_t>(kStatusNoMemory);
      }
      current_state->paging_progress = target;
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected DXG operation " << _IOC_NR(request);
      return static_cast<uint32_t>(kStatusNoMemory);
  }
}

amdf_dxg_provider_result_t FakeQueryAllocationLayout(
    amdf_dxg_provider_gpu_adapter_t* adapter, uint64_t byte_length,
    uint32_t* out_allocation_count,
    uint64_t* out_maximum_allocation_byte_length) {
  EXPECT_EQ(adapter,
            reinterpret_cast<amdf_dxg_provider_gpu_adapter_t*>(current_state));
  current_state->operations.push_back(Operation::kQueryLayout);
  EXPECT_EQ(byte_length, 65536u);
  *out_allocation_count = 1;
  *out_maximum_allocation_byte_length = 65536;
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_provider_result_t FakeCreateAllocation(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    const amdf_dxg_provider_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) {
  EXPECT_EQ(adapter,
            reinterpret_cast<amdf_dxg_provider_gpu_adapter_t*>(current_state));
  current_state->operations.push_back(Operation::kCreateAllocation);
  EXPECT_EQ(create_info->device_handle, 0x10u);
  EXPECT_EQ(create_info->domain, current_state->allocation_domain);
  EXPECT_EQ(create_info->byte_length, 65536u);
  if (current_state->allocation_domain ==
      AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_LOCAL) {
    EXPECT_EQ(create_info->placement_device_address, 0x100000u);
    EXPECT_EQ(create_info->host_pointer, nullptr);
  } else {
    EXPECT_EQ(create_info->placement_device_address, 0u);
    EXPECT_NE(create_info->host_pointer, nullptr);
    current_state->host_pointer = create_info->host_pointer;
  }
  if (current_state->failure_stage == FailureStage::kCreateAllocation) {
    *out_native_status = static_cast<uint32_t>(kStatusNoMemory);
    return AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE;
  }
  EXPECT_EQ(allocation_handle_capacity, 1u);
  out_allocation_handles[0] = 0x20;
  *out_resource_handle = 0;
  *out_allocation_count = 1;
  *out_native_status = 0;
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

class LinuxDxgMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    current_connection = &connection_;
    allocator_ = amdf_allocator_system();
    connection_.descriptor = 1;
    kmt_.connection = &connection_;
    kmt_.host_allocator = &allocator_;
    provider_api_.gpu_allocation_query_layout = FakeQueryAllocationLayout;
    provider_api_.gpu_allocation_create = FakeCreateAllocation;
    device_.host_allocator = allocator_;
    device_.adapter = 0x08;
    device_.adapter_luid = UINT64_C(0x1122334455667788);
    device_.memory_capabilities = {
        .virtual_address_bit_count = 48,
        .host_registration_granularity = 1,
        .read_only_memory_supported = 1,
        .no_execute_memory_supported = 1,
        .cache_coherent_memory_supported = 1,
        .atomic_update_64_supported = 1,
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
    device_.logical_device.kmt = &kmt_;
    device_.logical_device.native = 0x10;
    device_.logical_device.provider_adapter = {
        .api = &provider_api_,
        .native = reinterpret_cast<amdf_dxg_provider_gpu_adapter_t*>(&state_),
    };
    device_.paging_queue = {
        .kmt = &kmt_,
        .native = 0x40,
        .synchronization_object = 0x50,
        .progress_fence = &state_.paging_progress,
    };
    create_info_ = {
        .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .required_flags =
            AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        .byte_length = 65536,
        .minimum_alignment = 65536,
    };
  }

  void TearDown() override {
    current_connection = nullptr;
    current_state = nullptr;
  }

  amdf_memory_native_profile_t Profile(uint32_t ordinal) {
    amdf_memory_native_profile_t profile = {};
    EXPECT_EQ(amdf_gpu_linux_dxg_device_query_memory_profile(&device_, ordinal,
                                                             &profile),
              AMDF_STATUS_OK);
    return profile;
  }

  amdf_status_t Prepare(uint32_t profile_ordinal,
                        amdf_gpu_umd_memory_t** out_memory) {
    const amdf_memory_native_profile_t profile = Profile(profile_ordinal);
    amdf_gpu_umd_memory_result_t result = {};
    return amdf_gpu_linux_dxg_memory_prepare(
        &device_, 0, nullptr, &profile, &create_info_, out_memory, &result);
  }

  FakeMemoryState state_;
  amdf_allocator_t allocator_ = {};
  amdf_linux_dxg_connection_t connection_ = {};
  amdf_linux_dxg_kmt_t kmt_ = {};
  amdf_dxg_provider_api_t provider_api_ = {};
  amdf_gpu_umd_device_t device_ = {};
  amdf_memory_native_create_info_t create_info_ = {};
};

TEST_F(LinuxDxgMemoryTest, OwnsLocalMemoryThroughPublishedKmtOperations) {
  amdf_gpu_umd_memory_t* memory = nullptr;
  ASSERT_EQ(Prepare(1, &memory), AMDF_STATUS_OK);
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(state_.operations, (std::vector<Operation>{
                                   Operation::kQueryLayout,
                                   Operation::kReserveAddress,
                                   Operation::kCreateAllocation,
                                   Operation::kMap,
                                   Operation::kWait,
                                   Operation::kQueryDeviceState,
                                   Operation::kMakeResident,
                                   Operation::kWait,
                                   Operation::kQueryDeviceState,
                               }));

  ASSERT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.operations[state_.operations.size() - 2],
            Operation::kDestroyAllocation);
  EXPECT_EQ(state_.operations.back(), Operation::kFreeAddress);
}

TEST_F(LinuxDxgMemoryTest, OwnsSystemStorageAndPublishesHostWrites) {
  state_.allocation_domain = AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_SYSTEM;
  state_.synchronous_destroy = true;
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  const amdf_memory_native_profile_t profile = Profile(0);
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_linux_dxg_memory_prepare(&device_, 0, nullptr, &profile,
                                              &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  ASSERT_NE(state_.host_pointer, nullptr);

  amdf_gpu_umd_host_mapping_t* mapping = nullptr;
  amdf_gpu_umd_host_mapping_result_t mapping_result = {};
  const amdf_memory_map_info_t map_info = {
      .byte_offset = 64,
      .byte_length = 128,
  };
  ASSERT_EQ(amdf_gpu_linux_dxg_memory_map(memory, &profile.host_mapping,
                                          &map_info, &mapping, &mapping_result),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_result.pointer,
            static_cast<uint8_t*>(state_.host_pointer) + 64);
  ASSERT_EQ(amdf_gpu_linux_dxg_host_mapping_cache_control(
                mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 64, 128),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.invalidations, (std::vector<Invalidation>{{0x20, 64, 128}}));
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);
}

TEST_F(LinuxDxgMemoryTest, RegistersBorrowedHostStorageWithoutOwningIt) {
  alignas(4096) static uint8_t registered_storage[65536];
  state_.allocation_domain =
      AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_REGISTERED_HOST;
  state_.synchronous_destroy = true;
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  create_info_.registered_host_pointer = registered_storage;
  create_info_.minimum_alignment = 4096;

  amdf_gpu_umd_memory_t* memory = nullptr;
  ASSERT_EQ(Prepare(2, &memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.host_pointer, registered_storage);
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  registered_storage[0] = 1;
  EXPECT_EQ(registered_storage[0], 1u);
}

class LinuxDxgMemoryPreparationFailureTest
    : public LinuxDxgMemoryTest,
      public ::testing::WithParamInterface<FailureStage> {};

TEST_P(LinuxDxgMemoryPreparationFailureTest,
       PreservesExactProgressForOrderedRollback) {
  state_.failure_stage = GetParam();
  amdf_gpu_umd_memory_t* memory = nullptr;
  EXPECT_EQ(Prepare(1, &memory),
            amdf_linux_dxg_kmt_make_status(kStatusNoMemory));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);

  std::vector<Operation> expected = {
      Operation::kQueryLayout,
      Operation::kReserveAddress,
  };
  if (GetParam() != FailureStage::kReserveAddress) {
    expected.push_back(Operation::kCreateAllocation);
  }
  if (GetParam() > FailureStage::kCreateAllocation) {
    expected.push_back(Operation::kMap);
  }
  if (GetParam() > FailureStage::kMap) {
    expected.push_back(Operation::kWait);
  }
  if (GetParam() >= FailureStage::kMapWait) {
    expected.push_back(Operation::kQueryDeviceState);
  }
  if (GetParam() > FailureStage::kMapWait) {
    expected.push_back(Operation::kMakeResident);
  }
  if (GetParam() > FailureStage::kMakeResident) {
    expected.push_back(Operation::kWait);
    expected.push_back(Operation::kQueryDeviceState);
  }
  if (GetParam() > FailureStage::kCreateAllocation) {
    expected.push_back(Operation::kDestroyAllocation);
  }
  if (GetParam() != FailureStage::kReserveAddress) {
    expected.push_back(Operation::kFreeAddress);
  }
  EXPECT_EQ(state_.operations, expected);
}

INSTANTIATE_TEST_SUITE_P(NativeStage, LinuxDxgMemoryPreparationFailureTest,
                         ::testing::Values(FailureStage::kReserveAddress,
                                           FailureStage::kCreateAllocation,
                                           FailureStage::kMap,
                                           FailureStage::kMapWait,
                                           FailureStage::kMakeResident,
                                           FailureStage::kResidentWait));

TEST_F(LinuxDxgMemoryTest, DestroyFailureConsumesAllocationBeforeRetry) {
  amdf_gpu_umd_memory_t* memory = nullptr;
  ASSERT_EQ(Prepare(1, &memory), AMDF_STATUS_OK);
  state_.destroy_failures_remaining = 1;
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory),
            amdf_linux_dxg_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kDestroyAllocation),
            1);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kFreeAddress),
            1);
}

TEST_F(LinuxDxgMemoryTest, FreeFailureRetainsAddressForOneExactRetry) {
  amdf_gpu_umd_memory_t* memory = nullptr;
  ASSERT_EQ(Prepare(1, &memory), AMDF_STATUS_OK);
  state_.free_failures_remaining = 1;
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory),
            amdf_linux_dxg_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kDestroyAllocation),
            1);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kFreeAddress),
            2);
}

TEST_F(LinuxDxgMemoryTest, ResetFenceCannotPublishPagingCompletion) {
  state_.paging_progress = UINT64_MAX;
  amdf_gpu_umd_memory_t* memory = nullptr;
  EXPECT_EQ(Prepare(1, &memory),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kWait),
            0);
  EXPECT_EQ(std::count(state_.operations.begin(), state_.operations.end(),
                       Operation::kQueryDeviceState),
            0);
}

TEST_F(LinuxDxgMemoryTest, StoppedDeviceCannotPublishRetiredPaging) {
  state_.execution_state = D3DKMT_DEVICEEXECUTION_STOPPED;
  amdf_gpu_umd_memory_t* memory = nullptr;
  EXPECT_EQ(Prepare(1, &memory),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);
}

TEST_F(LinuxDxgMemoryTest, CachePublicationReturnsExactNativeFailure) {
  state_.allocation_domain = AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_SYSTEM;
  state_.synchronous_destroy = true;
  create_info_.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  const amdf_memory_native_profile_t profile = Profile(0);
  amdf_gpu_umd_memory_t* memory = nullptr;
  amdf_gpu_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_gpu_linux_dxg_memory_prepare(&device_, 0, nullptr, &profile,
                                              &create_info_, &memory, &result),
            AMDF_STATUS_OK);
  state_.invalidate_failures_remaining = 1;
  EXPECT_EQ(amdf_gpu_linux_dxg_host_mapping_cache_control(
                reinterpret_cast<amdf_gpu_umd_host_mapping_t*>(memory),
                AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 64),
            amdf_linux_dxg_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(amdf_gpu_linux_dxg_memory_destroy(memory), AMDF_STATUS_OK);
}

}  // namespace
