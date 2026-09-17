// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/memory.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/mcdm/device.h"

namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103u);
constexpr NTSTATUS kStatusNoMemory = static_cast<NTSTATUS>(0xC0000017u);

enum class FailurePoint {
  kNone,
  kCreate,
  kMap,
  kInvalidMapAddress,
  kFirstWait,
  kResident,
  kPartialResident,
  kSecondWait,
  kDestroy,
};

enum class Operation {
  kCreate,
  kMap,
  kWait,
  kResident,
  kDestroy,
};

struct FakeKmtState {
  // Exclusive device-address limit expected in the native mapping request.
  uint64_t address_limit = (UINT64_C(1) << 48) - UINT64_C(0x80000000);
  // Native allocation length expected after logical-length rounding.
  uint64_t byte_length = 65536;
  // Native mapping base returned when no malformed-alignment failure is set.
  uint64_t mapped_address = UINT64_C(0x13000);
  // Native operation selected for a failure response.
  FailurePoint failure_point = FailurePoint::kNone;
  // Number of allocation release calls rejected before consuming the handle.
  uint32_t destroy_failures_remaining = 0;
  // Real host backing borrowed by the modeled native allocation.
  const void* host_pointer = nullptr;
  // Memory and host-view metadata returned to the host allocator.
  uint32_t metadata_free_count = 0;
  // Flags observed in the residency request.
  D3DDDI_MAKERESIDENT_FLAGS resident_flags = {};
  // Native operations in call order.
  std::vector<Operation> operations;
  // Paging fence values observed by CPU waits.
  std::vector<uint64_t> wait_targets;
};

FakeKmtState* g_fake_state = nullptr;

NTSTATUS APIENTRY FakeCreateAllocation(D3DKMT_CREATEALLOCATION* create) {
  g_fake_state->operations.push_back(Operation::kCreate);
  if (g_fake_state->failure_point == FailurePoint::kCreate) {
    return kStatusNoMemory;
  }
  EXPECT_EQ(create->Flags.StandardAllocation, 1u);
  EXPECT_EQ(create->Flags.ExistingSysMem, 1u);
  EXPECT_EQ(create->NumAllocations, 1u);
  EXPECT_NE(create->pAllocationInfo2[0].pSystemMem, nullptr);
  EXPECT_EQ(create->pStandardAllocation->ExistingHeapData.Size,
            g_fake_state->byte_length);
  g_fake_state->host_pointer = create->pAllocationInfo2[0].pSystemMem;
  create->pAllocationInfo2[0].hAllocation = 0x20;
  return 0;
}

NTSTATUS APIENTRY
FakeDestroyAllocation(const D3DKMT_DESTROYALLOCATION2* destroy) {
  g_fake_state->operations.push_back(Operation::kDestroy);
  EXPECT_EQ(destroy->hDevice, 0x10u);
  EXPECT_EQ(destroy->hResource, 0u);
  EXPECT_EQ(destroy->AllocationCount, 1u);
  EXPECT_EQ(destroy->phAllocationList[0], 0x20u);
  EXPECT_EQ(destroy->Flags.AssumeNotInUse, 1u);
  EXPECT_EQ(destroy->Flags.SynchronousDestroy, 1u);
  if (g_fake_state->destroy_failures_remaining != 0) {
    --g_fake_state->destroy_failures_remaining;
    return kStatusNoMemory;
  }
  if (g_fake_state->failure_point == FailurePoint::kDestroy) {
    return kStatusNoMemory;
  }
  return 0;
}

NTSTATUS APIENTRY FakeMapGpuVirtualAddress(D3DDDI_MAPGPUVIRTUALADDRESS* map) {
  g_fake_state->operations.push_back(Operation::kMap);
  if (g_fake_state->failure_point == FailurePoint::kMap) {
    return kStatusNoMemory;
  }
  EXPECT_EQ(map->hPagingQueue, 0x30u);
  EXPECT_EQ(map->hAllocation, 0x20u);
  EXPECT_EQ(map->SizeInPages, g_fake_state->byte_length / 4096);
  EXPECT_EQ(map->BaseAddress, 0u);
  EXPECT_EQ(map->MinimumAddress, UINT64_C(65536));
  EXPECT_EQ(map->MaximumAddress, g_fake_state->address_limit);
  EXPECT_EQ(map->Protection.Write, 1u);
  EXPECT_EQ(map->Protection.Execute, 0u);
  map->VirtualAddress =
      g_fake_state->failure_point == FailurePoint::kInvalidMapAddress
          ? UINT64_C(0x12340001)
          : g_fake_state->mapped_address;
  map->PagingFenceValue = 1;
  return kStatusPending;
}

NTSTATUS APIENTRY FakeMakeResident(D3DDDI_MAKERESIDENT* resident) {
  g_fake_state->operations.push_back(Operation::kResident);
  g_fake_state->resident_flags = resident->Flags;
  if (g_fake_state->failure_point == FailurePoint::kResident) {
    return kStatusNoMemory;
  }
  EXPECT_EQ(resident->hPagingQueue, 0x30u);
  EXPECT_EQ(resident->NumAllocations, 1u);
  EXPECT_EQ(resident->AllocationList[0], 0x20u);
  if (g_fake_state->failure_point == FailurePoint::kPartialResident) {
    resident->NumAllocations = 0;
  }
  resident->PagingFenceValue = 2;
  return kStatusPending;
}

NTSTATUS APIENTRY
FakeWaitFromCpu(const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) {
  g_fake_state->operations.push_back(Operation::kWait);
  EXPECT_EQ(wait->hDevice, 0x10u);
  EXPECT_EQ(wait->ObjectCount, 1u);
  EXPECT_EQ(wait->ObjectHandleArray[0], 0x40u);
  const uint64_t target = wait->FenceValueArray[0];
  g_fake_state->wait_targets.push_back(target);
  const uint64_t failing_target =
      g_fake_state->failure_point == FailurePoint::kFirstWait    ? 1
      : g_fake_state->failure_point == FailurePoint::kSecondWait ? 2
                                                                 : 0;
  if (target == failing_target) {
    return kStatusNoMemory;
  }
  return 0;
}

class WindowsXdnaMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_fake_state = &state_;
    kmt_.create_allocation = FakeCreateAllocation;
    kmt_.destroy_allocation = FakeDestroyAllocation;
    kmt_.map_gpu_virtual_address = FakeMapGpuVirtualAddress;
    kmt_.make_resident = FakeMakeResident;
    kmt_.wait_from_cpu = FakeWaitFromCpu;
    device_.host_allocator = amdf_allocator_system();
    device_.host_allocator.user_data = &state_;
    device_.host_allocator.free = [](void* user_data, void* allocation) {
      ++static_cast<FakeKmtState*>(user_data)->metadata_free_count;
      amdf_free(amdf_allocator_system(), allocation);
    };
    device_.kmt = &kmt_;
    device_profile_.dma.address_bit_count = 48;
    device_profile_.dma.byte_offset = UINT32_C(0x80000000);
    device_.profile = &device_profile_;
    device_.device = 0x10;
    device_.paging_queue = 0x30;
    device_.paging_sync_object = 0x40;
    device_.paging_fence = &paging_fence_;

    create_info_.device_access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    create_info_.required_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    create_info_.byte_length = 4097;
    create_info_.minimum_alignment = 4096;
    ASSERT_TRUE(amdf_status_is_ok(
        amdf_xdna_umd_device_query_memory_profile(&device_, 0, &profile_)));
  }

  void TearDown() override { g_fake_state = nullptr; }

  void ReleaseLeakedBacking() {
    MEMORY_BASIC_INFORMATION information = {};
    ASSERT_NE(
        VirtualQuery(state_.host_pointer, &information, sizeof(information)),
        0u);
    EXPECT_EQ(information.State, MEM_COMMIT);
    // No real native allocation exists in this model. Reclaim the known test
    // backing without retrying the failed native operation.
    EXPECT_TRUE(VirtualFree(information.AllocationBase, 0, MEM_RELEASE));
  }

  // Native dependency results and ordered observations.
  FakeKmtState state_;
  // Native procedures borrowed by the production memory implementation.
  amdf_kmt_api_t kmt_ = {};
  // Explicitly live device borrowed by the memory constructor.
  amdf_xdna_umd_device_t device_ = {};
  // Immutable hardware address geometry consumed by native qualification.
  amdf_xdna_device_profile_t device_profile_ = {};
  // Monitored fence exposed to the production paging wait.
  volatile uint64_t paging_fence_ = 0;
  // System-memory request for an unaligned logical byte length.
  amdf_memory_native_create_info_t create_info_ = {};
  // Profile queried from the production native provider.
  amdf_memory_native_profile_t profile_ = {};
};

TEST_F(WindowsXdnaMemoryTest, CompletesOnlyAfterMapAndOrdinaryResidency) {
  amdf_xdna_umd_memory_t* memory = nullptr;
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_TRUE(amdf_status_is_ok(amdf_xdna_umd_memory_prepare(
      &device_, &profile_, &create_info_, &memory, &result)));
  ASSERT_NE(memory, nullptr);

  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kResident,
                                    Operation::kWait}));
  EXPECT_EQ(state_.resident_flags.CantTrimFurther, 0u);
  EXPECT_EQ(state_.resident_flags.MustSucceed, 0u);
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(result.flags,
            AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS);
  EXPECT_EQ(result.byte_length, UINT64_C(65536));
  EXPECT_EQ(result.alignment, UINT64_C(4096));
  EXPECT_TRUE(amdf_physical_memory_id_is_valid(&result.physical_backing_id));
  EXPECT_EQ(result.device_address, UINT64_C(0x13000));
  EXPECT_EQ(result.address_kinds,
            (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE) |
                (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA));
  EXPECT_EQ(result.dma_address, UINT64_C(0x80013000));
  EXPECT_EQ(result.address_kinds, profile_.address_kinds);

  amdf_memory_map_info_t map_info = {};
  map_info.byte_offset = 32;
  map_info.byte_length = 4096;
  map_info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  amdf_xdna_umd_host_mapping_t* mapping = nullptr;
  amdf_xdna_umd_host_mapping_result_t map_result = {};
  ASSERT_TRUE(amdf_status_is_ok(amdf_xdna_umd_memory_map(
      memory, &profile_.host_mapping, &map_info, &mapping, &map_result)));
  ASSERT_NE(mapping, nullptr);
  EXPECT_NE(map_result.pointer, nullptr);
  EXPECT_EQ(map_result.byte_length, map_info.byte_length);
  EXPECT_EQ(map_result.visibility.cacheability,
            AMDF_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(map_result.visibility.cache_line_size, 64u);
  ASSERT_NE(profile_.visibility.describe_host, nullptr);
  const auto qualified = profile_.visibility.describe_host(
      profile_.visibility.data, AMDF_EXTERNAL_MEMORY_TYPE_NONE,
      profile_.guaranteed_flags);
  EXPECT_EQ(map_result.visibility.cacheability, qualified.cacheability);
  EXPECT_EQ(map_result.visibility.cache_line_size, qualified.cache_line_size);
  for (auto member : {&amdf_memory_host_description_t::flush,
                      &amdf_memory_host_description_t::invalidate}) {
    const auto& actual = map_result.visibility.*member;
    const auto& expected = qualified.*member;
    SCOPED_TRACE(expected.host_operation);
    EXPECT_EQ(actual.kind, expected.kind);
    EXPECT_EQ(actual.executor, expected.executor);
    EXPECT_EQ(actual.operation, expected.operation);
    EXPECT_EQ(actual.host_operation, expected.host_operation);
    EXPECT_EQ(actual.host_instruction, expected.host_instruction);
    EXPECT_EQ(actual.host_fence_before, expected.host_fence_before);
    EXPECT_EQ(actual.host_fence_after, expected.host_fence_after);
    EXPECT_EQ(actual.range_granularity, expected.range_granularity);
  }
  EXPECT_TRUE(amdf_status_is_ok(amdf_xdna_umd_host_mapping_cache_control(
      mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, map_result.byte_length)));
  EXPECT_TRUE(amdf_status_is_ok(amdf_xdna_umd_host_mapping_cache_control(
      mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
      map_result.byte_length)));
  amdf_xdna_umd_host_mapping_destroy(mapping);

  EXPECT_TRUE(amdf_status_is_ok(amdf_xdna_umd_memory_destroy(memory)));
  EXPECT_EQ(state_.operations.back(), Operation::kDestroy);
}

TEST_F(WindowsXdnaMemoryTest, ExposesExactSystemMemoryProfile) {
  EXPECT_EQ(profile_.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(profile_.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE |
                                AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(profile_.guaranteed_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(profile_.supported_device_access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(profile_.device_address.address_bit_count, 48u);
  EXPECT_EQ(profile_.device_address.minimum_address, UINT64_C(65536));
  EXPECT_EQ(profile_.device_address.maximum_address, (UINT64_C(1) << 48) - 1);
  EXPECT_EQ(profile_.allocation.maximum_byte_length,
            (UINT64_C(1) << 48) - UINT64_C(0x80000000) - UINT64_C(65536));
  EXPECT_EQ(profile_.allocation.minimum_alignment, UINT64_C(4096));
  EXPECT_EQ(profile_.allocation.maximum_alignment, UINT64_C(4096));
  EXPECT_EQ(profile_.device_address.minimum_alignment, UINT64_C(4096));
  EXPECT_EQ(profile_.allocation.native_byte_length_granularity,
            UINT64_C(65536));
  EXPECT_TRUE(state_.operations.empty());
}

TEST_F(WindowsXdnaMemoryTest, RegistersExactCallerPagesWithoutTakingOwnership) {
  ASSERT_EQ(amdf_xdna_umd_device_query_memory_profile(&device_, 1, &profile_),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile_.ordinal, 1u);
  EXPECT_EQ(profile_.roles, AMDF_MEMORY_PROFILE_ROLE_REGISTER |
                                AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(profile_.allocation.maximum_byte_length, 0u);
  EXPECT_EQ(profile_.registration.byte_length_granularity, 65536u);
  EXPECT_EQ(profile_.registration.registered_host_pointer_alignment, 65536u);
  EXPECT_EQ(profile_.registration.registered_host_cacheability,
            AMDF_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(profile_.registration.minimum_alignment, 4096u);
  EXPECT_EQ(profile_.registration.maximum_alignment, 4096u);
  EXPECT_TRUE(state_.operations.empty());

  auto* pages = static_cast<uint8_t*>(
      VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  ASSERT_NE(pages, nullptr);
  std::memset(pages, 0xA5, 65536);
  create_info_.registered_host_pointer = pages;
  create_info_.byte_length = 65536;
  amdf_xdna_umd_memory_t* memory = nullptr;
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare(&device_, &profile_, &create_info_,
                                         &memory, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.host_pointer, pages);
  EXPECT_EQ(result.physical_backing_id.words[0],
            reinterpret_cast<uintptr_t>(pages));
  EXPECT_EQ(result.physical_backing_id.words[1], 65536u);
  EXPECT_EQ(result.device_address, state_.mapped_address);
  ASSERT_EQ(amdf_xdna_umd_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(state_.metadata_free_count, 1u);

  MEMORY_BASIC_INFORMATION information = {};
  ASSERT_NE(VirtualQuery(pages, &information, sizeof(information)), 0u);
  ASSERT_EQ(information.State, MEM_COMMIT);
  EXPECT_EQ(pages[0], 0xA5);
  EXPECT_EQ(pages[65535], 0xA5);
  pages[65535] = 0x5A;
  EXPECT_TRUE(VirtualFree(pages, 0, MEM_RELEASE));
}

TEST_F(WindowsXdnaMemoryTest, RegistrationFailuresNeverReleaseCallerPages) {
  ASSERT_EQ(amdf_xdna_umd_device_query_memory_profile(&device_, 1, &profile_),
            AMDF_STATUS_OK);
  auto* pages = static_cast<uint8_t*>(
      VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  ASSERT_NE(pages, nullptr);
  create_info_.registered_host_pointer = pages;
  create_info_.byte_length = 65536;
  for (FailurePoint failure_point :
       {FailurePoint::kCreate, FailurePoint::kMap,
        FailurePoint::kInvalidMapAddress, FailurePoint::kFirstWait,
        FailurePoint::kResident, FailurePoint::kPartialResident,
        FailurePoint::kSecondWait, FailurePoint::kDestroy}) {
    SCOPED_TRACE(static_cast<int>(failure_point));
    state_ = {};
    state_.failure_point = failure_point;
    amdf_xdna_umd_memory_t* memory = nullptr;
    amdf_xdna_umd_memory_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original = result;
    const auto status = amdf_xdna_umd_memory_prepare(
        &device_, &profile_, &create_info_, &memory, &result);
    ASSERT_NE(memory, nullptr);
    if (failure_point == FailurePoint::kDestroy) {
      EXPECT_EQ(status, AMDF_STATUS_OK);
      EXPECT_FALSE(amdf_status_is_ok(amdf_xdna_umd_memory_destroy(memory)));
      amdf_xdna_umd_memory_abandon(memory);
    } else {
      EXPECT_FALSE(amdf_status_is_ok(status));
      EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
      EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory), AMDF_STATUS_OK);
    }
    EXPECT_EQ(state_.metadata_free_count, 1u);
    MEMORY_BASIC_INFORMATION information = {};
    ASSERT_NE(VirtualQuery(pages, &information, sizeof(information)), 0u);
    ASSERT_EQ(information.State, MEM_COMMIT);
    // Only native dependencies are modeled; there is no remaining real DMA
    // registration after the injected final-release failure.
    std::memset(pages, 0xA5, 65536);
  }
  EXPECT_TRUE(VirtualFree(pages, 0, MEM_RELEASE));
}

TEST_F(WindowsXdnaMemoryTest, ConstrainsAndChecksCompleteNativeAddressRanges) {
  struct RangeCase {
    // Hardware representability used to derive the native mapping bounds.
    uint32_t address_bit_count;
    // Address returned by the native mapping dependency.
    uint64_t address;
    // Requested logical length, rounded by the production allocator.
    uint64_t logical_byte_length;
    // Expected complete native extent after rounding.
    uint64_t native_byte_length;
    // Status expected before publishing the native result.
    amdf_status_code_t status_code;
  };
  const RangeCase cases[] = {
      {32, (UINT64_C(1) << 32) - 0x80000000 - 65536, 4097, 65536,
       AMDF_STATUS_CODE_OK},
      {48, (UINT64_C(1) << 48) - 0x80000000 - 65536, 4097, 65536,
       AMDF_STATUS_CODE_OK},
      {48, (UINT64_C(1) << 48) - 65536, 4097, 65536, AMDF_STATUS_CODE_INTERNAL},
      {48, 0, 4097, 65536, AMDF_STATUS_CODE_INTERNAL},
      {48, UINT64_C(1) << 48, 4097, 65536, AMDF_STATUS_CODE_INTERNAL},
      {48, (UINT64_C(1) << 48) - 0x80000000 - 65536, 65537, 131072,
       AMDF_STATUS_CODE_INTERNAL},
      {64, UINT64_MAX - 0x80000000 - 65535, 4097, 65536, AMDF_STATUS_CODE_OK},
      {64, UINT64_MAX - 0x80000000 - 65535, 65537, 131072,
       AMDF_STATUS_CODE_INTERNAL},
  };
  for (const RangeCase& test : cases) {
    SCOPED_TRACE(test.address);
    state_ = {};
    state_.address_limit = (UINT64_MAX >> (64 - test.address_bit_count)) -
                           device_profile_.dma.byte_offset + 1;
    state_.mapped_address = test.address;
    state_.byte_length = test.native_byte_length;
    device_profile_.dma.address_bit_count = test.address_bit_count;
    ASSERT_EQ(amdf_xdna_umd_device_query_memory_profile(&device_, 0, &profile_),
              AMDF_STATUS_OK);
    EXPECT_EQ(profile_.device_address.address_bit_count,
              test.address_bit_count);
    EXPECT_EQ(profile_.device_address.maximum_address,
              UINT64_MAX >> (64 - test.address_bit_count));
    create_info_.byte_length = test.logical_byte_length;
    amdf_xdna_umd_memory_t* memory = nullptr;
    amdf_xdna_umd_memory_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const amdf_xdna_umd_memory_result_t original = result;
    const amdf_status_t status = amdf_xdna_umd_memory_prepare(
        &device_, &profile_, &create_info_, &memory, &result);
    EXPECT_EQ(amdf_status_code(status), test.status_code);
    ASSERT_NE(memory, nullptr);
    if (amdf_status_is_ok(status)) {
      EXPECT_EQ(result.device_address, test.address);
      EXPECT_EQ(result.dma_address,
                test.address + device_profile_.dma.byte_offset);
      EXPECT_EQ(result.byte_length, test.native_byte_length);
    } else {
      EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
      EXPECT_EQ(state_.operations.back(), Operation::kWait);
    }
    EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory), AMDF_STATUS_OK);
    EXPECT_EQ(state_.operations.back(), Operation::kDestroy);
  }
}

TEST_F(WindowsXdnaMemoryTest, RequiresAddressGeometryBeforeAdvertisingMemory) {
  device_profile_.dma.address_bit_count = 0;
  const amdf_memory_native_profile_t original = profile_;
  EXPECT_EQ(amdf_status_code(amdf_xdna_umd_device_query_memory_profile(
                &device_, 0, &profile_)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(std::memcmp(&profile_, &original, sizeof(profile_)), 0);
  EXPECT_TRUE(state_.operations.empty());
}

TEST_F(WindowsXdnaMemoryTest,
       ReclaimsEveryFailurePrefixThroughAllocationDestruction) {
  for (FailurePoint failure_point :
       {FailurePoint::kCreate, FailurePoint::kMap,
        FailurePoint::kInvalidMapAddress, FailurePoint::kFirstWait,
        FailurePoint::kResident, FailurePoint::kPartialResident,
        FailurePoint::kSecondWait}) {
    state_ = {};
    state_.failure_point = failure_point;
    amdf_xdna_umd_memory_t* memory = nullptr;
    amdf_xdna_umd_memory_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const amdf_xdna_umd_memory_result_t original_result = result;

    const amdf_status_t status = amdf_xdna_umd_memory_prepare(
        &device_, &profile_, &create_info_, &memory, &result);

    EXPECT_FALSE(amdf_status_is_ok(status));
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    EXPECT_EQ(state_.metadata_free_count, 0u);
    EXPECT_NE(state_.operations.back(), Operation::kDestroy);
    EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory), AMDF_STATUS_OK);
    EXPECT_EQ(state_.metadata_free_count, 1u);
    switch (failure_point) {
      case FailurePoint::kCreate:
        EXPECT_EQ(state_.operations,
                  (std::vector<Operation>{Operation::kCreate}));
        break;
      case FailurePoint::kMap:
        EXPECT_EQ(state_.operations,
                  (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                          Operation::kDestroy}));
        break;
      case FailurePoint::kInvalidMapAddress:
        EXPECT_EQ(
            state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kDestroy}));
        break;
      case FailurePoint::kFirstWait:
        EXPECT_EQ(
            state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kDestroy}));
        EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1}));
        break;
      case FailurePoint::kResident:
        EXPECT_EQ(state_.operations,
                  (std::vector<Operation>{
                      Operation::kCreate, Operation::kMap, Operation::kWait,
                      Operation::kResident, Operation::kDestroy}));
        break;
      case FailurePoint::kPartialResident:
        EXPECT_EQ(
            state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kResident,
                                    Operation::kWait, Operation::kDestroy}));
        break;
      case FailurePoint::kSecondWait:
        EXPECT_EQ(
            state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kWait, Operation::kResident,
                                    Operation::kWait, Operation::kDestroy}));
        EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{1, 2}));
        break;
      case FailurePoint::kDestroy:
      case FailurePoint::kNone:
        FAIL() << "unexpected failure point";
        break;
    }
  }
}

TEST_F(WindowsXdnaMemoryTest,
       ReclaimsOwnedBackingAfterPersistentPreparationWaitFailure) {
  for (FailurePoint failure_point :
       {FailurePoint::kFirstWait, FailurePoint::kSecondWait}) {
    SCOPED_TRACE(static_cast<int>(failure_point));
    state_ = {};
    state_.failure_point = failure_point;
    amdf_xdna_umd_memory_t* memory = nullptr;
    amdf_xdna_umd_memory_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const amdf_xdna_umd_memory_result_t original_result = result;

    EXPECT_EQ(amdf_xdna_umd_memory_prepare(&device_, &profile_, &create_info_,
                                           &memory, &result),
              amdf_kmt_make_status(kStatusNoMemory));
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    EXPECT_EQ(state_.metadata_free_count, 0u);
    const auto preparation_waits = state_.wait_targets;
    EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory), AMDF_STATUS_OK);
    EXPECT_EQ(state_.metadata_free_count, 1u);
    EXPECT_EQ(state_.operations.back(), Operation::kDestroy);
    EXPECT_EQ(state_.wait_targets, preparation_waits);

    MEMORY_BASIC_INFORMATION information = {};
    ASSERT_NE(
        VirtualQuery(state_.host_pointer, &information, sizeof(information)),
        0u);
    EXPECT_EQ(information.State, MEM_FREE);
  }
}

TEST_F(WindowsXdnaMemoryTest, PreservesBackingAfterFinalDestructionFailure) {
  amdf_xdna_umd_memory_t* memory = nullptr;
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_TRUE(amdf_status_is_ok(amdf_xdna_umd_memory_prepare(
      &device_, &profile_, &create_info_, &memory, &result)));
  ASSERT_NE(memory, nullptr);

  state_.failure_point = FailurePoint::kDestroy;
  EXPECT_FALSE(amdf_status_is_ok(amdf_xdna_umd_memory_destroy(memory)));
  EXPECT_EQ(state_.metadata_free_count, 0u);
  amdf_xdna_umd_memory_abandon(memory);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  ReleaseLeakedBacking();
}

TEST_F(WindowsXdnaMemoryTest,
       ReportsFailedNativeCleanupWithoutRetainingMemory) {
  state_.failure_point = FailurePoint::kMap;
  state_.destroy_failures_remaining = 1;
  amdf_xdna_umd_memory_t* memory = nullptr;
  amdf_xdna_umd_memory_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_xdna_umd_memory_result_t original_result = result;

  EXPECT_EQ(amdf_xdna_umd_memory_prepare(&device_, &profile_, &create_info_,
                                         &memory, &result),
            amdf_kmt_make_status(kStatusNoMemory));
  ASSERT_NE(memory, nullptr);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(state_.metadata_free_count, 0u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap}));
  EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory),
            amdf_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(state_.metadata_free_count, 0u);
  amdf_xdna_umd_memory_abandon(memory);
  EXPECT_EQ(state_.metadata_free_count, 1u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreate, Operation::kMap,
                                    Operation::kDestroy}));
  ReleaseLeakedBacking();
}

TEST(WindowsXdnaMemoryPairTest, DescribesAchievedPermissionsWithoutAtomics) {
  amdf_memory_access_info_t access = {};
  const amdf_queue_family_info_t family = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA,
      .format_version = AMDF_XDNA_QUEUE_FORMAT_VERSION_1,
      .roles = AMDF_QUEUE_ROLE_COMPUTE,
  };
  amdf_memory_site_query_t query = {
      .access = access.access,
      .queue_family_info = &family,
  };
  const amdf_memory_access_t permission_sets[] = {
      AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE};
  for (amdf_memory_access_t permissions : permission_sets) {
    query.access = permissions;
    amdf_memory_site_description_t description = {};
    ASSERT_EQ(amdf_xdna_umd_memory_describe_site(&query, &description),
              AMDF_STATUS_OK);
    amdf_memory_site_capabilities_t expected = 0;
    if (permissions & AMDF_MEMORY_ACCESS_READ) {
      expected |= AMDF_MEMORY_SITE_CAPABILITY_READ;
    }
    if (permissions & AMDF_MEMORY_ACCESS_WRITE) {
      expected |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
    }
    EXPECT_EQ(description.capabilities, expected);
    EXPECT_EQ(description.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    EXPECT_EQ(description.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    EXPECT_FALSE(
        amdf_memory_compatibility_domain_is_valid(&description.atomic_domain));
    EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
  }
}

TEST(WindowsXdnaMemoryPairTest, RejectsUnqualifiedFamiliesWithoutOutput) {
  const amdf_memory_access_info_t access = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
  };
  const amdf_queue_family_info_t families[] = {
      {.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
       .format_version = AMDF_XDNA_QUEUE_FORMAT_VERSION_1,
       .roles = AMDF_QUEUE_ROLE_COMPUTE},
      {.command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA,
       .format_version = AMDF_XDNA_QUEUE_FORMAT_VERSION_1 + 1,
       .roles = AMDF_QUEUE_ROLE_COMPUTE},
      {.command_type = AMDF_QUEUE_COMMAND_TYPE_XDNA,
       .format_version = AMDF_XDNA_QUEUE_FORMAT_VERSION_1,
       .roles = AMDF_QUEUE_ROLE_TRANSFER},
  };
  for (const auto& family : families) {
    amdf_memory_site_query_t query = {
        .access = access.access,
        .queue_family_info = &family,
    };
    amdf_memory_site_description_t description;
    std::memset(&description, 0xA5, sizeof(description));
    const auto original = description;
    EXPECT_EQ(amdf_xdna_umd_memory_describe_site(&query, &description),
              amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(std::memcmp(&description, &original, sizeof(description)), 0);
  }
}

}  // namespace
