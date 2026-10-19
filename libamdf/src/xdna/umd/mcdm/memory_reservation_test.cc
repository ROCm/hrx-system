// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/memory_reservation.h"

#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/memory_resource.h"
#include "libamdf/src/xdna/umd/mcdm/memory.h"

namespace {

constexpr NTSTATUS kNoMemory = static_cast<NTSTATUS>(0xC0000017u);
constexpr uint64_t kUnit = 4096;
constexpr uint64_t kAddress = 0x40000;
constexpr uint64_t kSourceByteLength = 3 * 65536;

struct Allocation {
  // Borrowed source pages named in native registration.
  const void* pointer;
  // Extent of the native registration.
  uint64_t byte_length;
  // Installed device address, zero until mapped.
  uint64_t address;
  // Whether the modeled native owner has consumed this handle.
  bool destroyed;
};

struct NativeState {
  // Native registrations; handles are one-based indices.
  std::vector<Allocation> allocations;
  // Destruction batches seen by the native dependency.
  std::vector<std::vector<D3DKMT_HANDLE>> destroys;
  // Residency batches seen by the native dependency.
  std::vector<std::vector<D3DKMT_HANDLE>> residents;
  // Paging points whose completion was requested.
  std::vector<uint64_t> waits;
  // Sequential paging point returned by map/residency.
  uint64_t point = 0;
  // Calls to release the explicitly reserved address range.
  uint32_t free_count = 0;
  // Exact accepted native reservation extent, including any alignment slack.
  uint64_t reserved_byte_length = 0;
  // One-based registration call to reject, or zero for none.
  size_t reject_create = 0;
  // One-based registration whose mapping is rejected, or zero for none.
  size_t reject_map = 0;
  // Native residency result injected after maps have been accepted.
  NTSTATUS residency_status = 0;
  // Native destroy failure after consuming the first handle of its batch.
  NTSTATUS destroy_status = 0;
  // Native execution state returned after paging retirement.
  D3DKMT_DEVICEEXECUTION_STATE execution_state = D3DKMT_DEVICEEXECUTION_ACTIVE;
};

NativeState* native = nullptr;

NTSTATUS APIENTRY CreateAllocation(D3DKMT_CREATEALLOCATION* create) {
  if (native->allocations.size() + 1 == native->reject_create) {
    return kNoMemory;
  }
  EXPECT_EQ(create->Flags.CreateResource, 0u);
  EXPECT_EQ(create->NumAllocations, 1u);
  native->allocations.push_back(
      {create->pAllocationInfo2->pSystemMem,
       create->pStandardAllocation->ExistingHeapData.Size, 0, false});
  create->pAllocationInfo2->hAllocation =
      static_cast<D3DKMT_HANDLE>(native->allocations.size());
  return 0;
}

NTSTATUS APIENTRY DestroyAllocation(const D3DKMT_DESTROYALLOCATION2* destroy) {
  EXPECT_EQ(destroy->hResource, 0u);
  EXPECT_EQ(destroy->Flags.AssumeNotInUse, 1u);
  EXPECT_EQ(destroy->Flags.SynchronousDestroy, 1u);
  native->destroys.emplace_back(
      destroy->phAllocationList,
      destroy->phAllocationList + destroy->AllocationCount);
  for (D3DKMT_HANDLE handle : native->destroys.back()) {
    auto& allocation = native->allocations.at(handle - 1);
    EXPECT_FALSE(allocation.destroyed) << "native handle was retried";
    allocation.destroyed = true;
    if (native->destroy_status != 0) {
      return native->destroy_status;
    }
  }
  return 0;
}

NTSTATUS APIENTRY Map(D3DDDI_MAPGPUVIRTUALADDRESS* map) {
  if (map->hAllocation == native->reject_map) {
    return kNoMemory;
  }
  auto& allocation = native->allocations.at(map->hAllocation - 1);
  EXPECT_FALSE(allocation.destroyed);
  EXPECT_EQ(map->SizeInPages * kUnit, allocation.byte_length);
  EXPECT_EQ(map->OffsetInPages, 0u);
  EXPECT_EQ(map->Protection.NoAccess, 0u);
  map->VirtualAddress = map->BaseAddress != 0 ? map->BaseAddress : 0x10000;
  allocation.address = map->VirtualAddress;
  map->PagingFenceValue = ++native->point;
  return 0x103;
}

NTSTATUS APIENTRY MakeResident(D3DDDI_MAKERESIDENT* resident) {
  native->residents.emplace_back(
      resident->AllocationList,
      resident->AllocationList + resident->NumAllocations);
  resident->PagingFenceValue = ++native->point;
  return native->residency_status != 0 ? native->residency_status : 0x103;
}

class WindowsXdnaReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native = &state_;
    kmt_.create_allocation = CreateAllocation;
    kmt_.destroy_allocation = DestroyAllocation;
    kmt_.map_gpu_virtual_address = Map;
    kmt_.make_resident = MakeResident;
    kmt_.reserve_gpu_virtual_address =
        [](D3DDDI_RESERVEGPUVIRTUALADDRESS* reserve) -> NTSTATUS {
      EXPECT_LE(reserve->MinimumAddress, kAddress);
      EXPECT_GE(reserve->MaximumAddress, kAddress + reserve->Size);
      native->reserved_byte_length = reserve->Size;
      reserve->VirtualAddress = kAddress;
      return 0;
    };
    kmt_.free_gpu_virtual_address =
        [](const D3DKMT_FREEGPUVIRTUALADDRESS* release) -> NTSTATUS {
      EXPECT_EQ(release->BaseAddress, kAddress);
      EXPECT_EQ(release->Size, native->reserved_byte_length);
      ++native->free_count;
      return 0;
    };
    kmt_.wait_from_cpu =
        [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
      native->waits.push_back(wait->FenceValueArray[0]);
      return 0;
    };
    kmt_.get_device_state = [](D3DKMT_GETDEVICESTATE* query) -> NTSTATUS {
      query->ExecutionState = native->execution_state;
      return 0;
    };
    target_.dma.address_bit_count = 48;
    target_.dma.byte_offset = 0x80000000;
    device_.host_allocator = amdf_allocator_system();
    device_.kmt = &kmt_;
    device_.profile = &target_;
    device_.device = 1;
    device_.adapter = 2;
    device_.paging_queue = 3;
    device_.paging_sync_object = 4;
    device_.paging_fence = &fence_;
    base_.host_allocator = amdf_allocator_system();
    base_.access_count = 1;
    base_.accesses = &access_;
    ASSERT_EQ(amdf_xdna_umd_device_query_memory_reservation_profile(&device_, 0,
                                                                    &profile_),
              AMDF_STATUS_OK);
    access_.vtable = profile_.vtable;

    // Use the real ordinary registration owner with valid borrowed pages.
    pages_ = VirtualAlloc(nullptr, kSourceByteLength, MEM_RESERVE | MEM_COMMIT,
                          PAGE_READWRITE);
    ASSERT_NE(pages_, nullptr);
    amdf_memory_native_profile_t memory_profile = {};
    ASSERT_EQ(
        amdf_xdna_umd_device_query_memory_profile(&device_, 1, &memory_profile),
        AMDF_STATUS_OK);
    const amdf_memory_native_create_info_t create = {
        .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .byte_length = kSourceByteLength,
        .registered_host_pointer = pages_,
    };
    amdf_xdna_umd_memory_result_t result = {};
    ASSERT_EQ(amdf_xdna_umd_memory_prepare(&device_, &memory_profile, &create,
                                           &source_, &result),
              AMDF_STATUS_OK);
    source_access_.native = source_;
    memory_.accesses = &source_access_;
    state_.residents.clear();
    state_.waits.clear();
  }

  void TearDown() override {
    if (access_.native != nullptr) {
      EXPECT_EQ(profile_.vtable->destroy(&base_, 0), AMDF_STATUS_OK);
    }
    // Dependency failures model uncertain consumption without real native DMA.
    state_.destroy_status = 0;
    if (source_ != nullptr) {
      EXPECT_EQ(amdf_xdna_umd_memory_destroy(source_), AMDF_STATUS_OK);
    }
    if (pages_ != nullptr) {
      EXPECT_TRUE(VirtualFree(pages_, 0, MEM_RELEASE));
    }
    native = nullptr;
  }

  void Reserve(uint64_t byte_length = 65536, uint64_t unit = kUnit) {
    const amdf_memory_reservation_access_t consumer = {
        .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .mapping_granularity = unit,
    };
    const amdf_memory_reservation_create_info_t create = {
        .byte_length = byte_length,
        .access_count = 1,
        .accesses = &consumer,
    };
    const uint32_t ordinal = 0;
    const amdf_memory_reservation_group_t group = {1, &ordinal, &profile_};
    ASSERT_EQ(profile_.vtable->prepare(&base_, &create, &group),
              AMDF_STATUS_OK);
    EXPECT_EQ(access_.info.address, kAddress + target_.dma.byte_offset);
    base_.host_allocator.allocate = [](void*, uint64_t, uint64_t) -> void* {
      ADD_FAILURE() << "native reservation handle storage must be pre-sized";
      return nullptr;
    };
  }

  amdf_memory_mapping_update_t Update(uint64_t target, uint64_t source,
                                      uint64_t count = 1) {
    return {
        .target_byte_offset = target * kUnit,
        .byte_length = count * kUnit,
        .memory = &memory_,
        .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .source_byte_offset = source * kUnit,
    };
  }

  amdf_status_t Apply(
      std::initializer_list<amdf_memory_mapping_update_t> updates) {
    bool mutated = false;
    const auto status = profile_.vtable->update(
        &base_, 0, static_cast<uint32_t>(updates.size()), updates.begin(),
        &mutated);
    EXPECT_TRUE(mutated);
    return status;
  }

  // Modeled KMT dependency and its observable native ownership.
  NativeState state_;
  // Native procedure table; no real driver calls occur.
  amdf_kmt_api_t kmt_ = {};
  // Device address geometry supplied to the production profile query.
  amdf_xdna_device_profile_t target_ = {};
  // Live native device borrowed by the source and reservation.
  amdf_xdna_umd_device_t device_ = {};
  // Modeled native paging progress; waits observe all new points explicitly.
  volatile uint64_t fence_ = 0;
  // Production reservation operations and construction profile.
  amdf_memory_reservation_profile_t profile_ = {};
  // Common owner containing the actual native reservation under test.
  amdf_memory_reservation_t base_ = {};
  // Native owner slot in the common reservation.
  amdf_memory_reservation_access_state_t access_ = {};
  // Common source whose native state was prepared by the real memory provider.
  amdf_memory_t memory_ = {};
  // Source access owning the ordinary native registration.
  amdf_memory_access_state_t source_access_ = {};
  // Ordinary source registration retained through all view changes.
  amdf_xdna_umd_memory_t* source_ = nullptr;
  // Real host pages borrowed by the modeled registrations.
  void* pages_ = nullptr;
};

TEST_F(WindowsXdnaReservationTest, ReplacesFullUnitsAndExactShortTail) {
  ASSERT_NO_FATAL_FAILURE(Reserve(3 * 65536, 2 * 65536));
  ASSERT_EQ(Apply({Update(0, 0, 48)}), AMDF_STATUS_OK);
  ASSERT_EQ(state_.allocations.size(), 3u);
  EXPECT_EQ(state_.allocations[1].byte_length, 2 * 65536u);
  EXPECT_EQ(state_.allocations[2].byte_length, 65536u);
  EXPECT_EQ(state_.allocations[2].address, kAddress + 2 * 65536);
  EXPECT_EQ(state_.residents.back(), (std::vector<D3DKMT_HANDLE>{2, 3}));
  EXPECT_EQ(state_.waits, (std::vector<uint64_t>{state_.point}));
  ASSERT_EQ(Apply({Update(32, 0, 16)}), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.back(), (std::vector<D3DKMT_HANDLE>{3}));
  EXPECT_FALSE(state_.allocations[1].destroyed);
  EXPECT_EQ(state_.allocations.back().byte_length, 65536u);
  EXPECT_EQ(state_.allocations.back().address, kAddress + 2 * 65536);
  ASSERT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.back(), (std::vector<D3DKMT_HANDLE>{2, 4}));
}

TEST_F(WindowsXdnaReservationTest, ReservesLessThanOneMappingUnit) {
  ASSERT_NO_FATAL_FAILURE(Reserve(65536, 2 * 65536));
  ASSERT_EQ(Apply({Update(0, 0, 16)}), AMDF_STATUS_OK);
  ASSERT_EQ(state_.allocations.size(), 2u);
  EXPECT_EQ(state_.allocations.back().byte_length, 65536u);
  EXPECT_EQ(state_.residents.back(), (std::vector<D3DKMT_HANDLE>{2}));
  ASSERT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.back(), (std::vector<D3DKMT_HANDLE>{2}));
}

TEST_F(WindowsXdnaReservationTest, FailedTailCreationRetainsFullPrefix) {
  ASSERT_NO_FATAL_FAILURE(Reserve(3 * 65536, 2 * 65536));
  state_.reject_create = 3;
  EXPECT_EQ(Apply({Update(0, 0, 48)}), amdf_kmt_make_status(kNoMemory));
  EXPECT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.back(), (std::vector<D3DKMT_HANDLE>{2}));
}

TEST_F(WindowsXdnaReservationTest,
       ReplacesOnlyAffectedViewsAndBatchesReadiness) {
  ASSERT_NO_FATAL_FAILURE(Reserve());
  ASSERT_EQ(Apply({Update(1, 2, 3)}), AMDF_STATUS_OK);
  ASSERT_EQ(state_.residents.size(), 1u);
  EXPECT_EQ(state_.residents[0], (std::vector<D3DKMT_HANDLE>{2, 3, 4}));
  EXPECT_EQ(state_.waits, (std::vector<uint64_t>{state_.point}));
  ASSERT_EQ(Apply({Update(1, 5), Update(3, 7)}), AMDF_STATUS_OK);
  ASSERT_EQ(state_.destroys.size(), 1u);
  EXPECT_EQ(state_.destroys[0], (std::vector<D3DKMT_HANDLE>{2, 4}));
  EXPECT_FALSE(state_.allocations[0].destroyed);  // Canonical source.
  EXPECT_FALSE(state_.allocations[2].destroyed);  // Untouched neighbor.
  EXPECT_EQ(state_.allocations[4].pointer,
            static_cast<uint8_t*>(pages_) + 5 * kUnit);
  EXPECT_EQ(state_.allocations[4].address, kAddress + kUnit);
  EXPECT_EQ(state_.allocations[5].address, kAddress + 3 * kUnit);
  EXPECT_EQ(state_.residents.back(), (std::vector<D3DKMT_HANDLE>{5, 6}));
  ASSERT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.back(), (std::vector<D3DKMT_HANDLE>{5, 3, 6}));
  const auto count = state_.destroys.size();
  EXPECT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.size(), count);
  EXPECT_EQ(state_.free_count, 0u);
  EXPECT_EQ(Apply({Update(1, 9)}), AMDF_STATUS_OK);
  EXPECT_EQ(state_.allocations.back().address, kAddress + kUnit);
}

TEST_F(WindowsXdnaReservationTest,
       FailedCreationLeavesItsSuccessfulPrefixOwned) {
  ASSERT_NO_FATAL_FAILURE(Reserve());
  state_.reject_create = 3;
  EXPECT_EQ(Apply({Update(1, 2, 3)}), amdf_kmt_make_status(kNoMemory));
  EXPECT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.back(), (std::vector<D3DKMT_HANDLE>{2}));
}

TEST_F(WindowsXdnaReservationTest,
       FailedMappingStillOwnsTheUnmappedRegistration) {
  ASSERT_NO_FATAL_FAILURE(Reserve());
  state_.reject_map = 3;
  EXPECT_EQ(Apply({Update(1, 2, 3)}), amdf_kmt_make_status(kNoMemory));
  EXPECT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.back(), (std::vector<D3DKMT_HANDLE>{2, 3}));
}

TEST_F(WindowsXdnaReservationTest, FailedResidencyIsFullyRemovable) {
  ASSERT_NO_FATAL_FAILURE(Reserve());
  state_.residency_status = kNoMemory;
  EXPECT_EQ(Apply({Update(1, 2, 3)}), amdf_kmt_make_status(kNoMemory));
  EXPECT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroys.back(), (std::vector<D3DKMT_HANDLE>{2, 3, 4}));
}

TEST_F(WindowsXdnaReservationTest,
       RetiredPagingWithDeviceFailureIsNotReadiness) {
  ASSERT_NO_FATAL_FAILURE(Reserve());
  state_.execution_state = D3DKMT_DEVICEEXECUTION_STOPPED;
  EXPECT_EQ(Apply({Update(1, 2, 3)}),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  EXPECT_EQ(profile_.vtable->detach(&base_, 0), AMDF_STATUS_OK);
}

TEST_F(WindowsXdnaReservationTest,
       FailedRemovalNeverRetriesConsumedHandlesOrFreesAddresses) {
  ASSERT_NO_FATAL_FAILURE(Reserve());
  ASSERT_EQ(Apply({Update(1, 2, 3)}), AMDF_STATUS_OK);
  state_.destroy_status = kNoMemory;
  EXPECT_EQ(Apply({Update(1, 5, 3)}), amdf_kmt_make_status(kNoMemory));
  EXPECT_EQ(profile_.vtable->detach(&base_, 0),
            amdf_kmt_make_status(kNoMemory));
  EXPECT_EQ(profile_.vtable->destroy(&base_, 0),
            amdf_kmt_make_status(kNoMemory));
  access_.native = nullptr;
  EXPECT_EQ(state_.destroys.size(), 1u);
  EXPECT_EQ(state_.allocations.size(), 4u);
  EXPECT_EQ(state_.free_count, 0u);
}

}  // namespace
