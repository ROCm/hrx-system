// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/pm4/mapping_fixture.h"

namespace {

class Pm4MappingTest : public Pm4MappingFixture {
 protected:
  void RunShortExtent(amdf_memory_scope_t* scope) {
    ASSERT_NO_FATAL_FAILURE(Prepare(scope));
    if (IsSkipped()) {
      return;
    }
    const uint64_t page = reservation_info_.mapping_granularity;
    ASSERT_LT(kPayloadByteOffset, page);
    ASSERT_LE(page, kSlabByteLength);
    ASSERT_NO_FATAL_FAILURE(ReleaseReservation(&reservation_));
    const amdf_memory_reservation_access_t access = {device_, 0, access_, page};
    const amdf_memory_reservation_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
        .structure_size = sizeof(create),
        .byte_length = 2 * kSlabByteLength + page,
        .minimum_alignment = page,
        .access_count = 1,
        .accesses = &access,
    };
    ASSERT_EQ(
        api_->memory_reservation_create(instance_, &create, &reservation_),
        AMDF_STATUS_OK);
    ASSERT_EQ(api_->memory_reservation_query_access_info(reservation_, 0,
                                                         &reservation_info_),
              AMDF_STATUS_OK);
    ASSERT_EQ(reservation_info_.byte_length, create.byte_length);
    ASSERT_EQ(reservation_info_.mapping_granularity, page);
    RecordProperty("mapping_reserved_byte_length",
                   std::to_string(reservation_info_.byte_length));
    RecordProperty("mapping_base", std::to_string(reservation_info_.address));

    GpuMemory* neighbor = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateBacking(kSlabByteLength, &neighbor));
    const amdf_memory_mapping_update_t first = {
        .byte_length = kSlabByteLength,
        .memory = neighbor->memory,
        .access = access_,
    };
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &first),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(WriteAndVerify(0, 4096, 37, *neighbor));
    std::array<GpuMemory*, 2> tails = {};
    for (auto& tail : tails) {
      ASSERT_NO_FATAL_FAILURE(CreateBacking(page, &tail));
    }
    amdf_memory_mapping_update_t last = {
        .target_byte_offset = 2 * kSlabByteLength,
        .byte_length = page,
        .access = access_,
    };
    for (uint32_t generation = 0; generation < tails.size(); ++generation) {
      last.memory = tails[generation]->memory;
      ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &last),
                AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(
          WriteAndVerify(last.target_byte_offset, page - kPayloadByteOffset,
                         generation + 1, *tails[generation]));
      for (uint32_t previous = 0; previous < generation; ++previous) {
        ASSERT_NO_FATAL_FAILURE(
            Verify(tails[previous]->device_address + kPayloadByteOffset,
                   page - kPayloadByteOffset, previous + 1));
      }
    }
    // Clearing the exact terminal page must preserve the distant mapping and
    // cannot consume physical padding beyond the requested reservation extent.
    last.memory = nullptr;
    last.access = 0;
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &last),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(
        Verify(reservation_info_.address + kPayloadByteOffset, 4096, 37));
    ASSERT_NO_FATAL_FAILURE(ReleaseReservation(&reservation_));
    for (auto* tail : tails) {
      ASSERT_NO_FATAL_FAILURE(ReleaseBacking(tail));
    }
    ASSERT_NO_FATAL_FAILURE(ReleaseBacking(neighbor));
  }

  void RunSplit(amdf_memory_scope_t* scope) {
    ASSERT_NO_FATAL_FAILURE(Prepare(scope));
    if (IsSkipped()) {
      return;
    }
    GpuMemory* original = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateBacking(9 * kSlabByteLength, &original));
    amdf_memory_mapping_update_t first = {
        .target_byte_offset = kPlacementByteOffset,
        .byte_length = 9 * kSlabByteLength,
        .memory = original->memory,
        .access = access_,
    };
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &first),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(WriteAndVerify(
        kPlacementByteOffset, kLargePayloadByteLength, 100, *original));
    const auto* captured =
        static_cast<const uint32_t*>(readback_->host.pointer);
    const std::vector<uint32_t> capture(
        captured, captured + kLargePayloadByteLength / sizeof(uint32_t));
    first.memory = nullptr;
    first.access = 0;
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &first),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(ReleaseBacking(original));

    constexpr std::array<uint64_t, 9> kPayloadLengths = {
        128, 4096, 4096, 4096, 4096, 128, 128, 4096, 4096};
    constexpr std::array<uint32_t, 3> kPhaseEnds = {3, 5, 9};
    std::array<GpuMemory*, 9> slabs = {};
    uint32_t begin = 0;
    for (const uint32_t end : kPhaseEnds) {
      std::array<amdf_memory_mapping_update_t, 4> updates = {};
      for (uint32_t i = begin; i < end; ++i) {
        ASSERT_NO_FATAL_FAILURE(CreateBacking(kSlabByteLength, &slabs[i]));
        updates[i - begin] = {
            .target_byte_offset = kPlacementByteOffset + i * kSlabByteLength,
            .byte_length = kSlabByteLength,
            .memory = slabs[i]->memory,
            .access = access_,
        };
      }
      ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, end - begin,
                                                updates.data()),
                AMDF_STATUS_OK);
      for (uint32_t i = begin; i < end; ++i) {
        ASSERT_NO_FATAL_FAILURE(
            WriteAndVerify(kPlacementByteOffset + i * kSlabByteLength,
                           kPayloadLengths[i], i + 1, *slabs[i]));
      }
      begin = end;
    }
    // Every owner remains mapped while later owners are written. Check them
    // again together, not just each newly mapped range in isolation.
    for (uint32_t i = 0; i < slabs.size(); ++i) {
      ASSERT_NO_FATAL_FAILURE(
          Verify(reservation_info_.address + kPlacementByteOffset +
                     i * kSlabByteLength + kPayloadByteOffset,
                 kPayloadLengths[i], i + 1));
    }
    for (uint64_t i = 0; i < capture.size(); ++i) {
      ASSERT_EQ(capture[i], TaggedWord(100, i));
    }
    constexpr std::array<uint32_t, 9> kReleaseOrder = {0, 2, 1, 6, 5,
                                                       4, 3, 7, 8};
    for (const uint32_t i : kReleaseOrder) {
      const amdf_memory_mapping_update_t detach = {
          .target_byte_offset = kPlacementByteOffset + i * kSlabByteLength,
          .byte_length = kSlabByteLength,
      };
      ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &detach),
                AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(ReleaseBacking(slabs[i]));
    }
    RecordProperty("mapping_verified_split_owners", slabs.size());
  }

  void RunReplacement(amdf_memory_scope_t* scope) {
    ASSERT_NO_FATAL_FAILURE(Prepare(scope));
    if (IsSkipped()) {
      return;
    }
    std::array<GpuMemory*, 3> slabs = {};
    for (auto& slab : slabs) {
      ASSERT_NO_FATAL_FAILURE(CreateBacking(kSlabByteLength, &slab));
    }
    // Descending descriptors also exercise admission and sorting of an ordinary
    // caller vector. Replacement requires no knowledge of the previous owners.
    constexpr uint64_t kHighOffset = (UINT64_C(5) << 30) + kSlabByteLength;
    for (uint32_t generation = 0; generation < slabs.size(); ++generation) {
      std::array<amdf_memory_mapping_update_t, 3> updates = {};
      for (uint32_t i = 0; i < updates.size(); ++i) {
        const uint32_t position = 2 - i;
        updates[i] = {
            .target_byte_offset = kHighOffset + position * kSlabByteLength,
            .byte_length = kSlabByteLength,
            .memory = slabs[(position + generation) % slabs.size()]->memory,
            .access = access_,
        };
      }
      ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, updates.size(),
                                                updates.data()),
                AMDF_STATUS_OK);
      for (uint32_t i = 0; i < slabs.size(); ++i) {
        ASSERT_NO_FATAL_FAILURE(WriteAndVerify(
            kHighOffset + i * kSlabByteLength, 4096, 10 * generation + i + 1,
            *slabs[(i + generation) % slabs.size()]));
      }
      // Reject the entire invalid vector before mutation, including an earlier
      // otherwise valid replacement. The installed generation must survive.
      updates[1].target_byte_offset = updates[0].target_byte_offset;
      updates[0].memory = slabs[generation]->memory;
      ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, updates.size(),
                                                updates.data()),
                amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
      for (uint32_t i = 0; i < slabs.size(); ++i) {
        ASSERT_NO_FATAL_FAILURE(Verify(reservation_info_.address + kHighOffset +
                                           i * kSlabByteLength +
                                           kPayloadByteOffset,
                                       4096, 10 * generation + i + 1));
      }
    }
    // Final release clears live mappings and holes while the same queue is
    // still alive. Backing is released before queue teardown can help.
    ASSERT_NO_FATAL_FAILURE(ReleaseReservation(&reservation_));
    for (auto* slab : slabs) {
      ASSERT_NO_FATAL_FAILURE(ReleaseBacking(slab));
    }
    RecordProperty("mapping_verified_retained_generations", 3);
  }

  void RunDetachReuse(amdf_memory_scope_t* scope) {
    ASSERT_NO_FATAL_FAILURE(Prepare(scope));
    if (IsSkipped()) {
      return;
    }
    std::array<GpuMemory*, 2> slabs = {};
    for (auto& slab : slabs) {
      ASSERT_NO_FATAL_FAILURE(CreateBacking(kSlabByteLength, &slab));
    }
    constexpr std::array<uint64_t, 2> kOffsets = {
        kPlacementByteOffset, (UINT64_C(5) << 30) + kSlabByteLength};
    std::array<amdf_memory_mapping_update_t, 2> updates = {};
    for (uint32_t i = 0; i < slabs.size(); ++i) {
      updates[i] = {
          .target_byte_offset = kOffsets[i],
          .byte_length = kSlabByteLength,
          .memory = slabs[i]->memory,
          .access = access_,
      };
    }
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, updates.size(),
                                              updates.data()),
              AMDF_STATUS_OK);
    for (uint32_t i = 0; i < slabs.size(); ++i) {
      ASSERT_NO_FATAL_FAILURE(
          WriteAndVerify(kOffsets[i], 4096, i + 1, *slabs[i]));
    }
    ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
    amdf_memory_reservation_access_info_t detached = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
        .structure_size = sizeof(detached),
    };
    ASSERT_EQ(
        api_->memory_reservation_query_access_info(reservation_, 0, &detached),
        AMDF_STATUS_OK);
    EXPECT_EQ(detached.address, reservation_info_.address);
    EXPECT_EQ(detached.byte_length, reservation_info_.byte_length);
    // Reclaim one slab while both the reservation and original queue remain
    // live. Recycle the other into its neighbor's former address, then use it
    // immediately without any allocation or queue recreation supplying a flush.
    ASSERT_NO_FATAL_FAILURE(ReleaseBacking(slabs[0]));
    updates[0].memory = slabs[1]->memory;
    ASSERT_EQ(
        api_->memory_reservation_update(reservation_, 0, 1, updates.data()),
        AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(WriteAndVerify(kOffsets[0], 4096, 3, *slabs[1]));
    ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(ReleaseBacking(slabs[1]));
    // Repeated full detach also covers an entirely empty, still-live group.
    ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
  }

  void RunSourceSlices(amdf_memory_scope_t* scope) {
    ASSERT_NO_FATAL_FAILURE(Prepare(scope));
    if (IsSkipped()) {
      return;
    }
    GpuMemory* backing = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateBacking(kSlabByteLength, &backing));
    const uint64_t page = reservation_info_.mapping_granularity;
    const amdf_memory_mapping_update_t aliases[] = {
        {
            .target_byte_offset = kPlacementByteOffset,
            .byte_length = 2 * page,
            .memory = backing->memory,
            .access = access_,
            .source_byte_offset = page,
        },
        {
            .target_byte_offset = kPlacementByteOffset + kSlabByteLength,
            .byte_length = 2 * page,
            .memory = backing->memory,
            .access = AMDF_MEMORY_ACCESS_READ | required_access_,
            .source_byte_offset = page,
        },
    };
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 2, aliases),
              AMDF_STATUS_OK);
    constexpr uint64_t kPayloadLength = 4096;
    auto* words = static_cast<uint32_t*>(upload_->host.pointer);
    for (uint64_t i = 0; i < kPayloadLength / sizeof(uint32_t); ++i) {
      words[i] = TaggedWord(37, i);
    }
    ASSERT_NO_FATAL_FAILURE(Copy(upload_->device_address,
                                 reservation_info_.address +
                                     aliases[0].target_byte_offset +
                                     kPayloadByteOffset,
                                 kPayloadLength));
    ASSERT_NO_FATAL_FAILURE(Verify(reservation_info_.address +
                                       aliases[1].target_byte_offset +
                                       kPayloadByteOffset,
                                   kPayloadLength, 37));
    ASSERT_NO_FATAL_FAILURE(
        Verify(backing->device_address + page + kPayloadByteOffset,
               kPayloadLength, 37));
    const amdf_memory_mapping_update_t detach = {
        .target_byte_offset = aliases[0].target_byte_offset,
        .byte_length = aliases[0].byte_length,
    };
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &detach),
              AMDF_STATUS_OK);
    // Removing one alias does not remove the source or its other projection.
    ASSERT_NO_FATAL_FAILURE(Verify(reservation_info_.address +
                                       aliases[1].target_byte_offset +
                                       kPayloadByteOffset,
                                   kPayloadLength, 37));
  }

  void RunSharedDomain(amdf_memory_scope_t* scope) {
    // This case deliberately owns a second logical device: its paging owner
    // and allocation handles must work with the first device's existing queue.
    amdf_gpu_device_create_info_t device_create = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
        .structure_size = sizeof(device_create),
    };
    ASSERT_EQ(
        gpu_api_->device_create(endpoint_, &device_create, &second_device_),
        AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(Prepare(scope, 65536, kSlabByteLength));
    if (IsSkipped()) {
      return;
    }
    amdf_memory_reservation_access_info_t second = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
        .structure_size = sizeof(second),
    };
    ASSERT_EQ(
        api_->memory_reservation_query_access_info(reservation_, 2, &second),
        AMDF_STATUS_OK);
    ASSERT_EQ(second.mapping_access_ordinal,
              reservation_info_.mapping_access_ordinal);
    ASSERT_EQ(second.address, reservation_info_.address);
    ASSERT_EQ(second.mapping_granularity, kSlabByteLength);
    ASSERT_EQ(reservation_info_.mapping_granularity, kSlabByteLength);
    ASSERT_EQ(api_->memory_reservation_detach(reservation_, 1), AMDF_STATUS_OK);

    GpuMemory* primary_backing = nullptr;
    GpuMemory* secondary_backing = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateBacking(kSlabByteLength, &primary_backing));
    ASSERT_NO_FATAL_FAILURE(
        CreateBacking(kSlabByteLength, &secondary_backing, second_device_));
    const uint64_t offset = (UINT64_C(4) << 30) + kSlabByteLength;
    amdf_memory_mapping_update_t update = {
        .target_byte_offset = offset,
        .byte_length = kSlabByteLength,
        .memory = secondary_backing->memory,
        .access = access_,
    };
    // The first GPU action after publication executes on the other logical
    // device. No allocation or diagnostic submission supplies readiness.
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 2, 1, &update),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(
        WriteAndVerify(offset, 4096, 1, *secondary_backing));
    update.memory = primary_backing->memory;
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &update),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(WriteAndVerify(offset, 4096, 2, *primary_backing));
    ASSERT_NO_FATAL_FAILURE(Verify(
        secondary_backing->device_address + kPayloadByteOffset, 4096, 1));
    ASSERT_EQ(api_->memory_reservation_detach(reservation_, 2), AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(ReleaseBacking(secondary_backing));
    ASSERT_NO_FATAL_FAILURE(ReleaseBacking(primary_backing));
  }

  void RunRegisteredSource() {
    ASSERT_NO_FATAL_FAILURE(Prepare(system_scope_));
    if (IsSkipped()) {
      return;
    }
    const amdf_memory_device_access_t attachment = {
        device_,
        {
            .access = access_,
            .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                     AMDF_MEMORY_FLAG_MAPPING_SOURCE |
                     AMDF_MEMORY_FLAG_HOST_COHERENT,
        },
    };
    amdf_memory_scope_info_t scope_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(scope_info),
    };
    ASSERT_EQ(api_->memory_scope_query_info(scope_, &scope_info),
              AMDF_STATUS_OK);
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    uint32_t ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t i = 0; i < scope_info.memory_profile_count; ++i) {
      const amdf_status_t status = api_->memory_scope_query_device_profile(
          scope_, i, 1, &attachment, &profile, &capabilities);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      constexpr auto kRoles =
          AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
      if ((profile.roles & kRoles) == kRoles &&
          (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) {
        ordinal = i;
        break;
      }
    }
    if (ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
      GTEST_SKIP() << "registered pages are not an advertised mapping source";
    }
    const uint64_t granularity =
        std::max(reservation_info_.mapping_granularity,
                 profile.registration.native_byte_length_granularity);
    const uint64_t alignment =
        std::max({granularity, profile.registration.minimum_alignment,
                  profile.registration.registered_host_pointer_alignment});
    ASSERT_GE(granularity, 4096u);
    ASSERT_LE(granularity, kLargePayloadByteLength);
    const uint64_t byte_length = 4 * granularity;
    ASSERT_LE(byte_length, profile.registration.maximum_byte_length);
    std::array<GpuMemory*, 2> sources = {};
    for (auto& source : sources) {
      void* storage = std::malloc(byte_length + alignment - 1);
      ASSERT_NE(storage, nullptr);
      const uintptr_t pointer =
          (reinterpret_cast<uintptr_t>(storage) + alignment - 1) &
          ~(alignment - 1);
      std::memset(reinterpret_cast<void*>(pointer), 0, byte_length);
      amdf_memory_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
          .structure_size = sizeof(create),
          .memory_profile_ordinal = ordinal,
          .access_count = 1,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .byte_length = byte_length,
          .minimum_alignment = alignment,
          .registered_host_pointer = reinterpret_cast<void*>(pointer),
          .accesses = &attachment,
          .registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
      };
      source = &backing_.emplace_back();
      source->Initialize(api_, scope_, create);
      // A failed construction rollback can retain native pins. Free caller
      // pages only through a published handle's successful final release.
      if (source->memory != nullptr) {
        caller_storage_.push_back(storage);
      }
      ASSERT_FALSE(HasFatalFailure());
    }
    const uint64_t offset = (UINT64_C(4) << 30) + kSlabByteLength;
    const uint64_t payload_length = granularity - kPayloadByteOffset;
    amdf_memory_mapping_update_t update = {
        .target_byte_offset = offset,
        .byte_length = granularity,
        .access = access_,
        .source_byte_offset = granularity,
    };
    for (uint32_t generation = 1; generation <= sources.size(); ++generation) {
      update.memory = sources[generation - 1]->memory;
      auto* words = static_cast<uint32_t*>(upload_->host.pointer);
      for (uint64_t i = 0; i < payload_length / 4; ++i) {
        words[i] = TaggedWord(generation, i);
      }
      ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &update),
                AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(
          Copy(upload_->device_address,
               reservation_info_.address + offset + kPayloadByteOffset,
               payload_length));
      ASSERT_NO_FATAL_FAILURE(
          Verify(reservation_info_.address + offset + kPayloadByteOffset,
                 payload_length, generation));
      for (uint32_t previous = 1; previous <= generation; ++previous) {
        const auto* source = sources[previous - 1];
        ASSERT_NO_FATAL_FAILURE(
            Verify(source->device_address + granularity + kPayloadByteOffset,
                   payload_length, previous));
        const auto* host = reinterpret_cast<const uint32_t*>(
            static_cast<const uint8_t*>(
                source->creation.registered_host_pointer) +
            granularity + kPayloadByteOffset);
        for (uint64_t i = 0; i < payload_length / 4; ++i) {
          ASSERT_EQ(host[i], TaggedWord(previous, i)) << "word=" << i;
        }
      }
    }
    ASSERT_NO_FATAL_FAILURE(ReleaseReservation(&reservation_));
    for (auto* source : sources) {
      ASSERT_NO_FATAL_FAILURE(ReleaseBacking(source));
    }
  }
};

TEST_F(Pm4MappingTest, SystemBackingReplacesExactShortTerminalExtent) {
  RunShortExtent(system_scope_);
}

TEST_F(Pm4MappingTest, LocalBackingReplacesExactShortTerminalExtent) {
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "endpoint has no local storage scope";
  }
  RunShortExtent(local_scope_);
}

TEST_F(Pm4MappingTest, SystemBackingSplitReuseIsImmediatelyVisibleToCompute) {
  RunSplit(system_scope_);
}

TEST_F(Pm4MappingTest, LocalBackingSplitReuseIsImmediatelyVisibleToCompute) {
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "endpoint has no local storage scope";
  }
  RunSplit(local_scope_);
}

TEST_F(Pm4MappingTest,
       SystemBackingReplacementPreservesOtherAliasesAndRejectsOverlap) {
  RunReplacement(system_scope_);
}

TEST_F(Pm4MappingTest,
       LocalBackingReplacementPreservesOtherAliasesAndRejectsOverlap) {
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "endpoint has no local storage scope";
  }
  RunReplacement(local_scope_);
}

TEST_F(Pm4MappingTest, SystemBackingDetachPreservesAddressesForReuse) {
  RunDetachReuse(system_scope_);
}

TEST_F(Pm4MappingTest, LocalBackingDetachPreservesAddressesForReuse) {
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "endpoint has no local storage scope";
  }
  RunDetachReuse(local_scope_);
}

TEST_F(Pm4MappingTest, SystemBackingSlicesAndAliasesSharePhysicalStorage) {
  RunSourceSlices(system_scope_);
}

TEST_F(Pm4MappingTest, LocalBackingSlicesAndAliasesSharePhysicalStorage) {
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "endpoint has no local storage scope";
  }
  RunSourceSlices(local_scope_);
}

TEST_F(Pm4MappingTest, SystemBackingPermutesRetainedSparseRanges) {
  const auto original_ranges = amdf::testing::CompactSparseRanges(1);
  const auto replacement_ranges = amdf::testing::CompactSparseRanges(2);
  RunRetainedSparse(system_scope_, original_ranges, replacement_ranges);
}

TEST_F(Pm4MappingTest, LocalBackingPermutesRetainedSparseRanges) {
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "endpoint has no local storage scope";
  }
  const auto original_ranges = amdf::testing::CompactSparseRanges(1);
  const auto replacement_ranges = amdf::testing::CompactSparseRanges(2);
  RunRetainedSparse(local_scope_, original_ranges, replacement_ranges);
}

TEST_F(Pm4MappingTest, SystemBackingReplacementThroughSharedLogicalDevices) {
  RunSharedDomain(system_scope_);
}

TEST_F(Pm4MappingTest, LocalBackingReplacementThroughSharedLogicalDevices) {
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "endpoint has no local storage scope";
  }
  RunSharedDomain(local_scope_);
}

TEST_F(Pm4MappingTest, RegisteredBackingReplacementPreservesCallerPages) {
  RunRegisteredSource();
}

}  // namespace
