// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/memory_reservation.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "libamdf/src/memory_test_fixture.h"

namespace amdf::testing {
namespace {

// Substitute native ownership and mutation; range admission and failure state
// are the production API. No hardware fault is required to exercise this edge.
struct NativeReservation {
  // Real common state under test.
  amdf_memory_reservation_t* reservation = nullptr;
  // Two logical consumers sharing native page tables.
  std::array<amdf_memory_reservation_access_state_t, 2> accesses = {};
  // Descriptors delivered to the native boundary by the last accepted call.
  std::vector<amdf_memory_mapping_update_t> updates;
  // Number of native mutation calls, excluding common admission rejections.
  uint32_t update_count = 0;
  // Whether the next injected failure occurs after native mutation starts.
  bool mutation_started = false;
  // Result returned by the next native operation.
  amdf_status_t update_status = AMDF_STATUS_OK;
  // Logical alias selecting the native group on the last full detach.
  uint32_t detach_ordinal = UINT32_MAX;
  // Number of native full-group removal calls.
  uint32_t detach_count = 0;
  // Result returned by full-group removal without consuming the owner.
  amdf_status_t detach_status = AMDF_STATUS_OK;
  // Number of terminal release calls after either success or failure.
  uint32_t destroy_count = 0;
  // Native terminal result; other owners must still be consumed after failure.
  amdf_status_t destroy_status = AMDF_STATUS_OK;
};

NativeReservation* NativeForAccess(amdf_memory_reservation_t* reservation,
                                   uint32_t ordinal) {
  return static_cast<NativeReservation*>(
      reservation->accesses[reservation->accesses[ordinal].native_owner_ordinal]
          .native);
}

const amdf_memory_reservation_vtable_t kNativeReservationVtable = {
    .update =
        [](amdf_memory_reservation_t* base, uint32_t ordinal, uint32_t count,
           const amdf_memory_mapping_update_t* updates, bool* mutated) {
          auto* native = NativeForAccess(base, ordinal);
          ++native->update_count;
          native->updates.assign(updates, updates + count);
          *mutated = native->mutation_started;
          return native->update_status;
        },
    .detach =
        [](amdf_memory_reservation_t* base, uint32_t access_ordinal) {
          auto* native = NativeForAccess(base, access_ordinal);
          native->detach_ordinal = access_ordinal;
          ++native->detach_count;
          return native->detach_status;
        },
    .destroy =
        [](amdf_memory_reservation_t* base, uint32_t ordinal) {
          auto* native = NativeForAccess(base, ordinal);
          ++native->destroy_count;
          return native->destroy_status;
        },
};

class MemoryReservationTest : public MemoryTest {
 protected:
  void SetUp() override {
    MemoryTest::SetUp();
    ASSERT_EQ(
        amdf_calloc(instance_.host_allocator, sizeof(*native_.reservation),
                    alignof(amdf_memory_reservation_t),
                    reinterpret_cast<void**>(&native_.reservation)),
        AMDF_STATUS_OK);
    native_.reservation->host_allocator = instance_.host_allocator;
    native_.reservation->access_count = native_.accesses.size();
    native_.reservation->accesses = native_.accesses.data();
    for (uint32_t i = 0; i < devices_.size(); ++i) {
      InitializeFakeDevice(i + 1, &instance_, &devices_[i]);
      devices_[i].profile.supported_flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
      devices_[i].profile.guaranteed_flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
      devices_[i].request.requirements.flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
      auto& access = native_.accesses[i];
      access.device = &devices_[i].base;
      access.vtable = &kNativeReservationVtable;
      access.native_access_ordinal = i;
      access.info.ordinal = i;
      access.info.address = 0x100000;
      access.info.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
      access.info.byte_length = 65536;
      access.info.mapping_granularity = 4096;
      access.required_access = AMDF_MEMORY_ACCESS_READ;
    }
    native_.accesses[0].native = &native_;
    auto create = MakeMemoryCreateInfo(devices_[0]);
    create.byte_length = 16384;
    ASSERT_EQ(amdf_memory_create(&instance_.system_memory_scope, &create,
                                 &memory_, nullptr),
              AMDF_STATUS_OK);
  }

  void TearDown() override {
    if (native_.reservation != nullptr) {
      EXPECT_EQ(amdf_memory_reservation_destroy(native_.reservation),
                AMDF_STATUS_OK);
    }
    if (memory_ != nullptr) {
      EXPECT_EQ(amdf_memory_destroy(memory_, nullptr), AMDF_STATUS_OK);
    }
  }

  // Explicit initialized consumers outliving the source and reservation.
  std::array<FakeDevice, 2> devices_ = {};
  // Native dependency whose common base receives the public operations.
  NativeReservation native_;
  // Real ordinary memory created through the shared construction path.
  amdf_memory_t* memory_ = nullptr;
};

TEST_F(MemoryReservationTest, ValidatesTheWholeVectorBeforeNativeMutation) {
  std::array<amdf_memory_mapping_update_t, 2> updates = {{
      {
          .target_byte_offset = 8192,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
          .source_byte_offset = 4096,
      },
      {
          .target_byte_offset = 0,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
      },
  }};
  const auto original = updates;
  ASSERT_EQ(amdf_memory_reservation_update(native_.reservation, 0,
                                           updates.size(), updates.data()),
            AMDF_STATUS_OK);
  ASSERT_EQ(native_.updates.size(), 2u);
  EXPECT_EQ(native_.updates[0].target_byte_offset, 0u);
  EXPECT_EQ(native_.updates[1].source_byte_offset, 4096u);
  EXPECT_EQ(std::memcmp(updates.data(), original.data(), sizeof(updates)), 0);

  for (uint32_t failure = 0; failure < 6; ++failure) {
    updates = original;
    uint32_t access = 0;
    switch (failure) {
      case 0:
        updates[1].target_byte_offset = 8192;
        break;
      case 1:
        updates[1].source_byte_offset = 16384;
        break;
      case 2:
        updates[1].source_byte_offset = 1;
        break;
      case 3:
        updates[1].access = AMDF_MEMORY_ACCESS_EXECUTE;
        break;
      case 4:
        updates[1].memory = nullptr;
        break;
      case 5:
        access = 1;
        break;
    }
    EXPECT_FALSE(amdf_status_is_ok(amdf_memory_reservation_update(
        native_.reservation, access, updates.size(), updates.data())))
        << failure;
    EXPECT_EQ(native_.update_count, 1u) << failure;
  }
  // Rejections do not poison the owner. A later valid update still executes.
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 0,
                                           original.size(), original.data()),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.update_count, 2u);
}

TEST_F(MemoryReservationTest, CoalescesOnlyContiguousMappingsAndDetaches) {
  std::array<amdf_memory_mapping_update_t, 7> updates = {{
      {
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
      },
      {
          .target_byte_offset = 4096,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
          .source_byte_offset = 4096,
      },
      {
          .target_byte_offset = 8192,
          .byte_length = 8192,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
          .source_byte_offset = 8192,
      },
      {
          .target_byte_offset = 24576,
          .byte_length = 4096,
      },
      {
          .target_byte_offset = 28672,
          .byte_length = 4096,
      },
      {
          .target_byte_offset = 40960,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
      },
      {
          .target_byte_offset = 45056,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
      },
  }};
  std::reverse(updates.begin(), updates.end());
  const auto original = updates;
  ASSERT_EQ(amdf_memory_reservation_update(native_.reservation, 0,
                                           updates.size(), updates.data()),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(updates.data(), original.data(), sizeof(updates)), 0);
  ASSERT_EQ(native_.updates.size(), 4u);
  EXPECT_EQ(native_.updates[0].target_byte_offset, 0u);
  EXPECT_EQ(native_.updates[0].byte_length, 16384u);
  EXPECT_EQ(native_.updates[0].source_byte_offset, 0u);
  EXPECT_EQ(native_.updates[0].memory, memory_);
  EXPECT_EQ(native_.updates[0].access, AMDF_MEMORY_ACCESS_READ);
  EXPECT_EQ(native_.updates[1].target_byte_offset, 24576u);
  EXPECT_EQ(native_.updates[1].byte_length, 8192u);
  EXPECT_EQ(native_.updates[1].memory, nullptr);
  // Repeated source pages remain distinct aliases, even at adjacent addresses.
  EXPECT_EQ(native_.updates[2].target_byte_offset, 40960u);
  EXPECT_EQ(native_.updates[3].target_byte_offset, 45056u);
  for (size_t i = 2; i < native_.updates.size(); ++i) {
    EXPECT_EQ(native_.updates[i].byte_length, 4096u);
    EXPECT_EQ(native_.updates[i].source_byte_offset, 0u);
    EXPECT_EQ(native_.updates[i].memory, memory_);
  }
  // Coalescing an earlier run must not conceal an overlapping later range.
  updates[4].target_byte_offset = 4096;
  EXPECT_EQ(amdf_status_code(amdf_memory_reservation_update(
                native_.reservation, 0, updates.size(), updates.data())),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(native_.update_count, 1u);
}

TEST_F(MemoryReservationTest, ValidatesPriorSourcesBeforeAnyNativeMutation) {
  const std::array<amdf_memory_mapping_update_t, 2> valid = {{
      {
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
      },
      {
          .target_byte_offset = 8192,
          .byte_length = 4096,
          .previous_memory = memory_,
          .previous_source_byte_offset = 4096,
      },
  }};
  for (uint32_t failure = 0; failure < 6; ++failure) {
    auto updates = valid;
    auto& removal = updates[1];
    amdf_status_code_t expected = AMDF_STATUS_CODE_INVALID_ARGUMENT;
    switch (failure) {
      case 0:
        removal.reserved = 1;
        break;
      case 1:
        removal.previous_memory = nullptr;
        break;
      case 2:
        removal.previous_memory = nullptr;
        removal.previous_source_byte_offset = 0;
        removal.previous_memory_access_ordinal = 1;
        break;
      case 3:
        removal.previous_source_byte_offset = 16384;
        expected = AMDF_STATUS_CODE_OUT_OF_RANGE;
        break;
      case 4:
        removal.previous_source_byte_offset = 1;
        expected = AMDF_STATUS_CODE_UNSUPPORTED;
        break;
      case 5:
        removal.previous_memory_access_ordinal = UINT32_MAX;
        expected = AMDF_STATUS_CODE_OUT_OF_RANGE;
        break;
    }
    EXPECT_EQ(amdf_status_code(amdf_memory_reservation_update(
                  native_.reservation, 0, updates.size(), updates.data())),
              expected)
        << failure;
    EXPECT_EQ(native_.update_count, 0u) << failure;
  }
  ASSERT_EQ(amdf_memory_reservation_update(native_.reservation, 0, valid.size(),
                                           valid.data()),
            AMDF_STATUS_OK);
  ASSERT_EQ(native_.updates.size(), 2u);
  EXPECT_EQ(native_.updates[1].previous_memory, memory_);
  EXPECT_EQ(native_.updates[1].previous_source_byte_offset, 4096u);
}

TEST_F(MemoryReservationTest, PriorSourceUsesItsSharedLogicalProjection) {
  const amdf_memory_mapping_update_t removal = {
      .byte_length = 4096,
      .previous_memory = memory_,
      .previous_source_byte_offset = 4096,
  };
  // Device 0 owns the backing, but device 1 publishes the shared page tables.
  ASSERT_EQ(amdf_memory_reservation_update(native_.reservation, 1, 1, &removal),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.update_count, 1u);
  ASSERT_EQ(native_.updates.size(), 1u);
  EXPECT_EQ(native_.updates[0].previous_memory, memory_);
  // The same source is not a valid prior binding in a distinct mapping group.
  native_.accesses[1].info.mapping_access_ordinal = 1;
  EXPECT_EQ(amdf_status_code(amdf_memory_reservation_update(native_.reservation,
                                                            1, 1, &removal)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(native_.update_count, 1u);
  native_.accesses[1].info.mapping_access_ordinal = 0;
}

TEST_F(MemoryReservationTest, CoalescingPreservesPriorSourceRuns) {
  std::array<amdf_memory_mapping_update_t, 2> updates = {{
      {
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
          .previous_memory = memory_,
          .previous_source_byte_offset = 8192,
      },
      {
          .target_byte_offset = 4096,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
          .source_byte_offset = 4096,
          .previous_memory = memory_,
          .previous_source_byte_offset = 12288,
      },
  }};
  ASSERT_EQ(amdf_memory_reservation_update(native_.reservation, 0,
                                           updates.size(), updates.data()),
            AMDF_STATUS_OK);
  ASSERT_EQ(native_.updates.size(), 1u);
  EXPECT_EQ(native_.updates[0].byte_length, 8192u);
  EXPECT_EQ(native_.updates[0].previous_source_byte_offset, 8192u);
  // Consecutive destinations replacing aliases of one source page must remain
  // separate even when the new backing is contiguous.
  updates[1].previous_source_byte_offset = 8192;
  ASSERT_EQ(amdf_memory_reservation_update(native_.reservation, 0,
                                           updates.size(), updates.data()),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.updates.size(), 2u);
  // An adjacent hole is not part of the prior source's surviving run.
  updates[1].previous_memory = nullptr;
  updates[1].previous_source_byte_offset = 0;
  ASSERT_EQ(amdf_memory_reservation_update(native_.reservation, 0,
                                           updates.size(), updates.data()),
            AMDF_STATUS_OK);
  ASSERT_EQ(native_.updates.size(), 2u);
  EXPECT_EQ(native_.updates[1].previous_memory, nullptr);
}

TEST_F(MemoryReservationTest,
       PreservesSourcePermissionAndDestinationBoundaries) {
  auto create = MakeMemoryCreateInfo(devices_[0]);
  create.byte_length = 16384;
  amdf_memory_t* other_memory = nullptr;
  ASSERT_EQ(amdf_memory_create(&instance_.system_memory_scope, &create,
                               &other_memory, nullptr),
            AMDF_STATUS_OK);
  const std::array<amdf_memory_mapping_update_t, 5> boundaries = {{
      {
          .target_byte_offset = 8192,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
          .source_byte_offset = 4096,
      },
      {
          .target_byte_offset = 4096,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ,
          .source_byte_offset = 8192,
      },
      {
          .target_byte_offset = 4096,
          .byte_length = 4096,
          .memory = other_memory,
          .access = AMDF_MEMORY_ACCESS_READ,
          .source_byte_offset = 4096,
      },
      {
          .target_byte_offset = 4096,
          .byte_length = 4096,
          .memory = memory_,
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
          .source_byte_offset = 4096,
      },
      {
          .target_byte_offset = 4096,
          .byte_length = 4096,
      },
  }};
  for (const auto& boundary : boundaries) {
    const amdf_memory_mapping_update_t updates[] = {
        {
            .byte_length = 4096,
            .memory = memory_,
            .access = AMDF_MEMORY_ACCESS_READ,
        },
        boundary,
    };
    EXPECT_EQ(
        amdf_memory_reservation_update(native_.reservation, 0, 2, updates),
        AMDF_STATUS_OK);
    EXPECT_EQ(native_.updates.size(), 2u);
    if (native_.updates.size() == 2) {
      EXPECT_EQ(native_.updates[0].byte_length, 4096u);
      EXPECT_EQ(native_.updates[1].target_byte_offset,
                boundary.target_byte_offset);
      EXPECT_EQ(native_.updates[1].memory, boundary.memory);
      EXPECT_EQ(native_.updates[1].access, boundary.access);
      EXPECT_EQ(native_.updates[1].source_byte_offset,
                boundary.source_byte_offset);
    }
  }
  EXPECT_EQ(amdf_memory_reservation_detach(native_.reservation, 0),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_memory_destroy(other_memory, nullptr), AMDF_STATUS_OK);
}

TEST_F(MemoryReservationTest,
       PartialNativeFailurePoisonsEveryAliasOnlyAfterMutation) {
  const amdf_memory_mapping_update_t detach = {.byte_length = 4096};
  const auto failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  native_.update_status = failure;
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 0, 1, &detach),
            failure);
  native_.update_status = AMDF_STATUS_OK;
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 1, 1, &detach),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.update_count, 2u);

  native_.update_status = failure;
  native_.mutation_started = true;
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 1, 1, &detach),
            failure);
  native_.update_status = AMDF_STATUS_OK;
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 0, 1, &detach),
            failure);
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 1, 0, nullptr),
            failure);
  EXPECT_EQ(native_.update_count, 3u);
  EXPECT_EQ(amdf_memory_reservation_destroy(native_.reservation),
            AMDF_STATUS_OK);
  native_.reservation = nullptr;
  EXPECT_EQ(native_.destroy_count, 1u);
}

TEST_F(MemoryReservationTest, DetachValidatesTheSelectedGroup) {
  EXPECT_EQ(amdf_memory_reservation_detach(nullptr, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(amdf_memory_reservation_detach(native_.reservation, 2),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  EXPECT_EQ(native_.detach_count, 0u);
  EXPECT_EQ(amdf_memory_reservation_detach(native_.reservation, 1),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.detach_ordinal, 1u);
  EXPECT_EQ(native_.destroy_count, 0u);
  const amdf_memory_mapping_update_t update = {
      .byte_length = 4096,
      .memory = memory_,
      .access = AMDF_MEMORY_ACCESS_READ,
  };
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 0, 1, &update),
            AMDF_STATUS_OK);
}

TEST_F(MemoryReservationTest,
       DetachAfterPartialFailurePreservesAddressesAndOriginalFailure) {
  const auto failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  const amdf_memory_mapping_update_t update = {
      .byte_length = 4096,
      .memory = memory_,
      .access = AMDF_MEMORY_ACCESS_READ,
  };
  native_.update_status = failure;
  native_.mutation_started = true;
  ASSERT_EQ(amdf_memory_reservation_update(native_.reservation, 0, 1, &update),
            failure);
  const amdf_memory_mapping_update_t full_remove = {.byte_length = 65536};
  EXPECT_EQ(
      amdf_memory_reservation_update(native_.reservation, 1, 1, &full_remove),
      failure);
  EXPECT_EQ(native_.update_count, 1u);

  const auto detach_failure =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  native_.detach_status = detach_failure;
  EXPECT_EQ(amdf_memory_reservation_detach(native_.reservation, 1),
            detach_failure);
  EXPECT_EQ(native_.detach_ordinal, 1u);
  EXPECT_EQ(native_.destroy_count, 0u);
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 0, 1, &update),
            failure);

  // Native removal is now independently successful. It does not recover the
  // group's original failed state or release its reserved addresses.
  native_.detach_status = AMDF_STATUS_OK;
  ASSERT_EQ(amdf_memory_reservation_detach(native_.reservation, 1),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.detach_count, 2u);
  EXPECT_EQ(native_.destroy_count, 0u);
  for (uint32_t i = 0; i < native_.reservation->access_count; ++i) {
    amdf_memory_reservation_access_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(amdf_memory_reservation_query_access_info(native_.reservation, i,
                                                        &info),
              AMDF_STATUS_OK);
    EXPECT_EQ(info.address, 0x100000u);
    EXPECT_EQ(info.byte_length, 65536u);
    EXPECT_EQ(
        amdf_memory_reservation_update(native_.reservation, i, 0, nullptr),
        failure);
  }
  // The caller can release detached backing while retaining the reservation.
  ASSERT_EQ(amdf_memory_destroy(memory_, nullptr), AMDF_STATUS_OK);
  memory_ = nullptr;
  EXPECT_EQ(amdf_memory_reservation_destroy(native_.reservation),
            AMDF_STATUS_OK);
  native_.reservation = nullptr;
  EXPECT_EQ(native_.destroy_count, 1u);
}

TEST_F(MemoryReservationTest, DetachFailurePoisonsOnlyItsNativeGroup) {
  native_.accesses[1].info.mapping_access_ordinal = 1;
  const auto failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  native_.detach_status = failure;
  EXPECT_EQ(amdf_memory_reservation_detach(native_.reservation, 1), failure);
  const amdf_memory_mapping_update_t remove = {.byte_length = 4096};
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 1, 1, &remove),
            failure);
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 0, 1, &remove),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.update_count, 1u);
  EXPECT_EQ(native_.destroy_count, 0u);
}

TEST_F(MemoryReservationTest, IndependentOwnersReleaseAfterTerminalFailure) {
  NativeReservation second;
  native_.accesses[1].native = &second;
  native_.accesses[1].native_owner_ordinal = 1;
  native_.accesses[1].info.mapping_access_ordinal = 1;
  const auto failure = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  native_.destroy_status = failure;
  EXPECT_EQ(amdf_memory_reservation_destroy(native_.reservation), failure);
  native_.reservation = nullptr;
  EXPECT_EQ(native_.destroy_count, 1u);
  EXPECT_EQ(second.destroy_count, 1u);
}

amdf_status_t PrepareNativeReservation(
    amdf_memory_reservation_t* reservation,
    const amdf_memory_reservation_create_info_t*,
    const amdf_memory_reservation_group_t* group) {
  const uint32_t ordinal = group->access_ordinals[0];
  auto* device =
      reinterpret_cast<FakeDevice*>(reservation->accesses[ordinal].device);
  ++device->create_call_count;
  device->prepared_access_count = group->access_count;
  reservation->accesses[ordinal].native = device;
  return device->create_status;
}

amdf_status_t DestroyNativeReservation(amdf_memory_reservation_t* reservation,
                                       uint32_t ordinal) {
  auto* device =
      static_cast<FakeDevice*>(reservation->accesses[ordinal].native);
  ++device->destroy_call_count;
  return device->destroy_status;
}

TEST_F(MemoryReservationTest, ConstructorRollsBackEveryPreparedNativeOwner) {
  const auto* original_vtable = devices_[0].base.vtable;
  auto vtable = *devices_[0].base.vtable;
  vtable.query_memory_reservation_profile =
      [](amdf_device_t* base, uint32_t,
         amdf_memory_reservation_profile_t* out) {
        static const amdf_memory_reservation_vtable_t preparations[] = {
            {
                .prepare = PrepareNativeReservation,
                .destroy = DestroyNativeReservation,
            },
            {
                .prepare = PrepareNativeReservation,
                .destroy = DestroyNativeReservation,
            },
        };
        *out = {
            .capabilities =
                {
                    .supported_access = AMDF_MEMORY_ACCESS_READ,
                    .required_access = AMDF_MEMORY_ACCESS_READ,
                    .maximum_byte_length = 65536,
                    .reservation_granularity = 4096,
                    .maximum_alignment = 65536,
                    .mapping_granularity = 4096,
                },
            .vtable = &preparations[reinterpret_cast<FakeDevice*>(base)
                                        ->profile.ordinal],
        };
        return AMDF_STATUS_OK;
      };
  std::array<amdf_memory_reservation_access_t, 2> accesses = {};
  for (uint32_t i = 0; i < devices_.size(); ++i) {
    devices_[i].base.vtable = &vtable;
    devices_[i].profile.ordinal = i;
    accesses[i] = {&devices_[i].base, 0, AMDF_MEMORY_ACCESS_READ};
  }
  const auto failure =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  devices_[1].create_status = failure;
  const amdf_memory_reservation_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(create),
      .byte_length = 4096,
      .access_count = accesses.size(),
      .accesses = accesses.data(),
  };
  auto* output = native_.reservation;
  const uint32_t previous_calls = devices_[0].create_call_count;
  EXPECT_EQ(amdf_memory_reservation_create(&instance_, &create, &output),
            failure);
  EXPECT_EQ(output, native_.reservation);
  EXPECT_EQ(devices_[0].create_call_count, previous_calls + 1);
  EXPECT_EQ(devices_[1].create_call_count, 1u);
  EXPECT_EQ(devices_[0].destroy_call_count, 1u);
  EXPECT_EQ(devices_[1].destroy_call_count, 1u);
  EXPECT_EQ(devices_[0].prepared_access_count, 1u);
  EXPECT_EQ(devices_[1].prepared_access_count, 1u);
  // Restore the borrowed device operation table before fixture cleanup.
  for (auto& device : devices_) {
    device.base.vtable = original_vtable;
  }
}

TEST_F(MemoryReservationTest,
       MappingAlignmentIncludesTheImportedLogicalPrefix) {
  auto& device = devices_[0];
  device.profile.external_memory_support[0].source_offset_alignment = 1;
  device.profile.external_memory_support[0].byte_length_alignment = 1;
  ReleaseState released = {};
  amdf_external_memory_t external = {
      .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
  };
  external.payload.file_descriptor = 101;
  external.source_byte_offset = 13;
  external.byte_length = 16371;
  external.physical_backing_id = device.backing_id;
  external.release = RecordRelease;
  external.release_user_data = &released;
  auto import = MakeMemoryImportInfo(device);
  import.minimum_alignment = 1;
  amdf_memory_t* imported = nullptr;
  ASSERT_EQ(amdf_memory_import(&instance_.system_memory_scope, &import,
                               &external, &imported, nullptr),
            AMDF_STATUS_OK);
  EXPECT_EQ(released.count, 1u);
  amdf_memory_mapping_update_t update = {
      .byte_length = 4096,
      .memory = imported,
      .access = AMDF_MEMORY_ACCESS_READ,
      .source_byte_offset = 4096,
  };
  EXPECT_EQ(amdf_status_code(amdf_memory_reservation_update(native_.reservation,
                                                            0, 1, &update)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(native_.update_count, 0u);
  update.source_byte_offset = 4096 - 13;
  EXPECT_EQ(amdf_memory_reservation_update(native_.reservation, 0, 1, &update),
            AMDF_STATUS_OK);
  ASSERT_EQ(native_.updates.size(), 1u);
  // The common layer validates native alignment but keeps the logical offset.
  // The GPU provider, not each common handoff, adds the native prefix.
  EXPECT_EQ(native_.updates[0].source_byte_offset, 4096u - 13u);
  EXPECT_EQ(amdf_memory_destroy(imported, nullptr), AMDF_STATUS_OK);
}

TEST_F(MemoryReservationTest,
       AdmissionAndProviderFailurePreserveConstructorOutput) {
  auto vtable = *devices_[0].base.vtable;
  vtable.query_memory_reservation_profile =
      [](amdf_device_t*, uint32_t, amdf_memory_reservation_profile_t* out) {
        static const amdf_memory_reservation_vtable_t preparation = {
            .prepare =
                [](amdf_memory_reservation_t*,
                   const amdf_memory_reservation_create_info_t* create,
                   const amdf_memory_reservation_group_t* group) {
                  ++reinterpret_cast<FakeDevice*>(
                        create->accesses[group->access_ordinals[0]].device)
                        ->create_call_count;
                  return amdf_make_api_status(
                      AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
                },
        };
        out->vtable = &preparation;
        out->capabilities = {
            .supported_access = AMDF_MEMORY_ACCESS_READ,
            .required_access = AMDF_MEMORY_ACCESS_READ,
            .maximum_byte_length = 65536,
            .reservation_granularity = 4096,
            .maximum_alignment = 65536,
            .mapping_granularity = 4096,
        };
        return AMDF_STATUS_OK;
      };
  devices_[0].base.vtable = &vtable;
  const amdf_memory_reservation_access_t access = {&devices_[0].base, 0,
                                                   AMDF_MEMORY_ACCESS_READ};
  amdf_memory_reservation_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(create),
      .byte_length = 65536,
      .access_count = 1,
      .accesses = &access,
  };
  auto* output = native_.reservation;
  const uint32_t previous_calls = devices_[0].create_call_count;
  create.minimum_alignment = 3;
  EXPECT_EQ(amdf_status_code(
                amdf_memory_reservation_create(&instance_, &create, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(devices_[0].create_call_count, previous_calls);
  EXPECT_EQ(output, native_.reservation);
  create.minimum_alignment = 0;
  create.flags = UINT32_C(0x80000000);
  EXPECT_EQ(amdf_status_code(
                amdf_memory_reservation_create(&instance_, &create, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(output, native_.reservation);
  EXPECT_EQ(devices_[0].create_call_count, previous_calls);
  create.flags = AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS;
  EXPECT_EQ(amdf_status_code(
                amdf_memory_reservation_create(&instance_, &create, &output)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(output, native_.reservation);
  EXPECT_EQ(devices_[0].create_call_count, previous_calls);
  create.flags = 0;
  EXPECT_EQ(amdf_status_code(
                amdf_memory_reservation_create(&instance_, &create, &output)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(devices_[0].create_call_count, previous_calls + 1);
  EXPECT_EQ(output, native_.reservation);
  devices_[0].base.vtable = devices_[1].base.vtable;
}

}  // namespace
}  // namespace amdf::testing
