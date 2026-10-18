// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>

#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/interop/gpu/xdna/recipes/device_fixture.h"
#include "libamdf/cts/interop/gpu/xdna/recipes/pm4_queue.h"
#include "libamdf/cts/xdna/programs/mul_i32.h"
#include "libamdf/cts/xdna/util/executable.h"
#include "libamdf/cts/xdna/util/execution.h"
#include "util/wait.h"

namespace {
constexpr auto kReadWrite = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
constexpr size_t kPayloadByteLength = 64;
constexpr size_t kReadbackByteOffset = 1024;
constexpr size_t kCompletionByteOffset = 2048;
struct Binding {
  // Retained allocation ordinal.
  uint32_t backing;
  // Source unit in that allocation.
  uint32_t unit;
};
struct Backing {
  // Optional CPU owner supplying pages to joint device registration.
  CtsMappedMemory storage;
  // Joint GPU/NPU allocation or registration over storage's retained pages.
  CtsMappedMemory memory;
  // Expected complete backing, including old locations and untouched neighbors.
  std::vector<uint8_t> expected;
};

class FabricReservationTest : public GpuXdnaDeviceFixture,
                              public ::testing::WithParamInterface<uint64_t> {
 protected:
  FabricReservationTest()
      : GpuXdnaDeviceFixture(AMDF_QUEUE_ROLE_TRANSFER |
                             AMDF_QUEUE_ROLE_CACHE_CONTROL) {}

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(GpuXdnaDeviceFixture::SetUp());
    if (IsSkipped()) {
      return;
    }
    for (uint32_t i = 0; i < capabilities_.size(); ++i) {
      auto& capability = capabilities_[i];
      capability.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES;
      capability.structure_size = sizeof(capability);
      const auto status =
          i == 2 ? api_->instance_query_host_memory_reservation_capabilities(
                       instance_, &capability)
                 : api_->device_query_memory_reservation_capabilities(
                       i == 0 ? xdna_device_ : device_, 0, &capability);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        GTEST_SKIP() << "consumer " << i << " does not support reservations";
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    // Consumer and source ordinals are caller-defined, not native owner order.
    // The NPU's biased DMA address is never interpreted as a CPU pointer.
    for (uint32_t i = 0; i < source_accesses_.size(); ++i) {
      source_accesses_[i] = accesses_[i];
      source_accesses_[i].requirements.access |=
          capabilities_[i].required_access;
      source_accesses_[i].requirements.flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
    }
    ASSERT_NO_FATAL_FAILURE(FindProfile(
        source_accesses_, AMDF_MEMORY_PROFILE_ROLE_CREATE, &source_profile_));
    if (source_profile_.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN ||
        (source_profile_.supported_flags &
         AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) == 0) {
      source_role_ = AMDF_MEMORY_PROFILE_ROLE_REGISTER;
      ASSERT_NO_FATAL_FAILURE(
          FindProfile(source_accesses_, source_role_, &source_profile_));
      if (source_profile_.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
        GTEST_SKIP() << "no joint mapping source allocation or registration";
      }
      ASSERT_NO_FATAL_FAILURE(
          FindProfile({}, AMDF_MEMORY_PROFILE_ROLE_CREATE, &host_profile_));
      ASSERT_NE(host_profile_.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
      ASSERT_NE(
          host_profile_.supported_flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE,
          0u);
    }
    const auto& geometry = source_role_ == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                               ? source_profile_.registration
                               : source_profile_.allocation;
    unit_ = std::max({GetParam(), capabilities_[0].reservation_granularity,
                      capabilities_[0].mapping_granularity,
                      capabilities_[1].reservation_granularity,
                      capabilities_[1].mapping_granularity,
                      capabilities_[2].reservation_granularity,
                      capabilities_[2].mapping_granularity,
                      geometry.native_byte_length_granularity,
                      host_profile_.allocation.native_byte_length_granularity});
    ASSERT_EQ(unit_ & (unit_ - 1), 0u);
    // Exercise the last native page even with a larger caller mapping unit.
    payload_offset_ = unit_ - capabilities_[2].mapping_granularity;
    access_ = source_accesses_[1].requirements.access;
    host_access_ = kReadWrite | capabilities_[0].required_access |
                   capabilities_[2].required_access;
  }

  void TearDown() override {
    if (!queue_.Release(api_) || !execution_.Release(api_, xdna_api_)) {
      return;
    }
    if (view_) {
      ASSERT_EQ(api_->host_mapping_destroy(view_), AMDF_STATUS_OK);
      view_ = nullptr;
    }
    if (reservation_) {
      bool detached = true;
      for (uint32_t ordinal = 0; ordinal < reservation_access_count_;
           ++ordinal) {
        const auto status =
            api_->memory_reservation_detach(reservation_, ordinal);
        EXPECT_EQ(status, AMDF_STATUS_OK);
        detached &= amdf_status_is_ok(status);
      }
      const auto status = api_->memory_reservation_destroy(reservation_);
      reservation_ = nullptr;
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (!detached && !amdf_status_is_ok(status)) {
        return;
      }
    }
    for (auto& backing : backings_) {
      if (!backing.memory.Release(api_) || !backing.storage.Release(api_)) {
        return;
      }
    }
    if (!unprepared_.Release(api_) || !staging_.Release(api_)) {
      return;
    }
  }

  void PrepareSources() {
    const auto& geometry = source_role_ == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                               ? source_profile_.registration
                               : source_profile_.allocation;
    amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = source_profile_.ordinal,
        .access_count = static_cast<uint32_t>(source_accesses_.size()),
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .byte_length = 6 * unit_,
        .minimum_alignment = geometry.minimum_alignment,
        .accesses = source_accesses_.data(),
    };
    for (auto& backing : backings_) {
      if (source_role_ == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        const amdf_memory_create_info_t host = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
            .structure_size = sizeof(host),
            .memory_profile_ordinal = host_profile_.ordinal,
            .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                              AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE,
            .byte_length = create.byte_length,
            .minimum_alignment =
                std::max(geometry.registered_host_pointer_alignment,
                         host_profile_.allocation.minimum_alignment),
        };
        ASSERT_NO_FATAL_FAILURE(
            backing.storage.Create(api_, system_scope_, host));
        create.registered_host_pointer = backing.storage.host.pointer;
        create.registered_host_cacheability = backing.storage.host.cacheability;
      } else {
        create.required_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
      }
      ASSERT_NO_FATAL_FAILURE(
          backing.memory.Create(api_, system_scope_, create));
      ASSERT_EQ(backing.memory.host.cacheability,
                AMDF_HOST_CACHEABILITY_WRITE_BACK);
      backing.expected.resize(create.byte_length);
    }
  }

  void PrepareBacking() {
    ASSERT_NO_FATAL_FAILURE(PrepareSources());
    const amdf_memory_reservation_access_t consumers[] = {
        {xdna_device_, capabilities_[0].address_domain_ordinal, host_access_,
         unit_},
        {device_, capabilities_[1].address_domain_ordinal, access_, unit_},
        {nullptr, capabilities_[2].address_domain_ordinal, host_access_, unit_},
    };
    const amdf_memory_reservation_create_info_t reserve = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
        .structure_size = sizeof(reserve),
        .byte_length = 3 * unit_,
        .minimum_alignment = unit_,
        .access_count = 3,
        .accesses = consumers,
    };
    ASSERT_EQ(
        api_->memory_reservation_create(instance_, &reserve, &reservation_),
        AMDF_STATUS_OK);
    reservation_access_count_ = reserve.access_count;
    for (uint32_t i = 0; i < projections_.size(); ++i) {
      auto& projection = projections_[i];
      projection.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO;
      projection.structure_size = sizeof(projection);
      ASSERT_EQ(api_->memory_reservation_query_access_info(reservation_, i,
                                                           &projection),
                AMDF_STATUS_OK);
      ASSERT_EQ(projection.mapping_granularity, unit_);
      ASSERT_EQ(projection.ordinal, i);
      ASSERT_EQ(projection.byte_length, 3 * unit_);
    }
    ASSERT_EQ(projections_[0].address_kind, AMDF_MEMORY_ADDRESS_XDNA_DMA);
    ASSERT_EQ(projections_[1].address_kind, AMDF_MEMORY_ADDRESS_GPU);
    ASSERT_EQ(projections_[2].address_kind, AMDF_MEMORY_ADDRESS_HOST);
    ASSERT_EQ(projections_[0].mapping_access_ordinal, 0u);
    ASSERT_EQ(projections_[1].mapping_access_ordinal, 1u);
    ASSERT_TRUE(projections_[2].mapping_access_ordinal == 0u ||
                projections_[2].mapping_access_ordinal == 2u);
    gpu_address_ = projections_[1].address;
    const amdf_memory_map_info_t map = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(map),
        .byte_length = 3 * unit_,
        .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    ASSERT_EQ(api_->memory_reservation_map(reservation_, 2, &map, &view_),
              AMDF_STATUS_OK);
    amdf_host_mapping_info_t host = {
        .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
        .structure_size = sizeof(host),
    };
    ASSERT_EQ(api_->host_mapping_query_info(view_, &host), AMDF_STATUS_OK);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(host.pointer),
              projections_[2].address);
    host_ = static_cast<uint8_t*>(host.pointer);
  }

  void Map(uint32_t ordinal, Binding binding) {
    const auto& backing = backings_[binding.backing];
    amdf_memory_mapping_update_t update = {
        .target_byte_offset = ordinal * unit_,
        .byte_length = unit_,
        .memory = backing.memory.memory,
        .memory_access_ordinal = 0,
        .access = host_access_,
        .source_byte_offset = binding.unit * unit_,
    };
    // Shared translations update through either alias. Independent consumers
    // each receive an explicit mapping of the same retained physical pages.
    const bool shared = projections_[2].mapping_access_ordinal == 0;
    ASSERT_EQ(api_->memory_reservation_update(
                  reservation_, shared && ordinal % 2 != 0 ? 2 : 0, 1, &update),
              AMDF_STATUS_OK);
    update.memory_access_ordinal = 1;
    update.access = access_;
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 1, 1, &update),
              AMDF_STATUS_OK);
    if (!shared) {
      update.memory = backing.storage.memory != nullptr ? backing.storage.memory
                                                        : backing.memory.memory;
      update.memory_access_ordinal = 0;
      update.access = host_access_;
      ASSERT_EQ(api_->memory_reservation_update(reservation_, 2, 1, &update),
                AMDF_STATUS_OK);
    }
  }

  void Detach() {
    for (uint32_t ordinal = 0; ordinal < reservation_access_count_; ++ordinal) {
      if (projections_[ordinal].mapping_access_ordinal == ordinal) {
        ASSERT_EQ(api_->memory_reservation_detach(reservation_, ordinal),
                  AMDF_STATUS_OK);
      }
    }
  }

  void PrepareExecution() {
    auto staging_access = accesses_[1];
    staging_access.requirements.flags |= AMDF_MEMORY_FLAG_QUEUE_STORAGE;
    amdf_memory_profile_t profile = {};
    ASSERT_NO_FATAL_FAILURE(FindProfile(
        {&staging_access, 1}, AMDF_MEMORY_PROFILE_ROLE_CREATE, &profile));
    ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = profile.ordinal,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .byte_length = unit_,
        .minimum_alignment = unit_,
        .accesses = &staging_access,
    };
    ASSERT_NO_FATAL_FAILURE(staging_.Create(api_, system_scope_, create));
    ASSERT_EQ(
        api_->memory_query_address(staging_.memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                   &staging_address_),
        AMDF_STATUS_OK);
    std::memset(staging_.host.pointer, 0, staging_.host.byte_length);
    ASSERT_NO_FATAL_FAILURE(queue_.Initialize(
        api_, gpu_api_, device_, system_scope_, gpu_family_, publication_mode_,
        reinterpret_cast<uintptr_t>(staging_.host.pointer) +
            kCompletionByteOffset,
        staging_address_ + kCompletionByteOffset));
    XdnaExecutable executable;
    const std::array<amdf_memory_access_t, 3> access = {
        AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_READ,
        AMDF_MEMORY_ACCESS_WRITE};
    bool admitted = false;
    const auto* images = amdf_cts_xdna_mul_i32_create();
    for (size_t i = 0; i < amdf_cts_xdna_mul_i32_size(); ++i) {
      const auto& image = images[i];
      if (executable.Initialize(
              {reinterpret_cast<const uint8_t*>(image.data), image.size},
              xdna_endpoint_info_, xdna_device_info_, 1, access)) {
        admitted = true;
        break;
      }
    }
    ASSERT_TRUE(admitted) << "no compatible finite arithmetic fixture";
    std::vector<uint8_t> storage(executable.allocation_byte_length());
    executable.Load(storage);
    std::array<uint64_t, 3> addresses = {};
    for (uint32_t i = 0; i < 3; ++i) {
      addresses[i] = projections_[0].address + i * unit_ + payload_offset_;
    }
    ASSERT_TRUE(executable.Bind(storage, addresses));
    ASSERT_NO_FATAL_FAILURE(
        execution_.Prepare(api_, xdna_api_, xdna_device_, xdna_family_, 1,
                           executable.ResolveInvocation(storage),
                           executable.allocation_alignment()));
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            execution_.instructions.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
            execution_.instructions.info.byte_length),
        AMDF_STATUS_OK);
  }

  // Stable CPU pointer borrowed from the public host view.
  uint8_t* host_ = nullptr;
  // One public owner for the NPU, GPU and host projections.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Number of caller projections owned by reservation_.
  uint32_t reservation_access_count_ = 0;
  // Common slice geometry admitted by both mapping providers.
  uint64_t unit_ = 0;
  // Last native page of each independently replaceable unit.
  uint64_t payload_offset_ = 0;
  // Exact permissions required by GPU mapping.
  amdf_memory_access_t access_ = 0;
  // Stable GPU address distinct from host_/NPU address.
  uint64_t gpu_address_ = 0;
  // Host/SVA protection envelope, independent of GPU-only permission bits.
  amdf_memory_access_t host_access_ = 0;
  // Native per-consumer limits queried before source selection.
  std::array<amdf_memory_reservation_capabilities_t, 3> capabilities_ = {};
  // Queried source profile satisfying both device requests.
  amdf_memory_profile_t source_profile_ = {};
  // Native role selected before construction, without a failure fallback.
  amdf_memory_profile_roles_t source_role_ = AMDF_MEMORY_PROFILE_ROLE_CREATE;
  // CPU-owned replaceable backing profile when devices register shared pages.
  amdf_memory_profile_t host_profile_ = {};
  // NPU then GPU requests retained for ordinary backing construction.
  std::array<amdf_memory_device_access_t, 2> source_accesses_ = {};
  // Immutable public addresses and native mapping-group identities.
  std::array<amdf_memory_reservation_access_info_t, 3> projections_ = {};
  // Direct cache view borrowing the stable host address interval.
  amdf_host_mapping_t* view_ = nullptr;
  // GPU-only source used to check missing NPU preparation before mutation.
  CtsMappedMemory unprepared_;
  // Retained ordinary backing, not recreated between epochs.
  std::array<Backing, 2> backings_;
  // Coherent GPU staging and completion storage.
  CtsMappedMemory staging_;
  // GPU address of staging storage.
  uint64_t staging_address_ = 0;
  // Existing finite GPU command transport.
  Pm4RecipeQueue queue_;
  // One immutable finite NPU invocation reused at every epoch.
  XdnaExecution execution_;
};

TEST_P(FabricReservationTest, NpuFirstWriterUsesFreshGpuBacking) {
  ASSERT_FALSE(IsSkipped());
  ASSERT_NO_FATAL_FAILURE(PrepareBacking());
  payload_offset_ += 3 * backings_[0].memory.host.cache_line_size;
  ASSERT_NO_FATAL_FAILURE(Map(0, {0, 1}));
  ASSERT_NO_FATAL_FAILURE(Map(1, {0, 2}));
  ASSERT_NO_FATAL_FAILURE(Map(2, {1, 3}));
  ASSERT_NO_FATAL_FAILURE(PrepareExecution());
  std::array<uint32_t, 16> expected = {};
  for (uint32_t i = 0; i < expected.size(); ++i) {
    const uint32_t lhs = 11 + 7 * i;
    const uint32_t rhs = 23 + 13 * i;
    std::memcpy(host_ + payload_offset_ + 4 * i, &lhs, sizeof(lhs));
    std::memcpy(host_ + unit_ + payload_offset_ + 4 * i, &rhs, sizeof(rhs));
    expected[i] = lhs * rhs;
  }
  for (uint32_t i = 0; i < 2; ++i) {
    ASSERT_EQ(api_->host_mapping_cache_control(
                  view_, AMDF_HOST_CACHE_OPERATION_FLUSH,
                  i * unit_ + payload_offset_, kPayloadByteLength),
              AMDF_STATUS_OK);
  }
  const amdf_xdna_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command_count = 1,
      .commands = &execution_.command,
  };
  uint64_t point = 0;
  ASSERT_EQ(xdna_api_->kernel_queue_submit(execution_.queue, &submit, &point),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->kernel_queue_wait(execution_.queue, point,
                                    amdf::cts::kWaitTimeoutNs,
                                    amdf::cts::kWaitPollIntervalNs),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->host_mapping_cache_control(
                view_, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                2 * unit_ + payload_offset_, kPayloadByteLength),
            AMDF_STATUS_OK);
  for (uint32_t i = 0; i < expected.size(); ++i) {
    uint32_t actual = 0;
    std::memcpy(&actual, host_ + 2 * unit_ + payload_offset_ + 4 * i,
                sizeof(actual));
    ASSERT_EQ(actual, expected[i]) << "word=" << i;
  }
  ASSERT_NO_FATAL_FAILURE(Detach());
}

TEST_P(FabricReservationTest, RejectsUnpreparedBackingBeforeAnyReplacement) {
  ASSERT_NO_FATAL_FAILURE(PrepareBacking());
  if (projections_[2].mapping_access_ordinal != 0) {
    GTEST_SKIP() << "CPU and NPU do not share translations";
  }
  ASSERT_NO_FATAL_FAILURE(Map(0, {0, 1}));
  ASSERT_NO_FATAL_FAILURE(Map(1, {0, 2}));
  amdf_memory_profile_t profile = {};
  ASSERT_NO_FATAL_FAILURE(FindProfile(
      {&source_accesses_[1], 1}, AMDF_MEMORY_PROFILE_ROLE_CREATE, &profile));
  ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  const amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory_profile_ordinal = profile.ordinal,
      .access_count = 1,
      .required_flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE,
      .byte_length = unit_,
      .minimum_alignment = profile.allocation.minimum_alignment,
      .accesses = &source_accesses_[1],
  };
  ASSERT_NO_FATAL_FAILURE(unprepared_.Create(api_, system_scope_, create));
  const uint32_t original = 0x13579BDF;
  std::memcpy(host_ + payload_offset_, &original, sizeof(original));
  const amdf_memory_mapping_update_t updates[] = {
      {
          .target_byte_offset = 0,
          .byte_length = unit_,
          .memory = backings_[1].memory.memory,
          .memory_access_ordinal = 0,
          .access = host_access_,
          .source_byte_offset = 3 * unit_,
      },
      {
          .target_byte_offset = unit_,
          .byte_length = unit_,
          .memory = unprepared_.memory,
          .memory_access_ordinal = 0,
          .access = host_access_,
      },
  };
  // Updating through the CPU alias must still validate the NPU's attachment.
  // The valid first descriptor cannot mutate before the second is rejected.
  EXPECT_EQ(amdf_status_code(
                api_->memory_reservation_update(reservation_, 2, 2, updates)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  uint32_t observed = 0;
  std::memcpy(&observed, host_ + payload_offset_, sizeof(observed));
  EXPECT_EQ(observed, original);
  const uint32_t replacement = 0x2468ACE0;
  std::memcpy(static_cast<uint8_t*>(backings_[1].memory.host.pointer) +
                  3 * unit_ + payload_offset_,
              &replacement, sizeof(replacement));
  // Validation failure leaves the native group healthy and usable.
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 2, 1, updates),
            AMDF_STATUS_OK);
  std::memcpy(&observed, host_ + payload_offset_, sizeof(observed));
  EXPECT_EQ(observed, replacement);
}

TEST_P(FabricReservationTest, EnforcesPermissionsAtMappingGroupBoundary) {
  ASSERT_NO_FATAL_FAILURE(PrepareBacking());
  const amdf_memory_reservation_access_t consumers[] = {
      {xdna_device_, capabilities_[0].address_domain_ordinal, host_access_,
       unit_},
      {nullptr, capabilities_[2].address_domain_ordinal,
       AMDF_MEMORY_ACCESS_READ, unit_},
  };
  const amdf_memory_reservation_create_info_t reserve = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(reserve),
      .byte_length = unit_,
      .minimum_alignment = unit_,
      .access_count = 2,
      .accesses = consumers,
  };
  // Independent page tables permit different protection envelopes; a shared
  // page table cannot represent them even when both requests are supported.
  amdf_memory_reservation_t* other = nullptr;
  const auto status =
      api_->memory_reservation_create(instance_, &reserve, &other);
  if (projections_[2].mapping_access_ordinal == 0) {
    EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(other, nullptr);
  } else {
    EXPECT_EQ(status, AMDF_STATUS_OK);
    EXPECT_NE(other, nullptr);
  }
  if (other != nullptr) {
    EXPECT_EQ(api_->memory_reservation_destroy(other), AMDF_STATUS_OK);
  }
}

TEST_P(FabricReservationTest, SupportsNpuOnlyAndHostFirstProjections) {
  ASSERT_NO_FATAL_FAILURE(PrepareSources());
  const amdf_memory_reservation_access_t consumers[] = {
      {nullptr, capabilities_[2].address_domain_ordinal, host_access_, unit_},
      {xdna_device_, capabilities_[0].address_domain_ordinal, host_access_,
       unit_},
  };
  for (uint32_t count : {1u, 2u}) {
    SCOPED_TRACE(count);
    const amdf_memory_reservation_create_info_t reserve = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
        .structure_size = sizeof(reserve),
        .byte_length = unit_,
        .minimum_alignment = unit_,
        .access_count = count,
        .accesses = count == 1 ? &consumers[1] : consumers,
    };
    ASSERT_EQ(
        api_->memory_reservation_create(instance_, &reserve, &reservation_),
        AMDF_STATUS_OK);
    reservation_access_count_ = count;
    for (uint32_t i = 0; i < count; ++i) {
      amdf_memory_reservation_access_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(
          api_->memory_reservation_query_access_info(reservation_, i, &info),
          AMDF_STATUS_OK);
      EXPECT_LE(info.mapping_access_ordinal, i);
      projections_[i] = info;
      EXPECT_EQ(info.address_kind, count == 2 && i == 0
                                       ? AMDF_MEMORY_ADDRESS_HOST
                                       : AMDF_MEMORY_ADDRESS_XDNA_DMA);
      EXPECT_EQ(info.address % unit_, 0u);
    }
    const amdf_memory_mapping_update_t update = {
        .byte_length = unit_,
        .memory = backings_[0].memory.memory,
        .memory_access_ordinal = 0,
        .access = host_access_,
        .source_byte_offset = 2 * unit_,
    };
    ASSERT_EQ(
        api_->memory_reservation_update(reservation_, count - 1, 1, &update),
        AMDF_STATUS_OK);
    if (count == 2) {
      if (projections_[1].mapping_access_ordinal != 0) {
        auto host_update = update;
        host_update.memory = backings_[0].storage.memory != nullptr
                                 ? backings_[0].storage.memory
                                 : backings_[0].memory.memory;
        ASSERT_EQ(
            api_->memory_reservation_update(reservation_, 0, 1, &host_update),
            AMDF_STATUS_OK);
      }
      const amdf_memory_map_info_t map = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
          .structure_size = sizeof(map),
          .byte_length = unit_,
          .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      };
      ASSERT_EQ(api_->memory_reservation_map(reservation_, 0, &map, &view_),
                AMDF_STATUS_OK);
      amdf_host_mapping_info_t host = {
          .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
          .structure_size = sizeof(host),
      };
      ASSERT_EQ(api_->host_mapping_query_info(view_, &host), AMDF_STATUS_OK);
      const uint32_t value = 0x1A2B3C4D;
      std::memcpy(static_cast<uint8_t*>(host.pointer) + payload_offset_, &value,
                  sizeof(value));
      uint32_t observed = 0;
      std::memcpy(&observed,
                  static_cast<uint8_t*>(backings_[0].memory.host.pointer) +
                      2 * unit_ + payload_offset_,
                  sizeof(observed));
      EXPECT_EQ(observed, value);
      ASSERT_EQ(api_->host_mapping_destroy(std::exchange(view_, nullptr)),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(Detach());
    ASSERT_EQ(
        api_->memory_reservation_destroy(std::exchange(reservation_, nullptr)),
        AMDF_STATUS_OK);
  }
}

TEST_P(FabricReservationTest, ReplacesBackingAcrossCpuGpuNpuExecution) {
  ASSERT_FALSE(IsSkipped());
  ASSERT_NO_FATAL_FAILURE(PrepareBacking());
  ASSERT_NO_FATAL_FAILURE(PrepareExecution());
  std::array<uint32_t, 1024> ingress_words = {}, egress_words = {};
  Pm4CommandWriter ingress(ingress_words.data(), *pm4_profile_);
  Pm4CommandWriter egress(egress_words.data(), *pm4_profile_);
  ingress.SystemBarrier();
  egress.SystemBarrier();
  for (size_t i = 0; i < 3; ++i) {
    for (size_t offset = 0; offset < kPayloadByteLength; offset += 4) {
      if (i != 1) {
        ingress.CopyData32(staging_address_ + i * kPayloadByteLength + offset,
                           gpu_address_ + i * unit_ + payload_offset_ + offset);
      }
      egress.CopyData32(gpu_address_ + i * unit_ + payload_offset_ + offset,
                        staging_address_ + kReadbackByteOffset +
                            i * kPayloadByteLength + offset);
    }
  }
  ingress.SystemBarrier();
  egress.SystemBarrier();
  const std::array<std::array<Binding, 3>, 4> epochs = {{
      {{{0, 1}, {0, 2}, {0, 3}}},
      {{{1, 2}, {0, 2}, {0, 3}}},
      {{{1, 2}, {1, 1}, {0, 3}}},
      {{{1, 2}, {1, 1}, {1, 4}}},
  }};
  const auto commands = execution_.instructions.bytes();
  const std::vector<uint8_t> original_commands(commands.begin(),
                                               commands.end());
  for (uint32_t epoch = 0; epoch < epochs.size() && !HasFailure(); ++epoch) {
    SCOPED_TRACE(epoch);
    for (uint32_t source = 0; source < backings_.size(); ++source) {
      auto& backing = backings_[source];
      auto* words = reinterpret_cast<uint32_t*>(backing.memory.host.pointer);
      for (size_t i = 0; i < backing.memory.info.byte_length / 4; ++i) {
        words[i] = 0xA0000000u + source * 0x100000 + epoch * 0x10000 + i;
      }
      std::memcpy(backing.expected.data(), words, backing.expected.size());
      // Publish fixture sentinels before the actual stable-view CPU write.
      ASSERT_EQ(api_->host_mapping_cache_control(
                    backing.memory.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                    backing.memory.info.byte_length),
                AMDF_STATUS_OK);
    }
    if (epoch == 2) {
      // Checked removal preserves all addresses and allows explicit refill.
      ASSERT_NO_FATAL_FAILURE(Detach());
    }
    for (uint32_t i = 0; i < 3; ++i) {
      if (epoch == 0 || epoch == 2 || i == epoch - 1) {
        ASSERT_NO_FATAL_FAILURE(Map(i, epochs[epoch][i]));
      }
    }
    auto* staging = static_cast<uint32_t*>(staging_.host.pointer);
    std::array<uint32_t, 48> expected = {};
    for (uint32_t i = 0; i < 16; ++i) {
      expected[i] = 11 + 43 * epoch + 7 * i;
      expected[16 + i] = 23 + 31 * epoch + 13 * i;
      expected[32 + i] = expected[i] * expected[16 + i];
      staging[i] = expected[i];
      staging[16 + i] = expected[16 + i];
      staging[32 + i] = ~expected[32 + i];
    }
    // The workload writes and flushes through the stable view itself. It never
    // translates the cache range back to the caller's backing table.
    std::memcpy(host_ + unit_ + payload_offset_, expected.data() + 16,
                kPayloadByteLength);
    for (uint32_t i = 0; i < 3; ++i) {
      const auto binding = epochs[epoch][i];
      std::memcpy(backings_[binding.backing].expected.data() +
                      binding.unit * unit_ + payload_offset_,
                  expected.data() + 16 * i, kPayloadByteLength);
    }
    std::memset(staging_.bytes().data() + kReadbackByteOffset, 0xCD,
                3 * kPayloadByteLength);
    ASSERT_EQ(api_->host_mapping_cache_control(
                  view_, AMDF_HOST_CACHE_OPERATION_FLUSH,
                  unit_ + payload_offset_, kPayloadByteLength),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->host_mapping_cache_control(staging_.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, staging_.info.byte_length),
              AMDF_STATUS_OK);
    ASSERT_TRUE(queue_.Publish(api_, gpu_api_,
                               {ingress_words.data(), ingress.word_count()},
                               2 * epoch + 1));
    ASSERT_TRUE(queue_.WaitComplete(api_));
    ASSERT_TRUE(queue_.Retire(api_));
    const amdf_xdna_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command_count = 1,
        .commands = &execution_.command,
    };
    uint64_t point = 0;
    ASSERT_EQ(xdna_api_->kernel_queue_submit(execution_.queue, &submit, &point),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->kernel_queue_wait(execution_.queue, point,
                                      amdf::cts::kWaitTimeoutNs,
                                      amdf::cts::kWaitPollIntervalNs),
              AMDF_STATUS_OK);
    ASSERT_TRUE(queue_.Publish(api_, gpu_api_,
                               {egress_words.data(), egress.word_count()},
                               2 * epoch + 2));
    ASSERT_TRUE(queue_.WaitComplete(api_));
    ASSERT_TRUE(queue_.Retire(api_));
    ASSERT_EQ(api_->host_mapping_cache_control(
                  staging_.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                  kReadbackByteOffset, 3 * kPayloadByteLength),
              AMDF_STATUS_OK);
    ASSERT_EQ(std::memcmp(staging_.bytes().data() + kReadbackByteOffset,
                          expected.data(), 3 * kPayloadByteLength),
              0);
    // Verify the stable view before touching the original backing views, so
    // their independent verification cannot supply missing cache maintenance.
    ASSERT_EQ(api_->host_mapping_cache_control(
                  view_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, 3 * unit_),
              AMDF_STATUS_OK);
    for (uint32_t i = 0; i < 3; ++i) {
      ASSERT_EQ(std::memcmp(host_ + i * unit_ + payload_offset_,
                            expected.data() + 16 * i, kPayloadByteLength),
                0);
    }
    for (const auto& backing : backings_) {
      ASSERT_EQ(
          api_->host_mapping_cache_control(backing.memory.mapping,
                                           AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                                           0, backing.memory.info.byte_length),
          AMDF_STATUS_OK);
      ASSERT_EQ(std::memcmp(backing.memory.host.pointer,
                            backing.expected.data(), backing.expected.size()),
                0);
    }
    ASSERT_EQ(
        std::memcmp(commands.data(), original_commands.data(), commands.size()),
        0);
  }
  ASSERT_FALSE(HasFailure());
  ASSERT_NO_FATAL_FAILURE(Detach());
}

INSTANTIATE_TEST_SUITE_P(MappingUnit, FabricReservationTest,
                         ::testing::Values(uint64_t{0}, UINT64_C(2) << 20),
                         [](const auto& info) {
                           return info.param == 0 ? "Native" : "TwoMiB";
                         });
}  // namespace
