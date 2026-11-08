// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>
#include <vector>

#include "libamdf/cts/xdna/programs/mul_i32.h"
#include "libamdf/cts/xdna/util/executable.h"
#include "libamdf/cts/xdna/util/execution.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"
#include "util/mapped_memory.h"
#include "util/wait.h"

namespace {

constexpr auto kReadWrite = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;

struct Binding {
  // Explicitly retained source allocation.
  uint32_t backing;
  // Mapping-unit offset within the source allocation.
  uint32_t unit;
};

class XdnaReservationTest
    : public XdnaDeviceFixture,
      public ::testing::WithParamInterface<
          std::tuple<amdf_memory_profile_roles_t, uint64_t>> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaDeviceFixture::SetUp());
    if (IsSkipped()) {
      return;
    }
    capabilities_.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES;
    capabilities_.structure_size = sizeof(capabilities_);
    const auto status = api_->device_query_memory_reservation_capabilities(
        device_, 0, &capabilities_);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native NPU reservations are unavailable";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    role_ = std::get<0>(GetParam());
    memory_access_.requirements.access |= capabilities_.required_access;
    memory_access_.requirements.flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
    memory_access_.requirements.address_kinds = UINT64_C(1)
                                                << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    amdf_memory_scope_info_t scope = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(scope),
    };
    ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope),
              AMDF_STATUS_OK);
    bool found = false;
    for (uint32_t ordinal = 0; ordinal < scope.memory_profile_count && !found;
         ++ordinal) {
      profile_.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile_.structure_size = sizeof(profile_);
      amdf_memory_access_capabilities_t access = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(access),
      };
      const auto query_status = QueryMemoryProfile(ordinal, &profile_, &access);
      if (query_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(query_status, AMDF_STATUS_OK);
      found =
          amdf_status_is_ok(profile_.availability) &&
          (profile_.roles & role_) != 0 &&
          (profile_.supported_flags & capabilities_.required_source_flags) ==
              capabilities_.required_source_flags;
    }
    if (!found) {
      GTEST_SKIP() << "ordinary source role " << role_
                   << " cannot supply independent NPU mappings";
    }
    const auto& geometry = role_ == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                               ? profile_.registration
                               : profile_.allocation;
    unit_ = std::max({std::get<1>(GetParam()),
                      capabilities_.reservation_granularity,
                      capabilities_.mapping_granularity,
                      geometry.native_byte_length_granularity});
    payload_offset_ = unit_ - 128;
    ASSERT_GE(unit_, 128u);
    ASSERT_LE(6 * unit_, geometry.maximum_byte_length);
    ASSERT_TRUE(FindXdnaKernelQueueFamily(api_, endpoint_, &family_));
  }

  void TearDown() override {
    if (!execution_.Release(api_, xdna_api_)) {
      return;
    }
    if (reservation_ != nullptr) {
      const auto detached = api_->memory_reservation_detach(reservation_, 0);
      EXPECT_EQ(detached, AMDF_STATUS_OK);
      const auto destroyed = api_->memory_reservation_destroy(reservation_);
      reservation_ = nullptr;
      EXPECT_EQ(destroyed, AMDF_STATUS_OK);
      if (!amdf_status_is_ok(detached) && !amdf_status_is_ok(destroyed)) {
        return;
      }
    }
    for (auto& source : sources_) {
      if (!source.Release(api_)) {
        return;
      }
    }
    for (auto& storage : storage_) {
      if (!storage.Release(api_)) {
        return;
      }
    }
  }

  void Prepare() {
    const auto& geometry = role_ == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                               ? profile_.registration
                               : profile_.allocation;
    amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = profile_.ordinal,
        .access_count = 1,
        .required_flags =
            AMDF_MEMORY_FLAG_HOST_VISIBLE | capabilities_.required_source_flags,
        .byte_length = 6 * unit_,
        .minimum_alignment = geometry.minimum_alignment,
        .accesses = &memory_access_,
    };
    for (size_t i = 0; i < sources_.size(); ++i) {
      if (role_ == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        const amdf_memory_create_info_t host = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
            .structure_size = sizeof(host),
            .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
            .byte_length = create.byte_length,
            .minimum_alignment = geometry.registered_host_pointer_alignment,
        };
        ASSERT_NO_FATAL_FAILURE(storage_[i].Create(api_, system_scope_, host));
        create.registered_host_pointer = storage_[i].host.pointer;
        create.registered_host_properties = storage_[i].host.properties;
      }
      ASSERT_NO_FATAL_FAILURE(sources_[i].Create(api_, system_scope_, create));
      expected_[i].resize(create.byte_length);
    }
    const amdf_memory_reservation_access_t access = {
        .device = device_,
        .address_domain_ordinal = capabilities_.address_domain_ordinal,
        .access = kReadWrite,
        .mapping_granularity = std::get<1>(GetParam()),
    };
    const amdf_memory_reservation_create_info_t reserve = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
        .structure_size = sizeof(reserve),
        .byte_length = 4 * unit_,
        .minimum_alignment = unit_,
        .access_count = 1,
        .accesses = &access,
    };
    ASSERT_EQ(
        api_->memory_reservation_create(instance_, &reserve, &reservation_),
        AMDF_STATUS_OK);
    projection_.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO;
    projection_.structure_size = sizeof(projection_);
    ASSERT_EQ(api_->memory_reservation_query_access_info(reservation_, 0,
                                                         &projection_),
              AMDF_STATUS_OK);
    ASSERT_EQ(projection_.address_kind, AMDF_MEMORY_ADDRESS_XDNA_DMA);
    ASSERT_EQ(
        projection_.mapping_granularity,
        std::max(std::get<1>(GetParam()), capabilities_.mapping_granularity));
    ASSERT_EQ(projection_.address % unit_, 0u);

    amdf_xdna_endpoint_info_t endpoint = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
        .structure_size = sizeof(endpoint),
    };
    amdf_xdna_device_info_t device = {
        .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
        .structure_size = sizeof(device),
    };
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &endpoint),
              AMDF_STATUS_OK);
    ASSERT_EQ(xdna_api_->device_query_info(device_, &device), AMDF_STATUS_OK);
    RecordProperty("amdf_xdna_target", endpoint.target_id);
    XdnaExecutable executable;
    constexpr std::array<amdf_memory_access_t, 3> binding_access = {
        AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_READ,
        AMDF_MEMORY_ACCESS_WRITE};
    bool admitted = false;
    const auto* images = amdf_cts_xdna_mul_i32_create();
    for (size_t i = 0; i < amdf_cts_xdna_mul_i32_size(); ++i) {
      const auto& image = images[i];
      if (executable.Initialize(
              {reinterpret_cast<const uint8_t*>(image.data), image.size},
              endpoint, device, 1, binding_access)) {
        admitted = true;
        break;
      }
    }
    ASSERT_TRUE(admitted) << "no compatible finite arithmetic fixture";
    std::vector<uint8_t> image_storage(executable.allocation_byte_length());
    executable.Load(image_storage);
    const std::array<uint64_t, 3> bindings = {
        projection_.address + payload_offset_,
        projection_.address + unit_ + payload_offset_,
        projection_.address + 2 * unit_ + payload_offset_,
    };
    ASSERT_TRUE(executable.Bind(image_storage, bindings));
    ASSERT_NO_FATAL_FAILURE(
        execution_.Prepare(api_, xdna_api_, device_, family_, 1,
                           executable.ResolveInvocation(image_storage),
                           executable.allocation_alignment()));
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            execution_.instructions.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
            execution_.instructions.host.byte_length),
        AMDF_STATUS_OK);
  }

  amdf_memory_mapping_update_t Mapping(uint32_t slot, Binding binding) {
    return {
        .target_byte_offset = slot * unit_,
        .byte_length = unit_,
        .memory = sources_[binding.backing].memory,
        .access = kReadWrite,
        .source_byte_offset = binding.unit * unit_,
    };
  }

  void Execute() {
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
  }

  void RunEpoch(uint32_t epoch, const std::array<Binding, 3>& bindings) {
    for (size_t i = 0; i < sources_.size(); ++i) {
      const auto bytes = sources_[i].bytes();
      std::fill(bytes.begin(), bytes.end(),
                static_cast<uint8_t>(0x80 + epoch * 7 + i));
    }
    std::array<uint32_t*, 3> words = {};
    for (size_t i = 0; i < bindings.size(); ++i) {
      words[i] = reinterpret_cast<uint32_t*>(
          sources_[bindings[i].backing].bytes().data() +
          bindings[i].unit * unit_ + payload_offset_);
    }
    for (uint32_t i = 0; i < 16; ++i) {
      words[0][i] = epoch * 31 + i * 7 + 13;
      words[1][i] = epoch * 17 + i * 3 + 23;
      words[2][i] = ~(words[0][i] * words[1][i]);
    }
    for (size_t i = 0; i < sources_.size(); ++i) {
      std::copy(sources_[i].bytes().begin(), sources_[i].bytes().end(),
                expected_[i].begin());
      ASSERT_EQ(api_->host_mapping_cache_control(
                    sources_[i].mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                    sources_[i].host.byte_length),
                AMDF_STATUS_OK);
    }
    for (size_t i = 0; i < 16; ++i) {
      const uint32_t product = words[0][i] * words[1][i];
      std::memcpy(expected_[bindings[2].backing].data() +
                      bindings[2].unit * unit_ + payload_offset_ +
                      i * sizeof(product),
                  &product, sizeof(product));
    }
    ASSERT_NO_FATAL_FAILURE(Execute());
    // Capture the output owner first, before any neighbor or source checks.
    for (size_t i :
         {size_t{bindings[2].backing}, size_t{1 - bindings[2].backing}}) {
      ASSERT_EQ(api_->host_mapping_cache_control(
                    sources_[i].mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                    0, sources_[i].host.byte_length),
                AMDF_STATUS_OK);
      EXPECT_EQ(std::memcmp(sources_[i].host.pointer, expected_[i].data(),
                            expected_[i].size()),
                0)
          << "full backing " << i << " at epoch " << epoch;
    }
  }

  // Queried native address-space geometry.
  amdf_memory_reservation_capabilities_t capabilities_ = {};
  // Selected source construction contract.
  amdf_memory_profile_t profile_ = {};
  // Parameter-selected construction role.
  amdf_memory_profile_roles_t role_ = 0;
  // Caller-selected independently replaceable unit, in bytes.
  uint64_t unit_ = 0;
  // Last cache-aligned payload near the end of each unit.
  uint64_t payload_offset_ = 0;
  // Reused native queue family.
  uint32_t family_ = 0;
  // Optional CPU owners retained through registered source destruction.
  std::array<CtsMappedMemory, 2> storage_;
  // Ordinary retained NPU sources; replacements never reconstruct these.
  std::array<CtsMappedMemory, 2> sources_;
  // Full expected source contents, including old output and untouched
  // neighbors.
  std::array<std::vector<uint8_t>, 2> expected_;
  // Case-owned stable NPU address range.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Immutable projection queried before installing any backing.
  amdf_memory_reservation_access_info_t projection_ = {};
  // One context, queue and immutable command used across all replacements.
  XdnaExecution execution_;
};

TEST_P(XdnaReservationTest, ReplacesRetainedBackingWithImmediateExecution) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  const std::array<std::array<Binding, 3>, 6> epochs = {{
      // Replacing one alias must not remove the other registration of the
      // same retained physical pages.
      {{{0, 1}, {0, 1}, {0, 3}}},
      {{{0, 1}, {0, 2}, {0, 3}}},
      {{{1, 2}, {0, 2}, {0, 3}}},
      {{{1, 2}, {1, 1}, {0, 3}}},
      {{{1, 2}, {1, 1}, {1, 4}}},
      {{{0, 1}, {0, 2}, {0, 3}}},
  }};
  for (uint32_t epoch = 0; epoch < epochs.size() && !HasFailure(); ++epoch) {
    SCOPED_TRACE(epoch);
    std::vector<amdf_memory_mapping_update_t> updates;
    for (uint32_t slot = 0; slot < 3; ++slot) {
      const auto binding = epochs[epoch][slot];
      if (epoch == 0 || binding.backing != epochs[epoch - 1][slot].backing ||
          binding.unit != epochs[epoch - 1][slot].unit) {
        updates.push_back(Mapping(slot, binding));
        if (epoch) {
          const auto previous = epochs[epoch - 1][slot];
          updates.back().previous_memory = sources_[previous.backing].memory;
          updates.back().previous_source_byte_offset = previous.unit * unit_;
        }
      }
    }
    ASSERT_EQ(api_->memory_reservation_update(
                  reservation_, 0, static_cast<uint32_t>(updates.size()),
                  updates.data()),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(RunEpoch(epoch, epochs[epoch]));
  }
  // Remove and refill only the middle unit while both neighbors remain live.
  const amdf_memory_mapping_update_t hole = {
      .target_byte_offset = unit_,
      .byte_length = unit_,
      .previous_memory = sources_[epochs.back()[1].backing].memory,
      .previous_source_byte_offset = epochs.back()[1].unit * unit_,
  };
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &hole),
            AMDF_STATUS_OK);
  const auto middle = Mapping(1, epochs.back()[1]);
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &middle),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(RunEpoch(6, epochs.back()));
  ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
  // Keep the address owner and execution alive; refill after checked removal.
  const std::array<amdf_memory_mapping_update_t, 3> refill = {
      Mapping(0, epochs[0][0]),
      Mapping(1, epochs[0][1]),
      Mapping(2, epochs[0][2]),
  };
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, refill.size(),
                                            refill.data()),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(RunEpoch(7, epochs[0]));
  ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
}

TEST_P(XdnaReservationTest, FirstDeviceWriterNeedsNoOutputPayloadPreparation) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  const std::array<amdf_memory_mapping_update_t, 3> mappings = {
      Mapping(0, {0, 1}),
      Mapping(1, {0, 2}),
      Mapping(2, {1, 3}),
  };
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, mappings.size(),
                                            mappings.data()),
            AMDF_STATUS_OK);
  auto* left = reinterpret_cast<uint32_t*>(sources_[0].bytes().data() + unit_ +
                                           payload_offset_);
  auto* right = reinterpret_cast<uint32_t*>(sources_[0].bytes().data() +
                                            2 * unit_ + payload_offset_);
  std::array<uint32_t, 16> expected;
  for (uint32_t i = 0; i < expected.size(); ++i) {
    left[i] = i + 19;
    right[i] = i * 3 + 31;
    expected[i] = left[i] * right[i];
  }
  ASSERT_EQ(api_->host_mapping_cache_control(sources_[0].mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             sources_[0].host.byte_length),
            AMDF_STATUS_OK);
  // No output load/store or cache operation has occurred since allocation.
  ASSERT_NO_FATAL_FAILURE(Execute());
  ASSERT_EQ(api_->host_mapping_cache_control(
                sources_[1].mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                3 * unit_ + payload_offset_, sizeof(expected)),
            AMDF_STATUS_OK);
  EXPECT_EQ(
      std::memcmp(sources_[1].bytes().data() + 3 * unit_ + payload_offset_,
                  expected.data(), sizeof(expected)),
      0);
  ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(
    SourceAndUnit, XdnaReservationTest,
    ::testing::Combine(::testing::Values(AMDF_MEMORY_PROFILE_ROLE_CREATE,
                                         AMDF_MEMORY_PROFILE_ROLE_REGISTER),
                       ::testing::Values(UINT64_C(0),
                                         UINT64_C(2) * 1024 * 1024)));

}  // namespace
