// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "libamdf/cts/gpu/pm4/command_fixture.h"

namespace {

class Pm4HostMappingTest
    : public Pm4CommandTest,
      public ::testing::WithParamInterface<
          std::tuple<uint64_t, amdf_memory_reservation_flags_t>> {
 protected:
  void ExpectPairEqual(const amdf_memory_pair_info_t& actual,
                       const amdf_memory_pair_info_t& expected) {
    EXPECT_EQ(actual.flags, expected.flags);
    EXPECT_EQ(actual.atomic_reach.scope_32, expected.atomic_reach.scope_32);
    EXPECT_EQ(actual.atomic_reach.scope_64, expected.atomic_reach.scope_64);
    EXPECT_EQ(actual.estimated_fixed_cost_nanoseconds,
              expected.estimated_fixed_cost_nanoseconds);
    const amdf_cache_transition_t* actual_sides[] = {&actual.release,
                                                     &actual.acquire};
    const amdf_cache_transition_t* expected_sides[] = {&expected.release,
                                                       &expected.acquire};
    for (size_t i = 0; i < 2; ++i) {
      SCOPED_TRACE(i);
      const auto& a = *actual_sides[i];
      const auto& b = *expected_sides[i];
      EXPECT_EQ(a.kind, b.kind);
      EXPECT_EQ(a.executor, b.executor);
      EXPECT_EQ(a.operation, b.operation);
      EXPECT_EQ(a.host_operation, b.host_operation);
      EXPECT_EQ(a.host_instruction, b.host_instruction);
      EXPECT_EQ(a.host_fence_before, b.host_fence_before);
      EXPECT_EQ(a.host_fence_after, b.host_fence_after);
      EXPECT_EQ(a.range_granularity, b.range_granularity);
    }
  }

  void TearDown() override {
    if (queue_ != nullptr) {
      ASSERT_TRUE(queue_->Release(api_));
    }
    if (view_ != nullptr) {
      ASSERT_EQ(api_->host_mapping_destroy(view_), AMDF_STATUS_OK);
    }
    if (reservation_ != nullptr) {
      bool detached = true;
      for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
        const amdf_status_t status =
            api_->memory_reservation_detach(reservation_, ordinal);
        EXPECT_EQ(status, AMDF_STATUS_OK);
        detached &= amdf_status_is_ok(status);
      }
      const amdf_status_t release =
          api_->memory_reservation_destroy(reservation_);
      EXPECT_EQ(release, AMDF_STATUS_OK);
      if (!detached && !amdf_status_is_ok(release)) {
        return;
      }
    }
    Pm4CommandTest::TearDown();
  }

  // Case-owned execution, removed before aliases or backing are released.
  GpuCommandQueue* queue_ = nullptr;
  // Address owner independent of both retained backing allocations.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Unchanged direct-cache view borrowing reservation addresses.
  amdf_host_mapping_t* view_ = nullptr;
};

TEST_P(Pm4HostMappingTest, ReplacesCpuAndGpuViewsBeforeImmediateUse) {
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  amdf_memory_reservation_capabilities_t gpu_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      .structure_size = sizeof(gpu_capabilities),
  };
  ASSERT_EQ(api_->device_query_memory_reservation_capabilities(
                device_, 0, &gpu_capabilities),
            AMDF_STATUS_OK);
  const amdf_memory_device_access_t consumer = {
      device_,
      {
          .access = kReadWrite | gpu_capabilities.required_access,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                   AMDF_MEMORY_FLAG_HOST_COHERENT |
                   AMDF_MEMORY_FLAG_MAPPING_SOURCE,
      }};
  const amdf_memory_flags_t flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
  amdf_memory_scope_info_t scope = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope),
  };
  ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope),
            AMDF_STATUS_OK);
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  amdf_memory_access_capabilities_t access_profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(access_profile),
  };
  uint32_t profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  for (uint32_t ordinal = 0; ordinal < scope.memory_profile_count; ++ordinal) {
    const auto status = api_->memory_scope_query_device_profile(
        system_scope_, ordinal, 1, &consumer, &profile, &access_profile);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) != 0 &&
        (flags & ~profile.supported_flags) == 0) {
      profile_ordinal = ordinal;
      break;
    }
  }
  if (profile_ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP() << "GPU backing does not supply replaceable host aliases";
  }
  // Qualification precedes both the address reservation and payload creation.
  // The same recipes must hold through every later source replacement.
  amdf_memory_profile_pair_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
      .structure_size = sizeof(query),
      .memory_profile_ordinal = profile_ordinal,
      .access_count = 1,
      .required_flags = flags,
      .accesses = &consumer,
  };
  query.producer.kind = AMDF_MEMORY_SITE_KIND_HOST;
  query.producer.flags = AMDF_MEMORY_PROFILE_SITE_FLAG_HOST_RESERVATION;
  query.producer.value.host_access =
      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  query.consumer.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
  query.consumer.value.device.queue_family_ordinal = family_.ordinal;
  std::array<amdf_memory_pair_info_t, 2> prospective = {};
  for (auto& pair : prospective) {
    pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair.structure_size = sizeof(pair);
    ASSERT_EQ(api_->memory_scope_query_pair_info(system_scope_, &query, &pair),
              AMDF_STATUS_OK);
    std::swap(query.producer, query.consumer);
  }
  RecordProperty("backing_profile_ordinal", profile_ordinal);
  amdf_memory_reservation_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      .structure_size = sizeof(capabilities),
  };
  ASSERT_EQ(api_->instance_query_host_memory_reservation_capabilities(
                instance_, &capabilities),
            AMDF_STATUS_OK);
  const uint64_t unit =
      std::max({std::get<0>(GetParam()), capabilities.mapping_granularity,
                gpu_capabilities.mapping_granularity,
                capabilities.reservation_granularity,
                gpu_capabilities.reservation_granularity,
                profile.allocation.native_byte_length_granularity});
  const size_t words = unit / sizeof(uint32_t);
  const amdf_memory_reservation_access_t accesses[] = {
      {nullptr, capabilities.address_domain_ordinal,
       kReadWrite | capabilities.required_access, unit},
      {device_, gpu_capabilities.address_domain_ordinal,
       consumer.requirements.access, unit},
  };
  const amdf_memory_reservation_create_info_t reservation_create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(reservation_create),
      .byte_length = 2 * unit,
      .minimum_alignment = unit,
      .access_count = 2,
      .flags = std::get<1>(GetParam()),
      .accesses = accesses,
  };
  ASSERT_EQ(reservation_create.flags & ~capabilities.supported_flags, 0u);
  ASSERT_EQ(reservation_create.flags & ~gpu_capabilities.supported_flags, 0u);
  ASSERT_EQ(api_->memory_reservation_create(instance_, &reservation_create,
                                            &reservation_),
            AMDF_STATUS_OK);
  amdf_memory_reservation_access_info_t gpu_address = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
      .structure_size = sizeof(gpu_address),
  };
  ASSERT_EQ(
      api_->memory_reservation_query_access_info(reservation_, 1, &gpu_address),
      AMDF_STATUS_OK);
  ASSERT_EQ(gpu_address.mapping_access_ordinal, 1u);
  const amdf_memory_map_info_t map = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map),
      .byte_length = 2 * unit,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  ASSERT_EQ(api_->memory_reservation_map(reservation_, 0, &map, &view_),
            AMDF_STATUS_OK);
  amdf_host_mapping_info_t host = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(host),
  };
  ASSERT_EQ(api_->host_mapping_query_info(view_, &host), AMDF_STATUS_OK);
  auto* pointer = static_cast<uint32_t*>(host.pointer);
  if (reservation_create.flags != 0) {
    ASSERT_EQ(reinterpret_cast<uintptr_t>(pointer), gpu_address.address);
  }
  const amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory_profile_ordinal = profile_ordinal,
      .access_count = 1,
      .required_flags = flags,
      .byte_length = 6 * unit,
      // Only reservation aliases need the requested mapping-unit alignment.
      // The persistent source view follows the allocation profile's geometry.
      .minimum_alignment = profile.allocation.minimum_alignment,
      .accesses = &consumer,
  };
  std::array<GpuMemory*, 2> backings = {};
  std::array<std::vector<uint32_t>, 2> expected;
  for (size_t i = 0; i < backings.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, create, &backings[i]));
    expected[i].resize(6 * words);
    for (size_t j = 0; j < expected[i].size(); ++j) {
      expected[i][j] = 0x91a73b45u ^ static_cast<uint32_t>(i * 0x1000000u + j);
    }
    std::memcpy(backings[i]->host.pointer, expected[i].data(), 6 * unit);
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            backings[i]->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 6 * unit),
        AMDF_STATUS_OK);
  }
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &completion));
  *static_cast<uint32_t*>(completion->host.pointer) = 0;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue_));
  ASSERT_GE(queue_->words().size_bytes(), 2048u);
  Pm4CommandWriter commands(queue_->words().data(), *pm4_profile_);
  constexpr size_t owners[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  constexpr size_t offsets[4][2] = {{1, 3}, {2, 3}, {2, 4}, {4, 4}};
  for (uint32_t epoch = 0; epoch < 4; ++epoch) {
    SCOPED_TRACE(epoch);
    std::array<amdf_memory_mapping_update_t, 2> updates = {};
    for (size_t slot = 0; slot < updates.size(); ++slot) {
      updates[slot] = {
          .target_byte_offset = slot * unit,
          .byte_length = unit,
          .memory = backings[owners[epoch][slot]]->memory,
          .access = accesses[0].access,
          .source_byte_offset = offsets[epoch][slot] * unit,
          .previous_memory =
              epoch ? backings[owners[epoch - 1][slot]]->memory : nullptr,
          .previous_source_byte_offset =
              epoch ? offsets[epoch - 1][slot] * unit : 0,
      };
    }
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, updates.size(),
                                              updates.data()),
              AMDF_STATUS_OK);
    for (auto& update : updates) {
      update.access = accesses[1].access;
    }
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 1, updates.size(),
                                              updates.data()),
              AMDF_STATUS_OK);
    amdf_memory_site_t host_site = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
        .structure_size = sizeof(host_site),
        .kind = AMDF_MEMORY_SITE_KIND_HOST,
    };
    host_site.value.host = {view_, updates[0].memory};
    const auto device_site =
        backings[owners[epoch][0]]->DeviceSite(family_.ordinal);
    amdf_memory_pair_info_t pair = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(pair),
    };
    ASSERT_EQ(api_->memory_query_pair_info(&host_site, &device_site, &pair),
              AMDF_STATUS_OK);
    ExpectPairEqual(pair, prospective[0]);
    ASSERT_EQ(api_->memory_query_pair_info(&device_site, &host_site, &pair),
              AMDF_STATUS_OK);
    ExpectPairEqual(pair, prospective[1]);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    for (size_t word = 0; word < words; ++word) {
      const uint32_t value =
          0x51389abcu ^ (epoch * 0x1010101u + static_cast<uint32_t>(word));
      pointer[word] = value;
      for (size_t slot = 0; slot < updates.size(); ++slot) {
        expected[owners[epoch][slot]][offsets[epoch][slot] * words + word] =
            value;
      }
    }
    ASSERT_EQ(api_->host_mapping_cache_control(
                  view_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, unit),
              AMDF_STATUS_OK);
    commands.SystemBarrier();
    commands.DmaCopy(gpu_address.address, gpu_address.address + unit,
                     static_cast<uint32_t>(unit));
    commands.WaitDma();
    commands.SystemBarrier();
    commands.WriteData32(completion->device_address, epoch + 1);
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(
        queue_->Publish(api_, gpu_api_, commands.word_count()));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer), epoch + 1));
    ASSERT_EQ(api_->host_mapping_cache_control(
                  view_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, unit, unit),
              AMDF_STATUS_OK);
    // Observe the stable view before touching the original source mappings.
    EXPECT_EQ(std::memcmp(
                  pointer + words,
                  expected[owners[epoch][1]].data() + offsets[epoch][1] * words,
                  unit),
              0);
    for (size_t i = 0; i < backings.size(); ++i) {
      ASSERT_EQ(api_->host_mapping_cache_control(
                    backings[i]->mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                    0, 6 * unit),
                AMDF_STATUS_OK);
      EXPECT_EQ(
          std::memcmp(backings[i]->host.pointer, expected[i].data(), 6 * unit),
          0)
          << "backing " << i;
    }
    ASSERT_NO_FATAL_FAILURE(queue_->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
  }
  ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
  ASSERT_EQ(api_->memory_reservation_detach(reservation_, 1), AMDF_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(
    MappingUnit, Pm4HostMappingTest,
    ::testing::Combine(
        ::testing::Values(uint64_t{0}, UINT64_C(2) << 20),
        ::testing::Values(
            amdf_memory_reservation_flags_t{0},
            amdf_memory_reservation_flags_t{
                AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS})),
    [](const auto& info) {
      return std::string(std::get<0>(info.param) == 0 ? "Native" : "TwoMiB") +
             (std::get<1>(info.param) == 0 ? "Independent" : "Common");
    });

}  // namespace
