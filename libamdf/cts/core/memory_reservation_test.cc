// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "amdf/amdf.h"
#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

class HostReservationTest : public ::testing::TestWithParam<uint64_t> {
 protected:
  void SetUp() override {
    ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_LATEST,
                                            AMDF_ABI_VERSION_LATEST, &api_),
              AMDF_STATUS_OK);
    ASSERT_EQ(GetCtsDeviceCache().GetInstance(&instance_), AMDF_STATUS_OK);
    capabilities_.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES;
    capabilities_.structure_size = sizeof(capabilities_);
    ASSERT_EQ(api_->instance_query_host_memory_reservation_capabilities(
                  instance_, &capabilities_),
              AMDF_STATUS_OK);
    uint32_t count = 0;
    ASSERT_EQ(amdf_status_code(api_->instance_enumerate_memory_scopes(
                  instance_, 0, nullptr, &count)),
              AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    std::vector<amdf_memory_scope_t*> scopes(count);
    ASSERT_EQ(api_->instance_enumerate_memory_scopes(instance_, count,
                                                     scopes.data(), &count),
              AMDF_STATUS_OK);
    for (amdf_memory_scope_t* scope : scopes) {
      amdf_memory_scope_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      if (info.kind != AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
        continue;
      }
      scope_ = scope;
      for (uint32_t ordinal = 0; ordinal < info.memory_profile_count;
           ++ordinal) {
        amdf_memory_profile_t profile = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
            .structure_size = sizeof(profile),
        };
        const amdf_status_t status = api_->memory_scope_query_device_profile(
            scope, ordinal, 0, nullptr, &profile, nullptr);
        if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
          continue;
        }
        ASSERT_EQ(status, AMDF_STATUS_OK);
        if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) != 0 &&
            (profile.supported_flags & capabilities_.required_source_flags) ==
                capabilities_.required_source_flags) {
          profile_ = profile;
        }
      }
    }
    ASSERT_NE(scope_, nullptr);
    ASSERT_NE(profile_.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE, 0u);
    unit_ = std::max(GetParam(), capabilities_.mapping_granularity);
    ASSERT_EQ(unit_ % capabilities_.mapping_granularity, 0u);
    access_ = capabilities_.required_access | AMDF_MEMORY_ACCESS_READ |
              AMDF_MEMORY_ACCESS_WRITE;
    ASSERT_EQ(access_ & ~capabilities_.supported_access, 0u);
  }

  void TearDown() override {
    for (amdf_host_mapping_t* view : views_) {
      EXPECT_EQ(api_->host_mapping_destroy(view), AMDF_STATUS_OK);
    }
    if (reservation_ != nullptr) {
      // Checked removal happens while the address owner and all sources live.
      // A failing detach cannot authorize recycling any possibly mapped source.
      const amdf_status_t detach =
          api_->memory_reservation_detach(reservation_, 0);
      EXPECT_EQ(detach, AMDF_STATUS_OK);
      const amdf_status_t release =
          api_->memory_reservation_destroy(reservation_);
      EXPECT_EQ(release, AMDF_STATUS_OK);
      if (!amdf_status_is_ok(detach) && !amdf_status_is_ok(release)) {
        return;
      }
    }
    for (amdf_memory_t* memory : sources_) {
      EXPECT_EQ(api_->memory_destroy(memory), AMDF_STATUS_OK);
    }
  }

  void Reserve(uint64_t length) {
    amdf_memory_reservation_access_t access = {
        .address_domain_ordinal = capabilities_.address_domain_ordinal,
        .access = access_,
        .mapping_granularity = unit_,
    };
    amdf_memory_reservation_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
        .structure_size = sizeof(create),
        .byte_length = length,
        .minimum_alignment = unit_,
        .access_count = 1,
        .accesses = &access,
    };
    ASSERT_EQ(
        api_->memory_reservation_create(instance_, &create, &reservation_),
        AMDF_STATUS_OK);
    address_.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO;
    address_.structure_size = sizeof(address_);
    ASSERT_EQ(
        api_->memory_reservation_query_access_info(reservation_, 0, &address_),
        AMDF_STATUS_OK);
    ASSERT_EQ(address_.address_kind, AMDF_MEMORY_ADDRESS_HOST);
    ASSERT_EQ(address_.mapping_granularity, unit_);
    ASSERT_EQ(address_.address % unit_, 0u);
  }

  void Allocate(uint64_t length, amdf_memory_flags_t flags,
                amdf_memory_t** out_memory, uint32_t** out_pointer) {
    amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal = profile_.ordinal,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE | flags,
        .byte_length = length,
    };
    amdf_memory_t* memory = nullptr;
    ASSERT_EQ(api_->memory_create(scope_, &create, &memory), AMDF_STATUS_OK);
    sources_.push_back(memory);
    amdf_memory_map_info_t request = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(request),
        .byte_length = length,
        .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    amdf_host_mapping_t* mapping = nullptr;
    ASSERT_EQ(api_->memory_map(memory, &request, &mapping), AMDF_STATUS_OK);
    views_.push_back(mapping);
    amdf_host_mapping_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->host_mapping_query_info(mapping, &info), AMDF_STATUS_OK);
    *out_memory = memory;
    *out_pointer = static_cast<uint32_t*>(info.pointer);
  }

  void View(uint64_t offset, uint64_t length, amdf_host_mapping_t** out_view,
            uint32_t** out_pointer) {
    amdf_memory_map_info_t request = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(request),
        .byte_offset = offset,
        .byte_length = length,
        .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    amdf_host_mapping_t* mapping = nullptr;
    ASSERT_EQ(api_->memory_reservation_map(reservation_, 0, &request, &mapping),
              AMDF_STATUS_OK);
    views_.push_back(mapping);
    amdf_host_mapping_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->host_mapping_query_info(mapping, &info), AMDF_STATUS_OK);
    ASSERT_EQ(info.memory_byte_offset, offset);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(info.pointer),
              address_.address + offset);
    EXPECT_EQ(info.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    *out_view = mapping;
    *out_pointer = static_cast<uint32_t*>(info.pointer);
  }

  amdf_memory_mapping_update_t Bind(uint64_t target, uint64_t length,
                                    amdf_memory_t* source, uint64_t offset) {
    return {
        .target_byte_offset = target,
        .byte_length = length,
        .memory = source,
        .access = access_,
        .source_byte_offset = offset,
    };
  }

  // Immutable API borrowed from the CTS loader.
  const amdf_api_t* api_ = nullptr;
  // Shared instance; these cases never activate a device.
  amdf_instance_t* instance_ = nullptr;
  // Passive system scope supplying CPU-only backing.
  amdf_memory_scope_t* scope_ = nullptr;
  // Queried CPU CREATE contract with explicit alias eligibility.
  amdf_memory_profile_t profile_ = {};
  // Native host geometry used to construct each request.
  amdf_memory_reservation_capabilities_t capabilities_ = {};
  // Independently replaceable byte extent selected for this case.
  uint64_t unit_ = 0;
  // Requested permissions, including native requirements.
  amdf_memory_access_t access_ = 0;
  // Sole address owner, destroyed after all of its lightweight views.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Stable address and geometry queried before backing exists.
  amdf_memory_reservation_access_info_t address_ = {};
  // Explicit view owners, borrowing either backing or reservation addresses.
  std::vector<amdf_host_mapping_t*> views_;
  // Physical owners retained through checked final detach.
  std::vector<amdf_memory_t*> sources_;
};

TEST_P(HostReservationTest, ReplacesBackingThroughAnUnchangedHostView) {
  ASSERT_NO_FATAL_FAILURE(Reserve(3 * unit_));
  amdf_host_mapping_t* view = nullptr;
  uint32_t* pointer = nullptr;
  ASSERT_NO_FATAL_FAILURE(View(0, 3 * unit_, &view, &pointer));
  std::array<amdf_memory_t*, 2> memories = {};
  std::array<uint32_t*, 2> original = {};
  std::array<std::vector<uint32_t>, 2> expected;
  const size_t words = unit_ / sizeof(uint32_t);
  for (size_t i = 0; i < memories.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(Allocate(6 * unit_,
                                     capabilities_.required_source_flags,
                                     &memories[i], &original[i]));
    expected[i].resize(6 * words);
    for (size_t j = 0; j < expected[i].size(); ++j) {
      original[i][j] = expected[i][j] =
          static_cast<uint32_t>(i * 0x10000000u + j);
    }
  }
  // Every epoch changes one destination while retaining independently checked
  // old backing and neighbors. No physical allocation is freed during rotation.
  constexpr size_t backing[4][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}};
  constexpr size_t offsets[4][3] = {{1, 2, 3}, {2, 2, 3}, {2, 1, 3}, {2, 1, 4}};
  for (size_t epoch = 0; epoch < 4; ++epoch) {
    SCOPED_TRACE(epoch);
    std::vector<amdf_memory_mapping_update_t> updates;
    for (size_t slot = 0; slot < 3; ++slot) {
      if (epoch == 0 || backing[epoch][slot] != backing[epoch - 1][slot]) {
        updates.push_back(Bind(slot * unit_, unit_,
                               memories[backing[epoch][slot]],
                               offsets[epoch][slot] * unit_));
      }
    }
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, updates.size(),
                                              updates.data()),
              AMDF_STATUS_OK);
    for (size_t slot = 0; slot < 3; ++slot) {
      auto& source = expected[backing[epoch][slot]];
      ASSERT_EQ(
          std::memcmp(pointer + slot * words,
                      source.data() + offsets[epoch][slot] * words, unit_),
          0);
      for (size_t word = 0; word < words; ++word) {
        const uint32_t value = static_cast<uint32_t>(
            0x80000000u | ((epoch + 1) << 24) | (slot << 20) | word);
        pointer[slot * words + word] = value;
        source[offsets[epoch][slot] * words + word] = value;
      }
    }
    ASSERT_EQ(api_->host_mapping_cache_control(
                  view, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 3 * unit_),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->host_mapping_cache_control(
                  view, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, 3 * unit_),
              AMDF_STATUS_OK);
    for (size_t slot = 0; slot < 3; ++slot) {
      ASSERT_EQ(std::memcmp(pointer + slot * words,
                            expected[backing[epoch][slot]].data() +
                                offsets[epoch][slot] * words,
                            unit_),
                0);
    }
    for (size_t source = 0; source < memories.size(); ++source) {
      ASSERT_EQ(
          std::memcmp(original[source], expected[source].data(), 6 * unit_), 0);
    }
  }
  ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
  const auto refill = Bind(0, 3 * unit_, memories[0], unit_);
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &refill),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(pointer, expected[0].data() + words, 3 * unit_), 0);
}

TEST_P(HostReservationTest, UpdatesDistantRangesWithoutPopulatingTheHoles) {
  const uint64_t length = UINT64_C(16) << 30;
  ASSERT_LE(length, capabilities_.maximum_byte_length);
  ASSERT_NO_FATAL_FAILURE(Reserve(length));
  amdf_memory_t* memory = nullptr;
  uint32_t* source = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      Allocate(unit_, capabilities_.required_source_flags, &memory, &source));
  std::fill_n(source, unit_ / sizeof(uint32_t), 0x17839abcu);
  const std::array updates = {Bind(0, unit_, memory, 0),
                              Bind(length - unit_, unit_, memory, 0)};
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, updates.size(),
                                            updates.data()),
            AMDF_STATUS_OK);
  amdf_host_mapping_t* view = nullptr;
  uint32_t* tail = nullptr;
  ASSERT_NO_FATAL_FAILURE(View(length - unit_, unit_, &view, &tail));
  EXPECT_EQ(std::memcmp(tail, source, unit_), 0);
  tail[0] = 0xac215673u;
  EXPECT_EQ(source[0], 0xac215673u);
  auto* head = reinterpret_cast<const uint32_t*>(
      static_cast<uintptr_t>(address_.address));
  EXPECT_EQ(head[0], 0xac215673u);
  const amdf_memory_mapping_update_t removal = {.byte_length = unit_};
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &removal),
            AMDF_STATUS_OK);
  EXPECT_EQ(tail[0], 0xac215673u);
}

TEST_P(HostReservationTest, QualifiesExplicitSourceWithoutDiscoveringBindings) {
  ASSERT_NO_FATAL_FAILURE(Reserve(unit_));
  amdf_host_mapping_t* view = nullptr;
  uint32_t* pointer = nullptr;
  ASSERT_NO_FATAL_FAILURE(View(0, unit_, &view, &pointer));
  amdf_memory_t* memory = nullptr;
  uint32_t* source = nullptr;
  ASSERT_NO_FATAL_FAILURE(Allocate(
      unit_,
      capabilities_.required_source_flags | AMDF_MEMORY_FLAG_HOST_FILE_IO,
      &memory, &source));
  amdf_memory_site_t producer = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(producer),
      .kind = AMDF_MEMORY_SITE_KIND_HOST,
  };
  producer.value.host.mapping = view;
  amdf_memory_site_t consumer = producer;
  consumer.value.host.mapping = views_.back();
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  const auto unchanged = pair;
  EXPECT_EQ(amdf_status_code(
                api_->memory_query_pair_info(&producer, &consumer, &pair)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&pair, &unchanged, sizeof(pair)), 0);
  // Source-qualified visibility is available before any backing is installed.
  producer.value.host.source_memory = memory;
  ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
            AMDF_STATUS_OK);
  EXPECT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
  EXPECT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  // Backing eligibility belongs to the original view, not every alias.
  amdf_host_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
  ASSERT_EQ(
      api_->host_mapping_query_info(consumer.value.host.mapping, &mapping_info),
      AMDF_STATUS_OK);
  EXPECT_NE(mapping_info.features & AMDF_HOST_MAPPING_FEATURE_FILE_IO, 0u);
  ASSERT_EQ(api_->host_mapping_query_info(view, &mapping_info), AMDF_STATUS_OK);
  EXPECT_EQ(mapping_info.features & AMDF_HOST_MAPPING_FEATURE_FILE_IO, 0u);
  producer.kind = AMDF_MEMORY_SITE_KIND_FILE_IO;
  const auto host_pair = pair;
  EXPECT_EQ(amdf_status_code(
                api_->memory_query_pair_info(&producer, &consumer, &pair)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&pair, &host_pair, sizeof(pair)), 0);
  producer.kind = AMDF_MEMORY_SITE_KIND_HOST;
  consumer.kind = AMDF_MEMORY_SITE_KIND_FILE_IO;
  ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
            AMDF_STATUS_OK);
  EXPECT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
  amdf_memory_t* anonymous = nullptr;
  uint32_t* anonymous_pointer = nullptr;
  ASSERT_NO_FATAL_FAILURE(Allocate(unit_, 0, &anonymous, &anonymous_pointer));
  producer.value.host.source_memory = anonymous;
  const auto qualified = pair;
  EXPECT_EQ(amdf_status_code(
                api_->memory_query_pair_info(&producer, &consumer, &pair)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&pair, &qualified, sizeof(pair)), 0);
  const auto update = Bind(0, unit_, memory, 0);
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &update),
            AMDF_STATUS_OK);
  source[0] = 0x97134abcu;
  EXPECT_EQ(pointer[0], source[0]);
}

TEST_P(HostReservationTest, RejectsTheEntireInvalidBatchBeforeReplacement) {
  ASSERT_NO_FATAL_FAILURE(Reserve(2 * unit_));
  amdf_memory_t* source = nullptr;
  uint32_t* source_pointer = nullptr;
  ASSERT_NO_FATAL_FAILURE(Allocate(2 * unit_,
                                   capabilities_.required_source_flags, &source,
                                   &source_pointer));
  std::fill_n(source_pointer, 2 * unit_ / sizeof(uint32_t), 0x631789abu);
  auto initial = Bind(0, 2 * unit_, source, 0);
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &initial),
            AMDF_STATUS_OK);
  amdf_memory_t* anonymous = nullptr;
  uint32_t* anonymous_pointer = nullptr;
  ASSERT_NO_FATAL_FAILURE(Allocate(unit_, 0, &anonymous, &anonymous_pointer));
  amdf_host_mapping_t* view = nullptr;
  uint32_t* pointer = nullptr;
  ASSERT_NO_FATAL_FAILURE(View(0, 2 * unit_, &view, &pointer));
  std::array<amdf_memory_mapping_update_t, 2> updates = {
      amdf_memory_mapping_update_t{.byte_length = unit_},
      Bind(unit_, unit_, anonymous, 0)};
  EXPECT_EQ(amdf_status_code(api_->memory_reservation_update(reservation_, 0, 2,
                                                             updates.data())),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(pointer, source_pointer, 2 * unit_), 0);
  updates[1] = Bind(unit_, unit_, source, 1);
  EXPECT_EQ(amdf_status_code(api_->memory_reservation_update(reservation_, 0, 2,
                                                             updates.data())),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(pointer, source_pointer, 2 * unit_), 0);
  // Rejection does not poison the reservation or prevent subsequent updates.
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &initial),
            AMDF_STATUS_OK);
}

TEST_P(HostReservationTest, KeepsTheViewAliveAcrossBackingDestruction) {
  ASSERT_NO_FATAL_FAILURE(Reserve(unit_));
  amdf_host_mapping_t* view = nullptr;
  uint32_t* pointer = nullptr;
  ASSERT_NO_FATAL_FAILURE(View(0, unit_, &view, &pointer));
  for (uint32_t generation = 1; generation <= 3; ++generation) {
    amdf_memory_t* memory = nullptr;
    uint32_t* source = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        Allocate(unit_, capabilities_.required_source_flags, &memory, &source));
    const auto update = Bind(0, unit_, memory, 0);
    ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &update),
              AMDF_STATUS_OK);
    std::fill_n(pointer, unit_ / sizeof(uint32_t), 0x167ab300u + generation);
    EXPECT_EQ(std::memcmp(pointer, source, unit_), 0);
    ASSERT_EQ(api_->memory_reservation_detach(reservation_, 0), AMDF_STATUS_OK);
    ASSERT_EQ(api_->host_mapping_destroy(views_.back()), AMDF_STATUS_OK);
    views_.pop_back();
    const amdf_status_t released = api_->memory_destroy(memory);
    sources_.pop_back();
    ASSERT_EQ(released, AMDF_STATUS_OK);
    // Both the reservation and its cache-operation handle stay unchanged.
    amdf_host_mapping_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->host_mapping_query_info(view, &info), AMDF_STATUS_OK);
    EXPECT_EQ(info.pointer, pointer);
  }
}

TEST_P(HostReservationTest, ReadOnlyMappingAndViewDoNotGrantWriteOperations) {
  ASSERT_NO_FATAL_FAILURE(Reserve(unit_));
  amdf_memory_t* memory = nullptr;
  uint32_t* source = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      Allocate(unit_, capabilities_.required_source_flags, &memory, &source));
  std::fill_n(source, unit_ / sizeof(uint32_t), 0x957139abu);
  auto update = Bind(0, unit_, memory, 0);
  update.access = capabilities_.required_access | AMDF_MEMORY_ACCESS_READ;
  ASSERT_EQ(update.access & AMDF_MEMORY_ACCESS_WRITE, 0u);
  ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &update),
            AMDF_STATUS_OK);
  amdf_memory_map_info_t request = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(request),
      .byte_offset = sizeof(uint32_t),
      .byte_length = unit_ - sizeof(uint32_t),
      .flags = AMDF_MEMORY_MAP_FLAG_READ,
  };
  amdf_host_mapping_t* view = nullptr;
  ASSERT_EQ(api_->memory_reservation_map(reservation_, 0, &request, &view),
            AMDF_STATUS_OK);
  views_.push_back(view);
  amdf_host_mapping_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->host_mapping_query_info(view, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.flags, AMDF_MEMORY_MAP_FLAG_READ);
  EXPECT_EQ(api_->host_mapping_cache_control(
                view, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, request.byte_length),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  ASSERT_EQ(
      api_->host_mapping_cache_control(
          view, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, request.byte_length),
      AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(info.pointer, source + 1, request.byte_length), 0);
  request.byte_offset = unit_;
  auto* unchanged = view;
  EXPECT_EQ(amdf_status_code(api_->memory_reservation_map(
                reservation_, 0, &request, &unchanged)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(unchanged, view);
}

INSTANTIATE_TEST_SUITE_P(MappingUnit, HostReservationTest,
                         ::testing::Values(uint64_t{0}, UINT64_C(2) << 20),
                         [](const auto& info) {
                           return info.param == 0 ? "Native" : "TwoMiB";
                         });

}  // namespace
