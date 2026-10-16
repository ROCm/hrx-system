// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/asan.h"

#include <map>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only the native reservation/backing boundary is simulated. The production
// owner performs commitment, publication, range allocation, and quarantine.
struct Native {
  struct Memory {
    // Native dependency owner tracking release order.
    Native* owner;
    // Physical bytes shared by every installed alias.
    std::vector<uint8_t> bytes;
  };
  struct Reservation {
    // Native dependency owner.
    Native* owner;
    // Stable address and extent supplied by the native API.
    amdf_memory_reservation_access_info_t info;
    // Installed granule mappings, borrowed until checked removal.
    std::map<uint64_t, Memory*> pages;
  };
  // Native API dependency, independent of the ASAN implementation.
  amdf_api_t api = {};
  // Observed reservation owners in construction order.
  std::vector<Reservation*> reservations;
  // Number of live physical allocations.
  uint32_t live_memory = 0;
  // Injected native host publication failure.
  bool fail_flush = false;

  Native() {
    api.device_query_memory_reservation_capabilities =
        [](amdf_device_t*, uint32_t,
           amdf_memory_reservation_capabilities_t* out) {
          out->mapping_granularity = 4096;
          return AMDF_STATUS_OK;
        };
    api.memory_scope_query_info = [](amdf_memory_scope_t*,
                                     amdf_memory_scope_info_t* out) {
      out->memory_profile_count = 1;
      return AMDF_STATUS_OK;
    };
    api.memory_scope_query_device_profile =
        [](amdf_memory_scope_t*, uint32_t, uint32_t,
           const amdf_memory_device_access_t*, amdf_memory_profile_t* out,
           amdf_memory_access_capabilities_t*) {
          out->roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                       AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
          out->supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
          out->host_mapping.supported_access =
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
          out->allocation.byte_length_granularity = 4096;
          out->allocation.minimum_alignment = 4096;
          out->allocation.maximum_byte_length = UINT64_MAX;
          return AMDF_STATUS_OK;
        };
    api.memory_reservation_create =
        [](amdf_instance_t* instance,
           const amdf_memory_reservation_create_info_t* info,
           amdf_memory_reservation_t** out) {
          auto* self = reinterpret_cast<Native*>(instance);
          auto* reservation = new Reservation{self, {}, {}};
          reservation->info.address =
              self->reservations.empty() ? 0x10000 : 0x100000;
          reservation->info.byte_length = info->byte_length;
          reservation->info.mapping_granularity = 4096;
          reservation->info.access = info->accesses[0].access;
          self->reservations.push_back(reservation);
          *out = reinterpret_cast<amdf_memory_reservation_t*>(reservation);
          return AMDF_STATUS_OK;
        };
    api.memory_reservation_query_access_info =
        [](amdf_memory_reservation_t* handle, uint32_t,
           amdf_memory_reservation_access_info_t* out) {
          *out = reinterpret_cast<Reservation*>(handle)->info;
          return AMDF_STATUS_OK;
        };
    api.memory_reservation_update =
        [](amdf_memory_reservation_t* handle, uint32_t, uint32_t count,
           const amdf_memory_mapping_update_t* updates) {
          auto* reservation = reinterpret_cast<Reservation*>(handle);
          for (uint32_t i = 0; i < count; ++i) {
            const auto& update = updates[i];
            EXPECT_EQ(update.byte_length % 4096, 0u);
            EXPECT_LE(update.target_byte_offset + update.byte_length,
                      reservation->info.byte_length);
            for (uint64_t offset = update.target_byte_offset;
                 offset < update.target_byte_offset + update.byte_length;
                 offset += 4096) {
              if (update.memory) {
                reservation->pages[offset] =
                    reinterpret_cast<Memory*>(update.memory);
              } else {
                reservation->pages.erase(offset);
              }
            }
          }
          return AMDF_STATUS_OK;
        };
    api.memory_reservation_destroy = [](amdf_memory_reservation_t* handle) {
      auto* reservation = reinterpret_cast<Reservation*>(handle);
      for (auto*& entry : reservation->owner->reservations) {
        if (entry == reservation) {
          entry = nullptr;
        }
      }
      delete reservation;
      return AMDF_STATUS_OK;
    };
    api.memory_create = [](amdf_memory_scope_t* scope,
                           const amdf_memory_create_info_t* info,
                           amdf_memory_t** out) {
      auto* self = reinterpret_cast<Native*>(scope);
      ++self->live_memory;
      *out = reinterpret_cast<amdf_memory_t*>(
          new Memory{self, std::vector<uint8_t>(info->byte_length, 0xcc)});
      return AMDF_STATUS_OK;
    };
    api.memory_query_info = [](amdf_memory_t* handle, amdf_memory_info_t* out) {
      out->byte_length = reinterpret_cast<Memory*>(handle)->bytes.size();
      return AMDF_STATUS_OK;
    };
    api.memory_map = [](amdf_memory_t* handle, const amdf_memory_map_info_t*,
                        amdf_host_mapping_t** out) {
      *out = reinterpret_cast<amdf_host_mapping_t*>(handle);
      return AMDF_STATUS_OK;
    };
    api.host_mapping_query_info = [](amdf_host_mapping_t* handle,
                                     amdf_host_mapping_info_t* out) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      out->pointer = memory->bytes.data();
      out->byte_length = memory->bytes.size();
      out->cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
      return AMDF_STATUS_OK;
    };
    api.host_mapping_cache_control = [](amdf_host_mapping_t* handle,
                                        amdf_host_cache_operation_t, uint64_t,
                                        uint64_t) {
      return reinterpret_cast<Memory*>(handle)->owner->fail_flush
                 ? amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL)
                 : AMDF_STATUS_OK;
    };
    api.host_mapping_destroy = [](amdf_host_mapping_t*) {
      return AMDF_STATUS_OK;
    };
    api.memory_destroy = [](amdf_memory_t* handle) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      for (auto* reservation : memory->owner->reservations) {
        if (!reservation) {
          continue;
        }
        for (const auto& [offset, installed] : reservation->pages) {
          EXPECT_NE(installed, memory)
              << "backing released before alias removal";
        }
      }
      --memory->owner->live_memory;
      delete memory;
      return AMDF_STATUS_OK;
    };
  }

  uint8_t Shadow(uint64_t address) {
    const uint64_t offset = address >> 3;
    auto* memory = reservations[1]->pages.at(offset & ~UINT64_C(4095));
    return memory->bytes[offset & 4095];
  }
};

class AsanTest : public ::testing::Test {
 protected:
  void Initialize(bool premapped = false) {
    iree_hal_amd_fabric_asan_options_t options;
    iree_hal_amd_fabric_asan_options_initialize(&options);
    options.shadow_size = 65536;
    options.shadow_slab_size = 4096;
    options.owned_application_size = 32768;
    options.quarantine_size = 8192;
    if (premapped) {
      options.shadow_mode = IREE_HAL_AMD_FABRIC_ASAN_SHADOW_MODE_PREMAPPED;
    }
    IREE_ASSERT_OK(iree_hal_amd_fabric_asan_initialize(
        &native_.api, reinterpret_cast<amdf_instance_t*>(&native_),
        reinterpret_cast<amdf_device_t*>(&native_), 0,
        reinterpret_cast<amdf_memory_scope_t*>(&native_), &options,
        iree_hal_device_event_sink_stderr(), iree_allocator_system(), &state_));
  }
  void TearDown() override {
    EXPECT_TRUE(iree_hal_amd_fabric_asan_deinitialize(&state_));
    if (application_backing_) {
      native_.api.memory_destroy(application_backing_);
    }
    EXPECT_EQ(native_.live_memory, 0u);
    for (auto* reservation : native_.reservations) {
      EXPECT_EQ(reservation, nullptr);
    }
  }
  // Controlled native API dependencies and their backing.
  Native native_;
  // Production device-owned sanitizer namespace.
  iree_hal_amd_fabric_asan_t state_ = {};
  // Physical test backing, released after all application aliases are removed.
  amdf_memory_t* application_backing_ = nullptr;
};

TEST_F(AsanTest, PreciseShadowPreservesRedzonesAndPartialTailAcrossSlabs) {
  Initialize();
  ASSERT_FALSE(HasFatalFailure());
  constexpr uint64_t kAddress = 32768 - 16;
  IREE_ASSERT_OK(iree_hal_amd_fabric_asan_map_range(&state_, kAddress, 64));
  iree_hal_amd_fabric_asan_advise_range(&state_, kAddress, 64, 8, 19);
  const uint8_t expected[] = {0xfa, 0, 0, 3, 0xfa, 0xfa, 0xfa, 0xfa};
  for (size_t i = 0; i < sizeof(expected); ++i) {
    EXPECT_EQ(native_.Shadow(kAddress + i * 8), expected[i]);
  }
  iree_hal_amd_fabric_asan_advise_range(&state_, kAddress, 64, 0, 0);
  for (uint64_t i = 0; i < 64; i += 8) {
    EXPECT_EQ(native_.Shadow(kAddress + i), 0xfa);
  }
  iree_hal_device_asan_observation_t observation;
  iree_hal_amd_fabric_asan_query_statistics(&state_, &observation);
  EXPECT_EQ(observation.shadow_mapped_slab_count, 2u);
  EXPECT_EQ(observation.shadow_committed_size, 8192u);
}

TEST_F(AsanTest, ImportedBoundariesWidenWithoutShrinkingAnotherLiveImport) {
  Initialize();
  ASSERT_FALSE(HasFatalFailure());
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_asan_publish_imported_range(&state_, 67, 11));
  EXPECT_EQ(native_.Shadow(64), 0u);
  EXPECT_EQ(native_.Shadow(72), 6u);
  EXPECT_EQ(native_.Shadow(80), 0xfau);
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_asan_publish_imported_range(&state_, 73, 1));
  EXPECT_EQ(native_.Shadow(72), 6u);
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_asan_publish_imported_range(&state_, 77, 3));
  EXPECT_EQ(native_.Shadow(72), 0u);
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_asan_publish_imported_range(&state_, 74, 1));
  EXPECT_EQ(native_.Shadow(72), 0u);
}

TEST_F(AsanTest, PremappedPoisonAliasesAreReplacedOnlyForCommittedSlabs) {
  Initialize(true);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(native_.live_memory, 1u);
  EXPECT_EQ(native_.Shadow(0), 0xfau);
  EXPECT_EQ(native_.Shadow(8 * 65536 - 1), 0xfau);
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_asan_publish_imported_range(&state_, 16, 8));
  EXPECT_EQ(native_.live_memory, 2u);
  EXPECT_EQ(native_.Shadow(16), 0u);
  EXPECT_EQ(native_.Shadow(8 * 4096 + 16), 0xfau);
}

TEST_F(AsanTest, ApplicationAliasesRespectAlignmentAndCoalesceAfterRelease) {
  Initialize();
  ASSERT_FALSE(HasFatalFailure());
  const amdf_memory_create_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(info),
      .byte_length = 32768,
  };
  ASSERT_EQ(native_.api.memory_create(
                reinterpret_cast<amdf_memory_scope_t*>(&native_), &info,
                &application_backing_),
            AMDF_STATUS_OK);
  iree_hal_amd_fabric_asan_application_range_t* ranges[3] = {};
  IREE_ASSERT_OK(iree_hal_amd_fabric_asan_map_application(
      &state_, application_backing_, 0, 4096, 4096, &ranges[0]));
  IREE_ASSERT_OK(iree_hal_amd_fabric_asan_map_application(
      &state_, application_backing_, 0, 4096, 16384, &ranges[1]));
  IREE_ASSERT_OK(iree_hal_amd_fabric_asan_map_application(
      &state_, application_backing_, 0, 8192, 4096, &ranges[2]));
  EXPECT_EQ(ranges[0]->address, 0x10000u);
  EXPECT_EQ(ranges[1]->address, 0x14000u);
  EXPECT_EQ(ranges[2]->address, 0x11000u);
  EXPECT_EQ(native_.reservations[0]->pages.size(), 4u);
  for (size_t i : {size_t{1}, size_t{0}, size_t{2}}) {
    EXPECT_TRUE(iree_hal_amd_fabric_asan_unmap_application(&state_, ranges[i]));
  }
  EXPECT_TRUE(native_.reservations[0]->pages.empty());
  IREE_ASSERT_OK(iree_hal_amd_fabric_asan_map_application(
      &state_, application_backing_, 0, 32768, 32768, &ranges[0]));
  EXPECT_EQ(ranges[0]->address, 0x10000u);
  EXPECT_EQ(ranges[0]->length, 32768u);
  EXPECT_TRUE(iree_hal_amd_fabric_asan_unmap_application(&state_, ranges[0]));
}

TEST_F(AsanTest, QuarantineRetainsFifoUntilPressureAndFinalTeardown) {
  Initialize();
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_amd_fabric_asan_quarantine_entry_t entries[3] = {};
  std::vector<int> releases;
  struct Release {
    // Callback output whose lifetime spans quarantine draining.
    std::vector<int>* values;
    // FIFO sequence used to distinguish release callbacks.
    int value;
  } contexts[] = {{&releases, 0}, {&releases, 1}, {&releases, 2}};
  for (size_t i = 0; i < 3; ++i) {
    iree_hal_amd_fabric_asan_quarantine(
        &state_, &entries[i], 4096,
        [](void* user_data) {
          auto* release = static_cast<Release*>(user_data);
          release->values->push_back(release->value);
        },
        &contexts[i]);
  }
  EXPECT_EQ(releases, (std::vector<int>{0}));
  iree_hal_device_asan_observation_t observation;
  iree_hal_amd_fabric_asan_query_statistics(&state_, &observation);
  EXPECT_EQ(observation.quarantine_size, 8192u);
  EXPECT_EQ(observation.quarantine_eviction_count, 1u);
  EXPECT_TRUE(iree_hal_amd_fabric_asan_deinitialize(&state_));
  EXPECT_EQ(releases, (std::vector<int>{0, 1, 2}));
}

TEST_F(AsanTest, FailedPoisonPublicationLatchesOnceAndStopsAdmission) {
  Initialize();
  ASSERT_FALSE(HasFatalFailure());
  IREE_ASSERT_OK(iree_hal_amd_fabric_asan_map_range(&state_, 64, 64));
  uint32_t notifications = 0;
  state_.failure_user_data = &notifications;
  state_.failure_callback = [](void* user_data, iree_status_code_t code) {
    EXPECT_EQ(code, IREE_STATUS_INTERNAL);
    ++*static_cast<uint32_t*>(user_data);
  };
  native_.fail_flush = true;
  iree_hal_amd_fabric_asan_advise_range(&state_, 64, 64, 0, 8);
  iree_hal_amd_fabric_asan_advise_range(&state_, 64, 64, 0, 0);
  EXPECT_EQ(notifications, 1u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_amd_fabric_asan_map_range(&state_, 128, 64));
}

}  // namespace
