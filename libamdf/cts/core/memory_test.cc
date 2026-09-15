// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "amdf/amdf.h"
#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

class HostMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_1,
                                            AMDF_ABI_VERSION_LATEST, &api_),
              AMDF_STATUS_OK);
    ASSERT_EQ(GetCtsDeviceCache().GetInstance(&instance_), AMDF_STATUS_OK);
    uint32_t count = 0;
    ASSERT_EQ(amdf_status_code(api_->instance_enumerate_memory_scopes(
                  instance_, 0, nullptr, &count)),
              AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    ASSERT_GT(count, 0u);
    std::vector<amdf_memory_scope_t*> scopes(count);
    ASSERT_EQ(api_->instance_enumerate_memory_scopes(instance_, count,
                                                     scopes.data(), &count),
              AMDF_STATUS_OK);
    for (amdf_memory_scope_t* scope : scopes) {
      amdf_memory_scope_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      info.structure_size = sizeof(info);
      ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
        scope_ = scope;
        scope_info_ = info;
        break;
      }
    }
    ASSERT_NE(scope_, nullptr);
  }

  void TearDown() override {
    for (amdf_host_mapping_t* mapping : mappings_) {
      EXPECT_EQ(api_->host_mapping_destroy(mapping), AMDF_STATUS_OK);
    }
    for (amdf_memory_t* memory : memories_) {
      EXPECT_EQ(api_->memory_destroy(memory), AMDF_STATUS_OK);
    }
  }

  void FindProfile(amdf_memory_profile_roles_t role,
                   amdf_memory_profile_t* out_profile) {
    for (uint32_t ordinal = 0; ordinal < scope_info_.memory_profile_count;
         ++ordinal) {
      amdf_memory_profile_t profile = {};
      profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile.structure_size = sizeof(profile);
      const amdf_status_t status = api_->memory_scope_query_profile(
          scope_, ordinal, 0, nullptr, &profile, nullptr);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED))
        continue;
      ASSERT_EQ(status, AMDF_STATUS_OK);
      if ((profile.roles & role) != 0) {
        *out_profile = profile;
        return;
      }
    }
    FAIL() << "CPU-only system scope has no requested acquisition contract";
  }

  void Map(amdf_memory_t* memory, uint64_t offset, uint64_t length,
           amdf_host_mapping_info_t* out_info) {
    amdf_memory_map_info_t map_info = {};
    map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map_info.structure_size = sizeof(map_info);
    map_info.byte_offset = offset;
    map_info.byte_length = length;
    map_info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    amdf_host_mapping_t* mapping = nullptr;
    ASSERT_EQ(api_->memory_map(memory, &map_info, &mapping), AMDF_STATUS_OK);
    mappings_.push_back(mapping);
    out_info->type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    out_info->structure_size = sizeof(*out_info);
    ASSERT_EQ(api_->host_mapping_query_info(mapping, out_info), AMDF_STATUS_OK);
  }

  // Public API and lifetime root borrowed from the shared CTS cache.
  const amdf_api_t* api_ = nullptr;
  // Instance shared with other CTS cases; no accelerator is initialized here.
  amdf_instance_t* instance_ = nullptr;
  // Borrowed system storage descriptor selected by its reported locality.
  amdf_memory_scope_t* scope_ = nullptr;
  // Complete passive scope facts.
  amdf_memory_scope_info_t scope_info_ = {};
  // Case-owned mappings, released before their memory resources.
  std::vector<amdf_host_mapping_t*> mappings_;
  // Case-owned resources, released in consumer-before-backing order.
  std::vector<amdf_memory_t*> memories_;
  // Caller-owned registration backing retained through fixture teardown.
  std::vector<uint8_t> storage_;
};

TEST_F(HostMemoryTest, LiveProfileWithoutDevicesDescribesCpuOnlyStorage) {
  amdf_memory_profile_t expected = {};
  ASSERT_NO_FATAL_FAILURE(
      FindProfile(AMDF_MEMORY_PROFILE_ROLE_CREATE, &expected));
  amdf_memory_profile_t live = {};
  live.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
  live.structure_size = sizeof(live);
  ASSERT_EQ(api_->memory_scope_query_device_profile(scope_, expected.ordinal, 0,
                                                    nullptr, &live, nullptr),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(&live, &expected, sizeof(live)), 0);
  const amdf_memory_profile_t original = live;
  EXPECT_EQ(amdf_status_code(api_->memory_scope_query_device_profile(
                scope_, scope_info_.memory_profile_count, 0, nullptr, &live,
                nullptr)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(std::memcmp(&live, &original, sizeof(live)), 0);
}

TEST_F(HostMemoryTest, AllocatesOneBackingWithoutAnAccelerator) {
  amdf_memory_profile_t profile = {};
  ASSERT_NO_FATAL_FAILURE(
      FindProfile(AMDF_MEMORY_PROFILE_ROLE_CREATE, &profile));
  EXPECT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  amdf_memory_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.memory_profile_ordinal = profile.ordinal;
  create_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create_info.byte_length = profile.allocation.minimum_alignment + 37;
  amdf_memory_t* memory = nullptr;
  ASSERT_EQ(api_->memory_create(scope_, &create_info, &memory), AMDF_STATUS_OK);
  memories_.push_back(memory);

  amdf_memory_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(api_->memory_query_info(memory, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.access_count, 0u);
  EXPECT_EQ(info.byte_length, create_info.byte_length);
  EXPECT_EQ(info.memory_profile_ordinal, profile.ordinal);
  EXPECT_EQ(info.flags, AMDF_MEMORY_FLAG_HOST_VISIBLE);
  EXPECT_GE(info.native_allocation_byte_length, info.byte_length);
  EXPECT_EQ(
      info.native_allocation_byte_length % info.native_allocation_granularity,
      0u);
  amdf_host_mapping_info_t first = {};
  ASSERT_NO_FATAL_FAILURE(Map(memory, 0, info.byte_length, &first));
  amdf_host_mapping_info_t interior = {};
  ASSERT_NO_FATAL_FAILURE(Map(memory, 17, info.byte_length - 17, &interior));
  EXPECT_EQ(interior.pointer, static_cast<uint8_t*>(first.pointer) + 17);
  std::memset(first.pointer, 0x53, info.byte_length);
  std::memset(interior.pointer, 0x7b, interior.byte_length);
  const auto* bytes = static_cast<const uint8_t*>(first.pointer);
  for (uint64_t i = 0; i < info.byte_length; ++i) {
    ASSERT_EQ(bytes[i], i < 17 ? 0x53 : 0x7b) << i;
  }
  EXPECT_EQ(first.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(first.flush.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  EXPECT_EQ(first.invalidate.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  amdf_memory_site_t producer = {};
  producer.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
  producer.structure_size = sizeof(producer);
  producer.kind = AMDF_MEMORY_SITE_KIND_HOST;
  producer.value.host_mapping = mappings_[0];
  amdf_memory_site_t consumer = producer;
  consumer.value.host_mapping = mappings_[1];
  amdf_memory_pair_info_t pair = {};
  pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
  pair.structure_size = sizeof(pair);
  ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
            AMDF_STATUS_OK);
  EXPECT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(pair.flags, AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE |
                            AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN);
  EXPECT_EQ(pair.estimated_fixed_cost_nanoseconds, 0u);
  EXPECT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
  EXPECT_EQ(pair.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
  for (auto operation : {AMDF_HOST_CACHE_OPERATION_FLUSH,
                         AMDF_HOST_CACHE_OPERATION_INVALIDATE}) {
    EXPECT_EQ(api_->host_mapping_cache_control(mappings_[0], operation, 3,
                                               info.byte_length - 3),
              AMDF_STATUS_OK);
  }
  uint64_t address = UINT64_MAX;
  EXPECT_EQ(amdf_status_code(api_->memory_query_address(
                memory, 0, AMDF_MEMORY_ADDRESS_GPU, &address)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(address, UINT64_MAX);
  EXPECT_EQ(amdf_status_code(api_->memory_destroy(memory)),
            AMDF_STATUS_CODE_BUSY);
}

TEST_F(HostMemoryTest, BorrowsCallerStorageWithoutTakingOwnership) {
  amdf_memory_profile_t profile = {};
  ASSERT_NO_FATAL_FAILURE(
      FindProfile(AMDF_MEMORY_PROFILE_ROLE_REGISTER, &profile));
  storage_.assign(4099, 0x42);
  amdf_memory_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.memory_profile_ordinal = profile.ordinal;
  create_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create_info.registered_host_pointer = storage_.data() + 1;
  create_info.byte_length = storage_.size() - 2;
  amdf_memory_t* memory = nullptr;
  ASSERT_EQ(api_->memory_create(scope_, &create_info, &memory), AMDF_STATUS_OK);
  memories_.push_back(memory);
  amdf_host_mapping_info_t info = {};
  ASSERT_NO_FATAL_FAILURE(Map(memory, 0, create_info.byte_length, &info));
  EXPECT_EQ(info.pointer, storage_.data() + 1);
  std::memset(info.pointer, 0x19, info.byte_length);
  ASSERT_EQ(api_->host_mapping_destroy(mappings_.back()), AMDF_STATUS_OK);
  mappings_.pop_back();
  ASSERT_EQ(api_->memory_destroy(memory), AMDF_STATUS_OK);
  memories_.pop_back();
  EXPECT_EQ(storage_.front(), 0x42);
  EXPECT_EQ(storage_.back(), 0x42);
  for (size_t i = 1; i + 1 < storage_.size(); ++i) ASSERT_EQ(storage_[i], 0x19);
}

TEST_F(HostMemoryTest, RejectsInvalidConstructionWithoutPublishingOutputs) {
  amdf_memory_profile_t profile = {};
  ASSERT_NO_FATAL_FAILURE(
      FindProfile(AMDF_MEMORY_PROFILE_ROLE_CREATE, &profile));
  amdf_memory_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.memory_profile_ordinal = profile.ordinal;
  create_info.byte_length = 4096;
  amdf_memory_t* sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  amdf_memory_t* memory = sentinel;
  create_info.access_count = 1;
  EXPECT_EQ(
      amdf_status_code(api_->memory_create(scope_, &create_info, &memory)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(memory, sentinel);
  create_info.access_count = 0;
  create_info.required_flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  EXPECT_EQ(
      amdf_status_code(api_->memory_create(scope_, &create_info, &memory)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(memory, sentinel);
  create_info.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  EXPECT_EQ(
      amdf_status_code(api_->memory_create(scope_, &create_info, &memory)),
      AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(memory, sentinel);
  create_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create_info.minimum_alignment = profile.allocation.maximum_alignment * 2;
  EXPECT_EQ(
      amdf_status_code(api_->memory_create(scope_, &create_info, &memory)),
      AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(memory, sentinel);

  amdf_memory_profile_t output = profile;
  EXPECT_EQ(amdf_status_code(api_->memory_scope_query_profile(
                scope_, scope_info_.memory_profile_count, 0, nullptr, &output,
                nullptr)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(std::memcmp(&output, &profile, sizeof(profile)), 0);
}

}  // namespace
