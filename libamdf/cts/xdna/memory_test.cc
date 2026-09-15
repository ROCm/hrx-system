// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "gtest/gtest.h"
#include "xdna_device_fixture.h"

namespace {

class XdnaMemoryTest : public XdnaDeviceFixture {
 protected:
  void TearDown() override {
    if (mapping_ != nullptr) {
      EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
    }
    if (memory_ != nullptr) {
      EXPECT_TRUE(amdf_status_is_ok(api_->memory_destroy(memory_)));
    }
    XdnaDeviceFixture::TearDown();
  }

  amdf_memory_create_info_t MakeMemoryCreateInfo() {
    amdf_memory_create_info_t create_info = {};
    create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create_info.structure_size = sizeof(create_info);
    create_info.access_count = 1;
    create_info.accesses = &memory_access_;
    create_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        create_info.required_flags);
    create_info.byte_length = 4097;
    create_info.minimum_alignment = 4096;
    return create_info;
  }

  amdf_memory_t* memory_ = nullptr;
  amdf_host_mapping_t* mapping_ = nullptr;
};

TEST_F(XdnaMemoryTest, ValidatesCreationArgumentsWithoutNativeAllocation) {
  amdf_memory_create_info_t create_info = MakeMemoryCreateInfo();
  amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  EXPECT_EQ(
      amdf_status_code(api_->memory_create(nullptr, &create_info, &output)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  EXPECT_EQ(
      amdf_status_code(api_->memory_create(system_scope_, nullptr, &output)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  create_info.type = AMDF_STRUCTURE_TYPE_NONE;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeMemoryCreateInfo();
  create_info.byte_length = 0;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeMemoryCreateInfo();
  create_info.minimum_alignment = 3;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeMemoryCreateInfo();
  create_info.required_flags = UINT64_C(1) << 63;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeMemoryCreateInfo();
  create_info.registered_host_pointer = &create_info;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
}

TEST_F(XdnaMemoryTest, OwnsStableAddressAndExplicitHostMapping) {
  const amdf_memory_create_info_t create_info = MakeMemoryCreateInfo();
  const amdf_status_t create_status =
      api_->memory_create(system_scope_, &create_info, &memory_);
  ASSERT_TRUE(amdf_status_is_ok(create_status))
      << "domain=" << amdf_status_domain(create_status)
      << " code=" << amdf_status_code(create_status);
  ASSERT_NE(memory_, nullptr);

  amdf_memory_info_t memory_info = {};
  memory_info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  memory_info.structure_size = sizeof(memory_info);
  ASSERT_TRUE(
      amdf_status_is_ok(api_->memory_query_info(memory_, &memory_info)));
  amdf_memory_access_info_t access_info = {};
  access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
  access_info.structure_size = sizeof(access_info);
  ASSERT_EQ(api_->memory_query_access_info(memory_, 0, &access_info),
            AMDF_STATUS_OK);
  EXPECT_EQ(memory_info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(
      (memory_info.flags | access_info.flags) & create_info.required_flags,
      create_info.required_flags);
  EXPECT_EQ(access_info.access, memory_access_.requirements.access);
  EXPECT_EQ(access_info.address_domain_ordinal, 0u);
  EXPECT_GE(memory_info.native_allocation_byte_length, memory_info.byte_length);
  EXPECT_NE(memory_info.native_allocation_granularity, 0u);
  EXPECT_EQ(memory_info.byte_length, create_info.byte_length);
  ASSERT_GE(memory_info.alignment, create_info.minimum_alignment);
  EXPECT_EQ(memory_info.alignment & (memory_info.alignment - 1), 0u);
  uint64_t address = 0;
  ASSERT_EQ(api_->memory_query_address(
                memory_, 0, AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE, &address),
            AMDF_STATUS_OK);
  EXPECT_NE(address, 0u);
  EXPECT_EQ(address & (memory_info.alignment - 1), 0u);
  amdf_memory_profile_t profile = {};
  profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
  profile.structure_size = sizeof(profile);
  amdf_memory_access_capabilities_t access_capabilities = {};
  access_capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
  access_capabilities.structure_size = sizeof(access_capabilities);
  ASSERT_EQ(QueryMemoryProfile(create_info.memory_profile_ordinal, &profile,
                               &access_capabilities),
            AMDF_STATUS_OK);
  EXPECT_EQ(access_info.address_kinds, access_capabilities.address_kinds);
  ASSERT_GT(access_capabilities.device_address.address_bit_count, 0u);
  ASSERT_LE(access_capabilities.device_address.address_bit_count, 64u);
  EXPECT_NE(access_info.address_kinds &
                (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE),
            0u);
  for (amdf_memory_address_kind_t kind :
       {AMDF_MEMORY_ADDRESS_GPU, AMDF_MEMORY_ADDRESS_XDNA_DMA,
        AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE}) {
    uint64_t queried_address = UINT64_MAX;
    const amdf_status_t status =
        api_->memory_query_address(memory_, 0, kind, &queried_address);
    if ((access_info.address_kinds & (UINT64_C(1) << kind)) != 0) {
      ASSERT_EQ(status, AMDF_STATUS_OK);
      EXPECT_EQ(queried_address & (memory_info.alignment - 1), 0u);
      EXPECT_GE(queried_address,
                access_capabilities.device_address.minimum_address);
      ASSERT_LE(queried_address,
                access_capabilities.device_address.maximum_address);
      EXPECT_LE(
          memory_info.byte_length - 1,
          access_capabilities.device_address.maximum_address - queried_address);
      uint64_t repeated_address = 0;
      ASSERT_EQ(api_->memory_query_address(memory_, 0, kind, &repeated_address),
                AMDF_STATUS_OK);
      EXPECT_EQ(repeated_address, queried_address);
    } else {
      EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_UNSUPPORTED);
      EXPECT_EQ(queried_address, UINT64_MAX);
    }
  }

  amdf_xdna_device_info_t device_info = {};
  device_info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
  device_info.structure_size = sizeof(device_info);
  ASSERT_TRUE(
      amdf_status_is_ok(xdna_api_->device_query_info(device_, &device_info)));
  EXPECT_EQ(access_info.reset_epoch, device_info.reset_epoch);
  EXPECT_TRUE(amdf_device_id_is_equal(&access_info.device_id, &device_info.id));
  EXPECT_EQ(memory_info.access_count, 1u);
  EXPECT_EQ(access_info.ordinal, 0u);

  amdf_memory_access_info_t repeated_access_info = access_info;
  ASSERT_EQ(api_->memory_query_access_info(memory_, 0, &repeated_access_info),
            AMDF_STATUS_OK);
  EXPECT_EQ(
      std::memcmp(&access_info, &repeated_access_info, sizeof(access_info)), 0);
  EXPECT_EQ(amdf_status_code(api_->memory_query_access_info(
                memory_, memory_info.access_count, &repeated_access_info)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(
      std::memcmp(&access_info, &repeated_access_info, sizeof(access_info)), 0);

  amdf_memory_info_t second_memory_info = {};
  second_memory_info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  second_memory_info.structure_size = sizeof(second_memory_info);
  ASSERT_TRUE(
      amdf_status_is_ok(api_->memory_query_info(memory_, &second_memory_info)));
  EXPECT_EQ(std::memcmp(&memory_info, &second_memory_info, sizeof(memory_info)),
            0);

  amdf_memory_info_t invalid_memory_info = memory_info;
  invalid_memory_info.type = AMDF_STRUCTURE_TYPE_NONE;
  const amdf_memory_info_t original_invalid_memory_info = invalid_memory_info;
  EXPECT_EQ(
      amdf_status_code(api_->memory_query_info(memory_, &invalid_memory_info)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&invalid_memory_info, &original_invalid_memory_info,
                        sizeof(invalid_memory_info)),
            0);

  amdf_memory_map_info_t map_info = {};
  map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  map_info.structure_size = sizeof(map_info);
  map_info.byte_offset = 32;
  map_info.byte_length = memory_info.byte_length - map_info.byte_offset;
  map_info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  ASSERT_TRUE(
      amdf_status_is_ok(api_->memory_map(memory_, &map_info, &mapping_)));
  ASSERT_NE(mapping_, nullptr);

  amdf_host_mapping_info_t mapping_info = {};
  mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  mapping_info.structure_size = sizeof(mapping_info);
  ASSERT_TRUE(amdf_status_is_ok(
      api_->host_mapping_query_info(mapping_, &mapping_info)));
  EXPECT_EQ(mapping_info.flags & map_info.flags, map_info.flags);
  EXPECT_EQ(mapping_info.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  ASSERT_NE(mapping_info.pointer, nullptr);
  EXPECT_EQ(mapping_info.byte_length, map_info.byte_length);
  EXPECT_EQ(mapping_info.memory_byte_offset, map_info.byte_offset);
  EXPECT_EQ(mapping_info.byte_offset_granularity, 1u);
  EXPECT_EQ(mapping_info.byte_length_granularity, 1u);
  ASSERT_NE(mapping_info.cache_line_size, 0u);
  EXPECT_EQ(mapping_info.flush.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  EXPECT_EQ(mapping_info.flush.executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
  EXPECT_EQ(mapping_info.flush.host_operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
  EXPECT_EQ(mapping_info.flush.host_instruction,
            AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
  EXPECT_EQ(mapping_info.flush.host_fence_after,
            AMDF_HOST_CACHE_FENCE_X86_MFENCE);
  EXPECT_EQ(mapping_info.flush.range_granularity, mapping_info.cache_line_size);
  EXPECT_EQ(mapping_info.invalidate.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  EXPECT_EQ(mapping_info.invalidate.executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
  EXPECT_EQ(mapping_info.invalidate.host_operation,
            AMDF_HOST_CACHE_OPERATION_INVALIDATE);
  EXPECT_EQ(mapping_info.invalidate.host_instruction,
            AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
  EXPECT_EQ(mapping_info.invalidate.host_fence_after,
            AMDF_HOST_CACHE_FENCE_X86_MFENCE);
  EXPECT_EQ(mapping_info.invalidate.range_granularity,
            mapping_info.cache_line_size);

  amdf_host_mapping_info_t invalid_mapping_info = mapping_info;
  invalid_mapping_info.type = AMDF_STRUCTURE_TYPE_NONE;
  const amdf_host_mapping_info_t original_invalid_mapping_info =
      invalid_mapping_info;
  EXPECT_EQ(amdf_status_code(
                api_->host_mapping_query_info(mapping_, &invalid_mapping_info)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&invalid_mapping_info, &original_invalid_mapping_info,
                        sizeof(invalid_mapping_info)),
            0);

  std::memset(mapping_info.pointer, 0xA5,
              static_cast<size_t>(mapping_info.byte_length));
  EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, mapping_info.byte_length)));
  EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
      mapping_info.byte_length)));
  EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, mapping_info.byte_length, 0)));
  EXPECT_EQ(amdf_status_code(api_->host_mapping_cache_control(
                mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH,
                mapping_info.byte_length, 1)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(api_->host_mapping_cache_control(
                mapping_, 0, 0, mapping_info.byte_length)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  amdf_memory_map_info_t read_only_map_info = map_info;
  read_only_map_info.flags = AMDF_MEMORY_MAP_FLAG_READ;
  amdf_host_mapping_t* read_only_mapping = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_map(memory_, &read_only_map_info, &read_only_mapping)));
  ASSERT_NE(read_only_mapping, nullptr);
  amdf_host_mapping_info_t read_only_mapping_info = {};
  read_only_mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  read_only_mapping_info.structure_size = sizeof(read_only_mapping_info);
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_query_info(
      read_only_mapping, &read_only_mapping_info)));
  EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      read_only_mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 0)));
  const amdf_status_t read_only_flush_status = api_->host_mapping_cache_control(
      read_only_mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 1);
  if ((read_only_mapping_info.flags & AMDF_MEMORY_MAP_FLAG_WRITE) != 0) {
    EXPECT_TRUE(amdf_status_is_ok(read_only_flush_status));
  } else {
    EXPECT_EQ(amdf_status_code(read_only_flush_status),
              AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(read_only_mapping)));

  EXPECT_EQ(amdf_status_code(api_->memory_destroy(memory_)),
            AMDF_STATUS_CODE_BUSY);

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
  mapping_ = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(api_->memory_destroy(memory_)));
  memory_ = nullptr;
}

TEST_F(XdnaMemoryTest, RejectsInvalidMappingRequestsBeforeNativeMapping) {
  const amdf_memory_create_info_t create_info = MakeMemoryCreateInfo();
  const amdf_status_t create_status =
      api_->memory_create(system_scope_, &create_info, &memory_);
  ASSERT_TRUE(amdf_status_is_ok(create_status));

  amdf_memory_map_info_t map_info = {};
  map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  map_info.structure_size = sizeof(map_info);
  map_info.byte_length = 1;
  map_info.flags = AMDF_MEMORY_MAP_FLAG_READ;
  amdf_host_mapping_t* output =
      reinterpret_cast<amdf_host_mapping_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_status_code(api_->memory_map(nullptr, &map_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  EXPECT_EQ(amdf_status_code(api_->memory_map(memory_, nullptr, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  map_info.flags = 0;
  EXPECT_EQ(amdf_status_code(api_->memory_map(memory_, &map_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  map_info.flags = UINT32_C(1) << 31;
  EXPECT_EQ(amdf_status_code(api_->memory_map(memory_, &map_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  map_info.flags = AMDF_MEMORY_MAP_FLAG_READ;
  map_info.byte_offset = UINT64_C(65536);
  map_info.byte_length = 1;
  EXPECT_EQ(amdf_status_code(api_->memory_map(memory_, &map_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
}

}  // namespace
