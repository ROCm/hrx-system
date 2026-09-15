// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>

#include "amdf/amdf.h"
#include "gtest/gtest.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"

namespace {

class XdnaLinuxMemoryTest : public XdnaDeviceFixture {
 protected:
  void TearDown() override {
    bool caller_pages_may_release = true;
    for (size_t i = 3; i > 0; --i) {
      const size_t ordinal = i - 1;
      if (mappings_[ordinal] != nullptr) {
        const amdf_status_t status =
            api_->host_mapping_destroy(mappings_[ordinal]);
        EXPECT_EQ(status, AMDF_STATUS_OK);
        if (amdf_status_is_ok(status)) {
          mappings_[ordinal] = nullptr;
        } else {
          caller_pages_may_release = false;
        }
      }
      if (memories_[ordinal] != nullptr && mappings_[ordinal] == nullptr) {
        const amdf_status_t status = api_->memory_destroy(memories_[ordinal]);
        EXPECT_EQ(status, AMDF_STATUS_OK);
        if (amdf_status_is_ok(status)) {
          memories_[ordinal] = nullptr;
        } else {
          caller_pages_may_release = false;
        }
      }
    }
    for (amdf_external_memory_t& external_memory : external_memories_) {
      api_->external_memory_release(&external_memory);
    }
    if (caller_pages_ != nullptr && caller_pages_may_release) {
      EXPECT_EQ(munmap(caller_pages_, caller_byte_length_), 0);
      caller_pages_ = nullptr;
    }
    XdnaDeviceFixture::TearDown();
  }

  void AllocateCallerPages() {
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    page_size_ = static_cast<size_t>(page_size);
    caller_byte_length_ = page_size_ * 8;
    void* pages = mmap(nullptr, caller_byte_length_, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(pages, MAP_FAILED);
    caller_pages_ = static_cast<uint8_t*>(pages);
    std::memset(caller_pages_, 0x5A, caller_byte_length_);
  }

  amdf_status_t MapMemory(amdf_memory_t* memory, uint32_t ordinal,
                          uint64_t byte_length) {
    const amdf_memory_map_info_t map_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(amdf_memory_map_info_t),
        .byte_length = byte_length,
        .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    amdf_status_t status =
        api_->memory_map(memory, &map_info, &mappings_[ordinal]);
    if (!amdf_status_is_ok(status)) return status;
    mapping_infos_[ordinal] = {};
    mapping_infos_[ordinal].type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapping_infos_[ordinal].structure_size = sizeof(mapping_infos_[ordinal]);
    return api_->host_mapping_query_info(mappings_[ordinal],
                                         &mapping_infos_[ordinal]);
  }

  // Host page size governing the native registration cover.
  size_t page_size_ = 0;
  // Complete caller-owned mapping retained through every registration.
  size_t caller_byte_length_ = 0;
  // First byte of the caller-owned mapping.
  uint8_t* caller_pages_ = nullptr;
  // Source and two independently owned XDNA attachments.
  amdf_memory_t* memories_[3] = {};
  // Explicit host views of each attachment.
  amdf_host_mapping_t* mappings_[3] = {};
  // Immutable facts for each host view.
  amdf_host_mapping_info_t mapping_infos_[3] = {};
  // Exported values retained until consumption or fixture teardown.
  amdf_external_memory_t external_memories_[2] = {};
};

TEST_F(XdnaLinuxMemoryTest,
       RegistersArbitraryOverlappingCallerSubrangesWithoutTakingOwnership) {
  ASSERT_NO_FATAL_FAILURE(AllocateCallerPages());
  const amdf_memory_flags_t required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  const amdf_memory_access_t device_access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  const uint32_t profile_ordinal = FindMemoryProfileOrdinal(
      AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      required_flags);
  ASSERT_NE(profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);

  amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(amdf_memory_create_info_t),
      .memory_profile_ordinal = profile_ordinal,
      .access_count = 1,
      .required_flags = required_flags,
      .byte_length = caller_byte_length_ / 2 + 17,
      .minimum_alignment = 1,
      .registered_host_pointer = caller_pages_ + 3,
      .accesses = &memory_access_,
  };

  amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  amdf_memory_create_info_t unsupported_info = create_info;
  amdf_memory_device_access_t unsupported_access = memory_access_;
  unsupported_access.requirements.access = AMDF_MEMORY_ACCESS_READ;
  unsupported_info.accesses = &unsupported_access;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &unsupported_info, &output)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  EXPECT_EQ(caller_pages_[3], 0x5A);

  ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &memories_[0]),
            AMDF_STATUS_OK);
  amdf_memory_info_t first_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(amdf_memory_info_t),
  };
  ASSERT_EQ(api_->memory_query_info(memories_[0], &first_info), AMDF_STATUS_OK);
  amdf_memory_access_info_t first_access_info = {};
  first_access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
  first_access_info.structure_size = sizeof(first_access_info);
  ASSERT_EQ(api_->memory_query_access_info(memories_[0], 0, &first_access_info),
            AMDF_STATUS_OK);
  EXPECT_EQ(first_info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(first_access_info.access, device_access);
  EXPECT_EQ((first_info.flags | first_access_info.flags) & required_flags,
            required_flags);
  EXPECT_EQ(first_info.source_byte_offset, 3u);
  EXPECT_EQ(first_info.byte_length, create_info.byte_length);
  EXPECT_EQ(first_info.alignment, 1u);
  EXPECT_EQ(first_info.native_allocation_byte_length,
            ((first_info.source_byte_offset + first_info.byte_length +
              page_size_ - 1) /
             page_size_) *
                page_size_);
  EXPECT_EQ(first_info.native_allocation_granularity, page_size_);
  EXPECT_FALSE(
      amdf_physical_memory_id_is_valid(&first_info.physical_backing_id));
  uint64_t first_address = 0;
  ASSERT_EQ(
      api_->memory_query_address(
          memories_[0], 0, AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE, &first_address),
      AMDF_STATUS_OK);
  EXPECT_EQ(first_address & (page_size_ - 1), first_info.source_byte_offset);
  ASSERT_EQ(MapMemory(memories_[0], 0, create_info.byte_length),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_infos_[0].pointer, create_info.registered_host_pointer);
  EXPECT_EQ(mapping_infos_[0].cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);

  create_info.registered_host_pointer = caller_pages_ + 19;
  create_info.byte_length = caller_byte_length_ / 2;
  ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &memories_[1]),
            AMDF_STATUS_OK);
  amdf_memory_info_t second_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(amdf_memory_info_t),
  };
  ASSERT_EQ(api_->memory_query_info(memories_[1], &second_info),
            AMDF_STATUS_OK);
  EXPECT_EQ(second_info.source_byte_offset, 19u);
  EXPECT_EQ(second_info.byte_length, create_info.byte_length);
  uint64_t second_address = 0;
  ASSERT_EQ(
      api_->memory_query_address(
          memories_[1], 0, AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE, &second_address),
      AMDF_STATUS_OK);
  EXPECT_EQ(second_address & (page_size_ - 1), second_info.source_byte_offset);
  ASSERT_EQ(MapMemory(memories_[1], 1, create_info.byte_length),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_infos_[1].pointer, create_info.registered_host_pointer);

  std::memset(mapping_infos_[1].pointer, 0xC3, create_info.byte_length);
  ASSERT_EQ(api_->host_mapping_cache_control(mappings_[1],
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             create_info.byte_length),
            AMDF_STATUS_OK);
  const auto* first_bytes =
      static_cast<const uint8_t*>(mapping_infos_[0].pointer);
  for (size_t i = 0; i < create_info.byte_length; ++i) {
    ASSERT_EQ(first_bytes[16 + i], 0xC3) << "byte " << i;
  }
  EXPECT_EQ(caller_pages_[2], 0x5A);
  EXPECT_EQ(caller_pages_[19 + create_info.byte_length], 0x5A);

  ASSERT_EQ(api_->host_mapping_destroy(mappings_[0]), AMDF_STATUS_OK);
  mappings_[0] = nullptr;
  ASSERT_EQ(api_->memory_destroy(memories_[0]), AMDF_STATUS_OK);
  memories_[0] = nullptr;
  EXPECT_EQ(static_cast<const uint8_t*>(mapping_infos_[1].pointer)[0], 0xC3);
  EXPECT_EQ(static_cast<const uint8_t*>(
                mapping_infos_[1].pointer)[create_info.byte_length - 1],
            0xC3);

  ASSERT_EQ(api_->host_mapping_destroy(mappings_[1]), AMDF_STATUS_OK);
  mappings_[1] = nullptr;
  ASSERT_EQ(api_->memory_destroy(memories_[1]), AMDF_STATUS_OK);
  memories_[1] = nullptr;
  std::memset(caller_pages_, 0x3C, caller_byte_length_);
  EXPECT_EQ(caller_pages_[caller_byte_length_ - 1], 0x3C);
}

TEST_F(XdnaLinuxMemoryTest,
       ExportsDuplicateSubrangesWithIndependentImportLifetimes) {
  const long native_page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(native_page_size, 0);
  page_size_ = static_cast<size_t>(native_page_size);
  const uint64_t native_byte_length = page_size_ * 3;
  const uint64_t source_byte_offset = page_size_ + 13;
  const uint64_t logical_byte_length = page_size_ - 29;
  const amdf_memory_flags_t owned_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  const amdf_memory_flags_t imported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  const uint32_t owned_profile_ordinal = FindMemoryProfileOrdinal(
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_EXPORT |
          AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      owned_flags);
  ASSERT_NE(owned_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  const uint32_t imported_profile_ordinal = FindMemoryProfileOrdinal(
      AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      imported_flags);
  ASSERT_NE(imported_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);

  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(amdf_memory_create_info_t),
      .memory_profile_ordinal = owned_profile_ordinal,
      .access_count = 1,
      .required_flags = owned_flags,
      .byte_length = native_byte_length,
      .minimum_alignment = page_size_,
      .accesses = &memory_access_,
  };
  ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &memories_[0]),
            AMDF_STATUS_OK);
  amdf_memory_info_t source_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(amdf_memory_info_t),
  };
  ASSERT_EQ(api_->memory_query_info(memories_[0], &source_info),
            AMDF_STATUS_OK);
  amdf_memory_access_info_t source_access_info = {};
  source_access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
  source_access_info.structure_size = sizeof(source_access_info);
  ASSERT_EQ(
      api_->memory_query_access_info(memories_[0], 0, &source_access_info),
      AMDF_STATUS_OK);
  EXPECT_EQ((source_info.flags | source_access_info.flags) & owned_flags,
            owned_flags);
  ASSERT_TRUE(
      amdf_physical_memory_id_is_valid(&source_info.physical_backing_id));
  ASSERT_EQ(MapMemory(memories_[0], 0, native_byte_length), AMDF_STATUS_OK);
  auto* source_bytes = static_cast<uint8_t*>(mapping_infos_[0].pointer);
  for (uint64_t i = 0; i < native_byte_length; ++i) {
    source_bytes[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
  }
  ASSERT_EQ(
      api_->host_mapping_cache_control(
          mappings_[0], AMDF_HOST_CACHE_OPERATION_FLUSH, 0, native_byte_length),
      AMDF_STATUS_OK);

  const amdf_memory_export_info_t export_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
      .structure_size = sizeof(amdf_memory_export_info_t),
      .external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
      .byte_offset = source_byte_offset,
      .byte_length = logical_byte_length,
  };
  for (amdf_external_memory_t& external_memory : external_memories_) {
    ASSERT_EQ(api_->memory_export(memories_[0], &export_info, &external_memory),
              AMDF_STATUS_OK);
    EXPECT_EQ(external_memory.type, AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
    EXPECT_EQ(external_memory.source_byte_offset, source_byte_offset);
    EXPECT_EQ(external_memory.byte_length, logical_byte_length);
    EXPECT_TRUE(
        amdf_physical_memory_id_is_equal(&external_memory.physical_backing_id,
                                         &source_info.physical_backing_id));
  }

  const amdf_memory_import_info_t import_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO,
      .structure_size = sizeof(amdf_memory_import_info_t),
      .memory_profile_ordinal = imported_profile_ordinal,
      .access_count = 1,
      .required_flags = imported_flags,
      .minimum_alignment = 1,
      .accesses = &memory_access_,
  };
  const amdf_external_memory_t empty_external_memory = {};
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(api_->memory_import(system_scope_, &import_info,
                                  &external_memories_[i], &memories_[i + 1]),
              AMDF_STATUS_OK);
    EXPECT_EQ(std::memcmp(&external_memories_[i], &empty_external_memory,
                          sizeof(empty_external_memory)),
              0);
    amdf_memory_info_t imported_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
        .structure_size = sizeof(amdf_memory_info_t),
    };
    ASSERT_EQ(api_->memory_query_info(memories_[i + 1], &imported_info),
              AMDF_STATUS_OK);
    EXPECT_EQ(imported_info.source_byte_offset, source_byte_offset);
    EXPECT_EQ(imported_info.byte_length, logical_byte_length);
    EXPECT_TRUE(amdf_physical_memory_id_is_equal(
        &imported_info.physical_backing_id, &source_info.physical_backing_id));
    ASSERT_EQ(MapMemory(memories_[i + 1], static_cast<uint32_t>(i + 1),
                        logical_byte_length),
              AMDF_STATUS_OK);
    EXPECT_EQ(
        std::memcmp(mapping_infos_[i + 1].pointer,
                    source_bytes + source_byte_offset, logical_byte_length),
        0);
  }

  auto* first_import_bytes = static_cast<uint8_t*>(mapping_infos_[1].pointer);
  auto* second_import_bytes = static_cast<uint8_t*>(mapping_infos_[2].pointer);

  amdf_external_memory_t rejected_export;
  std::memset(&rejected_export, 0x5A, sizeof(rejected_export));
  const amdf_external_memory_t original_rejected_export = rejected_export;
  amdf_memory_export_info_t imported_export_info = export_info;
  imported_export_info.byte_offset = 0;
  EXPECT_EQ(amdf_status_code(api_->memory_export(
                memories_[1], &imported_export_info, &rejected_export)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&rejected_export, &original_rejected_export,
                        sizeof(rejected_export)),
            0);

  first_import_bytes[0] = 0xA7;
  first_import_bytes[logical_byte_length - 1] = 0xD3;
  ASSERT_EQ(api_->host_mapping_cache_control(mappings_[1],
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             logical_byte_length),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->host_mapping_cache_control(
                mappings_[2], AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                logical_byte_length),
            AMDF_STATUS_OK);
  EXPECT_EQ(second_import_bytes[0], 0xA7);
  EXPECT_EQ(second_import_bytes[logical_byte_length - 1], 0xD3);

  ASSERT_EQ(api_->host_mapping_destroy(mappings_[0]), AMDF_STATUS_OK);
  mappings_[0] = nullptr;
  ASSERT_EQ(api_->memory_destroy(memories_[0]), AMDF_STATUS_OK);
  memories_[0] = nullptr;

  second_import_bytes[1] = 0x6C;
  ASSERT_EQ(api_->host_mapping_cache_control(mappings_[2],
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             logical_byte_length),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->host_mapping_cache_control(
                mappings_[1], AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                logical_byte_length),
            AMDF_STATUS_OK);
  EXPECT_EQ(first_import_bytes[1], 0x6C);
}

}  // namespace
