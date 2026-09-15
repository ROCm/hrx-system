// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <windows.h>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "gtest/gtest.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"

namespace {

class GpuMemoryTest : public GpuDeviceFixture {
 protected:
  void TearDown() override {
    if (mapping_ != nullptr) {
      const amdf_status_t status = api_->host_mapping_destroy(mapping_);
      EXPECT_TRUE(amdf_status_is_ok(status));
      if (amdf_status_is_ok(status)) {
        mapping_ = nullptr;
      }
    }
    if (memory_ != nullptr) {
      const amdf_status_t status = api_->memory_destroy(memory_);
      EXPECT_TRUE(amdf_status_is_ok(status));
      if (amdf_status_is_ok(status)) {
        memory_ = nullptr;
      }
    }
    if (memory_ == nullptr && registered_host_pointer_ != nullptr) {
      EXPECT_TRUE(VirtualFree(registered_host_pointer_, 0, MEM_RELEASE));
      registered_host_pointer_ = nullptr;
    }
    GpuDeviceFixture::TearDown();
  }

  amdf_memory_create_info_t MakeSystemMemoryCreateInfo() {
    amdf_memory_create_info_t create_info = {};
    create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create_info.structure_size = sizeof(create_info);
    memory_access_.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    memory_access_.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    create_info.access_count = 1;
    create_info.accesses = &memory_access_;
    create_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
        system_scope_,
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        create_info.required_flags, memory_access_.requirements);
    create_info.byte_length = 4097;
    create_info.minimum_alignment = 1024 * 1024;
    return create_info;
  }

  // Case-owned backing, released after its host view.
  amdf_memory_t* memory_ = nullptr;
  // Explicit host view borrowing memory_.
  amdf_host_mapping_t* mapping_ = nullptr;
  // Caller storage retained until the registration is gone.
  void* registered_host_pointer_ = nullptr;
};

TEST_F(GpuMemoryTest, ValidatesPlacementRequirementsBeforeNativeAllocation) {
  amdf_memory_create_info_t create_info = MakeSystemMemoryCreateInfo();
  amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});

  create_info.required_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeSystemMemoryCreateInfo();
  create_info.required_flags |= AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeSystemMemoryCreateInfo();
  ASSERT_NE(local_scope_, nullptr);
  create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
      local_scope_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
      AMDF_MEMORY_FLAG_DEVICE_LOCAL, memory_access_.requirements);
  create_info.required_flags |= AMDF_MEMORY_FLAG_HOST_VISIBLE;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(local_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeSystemMemoryCreateInfo();
  create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
      system_scope_,
      AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      AMDF_MEMORY_FLAG_HOST_VISIBLE, memory_access_.requirements);
  create_info.byte_length = 4096;
  create_info.registered_host_pointer = &create_info;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(system_scope_, &create_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
}

TEST_F(GpuMemoryTest, OwnsStableSystemAddressAndExplicitHostMapping) {
  const amdf_memory_create_info_t create_info = MakeSystemMemoryCreateInfo();
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_create(system_scope_, &create_info, &memory_)));
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
  EXPECT_GE(memory_info.byte_length, create_info.byte_length);
  EXPECT_GE(memory_info.alignment, create_info.minimum_alignment);
  uint64_t address = 0;
  ASSERT_EQ(
      api_->memory_query_address(memory_, 0, AMDF_MEMORY_ADDRESS_GPU, &address),
      AMDF_STATUS_OK);
  EXPECT_EQ(address & (memory_info.alignment - 1), 0u);
  EXPECT_NE(memory_info.physical_backing_id.words[0] |
                memory_info.physical_backing_id.words[1],
            0u);

  amdf_gpu_device_info_t device_info = {};
  device_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
  device_info.structure_size = sizeof(device_info);
  ASSERT_TRUE(
      amdf_status_is_ok(gpu_api_->device_query_info(device_, &device_info)));
  EXPECT_EQ(access_info.reset_epoch, device_info.reset_epoch);
  EXPECT_TRUE(amdf_device_id_is_equal(&access_info.device_id, &device_info.id));
  EXPECT_EQ(memory_info.access_count, 1u);
  EXPECT_EQ(access_info.ordinal, 0u);

  amdf_memory_map_info_t map_info = {};
  map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  map_info.structure_size = sizeof(map_info);
  map_info.byte_length = memory_info.byte_length;
  map_info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  ASSERT_TRUE(
      amdf_status_is_ok(api_->memory_map(memory_, &map_info, &mapping_)));

  amdf_host_mapping_info_t mapping_info = {};
  mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  mapping_info.structure_size = sizeof(mapping_info);
  ASSERT_TRUE(amdf_status_is_ok(
      api_->host_mapping_query_info(mapping_, &mapping_info)));
  ASSERT_NE(mapping_info.pointer, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping_info.pointer) &
                (memory_info.alignment - 1),
            0u);
  EXPECT_EQ(mapping_info.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(mapping_info.byte_length, memory_info.byte_length);
  ASSERT_NE(mapping_info.cache_line_size, 0u);
  EXPECT_EQ(mapping_info.flush.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  EXPECT_EQ(mapping_info.flush.executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API);
  EXPECT_EQ(mapping_info.flush.host_operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
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

  std::memset(mapping_info.pointer, 0xA5,
              static_cast<size_t>(mapping_info.byte_length));
  EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, mapping_info.byte_length)));
  EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
      mapping_info.byte_length)));

  EXPECT_EQ(amdf_status_code(api_->memory_destroy(memory_)),
            AMDF_STATUS_CODE_BUSY);
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
  mapping_ = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(api_->memory_destroy(memory_)));
  memory_ = nullptr;
}

TEST_F(GpuMemoryTest, CreatesDeviceLocalExecutableMemory) {
  ASSERT_NE(local_scope_, nullptr);
  const amdf_memory_device_access_t access = {
      device_,
      {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                 AMDF_MEMORY_ACCESS_EXECUTE,
       .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  amdf_memory_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.access_count = 1;
  create_info.accesses = &access;
  create_info.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  create_info.memory_profile_ordinal =
      FindMemoryProfileOrdinal(local_scope_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
                               create_info.required_flags, access.requirements);
  create_info.byte_length = 4097;
  create_info.minimum_alignment = 1024 * 1024;

  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_create(local_scope_, &create_info, &memory_)));
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
  EXPECT_EQ(memory_info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
  EXPECT_EQ(
      (memory_info.flags | access_info.flags) & create_info.required_flags,
      create_info.required_flags);
  EXPECT_EQ(access_info.access, access.requirements.access);
  EXPECT_EQ(access_info.flags & access.requirements.flags,
            access.requirements.flags);
  EXPECT_GE(memory_info.byte_length, create_info.byte_length);
  EXPECT_GE(memory_info.alignment, create_info.minimum_alignment);
  uint64_t address = 0;
  ASSERT_EQ(
      api_->memory_query_address(memory_, 0, AMDF_MEMORY_ADDRESS_GPU, &address),
      AMDF_STATUS_OK);
  EXPECT_EQ(address & (memory_info.alignment - 1), 0u);
  EXPECT_NE(address, 0u);

  amdf_memory_map_info_t map_info = {};
  map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  map_info.structure_size = sizeof(map_info);
  map_info.byte_length = memory_info.byte_length;
  map_info.flags = AMDF_MEMORY_MAP_FLAG_WRITE;
  amdf_host_mapping_t* output =
      reinterpret_cast<amdf_host_mapping_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_status_code(api_->memory_map(memory_, &map_info, &output)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
}

TEST_F(GpuMemoryTest, RegistersCallerOwnedCoherentHostPages) {
  constexpr uint64_t kByteLength = 64 * 1024;
  registered_host_pointer_ = VirtualAlloc(
      nullptr, kByteLength, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  ASSERT_NE(registered_host_pointer_, nullptr);

  const amdf_memory_device_access_t access = {
      device_,
      {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  amdf_memory_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.access_count = 1;
  create_info.accesses = &access;
  create_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
      system_scope_,
      AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      create_info.required_flags, access.requirements);
  create_info.byte_length = kByteLength;
  create_info.minimum_alignment = kByteLength;
  create_info.registered_host_pointer = registered_host_pointer_;
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_create(system_scope_, &create_info, &memory_)));

  amdf_memory_map_info_t map_info = {};
  map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  map_info.structure_size = sizeof(map_info);
  map_info.byte_offset = 128;
  map_info.byte_length = 4096;
  map_info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  ASSERT_TRUE(
      amdf_status_is_ok(api_->memory_map(memory_, &map_info, &mapping_)));

  amdf_host_mapping_info_t mapping_info = {};
  mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  mapping_info.structure_size = sizeof(mapping_info);
  ASSERT_TRUE(amdf_status_is_ok(
      api_->host_mapping_query_info(mapping_, &mapping_info)));
  EXPECT_EQ(
      mapping_info.pointer,
      static_cast<uint8_t*>(registered_host_pointer_) + map_info.byte_offset);
  EXPECT_EQ(mapping_info.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_NE(mapping_info.cache_line_size, 0u);
  EXPECT_EQ(mapping_info.flush.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  EXPECT_EQ(mapping_info.flush.executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
  EXPECT_EQ(mapping_info.flush.host_instruction,
            AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
  EXPECT_EQ(mapping_info.invalidate.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  EXPECT_EQ(mapping_info.invalidate.host_instruction,
            AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
  EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, mapping_info.byte_length)));

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
  mapping_ = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(api_->memory_destroy(memory_)));
  memory_ = nullptr;
  std::memset(registered_host_pointer_, 0x3C, kByteLength);
}

}  // namespace
