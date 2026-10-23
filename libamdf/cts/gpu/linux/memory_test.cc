// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <utility>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "gtest/gtest.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

struct DeviceAccessCase {
  // Exact device permissions requested for this case.
  amdf_memory_access_t access;
  // Consumer properties added for this case.
  amdf_memory_flags_t additional_flags;
};

constexpr std::array<DeviceAccessCase, 4> kDeviceAccessCases = {{
    {AMDF_MEMORY_ACCESS_READ, 0},
    {AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 0},
    {AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE, 0},
    {AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
         AMDF_MEMORY_ACCESS_EXECUTE,
     AMDF_MEMORY_FLAG_QUEUE_STORAGE},
}};

class GpuLinuxMemoryTest : public GpuDeviceFixture {
 protected:
  void TearDown() override {
    for (amdf_host_mapping_t*& mapping : mappings_) {
      if (mapping == nullptr) {
        continue;
      }
      ASSERT_EQ(api_->host_mapping_destroy(mapping), AMDF_STATUS_OK);
      mapping = nullptr;
    }
    for (amdf_memory_t*& memory : memories_) {
      if (memory == nullptr) {
        continue;
      }
      ASSERT_EQ(api_->memory_destroy(std::exchange(memory, nullptr), nullptr),
                AMDF_STATUS_OK);
    }
    if (!HasFailure() && caller_pages_ != nullptr) {
      ASSERT_EQ(munmap(caller_pages_, caller_byte_length_), 0);
      caller_pages_ = nullptr;
    }
    GpuDeviceFixture::TearDown();
  }

  amdf_memory_create_info_t MakeSystemMemoryCreateInfo() {
    amdf_memory_create_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(info),
    };
    memory_access_.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    memory_access_.requirements.flags =
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    info.access_count = 1;
    info.accesses = &memory_access_;
    info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    info.memory_profile_ordinal = FindMemoryProfileOrdinal(
        system_scope_,
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        info.required_flags, memory_access_.requirements);
    info.byte_length = 4097;
    info.minimum_alignment = 1024 * 1024;
    return info;
  }

  amdf_status_t MapMemory(amdf_memory_t* memory, uint32_t ordinal,
                          uint64_t byte_offset, uint64_t byte_length,
                          amdf_memory_map_flags_t flags) {
    amdf_memory_map_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(info),
        .byte_offset = byte_offset,
        .byte_length = byte_length,
        .flags = flags,
    };
    amdf_status_t status = api_->memory_map(memory, &info, &mappings_[ordinal]);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    mapping_infos_[ordinal] = {};
    mapping_infos_[ordinal].type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapping_infos_[ordinal].structure_size = sizeof(mapping_infos_[ordinal]);
    return api_->host_mapping_query_info(mappings_[ordinal],
                                         &mapping_infos_[ordinal]);
  }

  void AllocateCallerPages() {
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    caller_byte_length_ = static_cast<size_t>(page_size) * 8;
    void* pages = mmap(nullptr, caller_byte_length_, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(pages, MAP_FAILED);
    caller_pages_ = static_cast<uint8_t*>(pages);
    std::memset(caller_pages_, 0x5A, caller_byte_length_);
  }

  void ExerciseOwnedAccessMatrix(amdf_memory_scope_t* scope,
                                 amdf_memory_class_t memory_class,
                                 amdf_memory_flags_t required_flags) {
    const bool map_memory =
        (required_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0;
    for (size_t case_ordinal = 0; case_ordinal < kDeviceAccessCases.size();
         ++case_ordinal) {
      const DeviceAccessCase& access_case = kDeviceAccessCases[case_ordinal];
      const amdf_memory_device_access_t access = {
          device_,
          {
              .access = access_case.access,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                       (memory_class == AMDF_MEMORY_CLASS_SYSTEM
                            ? AMDF_MEMORY_FLAG_HOST_COHERENT
                            : 0) |
                       access_case.additional_flags,
          },
      };
      amdf_memory_create_info_t create_info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
          .structure_size = sizeof(create_info),
          .access_count = 1,
          .required_flags = required_flags,
          .accesses = &access,
      };
      create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
          scope,
          AMDF_MEMORY_PROFILE_ROLE_CREATE |
              (map_memory ? AMDF_MEMORY_PROFILE_ROLE_HOST_MAP : 0),
          create_info.required_flags, access.requirements);
      create_info.byte_length = 4097;
      create_info.minimum_alignment = 1024 * 1024;

      amdf_memory_profile_t profile = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(profile),
      };
      amdf_memory_access_capabilities_t capabilities = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(capabilities),
      };
      ASSERT_EQ(
          QueryMemoryProfile(scope, create_info.memory_profile_ordinal,
                             access.requirements, &profile, &capabilities),
          AMDF_STATUS_OK);
      const auto& geometry = profile.allocation;
      const uint64_t granularity = geometry.native_byte_length_granularity;
      ASSERT_GT(granularity, 0u);
      ASSERT_LE(
          geometry.maximum_byte_length,
          UINT64_MAX - geometry.native_byte_length_prefix - (granularity - 1));
      const uint64_t native_byte_length =
          ((create_info.byte_length + geometry.native_byte_length_prefix +
            granularity - 1) /
           granularity) *
          granularity;
      ASSERT_EQ(
          api_->memory_create(scope, &create_info, &memories_[0], nullptr),
          AMDF_STATUS_OK)
          << "access case " << case_ordinal;

      amdf_memory_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(api_->memory_query_info(memories_[0], &info), AMDF_STATUS_OK);
      amdf_memory_access_info_t access_info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
          .structure_size = sizeof(access_info),
      };
      ASSERT_EQ(api_->memory_query_access_info(memories_[0], 0, &access_info),
                AMDF_STATUS_OK);
      EXPECT_EQ(info.memory_profile_ordinal,
                create_info.memory_profile_ordinal);
      EXPECT_EQ(info.memory_class, memory_class);
      EXPECT_EQ(access_info.access, access.requirements.access);
      EXPECT_EQ(info.flags & profile.guaranteed_flags,
                profile.guaranteed_flags);
      EXPECT_EQ(info.flags & create_info.required_flags,
                create_info.required_flags);
      EXPECT_EQ(info.flags & ~profile.supported_flags, 0u);
      EXPECT_EQ(access_info.flags & capabilities.guaranteed_flags,
                capabilities.guaranteed_flags);
      EXPECT_EQ(access_info.flags & access.requirements.flags,
                access.requirements.flags);
      EXPECT_EQ(access_info.flags & ~capabilities.supported_flags, 0u);
      EXPECT_EQ(info.source_byte_offset, geometry.native_byte_length_prefix);
      EXPECT_EQ(info.byte_length, create_info.byte_length);
      EXPECT_EQ(info.native_allocation_byte_length, native_byte_length);
      EXPECT_EQ(info.native_allocation_granularity, granularity);
      EXPECT_GE(info.alignment, create_info.minimum_alignment);
      uint64_t address = 0;
      ASSERT_EQ(api_->memory_query_address(memories_[0], 0,
                                           AMDF_MEMORY_ADDRESS_GPU, &address),
                AMDF_STATUS_OK);
      EXPECT_EQ(address & (info.alignment - 1), 0u);

      if (map_memory) {
        ASSERT_EQ(
            MapMemory(memories_[0], 0, 0, info.byte_length,
                      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE),
            AMDF_STATUS_OK);
        ASSERT_NE(mapping_infos_[0].pointer, nullptr);
        EXPECT_EQ(mapping_infos_[0].cacheability,
                  ((info.flags | access_info.flags) &
                   AMDF_MEMORY_FLAG_HOST_COHERENT) != 0
                      ? AMDF_HOST_CACHEABILITY_WRITE_BACK
                      : AMDF_HOST_CACHEABILITY_WRITE_COMBINED);
        if (((info.flags | access_info.flags) &
             AMDF_MEMORY_FLAG_HOST_COHERENT) != 0) {
          EXPECT_NE(mapping_infos_[0].cache_line_size, 0u);
          EXPECT_EQ(mapping_infos_[0].flush.kind,
                    AMDF_CACHE_TRANSITION_KIND_RANGE);
          EXPECT_EQ(mapping_infos_[0].invalidate.kind,
                    AMDF_CACHE_TRANSITION_KIND_RANGE);
          EXPECT_EQ(mapping_infos_[0].flush.host_instruction,
                    AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
          EXPECT_EQ(mapping_infos_[0].invalidate.host_instruction,
                    AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
        } else {
          EXPECT_EQ(mapping_infos_[0].cache_line_size, 0u);
          if (mapping_infos_[0].flush.kind ==
              AMDF_CACHE_TRANSITION_KIND_UNKNOWN) {
            EXPECT_EQ(mapping_infos_[0].flush.executor,
                      AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
          } else {
            EXPECT_EQ(mapping_infos_[0].flush.kind,
                      AMDF_CACHE_TRANSITION_KIND_GLOBAL);
            EXPECT_EQ(mapping_infos_[0].flush.executor,
                      AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API);
            EXPECT_EQ(mapping_infos_[0].flush.host_operation,
                      AMDF_HOST_CACHE_OPERATION_FLUSH);
            EXPECT_EQ(mapping_infos_[0].flush.host_instruction,
                      AMDF_HOST_CACHE_INSTRUCTION_NONE);
            EXPECT_EQ(mapping_infos_[0].flush.host_fence_before,
                      AMDF_HOST_CACHE_FENCE_NONE);
            EXPECT_EQ(mapping_infos_[0].flush.host_fence_after,
                      AMDF_HOST_CACHE_FENCE_NONE);
          }
          EXPECT_EQ(mapping_infos_[0].invalidate.kind,
                    AMDF_CACHE_TRANSITION_KIND_GLOBAL);
          EXPECT_EQ(mapping_infos_[0].invalidate.executor,
                    AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
          EXPECT_EQ(mapping_infos_[0].invalidate.host_operation,
                    AMDF_HOST_CACHE_OPERATION_INVALIDATE);
          EXPECT_EQ(mapping_infos_[0].invalidate.host_fence_after,
                    AMDF_HOST_CACHE_FENCE_X86_MFENCE);
        }
        const uint8_t value = static_cast<uint8_t>(0x41 + case_ordinal);
        std::memset(mapping_infos_[0].pointer, value,
                    static_cast<size_t>(info.byte_length));
        EXPECT_EQ(static_cast<const uint8_t*>(
                      mapping_infos_[0].pointer)[info.byte_length - 1],
                  value);
        const amdf_status_t flush_status = api_->host_mapping_cache_control(
            mappings_[0], AMDF_HOST_CACHE_OPERATION_FLUSH, 0, info.byte_length);
        EXPECT_EQ(
            flush_status,
            mapping_infos_[0].flush.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN
                ? amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)
                : AMDF_STATUS_OK);
        ASSERT_EQ(api_->host_mapping_destroy(mappings_[0]), AMDF_STATUS_OK);
        mappings_[0] = nullptr;
      } else {
        amdf_memory_map_info_t map_info = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
            .structure_size = sizeof(map_info),
            .byte_length = 1,
            .flags = AMDF_MEMORY_MAP_FLAG_READ,
        };
        auto* const sentinel =
            reinterpret_cast<amdf_host_mapping_t*>(uintptr_t{1});
        amdf_host_mapping_t* output = sentinel;
        EXPECT_EQ(api_->memory_map(memories_[0], &map_info, &output),
                  amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
        EXPECT_EQ(output, sentinel);
      }

      ASSERT_EQ(
          api_->memory_destroy(std::exchange(memories_[0], nullptr), nullptr),
          AMDF_STATUS_OK);
    }
  }

  void ExerciseExactSystemDeviceAccess() {
    ASSERT_NO_FATAL_FAILURE(
        ExerciseOwnedAccessMatrix(system_scope_, AMDF_MEMORY_CLASS_SYSTEM,
                                  AMDF_MEMORY_FLAG_HOST_VISIBLE));

    amdf_memory_create_info_t create_info = MakeSystemMemoryCreateInfo();
    memory_access_.requirements.access = AMDF_MEMORY_ACCESS_WRITE;
    amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
    EXPECT_EQ(
        api_->memory_create(system_scope_, &create_info, &output, nullptr),
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  }

  void ExerciseLocalPlacementCapabilities() {
    if ((features_ & AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) == 0) {
      EXPECT_EQ(local_scope_, nullptr);
      return;
    }
    ASSERT_NE(local_scope_, nullptr);
    ASSERT_NO_FATAL_FAILURE(ExerciseOwnedAccessMatrix(
        local_scope_, AMDF_MEMORY_CLASS_LOCAL, AMDF_MEMORY_FLAG_DEVICE_LOCAL));

    if ((features_ & AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY) == 0) {
      amdf_memory_create_info_t create_info = MakeSystemMemoryCreateInfo();
      memory_access_.requirements.access = AMDF_MEMORY_ACCESS_READ;
      memory_access_.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
      create_info.required_flags =
          AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE;
      create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
          local_scope_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
          AMDF_MEMORY_FLAG_DEVICE_LOCAL, memory_access_.requirements);
      amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
      EXPECT_EQ(
          api_->memory_create(local_scope_, &create_info, &output, nullptr),
          amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
      EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
      return;
    }
    ASSERT_NO_FATAL_FAILURE(ExerciseOwnedAccessMatrix(
        local_scope_, AMDF_MEMORY_CLASS_LOCAL,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE));
  }

  // Exercises allocation and host-view lifetime under either instance policy.
  void ExerciseSystemMemory() {
    const amdf_memory_create_info_t create_info = MakeSystemMemoryCreateInfo();
    ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &memories_[0],
                                  nullptr),
              AMDF_STATUS_OK);
    amdf_memory_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->memory_query_info(memories_[0], &info), AMDF_STATUS_OK);
    amdf_memory_access_info_t access_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
        .structure_size = sizeof(access_info),
    };
    ASSERT_EQ(api_->memory_query_access_info(memories_[0], 0, &access_info),
              AMDF_STATUS_OK);
    EXPECT_EQ(info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    EXPECT_EQ((info.flags | access_info.flags) & create_info.required_flags,
              create_info.required_flags);
    EXPECT_GE(info.byte_length, create_info.byte_length);
    ASSERT_GE(info.alignment, create_info.minimum_alignment);
    uint64_t address = 0;
    ASSERT_EQ(api_->memory_query_address(memories_[0], 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
              AMDF_STATUS_OK);
    EXPECT_NE(address, 0u);
    EXPECT_EQ(address & (info.alignment - 1), 0u);
    EXPECT_NE(
        info.physical_backing_id.words[0] | info.physical_backing_id.words[1],
        0u);

    ASSERT_EQ(MapMemory(memories_[0], 0, 0, info.byte_length,
                        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE),
              AMDF_STATUS_OK);
    ASSERT_NE(mapping_infos_[0].pointer, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping_infos_[0].pointer) &
                  (info.alignment - 1),
              0u);
    EXPECT_EQ(mapping_infos_[0].cacheability,
              AMDF_HOST_CACHEABILITY_WRITE_BACK);
    EXPECT_NE(mapping_infos_[0].cache_line_size, 0u);
    EXPECT_EQ(mapping_infos_[0].flush.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    EXPECT_EQ(mapping_infos_[0].invalidate.kind,
              AMDF_CACHE_TRANSITION_KIND_RANGE);
    std::memset(mapping_infos_[0].pointer, 0xA5,
                static_cast<size_t>(info.byte_length));

    ASSERT_EQ(MapMemory(memories_[0], 1, 128, 256,
                        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE),
              AMDF_STATUS_OK);
    EXPECT_EQ(mapping_infos_[1].pointer,
              static_cast<uint8_t*>(mapping_infos_[0].pointer) + 128);
    const auto* bytes = static_cast<const uint8_t*>(mapping_infos_[1].pointer);
    for (size_t i = 0; i < 256; ++i) {
      EXPECT_EQ(bytes[i], 0xA5);
    }
    EXPECT_EQ(api_->host_mapping_cache_control(
                  mappings_[1], AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 256),
              AMDF_STATUS_OK);
    EXPECT_EQ(api_->host_mapping_cache_control(
                  mappings_[1], AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, 256),
              AMDF_STATUS_OK);
    EXPECT_EQ(amdf_status_code(api_->host_mapping_cache_control(
                  mappings_[1], AMDF_HOST_CACHE_OPERATION_FLUSH, 255, 2)),
              AMDF_STATUS_CODE_INVALID_ARGUMENT);
    ASSERT_EQ(api_->host_mapping_destroy(mappings_[0]), AMDF_STATUS_OK);
    mappings_[0] = nullptr;
    amdf_memory_info_t after = info;
    ASSERT_EQ(api_->memory_query_info(memories_[0], &after), AMDF_STATUS_OK);
    uint64_t address_after = 0;
    ASSERT_EQ(api_->memory_query_address(
                  memories_[0], 0, AMDF_MEMORY_ADDRESS_GPU, &address_after),
              AMDF_STATUS_OK);
    EXPECT_EQ(address_after, address);
    ASSERT_EQ(MapMemory(memories_[0], 0, 0, info.byte_length,
                        AMDF_MEMORY_MAP_FLAG_READ),
              AMDF_STATUS_OK);
    EXPECT_EQ(mapping_infos_[0].flags,
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
    EXPECT_EQ(static_cast<const uint8_t*>(mapping_infos_[0].pointer)[128],
              0xA5);
    EXPECT_EQ(api_->host_mapping_cache_control(
                  mappings_[0], AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 1),
              AMDF_STATUS_OK);

    for (amdf_host_mapping_t*& mapping : mappings_) {
      if (mapping == nullptr) {
        continue;
      }
      ASSERT_EQ(api_->host_mapping_destroy(mapping), AMDF_STATUS_OK);
      mapping = nullptr;
    }
    ASSERT_EQ(
        api_->memory_destroy(std::exchange(memories_[0], nullptr), nullptr),
        AMDF_STATUS_OK);
  }

  // Owned backing handles released after every host view.
  std::array<amdf_memory_t*, 4> memories_ = {};
  // Explicit views borrowing the memory handles above.
  std::array<amdf_host_mapping_t*, 4> mappings_ = {};
  // Cached properties of each explicit view.
  std::array<amdf_host_mapping_info_t, 4> mapping_infos_ = {};
  // Caller-owned anonymous pages retained until every registration is gone.
  uint8_t* caller_pages_ = nullptr;
  // Length of the caller-owned mmap reservation in bytes.
  size_t caller_byte_length_ = 0;
};

TEST_F(GpuLinuxMemoryTest, OwnsAlignedSystemMemoryAndBorrowedHostViews) {
  ASSERT_NO_FATAL_FAILURE(ExerciseSystemMemory());
}

TEST_F(GpuLinuxMemoryTest, HonorsExactSystemDeviceAccess) {
  ASSERT_NO_FATAL_FAILURE(ExerciseExactSystemDeviceAccess());
}

TEST_F(GpuLinuxMemoryTest, UnifiedExecutableStorageRetainsCachedHostAccess) {
  memory_access_.requirements = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                AMDF_MEMORY_ACCESS_EXECUTE,
      .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
  };
  const amdf_memory_flags_t flags =
      AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE;
  const uint32_t ordinal = FindMemoryProfileOrdinal(
      system_scope_,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      flags, memory_access_.requirements);
  if (ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP() << "No device-local unified storage profile";
  }
  const amdf_memory_create_info_t request = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(request),
      .memory_profile_ordinal = ordinal,
      .access_count = 1,
      .required_flags = flags,
      .byte_length = 4096,
      .minimum_alignment = 256,
      .accesses = &memory_access_,
  };
  ASSERT_EQ(
      api_->memory_create(system_scope_, &request, &memories_[0], nullptr),
      AMDF_STATUS_OK);
  amdf_memory_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->memory_query_info(memories_[0], &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(info.flags & flags, flags);
  ASSERT_EQ(MapMemory(memories_[0], 0, 0, request.byte_length,
                      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_infos_[0].cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  std::memset(mapping_infos_[0].pointer, 0x5A, request.byte_length);
  ASSERT_EQ(api_->host_mapping_cache_control(mappings_[0],
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             request.byte_length),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->host_mapping_cache_control(
                mappings_[0], AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                request.byte_length),
            AMDF_STATUS_OK);
  EXPECT_EQ(static_cast<const uint8_t*>(mapping_infos_[0].pointer)[4095], 0x5A);
}

TEST_F(GpuLinuxMemoryTest, HonorsSystemPlacementCapabilities) {
  amdf_memory_create_info_t info = MakeSystemMemoryCreateInfo();
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  amdf_memory_access_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(capabilities),
  };
  ASSERT_EQ(
      QueryMemoryProfile(system_scope_, info.memory_profile_ordinal,
                         memory_access_.requirements, &profile, &capabilities),
      AMDF_STATUS_OK);
  info.required_flags |= AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  const amdf_status_t status =
      api_->memory_create(system_scope_, &info, &output, nullptr);
  if (amdf_status_is_ok(status)) {
    memories_[0] = output;
  }
  if (profile.supported_flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL) {
    EXPECT_EQ(status, AMDF_STATUS_OK);
  } else {
    EXPECT_EQ(status, amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  }
}

TEST_F(GpuLinuxMemoryTest, HonorsLocalPlacementCapabilities) {
  ASSERT_NO_FATAL_FAILURE(ExerciseLocalPlacementCapabilities());
}

TEST_F(GpuLinuxMemoryTest, OmitsRegistrationWhenLifetimeDoesNotSupportIt) {
  if ((features_ & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION) != 0) {
    GTEST_SKIP() << "selected lifetime supports host registration";
  }
  for (uint32_t ordinal = 0;; ++ordinal) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t status =
        QueryMemoryProfile(system_scope_, ordinal, memory_access_.requirements,
                           &profile, &capabilities);
    if (amdf_status_code(status) == AMDF_STATUS_CODE_OUT_OF_RANGE) {
      break;
    }
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    EXPECT_NE(profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER,
              AMDF_MEMORY_PROFILE_ROLE_REGISTER);
  }
}

TEST_F(GpuLinuxMemoryTest, RegistersOverlappingCallerPagesWithExactAccess) {
  if (!(features_ & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION)) {
    GTEST_SKIP()
        << "host registration is unavailable under this native lifetime";
  }
  amdf_gpu_device_info_t device_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(device_info),
  };
  ASSERT_EQ(gpu_api_->device_query_info(device_, &device_info), AMDF_STATUS_OK);
  EXPECT_EQ(device_info.features, features_);
  ASSERT_NO_FATAL_FAILURE(AllocateCallerPages());

  const long native_page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(native_page_size, 0);
  const size_t page_size = static_cast<size_t>(native_page_size);
  constexpr std::array<size_t, 4> kCallerOffsets = {3, 19, 35, 51};
  const uint64_t logical_byte_length = page_size * 4 + 17;
  const amdf_memory_flags_t base_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  std::array<amdf_memory_info_t, 4> memory_infos = {};
  std::array<uint64_t, 4> addresses = {};
  for (size_t case_ordinal = 0; case_ordinal < kDeviceAccessCases.size();
       ++case_ordinal) {
    const DeviceAccessCase& access_case = kDeviceAccessCases[case_ordinal];
    const amdf_memory_device_access_t access = {
        device_,
        {
            .access = access_case.access,
            .flags = AMDF_MEMORY_FLAG_HOST_COHERENT |
                     AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                     access_case.additional_flags,
        },
    };
    amdf_memory_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create_info),
        .access_count = 1,
        .required_flags = base_flags,
        .accesses = &access,
    };
    create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
        system_scope_,
        AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        create_info.required_flags, access.requirements);
    create_info.byte_length = logical_byte_length;
    create_info.minimum_alignment = 1;
    create_info.registered_host_pointer =
        caller_pages_ + kCallerOffsets[case_ordinal];
    create_info.registered_host_cacheability =
        AMDF_HOST_CACHEABILITY_WRITE_BACK;
    ASSERT_EQ(api_->memory_create(system_scope_, &create_info,
                                  &memories_[case_ordinal], nullptr),
              AMDF_STATUS_OK)
        << "access case " << case_ordinal;

    amdf_memory_info_t& info = memory_infos[case_ordinal];
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(memories_[case_ordinal], &info),
              AMDF_STATUS_OK);
    amdf_memory_access_info_t access_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
        .structure_size = sizeof(access_info),
    };
    ASSERT_EQ(api_->memory_query_access_info(memories_[case_ordinal], 0,
                                             &access_info),
              AMDF_STATUS_OK);
    EXPECT_EQ(info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    EXPECT_EQ(access_info.access, access.requirements.access);
    EXPECT_EQ(access_info.flags & access.requirements.flags,
              access.requirements.flags);
    EXPECT_EQ((info.flags | access_info.flags) & create_info.required_flags,
              create_info.required_flags);
    EXPECT_EQ(info.source_byte_offset, kCallerOffsets[case_ordinal]);
    EXPECT_EQ(info.byte_length, logical_byte_length);
    EXPECT_EQ(info.alignment, 1u);
    EXPECT_EQ(info.native_allocation_granularity, page_size);
    EXPECT_EQ(
        info.native_allocation_byte_length,
        ((kCallerOffsets[case_ordinal] + logical_byte_length + page_size - 1) /
         page_size) *
            page_size);
    ASSERT_EQ(api_->memory_query_address(memories_[case_ordinal], 0,
                                         AMDF_MEMORY_ADDRESS_GPU,
                                         &addresses[case_ordinal]),
              AMDF_STATUS_OK);
    EXPECT_EQ(addresses[case_ordinal] & (info.alignment - 1), 0u);
    for (size_t prior_ordinal = 0; prior_ordinal < case_ordinal;
         ++prior_ordinal) {
      EXPECT_NE(addresses[case_ordinal], addresses[prior_ordinal]);
      EXPECT_TRUE(amdf_physical_memory_id_is_equal(
          &info.physical_backing_id,
          &memory_infos[prior_ordinal].physical_backing_id));
    }

    ASSERT_EQ(
        MapMemory(memories_[case_ordinal], case_ordinal, 0, logical_byte_length,
                  AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE),
        AMDF_STATUS_OK);
    EXPECT_EQ(mapping_infos_[case_ordinal].pointer,
              create_info.registered_host_pointer);
    EXPECT_EQ(mapping_infos_[case_ordinal].memory_byte_offset, 0u);
    EXPECT_EQ(mapping_infos_[case_ordinal].byte_length, logical_byte_length);
    EXPECT_EQ(mapping_infos_[case_ordinal].cacheability,
              AMDF_HOST_CACHEABILITY_WRITE_BACK);
    EXPECT_EQ(api_->host_mapping_cache_control(mappings_[case_ordinal],
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, logical_byte_length),
              AMDF_STATUS_OK);
  }

  std::memset(mapping_infos_.back().pointer, 0xC3,
              static_cast<size_t>(logical_byte_length));
  const auto* first_bytes =
      static_cast<const uint8_t*>(mapping_infos_.front().pointer);
  for (size_t i = kCallerOffsets.back() - kCallerOffsets.front();
       i < logical_byte_length; ++i) {
    ASSERT_EQ(first_bytes[i], 0xC3) << "byte " << i;
  }
  EXPECT_EQ(caller_pages_[kCallerOffsets.front() - 1], 0x5A);
  EXPECT_EQ(caller_pages_[kCallerOffsets.back() + logical_byte_length], 0x5A);

  amdf_memory_map_info_t overrun = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(overrun),
      .byte_length = logical_byte_length + 1,
      .flags = AMDF_MEMORY_MAP_FLAG_READ,
  };
  amdf_host_mapping_t* output =
      reinterpret_cast<amdf_host_mapping_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_status_code(api_->memory_map(memories_[0], &overrun, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  amdf_memory_create_info_t invalid_access = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(invalid_access),
      .memory_profile_ordinal = memory_infos[0].memory_profile_ordinal,
  };
  const amdf_memory_device_access_t write_only_access = {
      device_,
      {
          .access = AMDF_MEMORY_ACCESS_WRITE,
          .flags =
              AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      }};
  invalid_access.access_count = 1;
  invalid_access.accesses = &write_only_access;
  invalid_access.required_flags = base_flags;
  invalid_access.byte_length = 1;
  invalid_access.minimum_alignment = 1;
  invalid_access.registered_host_pointer = caller_pages_ + 67;
  invalid_access.registered_host_cacheability =
      AMDF_HOST_CACHEABILITY_WRITE_BACK;
  amdf_memory_t* invalid_output =
      reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  EXPECT_EQ(api_->memory_create(system_scope_, &invalid_access, &invalid_output,
                                nullptr),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(invalid_output), uintptr_t{1});

  for (amdf_host_mapping_t*& mapping : mappings_) {
    ASSERT_EQ(api_->host_mapping_destroy(mapping), AMDF_STATUS_OK);
    mapping = nullptr;
  }
  for (amdf_memory_t*& memory : memories_) {
    ASSERT_EQ(api_->memory_destroy(std::exchange(memory, nullptr), nullptr),
              AMDF_STATUS_OK);
  }
  std::memset(caller_pages_, 0x3C, caller_byte_length_);
  EXPECT_EQ(caller_pages_[caller_byte_length_ - 1], 0x3C);
  ASSERT_EQ(munmap(caller_pages_, caller_byte_length_), 0);
  caller_pages_ = nullptr;
}

class GpuHostNumaMemoryTest : public GpuCommandTest {
 protected:
  GpuHostNumaMemoryTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                                 AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}
};

TEST_F(GpuHostNumaMemoryTest, CopiesUsingPagesOnEveryAllowedHostNode) {
  const amdf_memory_device_access_t access = {
      device_,
      {
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
          .flags =
              AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      },
  };
  amdf_memory_scope_info_t scope_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope_info),
  };
  ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope_info),
            AMDF_STATUS_OK);
  uint32_t profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  for (uint32_t ordinal = 0; ordinal < scope_info.memory_profile_count;
       ++ordinal) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const auto status = api_->memory_scope_query_device_profile(
        system_scope_, ordinal, 1, &access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) != 0 &&
        (profile.allocation.placement_kinds &
         (UINT32_C(1) << AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA)) != 0) {
      profile_ordinal = ordinal;
      break;
    }
  }
  if (profile_ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP()
        << "Selected native lifetime/device has no placed host registration";
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  constexpr size_t kWordBits = sizeof(unsigned long) * CHAR_BIT;
  std::vector<unsigned long> allowed(page_size / sizeof(unsigned long));
  const size_t node_count = allowed.size() * kWordBits;
  ASSERT_EQ(syscall(SYS_get_mempolicy, nullptr, allowed.data(), node_count + 1,
                    nullptr, MPOL_F_MEMS_ALLOWED),
            0)
      << std::strerror(errno);
  uint64_t allowed_node_count = 0;
  for (unsigned long word : allowed) {
    allowed_node_count += std::popcount(word);
  }
  constexpr std::array<amdf_memory_flags_t, 2> kMemoryFlags = {
      0, AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE};
  constexpr uint32_t kEpochCount = 2;
  // Each epoch has two copies, optional acquire/release, and a fence. The
  // finite stream appends across all nodes and leaves one ring word free.
  constexpr size_t kMaximumEpochWords = 2 * 7 + 2 * 5 + 4;
  const uint64_t command_word_capacity = allowed_node_count *
                                             kMemoryFlags.size() * kEpochCount *
                                             kMaximumEpochWords +
                                         1;
  const uint64_t command_byte_length =
      std::bit_ceil(std::max(family_.minimum_ring_byte_length,
                             command_word_capacity * sizeof(uint32_t)));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue, command_byte_length));
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  uint32_t tested_nodes = 0;
  for (uint32_t node = 0; node < node_count; ++node) {
    if ((allowed[node / kWordBits] & (1UL << (node % kWordBits))) == 0) {
      continue;
    }
    ++tested_nodes;
    for (amdf_memory_flags_t flags : kMemoryFlags) {
      SCOPED_TRACE(::testing::Message()
                   << "node=" << node << " flags=" << flags);
      const amdf_memory_create_info_t create_info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
          .structure_size = sizeof(create_info),
          .memory_profile_ordinal = profile_ordinal,
          .access_count = 1,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE | flags,
          .byte_length = static_cast<uint64_t>(page_size) * 4,
          .accesses = &access,
          .placement = {AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA, node},
      };
      GpuMemory* memory = nullptr;
      ASSERT_NO_FATAL_FAILURE(
          CreateMemory(system_scope_, create_info, &memory));
      EXPECT_EQ(memory->info.placement.kind, create_info.placement.kind);
      EXPECT_EQ(memory->info.placement.node, node);
      amdf_memory_profile_pair_query_t pair_query = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
          .structure_size = sizeof(pair_query),
          .memory_profile_ordinal = profile_ordinal,
          .access_count = 1,
          .required_flags = create_info.required_flags,
          .accesses = &access,
      };
      pair_query.producer.kind = AMDF_MEMORY_SITE_KIND_HOST;
      pair_query.producer.value.host_access =
          AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
      pair_query.consumer.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
      pair_query.consumer.value.device = {0, family_.ordinal};
      auto source = memory->HostSite();
      auto target = memory->DeviceSite(family_.ordinal);
      amdf_cache_operations_t cache_operations = 0;
      for (uint32_t direction = 0; direction < 2; ++direction) {
        amdf_memory_pair_info_t prospective = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
            .structure_size = sizeof(prospective),
        };
        auto concrete = prospective;
        ASSERT_EQ(api_->memory_scope_query_pair_info(system_scope_, &pair_query,
                                                     &prospective),
                  AMDF_STATUS_OK);
        ASSERT_EQ(api_->memory_query_pair_info(&source, &target, &concrete),
                  AMDF_STATUS_OK);
        EXPECT_EQ(prospective.flags, concrete.flags);
        EXPECT_EQ(prospective.release.kind, concrete.release.kind);
        EXPECT_EQ(prospective.release.executor, concrete.release.executor);
        EXPECT_EQ(prospective.acquire.kind, concrete.acquire.kind);
        EXPECT_EQ(prospective.acquire.executor, concrete.acquire.executor);
        const auto& host_transition =
            direction == 0 ? concrete.release : concrete.acquire;
        ASSERT_EQ(host_transition.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
        const auto& device_transition =
            direction == 0 ? concrete.acquire : concrete.release;
        if (device_transition.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
          ASSERT_EQ(device_transition.executor,
                    AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
          ASSERT_EQ(device_transition.operation, AMDF_CACHE_OPERATION_NONE);
        } else {
          const auto operation = direction == 0
                                     ? AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
                                     : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
          ASSERT_EQ(device_transition.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
          ASSERT_EQ(device_transition.executor,
                    AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
          ASSERT_EQ(device_transition.operation, operation);
          ASSERT_NE(
              family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
              0u);
          cache_operations |= UINT64_C(1) << operation;
        }
        std::swap(pair_query.producer, pair_query.consumer);
        std::swap(source, target);
      }
      auto* bytes = static_cast<uint8_t*>(memory->host.pointer);
      const uint64_t completion =
          reinterpret_cast<uintptr_t>(bytes + page_size * 3);
      GpuStoreRelease<uint32_t>(completion, 0);
      for (uint32_t epoch = 1; epoch <= kEpochCount; ++epoch) {
        for (long i = 0; i < page_size; ++i) {
          bytes[i] = (i * 17 + epoch + node) & 0xFF;
        }
        std::memset(bytes + page_size, 0xA5, page_size * 2);
        ASSERT_GT(queue->words().size(),
                  commands.word_count() + kMaximumEpochWords);
        if ((cache_operations & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) !=
            0) {
          commands.AcquireFromSystem();
        }
        for (uint32_t destination = 1; destination <= 2; ++destination) {
          commands.CopyLinear(memory->device_address,
                              memory->device_address + page_size * destination,
                              static_cast<uint32_t>(page_size) - 1);
        }
        if ((cache_operations & AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
          commands.ReleaseToSystem();
        }
        commands.Fence32(memory->device_address + page_size * 3, epoch);
        ASSERT_NO_FATAL_FAILURE(
            queue->Publish(api_, gpu_api_, commands.word_count()));
        ASSERT_TRUE(GpuWaitEqual<uint32_t>(completion, epoch));
        for (uint32_t destination = 1; destination <= 2; ++destination) {
          EXPECT_EQ(std::memcmp(bytes, bytes + page_size * destination,
                                page_size - 1),
                    0);
          EXPECT_EQ(bytes[page_size * (destination + 1) - 1], 0xA5);
        }
        ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
        for (uint32_t page = 0; page < 4; ++page) {
          int resident_node = -1;
          ASSERT_EQ(
              syscall(SYS_get_mempolicy, &resident_node, nullptr, 0,
                      bytes + page_size * page, MPOL_F_NODE | MPOL_F_ADDR),
              0)
              << std::strerror(errno);
          EXPECT_EQ(resident_node, static_cast<int>(node)) << "page " << page;
        }
      }
    }
  }
  EXPECT_GT(tested_nodes, 0u);
  RecordProperty("numa_nodes", tested_nodes);
  RecordProperty("command_bytes", commands.word_count() * sizeof(uint32_t));
}

}  // namespace
