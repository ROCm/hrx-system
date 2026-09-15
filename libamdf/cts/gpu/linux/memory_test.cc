// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "gtest/gtest.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"

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
      if (mapping == nullptr) continue;
      ASSERT_EQ(api_->host_mapping_destroy(mapping), AMDF_STATUS_OK);
      mapping = nullptr;
    }
    for (amdf_memory_t*& memory : memories_) {
      if (memory == nullptr) continue;
      ASSERT_EQ(api_->memory_destroy(memory), AMDF_STATUS_OK);
      memory = nullptr;
    }
    if (caller_pages_ != nullptr) {
      ASSERT_EQ(munmap(caller_pages_, caller_byte_length_), 0);
      caller_pages_ = nullptr;
    }
    GpuDeviceFixture::TearDown();
  }

  amdf_memory_create_info_t MakeSystemMemoryCreateInfo() {
    amdf_memory_create_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    info.structure_size = sizeof(info);
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
    amdf_memory_map_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    info.structure_size = sizeof(info);
    info.byte_offset = byte_offset;
    info.byte_length = byte_length;
    info.flags = flags;
    amdf_status_t status = api_->memory_map(memory, &info, &mappings_[ordinal]);
    if (!amdf_status_is_ok(status)) return status;
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
          {.access = access_case.access,
           .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                    (memory_class == AMDF_MEMORY_CLASS_SYSTEM
                         ? AMDF_MEMORY_FLAG_HOST_COHERENT
                         : 0) |
                    access_case.additional_flags},
      };
      amdf_memory_create_info_t create_info = {};
      create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
      create_info.structure_size = sizeof(create_info);
      create_info.access_count = 1;
      create_info.accesses = &access;
      create_info.required_flags = required_flags;
      create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
          scope,
          AMDF_MEMORY_PROFILE_ROLE_CREATE |
              (map_memory ? AMDF_MEMORY_PROFILE_ROLE_HOST_MAP : 0),
          create_info.required_flags, access.requirements);
      create_info.byte_length = 4097;
      create_info.minimum_alignment = 1024 * 1024;

      amdf_memory_profile_t profile = {};
      profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile.structure_size = sizeof(profile);
      amdf_memory_access_capabilities_t capabilities = {};
      capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      capabilities.structure_size = sizeof(capabilities);
      ASSERT_EQ(
          QueryMemoryProfile(scope, create_info.memory_profile_ordinal,
                             access.requirements, &profile, &capabilities),
          AMDF_STATUS_OK);
      ASSERT_EQ(api_->memory_create(scope, &create_info, &memories_[0]),
                AMDF_STATUS_OK)
          << "access case " << case_ordinal;

      amdf_memory_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
      info.structure_size = sizeof(info);
      ASSERT_EQ(api_->memory_query_info(memories_[0], &info), AMDF_STATUS_OK);
      amdf_memory_access_info_t access_info = {};
      access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
      access_info.structure_size = sizeof(access_info);
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
      EXPECT_EQ(info.source_byte_offset, 0u);
      EXPECT_EQ(info.byte_length, create_info.byte_length);
      EXPECT_GE(info.native_allocation_byte_length, info.byte_length);
      EXPECT_EQ(info.native_allocation_byte_length %
                    profile.allocation.native_byte_length_granularity,
                0u);
      EXPECT_GE(info.byte_length, create_info.byte_length);
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
          EXPECT_EQ(mapping_infos_[0].flush.kind,
                    AMDF_CACHE_TRANSITION_KIND_GLOBAL);
          EXPECT_EQ(mapping_infos_[0].flush.executor,
                    AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
          EXPECT_EQ(mapping_infos_[0].flush.host_operation,
                    AMDF_HOST_CACHE_OPERATION_FLUSH);
          EXPECT_EQ(mapping_infos_[0].flush.host_fence_after,
                    AMDF_HOST_CACHE_FENCE_X86_MFENCE);
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
        EXPECT_EQ(api_->host_mapping_cache_control(
                      mappings_[0], AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                      info.byte_length),
                  AMDF_STATUS_OK);
        ASSERT_EQ(api_->host_mapping_destroy(mappings_[0]), AMDF_STATUS_OK);
        mappings_[0] = nullptr;
      } else {
        amdf_memory_map_info_t map_info = {};
        map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
        map_info.structure_size = sizeof(map_info);
        map_info.byte_length = 1;
        map_info.flags = AMDF_MEMORY_MAP_FLAG_READ;
        auto* const sentinel =
            reinterpret_cast<amdf_host_mapping_t*>(uintptr_t{1});
        amdf_host_mapping_t* output = sentinel;
        EXPECT_EQ(api_->memory_map(memories_[0], &map_info, &output),
                  amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
        EXPECT_EQ(output, sentinel);
      }

      ASSERT_EQ(api_->memory_destroy(memories_[0]), AMDF_STATUS_OK);
      memories_[0] = nullptr;
    }
  }

  void ExerciseExactSystemDeviceAccess() {
    ASSERT_NO_FATAL_FAILURE(
        ExerciseOwnedAccessMatrix(system_scope_, AMDF_MEMORY_CLASS_SYSTEM,
                                  AMDF_MEMORY_FLAG_HOST_VISIBLE));

    amdf_memory_create_info_t create_info = MakeSystemMemoryCreateInfo();
    memory_access_.requirements.access = AMDF_MEMORY_ACCESS_WRITE;
    amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
    EXPECT_EQ(api_->memory_create(system_scope_, &create_info, &output),
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
      EXPECT_EQ(api_->memory_create(local_scope_, &create_info, &output),
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
    ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &memories_[0]),
              AMDF_STATUS_OK);
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(memories_[0], &info), AMDF_STATUS_OK);
    amdf_memory_access_info_t access_info = {};
    access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
    access_info.structure_size = sizeof(access_info);
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
    for (size_t i = 0; i < 256; ++i) EXPECT_EQ(bytes[i], 0xA5);
    EXPECT_EQ(api_->host_mapping_cache_control(
                  mappings_[1], AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 256),
              AMDF_STATUS_OK);
    EXPECT_EQ(api_->host_mapping_cache_control(
                  mappings_[1], AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, 256),
              AMDF_STATUS_OK);
    EXPECT_EQ(amdf_status_code(api_->host_mapping_cache_control(
                  mappings_[1], AMDF_HOST_CACHE_OPERATION_FLUSH, 255, 2)),
              AMDF_STATUS_CODE_INVALID_ARGUMENT);
    EXPECT_EQ(amdf_status_code(api_->memory_destroy(memories_[0])),
              AMDF_STATUS_CODE_BUSY);

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
      if (mapping == nullptr) continue;
      ASSERT_EQ(api_->host_mapping_destroy(mapping), AMDF_STATUS_OK);
      mapping = nullptr;
    }
    ASSERT_EQ(api_->memory_destroy(memories_[0]), AMDF_STATUS_OK);
    memories_[0] = nullptr;
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

TEST_F(GpuLinuxMemoryTest, RejectsUnachievableSystemPlacement) {
  amdf_memory_create_info_t info = MakeSystemMemoryCreateInfo();
  info.required_flags |= AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  EXPECT_EQ(api_->memory_create(system_scope_, &info, &output),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
}

TEST_F(GpuLinuxMemoryTest, HonorsLocalPlacementCapabilities) {
  ASSERT_NO_FATAL_FAILURE(ExerciseLocalPlacementCapabilities());
}

TEST_F(GpuLinuxMemoryTest, OmitsRegistrationWhenLifetimeDoesNotSupportIt) {
  if ((features_ & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION) != 0) {
    GTEST_SKIP() << "selected lifetime supports host registration";
  }
  for (uint32_t ordinal = 0;; ++ordinal) {
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    const amdf_status_t status =
        QueryMemoryProfile(system_scope_, ordinal, memory_access_.requirements,
                           &profile, &capabilities);
    if (amdf_status_code(status) == AMDF_STATUS_CODE_OUT_OF_RANGE) break;
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
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
  amdf_gpu_device_info_t device_info = {};
  device_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
  device_info.structure_size = sizeof(device_info);
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
        {.access = access_case.access,
         .flags = AMDF_MEMORY_FLAG_HOST_COHERENT |
                  AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                  access_case.additional_flags},
    };
    amdf_memory_create_info_t create_info = {};
    create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create_info.structure_size = sizeof(create_info);
    create_info.access_count = 1;
    create_info.accesses = &access;
    create_info.required_flags = base_flags;
    create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
        system_scope_,
        AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        create_info.required_flags, access.requirements);
    create_info.byte_length = logical_byte_length;
    create_info.minimum_alignment = 1;
    create_info.registered_host_pointer =
        caller_pages_ + kCallerOffsets[case_ordinal];
    ASSERT_EQ(api_->memory_create(system_scope_, &create_info,
                                  &memories_[case_ordinal]),
              AMDF_STATUS_OK)
        << "access case " << case_ordinal;

    amdf_memory_info_t& info = memory_infos[case_ordinal];
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(memories_[case_ordinal], &info),
              AMDF_STATUS_OK);
    amdf_memory_access_info_t access_info = {};
    access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
    access_info.structure_size = sizeof(access_info);
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

  amdf_memory_map_info_t overrun = {};
  overrun.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  overrun.structure_size = sizeof(overrun);
  overrun.flags = AMDF_MEMORY_MAP_FLAG_READ;
  overrun.byte_length = logical_byte_length + 1;
  amdf_host_mapping_t* output =
      reinterpret_cast<amdf_host_mapping_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_status_code(api_->memory_map(memories_[0], &overrun, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  amdf_memory_create_info_t invalid_access = {};
  invalid_access.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  invalid_access.structure_size = sizeof(invalid_access);
  invalid_access.memory_profile_ordinal =
      memory_infos[0].memory_profile_ordinal;
  const amdf_memory_device_access_t write_only_access = {
      device_,
      {.access = AMDF_MEMORY_ACCESS_WRITE,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  invalid_access.access_count = 1;
  invalid_access.accesses = &write_only_access;
  invalid_access.required_flags = base_flags;
  invalid_access.byte_length = 1;
  invalid_access.minimum_alignment = 1;
  invalid_access.registered_host_pointer = caller_pages_ + 67;
  amdf_memory_t* invalid_output =
      reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  EXPECT_EQ(
      api_->memory_create(system_scope_, &invalid_access, &invalid_output),
      amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(invalid_output), uintptr_t{1});

  for (amdf_host_mapping_t*& mapping : mappings_) {
    ASSERT_EQ(api_->host_mapping_destroy(mapping), AMDF_STATUS_OK);
    mapping = nullptr;
  }
  for (amdf_memory_t*& memory : memories_) {
    ASSERT_EQ(api_->memory_destroy(memory), AMDF_STATUS_OK);
    memory = nullptr;
  }
  std::memset(caller_pages_, 0x3C, caller_byte_length_);
  EXPECT_EQ(caller_pages_[caller_byte_length_ - 1], 0x3C);
  ASSERT_EQ(munmap(caller_pages_, caller_byte_length_), 0);
  caller_pages_ = nullptr;
}

}  // namespace
