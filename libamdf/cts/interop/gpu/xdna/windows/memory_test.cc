// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "amdf/amdf.h"
#include "gtest/gtest.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/util/device_cache.h"

namespace {

class WindowsGpuXdnaMemoryTest
    : public GpuDeviceFixture,
      public ::testing::WithParamInterface<uint32_t> {
 protected:
  static constexpr uint64_t kByteLength = 2 * 65536;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(GpuDeviceFixture::SetUp());
    if (IsSkipped()) {
      return;
    }
    uint32_t count = 0;
    ASSERT_EQ(api_->endpoint_enumerate(instance_, 0, nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_endpoint_summary_t> endpoints(count);
    ASSERT_EQ(
        api_->endpoint_enumerate(instance_, count, endpoints.data(), &count),
        AMDF_STATUS_OK);
    amdf_endpoint_t* xdna_endpoint = nullptr;
    amdf_device_t* xdna_device = nullptr;
    for (const auto& endpoint : endpoints) {
      if (endpoint.engine_kind != AMDF_ENGINE_KIND_XDNA) {
        continue;
      }
      ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(endpoint.id, &xdna_endpoint),
                AMDF_STATUS_OK);
      ASSERT_EQ(GetCtsDeviceCache().GetXdnaDevice(xdna_endpoint, &xdna_device),
                AMDF_STATUS_OK);
      break;
    }
    ASSERT_NE(xdna_device, nullptr) << "required XDNA endpoint is absent";
    gpu_ordinal_ = GetParam();
    xdna_ordinal_ = 1 - gpu_ordinal_;
    accesses_[gpu_ordinal_] = memory_access_;
    accesses_[gpu_ordinal_].requirements.address_kinds =
        UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    accesses_[xdna_ordinal_] = {
        xdna_device,
        {
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
            .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA,
        }};
    ASSERT_NO_FATAL_FAILURE(FindFamily(
        endpoint_, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, &families_[gpu_ordinal_]));
    ASSERT_NO_FATAL_FAILURE(FindFamily(xdna_endpoint,
                                       AMDF_QUEUE_COMMAND_TYPE_XDNA,
                                       &families_[xdna_ordinal_]));
    amdf_memory_scope_info_t scope = {};
    scope.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
    scope.structure_size = sizeof(scope);
    ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope),
              AMDF_STATUS_OK);
    profile_.ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ordinal = 0; ordinal < scope.memory_profile_count;
         ++ordinal) {
      amdf_memory_profile_t profile = {};
      profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile.structure_size = sizeof(profile);
      for (auto& capability : capabilities_) {
        capability = {};
        capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capability.structure_size = sizeof(capability);
      }
      const auto status = api_->memory_scope_query_device_profile(
          system_scope_, ordinal, accesses_.size(), accesses_.data(), &profile,
          capabilities_.data());
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
        profile_ = profile;
        break;
      }
    }
    ASSERT_NE(profile_.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  }

  void TearDown() override {
    if (mapping_) {
      ASSERT_EQ(api_->host_mapping_destroy(std::exchange(mapping_, nullptr)),
                AMDF_STATUS_OK);
    }
    if (memory_) {
      ASSERT_EQ(api_->memory_destroy(std::exchange(memory_, nullptr)),
                AMDF_STATUS_OK);
    }
    if (pages_) {
      EXPECT_TRUE(VirtualFree(pages_, 0, MEM_RELEASE));
    }
    GpuDeviceFixture::TearDown();
  }

  void FindFamily(amdf_endpoint_t* endpoint, amdf_queue_command_type_t type,
                  uint32_t* out_ordinal) {
    amdf_endpoint_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->endpoint_query_info(endpoint, &info), AMDF_STATUS_OK);
    for (uint32_t ordinal = 0; ordinal < info.queue_family_count; ++ordinal) {
      amdf_queue_family_info_t family = {};
      family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family.structure_size = sizeof(family);
      ASSERT_EQ(
          api_->endpoint_query_queue_family_info(endpoint, ordinal, &family),
          AMDF_STATUS_OK);
      if (family.command_type == type) {
        *out_ordinal = ordinal;
        return;
      }
    }
    FAIL() << "required queue family is absent";
  }

  void AllocatePages() {
    pages_ = static_cast<uint8_t*>(VirtualAlloc(
        nullptr, kByteLength, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    ASSERT_NE(pages_, nullptr);
    std::memset(pages_, 0xA5, kByteLength);
  }

  amdf_memory_create_info_t MakeCreateInfo() {
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = profile_.ordinal;
    create.access_count = accesses_.size();
    create.accesses = accesses_.data();
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length = kByteLength;
    create.minimum_alignment = profile_.registration.minimum_alignment;
    create.registered_host_pointer = pages_;
    create.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    return create;
  }

  // Complete consumers in the selected caller order.
  std::array<amdf_memory_device_access_t, 2> accesses_ = {};
  // Stable queue-family ordinals paired with the caller's consumers.
  std::array<uint32_t, 2> families_ = {};
  // Composite registration contract queried before allocating host pages.
  amdf_memory_profile_t profile_ = {};
  // Per-consumer address and access guarantees from the same cold query.
  std::array<amdf_memory_access_capabilities_t, 2> capabilities_ = {};
  // GPU position in the caller-ordered consumer list.
  uint32_t gpu_ordinal_ = 0;
  // XDNA position in the caller-ordered consumer list.
  uint32_t xdna_ordinal_ = 0;
  // Caller-owned VirtualAlloc storage, released after native registration.
  uint8_t* pages_ = nullptr;
  // Case-owned joint registration, consumed before caller storage is reused.
  amdf_memory_t* memory_ = nullptr;
  // Explicit subrange host view, destroyed before its memory owner.
  amdf_host_mapping_t* mapping_ = nullptr;
};

TEST_P(WindowsGpuXdnaMemoryTest, QualifiesSharedPagesBeforeAllocation) {
  EXPECT_EQ(profile_.roles, AMDF_MEMORY_PROFILE_ROLE_REGISTER |
                                AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(profile_.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(profile_.registration.registered_host_pointer_alignment, 65536u);
  EXPECT_EQ(profile_.registration.byte_length_granularity, 65536u);
  EXPECT_EQ(profile_.registration.minimum_alignment, 4096u);
  EXPECT_EQ(profile_.registration.maximum_alignment, 4096u);
  EXPECT_EQ(profile_.registration.registered_host_cacheability,
            AMDF_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(profile_.external_memory_support_count, 0u);
  for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
    const auto& capability = capabilities_[ordinal];
    EXPECT_EQ(capability.address_kinds &
                  accesses_[ordinal].requirements.address_kinds,
              accesses_[ordinal].requirements.address_kinds);
    EXPECT_EQ(capability.device_address.minimum_alignment, 4096u);
  }
  amdf_memory_profile_pair_query_t query = {};
  query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
  query.structure_size = sizeof(query);
  query.memory_profile_ordinal = profile_.ordinal;
  query.access_count = accesses_.size();
  query.accesses = accesses_.data();
  query.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  query.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
  for (uint32_t producer = 0; producer < 2; ++producer) {
    const uint32_t consumer = 1 - producer;
    query.producer.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    query.producer.value.device.access_ordinal = producer;
    query.producer.value.device.queue_family_ordinal = families_[producer];
    query.consumer.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    query.consumer.value.device.access_ordinal = consumer;
    query.consumer.value.device.queue_family_ordinal = families_[consumer];
    amdf_memory_pair_info_t pair = {};
    pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair.structure_size = sizeof(pair);
    ASSERT_EQ(api_->memory_scope_query_pair_info(system_scope_, &query, &pair),
              AMDF_STATUS_OK);
    EXPECT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    EXPECT_EQ(pair.release.kind, producer == gpu_ordinal_
                                     ? AMDF_CACHE_TRANSITION_KIND_GLOBAL
                                     : AMDF_CACHE_TRANSITION_KIND_NONE);
    EXPECT_EQ(pair.acquire.kind, consumer == gpu_ordinal_
                                     ? AMDF_CACHE_TRANSITION_KIND_GLOBAL
                                     : AMDF_CACHE_TRANSITION_KIND_NONE);
  }
}

TEST_P(WindowsGpuXdnaMemoryTest, RejectsInvalidRegistrationWithoutOutput) {
  ASSERT_NO_FATAL_FAILURE(AllocatePages());
  amdf_memory_t* output = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  const auto original = MakeCreateInfo();
  auto create = original;
  create.registered_host_pointer = pages_ + 4096;
  EXPECT_EQ(api_->memory_create(system_scope_, &create, &output),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  create = original;
  --create.byte_length;
  EXPECT_EQ(api_->memory_create(system_scope_, &create, &output),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  for (auto cacheability : {AMDF_HOST_CACHEABILITY_UNKNOWN,
                            AMDF_HOST_CACHEABILITY_WRITE_COMBINED}) {
    create = original;
    create.registered_host_cacheability = cacheability;
    EXPECT_EQ(api_->memory_create(system_scope_, &create, &output),
              amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  }
  create = original;
  accesses_[xdna_ordinal_].requirements.access |= AMDF_MEMORY_ACCESS_EXECUTE;
  EXPECT_EQ(api_->memory_create(system_scope_, &create, &output),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  std::memset(pages_, 0x3C, kByteLength);
}

TEST_P(WindowsGpuXdnaMemoryTest,
       PreservesCallerStorageAcrossRegistrationLifetimes) {
  ASSERT_NO_FATAL_FAILURE(AllocatePages());
  const auto create = MakeCreateInfo();
  for (uint32_t generation = 0; generation < 3; ++generation) {
    SCOPED_TRACE(generation);
    const uint8_t marker = static_cast<uint8_t>(0x51 + generation);
    std::memset(pages_, marker, kByteLength);
    ASSERT_EQ(api_->memory_create(system_scope_, &create, &memory_),
              AMDF_STATUS_OK);
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(memory_, &info), AMDF_STATUS_OK);
    EXPECT_EQ(info.access_count, accesses_.size());
    EXPECT_EQ(info.byte_length, kByteLength);
    EXPECT_EQ(info.native_allocation_byte_length, kByteLength);
    for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
      amdf_memory_access_info_t access = {};
      access.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
      access.structure_size = sizeof(access);
      ASSERT_EQ(api_->memory_query_access_info(memory_, ordinal, &access),
                AMDF_STATUS_OK);
      EXPECT_EQ(access.ordinal, ordinal);
      EXPECT_EQ(access.access, accesses_[ordinal].requirements.access);
      const auto kind = ordinal == gpu_ordinal_ ? AMDF_MEMORY_ADDRESS_GPU
                                                : AMDF_MEMORY_ADDRESS_XDNA_DMA;
      uint64_t address = 0;
      ASSERT_EQ(api_->memory_query_address(memory_, ordinal, kind, &address),
                AMDF_STATUS_OK);
      EXPECT_NE(address, 0u);
      EXPECT_EQ(address % 4096, 0u);
      uint64_t repeated = 0;
      ASSERT_EQ(api_->memory_query_address(memory_, ordinal, kind, &repeated),
                AMDF_STATUS_OK);
      EXPECT_EQ(address, repeated);
    }
    amdf_memory_map_info_t map = {};
    map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map.structure_size = sizeof(map);
    map.byte_offset = 128;
    map.byte_length = kByteLength - 256;
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    ASSERT_EQ(api_->memory_map(memory_, &map, &mapping_), AMDF_STATUS_OK);
    amdf_host_mapping_info_t mapping = {};
    mapping.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapping.structure_size = sizeof(mapping);
    ASSERT_EQ(api_->host_mapping_query_info(mapping_, &mapping),
              AMDF_STATUS_OK);
    ASSERT_EQ(mapping.pointer, pages_ + map.byte_offset);
    EXPECT_EQ(mapping.byte_length, map.byte_length);
    EXPECT_EQ(mapping.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    EXPECT_EQ(static_cast<uint8_t*>(mapping.pointer)[map.byte_length - 1],
              marker);
    ASSERT_EQ(api_->host_mapping_destroy(std::exchange(mapping_, nullptr)),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->memory_destroy(std::exchange(memory_, nullptr)),
              AMDF_STATUS_OK);
    MEMORY_BASIC_INFORMATION native = {};
    ASSERT_NE(VirtualQuery(pages_, &native, sizeof(native)), 0u);
    ASSERT_EQ(native.State, MEM_COMMIT);
    EXPECT_EQ(pages_[0], marker);
    EXPECT_EQ(pages_[kByteLength - 1], marker);
    std::memset(pages_, 0x3C, kByteLength);
  }
}

INSTANTIATE_TEST_SUITE_P(ConsumerOrder, WindowsGpuXdnaMemoryTest,
                         ::testing::Values(0u, 1u),
                         [](const ::testing::TestParamInfo<uint32_t>& info) {
                           return info.param == 0 ? "GpuFirst" : "XdnaFirst";
                         });

}  // namespace
