// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gpu_device_fixture.h"

namespace {

struct GroupAcquisition {
  // Physical placement of the backing, independent of consumer order.
  amdf_memory_class_t memory_class;
  // Native acquisition role exercised within the selected scope.
  amdf_memory_profile_roles_t role;
};

class GpuMemoryGroupTest
    : public GpuDeviceFixture,
      public ::testing::WithParamInterface<GroupAcquisition> {
 protected:
  void TearDown() override {
    if (mapping_ != nullptr) {
      ASSERT_EQ(api_->host_mapping_destroy(mapping_), AMDF_STATUS_OK);
      mapping_ = nullptr;
    }
    if (memory_ != nullptr) {
      ASSERT_EQ(api_->memory_destroy(memory_), AMDF_STATUS_OK);
      memory_ = nullptr;
    }
    std::free(caller_storage_);
    caller_storage_ = nullptr;
    GpuDeviceFixture::TearDown();
  }

  // Original caller allocation, freed only after successful native teardown.
  uint8_t* caller_storage_ = nullptr;
  // Complete group allocation owned by this case, not by the device cache.
  amdf_memory_t* memory_ = nullptr;
  // Explicit host view released before its backing.
  amdf_host_mapping_t* mapping_ = nullptr;
};

INSTANTIATE_TEST_SUITE_P(
    Acquisition, GpuMemoryGroupTest,
    ::testing::Values(GroupAcquisition{AMDF_MEMORY_CLASS_SYSTEM,
                                       AMDF_MEMORY_PROFILE_ROLE_CREATE},
                      GroupAcquisition{AMDF_MEMORY_CLASS_SYSTEM,
                                       AMDF_MEMORY_PROFILE_ROLE_REGISTER},
                      GroupAcquisition{AMDF_MEMORY_CLASS_LOCAL,
                                       AMDF_MEMORY_PROFILE_ROLE_CREATE}));

TEST_P(GpuMemoryGroupTest, OneBackingForTwoPhysicalConsumers) {
  const amdf_memory_profile_roles_t role = GetParam().role;
  const bool local = GetParam().memory_class == AMDF_MEMORY_CLASS_LOCAL;
  amdf_memory_scope_t* scope = local ? local_scope_ : system_scope_;
  if (scope == nullptr) GTEST_SKIP() << "no local backing scope";
  const bool registered = role == AMDF_MEMORY_PROFILE_ROLE_REGISTER;
  uint32_t endpoint_count = 0;
  ASSERT_EQ(api_->endpoint_enumerate(instance_, 0, nullptr, &endpoint_count),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> summaries(endpoint_count);
  ASSERT_EQ(api_->endpoint_enumerate(instance_, endpoint_count,
                                     summaries.data(), &endpoint_count),
            AMDF_STATUS_OK);
  amdf_endpoint_t* peer_endpoint = nullptr;
  for (const auto& summary : summaries) {
    if (summary.engine_kind != AMDF_ENGINE_KIND_GPU) continue;
    amdf_endpoint_t* candidate = nullptr;
    ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(summary.id, &candidate),
              AMDF_STATUS_OK);
    if (candidate == endpoint_) continue;
    peer_endpoint = candidate;
    break;
  }
  if (peer_endpoint == nullptr) GTEST_SKIP() << "requires two physical GPUs";

  const amdf_memory_access_requirements_t requirements = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
  };
  const std::array<amdf_memory_endpoint_access_t, 2> expected_accesses = {
      {{peer_endpoint, requirements}, {endpoint_, requirements}}};
  amdf_memory_scope_info_t scope_info = {};
  scope_info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
  scope_info.structure_size = sizeof(scope_info);
  ASSERT_EQ(api_->memory_scope_query_info(scope, &scope_info), AMDF_STATUS_OK);
  amdf_memory_profile_t profile = {};
  profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
  profile.structure_size = sizeof(profile);
  std::array<amdf_memory_access_capabilities_t, 2> capabilities = {};
  for (auto& capability : capabilities) {
    capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capability.structure_size = sizeof(capability);
  }
  uint32_t selected = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  for (uint32_t i = 0; i < scope_info.memory_profile_count; ++i) {
    const amdf_status_t status = api_->memory_scope_query_profile(
        scope, i, expected_accesses.size(), expected_accesses.data(), &profile,
        capabilities.data());
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
    ASSERT_EQ(status, AMDF_STATUS_OK);
    const auto required_roles =
        role | (local ? 0 : AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
    if ((profile.roles & required_roles) != required_roles) {
      continue;
    }
    if (!local &&
        (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) == 0)
      continue;
    selected = i;
    break;
  }
  if (selected == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP() << "no joint construction profile for this physical pair";
  }
  amdf_device_t* peer_device = nullptr;
  const amdf_status_t activation =
      GetCtsDeviceCache().GetGpuDevice(peer_endpoint, &peer_device);
  if (activation == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "peer activation is unavailable for this native lifetime";
  }
  ASSERT_EQ(activation, AMDF_STATUS_OK);
  const std::array<amdf_memory_device_access_t, 2> accesses = {
      {{peer_device, requirements}, {device_, requirements}}};
  ASSERT_EQ(api_->memory_scope_query_device_profile(
                scope, selected, accesses.size(), accesses.data(), &profile,
                capabilities.data()),
            AMDF_STATUS_OK);
  amdf_memory_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.memory_profile_ordinal = selected;
  create_info.access_count = accesses.size();
  create_info.accesses = accesses.data();
  create_info.required_flags =
      local ? AMDF_MEMORY_FLAG_DEVICE_LOCAL : AMDF_MEMORY_FLAG_HOST_VISIBLE;

  const auto& construction =
      registered ? profile.registration : profile.allocation;
  create_info.byte_length = construction.native_byte_length_granularity;
  create_info.minimum_alignment = construction.minimum_alignment;
  ASSERT_NE(create_info.byte_length, 0u);
  ASSERT_LE(create_info.byte_length, construction.maximum_byte_length);
  size_t caller_offset = 0;
  size_t caller_length = 0;
  uint8_t* caller_storage = nullptr;
  if (registered) {
    const uint64_t alignment = construction.registered_host_pointer_alignment;
    ASSERT_NE(alignment, 0u);
    // Use the queried host-pointer alignment, not an OS page-size assumption.
    // The preceding/trailing storage keeps the full native page cover live.
    const uint64_t granularity = construction.native_byte_length_granularity;
    ASSERT_LE(granularity, SIZE_MAX / 4);
    ASSERT_LE(alignment, granularity);
    caller_length = static_cast<size_t>(granularity * 4);
    caller_storage = static_cast<uint8_t*>(std::malloc(caller_length));
    ASSERT_NE(caller_storage, nullptr);
    std::memset(caller_storage, 0xA7, caller_length);
    const uintptr_t candidate =
        reinterpret_cast<uintptr_t>(caller_storage) + granularity + 1;
    const uintptr_t pointer =
        candidate + (alignment - candidate % alignment) % alignment;
    caller_offset = pointer - reinterpret_cast<uintptr_t>(caller_storage);
    create_info.registered_host_pointer = caller_storage + caller_offset;
  }
  // Failed native rollback can retain pins. Only a published registration
  // establishes a teardown path that permits this fixture to free the pages.
  ASSERT_EQ(api_->memory_create(scope, &create_info, &memory_), AMDF_STATUS_OK);
  caller_storage_ = caller_storage;
  amdf_memory_info_t memory_info = {};
  memory_info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  memory_info.structure_size = sizeof(memory_info);
  ASSERT_EQ(api_->memory_query_info(memory_, &memory_info), AMDF_STATUS_OK);
  EXPECT_EQ(memory_info.access_count, accesses.size());
  EXPECT_EQ(memory_info.memory_class, GetParam().memory_class);
  EXPECT_EQ(memory_info.byte_length, create_info.byte_length);
  uint64_t common_address = 0;
  for (uint32_t i = 0; i < accesses.size(); ++i) {
    amdf_memory_access_info_t access_info = {};
    access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
    access_info.structure_size = sizeof(access_info);
    ASSERT_EQ(api_->memory_query_access_info(memory_, i, &access_info),
              AMDF_STATUS_OK);
    EXPECT_EQ(access_info.ordinal, i);
    EXPECT_EQ(access_info.access, requirements.access);
    uint64_t address = 0;
    ASSERT_EQ(api_->memory_query_address(memory_, i, AMDF_MEMORY_ADDRESS_GPU,
                                         &address),
              AMDF_STATUS_OK);
    if (i == 0) common_address = address;
    EXPECT_EQ(address, common_address);
    EXPECT_GE(address, capabilities[i].device_address.minimum_address);
    EXPECT_LE(address, capabilities[i].device_address.maximum_address);
    EXPECT_EQ(address % capabilities[i].device_address.minimum_alignment, 0u);
  }
  if (local) return;
  amdf_memory_map_info_t map_info = {};
  map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  map_info.structure_size = sizeof(map_info);
  map_info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  map_info.byte_length = create_info.byte_length;
  ASSERT_EQ(api_->memory_map(memory_, &map_info, &mapping_), AMDF_STATUS_OK);
  amdf_host_mapping_info_t mapping_info = {};
  mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  mapping_info.structure_size = sizeof(mapping_info);
  ASSERT_EQ(api_->host_mapping_query_info(mapping_, &mapping_info),
            AMDF_STATUS_OK);
  if (registered) {
    EXPECT_EQ(mapping_info.pointer, create_info.registered_host_pointer);
  }
  std::memset(mapping_info.pointer, 0x5A, mapping_info.byte_length);
  if (registered) {
    ASSERT_EQ(api_->host_mapping_destroy(mapping_), AMDF_STATUS_OK);
    mapping_ = nullptr;
    ASSERT_EQ(api_->memory_destroy(memory_), AMDF_STATUS_OK);
    memory_ = nullptr;
    for (size_t i = 0; i < caller_length; ++i) {
      const uint8_t expected =
          i >= caller_offset && i < caller_offset + create_info.byte_length
              ? 0x5A
              : 0xA7;
      ASSERT_EQ(caller_storage_[i], expected) << "byte " << i;
    }
    std::memset(caller_storage_, 0x69, caller_length);
  }
}

}  // namespace
