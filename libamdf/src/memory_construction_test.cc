// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "libamdf/src/allocator.h"
#include "libamdf/src/host_mapping.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/memory_test_fixture.h"

namespace amdf::testing {
namespace {

using MemoryConstructionTest = MemoryTest;

// Native group metadata is qualified before preparation, not by the common
// memory layer interpreting a native domain or implementation identity.
static bool QueryGroupAccess(const amdf_memory_native_profile_t* backing,
                             const amdf_memory_native_profile_t* candidate,
                             amdf_memory_native_profile_t* out_profile) {
  if (backing->construction.data != candidate->construction.data) return false;
  *out_profile = *candidate;
  return true;
}

struct AllocationState {
  // Allocation attempt to fail, or UINT32_MAX when all allocations succeed.
  uint32_t failure_ordinal = UINT32_MAX;
  // Number of allocation attempts through this host allocator.
  uint32_t allocation_count = 0;
  // Number of successful host allocations not yet freed.
  uint32_t live_count = 0;

  amdf_allocator_t allocator() {
    amdf_allocator_t value = {};
    value.user_data = this;
    value.allocate = [](void* user_data, uint64_t byte_length,
                        uint64_t minimum_alignment) -> void* {
      auto* state = static_cast<AllocationState*>(user_data);
      if (state->allocation_count++ == state->failure_ordinal) return nullptr;
      const amdf_allocator_t system = amdf_allocator_system();
      void* pointer =
          system.allocate(system.user_data, byte_length, minimum_alignment);
      if (pointer != nullptr) ++state->live_count;
      return pointer;
    };
    value.free = [](void* user_data, void* pointer) {
      auto* state = static_cast<AllocationState*>(user_data);
      EXPECT_GT(state->live_count, 0u);
      --state->live_count;
      const amdf_allocator_t system = amdf_allocator_system();
      system.free(system.user_data, pointer);
    };
    return value;
  }
};

// The same owner, permission and address contracts apply to both acquisition
// roles; the device dependency supplies the role-specific native geometry.
class MemoryGroupTest : public MemoryTest,
                        public ::testing::WithParamInterface<uint32_t> {
 protected:
  void InitializeGroupDevice(uint64_t identity, FakeDevice* device) {
    InitializeFakeDevice(identity, &instance_, device);
    if (GetParam() == 0) return;
    device->profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    device->profile.registration = device->profile.allocation;
    device->profile.registration.registered_host_pointer_alignment = 1;
    device->profile.external_memory_support_count = 0;
  }

  amdf_memory_create_info_t MakeGroupCreateInfo(FakeDevice& device) {
    amdf_memory_create_info_t info = MakeMemoryCreateInfo(device);
    info.memory_profile_ordinal = GetParam();
    if (GetParam() != 0) {
      info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
      info.registered_host_pointer = pages_.data();
    }
    return info;
  }

  // Borrowed input retained until all test-owned native registrations are gone.
  alignas(4096) std::array<uint8_t, 8192> pages_ = {};
};

INSTANTIATE_TEST_SUITE_P(Acquisition, MemoryGroupTest,
                         ::testing::Values(0u, 1u));

TEST_P(MemoryGroupTest, NativeGroupUsesOneOwnerWithoutExternalTransport) {
  FakeDevice devices[2];
  amdf_memory_device_access_t accesses[2];
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeGroupDevice(i + 1, &devices[i]);
    devices[i].profile.construction = {QueryGroupAccess, devices};
    devices[i].profile.roles &= ~AMDF_MEMORY_PROFILE_ROLE_IMPORT;
    devices[i].profile.external_memory_support_count = 0;
    accesses[i] = devices[i].request;
  }
  devices[1].profile.allocation.maximum_byte_length = 8192;
  devices[1].profile.registration.maximum_byte_length = 8192;
  devices[1].profile.device_address.address_domain_ordinal = 7;
  amdf_memory_profile_t profile = {};
  profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
  profile.structure_size = sizeof(profile);
  amdf_memory_access_capabilities_t capabilities[2] = {};
  for (auto& capability : capabilities) {
    capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capability.structure_size = sizeof(capability);
  }
  ASSERT_EQ(amdf_memory_scope_query_device_profile(
                &instance_.system_memory_scope, GetParam(), 2, accesses,
                &profile, capabilities),
            AMDF_STATUS_OK);
  EXPECT_EQ(GetParam() == 0 ? profile.allocation.maximum_byte_length
                            : profile.registration.maximum_byte_length,
            8192u);
  EXPECT_EQ(capabilities[1].device_address.address_domain_ordinal, 7u);
  EXPECT_EQ(devices[0].create_call_count, 0u);
  amdf_memory_create_info_t info = MakeGroupCreateInfo(devices[0]);
  info.access_count = 2;
  info.accesses = accesses;
  amdf_memory_t* memory = nullptr;
  ASSERT_EQ(amdf_memory_create(&instance_.system_memory_scope, &info, &memory),
            AMDF_STATUS_OK);
  EXPECT_EQ(devices[0].create_call_count, 1u);
  EXPECT_EQ(devices[0].prepared_access_count, 2u);
  EXPECT_EQ(devices[1].create_call_count, 0u);
  EXPECT_EQ(devices[1].import_call_count, 0u);
  EXPECT_EQ(memory->accesses[1].native_owner_ordinal, 0u);
  EXPECT_EQ(memory->accesses[1].native, nullptr);
  EXPECT_EQ(memory->accesses[1].info.ordinal, 1u);
  EXPECT_EQ(memory->accesses[1].info.address_domain_ordinal, 7u);
  EXPECT_EQ(memory->accesses[0].addresses[AMDF_MEMORY_ADDRESS_GPU],
            memory->accesses[1].addresses[AMDF_MEMORY_ADDRESS_GPU]);
  EXPECT_EQ(amdf_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(devices[0].destroy_call_count, 1u);
  EXPECT_EQ(devices[1].destroy_call_count, 0u);
}

TEST_P(MemoryGroupTest, NativeGroupRejectsIncompatibleContractsBeforeCreate) {
  FakeDevice devices[2];
  amdf_memory_device_access_t accesses[2];
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeGroupDevice(i + 1, &devices[i]);
    devices[i].profile.construction = {QueryGroupAccess, devices};
    devices[i].profile.roles &= ~AMDF_MEMORY_PROFILE_ROLE_IMPORT;
    devices[i].profile.external_memory_support_count = 0;
    accesses[i] = devices[i].request;
  }
  amdf_memory_create_info_t info = MakeGroupCreateInfo(devices[0]);
  info.access_count = 2;
  info.accesses = accesses;
  amdf_memory_t* sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  const auto expect_rejection = [&]() {
    amdf_memory_t* memory = sentinel;
    EXPECT_EQ(amdf_status_code(amdf_memory_create(
                  &instance_.system_memory_scope, &info, &memory)),
              AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(memory, sentinel);
    EXPECT_EQ(devices[0].create_call_count, 0u);
    EXPECT_EQ(devices[1].create_call_count, 0u);
  };
  accesses[1].requirements.access = AMDF_MEMORY_ACCESS_READ;
  expect_rejection();
  accesses[1] = devices[1].request;
  devices[1].profile.construction = {QueryGroupAccess, &devices[1]};
  expect_rejection();
  devices[1].profile.construction = {QueryGroupAccess, devices};
  devices[0].profile.device_address.maximum_address = 0x1FFFF;
  devices[1].profile.device_address.minimum_address = 0x20000;
  expect_rejection();
}

TEST_P(MemoryGroupTest, NativeGroupIntersectsTheSharedAddressEnvelope) {
  FakeDevice devices[2];
  amdf_memory_device_access_t accesses[2];
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeGroupDevice(i + 1, &devices[i]);
    devices[i].profile.construction = {QueryGroupAccess, devices};
    accesses[i] = devices[i].request;
  }
  devices[0].profile.device_address.maximum_address = 0x11FFFF;
  devices[1].profile.device_address.minimum_address = 0x110000;
  amdf_memory_access_capabilities_t capabilities[2] = {};
  for (auto& capability : capabilities) {
    capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capability.structure_size = sizeof(capability);
  }
  amdf_memory_profile_t profile = {};
  profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
  profile.structure_size = sizeof(profile);
  ASSERT_EQ(amdf_memory_scope_query_device_profile(
                &instance_.system_memory_scope, GetParam(), 2, accesses,
                &profile, capabilities),
            AMDF_STATUS_OK);
  EXPECT_EQ(GetParam() == 0 ? profile.allocation.maximum_byte_length
                            : profile.registration.maximum_byte_length,
            65536u);
  EXPECT_EQ(devices[0].create_call_count, 0u);
  EXPECT_EQ(devices[1].create_call_count, 0u);
}

TEST_P(MemoryGroupTest, NativeGroupRejectsNarrowedPeerAccess) {
  FakeDevice devices[2];
  amdf_memory_device_access_t accesses[2];
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeGroupDevice(i + 1, &devices[i]);
    devices[i].profile.external_memory_support_count = 0;
    devices[i].profile.construction.query_access =
        [](const amdf_memory_native_profile_t* backing,
           const amdf_memory_native_profile_t* candidate,
           amdf_memory_native_profile_t* out_profile) {
          (void)backing;
          *out_profile = *candidate;
          out_profile->supported_device_access = AMDF_MEMORY_ACCESS_READ;
          return true;
        };
    accesses[i] = devices[i].request;
  }
  amdf_memory_create_info_t info = MakeGroupCreateInfo(devices[0]);
  info.access_count = 2;
  info.accesses = accesses;
  auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  amdf_memory_t* memory = sentinel;
  EXPECT_EQ(amdf_status_code(amdf_memory_create(&instance_.system_memory_scope,
                                                &info, &memory)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(memory, sentinel);
  EXPECT_EQ(devices[0].create_call_count, 0u);
  EXPECT_EQ(devices[1].create_call_count, 0u);
}

TEST_F(MemoryConstructionTest,
       NativeGroupPrecedesExternalConsumersAndOutlivesThem) {
  for (bool fail_import : {false, true}) {
    FakeDevice devices[3];
    amdf_memory_device_access_t accesses[3];
    std::vector<uint64_t> release_order;
    for (uint32_t i = 0; i < 3; ++i) {
      InitializeFakeDevice(i + 1, &instance_, &devices[i]);
      devices[i].release_order = &release_order;
      accesses[i] = devices[i].request;
    }
    // Place the external consumer first to exercise nonzero backing ownership.
    devices[0].profile.roles &= ~AMDF_MEMORY_PROFILE_ROLE_CREATE;
    devices[1].profile.construction = {QueryGroupAccess, devices};
    devices[2].profile.construction = {QueryGroupAccess, devices};
    if (fail_import) {
      devices[0].import_failure_stage = ImportFailureStage::kAfterAttachment;
    }
    amdf_memory_create_info_t info = MakeMemoryCreateInfo(devices[1]);
    info.access_count = 3;
    info.accesses = accesses;
    amdf_memory_t* memory = nullptr;
    const amdf_status_t status =
        amdf_memory_create(&instance_.system_memory_scope, &info, &memory);
    if (fail_import) {
      EXPECT_EQ(status, devices[0].import_status);
      EXPECT_EQ(memory, nullptr);
    } else {
      ASSERT_EQ(status, AMDF_STATUS_OK);
      EXPECT_EQ(memory->backing_access_ordinal, 1u);
      EXPECT_EQ(memory->accesses[2].native_owner_ordinal, 1u);
      EXPECT_EQ(amdf_memory_destroy(memory), AMDF_STATUS_OK);
    }
    EXPECT_EQ(devices[1].prepared_access_count, 2u);
    EXPECT_EQ(devices[1].create_call_count, 1u);
    EXPECT_EQ(devices[0].import_call_count, 1u);
    EXPECT_EQ(devices[2].import_call_count, 0u);
    EXPECT_EQ(devices[1].export_release.count, 1u);
    EXPECT_EQ(release_order, (std::vector<uint64_t>{1, 2}));
  }
}

TEST_P(MemoryGroupTest, NativeGroupFailureRollsBackOneOwner) {
  FakeDevice devices[2];
  amdf_memory_device_access_t accesses[2];
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeGroupDevice(i + 1, &devices[i]);
    devices[i].profile.construction = {QueryGroupAccess, devices};
    accesses[i] = devices[i].request;
  }
  devices[0].create_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  amdf_memory_create_info_t info = MakeGroupCreateInfo(devices[0]);
  info.access_count = 2;
  info.accesses = accesses;
  amdf_memory_t* memory = nullptr;
  EXPECT_EQ(amdf_memory_create(&instance_.system_memory_scope, &info, &memory),
            devices[0].create_status);
  EXPECT_EQ(memory, nullptr);
  EXPECT_EQ(devices[0].prepared_access_count, 2u);
  EXPECT_EQ(devices[0].destroy_call_count, 1u);
  EXPECT_EQ(devices[1].destroy_call_count, 0u);
}

TEST_F(MemoryConstructionTest, LiveProfileConstrainsTheConstructedAccessSet) {
  FakeDevice devices[2];
  amdf_memory_device_access_t accesses[2];
  amdf_memory_access_capabilities_t capabilities[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeFakeDevice(i + 1, &instance_, &devices[i]);
    accesses[i] = devices[i].request;
    capabilities[i].type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities[i].structure_size = sizeof(capabilities[i]);
  }
  // The second consumer narrows the joint length while retaining its own
  // address envelope. Selection and publication must preserve both facts.
  devices[1].profile.import.maximum_byte_length = 8192;
  devices[1].profile.device_address.maximum_address = (UINT64_C(1) << 40) - 1;
  amdf_memory_profile_t profile = {};
  profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
  profile.structure_size = sizeof(profile);
  ASSERT_EQ(amdf_memory_scope_query_device_profile(
                &instance_.system_memory_scope, 0, 2, accesses, &profile,
                capabilities),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile.allocation.maximum_byte_length, 8192u);
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(capabilities[i].device_address.maximum_address,
              devices[i].profile.device_address.maximum_address);
    EXPECT_EQ(devices[i].create_call_count, 0u);
    EXPECT_EQ(devices[i].import_call_count, 0u);
  }

  amdf_memory_create_info_t create_info = MakeMemoryCreateInfo(devices[0]);
  create_info.access_count = 2;
  create_info.accesses = accesses;
  create_info.byte_length = profile.allocation.maximum_byte_length + 1;
  amdf_memory_t* memory = nullptr;
  EXPECT_EQ(amdf_status_code(amdf_memory_create(&instance_.system_memory_scope,
                                                &create_info, &memory)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(memory, nullptr);
  EXPECT_EQ(devices[0].create_call_count, 0u);
  EXPECT_EQ(devices[1].import_call_count, 0u);
  create_info.byte_length = profile.allocation.maximum_byte_length;
  ASSERT_EQ(
      amdf_memory_create(&instance_.system_memory_scope, &create_info, &memory),
      AMDF_STATUS_OK);
  EXPECT_EQ(memory->info.byte_length, create_info.byte_length);
  EXPECT_EQ(memory->info.access_count, 2u);
  EXPECT_EQ(devices[0].create_call_count, 1u);
  EXPECT_EQ(devices[1].import_call_count, 1u);
  EXPECT_EQ(amdf_memory_destroy(memory), AMDF_STATUS_OK);
}

TEST_F(MemoryConstructionTest, LiveProfileFailurePublishesNoPartialOutputs) {
  FakeDevice devices[2];
  amdf_memory_device_access_t accesses[2];
  amdf_memory_access_capabilities_t capabilities[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeFakeDevice(i + 1, &instance_, &devices[i]);
    accesses[i] = devices[i].request;
    capabilities[i].type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities[i].structure_size = sizeof(capabilities[i]);
    capabilities[i].device_address.maximum_address = 73 + i;
  }
  amdf_memory_profile_t profile = {};
  profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
  profile.structure_size = sizeof(profile);
  profile.ordinal = 91;
  const amdf_memory_profile_t original_profile = profile;
  amdf_memory_access_capabilities_t original_capabilities[2];
  std::memcpy(original_capabilities, capabilities, sizeof(capabilities));
  const auto expect_failure = [&](amdf_status_code_t code) {
    EXPECT_EQ(amdf_status_code(amdf_memory_scope_query_device_profile(
                  &instance_.system_memory_scope, 0, 2, accesses, &profile,
                  capabilities)),
              code);
    EXPECT_EQ(std::memcmp(&profile, &original_profile, sizeof(profile)), 0);
    EXPECT_EQ(
        std::memcmp(capabilities, original_capabilities, sizeof(capabilities)),
        0);
  };
  accesses[1].device = nullptr;
  expect_failure(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  accesses[1].device = accesses[0].device;
  expect_failure(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  accesses[1] = devices[1].request;
  amdf_instance_t other_instance = {};
  devices[1].base.provider_instance = &other_instance;
  expect_failure(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  devices[1].base.provider_instance = &instance_;
  devices[1].profile_status =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  expect_failure(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(devices[i].create_call_count, 0u);
    EXPECT_EQ(devices[i].import_call_count, 0u);
  }
}

TEST_F(MemoryConstructionTest,
       NativeRoundingDoesNotExpandTheSharedLogicalRange) {
  FakeDevice devices[2];
  amdf_memory_device_access_t accesses[2];
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeFakeDevice(i + 1, &instance_, &devices[i]);
    accesses[i] = devices[i].request;
    devices[i].profile.external_memory_support[0].byte_length_alignment = 1;
  }
  const uint64_t length = 4099;
  for (uint32_t count = 1; count <= 2; ++count) {
    amdf_memory_create_info_t create_info = MakeMemoryCreateInfo(devices[0]);
    create_info.access_count = count;
    create_info.accesses = accesses;
    create_info.byte_length = length;
    amdf_memory_t* memory = nullptr;
    ASSERT_EQ(amdf_memory_create(&instance_.system_memory_scope, &create_info,
                                 &memory),
              AMDF_STATUS_OK);
    EXPECT_EQ(memory->info.byte_length, length);
    EXPECT_EQ(memory->info.native_allocation_byte_length, 8192u);
    amdf_memory_export_info_t export_info = MakeMemoryExportInfo();
    export_info.byte_offset = length;
    export_info.byte_length = 1;
    amdf_external_memory_t output = {};
    EXPECT_EQ(
        amdf_status_code(amdf_memory_export(memory, &export_info, &output)),
        AMDF_STATUS_CODE_INVALID_ARGUMENT);
    EXPECT_EQ(amdf_memory_destroy(memory), AMDF_STATUS_OK);
  }
}

TEST_F(MemoryConstructionTest,
       PreservesCallerOrderWhenBackingComesFromLaterAccess) {
  std::array<FakeDevice, 3> devices;
  std::array<amdf_memory_device_access_t, 3> accesses;
  std::vector<uint64_t> release_order;
  for (size_t i = 0; i < devices.size(); ++i) {
    InitializeFakeDevice(i + 1, &instance_, &devices[i]);
    devices[i].addresses[AMDF_MEMORY_ADDRESS_GPU] = (i + 1) * 0x100000;
    devices[i].release_order = &release_order;
    accesses[i] = devices[i].request;
  }
  devices[0].profile.roles &= ~AMDF_MEMORY_PROFILE_ROLE_CREATE;
  accesses[0].requirements.access = AMDF_MEMORY_ACCESS_READ;
  amdf_memory_create_info_t create_info = MakeMemoryCreateInfo(devices[1]);
  create_info.access_count = accesses.size();
  create_info.accesses = accesses.data();
  amdf_memory_t* memory = nullptr;
  ASSERT_EQ(
      amdf_memory_create(&instance_.system_memory_scope, &create_info, &memory),
      AMDF_STATUS_OK);
  EXPECT_EQ(devices[0].create_call_count, 0u);
  EXPECT_EQ(devices[1].create_call_count, 1u);
  EXPECT_EQ(devices[2].create_call_count, 0u);
  EXPECT_EQ(devices[0].import_call_count, 1u);
  EXPECT_EQ(devices[1].import_call_count, 0u);
  EXPECT_EQ(devices[2].import_call_count, 1u);
  EXPECT_EQ(devices[1].export_release.count, 1u);
  EXPECT_EQ(memory->backing_access_ordinal, 1u);
  EXPECT_EQ(memory->info.access_count, accesses.size());
  EXPECT_TRUE(amdf_physical_memory_id_is_equal(
      &memory->info.physical_backing_id, &devices[1].backing_id));
  for (uint32_t i = 0; i < accesses.size(); ++i) {
    devices[i].profile_status =
        amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
    amdf_memory_access_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(amdf_memory_query_access_info(memory, i, &info), AMDF_STATUS_OK);
    EXPECT_EQ(info.ordinal, i);
    EXPECT_EQ(info.access, accesses[i].requirements.access);
    uint64_t address = 0;
    ASSERT_EQ(
        amdf_memory_query_address(memory, i, AMDF_MEMORY_ADDRESS_GPU, &address),
        AMDF_STATUS_OK);
    EXPECT_EQ(address, devices[i].addresses[AMDF_MEMORY_ADDRESS_GPU]);
  }
  // The handle itself establishes shared backing; no physical-ID lookup is
  // needed.
  memory->info.physical_backing_id = {};
  amdf_memory_site_t producer = MakeMemorySite(memory, 3);
  producer.value.device.access_ordinal = 1;
  amdf_memory_site_t consumer = MakeMemorySite(memory, 5);
  amdf_memory_pair_info_t pair = {};
  pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
  pair.structure_size = sizeof(pair);
  ASSERT_EQ(amdf_memory_query_pair_info(&producer, &consumer, &pair),
            AMDF_STATUS_OK);
  EXPECT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
  EXPECT_EQ(amdf_memory_destroy(memory), AMDF_STATUS_OK);
  EXPECT_EQ(release_order, (std::vector<uint64_t>{3, 1, 2}));
  EXPECT_EQ(devices[1].export_release.count, 1u);
}

TEST_F(MemoryConstructionTest,
       FailedAggregatePreservesBackingAfterConsumerReleaseError) {
  const amdf_status_t release_error =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  for (ImportFailureStage stage : {ImportFailureStage::kBeforeAttachment,
                                   ImportFailureStage::kAfterAttachment}) {
    for (amdf_status_t destroy_status : {AMDF_STATUS_OK, release_error}) {
      AllocationState allocations;
      instance_.host_allocator = allocations.allocator();
      std::array<FakeDevice, 3> devices;
      std::array<amdf_memory_device_access_t, 3> accesses;
      std::vector<uint64_t> release_order;
      for (size_t i = 0; i < devices.size(); ++i) {
        InitializeFakeDevice(i + 1, &instance_, &devices[i]);
        devices[i].release_order = &release_order;
        accesses[i] = devices[i].request;
      }
      devices[0].profile.roles &= ~AMDF_MEMORY_PROFILE_ROLE_CREATE;
      devices[0].destroy_status = destroy_status;
      devices[2].import_failure_stage = stage;
      amdf_memory_create_info_t create_info = MakeMemoryCreateInfo(devices[1]);
      create_info.access_count = accesses.size();
      create_info.accesses = accesses.data();
      auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
      amdf_memory_t* memory = sentinel;
      EXPECT_EQ(amdf_memory_create(&instance_.system_memory_scope, &create_info,
                                   &memory),
                amdf_status_is_ok(destroy_status) ? devices[2].import_status
                                                  : destroy_status);
      EXPECT_EQ(memory, sentinel);
      EXPECT_EQ(devices[1].export_release.count, 1u);
      std::vector<uint64_t> expected_order;
      if (stage == ImportFailureStage::kAfterAttachment)
        expected_order.push_back(3);
      expected_order.push_back(1);
      if (amdf_status_is_ok(destroy_status)) expected_order.push_back(2);
      EXPECT_EQ(release_order, expected_order);
      EXPECT_EQ(devices[0].abandon_call_count,
                amdf_status_is_ok(destroy_status) ? 0u : 1u);
      EXPECT_EQ(devices[1].abandon_call_count,
                amdf_status_is_ok(destroy_status) ? 0u : 1u);
      EXPECT_EQ(devices[2].abandon_call_count, 0u);
      EXPECT_EQ(allocations.live_count, 0u);
    }
  }
}

TEST_F(MemoryConstructionTest,
       RegistersTheSameCallerPagesAndRollsBackPartialConsumers) {
  alignas(4096) std::array<uint8_t, 8192> pages = {};
  for (bool grouped : {false, true}) {
    const uint32_t successful_case = grouped ? 2 : 3;
    for (uint32_t failing_consumer = 0; failing_consumer <= successful_case;
         ++failing_consumer) {
      SCOPED_TRACE(failing_consumer);
      AllocationState allocations;
      instance_.host_allocator = allocations.allocator();
      std::array<FakeDevice, 3> devices;
      std::array<amdf_memory_device_access_t, 3> accesses;
      std::vector<uint64_t> release_order;
      for (uint32_t i = 0; i < devices.size(); ++i) {
        InitializeFakeDevice(i + 1, &instance_, &devices[i]);
        auto& device = devices[i];
        device.profile.roles = AMDF_MEMORY_PROFILE_ROLE_REGISTER |
                               AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
        device.profile.registration = device.profile.allocation;
        device.profile.registration.registered_host_pointer_alignment = 1;
        device.profile.allocation = {};
        device.profile.import = {};
        device.profile.external_memory_support_count = 0;
        device.profile.supported_flags &= ~AMDF_MEMORY_FLAG_SHAREABLE;
        if (grouped ? i == 0 : i != 0) {
          device.base.engine_kind = AMDF_ENGINE_KIND_XDNA;
          device.profile.address_kinds = UINT64_C(1)
                                         << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
        }
        if (grouped && i != 0)
          device.profile.construction = {QueryGroupAccess, devices.data()};
        if (i == failing_consumer) {
          device.create_status =
              amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
        }
        device.release_order = &release_order;
        accesses[i] = device.request;
      }
      amdf_memory_create_info_t info = MakeMemoryCreateInfo(devices[0]);
      info.memory_profile_ordinal = 1;
      info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
      info.byte_length = 4099;
      info.registered_host_pointer = pages.data();
      info.access_count = accesses.size();
      info.accesses = accesses.data();
      auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
      amdf_memory_t* memory = sentinel;
      const amdf_status_t status =
          amdf_memory_create(&instance_.system_memory_scope, &info, &memory);
      if (failing_consumer == successful_case) {
        ASSERT_EQ(status, AMDF_STATUS_OK);
        EXPECT_EQ(memory->info.byte_length, info.byte_length);
        EXPECT_EQ(memory->info.access_count, accesses.size());
        if (grouped) {
          EXPECT_EQ(memory->backing_access_ordinal, 1u);
          EXPECT_EQ(memory->accesses[2].native_owner_ordinal, 1u);
          EXPECT_EQ(memory->accesses[2].native, nullptr);
        }
        EXPECT_EQ(amdf_memory_destroy(memory), AMDF_STATUS_OK);
      } else {
        EXPECT_EQ(amdf_status_code(status),
                  AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
        EXPECT_EQ(memory, sentinel);
      }
      const uint32_t prepared_count =
          failing_consumer < 3 ? failing_consumer + 1 : 3;
      std::vector<uint64_t> expected_order;
      for (uint32_t i = prepared_count; i != 0; --i) {
        expected_order.push_back(i);
      }
      if (grouped)
        expected_order = failing_consumer == 1 ? std::vector<uint64_t>{2}
                                               : std::vector<uint64_t>{1, 2};
      EXPECT_EQ(release_order, expected_order);
      for (uint32_t i = 0; i < devices.size(); ++i) {
        const bool prepared =
            grouped ? (i == 1 || (i == 0 && failing_consumer != 1))
                    : i < prepared_count;
        EXPECT_EQ(devices[i].create_call_count, prepared ? 1u : 0u);
        EXPECT_EQ(devices[i].registered_host_pointer,
                  prepared ? pages.data() : nullptr);
        EXPECT_EQ(devices[i].import_call_count, 0u);
        EXPECT_EQ(devices[i].export_release.count, 0u);
        EXPECT_EQ(devices[i].abandon_call_count, 0u);
      }
      EXPECT_EQ(allocations.live_count, 0u);
      if (grouped) continue;
      // Independently registering another GPU cannot establish one common GPU
      // pointer. Reject that joint contract before any native preparation.
      devices[1].profile.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
      devices[1].base.engine_kind = AMDF_ENGINE_KIND_GPU;
      memory = sentinel;
      EXPECT_EQ(amdf_status_code(amdf_memory_create(
                    &instance_.system_memory_scope, &info, &memory)),
                AMDF_STATUS_CODE_UNSUPPORTED);
      EXPECT_EQ(memory, sentinel);
      EXPECT_EQ(release_order, expected_order);
      for (uint32_t i = 0; i < devices.size(); ++i) {
        const bool prepared =
            grouped ? (i == 1 || (i == 0 && failing_consumer != 1))
                    : i < prepared_count;
        EXPECT_EQ(devices[i].create_call_count, prepared ? 1u : 0u);
      }
      if (memory != sentinel) {
        EXPECT_EQ(amdf_memory_destroy(memory), AMDF_STATUS_OK);
      }
      EXPECT_EQ(allocations.live_count, 0u);
    }
  }
}

TEST_F(MemoryConstructionTest,
       RejectsDuplicateAndForeignLiveConsumersBeforeNativeWork) {
  FakeDevice device;
  InitializeFakeDevice(1, &instance_, &device);
  std::array<amdf_memory_device_access_t, 2> accesses = {device.request,
                                                         device.request};
  amdf_memory_create_info_t create_info = MakeMemoryCreateInfo(device);
  create_info.access_count = accesses.size();
  create_info.accesses = accesses.data();
  auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  amdf_memory_t* memory = sentinel;
  EXPECT_EQ(
      amdf_memory_create(&instance_.system_memory_scope, &create_info, &memory),
      amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(memory, sentinel);
  amdf_instance_t other_instance = {};
  device.base.provider_instance = &other_instance;
  create_info.access_count = 1;
  EXPECT_EQ(
      amdf_memory_create(&instance_.system_memory_scope, &create_info, &memory),
      amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(memory, sentinel);
  EXPECT_EQ(device.create_call_count, 0u);
  EXPECT_EQ(device.import_call_count, 0u);
}

TEST_F(MemoryConstructionTest, AllocationFailureHasNoNativeReleaseObligation) {
  for (uint32_t failure_ordinal : {0u, 1u, 2u}) {
    AllocationState allocations;
    allocations.failure_ordinal = failure_ordinal;
    FakeDevice device;
    InitializeFakeDevice(7, &instance_, &device);
    device.base.host_allocator = allocations.allocator();
    instance_.host_allocator = allocations.allocator();
    const amdf_memory_create_info_t create_info = MakeMemoryCreateInfo(device);
    auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
    amdf_memory_t* memory = sentinel;
    EXPECT_EQ(amdf_memory_create(&instance_.system_memory_scope, &create_info,
                                 &memory),
              amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED));
    EXPECT_EQ(memory, sentinel);
    EXPECT_EQ(device.create_call_count, failure_ordinal == 2 ? 1u : 0u);
    EXPECT_EQ(device.destroy_call_count, 0u);
    EXPECT_EQ(device.abandon_call_count, 0u);
    EXPECT_EQ(allocations.live_count, 0u);
  }
}

TEST_F(MemoryConstructionTest,
       AggregateAllocationFailuresPreserveTheInputMove) {
  for (amdf_memory_profile_roles_t role :
       {AMDF_MEMORY_PROFILE_ROLE_CREATE, AMDF_MEMORY_PROFILE_ROLE_IMPORT}) {
    // Plan, common owner, then three independently prepared native accesses.
    for (uint32_t failure_ordinal = 0; failure_ordinal <= 5;
         ++failure_ordinal) {
      SCOPED_TRACE(::testing::Message()
                   << "role=" << role << " allocation=" << failure_ordinal);
      AllocationState allocations;
      allocations.failure_ordinal = failure_ordinal;
      instance_.host_allocator = allocations.allocator();
      std::array<FakeDevice, 3> devices;
      std::array<amdf_memory_device_access_t, 3> accesses;
      for (uint32_t i = 0; i < devices.size(); ++i) {
        InitializeFakeDevice(i + 1, &instance_, &devices[i]);
        accesses[i] = devices[i].request;
      }
      devices[0].profile.roles &= ~AMDF_MEMORY_PROFILE_ROLE_CREATE;
      ReleaseState release = {};
      amdf_external_memory_t external = {};
      external.type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
      external.payload.file_descriptor = 83;
      external.byte_length = 4096;
      external.physical_backing_id = devices[0].backing_id;
      external.release = RecordRelease;
      external.release_user_data = &release;
      const amdf_external_memory_t original = external;
      auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
      amdf_memory_t* memory = sentinel;
      amdf_status_t status;
      if (role == AMDF_MEMORY_PROFILE_ROLE_CREATE) {
        amdf_memory_create_info_t create_info =
            MakeMemoryCreateInfo(devices[1]);
        create_info.access_count = accesses.size();
        create_info.accesses = accesses.data();
        status = amdf_memory_create(&instance_.system_memory_scope,
                                    &create_info, &memory);
        EXPECT_EQ(devices[1].export_release.count,
                  failure_ordinal >= 3 ? 1u : 0u);
      } else {
        amdf_memory_import_info_t import_info =
            MakeMemoryImportInfo(devices[0]);
        import_info.access_count = accesses.size();
        import_info.accesses = accesses.data();
        status = amdf_memory_import(&instance_.system_memory_scope,
                                    &import_info, &external, &memory);
        EXPECT_EQ(release.count, failure_ordinal == 5 ? 1u : 0u);
      }
      if (failure_ordinal == 5) {
        ASSERT_EQ(status, AMDF_STATUS_OK);
        EXPECT_EQ(amdf_memory_destroy(memory), AMDF_STATUS_OK);
      } else {
        EXPECT_EQ(amdf_status_code(status),
                  AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
        EXPECT_EQ(memory, sentinel);
        EXPECT_EQ(std::memcmp(&external, &original, sizeof(external)), 0);
      }
      uint32_t release_count = 0;
      for (const auto& device : devices) {
        release_count += device.destroy_call_count;
        EXPECT_EQ(device.abandon_call_count, 0u);
      }
      EXPECT_EQ(release_count, failure_ordinal > 2 ? failure_ordinal - 2 : 0u);
      EXPECT_EQ(allocations.live_count, 0u);
      amdf_external_memory_release(&external);
      EXPECT_EQ(release.count, 1u);
    }
  }
}

TEST_F(MemoryConstructionTest, FailedPrepareDiscardsUnpublishedNativeMetadata) {
  const amdf_status_t release_error =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  for (amdf_status_t destroy_status : {AMDF_STATUS_OK, release_error}) {
    AllocationState allocations;
    FakeDevice device;
    InitializeFakeDevice(9, &instance_, &device);
    device.base.host_allocator = allocations.allocator();
    instance_.host_allocator = allocations.allocator();
    device.create_status =
        amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    device.destroy_status = destroy_status;
    const amdf_memory_create_info_t create_info = MakeMemoryCreateInfo(device);
    auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
    amdf_memory_t* memory = sentinel;
    EXPECT_EQ(amdf_memory_create(&instance_.system_memory_scope, &create_info,
                                 &memory),
              amdf_status_is_ok(destroy_status) ? device.create_status
                                                : destroy_status);
    EXPECT_EQ(memory, sentinel);
    EXPECT_EQ(device.destroy_call_count, 1u);
    EXPECT_EQ(device.abandon_call_count,
              amdf_status_is_ok(destroy_status) ? 0u : 1u);
    EXPECT_EQ(allocations.live_count, 0u);
  }
}

TEST_F(MemoryConstructionTest, FailedImportPreservesInputOnReleaseError) {
  AllocationState allocations;
  FakeDevice device;
  InitializeFakeDevice(13, &instance_, &device);
  device.base.host_allocator = allocations.allocator();
  instance_.host_allocator = allocations.allocator();
  device.import_failure_stage = ImportFailureStage::kAfterAttachment;
  device.destroy_status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  ReleaseState release = {};
  amdf_external_memory_t external = {};
  external.type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
  external.byte_length = 4096;
  external.payload.file_descriptor = 91;
  external.physical_backing_id = device.backing_id;
  external.release = RecordRelease;
  external.release_user_data = &release;
  const amdf_external_memory_t original = external;
  const amdf_memory_import_info_t import_info = MakeMemoryImportInfo(device);
  auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  amdf_memory_t* memory = sentinel;
  EXPECT_EQ(amdf_memory_import(&instance_.system_memory_scope, &import_info,
                               &external, &memory),
            device.destroy_status);
  EXPECT_EQ(memory, sentinel);
  EXPECT_EQ(std::memcmp(&external, &original, sizeof(external)), 0);
  EXPECT_EQ(release.count, 0u);
  EXPECT_EQ(device.destroy_call_count, 1u);
  EXPECT_EQ(device.abandon_call_count, 1u);
  EXPECT_EQ(allocations.live_count, 0u);
  amdf_external_memory_release(&external);
  EXPECT_EQ(release.count, 1u);
}

}  // namespace
}  // namespace amdf::testing
