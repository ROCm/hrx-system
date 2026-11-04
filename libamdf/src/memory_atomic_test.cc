// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>

#include "libamdf/src/host_mapping.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/memory_pair.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/memory_test_fixture.h"

namespace amdf::testing {
namespace {

constexpr amdf_atomic_operations_t kLoadStore =
    AMDF_ATOMIC_OPERATION_LOAD | AMDF_ATOMIC_OPERATION_STORE;
constexpr amdf_atomic_operations_t kReadModifyWrite =
    AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE | AMDF_ATOMIC_OPERATION_FETCH_ADD;

// Native dependency policy; the actual public planner, construction, retained
// metadata, permission filtering, and pair composition remain under test.
amdf_memory_atomic_capabilities_t ProgramPolicy(
    const amdf_memory_native_profile_t*, amdf_memory_flags_t flags) {
  const bool uncached = (flags & AMDF_MEMORY_FLAG_DEVICE_UNCACHED) != 0;
  return {
      .device_scope_32 = kLoadStore | kReadModifyWrite,
      .device_scope_64 = kLoadStore | AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE,
      .system_scope_32 = uncached ? kLoadStore | kReadModifyWrite : 0,
      .system_scope_64 = uncached ? kLoadStore : 0,
  };
}

void ConfigureProgram(FakeDevice& device) {
  device.profile.query_program_atomics = ProgramPolicy;
  device.profile.visibility.describe_program =
      [](amdf_memory_access_t access, const amdf_device_t* base,
         amdf_memory_atomic_capabilities_t operations) {
        return amdf_memory_site_description_t{
            .capabilities = ((access & AMDF_MEMORY_ACCESS_READ) != 0
                                 ? AMDF_MEMORY_SITE_CAPABILITY_READ
                                 : UINT64_C(0)) |
                            ((access & AMDF_MEMORY_ACCESS_WRITE) != 0
                                 ? AMDF_MEMORY_SITE_CAPABILITY_WRITE
                                 : UINT64_C(0)),
            .release =
                {
                    .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                    .executor = AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM,
                    .operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
                },
            .acquire =
                {
                    .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                    .executor = AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM,
                    .operation = AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
                },
            .atomic_domain = {{19, 0}},
            .atomic_reach =
                {
                    .scope_32 = operations.system_scope_32 != 0
                                    ? uint32_t{AMDF_ATOMIC_SCOPE_SYSTEM}
                                    : 0u,
                    .scope_64 = operations.system_scope_64 != 0
                                    ? uint32_t{AMDF_ATOMIC_SCOPE_SYSTEM}
                                    : 0u,
                },
            .atomic_operations = operations,
            .device_atomic_domain = {{base->endpoint_id.words[0],
                                      base->endpoint_id.words[1]}},
        };
      };
  device.profile.guaranteed_device_access = 0;
  device.profile.guaranteed_flags |= AMDF_MEMORY_FLAG_HOST_COHERENT;
  device.profile.supported_flags |=
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
}

TEST_F(MemoryTest, ProgramQueriesPreservePolicyPermissionsAndCapturedFacts) {
  constexpr amdf_memory_access_t kPermissions[] = {
      AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE};
  for (const amdf_memory_access_t permissions : kPermissions) {
    for (const bool uncached : {false, true}) {
      SCOPED_TRACE(permissions);
      SCOPED_TRACE(uncached);
      FakeDevice device;
      InitializeFakeDevice(11, &instance_, &device);
      ConfigureProgram(device);
      device.request.requirements.access = permissions;
      if (uncached) {
        device.request.requirements.flags |= AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
      }
      amdf_memory_program_atomics_t prospective = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROGRAM_ATOMICS,
          .structure_size = sizeof(prospective),
      };
      amdf_memory_access_capabilities_t access = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(access),
          .next = &prospective,
      };
      amdf_memory_profile_t profile = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(profile),
      };
      ASSERT_EQ(amdf_memory_scope_query_device_profile(
                    &instance_.system_memory_scope, 0, 1, &device.request,
                    &profile, &access),
                AMDF_STATUS_OK);
      EXPECT_EQ(device.create_call_count, 0u);
      EXPECT_EQ(device.import_call_count, 0u);
      const bool read = (permissions & AMDF_MEMORY_ACCESS_READ) != 0;
      const bool write = (permissions & AMDF_MEMORY_ACCESS_WRITE) != 0;
      const amdf_atomic_operations_t expected =
          (read ? AMDF_ATOMIC_OPERATION_LOAD : 0) |
          (write ? AMDF_ATOMIC_OPERATION_STORE : 0) |
          (read && write ? kReadModifyWrite : 0);
      EXPECT_EQ(prospective.operations.device_scope_32, expected);
      EXPECT_EQ(prospective.operations.system_scope_32,
                uncached ? expected : 0u);
      EXPECT_EQ(prospective.operations.fabric_scope_32, 0u);
      EXPECT_EQ(prospective.operations.fabric_scope_64, 0u);

      const auto create_info = MakeMemoryCreateInfo(device);
      amdf_memory_t* memory = nullptr;
      ASSERT_EQ(amdf_memory_create(&instance_.system_memory_scope, &create_info,
                                   &memory, nullptr),
                AMDF_STATUS_OK);
      // An achieved query cannot call back into a profile producer or recover
      // policy from the now-changeable test dependency.
      device.profile_status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      amdf_memory_program_atomics_t achieved = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROGRAM_ATOMICS,
          .structure_size = sizeof(achieved),
      };
      amdf_memory_access_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
          .structure_size = sizeof(info),
          .next = &achieved,
      };
      ASSERT_EQ(amdf_memory_query_access_info(memory, 0, &info),
                AMDF_STATUS_OK);
      EXPECT_EQ(std::memcmp(&prospective.operations, &achieved.operations,
                            sizeof(achieved.operations)),
                0);
      const auto original_info = info;
      const auto original_atomics = achieved;
      EXPECT_EQ(
          amdf_status_code(amdf_memory_query_access_info(memory, 1, &info)),
          AMDF_STATUS_CODE_OUT_OF_RANGE);
      EXPECT_EQ(std::memcmp(&info, &original_info, sizeof(info)), 0);
      EXPECT_EQ(std::memcmp(&achieved, &original_atomics, sizeof(achieved)), 0);
      EXPECT_EQ(amdf_memory_destroy(memory, nullptr), AMDF_STATUS_OK);
    }
  }
}

TEST_F(MemoryTest, ProgramExtensionValidationIsTransactionalAcrossConsumers) {
  FakeDevice devices[2];
  amdf_memory_device_access_t requests[2];
  std::array<amdf_memory_program_atomics_t, 2> programs;
  std::array<amdf_memory_access_capabilities_t, 2> accesses;
  for (uint32_t i = 0; i < 2; ++i) {
    InitializeFakeDevice(i + 1, &instance_, &devices[i]);
    ConfigureProgram(devices[i]);
    requests[i] = devices[i].request;
    programs[i] = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROGRAM_ATOMICS,
        .structure_size = sizeof(programs[i]),
        .operations = {.device_scope_32 = UINT64_MAX},
    };
    accesses[i] = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(accesses[i]),
        .next = &programs[i],
        .guaranteed_access = UINT32_MAX,
    };
  }
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
      .ordinal = UINT32_MAX,
  };
  for (const uint32_t invalid_size : {0u, uint32_t{sizeof(programs[1]) - 1}}) {
    programs[1].structure_size = invalid_size;
    const auto original_programs = programs;
    const auto original_accesses = accesses;
    const auto original_profile = profile;
    EXPECT_EQ(amdf_status_code(amdf_memory_scope_query_device_profile(
                  &instance_.system_memory_scope, 0, 2, requests, &profile,
                  accesses.data())),
              AMDF_STATUS_CODE_INVALID_ARGUMENT);
    EXPECT_EQ(std::memcmp(programs.data(), original_programs.data(),
                          sizeof(programs)),
              0);
    EXPECT_EQ(std::memcmp(accesses.data(), original_accesses.data(),
                          sizeof(accesses)),
              0);
    EXPECT_EQ(std::memcmp(&profile, &original_profile, sizeof(profile)), 0);
  }
  programs[1].structure_size = sizeof(programs[1]);
  programs[1].next = &programs[1];
  EXPECT_EQ(amdf_status_code(amdf_memory_scope_query_device_profile(
                &instance_.system_memory_scope, 0, 2, requests, &profile,
                accesses.data())),
            AMDF_STATUS_CODE_UNSUPPORTED);
  programs[1].next = nullptr;
  devices[1].profile.query_program_atomics = nullptr;
  ASSERT_EQ(amdf_memory_scope_query_device_profile(
                &instance_.system_memory_scope, 0, 2, requests, &profile,
                accesses.data()),
            AMDF_STATUS_OK);
  const amdf_memory_atomic_capabilities_t empty = {};
  EXPECT_EQ(std::memcmp(&programs[1].operations, &empty, sizeof(empty)), 0);
  EXPECT_NE(programs[0].operations.device_scope_32, 0u);
  EXPECT_EQ(devices[0].create_call_count, 0u);
  EXPECT_EQ(devices[1].create_call_count, 0u);
}

TEST_F(MemoryTest, ProgramPairsDistinguishShaderDomainAndHostPublication) {
  FakeDevice device;
  InitializeFakeDevice(11, &instance_, &device);
  ConfigureProgram(device);
  device.request.requirements.flags |= AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
  const auto create_info = MakeMemoryCreateInfo(device);
  amdf_memory_t* memory = nullptr;
  ASSERT_EQ(amdf_memory_create(&instance_.system_memory_scope, &create_info,
                               &memory, nullptr),
            AMDF_STATUS_OK);
  amdf_memory_site_t shader = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(shader),
      .kind = AMDF_MEMORY_SITE_KIND_PROGRAM,
  };
  shader.value.program.memory = memory;
  amdf_memory_pair_atomics_t atomics = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_ATOMICS,
      .structure_size = sizeof(atomics),
  };
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
      .next = &atomics,
  };
  ASSERT_EQ(amdf_memory_query_pair_info(&shader, &shader, &pair),
            AMDF_STATUS_OK);
  EXPECT_EQ(atomics.producer.device_scope_32, kLoadStore | kReadModifyWrite);
  EXPECT_EQ(atomics.consumer.device_scope_32, atomics.producer.device_scope_32);
  EXPECT_EQ(pair.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM);
  EXPECT_EQ(pair.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM);

  amdf_host_mapping_t mapping = {.memory = memory};
  mapping.info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  mapping.info.properties = AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK;
  mapping.info.flush.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  mapping.info.invalidate.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  amdf_memory_site_t host = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(host),
      .kind = AMDF_MEMORY_SITE_KIND_HOST,
  };
  host.value.host.mapping = &mapping;
  ASSERT_EQ(amdf_memory_query_pair_info(&shader, &host, &pair), AMDF_STATUS_OK);
  EXPECT_EQ(atomics.producer.system_scope_32, kLoadStore | kReadModifyWrite);
  EXPECT_EQ(atomics.consumer.system_scope_32,
            kLoadStore | AMDF_ATOMIC_OPERATION_WAIT);
  EXPECT_EQ(atomics.producer.device_scope_32, 0u);
  EXPECT_EQ(atomics.consumer.device_scope_32, 0u);
  EXPECT_EQ(atomics.consumer.system_scope_32 & kReadModifyWrite, 0u);
  EXPECT_EQ(pair.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM);
  EXPECT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  ASSERT_EQ(amdf_memory_query_pair_info(&host, &shader, &pair), AMDF_STATUS_OK);
  EXPECT_EQ(atomics.consumer.system_scope_32, kLoadStore | kReadModifyWrite);
  EXPECT_EQ(atomics.producer.system_scope_32 & kReadModifyWrite, 0u);
  EXPECT_EQ(pair.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM);

  amdf_memory_profile_pair_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
      .structure_size = sizeof(query),
      .memory_profile_ordinal = 0,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .accesses = &device.request,
      .producer = {.kind = AMDF_MEMORY_SITE_KIND_HOST},
      .consumer = {.kind = AMDF_MEMORY_SITE_KIND_PROGRAM},
  };
  query.producer.value.host_access = mapping.info.flags;
  // The prospective CPU view uses the native profile's immutable host policy.
  device.profile.host_mapping.properties = AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK;
  device.profile.visibility.describe_host =
      [](const void*, amdf_external_memory_type_t, amdf_memory_flags_t,
         amdf_host_memory_properties_t properties) {
        return amdf_memory_host_description_t{
            .properties = properties,
            .flush = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE},
            .invalidate = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE},
        };
      };
  const auto achieved = atomics;
  ASSERT_EQ(amdf_memory_scope_query_pair_info(&instance_.system_memory_scope,
                                              &query, &pair),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(&atomics, &achieved, sizeof(atomics)), 0);
  const auto original_pair = pair;
  const auto original_atomics = atomics;
  shader.value.program.reserved = 1;
  EXPECT_EQ(
      amdf_status_code(amdf_memory_query_pair_info(&shader, &host, &pair)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&pair, &original_pair, sizeof(pair)), 0);
  EXPECT_EQ(std::memcmp(&atomics, &original_atomics, sizeof(atomics)), 0);
  EXPECT_EQ(amdf_memory_destroy(memory, nullptr), AMDF_STATUS_OK);
}

}  // namespace
}  // namespace amdf::testing
