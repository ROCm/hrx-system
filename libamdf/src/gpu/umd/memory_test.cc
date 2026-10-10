// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory.h"

#include <cstring>

#include "amdf/gpu.h"
#include "gtest/gtest.h"

namespace {

TEST(GpuMemoryPairTest, DescribesExactLocalQueueSites) {
  amdf_queue_family_info_t family = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
      .roles = AMDF_QUEUE_ROLE_CACHE_CONTROL,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
  };
  const amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = AMDF_MEMORY_FLAG_HOST_COHERENT,
      .queue_family_info = &family,
  };
  for (const auto command_type :
       {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
        AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA}) {
    SCOPED_TRACE(command_type);
    family.command_type = command_type;
    family.format_features = command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA
                                 ? AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR
                                 : 0;
    amdf_memory_site_description_t description = {};
    ASSERT_EQ(amdf_gpu_umd_memory_describe_site(&query, &description),
              AMDF_STATUS_OK);
    EXPECT_EQ(description.capabilities, AMDF_MEMORY_SITE_CAPABILITY_READ |
                                            AMDF_MEMORY_SITE_CAPABILITY_WRITE);
    EXPECT_EQ(description.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    EXPECT_EQ(description.release.executor,
              AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    EXPECT_EQ(description.release.operation,
              AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
    EXPECT_EQ(description.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    EXPECT_EQ(description.acquire.operation,
              AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
    EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
  }

  amdf_memory_site_description_t description = {};
  family.format_features = 0;
  for (const auto command_type :
       {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, AMDF_QUEUE_COMMAND_TYPE_UNKNOWN}) {
    SCOPED_TRACE(command_type);
    family.command_type = command_type;
    std::memset(&description, 0xA5, sizeof(description));
    const amdf_memory_site_description_t original = description;
    EXPECT_EQ(amdf_status_code(
                  amdf_gpu_umd_memory_describe_site(&query, &description)),
              AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(std::memcmp(&description, &original, sizeof(description)), 0);
  }
}

TEST(GpuMemoryPairTest, ScopedSdmaUsesPerCommandSystemVisibility) {
  const amdf_queue_family_info_t family = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
      .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
      .format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE,
      .roles = AMDF_QUEUE_ROLE_TRANSFER,
  };
  constexpr amdf_memory_access_t kAccesses[] = {
      0, AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE};
  for (const amdf_memory_access_t access : kAccesses) {
    SCOPED_TRACE(access);
    const amdf_memory_site_query_t query = {
        .access = access,
        .flags = AMDF_MEMORY_FLAG_HOST_COHERENT,
        .queue_family_info = &family,
    };
    amdf_memory_site_description_t description = {};
    ASSERT_EQ(amdf_gpu_umd_memory_describe_site(&query, &description),
              AMDF_STATUS_OK);
    amdf_memory_site_capabilities_t expected_capabilities =
        AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN |
        AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN;
    if ((access & AMDF_MEMORY_ACCESS_READ) != 0) {
      expected_capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
    }
    if ((access & AMDF_MEMORY_ACCESS_WRITE) != 0) {
      expected_capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
    }
    EXPECT_EQ(description.capabilities, expected_capabilities);
    for (const auto* transition :
         {&description.release, &description.acquire}) {
      EXPECT_EQ(transition->kind, AMDF_CACHE_TRANSITION_KIND_NONE);
      EXPECT_EQ(transition->executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
      EXPECT_EQ(transition->operation, AMDF_CACHE_OPERATION_NONE);
      EXPECT_EQ(transition->host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
      EXPECT_EQ(transition->host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
      EXPECT_EQ(transition->host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
      EXPECT_EQ(transition->host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
      EXPECT_EQ(transition->range_granularity, 0u);
    }
    EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
  }
}

TEST(GpuMemoryPairTest, SdmaPayloadPoliciesExcludeHostVisibleLocalApertures) {
  amdf_queue_family_info_t family = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
      .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
      .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
  };
  amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL,
      .queue_family_info = &family,
  };
  for (const amdf_queue_format_features_t feature :
       {AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
        AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE}) {
    SCOPED_TRACE(feature);
    family.format_features = feature;
    query.flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL;
    amdf_memory_site_description_t description = {};
    ASSERT_EQ(amdf_gpu_umd_memory_describe_site(&query, &description),
              AMDF_STATUS_OK);
    for (const auto flags :
         {AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE,
          AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE |
              AMDF_MEMORY_FLAG_HOST_COHERENT}) {
      SCOPED_TRACE(flags);
      query.flags = flags;
      std::memset(&description, 0xA5, sizeof(description));
      const amdf_memory_site_description_t original = description;
      EXPECT_EQ(amdf_status_code(
                    amdf_gpu_umd_memory_describe_site(&query, &description)),
                AMDF_STATUS_CODE_UNSUPPORTED);
      EXPECT_EQ(std::memcmp(&description, &original, sizeof(description)), 0);
    }
  }
}

TEST(GpuMemoryPairTest, SystemStoreReachRequiresBothMemoryAndQueueWidths) {
  amdf_queue_family_info_t family = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
      .roles = AMDF_QUEUE_ROLE_CACHE_CONTROL | AMDF_QUEUE_ROLE_ATOMIC,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
  };
  amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = AMDF_MEMORY_FLAG_HOST_COHERENT,
      .queue_family_info = &family,
  };
  for (uint32_t memory_widths = 0; memory_widths < 4; ++memory_widths) {
    query.atomic_operations_32 =
        (memory_widths & 1) ? AMDF_ATOMIC_OPERATION_STORE : 0;
    query.atomic_operations_64 =
        (memory_widths & 2) ? AMDF_ATOMIC_OPERATION_STORE : 0;
    for (uint32_t queue_widths = 0; queue_widths < 4; ++queue_widths) {
      SCOPED_TRACE(::testing::Message()
                   << "memory=" << memory_widths << " queue=" << queue_widths);
      family.atomic_capabilities.operations_32 =
          (queue_widths & 1) ? AMDF_ATOMIC_OPERATION_STORE : 0;
      family.atomic_capabilities.operations_64 =
          (queue_widths & 2) ? AMDF_ATOMIC_OPERATION_STORE : 0;
      amdf_memory_site_description_t description = {};
      ASSERT_EQ(
          amdf_gpu_umd_memory_describe_system_store_site(&query, &description),
          AMDF_STATUS_OK);
      const uint32_t shared_widths = memory_widths & queue_widths;
      EXPECT_EQ(description.atomic_reach.scope_32,
                (shared_widths & 1) ? AMDF_ATOMIC_SCOPE_SYSTEM
                                    : AMDF_ATOMIC_SCOPE_NONE);
      EXPECT_EQ(description.atomic_reach.scope_64,
                (shared_widths & 2) ? AMDF_ATOMIC_SCOPE_SYSTEM
                                    : AMDF_ATOMIC_SCOPE_NONE);
      EXPECT_EQ(
          amdf_memory_compatibility_domain_is_valid(&description.atomic_domain),
          shared_widths != 0);
      EXPECT_EQ(description.release.operation,
                AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
      EXPECT_EQ(description.acquire.operation,
                AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
    }
  }
}

TEST(GpuMemoryPairTest, SystemStoresDoNotGrantOtherAccessOrQueueContracts) {
  amdf_queue_family_info_t family = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
      .roles = AMDF_QUEUE_ROLE_CACHE_CONTROL | AMDF_QUEUE_ROLE_ATOMIC,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      .atomic_capabilities =
          {
              .operations_32 = AMDF_ATOMIC_OPERATION_STORE,
              .operations_64 = AMDF_ATOMIC_OPERATION_STORE,
          },
  };
  amdf_memory_site_query_t query = {
      .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .flags = AMDF_MEMORY_FLAG_HOST_COHERENT,
      .atomic_operations_32 = AMDF_ATOMIC_OPERATION_STORE,
      .atomic_operations_64 = AMDF_ATOMIC_OPERATION_STORE,
      .queue_family_info = &family,
  };
  const auto expect_no_reach = [&]() {
    amdf_memory_site_description_t description = {};
    ASSERT_EQ(
        amdf_gpu_umd_memory_describe_system_store_site(&query, &description),
        AMDF_STATUS_OK);
    EXPECT_EQ(description.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    EXPECT_EQ(description.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
    EXPECT_FALSE(
        amdf_memory_compatibility_domain_is_valid(&description.atomic_domain));
  };
  constexpr amdf_memory_flags_t kOtherBackingFlags[] = {
      AMDF_MEMORY_FLAG_HOST_VISIBLE,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_LOCAL};
  for (const auto flags : kOtherBackingFlags) {
    query.flags = flags;
    expect_no_reach();
  }
  query.flags = AMDF_MEMORY_FLAG_HOST_COHERENT;
  for (const auto access :
       {AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE}) {
    query.access = access;
    expect_no_reach();
  }
  query.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  family.roles = AMDF_QUEUE_ROLE_CACHE_CONTROL;
  expect_no_reach();
  family.roles |= AMDF_QUEUE_ROLE_ATOMIC;
  for (const auto command :
       {AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA}) {
    family.command_type = command;
    family.format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
    expect_no_reach();
  }
  family.command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  query.atomic_operations_32 = AMDF_ATOMIC_OPERATION_ADD;
  query.atomic_operations_64 = AMDF_ATOMIC_OPERATION_ADD;
  expect_no_reach();
}

}  // namespace
