// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/util/memory.h"

#include "gtest/gtest.h"

void GpuMemory::Initialize(const amdf_api_t* api, amdf_memory_scope_t* scope,
                           const amdf_memory_create_info_t& create_info) {
  ASSERT_EQ(create_info.access_count, 1u);
  attachment = create_info.accesses[0];
  creation = create_info;
  creation.accesses = &attachment;
  ASSERT_EQ(api->memory_create(scope, &creation, &memory), AMDF_STATUS_OK);

  info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(api->memory_query_info(memory, &info), AMDF_STATUS_OK);
  access_info.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO;
  access_info.structure_size = sizeof(access_info);
  ASSERT_EQ(api->memory_query_access_info(memory, 0, &access_info),
            AMDF_STATUS_OK);
  ASSERT_EQ(info.memory_profile_ordinal, creation.memory_profile_ordinal);
  ASSERT_EQ(info.flags & creation.required_flags, creation.required_flags);
  ASSERT_EQ(info.byte_length, creation.byte_length);
  ASSERT_EQ(access_info.access, attachment.requirements.access);
  ASSERT_EQ(access_info.flags & attachment.requirements.flags,
            attachment.requirements.flags);
  ASSERT_GE(info.alignment, creation.minimum_alignment);
  if ((access_info.flags & AMDF_MEMORY_FLAG_DEVICE_ADDRESS) != 0) {
    ASSERT_EQ(api->memory_query_address(memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                        &device_address),
              AMDF_STATUS_OK);
  }

  if ((creation.required_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) == 0) {
    return;
  }
  amdf_memory_map_info_t map = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map),
      .byte_length = creation.byte_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  ASSERT_EQ(api->memory_map(memory, &map, &mapping), AMDF_STATUS_OK);
  host.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  host.structure_size = sizeof(host);
  ASSERT_EQ(api->host_mapping_query_info(mapping, &host), AMDF_STATUS_OK);
  ASSERT_NE(host.pointer, nullptr);
  ASSERT_EQ(host.byte_length, creation.byte_length);
  // CPU cacheability and GPU coherence are independent properties. Each
  // witness uses the returned host policy instead of inferring it from flags.
  // Initialization performs no cache maintenance. Each witness supplies its
  // own publication and visibility operations, including qualified no-ops.
}

bool GpuMemory::Release(const amdf_api_t* api) {
  if (mapping != nullptr) {
    const amdf_status_t status = api->host_mapping_destroy(mapping);
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
    mapping = nullptr;
  }
  if (memory != nullptr) {
    const amdf_status_t status = api->memory_destroy(memory);
    memory = nullptr;  // Every result consumes this handle.
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
  }
  return true;
}

amdf_memory_site_t GpuMemory::HostSite() const {
  amdf_memory_site_t site = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(site),
      .kind = AMDF_MEMORY_SITE_KIND_HOST,
  };
  site.value.host.mapping = mapping;
  return site;
}

amdf_memory_site_t GpuMemory::DeviceSite(uint32_t queue_family_ordinal) const {
  amdf_memory_site_t site = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(site),
      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
  };
  site.value.device.memory = memory;
  site.value.device.queue_family_ordinal = queue_family_ordinal;
  return site;
}

void GpuMemory::TransferHostCache(const amdf_api_t* api,
                                  uint32_t queue_family_ordinal,
                                  amdf_host_cache_operation_t operation) const {
  const auto host_site = HostSite();
  const auto device_site = DeviceSite(queue_family_ordinal);
  const bool flush = operation == AMDF_HOST_CACHE_OPERATION_FLUSH;
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  ASSERT_EQ(
      api->memory_query_pair_info(flush ? &host_site : &device_site,
                                  flush ? &device_site : &host_site, &pair),
      AMDF_STATUS_OK);
  ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
  const auto& transition = flush ? pair.release : pair.acquire;
  if (transition.kind != AMDF_CACHE_TRANSITION_KIND_NONE) {
    ASSERT_TRUE(transition.executor ==
                    AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT ||
                transition.executor == AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API);
    ASSERT_EQ(transition.host_operation, operation);
    ASSERT_EQ(api->host_mapping_cache_control(mapping, operation, 0,
                                              info.byte_length),
              AMDF_STATUS_OK);
  }
}
