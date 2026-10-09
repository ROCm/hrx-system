// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "util/mapped_memory.h"

#include <utility>

#include "gtest/gtest.h"

void CtsMappedMemory::Create(const amdf_api_t* api, amdf_memory_scope_t* scope,
                             const amdf_memory_create_info_t& create_info) {
  ASSERT_EQ(api->memory_create(scope, &create_info, &memory), AMDF_STATUS_OK);
  info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(api->memory_query_info(memory, &info), AMDF_STATUS_OK);
  ASSERT_EQ(info.byte_length, create_info.byte_length);
  ASSERT_EQ(info.flags & create_info.required_flags,
            create_info.required_flags);

  amdf_memory_map_info_t map = {};
  map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
  map.structure_size = sizeof(map);
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  map.byte_length = info.byte_length;
  ASSERT_EQ(api->memory_map(memory, &map, &mapping), AMDF_STATUS_OK);
  host.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  host.structure_size = sizeof(host);
  ASSERT_EQ(api->host_mapping_query_info(mapping, &host), AMDF_STATUS_OK);
  ASSERT_NE(host.pointer, nullptr);
  ASSERT_EQ(host.byte_length, info.byte_length);
}

bool CtsMappedMemory::Release(const amdf_api_t* api) {
  if (mapping) {
    const auto status =
        api->host_mapping_destroy(std::exchange(mapping, nullptr));
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
  }
  if (memory) {
    const auto status = api->memory_destroy(std::exchange(memory, nullptr));
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
  }
  return true;
}

amdf_memory_site_t CtsMappedMemory::HostSite() const {
  amdf_memory_site_t site = {.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                             .structure_size = sizeof(site),
                             .kind = AMDF_MEMORY_SITE_KIND_HOST};
  site.value.host_mapping = mapping;
  return site;
}

amdf_memory_site_t CtsMappedMemory::DeviceSite(
    uint32_t access_ordinal, uint32_t queue_family_ordinal) const {
  amdf_memory_site_t site = {.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                             .structure_size = sizeof(site),
                             .kind = AMDF_MEMORY_SITE_KIND_DEVICE};
  site.value.device.memory = memory;
  site.value.device.access_ordinal = access_ordinal;
  site.value.device.queue_family_ordinal = queue_family_ordinal;
  return site;
}
