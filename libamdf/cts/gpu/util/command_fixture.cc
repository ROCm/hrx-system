// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/util/command_fixture.h"

amdf_status_t FindGpuQueueFamily(const amdf_api_t* api,
                                 amdf_endpoint_t* endpoint,
                                 const GpuQueueRequirements& requirements,
                                 amdf_queue_family_info_t* out_family,
                                 bool* out_matches) {
  amdf_endpoint_info_t endpoint_info = {};
  endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
  endpoint_info.structure_size = sizeof(endpoint_info);
  amdf_status_t status = api->endpoint_query_info(endpoint, &endpoint_info);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  bool matches = false;
  for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count;
       ++ordinal) {
    amdf_queue_family_info_t family = {};
    family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
    family.structure_size = sizeof(family);
    status = api->endpoint_query_queue_family_info(endpoint, ordinal, &family);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (family.command_type == requirements.command_type &&
        family.format_version == 1 &&
        (family.roles & requirements.roles) == requirements.roles &&
        (family.format_features & requirements.format_features) ==
            requirements.format_features &&
        (family.cache_operations & requirements.cache_operations) ==
            requirements.cache_operations &&
        (family.cache_transition_kinds & requirements.cache_transition_kinds) ==
            requirements.cache_transition_kinds &&
        SelectGpuHostPublication(family, requirements.publication_modes) != 0) {
      *out_family = family;
      matches = true;
      break;
    }
  }
  *out_matches = matches;
  return AMDF_STATUS_OK;
}

amdf_status_t GpuCommandTest::MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                               bool* out_matches) {
  return FindGpuQueueFamily(api_, endpoint, requirements_, &family_,
                            out_matches);
}

void GpuCommandTest::CreateMemory(amdf_memory_access_t access,
                                  uint64_t byte_length,
                                  GpuMemory** out_memory) {
  const amdf_memory_device_access_t attachment = {
      device_,
      {.access = access,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  amdf_memory_create_info_t creation = {};
  creation.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  creation.structure_size = sizeof(creation);
  creation.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  creation.memory_profile_ordinal = FindGpuMemoryProfileOrdinal(
      api_, system_scope_, device_,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      creation.required_flags, attachment.requirements);
  ASSERT_NE(creation.memory_profile_ordinal,
            AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  creation.access_count = 1;
  creation.accesses = &attachment;
  creation.byte_length = byte_length;
  creation.minimum_alignment = 4096;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, creation, out_memory));
}

void GpuCommandTest::CreateMemory(amdf_memory_scope_t* scope,
                                  const amdf_memory_create_info_t& create_info,
                                  GpuMemory** out_memory) {
  auto& memory = memories_.emplace_back();
  ASSERT_NO_FATAL_FAILURE(memory.Initialize(api_, scope, create_info));
  *out_memory = &memory;
}

void GpuCommandTest::CreateQueue(GpuUserQueue** out_queue,
                                 amdf_queue_producer_mode_t producer_mode,
                                 const amdf_gpu_queue_scratch_t& scratch) {
  CreateQueue(family_, out_queue, producer_mode, scratch);
}

void GpuCommandTest::CreateQueue(const amdf_queue_family_info_t& family,
                                 GpuUserQueue** out_queue,
                                 amdf_queue_producer_mode_t producer_mode,
                                 const amdf_gpu_queue_scratch_t& scratch) {
  auto& queue = queues_.emplace_back();
  ASSERT_NO_FATAL_FAILURE(queue.Initialize(api_, gpu_api_, device_, family,
                                           producer_mode, scratch));
  *out_queue = &queue;
}

void GpuCommandTest::CreateQueue(GpuCommandQueue** out_queue) {
  CreateQueue(family_, out_queue);
}

void GpuCommandTest::CreateQueue(const amdf_queue_family_info_t& family,
                                 GpuCommandQueue** out_queue) {
  const auto publication =
      SelectGpuHostPublication(family, requirements_.publication_modes);
  ASSERT_NE(publication, 0u);
  auto& queue = command_queues_.emplace_back();
  ASSERT_NO_FATAL_FAILURE(queue.Initialize(api_, gpu_api_, device_,
                                           system_scope_, family, publication));
  *out_queue = &queue;
}

void GpuCommandTest::TearDown() {
  // Stop after any failure: a consumed queue handle alone does not authorize
  // releasing its backing or unloading the provider. The outer cache retains
  // native parents when their children cannot be removed.
  for (auto& queue : queues_) {
    ASSERT_TRUE(queue.Release(api_));
  }
  for (auto& queue : command_queues_) {
    ASSERT_TRUE(queue.Release(api_));
  }
  for (auto& memory : memories_) {
    ASSERT_TRUE(memory.Release(api_));
  }
}
