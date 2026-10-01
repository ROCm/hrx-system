// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/recipes/pm4_sdma_fixture.h"

#include <string>

namespace {

void CheckTransition(const amdf_cache_transition_t& transition,
                     amdf_cache_operation_t operation) {
  const bool global = operation != AMDF_CACHE_OPERATION_NONE;
  ASSERT_EQ(transition.kind, global ? AMDF_CACHE_TRANSITION_KIND_GLOBAL
                                    : AMDF_CACHE_TRANSITION_KIND_NONE);
  ASSERT_EQ(transition.executor, global ? AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE
                                        : AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  ASSERT_EQ(transition.operation, operation);
  ASSERT_EQ(transition.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
  ASSERT_EQ(transition.host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
  ASSERT_EQ(transition.host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.range_granularity, 0u);
}

std::string DescribeTransition(const amdf_cache_transition_t& transition) {
  return "kind=" + std::to_string(transition.kind) +
         ",executor=" + std::to_string(transition.executor) +
         ",operation=" + std::to_string(transition.operation);
}

}  // namespace

amdf_status_t Pm4SdmaTest::MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                            bool* out_matches) {
  const GpuQueueRequirements requirements = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
      .roles = AMDF_QUEUE_ROLE_TRANSFER,
      .publication_modes = publication_modes_,
  };
  amdf_queue_family_info_t sdma_family = {};
  bool matches = false;
  amdf_status_t status =
      FindGpuQueueFamily(api_, endpoint, requirements, &sdma_family, &matches);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (!matches) {
    *out_matches = false;
    return AMDF_STATUS_OK;
  }
  status = Pm4DispatchTest::MatchGpuEndpoint(endpoint, &matches);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (matches) {
    sdma_family_ = sdma_family;
  }
  *out_matches = matches;
  return AMDF_STATUS_OK;
}

void Pm4SdmaTest::SelectCreation(Backing& backing) {
  backing.attachment = {device_,
                        {.access = backing.access,
                         .flags = AMDF_MEMORY_FLAG_HOST_COHERENT |
                                  AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  auto& creation = backing.creation;
  creation.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  creation.structure_size = sizeof(creation);
  creation.memory_profile_ordinal = FindGpuMemoryProfileOrdinal(
      api_, system_scope_, device_,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      AMDF_MEMORY_FLAG_HOST_VISIBLE, backing.attachment.requirements);
  ASSERT_NE(creation.memory_profile_ordinal,
            AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  creation.access_count = 1;
  creation.accesses = &backing.attachment;
  creation.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  creation.byte_length = backing.byte_length;
  creation.minimum_alignment = 4096;
}

amdf_memory_profile_site_t Pm4SdmaTest::ProfileSite(
    Site site, amdf_memory_map_flags_t host_access) {
  amdf_memory_profile_site_t result = {};
  result.kind = site == Site::kHost ? AMDF_MEMORY_SITE_KIND_HOST
                                    : AMDF_MEMORY_SITE_KIND_DEVICE;
  if (site == Site::kHost) {
    result.value.host_access = host_access;
  } else {
    result.value.device.access_ordinal = 0;
    result.value.device.queue_family_ordinal =
        site == Site::kPm4 ? family_.ordinal : sdma_family_.ordinal;
  }
  return result;
}

amdf_memory_site_t Pm4SdmaTest::ConcreteSite(const GpuMemory& memory,
                                             Site site) {
  return site == Site::kHost
             ? memory.HostSite()
             : memory.DeviceSite(site == Site::kPm4 ? family_.ordinal
                                                    : sdma_family_.ordinal);
}

void Pm4SdmaTest::ResolveTransition(
    const amdf_cache_transition_t& transition, Site site,
    amdf_cache_operation_t operation,
    amdf_cache_operations_t* inout_sdma_operations) {
  if (site != Site::kSdma) {
    ASSERT_NO_FATAL_FAILURE(CheckTransition(
        transition,
        site == Site::kPm4 ? operation : AMDF_CACHE_OPERATION_NONE));
    return;
  }
  if (transition.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    ASSERT_NO_FATAL_FAILURE(
        CheckTransition(transition, AMDF_CACHE_OPERATION_NONE));
    return;
  }
  ASSERT_NO_FATAL_FAILURE(CheckTransition(transition, operation));
  ASSERT_NE(
      sdma_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR, 0u);
  const amdf_cache_operations_t bit = UINT64_C(1) << operation;
  ASSERT_NE(sdma_family_.cache_operations & bit, 0u);
  ASSERT_NE(
      sdma_family_.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      0u);
  *inout_sdma_operations |= bit;
}

void Pm4SdmaTest::ResolvePairs(
    PairQuery query_kind, const std::array<Backing, kBackingCount>& backings,
    std::span<const Edge> edges,
    std::array<amdf_cache_operations_t, kTransferPhaseCount>*
        inout_operations) {
  for (const Edge& edge : edges) {
    SCOPED_TRACE(edge.name);
    const Backing& backing = backings[edge.backing];
    amdf_memory_pair_info_t pair = {};
    pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair.structure_size = sizeof(pair);
    if (query_kind == PairQuery::kProfile) {
      const auto& creation = backing.creation;
      amdf_memory_profile_pair_query_t query = {};
      query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
      query.structure_size = sizeof(query);
      query.memory_profile_ordinal = creation.memory_profile_ordinal;
      query.required_flags = creation.required_flags;
      query.access_count = creation.access_count;
      query.accesses = creation.accesses;
      query.registered_host_cacheability =
          creation.registered_host_cacheability;
      query.producer = ProfileSite(edge.producer, AMDF_MEMORY_MAP_FLAG_WRITE);
      query.consumer = ProfileSite(edge.consumer, AMDF_MEMORY_MAP_FLAG_READ);
      ASSERT_EQ(
          api_->memory_scope_query_pair_info(system_scope_, &query, &pair),
          AMDF_STATUS_OK);
    } else {
      const auto producer = ConcreteSite(*backing.memory, edge.producer);
      const auto consumer = ConcreteSite(*backing.memory, edge.consumer);
      ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
                AMDF_STATUS_OK);
    }
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_NO_FATAL_FAILURE(ResolveTransition(
        pair.release, edge.producer, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
        &(*inout_operations)[edge.transfer_phase]));
    ASSERT_NO_FATAL_FAILURE(ResolveTransition(
        pair.acquire, edge.consumer, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        &(*inout_operations)[edge.transfer_phase]));
    const std::string prefix = std::string("pm4_sdma_") + edge.name;
    RecordProperty(prefix + "_flags", std::to_string(pair.flags));
    RecordProperty(prefix + "_release", DescribeTransition(pair.release));
    RecordProperty(prefix + "_acquire", DescribeTransition(pair.acquire));
  }
}

void Pm4SdmaTest::PrepareCoherentHandoff(
    const kernels::Kernel& kernel, PairQuery query_kind,
    std::span<const Edge> edges, std::array<Backing, kBackingCount>& backings,
    Pm4ComputeProgram* out_program,
    std::array<amdf_cache_operations_t, kTransferPhaseCount>*
        inout_operations) {
  for (auto& backing : backings) {
    ASSERT_NO_FATAL_FAILURE(SelectCreation(backing));
  }
  RecordProperty("pm4_sdma_pair_query_mode",
                 query_kind == PairQuery::kProfile ? "profile" : "concrete");
  if (query_kind == PairQuery::kProfile) {
    // Every answer precedes all seven native allocations, including code.
    ASSERT_NO_FATAL_FAILURE(
        ResolvePairs(query_kind, backings, edges, inout_operations));
  }
  for (size_t i = 0; i < kCode; ++i) {
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(system_scope_, backings[i].creation, &backings[i].memory));
  }
  *out_program = {
      0,
      kernel.program.resource1,
      kernel.program.resource2,
      kernel.program.resource3,
      kernel.group_segment_byte_length,
      {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
  };
  // Code uses the same cold READ|EXECUTE creation inputs.
  // Check its retained descriptor against the prospective selection below.
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel.executable,
                                         kernel.entry_byte_offset, out_program,
                                         "pm4_sdma", &backings[kCode].memory));
  for (const Backing& backing : backings) {
    const auto& memory = *backing.memory;
    const auto& creation = memory.creation;
    ASSERT_EQ(creation.memory_profile_ordinal,
              backing.creation.memory_profile_ordinal);
    ASSERT_EQ(creation.required_flags, backing.creation.required_flags);
    ASSERT_EQ(creation.byte_length, backing.creation.byte_length);
    ASSERT_EQ(creation.minimum_alignment, backing.creation.minimum_alignment);
    ASSERT_EQ(creation.access_count, 1u);
    ASSERT_EQ(creation.registered_host_pointer, nullptr);
    ASSERT_EQ(creation.registered_host_cacheability,
              AMDF_HOST_CACHEABILITY_UNKNOWN);
    ASSERT_EQ(memory.attachment.device, device_);
    ASSERT_EQ(memory.attachment.requirements.access,
              backing.attachment.requirements.access);
    ASSERT_EQ(memory.attachment.requirements.flags,
              backing.attachment.requirements.flags);
    ASSERT_EQ(memory.attachment.requirements.address_kinds,
              backing.attachment.requirements.address_kinds);
    ASSERT_EQ(memory.access_info.access, backing.access);
    ASSERT_EQ(memory.access_info.flags & backing.attachment.requirements.flags,
              backing.attachment.requirements.flags);
    ASSERT_EQ(memory.info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    ASSERT_NE(memory.info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
    ASSERT_EQ(memory.info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
    ASSERT_EQ(memory.host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    ASSERT_EQ(memory.host.flags,
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
    ASSERT_EQ(memory.host.memory_byte_offset, 0u);
    ASSERT_EQ(memory.host.byte_length, backing.byte_length);
    const std::string prefix = std::string("pm4_sdma_") + backing.name;
    RecordProperty(prefix + "_profile", creation.memory_profile_ordinal);
    RecordProperty(prefix + "_access", memory.access_info.access);
    RecordProperty(prefix + "_backing_flags",
                   std::to_string(memory.info.flags));
    RecordProperty(prefix + "_access_flags",
                   std::to_string(memory.access_info.flags));
    RecordProperty(prefix + "_byte_length",
                   std::to_string(backing.byte_length));
    RecordProperty(prefix + "_address", std::to_string(memory.device_address));
  }
  if (query_kind == PairQuery::kConcrete) {
    ASSERT_NO_FATAL_FAILURE(
        ResolvePairs(query_kind, backings, edges, inout_operations));
  }
  RecordProperty("pm4_sdma_upload_operations",
                 std::to_string((*inout_operations)[kUpload]));
  RecordProperty("pm4_sdma_download_operations",
                 std::to_string((*inout_operations)[kDownload]));
}
