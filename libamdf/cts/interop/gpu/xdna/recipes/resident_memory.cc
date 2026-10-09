// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/interop/gpu/xdna/recipes/resident_memory.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>

namespace {

void FindEndpoint(const amdf_api_t* api, amdf_memory_scope_t* scope,
                  amdf_memory_profile_roles_t roles, amdf_memory_flags_t flags,
                  amdf_external_memory_type_t transport,
                  amdf_external_memory_support_flags_t transport_flags,
                  const amdf_external_memory_provenance_t* provenance,
                  ResidentImportPlan::Endpoint& endpoint) {
  endpoint.profile.ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info)};
  ASSERT_EQ(api->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
  for (uint32_t ordinal = 0; ordinal < info.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile = {.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                                     .structure_size = sizeof(profile)};
    amdf_memory_access_capabilities_t access = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(access)};
    const auto status = api->memory_scope_query_device_profile(
        scope, ordinal, 1, &endpoint.access, &profile, &access);
    if (amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    if (profile.memory_class != AMDF_MEMORY_CLASS_SYSTEM ||
        (profile.roles & roles) != roles ||
        (profile.supported_flags & flags) != flags) {
      continue;
    }
    for (uint32_t i = 0; i < profile.external_memory_support_count; ++i) {
      const auto& support = profile.external_memory_support[i];
      if (support.type == transport &&
          (support.flags & transport_flags) == transport_flags &&
          (!provenance || amdf_external_memory_provenance_is_equal(
                              &support.provenance, provenance))) {
        endpoint.profile = profile;
        endpoint.transport = support;
        return;
      }
    }
  }
}

uint64_t RoundUp(uint64_t value, uint64_t granularity) {
  return (value + granularity - 1) / granularity * granularity;
}

}  // namespace

void ResidentImportPlan::Find(const amdf_api_t* api, amdf_memory_scope_t* scope,
                              const amdf_memory_device_access_t& gpu_access,
                              const amdf_memory_device_access_t& xdna_access,
                              amdf_external_memory_type_t transport,
                              ResidentSourceOffset source_offset) {
  source.access = gpu_access;
  destination.access = xdna_access;
  ASSERT_NO_FATAL_FAILURE(FindEndpoint(
      api, scope,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_EXPORT |
          AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE, transport,
      AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
          AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET,
      nullptr, source));
  if (source.profile.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP() << "GPU export transport " << transport
                 << " is not advertised";
  }
  ASSERT_NO_FATAL_FAILURE(
      FindEndpoint(api, scope, AMDF_MEMORY_PROFILE_ROLE_IMPORT,
                   AMDF_MEMORY_FLAG_HOST_VISIBLE, transport,
                   AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                       AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET,
                   &source.transport.provenance, destination));
  if (destination.profile.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP() << "matching XDNA import transport " << transport
                 << " and provenance are not advertised";
  }
  ASSERT_GT(source.profile.allocation.minimum_alignment, 0u);
  ASSERT_GT(source.profile.allocation.byte_length_granularity, 0u);
  ASSERT_GT(source.profile.allocation.native_byte_length_granularity, 0u);
  ASSERT_GT(destination.profile.import.minimum_alignment, 0u);
  ASSERT_GT(destination.profile.import.byte_length_granularity, 0u);
  ASSERT_GT(source.transport.source_offset_alignment, 0u);
  ASSERT_GT(destination.transport.source_offset_alignment, 0u);
  ASSERT_GT(source.transport.byte_length_alignment, 0u);
  ASSERT_GT(destination.transport.byte_length_alignment, 0u);
  minimum_alignment =
      std::max(uint64_t{64}, destination.profile.import.minimum_alignment);
  ASSERT_LE(minimum_alignment, destination.profile.import.maximum_alignment);
  ASSERT_EQ(
      source.profile.allocation.native_byte_length_prefix % minimum_alignment,
      0u);
  uint64_t offset_alignment =
      std::lcm(minimum_alignment, source.profile.allocation.minimum_alignment);
  offset_alignment =
      std::lcm(offset_alignment, source.transport.source_offset_alignment);
  offset_alignment =
      std::lcm(offset_alignment, destination.transport.source_offset_alignment);
  source_byte_offset =
      source_offset == ResidentSourceOffset::kAligned ? offset_alignment : 0;
  byte_length_granularity =
      std::lcm(destination.profile.import.byte_length_granularity,
               std::lcm(source.transport.byte_length_alignment,
                        destination.transport.byte_length_alignment));
  byte_length_granularity = std::lcm(byte_length_granularity, uint64_t{64});
}

void ResidentBuffer::CreateImported(const amdf_api_t* api,
                                    amdf_memory_scope_t* scope,
                                    const ResidentImportPlan& plan,
                                    uint64_t minimum_byte_length) {
  logical.byte_offset = plan.source_byte_offset;
  logical.byte_length =
      RoundUp(minimum_byte_length, plan.byte_length_granularity);
  ASSERT_LE(logical.byte_length,
            plan.destination.profile.import.maximum_byte_length);
  amdf_memory_create_info_t create = {};
  create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create.structure_size = sizeof(create);
  create.memory_profile_ordinal = plan.source.profile.ordinal;
  create.access_count = 1;
  create.accesses = &plan.source.access;
  create.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  create.minimum_alignment = std::max(
      plan.minimum_alignment, plan.source.profile.allocation.minimum_alignment);
  ASSERT_LE(create.minimum_alignment,
            plan.source.profile.allocation.maximum_alignment);
  create.byte_length = RoundUp(
      logical.byte_offset + logical.byte_length + create.minimum_alignment,
      std::lcm(plan.source.profile.allocation.byte_length_granularity,
               plan.source.profile.allocation.native_byte_length_granularity));
  ASSERT_LE(create.byte_length,
            plan.source.profile.allocation.maximum_byte_length);
  ASSERT_NO_FATAL_FAILURE(memory.Create(api, scope, create));
  ASSERT_EQ(memory.info.source_byte_offset,
            plan.source.profile.allocation.native_byte_length_prefix);
  ASSERT_TRUE(
      amdf_physical_memory_id_is_valid(&memory.info.physical_backing_id));

  amdf_memory_export_info_t export_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
      .structure_size = sizeof(export_info),
      .external_memory_type = plan.source.transport.type,
      .byte_offset = logical.byte_offset,
      .byte_length = logical.byte_length};
  ASSERT_EQ(api->memory_export(memory.memory, &export_info, &external),
            AMDF_STATUS_OK);
  ASSERT_EQ(external.type, plan.source.transport.type);
  ASSERT_TRUE(amdf_external_memory_provenance_is_equal(
      &external.provenance, &plan.source.transport.provenance));
  const uint64_t physical_offset =
      memory.info.source_byte_offset + logical.byte_offset;
  ASSERT_EQ(external.source_byte_offset, physical_offset);
  ASSERT_EQ(external.byte_length, logical.byte_length);
  ASSERT_TRUE(amdf_physical_memory_id_is_equal(
      &external.physical_backing_id, &memory.info.physical_backing_id));

  amdf_memory_import_info_t import_info = {};
  import_info.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
  import_info.structure_size = sizeof(import_info);
  import_info.memory_profile_ordinal = plan.destination.profile.ordinal;
  import_info.access_count = 1;
  import_info.accesses = &plan.destination.access;
  import_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  import_info.minimum_alignment = plan.minimum_alignment;
  ASSERT_EQ(api->memory_import(scope, &import_info, &external, &xdna_import),
            AMDF_STATUS_OK);
  const amdf_external_memory_t empty = {};
  ASSERT_EQ(std::memcmp(&external, &empty, sizeof(empty)), 0);
  amdf_memory_info_t imported_info = {.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
                                      .structure_size = sizeof(imported_info)};
  ASSERT_EQ(api->memory_query_info(xdna_import, &imported_info),
            AMDF_STATUS_OK);
  ASSERT_EQ(imported_info.memory_profile_ordinal,
            plan.destination.profile.ordinal);
  ASSERT_EQ(imported_info.source_byte_offset, physical_offset);
  ASSERT_EQ(imported_info.byte_length, logical.byte_length);
  ASSERT_EQ(imported_info.access_count, 1u);
  ASSERT_TRUE(amdf_physical_memory_id_is_equal(
      &imported_info.physical_backing_id, &memory.info.physical_backing_id));
  amdf_memory_access_info_t access = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
      .structure_size = sizeof(access)};
  ASSERT_EQ(api->memory_query_access_info(xdna_import, 0, &access),
            AMDF_STATUS_OK);
  const auto& required = plan.destination.access.requirements;
  ASSERT_EQ(access.access, required.access);
  ASSERT_EQ(access.flags & required.flags, required.flags);
  ASSERT_EQ(access.address_kinds & required.address_kinds,
            required.address_kinds);
  ASSERT_EQ(api->memory_query_address(memory.memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                      &gpu_address),
            AMDF_STATUS_OK);
  gpu_address += logical.byte_offset;
  ASSERT_EQ(api->memory_query_address(
                xdna_import, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &npu_address),
            AMDF_STATUS_OK);
  ASSERT_EQ(npu_address % plan.minimum_alignment, 0u);
}

void ResidentBuffer::QueryImportedPairs(const amdf_api_t* api,
                                        uint32_t gpu_family,
                                        uint32_t xdna_family) {
  const auto host = memory.HostSite();
  const auto gpu = memory.DeviceSite(0, gpu_family);
  auto xdna = gpu;
  xdna.value.device.memory = xdna_import;
  xdna.value.device.queue_family_ordinal = xdna_family;
  const std::array sites = {host, gpu, xdna};
  const amdf_cache_transition_t none = {.kind =
                                            AMDF_CACHE_TRANSITION_KIND_NONE};
  const amdf_cache_transition_t release = {
      .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
      .operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM};
  const amdf_cache_transition_t acquire = {
      .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
      .operation = AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM};
  for (size_t i = 0; i < kGpuXdnaJointEdges.size(); ++i) {
    SCOPED_TRACE(i);
    const auto edge = kGpuXdnaJointEdges[i];
    auto& pair = pairs[i];
    pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair.structure_size = sizeof(pair);
    ASSERT_EQ(api->memory_query_pair_info(
                  &sites[static_cast<size_t>(edge.producer)],
                  &sites[static_cast<size_t>(edge.consumer)], &pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    const bool gpu_peer = edge.producer == GpuXdnaSite::kGpu ||
                          edge.consumer == GpuXdnaSite::kGpu;
    auto host_release = memory.host.flush;
    auto host_acquire = memory.host.invalidate;
    if (gpu_peer &&
        memory.host.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK) {
      if (host_release.executor != AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API) {
        host_release = none;
      }
      if (host_acquire.executor != AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API) {
        host_acquire = none;
      }
    }
    ASSERT_NO_FATAL_FAILURE(CheckGpuXdnaTransition(
        pair.release, edge.producer == GpuXdnaSite::kGpu    ? release
                      : edge.producer == GpuXdnaSite::kXdna ? none
                                                            : host_release));
    ASSERT_NO_FATAL_FAILURE(CheckGpuXdnaTransition(
        pair.acquire, edge.consumer == GpuXdnaSite::kGpu    ? acquire
                      : edge.consumer == GpuXdnaSite::kXdna ? none
                                                            : host_acquire));
    if (edge.producer == GpuXdnaSite::kXdna ||
        edge.consumer == GpuXdnaSite::kXdna) {
      ASSERT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
      ASSERT_EQ(pair.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
    }
  }
}

bool ResidentBuffer::Release(const amdf_api_t* api) {
  if (external.type != 0) {
    api->external_memory_release(&external);
  }
  if (xdna_import) {
    const auto status =
        api->memory_destroy(std::exchange(xdna_import, nullptr));
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
  }
  return memory.Release(api) && registration.Release(api);
}
