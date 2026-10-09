// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/interop/gpu/xdna/recipes/device_fixture.h"

#include <vector>

#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/xdna/util/execution.h"

void CheckGpuXdnaTransition(const amdf_cache_transition_t& actual,
                            const amdf_cache_transition_t& expected) {
  ASSERT_EQ(actual.kind, expected.kind);
  ASSERT_EQ(actual.executor, expected.executor);
  ASSERT_EQ(actual.operation, expected.operation);
  ASSERT_EQ(actual.host_operation, expected.host_operation);
  ASSERT_EQ(actual.host_instruction, expected.host_instruction);
  ASSERT_EQ(actual.host_fence_before, expected.host_fence_before);
  ASSERT_EQ(actual.host_fence_after, expected.host_fence_after);
  ASSERT_EQ(actual.range_granularity, expected.range_granularity);
}

GpuXdnaDeviceFixture::GpuXdnaDeviceFixture(
    amdf_queue_roles_t required_gpu_roles)
    : required_gpu_roles_(required_gpu_roles) {}

amdf_status_t GpuXdnaDeviceFixture::MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                                     bool* out_matches) {
  *out_matches = false;
  amdf_gpu_endpoint_info_t target = {
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(target)};
  auto status = gpu_api_->endpoint_query_info(endpoint, &target);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const auto* profile = Pm4CommandProfile::Find(target);
  if (!profile) {
    return AMDF_STATUS_OK;
  }
  amdf_endpoint_info_t info = {.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
                               .structure_size = sizeof(info)};
  status = api_->endpoint_query_info(endpoint, &info);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  for (uint32_t ordinal = 0; ordinal < info.queue_family_count; ++ordinal) {
    amdf_queue_family_info_t family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family)};
    status = api_->endpoint_query_queue_family_info(endpoint, ordinal, &family);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    constexpr auto kCacheOperations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                      AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
    if (family.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 ||
        family.format_version != AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 ||
        (family.roles & required_gpu_roles_) != required_gpu_roles_ ||
        (family.format_features &
         AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR) == 0 ||
        (family.cache_operations & kCacheOperations) != kCacheOperations ||
        (family.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL) ==
            0) {
      continue;
    }
    const bool user =
        (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) != 0 &&
        (family.user_queue_capabilities &
         AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) != 0 &&
        (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE) != 0 &&
        (family.priority_capabilities &
         AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL) != 0 &&
        family.maximum_ring_byte_length >= 4096;
    const bool kernel =
        (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) != 0;
    if (user || (kernel && !*out_matches)) {
      gpu_endpoint_info_ = target;
      pm4_profile_ = profile;
      gpu_family_ = family;
      publication_mode_ = user ? AMDF_QUEUE_PUBLICATION_MODE_USER
                               : AMDF_QUEUE_PUBLICATION_MODE_KERNEL;
      *out_matches = true;
      if (user) {
        break;
      }
    }
  }
  return AMDF_STATUS_OK;
}

void GpuXdnaDeviceFixture::SetUp() {
  ASSERT_NO_FATAL_FAILURE(GpuDeviceFixture::SetUp());
  if (IsSkipped()) {
    return;
  }
  const void* extension = nullptr;
  ASSERT_EQ(
      api_->query_extension(AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
                            AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension),
      AMDF_STATUS_OK);
  xdna_api_ = static_cast<const amdf_xdna_api_t*>(extension);
  uint32_t count = 0;
  ASSERT_EQ(api_->endpoint_enumerate(instance_, 0, nullptr, &count),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> summaries(count);
  ASSERT_EQ(
      api_->endpoint_enumerate(instance_, count, summaries.data(), &count),
      AMDF_STATUS_OK);
  amdf_endpoint_t* xdna_endpoint = nullptr;
  for (const auto& summary : summaries) {
    if (summary.engine_kind == AMDF_ENGINE_KIND_XDNA) {
      ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(summary.id, &xdna_endpoint),
                AMDF_STATUS_OK);
      break;
    }
  }
  ASSERT_NE(xdna_endpoint, nullptr) << "required XDNA endpoint is absent";
  xdna_endpoint_info_.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
  xdna_endpoint_info_.structure_size = sizeof(xdna_endpoint_info_);
  ASSERT_EQ(xdna_api_->endpoint_query_info(xdna_endpoint, &xdna_endpoint_info_),
            AMDF_STATUS_OK);
  ASSERT_EQ(GetCtsDeviceCache().GetXdnaDevice(xdna_endpoint, &xdna_device_),
            AMDF_STATUS_OK);
  xdna_device_info_.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
  xdna_device_info_.structure_size = sizeof(xdna_device_info_);
  ASSERT_EQ(xdna_api_->device_query_info(xdna_device_, &xdna_device_info_),
            AMDF_STATUS_OK);
  ASSERT_TRUE(FindXdnaKernelQueueFamily(api_, xdna_endpoint, &xdna_family_));
  accesses_[0] = {
      xdna_device_,
      {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
       .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
       .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA}};
  accesses_[1] = {device_,
                  {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   .flags = AMDF_MEMORY_FLAG_HOST_COHERENT |
                            AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                   .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU}};
  RecordProperty("amdf_xdna_target", xdna_endpoint_info_.target_id);
  RecordProperty("gpu_xdna_publication_mode",
                 publication_mode_ == AMDF_QUEUE_PUBLICATION_MODE_USER
                     ? "user"
                     : "kernel");
  RecordProperty("gpu_xdna_gpu_family", gpu_family_.ordinal);
  RecordProperty("gpu_xdna_xdna_family", xdna_family_);
}

void GpuXdnaDeviceFixture::FindProfile(
    std::span<const amdf_memory_device_access_t> accesses,
    amdf_memory_profile_roles_t role, amdf_memory_profile_t* result) {
  result->ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  amdf_memory_scope_info_t scope = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope)};
  ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope),
            AMDF_STATUS_OK);
  for (uint32_t ordinal = 0; ordinal < scope.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile = {.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                                     .structure_size = sizeof(profile)};
    std::array<amdf_memory_access_capabilities_t, 2> capabilities = {};
    for (auto& capability : capabilities) {
      capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      capability.structure_size = sizeof(capability);
    }
    const auto status = api_->memory_scope_query_device_profile(
        system_scope_, ordinal, accesses.size(), accesses.data(), &profile,
        capabilities.data());
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    const auto roles = role | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    if ((profile.roles & roles) == roles &&
        (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) {
      *result = profile;
      return;
    }
  }
}

amdf_memory_profile_site_t GpuXdnaDeviceFixture::ProfileSite(
    GpuXdnaSite actor, uint32_t gpu_ordinal) const {
  amdf_memory_profile_site_t site = {};
  if (actor == GpuXdnaSite::kHost) {
    site.kind = AMDF_MEMORY_SITE_KIND_HOST;
    site.value.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  } else {
    site.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    site.value.device.access_ordinal =
        actor == GpuXdnaSite::kGpu ? gpu_ordinal : 0;
    site.value.device.queue_family_ordinal =
        actor == GpuXdnaSite::kGpu ? gpu_family_.ordinal : xdna_family_;
  }
  return site;
}

amdf_memory_site_t GpuXdnaDeviceFixture::ConcreteSite(
    const CtsMappedMemory& memory, GpuXdnaSite actor,
    uint32_t gpu_ordinal) const {
  return actor == GpuXdnaSite::kHost
             ? memory.HostSite()
             : memory.DeviceSite(actor == GpuXdnaSite::kGpu ? gpu_ordinal : 0,
                                 actor == GpuXdnaSite::kGpu
                                     ? gpu_family_.ordinal
                                     : xdna_family_);
}

void GpuXdnaDeviceFixture::QueryProfilePairs(
    const amdf_memory_create_info_t& create, uint32_t gpu_ordinal,
    std::span<const GpuXdnaEdge> edges,
    std::span<amdf_memory_pair_info_t> pairs) {
  amdf_memory_profile_pair_query_t query =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
           // differs from declaration order.
  query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
  query.structure_size = sizeof(query);
  query.memory_profile_ordinal = create.memory_profile_ordinal;
  query.access_count = create.access_count;
  query.accesses = create.accesses;
  query.required_flags = create.required_flags;
  query.registered_host_cacheability = create.registered_host_cacheability;
  for (size_t i = 0; i < edges.size(); ++i) {
    SCOPED_TRACE(i);
    query.producer = ProfileSite(edges[i].producer, gpu_ordinal);
    query.consumer = ProfileSite(edges[i].consumer, gpu_ordinal);
    pairs[i].type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pairs[i].structure_size = sizeof(pairs[i]);
    ASSERT_EQ(
        api_->memory_scope_query_pair_info(system_scope_, &query, &pairs[i]),
        AMDF_STATUS_OK);
  }
}

void GpuXdnaDeviceFixture::CheckConcretePairs(
    const CtsMappedMemory& memory, uint32_t gpu_ordinal,
    std::span<const GpuXdnaEdge> edges,
    std::span<const amdf_memory_pair_info_t> expected) {
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
  for (size_t i = 0; i < edges.size(); ++i) {
    SCOPED_TRACE(i);
    const auto producer = ConcreteSite(memory, edges[i].producer, gpu_ordinal);
    const auto consumer = ConcreteSite(memory, edges[i].consumer, gpu_ordinal);
    amdf_memory_pair_info_t pair = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(pair)};
    ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, &pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair.flags, expected[i].flags);
    ASSERT_EQ(pair.atomic_reach.scope_32, expected[i].atomic_reach.scope_32);
    ASSERT_EQ(pair.atomic_reach.scope_64, expected[i].atomic_reach.scope_64);
    ASSERT_NO_FATAL_FAILURE(
        CheckGpuXdnaTransition(pair.release, expected[i].release));
    ASSERT_NO_FATAL_FAILURE(
        CheckGpuXdnaTransition(pair.acquire, expected[i].acquire));
    auto host_release = memory.host.flush;
    auto host_acquire = memory.host.invalidate;
    const bool gpu_peer = edges[i].producer == GpuXdnaSite::kGpu ||
                          edges[i].consumer == GpuXdnaSite::kGpu;
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
        pair.release, edges[i].producer == GpuXdnaSite::kGpu ? release
                      : edges[i].producer == GpuXdnaSite::kXdna
                          ? none
                          : host_release));
    ASSERT_NO_FATAL_FAILURE(CheckGpuXdnaTransition(
        pair.acquire, edges[i].consumer == GpuXdnaSite::kGpu ? acquire
                      : edges[i].consumer == GpuXdnaSite::kXdna
                          ? none
                          : host_acquire));
    if (edges[i].producer == GpuXdnaSite::kXdna ||
        edges[i].consumer == GpuXdnaSite::kXdna) {
      ASSERT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
      ASSERT_EQ(pair.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
    }
  }
}

void GpuXdnaDeviceFixture::CheckAccesses(
    const CtsMappedMemory& memory,
    std::span<const amdf_memory_device_access_t> accesses) {
  ASSERT_EQ(memory.info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  ASSERT_EQ(memory.info.access_count, accesses.size());
  ASSERT_EQ(memory.host.flags &
                (AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE),
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
  for (uint32_t ordinal = 0; ordinal < accesses.size(); ++ordinal) {
    amdf_memory_access_info_t actual = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
        .structure_size = sizeof(actual)};
    ASSERT_EQ(api_->memory_query_access_info(memory.memory, ordinal, &actual),
              AMDF_STATUS_OK);
    const auto& required = accesses[ordinal].requirements;
    ASSERT_EQ(actual.access, required.access);
    ASSERT_EQ(actual.flags & required.flags, required.flags);
    ASSERT_EQ(actual.address_kinds & required.address_kinds,
              required.address_kinds);
  }
}

amdf_status_t GpuXdnaDeviceFixture::HostTransition(
    const CtsMappedMemory& memory,
    const amdf_cache_transition_t& transition) const {
  if (transition.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    return AMDF_STATUS_OK;
  }
  return api_->host_mapping_cache_control(
      memory.mapping, transition.host_operation, 0, memory.host.byte_length);
}

void GpuXdnaDeviceFixture::CreateShaderMemory(
    amdf_memory_access_t device_access, uint64_t byte_length,
    CtsMappedMemory& memory, uint64_t& address,
    amdf_cache_transition_t& host_release) {
  auto access = accesses_[1];
  access.requirements.access = device_access;
  const std::span<const amdf_memory_device_access_t> accesses(&access, 1);
  amdf_memory_profile_t profile = {};
  ASSERT_NO_FATAL_FAILURE(
      FindProfile(accesses, AMDF_MEMORY_PROFILE_ROLE_CREATE, &profile));
  ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  const uint64_t granularity = profile.allocation.byte_length_granularity;
  ASSERT_GT(granularity, 0u);
  amdf_memory_create_info_t create =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
           // differs from declaration order.
  create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create.structure_size = sizeof(create);
  create.memory_profile_ordinal = profile.ordinal;
  create.access_count = 1;
  create.accesses = &access;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create.byte_length =
      (byte_length + granularity - 1) / granularity * granularity;
  create.minimum_alignment = profile.allocation.minimum_alignment;
  constexpr std::array<GpuXdnaEdge, 1> edges = {
      {{GpuXdnaSite::kHost, GpuXdnaSite::kGpu}}};
  std::array<amdf_memory_pair_info_t, 1> pairs = {};
  ASSERT_NO_FATAL_FAILURE(QueryProfilePairs(create, 0, edges, pairs));
  ASSERT_NO_FATAL_FAILURE(memory.Create(api_, system_scope_, create));
  ASSERT_NO_FATAL_FAILURE(CheckAccesses(memory, accesses));
  ASSERT_NO_FATAL_FAILURE(CheckConcretePairs(memory, 0, edges, pairs));
  ASSERT_EQ(api_->memory_query_address(memory.memory, 0,
                                       AMDF_MEMORY_ADDRESS_GPU, &address),
            AMDF_STATUS_OK);
  ASSERT_NE(address, 0u);
  host_release = pairs[0].release;
}
