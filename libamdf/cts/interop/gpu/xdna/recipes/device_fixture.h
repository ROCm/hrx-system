// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_INTEROP_GPU_XDNA_RECIPES_DEVICE_FIXTURE_H_
#define AMDF_CTS_INTEROP_GPU_XDNA_RECIPES_DEVICE_FIXTURE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "amdf/xdna.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/profile.h"
#include "util/mapped_memory.h"

// Checks every semantic cache-transition field without comparing ABI padding.
void CheckGpuXdnaTransition(const amdf_cache_transition_t& actual,
                            const amdf_cache_transition_t& expected);

enum class GpuXdnaSite { kHost, kGpu, kXdna };

struct GpuXdnaEdge {
  // Actor publishing the backing before the explicit ordering edge.
  GpuXdnaSite producer;
  // Actor acquiring the same backing after that ordering edge.
  GpuXdnaSite consumer;
};

enum GpuXdnaJointEdge : size_t {
  kGpuXdnaHostToGpu,
  kGpuXdnaHostToXdna,
  kGpuXdnaGpuToXdna,
  kGpuXdnaXdnaToGpu,
  kGpuXdnaGpuToHost,
  kGpuXdnaXdnaToHost,
};

inline constexpr std::array<GpuXdnaEdge, 6> kGpuXdnaJointEdges = {{
    {GpuXdnaSite::kHost, GpuXdnaSite::kGpu},
    {GpuXdnaSite::kHost, GpuXdnaSite::kXdna},
    {GpuXdnaSite::kGpu, GpuXdnaSite::kXdna},
    {GpuXdnaSite::kXdna, GpuXdnaSite::kGpu},
    {GpuXdnaSite::kGpu, GpuXdnaSite::kHost},
    {GpuXdnaSite::kXdna, GpuXdnaSite::kHost},
}};
inline constexpr std::array<GpuXdnaEdge, 2> kGpuXdnaStagingEdges = {{
    {GpuXdnaSite::kHost, GpuXdnaSite::kGpu},
    {GpuXdnaSite::kGpu, GpuXdnaSite::kHost},
}};

// Borrows one cached device per endpoint and admits the native queue and memory
// contracts shared by GPU/XDNA recipes. Workload images, allocations, queues,
// contexts, publication and joined cleanup remain owned by the derived case.
class GpuXdnaDeviceFixture : public GpuDeviceFixture {
 protected:
  explicit GpuXdnaDeviceFixture(amdf_queue_roles_t required_gpu_roles);

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override;
  void SetUp() override;

  // Finds a host-visible profile for one GPU access or the two joint accesses.
  // A missing advertised construction role leaves ordinal UNKNOWN; the caller
  // decides whether that optional service is required by its selected case.
  void FindProfile(std::span<const amdf_memory_device_access_t> accesses,
                   amdf_memory_profile_roles_t role,
                   amdf_memory_profile_t* result);
  // Joint requests place XDNA access first; gpu_ordinal identifies GPU access
  // in either a joint request or a GPU-only request.
  amdf_memory_profile_site_t ProfileSite(GpuXdnaSite actor,
                                         uint32_t gpu_ordinal) const;
  amdf_memory_site_t ConcreteSite(const CtsMappedMemory& memory,
                                  GpuXdnaSite actor,
                                  uint32_t gpu_ordinal) const;
  // The output and expected pair spans correspond one-to-one with edges.
  void QueryProfilePairs(const amdf_memory_create_info_t& create,
                         uint32_t gpu_ordinal,
                         std::span<const GpuXdnaEdge> edges,
                         std::span<amdf_memory_pair_info_t> pairs);
  void CheckConcretePairs(const CtsMappedMemory& memory, uint32_t gpu_ordinal,
                          std::span<const GpuXdnaEdge> edges,
                          std::span<const amdf_memory_pair_info_t> expected);
  void CheckAccesses(const CtsMappedMemory& memory,
                     std::span<const amdf_memory_device_access_t> accesses);
  // Applies the named host transition to the full mapped allocation.
  amdf_status_t HostTransition(const CtsMappedMemory& memory,
                               const amdf_cache_transition_t& transition) const;
  // Creates caller-owned GPU-only storage and checks both prospective and
  // concrete HOST-to-GPU publication recipes without publishing any contents.
  void CreateShaderMemory(amdf_memory_access_t device_access,
                          uint64_t byte_length, CtsMappedMemory& memory,
                          uint64_t& address,
                          amdf_cache_transition_t& host_release,
                          amdf_memory_flags_t flags = 0);

  // Exact native GPU facts retained by passive endpoint matching.
  amdf_gpu_endpoint_info_t gpu_endpoint_info_ = {};
  // Native PM4 family chosen before cached GPU activation.
  amdf_queue_family_info_t gpu_family_ = {};
  // Static compute encoding selected before native GPU activation.
  const Pm4CommandProfile* pm4_profile_ = nullptr;
  // One admitted transport, never switched after a native failure.
  amdf_queue_publication_modes_t publication_mode_ = 0;
  // XDNA API table borrowed from the same provider instance.
  const amdf_xdna_api_t* xdna_api_ = nullptr;
  // Cached XDNA device; each case owns its execution context and children.
  amdf_device_t* xdna_device_ = nullptr;
  // Exact XDNA target facts used by the derived case's program admission.
  amdf_xdna_endpoint_info_t xdna_endpoint_info_ = {};
  // Achieved native XDNA device facts used by program admission.
  amdf_xdna_device_info_t xdna_device_info_ = {};
  // Exact XDNA family used by native submissions and pair queries.
  uint32_t xdna_family_ = UINT32_MAX;
  // Complete joint access order: XDNA DMA, then coherent GPU addresses.
  std::array<amdf_memory_device_access_t, 2> accesses_ = {};

 private:
  // Queue roles required by the derived case before endpoint activation.
  const amdf_queue_roles_t required_gpu_roles_;
};

#endif  // AMDF_CTS_INTEROP_GPU_XDNA_RECIPES_DEVICE_FIXTURE_H_
