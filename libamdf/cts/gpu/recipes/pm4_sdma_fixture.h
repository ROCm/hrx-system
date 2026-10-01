// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_RECIPES_PM4_SDMA_FIXTURE_H_
#define AMDF_CTS_GPU_RECIPES_PM4_SDMA_FIXTURE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "libamdf/cts/gpu/kernels/kernel.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

// Prepares coherent backing and directional visibility for PM4/SDMA recipes.
// The inherited case owner retains every native resource. Callers encode their
// own packet graphs and independently join payload use and queue retirement.
class Pm4SdmaTest : public Pm4DispatchTest {
 protected:
  enum class PairQuery { kConcrete, kProfile };
  enum class Site { kHost, kPm4, kSdma };
  enum TransferPhase : size_t { kUpload, kDownload, kTransferPhaseCount };
  enum BackingIndex : size_t {
    kSource,
    kInput,
    kOutput,
    kReadback,
    kArguments,
    kControl,
    kCode,
    kBackingCount,
  };

  struct Backing {
    // Receipt name for this particular owned allocation.
    const char* name;
    // Complete initialized and observed logical extent.
    uint64_t byte_length;
    // Exact permissions of the only GPU attachment.
    amdf_memory_access_t access;
    // Stable creation input, borrowed by the descriptor below.
    amdf_memory_device_access_t attachment = {};
    // Exact cold inputs shared by the profile query and native allocation.
    amdf_memory_create_info_t creation = {};
    // Borrowed from the fixture's queue-first lifetime owner.
    GpuMemory* memory = nullptr;
  };

  struct Edge {
    // Receipt name for this directional same-backing query.
    const char* name;
    // One allocation containing both sites of the edge.
    BackingIndex backing;
    // Actor publishing bytes in this allocation.
    Site producer;
    // Actor consuming those bytes after the separate ordering edge.
    Site consumer;
    // Transfer phase that supplies this edge's SDMA-side cache operation.
    // Edges without an SDMA site leave this field unused.
    TransferPhase transfer_phase;
  };

  explicit Pm4SdmaTest(amdf_queue_publication_modes_t publication_modes =
                           AMDF_QUEUE_PUBLICATION_MODE_USER |
                           AMDF_QUEUE_PUBLICATION_MODE_KERNEL)
      : Pm4DispatchTest(publication_modes),
        publication_modes_(publication_modes) {}

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override;

  // Selects exact descriptors, creates the seven native allocations, checks
  // their retained properties and queries the caller's actual visibility edges.
  // The backing array stays at a stable address through preparation. The caller
  // initializes transfer operations with any explicitly required cache work;
  // queries add the minimum operations for each phase. Packet emission and cold
  // instruction-cache publication remain the caller's responsibility.
  void PrepareCoherentHandoff(
      const kernels::Kernel& kernel, PairQuery query_kind,
      std::span<const Edge> edges, std::array<Backing, kBackingCount>& backings,
      Pm4ComputeProgram* out_program,
      std::array<amdf_cache_operations_t, kTransferPhaseCount>*
          inout_operations);

  // Transfer family selected passively on the same endpoint as compute.
  amdf_queue_family_info_t sdma_family_ = {};

 private:
  void SelectCreation(Backing& backing);
  amdf_memory_profile_site_t ProfileSite(Site site,
                                         amdf_memory_map_flags_t host_access);
  amdf_memory_site_t ConcreteSite(const GpuMemory& memory, Site site);
  void ResolveTransition(const amdf_cache_transition_t& transition, Site site,
                         amdf_cache_operation_t operation,
                         amdf_cache_operations_t* inout_sdma_operations);
  void ResolvePairs(PairQuery query_kind,
                    const std::array<Backing, kBackingCount>& backings,
                    std::span<const Edge> edges,
                    std::array<amdf_cache_operations_t, kTransferPhaseCount>*
                        inout_operations);

  // Publication modes required on both engines before native activation.
  const amdf_queue_publication_modes_t publication_modes_;
};

#endif  // AMDF_CTS_GPU_RECIPES_PM4_SDMA_FIXTURE_H_
