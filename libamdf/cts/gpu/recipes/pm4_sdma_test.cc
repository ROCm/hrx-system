// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

enum class PairQuery { kConcrete, kProfile };
enum class AcquireMode { kFullBarrier, kOrderedData };
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

// Besides the four main payload edges, preserve the CPU seed and diagnostic
// paths. Code publication additionally needs the qualified instruction-cache
// operation in the first SystemBarrier below.
constexpr std::array<Edge, 13> kEdges = {{
    {"source_host_to_sdma", kSource, Site::kHost, Site::kSdma, kUpload},
    {"input_sdma_to_pm4", kInput, Site::kSdma, Site::kPm4, kUpload},
    {"output_pm4_to_sdma", kOutput, Site::kPm4, Site::kSdma, kDownload},
    {"readback_sdma_to_host", kReadback, Site::kSdma, Site::kHost, kDownload},
    {"input_seed_host_to_pm4", kInput, Site::kHost, Site::kPm4, kUpload},
    {"output_seed_host_to_sdma", kOutput, Site::kHost, Site::kSdma, kDownload},
    {"input_sdma_to_host", kInput, Site::kSdma, Site::kHost, kUpload},
    {"output_pm4_to_host", kOutput, Site::kPm4, Site::kHost, kDownload},
    {"arguments_host_to_pm4", kArguments, Site::kHost, Site::kPm4, kUpload},
    {"code_host_to_pm4", kCode, Site::kHost, Site::kPm4, kUpload},
    {"control_host_to_sdma", kControl, Site::kHost, Site::kSdma, kUpload},
    {"control_pm4_to_sdma", kControl, Site::kPm4, Site::kSdma, kDownload},
    {"control_sdma_to_host", kControl, Site::kSdma, Site::kHost, kDownload},
}};

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

class Pm4SdmaRecipeTest : public Pm4DispatchTest {
 protected:
  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    const GpuQueueRequirements requirements = {
        .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
        .roles = AMDF_QUEUE_ROLE_TRANSFER,
        .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                             AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
    };
    amdf_queue_family_info_t sdma_family = {};
    bool matches = false;
    amdf_status_t status =
        FindQueueFamily(endpoint, requirements, &sdma_family, &matches);
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

  void SelectCreation(Backing& backing) {
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

  amdf_memory_profile_site_t ProfileSite(Site site,
                                         amdf_memory_map_flags_t host_access) {
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

  amdf_memory_site_t ConcreteSite(const GpuMemory& memory, Site site) {
    return site == Site::kHost
               ? memory.HostSite()
               : memory.DeviceSite(site == Site::kPm4 ? family_.ordinal
                                                      : sdma_family_.ordinal);
  }

  void ResolveTransition(const amdf_cache_transition_t& transition, Site site,
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
        sdma_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
        0u);
    const amdf_cache_operations_t bit = UINT64_C(1) << operation;
    ASSERT_NE(sdma_family_.cache_operations & bit, 0u);
    ASSERT_NE(sdma_family_.cache_transition_kinds &
                  AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
              0u);
    *inout_sdma_operations |= bit;
  }

  void ResolvePairs(PairQuery query_kind,
                    const std::array<Backing, kBackingCount>& backings,
                    std::array<amdf_cache_operations_t, kTransferPhaseCount>*
                        out_operations) {
    for (const Edge& edge : kEdges) {
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
      ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
      ASSERT_NO_FATAL_FAILURE(ResolveTransition(
          pair.release, edge.producer, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
          &(*out_operations)[edge.transfer_phase]));
      ASSERT_NO_FATAL_FAILURE(ResolveTransition(
          pair.acquire, edge.consumer, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
          &(*out_operations)[edge.transfer_phase]));
      const std::string prefix = std::string("pm4_sdma_") + edge.name;
      RecordProperty(prefix + "_flags", std::to_string(pair.flags));
      RecordProperty(prefix + "_release", DescribeTransition(pair.release));
      RecordProperty(prefix + "_acquire", DescribeTransition(pair.acquire));
    }
  }

  template <uint32_t kGraphCount = 1>
  void RunCoherentHandoff(
      PairQuery query_kind,
      AcquireMode acquire_mode = AcquireMode::kFullBarrier,
      amdf_cache_operations_t required_sdma_operations = 0) {
    if (required_sdma_operations != 0 &&
        (sdma_family_.format_features &
         AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) == 0) {
      GTEST_SKIP() << "explicit SDMA cache operations require USER_GCR";
    }
    ASSERT_EQ(sdma_family_.cache_operations & required_sdma_operations,
              required_sdma_operations);
    const auto* kernel_product =
        kernels::transform::kKernels.Find(gpu_endpoint_info_);
    ASSERT_NE(kernel_product, nullptr)
        << "missing compiled transform kernel for endpoint";
    const auto& kernel = *kernel_product;
    RecordProperty("transform_kernel_target", kernel.target);

    constexpr amdf_memory_access_t kReadWrite =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    constexpr uint32_t kGridSize = 1024;
    constexpr uint32_t kPayloadWordCount = 2048;
    constexpr uint32_t kPageWordCount = 1024;
    constexpr uint32_t kPayloadOffset = 16;
    constexpr uint32_t kMaximumPm4WordsPerGraph = 64;
    constexpr uint32_t kMinimumSdmaWordsPerGraph = 28;
    constexpr uint32_t kMaximumSdmaWordsPerGraph =
        kMinimumSdmaWordsPerGraph + 20;
    constexpr uint32_t kEpochCount = 2;
    constexpr uint32_t kControlGuard = 0x68d329b7u;
    constexpr std::array<uint32_t, 4> kGuards = {0x759bf13du, 0x26a4e8c3u,
                                                 0x93b57fd1u, 0x4cd218a7u};
    constexpr std::array<uint32_t, kEpochCount> kCounts = {1003, 997};
    constexpr std::array<uint32_t, kEpochCount> kAddends = {7, 0x80000023u};

    static_assert(sizeof(kernels::transform::Arguments) == 32);
    std::array<Backing, kBackingCount> backings = {{
        {"source", kGraphCount * 8192, AMDF_MEMORY_ACCESS_READ},
        {"input", kGraphCount * 8192, kReadWrite},
        {"output", kGraphCount * 8192, kReadWrite},
        {"readback", kGraphCount * 8192, kReadWrite},
        {"arguments", kGraphCount * 4096, AMDF_MEMORY_ACCESS_READ},
        {"control", kGraphCount * 4096, kReadWrite},
        {"code", 4096, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE},
    }};
    for (auto& backing : backings) {
      ASSERT_NO_FATAL_FAILURE(SelectCreation(backing));
    }
    RecordProperty("pm4_sdma_pair_query_mode",
                   query_kind == PairQuery::kProfile ? "profile" : "concrete");
    // An explicit GCR case exercises both phases even when the queried
    // minimum for coherent backing is NONE. It retains every pair answer.
    std::array<amdf_cache_operations_t, kTransferPhaseCount> sdma_operations = {
        required_sdma_operations, required_sdma_operations};
    if (query_kind == PairQuery::kProfile) {
      // Every answer precedes all seven native allocations, including code.
      ASSERT_NO_FATAL_FAILURE(
          ResolvePairs(query_kind, backings, &sdma_operations));
    }
    for (size_t i = 0; i < kCode; ++i) {
      ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, backings[i].creation,
                                           &backings[i].memory));
    }
    Pm4ComputeProgram program = {
        0,
        kernel.program.resource1,
        kernel.program.resource2,
        kernel.program.resource3,
        kernel.group_segment_byte_length,
        {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
    };
    // This unchanged helper chooses the same cold READ|EXECUTE creation inputs.
    // Check its retained descriptor against the prospective selection below.
    ASSERT_NO_FATAL_FAILURE(
        PrepareProgram(kernel.executable, kernel.entry_byte_offset, &program,
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
      ASSERT_EQ(
          memory.access_info.flags & backing.attachment.requirements.flags,
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
      RecordProperty(prefix + "_address",
                     std::to_string(memory.device_address));
    }
    if (query_kind == PairQuery::kConcrete) {
      ASSERT_NO_FATAL_FAILURE(
          ResolvePairs(query_kind, backings, &sdma_operations));
    }
    RecordProperty("pm4_sdma_upload_operations",
                   std::to_string(sdma_operations[kUpload]));
    RecordProperty("pm4_sdma_download_operations",
                   std::to_string(sdma_operations[kDownload]));
    auto& source = *backings[kSource].memory;
    auto& input = *backings[kInput].memory;
    auto& output = *backings[kOutput].memory;
    auto& readback = *backings[kReadback].memory;
    auto& arguments = *backings[kArguments].memory;
    auto& control = *backings[kControl].memory;
    auto& code = *backings[kCode].memory;
    ASSERT_EQ(arguments.device_address % kernel.arguments.alignment, 0u);
    ASSERT_EQ(control.device_address % 64, 0u);
    // Progress words use the selected coherent mapping and the ordinary
    // memory-wait, uncached SDMA FENCE32 and confirmed PM4 EOP protocols. The U
    // poll precedes the payload acquire, which cannot publish its predicate.
    // This separate packet/mapping contract is part of source admission; a
    // generic pair answer alone does not establish it. C's host observation
    // uses ReleaseSystem32's qualified completion contract. These are plain
    // DWORDs, with no HSA signal ABI and no host reset after this
    // initialization.
    std::array<uint32_t, kGraphCount * kPageWordCount> expected_control;
    expected_control.fill(kControlGuard);
    for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
      const size_t base = graph * kPageWordCount;
      expected_control[base] = expected_control[base + 16] =
          expected_control[base + 32] = 0;
    }
    std::memcpy(control.host.pointer, expected_control.data(),
                sizeof(expected_control));
    std::array<uint32_t, kPageWordCount> expected_code = {};
    std::memcpy(expected_code.data(), kernel.executable.words,
                kernel.executable.byte_length);

    GpuCommandQueue* pm4_queue = nullptr;
    GpuCommandQueue* sdma_queue = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&pm4_queue));
    ASSERT_NO_FATAL_FAILURE(CreateQueue(sdma_family_, &sdma_queue));
    ASSERT_TRUE(amdf_device_id_is_equal(&pm4_queue->device_id(),
                                        &sdma_queue->device_id()));
    ASSERT_NE(pm4_queue->native_handle(), sdma_queue->native_handle());
    ASSERT_GT(pm4_queue->words().size(),
              kEpochCount * kGraphCount * kMaximumPm4WordsPerGraph);
    ASSERT_GT(sdma_queue->words().size(),
              kEpochCount * kGraphCount * kMaximumSdmaWordsPerGraph);
    Pm4CommandWriter pm4(pm4_queue->words().data(), *pm4_profile_);
    SdmaCommandWriter sdma(sdma_queue->words().data(),
                           sdma_family_.format_features);
    std::array<size_t, kEpochCount> pm4_frontiers;
    std::array<size_t, kEpochCount> sdma_frontiers;
    for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
      for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
        const uint32_t generation = epoch * kGraphCount + graph + 1;
        const uint64_t payload_offset = graph * 8192 + 64;
        const uint64_t page_offset = graph * 4096;
        const uint64_t progress_address = control.device_address + page_offset;
        pm4.WaitMemory32(progress_address, generation);
        if (acquire_mode == AcquireMode::kFullBarrier || generation == 1) {
          // Queried GLOBAL data acquire and cold code publication.
          pm4.SystemBarrier();
        } else {
          // SDMA reaches this upload marker only after observing the previous
          // graph's shader completion and downloading its output. Between
          // epochs, the host also joins both queues before rewriting inputs.
          // Code remains immutable after its initial publication.
          pm4.AcquireFromSystem();
        }
        pm4.BindCompute(program, arguments.device_address + page_offset);
        pm4.DispatchWave32(kGridSize, 1, 1);
        pm4.ReleaseSystem32(progress_address + 64, generation);
        pm4.PadToEightWords();
        if ((sdma_operations[kUpload] &
             AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
          sdma.AcquireFromSystem();
        }
        sdma.CopyLinear(source.device_address + payload_offset,
                        input.device_address + payload_offset,
                        kGridSize * sizeof(uint32_t));
        if ((sdma_operations[kUpload] &
             AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
          sdma.ReleaseToSystem();
        }
        sdma.Fence32(progress_address, generation);
        sdma.WaitMemory32(progress_address + 64, generation);
        if ((sdma_operations[kDownload] &
             AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
          sdma.AcquireFromSystem();
        }
        sdma.CopyLinear(output.device_address + payload_offset,
                        readback.device_address + payload_offset,
                        kGridSize * sizeof(uint32_t));
        if ((sdma_operations[kDownload] &
             AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
          sdma.ReleaseToSystem();
        }
        sdma.Fence32(progress_address + 128, generation);
        ASSERT_LE(pm4.word_count(), generation * kMaximumPm4WordsPerGraph);
        ASSERT_GE(sdma.word_count(), generation * kMinimumSdmaWordsPerGraph);
        ASSERT_LE(sdma.word_count(), generation * kMaximumSdmaWordsPerGraph);
      }
      pm4_frontiers[epoch] = pm4.word_count();
      sdma_frontiers[epoch] = sdma.word_count();
    }
    // Both complete streams are resident before the first publication. PM4's
    // index uses DWORDs; SDMA's index uses bytes. Neither stream wraps.
    RecordProperty("pm4_sdma_pm4_family", family_.ordinal);
    RecordProperty("pm4_sdma_sdma_family", sdma_family_.ordinal);
    RecordProperty("pm4_sdma_sdma_format_features",
                   std::to_string(sdma_family_.format_features));
    RecordProperty("pm4_sdma_pm4_capacity_dwords",
                   std::to_string(pm4_queue->words().size()));
    RecordProperty("pm4_sdma_sdma_capacity_bytes",
                   std::to_string(sdma_queue->words().size_bytes()));
    RecordProperty("pm4_sdma_payload_byte_offset", 64);
    RecordProperty("pm4_sdma_copy_byte_length", kGridSize * sizeof(uint32_t));
    RecordProperty("pm4_sdma_grid_size", kGridSize);
    RecordProperty("pm4_sdma_workgroup_size", kernel.workgroup_size());
    RecordProperty("pm4_sdma_kernarg_byte_length",
                   kernel.arguments.byte_length);
    RecordProperty("pm4_sdma_host_cacheability",
                   AMDF_HOST_CACHEABILITY_WRITE_BACK);
    RecordProperty("pm4_sdma_control_offsets", "0,64,128");
    RecordProperty("pm4_sdma_publication_order", "pm4,sdma");

    std::array<std::array<uint32_t, kGraphCount * kPayloadWordCount>, 4>
        expected;
    std::array<std::array<uint32_t, kGraphCount * kPayloadWordCount>, 4>
        observed;
    std::array<uint32_t, kGraphCount * kPageWordCount> expected_arguments;
    std::array<uint32_t, kGraphCount * kPageWordCount> observed_arguments;
    std::array<uint32_t, kGraphCount * kPageWordCount> observed_control;
    std::array<uint32_t, kPageWordCount> observed_code;
    for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
      SCOPED_TRACE(epoch);
      for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
        const uint32_t ordinal = epoch * kGraphCount + graph;
        const size_t payload_base = graph * kPayloadWordCount;
        const size_t page_base = graph * kPageWordCount;
        for (size_t owner = 0; owner < expected.size(); ++owner) {
          std::fill_n(expected[owner].data() + payload_base, kPayloadWordCount,
                      kGuards[owner] ^ (ordinal * 0x1020304u));
        }
        for (uint32_t i = 0; i < kGridSize; ++i) {
          const size_t word = payload_base + kPayloadOffset + i;
          const uint32_t value = static_cast<uint32_t>(
              UINT64_C(0xfffffff0) + uint64_t{i} * 0x01030507u +
              uint64_t{ordinal} * 0x11111111u);
          expected[kSource][word] = value;
          expected[kInput][word] = value;
          if (i < kCounts[ordinal % kCounts.size()]) {
            expected[kOutput][word] = static_cast<uint32_t>(
                uint64_t{value} * 3 + kAddends[ordinal % kAddends.size()]);
          }
          expected[kReadback][word] = expected[kOutput][word];
        }
        const kernels::transform::Arguments payload = {
            input.device_address + graph * 8192 + 64,
            output.device_address + graph * 8192 + 64,
            kCounts[ordinal % kCounts.size()],
            kAddends[ordinal % kAddends.size()]};
        std::fill_n(expected_arguments.data() + page_base, kPageWordCount,
                    UINT32_C(0x713ace09) ^ ordinal);
        std::memcpy(expected_arguments.data() + page_base, &payload,
                    kernel.arguments.byte_length);
        expected_control[page_base] = expected_control[page_base + 16] =
            expected_control[page_base + 32] = ordinal + 1;
      }
      for (size_t owner = 0; owner < expected.size(); ++owner) {
        std::memcpy(backings[owner].memory->host.pointer,
                    expected[owner].data(), sizeof(expected[owner]));
      }
      for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
        const uint32_t ordinal = epoch * kGraphCount + graph;
        for (uint32_t i = 0; i < kGridSize; ++i) {
          const size_t word = graph * kPayloadWordCount + kPayloadOffset + i;
          static_cast<uint32_t*>(input.host.pointer)[word] =
              ~expected[kInput][word];
          if (i < kCounts[ordinal % kCounts.size()]) {
            static_cast<uint32_t*>(output.host.pointer)[word] =
                ~expected[kOutput][word];
          }
          static_cast<uint32_t*>(readback.host.pointer)[word] =
              ~expected[kReadback][word];
        }
      }
      std::memcpy(arguments.host.pointer, expected_arguments.data(),
                  sizeof(expected_arguments));
      const uint64_t pm4_frontier = pm4_frontiers[epoch];
      const uint64_t sdma_frontier = sdma_frontiers[epoch];
      ASSERT_NO_FATAL_FAILURE(pm4_queue->Publish(api_, gpu_api_, pm4_frontier));
      ASSERT_NO_FATAL_FAILURE(
          sdma_queue->Publish(api_, gpu_api_, sdma_frontier));
      const auto last_control =
          reinterpret_cast<uintptr_t>(control.host.pointer) +
          (kGraphCount - 1) * 4096;
      const uint32_t last_generation = (epoch + 1) * kGraphCount;
      GpuWaitEqual<uint32_t>(last_control + 128, last_generation);
      // One final download joins the complete batch. Snapshot every readback
      // before any independent shader join: this is the dependency observation.
      std::memcpy(observed[kReadback].data(), readback.host.pointer,
                  sizeof(observed[kReadback]));
      // Independently join the shader before reading its other owners, even if
      // the dependency under test allowed SDMA to publish D prematurely.
      GpuWaitEqual<uint32_t>(last_control + 64, last_generation);
      for (size_t owner = kSource; owner <= kOutput; ++owner) {
        std::memcpy(observed[owner].data(),
                    backings[owner].memory->host.pointer,
                    sizeof(observed[owner]));
      }
      std::memcpy(observed_arguments.data(), arguments.host.pointer,
                  sizeof(observed_arguments));
      std::memcpy(observed_control.data(), control.host.pointer,
                  sizeof(observed_control));
      std::memcpy(observed_code.data(), code.host.pointer,
                  sizeof(observed_code));

      for (size_t owner = 0; owner < expected.size(); ++owner) {
        for (size_t word = 0; word < expected[owner].size(); ++word) {
          EXPECT_EQ(observed[owner][word], expected[owner][word])
              << backings[owner].name << " word=" << word;
        }
      }
      for (size_t word = 0; word < expected_arguments.size(); ++word) {
        EXPECT_EQ(observed_arguments[word], expected_arguments[word])
            << "arguments word=" << word;
        EXPECT_EQ(observed_control[word], expected_control[word])
            << "control word=" << word;
      }
      for (size_t word = 0; word < expected_code.size(); ++word) {
        EXPECT_EQ(observed_code[word], expected_code[word])
            << "code word=" << word;
      }
      // Both retirement calls run after every nonfatal oracle, then any failure
      // stops the next CPU rewrite. Native teardown failure retains all owners.
      EXPECT_NO_FATAL_FAILURE(pm4_queue->WaitRetired(api_));
      EXPECT_NO_FATAL_FAILURE(sdma_queue->WaitRetired(api_));
      if (HasFailure()) {
        return;
      }
      const std::string prefix = "pm4_sdma_epoch_" + std::to_string(epoch);
      for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
        const uint32_t ordinal = epoch * kGraphCount + graph;
        const std::string graph_prefix =
            prefix + "_graph_" + std::to_string(graph);
        RecordProperty(graph_prefix + "_count",
                       kCounts[ordinal % kCounts.size()]);
        RecordProperty(graph_prefix + "_addend",
                       std::to_string(kAddends[ordinal % kAddends.size()]));
        RecordProperty(graph_prefix + "_completion",
                       observed_control[graph * kPageWordCount + 32]);
      }
      RecordProperty(prefix + "_pm4_frontier", std::to_string(pm4_frontier));
      RecordProperty(prefix + "_sdma_word_frontier",
                     std::to_string(sdma_frontier));
    }
    RecordProperty("pm4_sdma_completed_epochs", kEpochCount);
    RecordProperty("pm4_sdma_graphs_per_epoch", kGraphCount);
    RecordProperty("pm4_sdma_acquire_mode",
                   acquire_mode == AcquireMode::kFullBarrier ? "full_barrier"
                                                             : "ordered_data");
    RecordProperty("pm4_sdma_pm4_command_dwords",
                   std::to_string(pm4.word_count()));
    RecordProperty("pm4_sdma_sdma_command_dwords",
                   std::to_string(sdma.word_count()));
  }

  // Transfer family selected passively on the same endpoint as compute.
  amdf_queue_family_info_t sdma_family_ = {};
};

TEST_F(Pm4SdmaRecipeTest, ConcreteCoherentUploadDispatchDownload) {
  RunCoherentHandoff(PairQuery::kConcrete);
}

TEST_F(Pm4SdmaRecipeTest, ProfileCoherentUploadDispatchDownload) {
  RunCoherentHandoff(PairQuery::kProfile);
}

TEST_F(Pm4SdmaRecipeTest, BatchedCoherentUploadDispatchDownload) {
  RunCoherentHandoff<4>(PairQuery::kConcrete);
}

TEST_F(Pm4SdmaRecipeTest, OrderedDataAcquireBatchedUploadDispatchDownload) {
  RunCoherentHandoff<4>(PairQuery::kConcrete, AcquireMode::kOrderedData);
}

TEST_F(Pm4SdmaRecipeTest, UserGcrUploadDispatchDownload) {
  RunCoherentHandoff(PairQuery::kConcrete, AcquireMode::kFullBarrier,
                     AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM |
                         AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM);
}

}  // namespace
