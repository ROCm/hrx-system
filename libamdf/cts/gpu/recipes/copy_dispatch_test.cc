// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

enum class PairQuery { kConcrete, kProfile };

struct PayloadPairs {
  // Host-written source consumed by SDMA upload.
  amdf_memory_pair_info_t ingress = {};
  // SDMA-written input consumed by the compiled dispatch.
  amdf_memory_pair_info_t upload = {};
  // Dispatch-written output consumed by SDMA download.
  amdf_memory_pair_info_t download = {};
  // SDMA-written readback consumed by the host.
  amdf_memory_pair_info_t egress = {};
};

struct PayloadTransitions {
  // AQL packet scopes for the upload-to-dispatch and dispatch-to-download
  // edges.
  aql::FenceScopes dispatch = {aql::FenceScope::kNone, aql::FenceScope::kNone};
  // SDMA acquire before upload and release before its completion signal.
  amdf_cache_operations_t upload = 0;
  // SDMA acquire after the dispatch wait and release before host completion.
  amdf_cache_operations_t download = 0;
};

struct PayloadSet {
  // Logical length of each of the four owned allocations.
  uint64_t byte_length;
  // Start of the copied window, leaving a distinct prefix and suffix guard.
  uint64_t payload_byte_offset;
  // Host-written upload source, borrowed from the case memory owner.
  GpuMemory* source = nullptr;
  // Upload destination and compiled-kernel input.
  GpuMemory* input = nullptr;
  // Compiled-kernel output and download source.
  GpuMemory* output = nullptr;
  // Download destination observed before any other payload read.
  GpuMemory* readback = nullptr;
  // Queue operations resolved from this set's actual queried transitions.
  PayloadTransitions transitions;
};

struct StagedPayloadPairs {
  // Source ingress, input upload, output download and output readback edges.
  PayloadPairs payload;
  // Host-written output seed consumed by the second SDMA upload.
  amdf_memory_pair_info_t seed_ingress = {};
  // SDMA-written output seed preserved outside the compiled kernel's stores.
  amdf_memory_pair_info_t seed_upload = {};
  // Uploaded input downloaded unchanged by the same SDMA queue.
  amdf_memory_pair_info_t input_download = {};
  // Diagnostic input readback consumed by the host.
  amdf_memory_pair_info_t input_egress = {};
};

struct StagedPayloadSet {
  // Source, LOCAL input/output and SYSTEM output readback for the main path.
  PayloadSet payload;
  // Coherent SYSTEM seed copied to the complete LOCAL output backing.
  GpuMemory* seed = nullptr;
  // Coherent SYSTEM diagnostic download of the complete unchanged input.
  GpuMemory* input_readback = nullptr;
};

std::string DescribeTransition(const amdf_cache_transition_t& transition) {
  return "kind=" + std::to_string(transition.kind) +
         ",executor=" + std::to_string(transition.executor) +
         ",operation=" + std::to_string(transition.operation);
}

void CheckTransition(const amdf_cache_transition_t& transition,
                     amdf_cache_transition_kind_t kind,
                     amdf_cache_transition_executor_t executor,
                     amdf_cache_operation_t operation) {
  ASSERT_EQ(transition.kind, kind);
  ASSERT_EQ(transition.executor, executor);
  ASSERT_EQ(transition.operation, operation);
  ASSERT_EQ(transition.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
  ASSERT_EQ(transition.host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
  ASSERT_EQ(transition.host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.range_granularity, 0u);
}

void CheckNoCacheTransition(const amdf_cache_transition_t& transition) {
  ASSERT_NO_FATAL_FAILURE(CheckTransition(
      transition, AMDF_CACHE_TRANSITION_KIND_NONE,
      AMDF_CACHE_TRANSITION_EXECUTOR_NONE, AMDF_CACHE_OPERATION_NONE));
}

void ResolveDispatchScope(const amdf_cache_transition_t& transition,
                          amdf_cache_operation_t operation,
                          aql::FenceScope* out_scope) {
  ASSERT_NO_FATAL_FAILURE(
      CheckTransition(transition, AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                      AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE, operation));
  // AQL v1 realizes this exact global queue operation with SYSTEM scope.
  *out_scope = aql::FenceScope::kSystem;
}

class CopyDispatchRecipeTest : public AqlDispatchTest {
 protected:
  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    const GpuQueueRequirements requirements = {
        .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
        .roles = AMDF_QUEUE_ROLE_TRANSFER,
    };
    amdf_queue_family_info_t sdma_family = {};
    bool matches = false;
    amdf_status_t status = FindGpuQueueFamily(api_, endpoint, requirements,
                                              &sdma_family, &matches);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (!matches) {
      *out_matches = false;
      return AMDF_STATUS_OK;
    }
    status = AqlDispatchTest::MatchGpuEndpoint(endpoint, &matches);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (matches) {
      sdma_family_ = sdma_family;
    }
    *out_matches = matches;
    return AMDF_STATUS_OK;
  }

  void SelectCreation(const amdf_memory_device_access_t& attachment,
                      amdf_memory_create_info_t* out_creation) {
    const uint32_t profile = FindGpuMemoryProfileOrdinal(
        api_, system_scope_, attachment.device,
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        AMDF_MEMORY_FLAG_HOST_VISIBLE, attachment.requirements);
    ASSERT_NE(profile, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    *out_creation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(*out_creation),
        .memory_profile_ordinal = profile,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .minimum_alignment = 4096,
        .accesses = &attachment,
    };
  }

  void SelectLocalCreation(const amdf_memory_device_access_t& attachment,
                           amdf_memory_create_info_t* out_creation) {
    const uint32_t profile = FindGpuMemoryProfileOrdinal(
        api_, local_scope_, attachment.device, AMDF_MEMORY_PROFILE_ROLE_CREATE,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL, attachment.requirements);
    ASSERT_NE(profile, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    *out_creation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(*out_creation),
        .memory_profile_ordinal = profile,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL,
        .minimum_alignment = 4096,
        .accesses = &attachment,
    };
  }

  void QueryProfilePair(amdf_memory_scope_t* scope,
                        const amdf_memory_create_info_t& creation,
                        const amdf_memory_profile_site_t& producer,
                        const amdf_memory_profile_site_t& consumer,
                        amdf_memory_pair_info_t* out_pair) {
    amdf_memory_profile_pair_query_t query = {};
    query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
    query.structure_size = sizeof(query);
    query.memory_profile_ordinal = creation.memory_profile_ordinal;
    query.required_flags = creation.required_flags;
    query.access_count = creation.access_count;
    query.accesses = creation.accesses;
    query.registered_host_cacheability = creation.registered_host_cacheability;
    query.producer = producer;
    query.consumer = consumer;
    out_pair->type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    out_pair->structure_size = sizeof(*out_pair);
    ASSERT_EQ(api_->memory_scope_query_pair_info(scope, &query, out_pair),
              AMDF_STATUS_OK);
    ASSERT_NE(out_pair->flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
              0u);
  }

  void QueryProfilePairs(amdf_memory_scope_t* payload_scope,
                         const amdf_memory_create_info_t& source_creation,
                         const amdf_memory_create_info_t& payload_creation,
                         const amdf_memory_create_info_t& readback_creation,
                         PayloadPairs* out_pairs) {
    const amdf_memory_profile_site_t host_write = {
        .kind = AMDF_MEMORY_SITE_KIND_HOST,
        .value = {.host_access = AMDF_MEMORY_MAP_FLAG_WRITE},
    };
    const amdf_memory_profile_site_t host_read = {
        .kind = AMDF_MEMORY_SITE_KIND_HOST,
        .value = {.host_access = AMDF_MEMORY_MAP_FLAG_READ},
    };
    const amdf_memory_profile_site_t sdma = {
        .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
        .value = {.device = {0, sdma_family_.ordinal}},
    };
    const amdf_memory_profile_site_t aql = {
        .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
        .value = {.device = {0, family_.ordinal}},
    };
    ASSERT_NO_FATAL_FAILURE(QueryProfilePair(
        system_scope_, source_creation, host_write, sdma, &out_pairs->ingress));
    ASSERT_NO_FATAL_FAILURE(QueryProfilePair(payload_scope, payload_creation,
                                             sdma, aql, &out_pairs->upload));
    ASSERT_NO_FATAL_FAILURE(QueryProfilePair(payload_scope, payload_creation,
                                             aql, sdma, &out_pairs->download));
    ASSERT_NO_FATAL_FAILURE(QueryProfilePair(
        system_scope_, readback_creation, sdma, host_read, &out_pairs->egress));
  }

  void QueryConcretePair(const amdf_memory_site_t& producer,
                         const amdf_memory_site_t& consumer,
                         amdf_memory_pair_info_t* out_pair) {
    out_pair->type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    out_pair->structure_size = sizeof(*out_pair);
    ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, out_pair),
              AMDF_STATUS_OK);
    ASSERT_NE(out_pair->flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
              0u);
  }

  void QueryConcretePairs(const PayloadSet& payload, PayloadPairs* out_pairs) {
    ASSERT_NO_FATAL_FAILURE(QueryConcretePair(
        payload.source->HostSite(),
        payload.source->DeviceSite(sdma_family_.ordinal), &out_pairs->ingress));
    ASSERT_NO_FATAL_FAILURE(QueryConcretePair(
        payload.input->DeviceSite(sdma_family_.ordinal),
        payload.input->DeviceSite(family_.ordinal), &out_pairs->upload));
    ASSERT_NO_FATAL_FAILURE(
        QueryConcretePair(payload.output->DeviceSite(family_.ordinal),
                          payload.output->DeviceSite(sdma_family_.ordinal),
                          &out_pairs->download));
    ASSERT_NO_FATAL_FAILURE(
        QueryConcretePair(payload.readback->DeviceSite(sdma_family_.ordinal),
                          payload.readback->HostSite(), &out_pairs->egress));
  }

  void RecordPair(const std::string& name,
                  const amdf_memory_pair_info_t& pair) {
    RecordProperty(name + "_flags", std::to_string(pair.flags));
    RecordProperty(name + "_release", DescribeTransition(pair.release));
    RecordProperty(name + "_acquire", DescribeTransition(pair.acquire));
  }

  void AccumulateSdmaTransition(const amdf_cache_transition_t& transition,
                                amdf_cache_operation_t operation,
                                amdf_cache_operations_t* inout_operations) {
    if (transition.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
      ASSERT_NO_FATAL_FAILURE(CheckNoCacheTransition(transition));
      return;
    }
    ASSERT_NO_FATAL_FAILURE(
        CheckTransition(transition, AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                        AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE, operation));
    ASSERT_NE(
        sdma_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
        0u);
    const amdf_cache_operations_t bit = UINT64_C(1) << operation;
    ASSERT_NE(sdma_family_.cache_operations & bit, 0u);
    ASSERT_NE(sdma_family_.cache_transition_kinds &
                  AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
              0u);
    *inout_operations |= bit;
  }

  void ResolvePairs(const PayloadPairs& pairs, const std::string& prefix,
                    PayloadTransitions* out_transitions) {
    RecordPair(prefix + "_host_to_sdma", pairs.ingress);
    RecordPair(prefix + "_sdma_to_aql", pairs.upload);
    RecordPair(prefix + "_aql_to_sdma", pairs.download);
    RecordPair(prefix + "_sdma_to_host", pairs.egress);
    ASSERT_NO_FATAL_FAILURE(CheckNoCacheTransition(pairs.ingress.release));
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.ingress.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        &out_transitions->upload));
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.upload.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
        &out_transitions->upload));
    ASSERT_NO_FATAL_FAILURE(ResolveDispatchScope(
        pairs.upload.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        &out_transitions->dispatch.acquire));
    ASSERT_NO_FATAL_FAILURE(ResolveDispatchScope(
        pairs.download.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
        &out_transitions->dispatch.release));
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.download.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        &out_transitions->download));
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.egress.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
        &out_transitions->download));
    ASSERT_NO_FATAL_FAILURE(CheckNoCacheTransition(pairs.egress.acquire));
  }

  void ResolveStagedPairs(const StagedPayloadPairs& pairs,
                          const std::string& prefix,
                          PayloadTransitions* out_transitions) {
    ASSERT_NO_FATAL_FAILURE(
        ResolvePairs(pairs.payload, prefix, out_transitions));
    RecordPair(prefix + "_seed_host_to_sdma", pairs.seed_ingress);
    RecordPair(prefix + "_seed_sdma_to_aql", pairs.seed_upload);
    RecordPair(prefix + "_input_sdma_to_sdma", pairs.input_download);
    RecordPair(prefix + "_input_sdma_to_host", pairs.input_egress);
    ASSERT_NO_FATAL_FAILURE(CheckNoCacheTransition(pairs.seed_ingress.release));
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.seed_ingress.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        &out_transitions->upload));
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.seed_upload.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
        &out_transitions->upload));
    aql::FenceScope seed_acquire = aql::FenceScope::kNone;
    ASSERT_NO_FATAL_FAILURE(ResolveDispatchScope(
        pairs.seed_upload.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        &seed_acquire));
    // One global dispatch acquire satisfies both LOCAL input and output-seed
    // transitions. Each allocation's returned operation must agree.
    ASSERT_EQ(seed_acquire, out_transitions->dispatch.acquire);
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.input_download.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
        &out_transitions->upload));
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.input_download.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        &out_transitions->download));
    ASSERT_NO_FATAL_FAILURE(AccumulateSdmaTransition(
        pairs.input_egress.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
        &out_transitions->download));
    ASSERT_NO_FATAL_FAILURE(CheckNoCacheTransition(pairs.input_egress.acquire));
  }

  void RunCoherentHandoff(PairQuery query_kind,
                          amdf_cache_operations_t required_sdma_operations = 0);
  void RunStagedHandoff(PairQuery query_kind);

  // Transfer family retained from the same passive endpoint match as AQL.
  amdf_queue_family_info_t sdma_family_ = {};
};

struct alignas(64) CompletionState {
  // Written 1 to 0 by SDMA; AQL depends on the complete native signal block.
  aql::Signal upload;
  // Decremented 1 to 0 by CP; SDMA polls the low word of its value.
  aql::Signal compute;
  // Written only by SDMA after download; one nonzero value per complete epoch.
  alignas(64) uint32_t download;
};

constexpr uint32_t kPageByteLength = 4096;
// Bytes outside the two native signal blocks and the final download word.
constexpr uint32_t kControlGuardByteOffset =
    offsetof(CompletionState, download) + sizeof(uint32_t);

void CopyDispatchRecipeTest::RunCoherentHandoff(
    PairQuery query_kind, amdf_cache_operations_t required_sdma_operations) {
  if (required_sdma_operations != 0 &&
      (sdma_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) ==
          0) {
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

  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kMaximumByteLength = 12288;
  constexpr uint32_t kMaximumWordCount = kMaximumByteLength / sizeof(uint32_t);
  constexpr uint32_t kPayloadByteLength = kGridSize * sizeof(uint32_t);
  constexpr uint32_t kSourceGuard = 0x759bf13du;
  constexpr uint32_t kInputGuard = 0x26a4e8c3u;
  constexpr uint32_t kOutputGuard = 0x93b57fd1u;
  constexpr uint32_t kReadbackGuard = 0x4cd218a7u;
  constexpr size_t kMinimumChainWordCount = 7 + 4 + 6 + 7 + 4;
  constexpr size_t kMaximumChainWordCount = kMinimumChainWordCount + 20;
  constexpr uint64_t kMinimumChainByteLength =
      kMinimumChainWordCount * sizeof(uint32_t);
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  const amdf_memory_device_access_t read_access = {
      device_,
      {.access = AMDF_MEMORY_ACCESS_READ,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  const amdf_memory_device_access_t read_write_access = {
      device_,
      {.access = kReadWrite,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  amdf_memory_create_info_t read_creation = {};
  amdf_memory_create_info_t read_write_creation = {};
  ASSERT_NO_FATAL_FAILURE(SelectCreation(read_access, &read_creation));
  ASSERT_NO_FATAL_FAILURE(
      SelectCreation(read_write_access, &read_write_creation));
  RecordProperty("pair_query_mode",
                 query_kind == PairQuery::kProfile ? "profile" : "concrete");
  RecordProperty("source_memory_profile_ordinal",
                 read_creation.memory_profile_ordinal);
  RecordProperty("read_write_memory_profile_ordinal",
                 read_write_creation.memory_profile_ordinal);
  RecordProperty("payload_required_flags",
                 std::to_string(read_creation.required_flags));
  RecordProperty("payload_access_flags",
                 std::to_string(read_access.requirements.flags));
  RecordProperty("source_device_access", read_access.requirements.access);
  RecordProperty("read_write_device_access",
                 read_write_access.requirements.access);
  RecordProperty("payload_minimum_alignment",
                 std::to_string(read_creation.minimum_alignment));

  PayloadTransitions profile_transitions;
  if (query_kind == PairQuery::kProfile) {
    // Both sets will use these answers, obtained before any payload backing.
    PayloadPairs pairs;
    ASSERT_NO_FATAL_FAILURE(QueryProfilePairs(system_scope_, read_creation,
                                              read_write_creation,
                                              read_write_creation, &pairs));
    ASSERT_NO_FATAL_FAILURE(
        ResolvePairs(pairs, "profile", &profile_transitions));
  }
  std::array<PayloadSet, 2> payload_sets = {{
      {.byte_length = 8192, .payload_byte_offset = 64},
      {.byte_length = kMaximumByteLength, .payload_byte_offset = 128},
  }};
  for (size_t i = 0; i < payload_sets.size(); ++i) {
    auto& payload = payload_sets[i];
    auto source_creation = read_creation;
    source_creation.byte_length = payload.byte_length;
    auto target_creation = read_write_creation;
    target_creation.byte_length = payload.byte_length;
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(system_scope_, source_creation, &payload.source));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(system_scope_, target_creation, &payload.input));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(system_scope_, target_creation, &payload.output));
    // KFD mappings always grant GPU read access, including download targets.
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(system_scope_, target_creation, &payload.readback));
    const std::string prefix = "payload_set_" + std::to_string(i);
    RecordProperty(prefix + "_byte_length",
                   std::to_string(payload.byte_length));
    RecordProperty(prefix + "_byte_offset",
                   std::to_string(payload.payload_byte_offset));
    RecordProperty(prefix + "_source_address",
                   std::to_string(payload.source->device_address));
    RecordProperty(prefix + "_input_address",
                   std::to_string(payload.input->device_address));
    RecordProperty(prefix + "_output_address",
                   std::to_string(payload.output->device_address));
    RecordProperty(prefix + "_readback_address",
                   std::to_string(payload.readback->device_address));
    if (query_kind == PairQuery::kConcrete) {
      PayloadPairs pairs;
      ASSERT_NO_FATAL_FAILURE(QueryConcretePairs(payload, &pairs));
      ASSERT_NO_FATAL_FAILURE(
          ResolvePairs(pairs, prefix, &payload.transitions));
    } else {
      payload.transitions = profile_transitions;
    }
    // Explicit-GCR coverage requests the stronger stream recipe even when
    // this backing's queried minimum is NONE. The pair answers stay intact.
    payload.transitions.upload |= required_sdma_operations;
    payload.transitions.download |= required_sdma_operations;
    RecordProperty(prefix + "_sdma_upload_operations",
                   std::to_string(payload.transitions.upload));
    RecordProperty(prefix + "_sdma_download_operations",
                   std::to_string(payload.transitions.download));
    RecordProperty(prefix + "_dispatch_acquire_scope",
                   static_cast<uint32_t>(payload.transitions.dispatch.acquire));
    RecordProperty(prefix + "_dispatch_release_scope",
                   static_cast<uint32_t>(payload.transitions.dispatch.release));
  }
  GpuMemory* arguments = nullptr;
  GpuMemory* control = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, kPageByteLength, &control));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_EQ(control->device_address % alignof(CompletionState), 0u);
  std::memset(control->host.pointer, 0, control->info.byte_length);
  auto& completion = *static_cast<CompletionState*>(control->host.pointer);
  completion.upload.kind = completion.compute.kind = 1;
  const uint64_t upload_signal_address =
      control->device_address + offsetof(CompletionState, upload);
  const uint64_t compute_signal_address =
      control->device_address + offsetof(CompletionState, compute);
  const uint64_t download_address =
      control->device_address + offsetof(CompletionState, download);
  RecordProperty("copy_dispatch_argument_byte_length", kPageByteLength);
  RecordProperty("copy_dispatch_control_guard_byte_offset",
                 kControlGuardByteOffset);
  RecordProperty("copy_dispatch_control_guard_byte_length",
                 kPageByteLength - kControlGuardByteOffset);

  GpuUserQueue* aql_queue = nullptr;
  GpuUserQueue* sdma_queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&aql_queue));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(sdma_family_, &sdma_queue));
  ASSERT_TRUE(amdf_device_id_is_equal(&aql_queue->info.device_id,
                                      &sdma_queue->info.device_id));
  ASSERT_FALSE(amdf_queue_id_is_equal(&aql_queue->info.queue_id,
                                      &sdma_queue->info.queue_id));
  const uint64_t aql_capacity =
      aql_queue->host.ring_byte_length / sizeof(aql::Packet);
  const uint64_t sdma_capacity = sdma_queue->host.ring_byte_length;
  ASSERT_GE(aql_capacity, 2u);
  ASSERT_GT(sdma_capacity, 2 * kMaximumChainWordCount * sizeof(uint32_t));
  const uint64_t epoch_count =
      2 * std::max((aql_capacity + 1) / 2,
                   sdma_capacity / kMinimumChainByteLength) +
      1;
  ASSERT_LE(epoch_count, UINT32_MAX);
  RecordProperty("copy_dispatch_epoch_count", std::to_string(epoch_count));
  RecordProperty("aql_queue_family_ordinal", family_.ordinal);
  RecordProperty("sdma_queue_family_ordinal", sdma_family_.ordinal);
  RecordProperty("aql_ring_packet_capacity", std::to_string(aql_capacity));
  RecordProperty("sdma_ring_byte_capacity", std::to_string(sdma_capacity));
  uint64_t aql_index = 0;
  uint64_t sdma_index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*aql_queue, kernel, "aql_kernel",
                                        &aql_index, &descriptor_address));

  auto* sdma_ring = reinterpret_cast<uint8_t*>(sdma_queue->host.ring_address);
  std::array<uint32_t, kMaximumWordCount> expected_source;
  std::array<uint32_t, kMaximumWordCount> expected_input;
  std::array<uint32_t, kMaximumWordCount> expected_output;
  std::array<uint32_t, kMaximumWordCount> expected_readback;
  std::array<uint32_t, kMaximumWordCount> downloaded;
  std::array<uint32_t, kMaximumWordCount> observed_source;
  std::array<uint32_t, kMaximumWordCount> observed_input;
  std::array<uint32_t, kMaximumWordCount> observed_output;
  std::array<uint8_t, kPageByteLength> expected_arguments;
  std::array<uint8_t, kPageByteLength> observed_arguments;
  CompletionState observed_completion;
  std::array<uint8_t, kPageByteLength - kControlGuardByteOffset>
      observed_control_guards;
  std::array<uint64_t, 2> completed_epochs = {};
  for (uint64_t epoch_index = 0; epoch_index < epoch_count; ++epoch_index) {
    // Each set sees both count variants before the next set uses the queues.
    const size_t set_index = (epoch_index / 2) % payload_sets.size();
    const auto& buffers = payload_sets[set_index];
    const uint32_t word_count =
        static_cast<uint32_t>(buffers.byte_length / sizeof(uint32_t));
    const uint32_t payload_word_offset =
        static_cast<uint32_t>(buffers.payload_byte_offset / sizeof(uint32_t));
    auto* source_words = static_cast<uint32_t*>(buffers.source->host.pointer);
    auto* input_words = static_cast<uint32_t*>(buffers.input->host.pointer);
    auto* output_words = static_cast<uint32_t*>(buffers.output->host.pointer);
    auto* readback_words =
        static_cast<uint32_t*>(buffers.readback->host.pointer);
    const uint32_t epoch = static_cast<uint32_t>(epoch_index + 1);
    const uint32_t count = epoch % 2 == 1 ? 1003 : 997;
    const uint32_t addend = 0x80000001u + 2 * epoch;
    expected_source.fill(kSourceGuard);
    expected_input.fill(kInputGuard);
    expected_output.fill(kOutputGuard);
    expected_readback.fill(kReadbackGuard);
    std::fill_n(source_words, word_count, kSourceGuard);
    std::fill_n(input_words, word_count, kInputGuard);
    std::fill_n(output_words, word_count, kOutputGuard);
    std::fill_n(readback_words, word_count, kReadbackGuard);
    for (uint32_t i = 0; i < kGridSize; ++i) {
      const uint32_t position = payload_word_offset + i;
      const uint32_t value = 0xfffffff0u + i * 0x01030507u + epoch;
      source_words[position] = value;
      input_words[position] = ~value;
      expected_source[position] = expected_input[position] = value;
      if (i < count) {
        // The host oracle is independent of GPU data. Common active lanes
        // change by five per epoch modulo 2^32, including overflowing values.
        const uint32_t result =
            static_cast<uint32_t>(uint64_t{value} * 3 + addend);
        expected_output[position] = result;
        output_words[position] = result ^ 0x5a5a5a5au;
      }
      expected_readback[position] = expected_output[position];
      // Inactive lanes retain CPU-written guards under the coherent SDMA
      // policy. Poison readback too so the download must actually copy them.
      readback_words[position] = expected_readback[position] ^ 0xa5a5a5a5u;
    }
    const kernels::transform::Arguments payload = {
        buffers.input->device_address + buffers.payload_byte_offset,
        buffers.output->device_address + buffers.payload_byte_offset,
        count,
        addend,
    };
    // Kernargs stay immutable through completion; normal AQL publication and
    // the queried dispatch acquire provide their coherent native contract.
    expected_arguments.fill(0);
    std::memcpy(expected_arguments.data(), &payload,
                kernel.arguments.byte_length);
    std::memcpy(arguments->host.pointer, expected_arguments.data(),
                expected_arguments.size());
    // Previous final download and both consumed frontiers precede rearming.
    // Only the low word changes on the SDMA upload's finite 1-to-0 transition.
    completion.upload.value = completion.compute.value = 1;
    // Native signals supply ordering independently of payload pair queries.
    // AND adds no global maintenance before the queried dispatch operation.
    const auto dependency =
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
                     {upload_signal_address},
                     {aql::FenceScope::kNone, aql::FenceScope::kNone});
    const auto dispatch = aql::Dispatch(
        aql::HeaderBarrier::kDisabled,
        {1,
         {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
         {kGridSize, 1, 1}},
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        descriptor_address, arguments->device_address, compute_signal_address,
        buffers.transitions.dispatch);
    std::array<uint32_t, kMaximumChainWordCount> stream = {};
    SdmaCommandWriter commands(stream.data(), sdma_family_.format_features);
    if ((buffers.transitions.upload &
         AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(
        buffers.source->device_address + buffers.payload_byte_offset,
        buffers.input->device_address + buffers.payload_byte_offset,
        kPayloadByteLength);
    if ((buffers.transitions.upload &
         AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(upload_signal_address + offsetof(aql::Signal, value), 0);
    commands.WaitMemory32(compute_signal_address + offsetof(aql::Signal, value),
                          0);
    if ((buffers.transitions.download &
         AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(
        buffers.output->device_address + buffers.payload_byte_offset,
        buffers.readback->device_address + buffers.payload_byte_offset,
        kPayloadByteLength);
    if ((buffers.transitions.download &
         AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(download_address, epoch);
    ASSERT_GE(commands.word_count(), kMinimumChainWordCount);
    ASSERT_LE(commands.word_count(), stream.size());
    const uint64_t chain_byte_length = commands.word_count() * sizeof(uint32_t);
    uint64_t offset = sdma_index % sdma_capacity;
    const uint64_t remaining = sdma_capacity - offset;
    if (remaining < chain_byte_length) {
      // Native zero NOP dwords keep every packet contiguous at ring wrap.
      std::memset(sdma_ring + offset, 0, remaining);
      sdma_index += remaining;
      offset = 0;
    }
    std::memcpy(sdma_ring + offset, stream.data(), chain_byte_length);
    sdma_index += chain_byte_length;

    // Consumer-first publication leaves every dataflow edge on the device.
    // Both rings are free from the preceding epoch before this reservation.
    GpuStoreRelease(aql_queue->host.write_index_address, aql_index + 2);
    aql::Publish(*aql_queue, aql_index++, dependency);
    aql::Publish(*aql_queue, aql_index++, dispatch);
    ASSERT_NO_FATAL_FAILURE(sdma_queue->PublishStream(sdma_index));
    GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(&completion.download),
                           epoch);

    // Observe the actual download before any native consumption/retirement
    // call or other payload read could maintain this visibility edge for us.
    std::memcpy(downloaded.data(), readback_words, buffers.byte_length);
    // Join the shader independently of the SDMA poll under test before
    // reading its output or reusing its arguments. The final download fence
    // already joins all SDMA work, including the upload signal write.
    GpuWaitEqual<int64_t>(
        reinterpret_cast<uintptr_t>(&completion.compute.value), 0);
    std::memcpy(observed_source.data(), source_words, buffers.byte_length);
    std::memcpy(observed_input.data(), input_words, buffers.byte_length);
    std::memcpy(observed_output.data(), output_words, buffers.byte_length);
    std::memcpy(observed_arguments.data(), arguments->host.pointer,
                observed_arguments.size());
    std::memcpy(&observed_completion, control->host.pointer,
                sizeof(observed_completion));
    std::memcpy(observed_control_guards.data(),
                static_cast<const uint8_t*>(control->host.pointer) +
                    kControlGuardByteOffset,
                observed_control_guards.size());
    // Every observation precedes diagnostics, which cannot repair a captured
    // visibility result or bypass either queue's retirement on failure.
    for (uint32_t i = 0; i < word_count; ++i) {
      EXPECT_EQ(downloaded[i], expected_readback[i])
          << "set=" << set_index << " epoch=" << epoch
          << " readback word=" << i;
    }
    for (uint32_t i = 0; i < word_count; ++i) {
      EXPECT_EQ(observed_source[i], expected_source[i])
          << "set=" << set_index << " epoch=" << epoch << " source word=" << i;
      EXPECT_EQ(observed_input[i], expected_input[i])
          << "set=" << set_index << " epoch=" << epoch << " input word=" << i;
      EXPECT_EQ(observed_output[i], expected_output[i])
          << "set=" << set_index << " epoch=" << epoch << " output word=" << i;
    }
    for (uint32_t i = 0; i < expected_arguments.size(); ++i) {
      EXPECT_EQ(observed_arguments[i], expected_arguments[i])
          << "set=" << set_index << " epoch=" << epoch << " kernarg byte=" << i;
    }
    EXPECT_EQ(observed_completion.upload.kind, 1);
    EXPECT_EQ(observed_completion.upload.value, 0);
    EXPECT_EQ(observed_completion.compute.kind, 1);
    EXPECT_EQ(observed_completion.compute.value, 0);
    EXPECT_EQ(observed_completion.download, epoch);
    for (uint32_t i = 0; i < observed_control_guards.size(); ++i) {
      EXPECT_EQ(observed_control_guards[i], 0)
          << "set=" << set_index << " epoch=" << epoch
          << " control guard byte=" << kControlGuardByteOffset + i;
    }
    EXPECT_NO_FATAL_FAILURE(aql_queue->WaitConsumed(api_, aql_index));
    EXPECT_NO_FATAL_FAILURE(sdma_queue->WaitConsumed(api_, sdma_index));
    if (HasFailure()) {
      return;
    }
    ++completed_epochs[set_index];
  }
  ASSERT_GT(aql_index, 2 * aql_capacity);
  ASSERT_GT(sdma_index, 2 * sdma_capacity);
  RecordProperty("copy_dispatch_completed_epochs", std::to_string(epoch_count));
  RecordProperty("aql_final_packet_index", std::to_string(aql_index));
  RecordProperty("sdma_final_byte_index", std::to_string(sdma_index));
  for (size_t i = 0; i < completed_epochs.size(); ++i) {
    RecordProperty("payload_set_" + std::to_string(i) + "_completed_epochs",
                   std::to_string(completed_epochs[i]));
  }
}

void CopyDispatchRecipeTest::RunStagedHandoff(PairQuery query_kind) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  if ((features_ & AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) == 0) {
    GTEST_SKIP() << "staged dataflow requires owned LOCAL memory";
  }
  ASSERT_NE(local_scope_, nullptr);
  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kMaximumByteLength = 12288;
  constexpr size_t kMaximumWordCount = kMaximumByteLength / sizeof(uint32_t);
  constexpr size_t kMinimumChainWordCount = 7 + 7 + 4 + 6 + 7 + 7 + 4;
  constexpr size_t kMaximumChainWordCount = kMinimumChainWordCount + 20;
  constexpr uint64_t kMinimumChainByteLength =
      kMinimumChainWordCount * sizeof(uint32_t);
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  const amdf_memory_device_access_t source_access = {
      device_,
      {.access = AMDF_MEMORY_ACCESS_READ,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  const amdf_memory_device_access_t readback_access = {
      device_,
      {.access = kReadWrite,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  const amdf_memory_device_access_t local_access = {
      device_,
      {.access = kReadWrite, .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  amdf_memory_create_info_t source_creation = {};
  amdf_memory_create_info_t readback_creation = {};
  amdf_memory_create_info_t local_creation = {};
  ASSERT_NO_FATAL_FAILURE(SelectCreation(source_access, &source_creation));
  ASSERT_NO_FATAL_FAILURE(SelectCreation(readback_access, &readback_creation));
  ASSERT_NO_FATAL_FAILURE(SelectLocalCreation(local_access, &local_creation));
  RecordProperty("pair_query_mode",
                 query_kind == PairQuery::kProfile ? "profile" : "concrete");
  RecordProperty("source_memory_profile_ordinal",
                 source_creation.memory_profile_ordinal);
  RecordProperty("readback_memory_profile_ordinal",
                 readback_creation.memory_profile_ordinal);
  RecordProperty("local_memory_profile_ordinal",
                 local_creation.memory_profile_ordinal);
  RecordProperty("local_required_flags",
                 std::to_string(local_creation.required_flags));
  RecordProperty("local_access_flags",
                 std::to_string(local_access.requirements.flags));
  RecordProperty("local_device_access", local_access.requirements.access);
  RecordProperty("sdma_minimum_chain_word_count",
                 std::to_string(kMinimumChainWordCount));

  PayloadTransitions profile_transitions;
  if (query_kind == PairQuery::kProfile) {
    // These exact profiles and requirements create every matching allocation
    // below. Query all directional roles before any payload backing exists.
    StagedPayloadPairs pairs;
    ASSERT_NO_FATAL_FAILURE(QueryProfilePairs(local_scope_, source_creation,
                                              local_creation, readback_creation,
                                              &pairs.payload));
    const amdf_memory_profile_site_t host_write = {
        .kind = AMDF_MEMORY_SITE_KIND_HOST,
        .value = {.host_access = AMDF_MEMORY_MAP_FLAG_WRITE},
    };
    const amdf_memory_profile_site_t host_read = {
        .kind = AMDF_MEMORY_SITE_KIND_HOST,
        .value = {.host_access = AMDF_MEMORY_MAP_FLAG_READ},
    };
    const amdf_memory_profile_site_t sdma = {
        .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
        .value = {.device = {0, sdma_family_.ordinal}},
    };
    const amdf_memory_profile_site_t aql = {
        .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
        .value = {.device = {0, family_.ordinal}},
    };
    ASSERT_NO_FATAL_FAILURE(QueryProfilePair(
        system_scope_, source_creation, host_write, sdma, &pairs.seed_ingress));
    ASSERT_NO_FATAL_FAILURE(QueryProfilePair(local_scope_, local_creation, sdma,
                                             aql, &pairs.seed_upload));
    ASSERT_NO_FATAL_FAILURE(QueryProfilePair(local_scope_, local_creation, sdma,
                                             sdma, &pairs.input_download));
    ASSERT_NO_FATAL_FAILURE(QueryProfilePair(system_scope_, readback_creation,
                                             sdma, host_read,
                                             &pairs.input_egress));
    ASSERT_NO_FATAL_FAILURE(
        ResolveStagedPairs(pairs, "profile", &profile_transitions));
  }

  std::array<StagedPayloadSet, 2> payload_sets = {{
      {.payload = {.byte_length = 8192, .payload_byte_offset = 64}},
      {.payload = {.byte_length = kMaximumByteLength,
                   .payload_byte_offset = 128}},
  }};
  for (size_t i = 0; i < payload_sets.size(); ++i) {
    auto& staged = payload_sets[i];
    auto& payload = staged.payload;
    auto staged_source_creation = source_creation;
    staged_source_creation.byte_length = payload.byte_length;
    auto staged_readback_creation = readback_creation;
    staged_readback_creation.byte_length = payload.byte_length;
    auto staged_local_creation = local_creation;
    staged_local_creation.byte_length = payload.byte_length;
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(system_scope_, staged_source_creation, &payload.source));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(system_scope_, staged_source_creation, &staged.seed));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(local_scope_, staged_local_creation, &payload.input));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(local_scope_, staged_local_creation, &payload.output));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        system_scope_, staged_readback_creation, &payload.readback));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        system_scope_, staged_readback_creation, &staged.input_readback));
    const std::array<GpuMemory*, 2> local_memories = {payload.input,
                                                      payload.output};
    for (auto* memory : local_memories) {
      ASSERT_EQ(memory->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
      ASSERT_NE(memory->info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
      ASSERT_EQ(memory->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
      ASSERT_EQ(memory->mapping, nullptr);
      ASSERT_EQ(memory->host.pointer, nullptr);
      const amdf_memory_map_info_t map = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
          .structure_size = sizeof(map),
          .byte_length = memory->info.byte_length,
          .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      };
      ASSERT_EQ(amdf_status_code(
                    api_->memory_map(memory->memory, &map, &memory->mapping)),
                AMDF_STATUS_CODE_UNSUPPORTED);
      ASSERT_EQ(memory->mapping, nullptr);
    }
    const std::string prefix = "payload_set_" + std::to_string(i);
    RecordProperty(prefix + "_byte_length",
                   std::to_string(payload.byte_length));
    RecordProperty(prefix + "_byte_offset",
                   std::to_string(payload.payload_byte_offset));
    RecordProperty(prefix + "_source_address",
                   std::to_string(payload.source->device_address));
    RecordProperty(prefix + "_seed_address",
                   std::to_string(staged.seed->device_address));
    RecordProperty(prefix + "_input_address",
                   std::to_string(payload.input->device_address));
    RecordProperty(prefix + "_output_address",
                   std::to_string(payload.output->device_address));
    RecordProperty(prefix + "_readback_address",
                   std::to_string(payload.readback->device_address));
    RecordProperty(prefix + "_input_readback_address",
                   std::to_string(staged.input_readback->device_address));
    RecordProperty(prefix + "_input_backing_flags",
                   std::to_string(payload.input->info.flags));
    RecordProperty(prefix + "_output_backing_flags",
                   std::to_string(payload.output->info.flags));
    RecordProperty(prefix + "_input_access_flags",
                   std::to_string(payload.input->access_info.flags));
    RecordProperty(prefix + "_output_access_flags",
                   std::to_string(payload.output->access_info.flags));
    if (query_kind == PairQuery::kConcrete) {
      StagedPayloadPairs pairs;
      ASSERT_NO_FATAL_FAILURE(QueryConcretePairs(payload, &pairs.payload));
      ASSERT_NO_FATAL_FAILURE(QueryConcretePair(
          staged.seed->HostSite(),
          staged.seed->DeviceSite(sdma_family_.ordinal), &pairs.seed_ingress));
      ASSERT_NO_FATAL_FAILURE(QueryConcretePair(
          payload.output->DeviceSite(sdma_family_.ordinal),
          payload.output->DeviceSite(family_.ordinal), &pairs.seed_upload));
      ASSERT_NO_FATAL_FAILURE(
          QueryConcretePair(payload.input->DeviceSite(sdma_family_.ordinal),
                            payload.input->DeviceSite(sdma_family_.ordinal),
                            &pairs.input_download));
      ASSERT_NO_FATAL_FAILURE(QueryConcretePair(
          staged.input_readback->DeviceSite(sdma_family_.ordinal),
          staged.input_readback->HostSite(), &pairs.input_egress));
      ASSERT_NO_FATAL_FAILURE(
          ResolveStagedPairs(pairs, prefix, &payload.transitions));
    } else {
      payload.transitions = profile_transitions;
    }
    RecordProperty(prefix + "_dispatch_acquire_scope",
                   static_cast<uint32_t>(payload.transitions.dispatch.acquire));
    RecordProperty(prefix + "_dispatch_release_scope",
                   static_cast<uint32_t>(payload.transitions.dispatch.release));
  }

  GpuMemory* arguments = nullptr;
  GpuMemory* control = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, kPageByteLength, &control));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_EQ(control->device_address % alignof(CompletionState), 0u);
  std::memset(control->host.pointer, 0, control->info.byte_length);
  auto& completion = *static_cast<CompletionState*>(control->host.pointer);
  completion.upload.kind = completion.compute.kind = 1;
  const uint64_t upload_signal_address =
      control->device_address + offsetof(CompletionState, upload);
  const uint64_t compute_signal_address =
      control->device_address + offsetof(CompletionState, compute);
  const uint64_t download_address =
      control->device_address + offsetof(CompletionState, download);
  RecordProperty("copy_dispatch_argument_byte_length", kPageByteLength);
  RecordProperty("copy_dispatch_control_guard_byte_offset",
                 kControlGuardByteOffset);
  RecordProperty("copy_dispatch_control_guard_byte_length",
                 kPageByteLength - kControlGuardByteOffset);

  GpuUserQueue* aql_queue = nullptr;
  GpuUserQueue* sdma_queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&aql_queue));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(sdma_family_, &sdma_queue));
  ASSERT_TRUE(amdf_device_id_is_equal(&aql_queue->info.device_id,
                                      &sdma_queue->info.device_id));
  ASSERT_FALSE(amdf_queue_id_is_equal(&aql_queue->info.queue_id,
                                      &sdma_queue->info.queue_id));
  const uint64_t aql_capacity =
      aql_queue->host.ring_byte_length / sizeof(aql::Packet);
  const uint64_t sdma_capacity = sdma_queue->host.ring_byte_length;
  ASSERT_GE(aql_capacity, 2u);
  ASSERT_GT(sdma_capacity, 2 * kMaximumChainWordCount * sizeof(uint32_t));
  const uint64_t epoch_count =
      2 * std::max((aql_capacity + 1) / 2,
                   sdma_capacity / kMinimumChainByteLength) +
      1;
  ASSERT_LE(epoch_count, UINT32_MAX);
  RecordProperty("copy_dispatch_epoch_count", std::to_string(epoch_count));
  RecordProperty("aql_queue_family_ordinal", family_.ordinal);
  RecordProperty("sdma_queue_family_ordinal", sdma_family_.ordinal);
  RecordProperty("aql_ring_packet_capacity", std::to_string(aql_capacity));
  RecordProperty("sdma_ring_byte_capacity", std::to_string(sdma_capacity));
  uint64_t aql_index = 0;
  uint64_t sdma_index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*aql_queue, kernel, "aql_kernel",
                                        &aql_index, &descriptor_address));

  auto* sdma_ring = reinterpret_cast<uint8_t*>(sdma_queue->host.ring_address);
  std::array<uint32_t, kMaximumWordCount> expected_input;
  std::array<uint32_t, kMaximumWordCount> expected_seed;
  std::array<uint32_t, kMaximumWordCount> expected_output;
  std::array<uint32_t, kMaximumWordCount> downloaded;
  std::array<uint32_t, kMaximumWordCount> observed_input_readback;
  std::array<uint32_t, kMaximumWordCount> observed_source;
  std::array<uint32_t, kMaximumWordCount> observed_seed;
  std::array<uint8_t, kPageByteLength> expected_arguments;
  std::array<uint8_t, kPageByteLength> observed_arguments;
  CompletionState observed_completion;
  std::array<uint8_t, kPageByteLength - kControlGuardByteOffset>
      observed_control_guards;
  std::array<uint64_t, 2> completed_epochs = {};
  for (uint64_t epoch_index = 0; epoch_index < epoch_count; ++epoch_index) {
    // Each backing sees both active-count variants before the next backing.
    const size_t set_index = (epoch_index / 2) % payload_sets.size();
    const auto& staged = payload_sets[set_index];
    const auto& buffers = staged.payload;
    const uint32_t byte_length = static_cast<uint32_t>(buffers.byte_length);
    const uint32_t word_count = byte_length / sizeof(uint32_t);
    const uint32_t payload_word_offset =
        static_cast<uint32_t>(buffers.payload_byte_offset / sizeof(uint32_t));
    auto* source_words = static_cast<uint32_t*>(buffers.source->host.pointer);
    auto* seed_words = static_cast<uint32_t*>(staged.seed->host.pointer);
    auto* readback_words =
        static_cast<uint32_t*>(buffers.readback->host.pointer);
    auto* input_readback_words =
        static_cast<uint32_t*>(staged.input_readback->host.pointer);
    const uint32_t epoch = static_cast<uint32_t>(epoch_index + 1);
    const uint32_t count = epoch % 2 == 1 ? 1003 : 997;
    const uint32_t addend = 0x80000001u + 2 * epoch;
    for (uint32_t i = 0; i < word_count; ++i) {
      expected_input[i] = 0xfffffff0u + i * 0x01030507u + 5 * epoch +
                          static_cast<uint32_t>(set_index) * 0x103u;
      expected_seed[i] =
          0x93b57fd1u ^ (i * 0x9e3779b9u + epoch * 0x01010101u +
                         static_cast<uint32_t>(set_index) * 0x1234u);
      expected_output[i] = expected_seed[i];
    }
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t position = payload_word_offset + i;
      // The CPU oracle uses only generated inputs, never GPU observations.
      const uint32_t result = static_cast<uint32_t>(
          uint64_t{expected_input[position]} * 3 + addend);
      expected_output[position] = result;
      expected_seed[position] = result ^ 0x5a5a5a5au;
    }
    std::memcpy(source_words, expected_input.data(), byte_length);
    std::memcpy(seed_words, expected_seed.data(), byte_length);
    for (uint32_t i = 0; i < word_count; ++i) {
      readback_words[i] = ~expected_output[i];
      input_readback_words[i] = ~expected_input[i];
    }
    const kernels::transform::Arguments payload = {
        buffers.input->device_address + buffers.payload_byte_offset,
        buffers.output->device_address + buffers.payload_byte_offset,
        count,
        addend,
    };
    expected_arguments.fill(0);
    std::memcpy(expected_arguments.data(), &payload,
                kernel.arguments.byte_length);
    std::memcpy(arguments->host.pointer, expected_arguments.data(),
                expected_arguments.size());
    // Rearming follows the previous epoch's download and both consumed
    // frontiers. The complete native values always have a zero high word.
    completion.upload.value = completion.compute.value = 1;
    const auto dependency =
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
                     {upload_signal_address},
                     {aql::FenceScope::kNone, aql::FenceScope::kNone});
    const auto dispatch = aql::Dispatch(
        aql::HeaderBarrier::kDisabled,
        {1,
         {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
         {kGridSize, 1, 1}},
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        descriptor_address, arguments->device_address, compute_signal_address,
        buffers.transitions.dispatch);
    std::array<uint32_t, kMaximumChainWordCount> stream = {};
    SdmaCommandWriter commands(stream.data(), sdma_family_.format_features);
    if ((buffers.transitions.upload &
         AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(buffers.source->device_address,
                        buffers.input->device_address, byte_length);
    commands.CopyLinear(staged.seed->device_address,
                        buffers.output->device_address, byte_length);
    if ((buffers.transitions.upload &
         AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(upload_signal_address + offsetof(aql::Signal, value), 0);
    commands.WaitMemory32(compute_signal_address + offsetof(aql::Signal, value),
                          0);
    if ((buffers.transitions.download &
         AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(buffers.output->device_address,
                        buffers.readback->device_address, byte_length);
    commands.CopyLinear(buffers.input->device_address,
                        staged.input_readback->device_address, byte_length);
    if ((buffers.transitions.download &
         AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(download_address, epoch);
    ASSERT_GE(commands.word_count(), kMinimumChainWordCount);
    ASSERT_LE(commands.word_count(), stream.size());
    const uint64_t chain_byte_length = commands.word_count() * sizeof(uint32_t);
    uint64_t offset = sdma_index % sdma_capacity;
    const uint64_t remaining = sdma_capacity - offset;
    if (remaining < chain_byte_length) {
      // Native zero NOP dwords keep each complete packet contiguous.
      std::memset(sdma_ring + offset, 0, remaining);
      sdma_index += remaining;
      offset = 0;
    }
    std::memcpy(sdma_ring + offset, stream.data(), chain_byte_length);
    sdma_index += chain_byte_length;

    // Consumer-first publication leaves both inter-engine edges on device.
    GpuStoreRelease(aql_queue->host.write_index_address, aql_index + 2);
    aql::Publish(*aql_queue, aql_index++, dependency);
    aql::Publish(*aql_queue, aql_index++, dispatch);
    ASSERT_NO_FATAL_FAILURE(sdma_queue->PublishStream(sdma_index));
    GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(&completion.download),
                           epoch);

    // Capture the full output first. Diagnostic payload reads and native
    // consumed/retirement calls cannot repair this already captured result.
    std::memcpy(downloaded.data(), readback_words, byte_length);
    // The final SDMA fence joins all copies. Independently join the shader
    // before observing its arguments/control or allowing another epoch.
    GpuWaitEqual<int64_t>(
        reinterpret_cast<uintptr_t>(&completion.compute.value), 0);
    std::memcpy(observed_input_readback.data(), input_readback_words,
                byte_length);
    std::memcpy(observed_source.data(), source_words, byte_length);
    std::memcpy(observed_seed.data(), seed_words, byte_length);
    std::memcpy(observed_arguments.data(), arguments->host.pointer,
                observed_arguments.size());
    std::memcpy(&observed_completion, control->host.pointer,
                sizeof(observed_completion));
    std::memcpy(observed_control_guards.data(),
                static_cast<const uint8_t*>(control->host.pointer) +
                    kControlGuardByteOffset,
                observed_control_guards.size());
    for (uint32_t i = 0; i < word_count; ++i) {
      EXPECT_EQ(downloaded[i], expected_output[i])
          << "set=" << set_index << " epoch=" << epoch << " output word=" << i;
    }
    for (uint32_t i = 0; i < word_count; ++i) {
      EXPECT_EQ(observed_input_readback[i], expected_input[i])
          << "set=" << set_index << " epoch=" << epoch
          << " input readback word=" << i;
      EXPECT_EQ(observed_source[i], expected_input[i])
          << "set=" << set_index << " epoch=" << epoch
          << " input staging word=" << i;
      EXPECT_EQ(observed_seed[i], expected_seed[i])
          << "set=" << set_index << " epoch=" << epoch
          << " output seed word=" << i;
    }
    for (uint32_t i = 0; i < expected_arguments.size(); ++i) {
      EXPECT_EQ(observed_arguments[i], expected_arguments[i])
          << "set=" << set_index << " epoch=" << epoch << " kernarg byte=" << i;
    }
    EXPECT_EQ(observed_completion.upload.kind, 1);
    EXPECT_EQ(observed_completion.upload.value, 0);
    EXPECT_EQ(observed_completion.compute.kind, 1);
    EXPECT_EQ(observed_completion.compute.value, 0);
    EXPECT_EQ(observed_completion.download, epoch);
    for (uint32_t i = 0; i < observed_control_guards.size(); ++i) {
      EXPECT_EQ(observed_control_guards[i], 0)
          << "set=" << set_index << " epoch=" << epoch
          << " control guard byte=" << kControlGuardByteOffset + i;
    }
    EXPECT_NO_FATAL_FAILURE(aql_queue->WaitConsumed(api_, aql_index));
    EXPECT_NO_FATAL_FAILURE(sdma_queue->WaitConsumed(api_, sdma_index));
    if (HasFailure()) {
      return;
    }
    ++completed_epochs[set_index];
  }
  ASSERT_GT(aql_index, 2 * aql_capacity);
  ASSERT_GT(sdma_index, 2 * sdma_capacity);
  RecordProperty("copy_dispatch_completed_epochs", std::to_string(epoch_count));
  RecordProperty("aql_final_packet_index", std::to_string(aql_index));
  RecordProperty("sdma_final_byte_index", std::to_string(sdma_index));
  for (size_t i = 0; i < payload_sets.size(); ++i) {
    ASSERT_GE(completed_epochs[i], 2u);
    RecordProperty("payload_set_" + std::to_string(i) + "_completed_epochs",
                   std::to_string(completed_epochs[i]));
  }
}

TEST_F(CopyDispatchRecipeTest,
       ConcreteCoherentSystemUploadDispatchDownloadReusesBothRings) {
  RunCoherentHandoff(PairQuery::kConcrete);
}

TEST_F(CopyDispatchRecipeTest,
       ProfileCoherentSystemUploadDispatchDownloadReusesBothRings) {
  RunCoherentHandoff(PairQuery::kProfile);
}

TEST_F(CopyDispatchRecipeTest, UserGcrUploadDispatchDownloadReusesBothRings) {
  RunCoherentHandoff(PairQuery::kConcrete,
                     AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM |
                         AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM);
}

TEST_F(CopyDispatchRecipeTest,
       ConcreteStagedLocalUploadDispatchDownloadReusesBothRings) {
  RunStagedHandoff(PairQuery::kConcrete);
}

TEST_F(CopyDispatchRecipeTest,
       ProfileStagedLocalUploadDispatchDownloadReusesBothRings) {
  RunStagedHandoff(PairQuery::kProfile);
}

}  // namespace
