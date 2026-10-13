// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/recipes/device_sdma_fixture.h"

template <typename DispatchTest>
amdf_status_t DeviceSdmaTest<DispatchTest>::MatchGpuEndpoint(
    amdf_endpoint_t* endpoint, bool* out_matches) {
  const GpuQueueRequirements requirements = {
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
      .roles = AMDF_QUEUE_ROLE_TRANSFER,
      .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
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
  status = DispatchTest::MatchGpuEndpoint(endpoint, &matches);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (matches) {
    sdma_family_ = sdma_family;
  }
  *out_matches = matches;
  return AMDF_STATUS_OK;
}

template <typename DispatchTest>
void DeviceSdmaTest<DispatchTest>::QueryPair(
    const amdf_memory_site_t& producer, const amdf_memory_site_t& consumer,
    amdf_memory_pair_info_t* out_pair) {
  out_pair->type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
  out_pair->structure_size = sizeof(*out_pair);
  ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, out_pair),
            AMDF_STATUS_OK);
  ASSERT_NE(out_pair->flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
            0u);
}

template <typename DispatchTest>
void DeviceSdmaTest<DispatchTest>::CheckTransition(
    const amdf_cache_transition_t& transition,
    amdf_cache_operation_t operation) {
  ASSERT_EQ(transition.kind, operation == AMDF_CACHE_OPERATION_NONE
                                 ? AMDF_CACHE_TRANSITION_KIND_NONE
                                 : AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  ASSERT_EQ(transition.executor, operation == AMDF_CACHE_OPERATION_NONE
                                     ? AMDF_CACHE_TRANSITION_EXECUTOR_NONE
                                     : AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  ASSERT_EQ(transition.operation, operation);
  ASSERT_EQ(transition.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
  ASSERT_EQ(transition.host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
  ASSERT_EQ(transition.host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.range_granularity, 0u);
}

template <typename DispatchTest>
void DeviceSdmaTest<DispatchTest>::ResolveSdmaTransition(
    const amdf_cache_transition_t& transition, amdf_cache_operation_t operation,
    uint32_t* inout_cache_flags) {
  if (transition.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    ASSERT_NO_FATAL_FAILURE(
        CheckTransition(transition, AMDF_CACHE_OPERATION_NONE));
    return;
  }
  ASSERT_NO_FATAL_FAILURE(CheckTransition(transition, operation));
  ASSERT_NE(
      sdma_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR, 0u);
  ASSERT_NE(sdma_family_.cache_operations & (UINT64_C(1) << operation), 0u);
  ASSERT_NE(
      sdma_family_.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      0u);
  *inout_cache_flags |=
      operation == AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM ? 1u : 2u;
}

template <typename DispatchTest>
void DeviceSdmaTest<DispatchTest>::QueryTransferCacheFlags(
    GpuMemory* source, GpuMemory* destination, GpuMemory* records,
    uint32_t* out_flags) {
  amdf_memory_pair_info_t ingress = {};
  amdf_memory_pair_info_t copied = {};
  amdf_memory_pair_info_t egress = {};
  ASSERT_NO_FATAL_FAILURE(QueryPair(
      source->HostSite(), source->DeviceSite(sdma_family_.ordinal), &ingress));
  ASSERT_NO_FATAL_FAILURE(
      QueryPair(destination->DeviceSite(sdma_family_.ordinal),
                destination->DeviceSite(family_.ordinal), &copied));
  ASSERT_NO_FATAL_FAILURE(QueryPair(records->DeviceSite(family_.ordinal),
                                    records->HostSite(), &egress));
  ASSERT_NO_FATAL_FAILURE(
      CheckTransition(ingress.release, AMDF_CACHE_OPERATION_NONE));
  ASSERT_NO_FATAL_FAILURE(CheckTransition(
      copied.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
  ASSERT_NO_FATAL_FAILURE(
      CheckTransition(egress.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
  ASSERT_NO_FATAL_FAILURE(
      CheckTransition(egress.acquire, AMDF_CACHE_OPERATION_NONE));
  ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
      ingress.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM, out_flags));
  ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
      copied.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM, out_flags));
}

template <typename DispatchTest>
void DeviceSdmaTest<DispatchTest>::QueryDownloadCacheFlags(
    GpuMemory* output, GpuMemory* readback, uint32_t* out_flags) {
  amdf_memory_pair_info_t produced = {};
  amdf_memory_pair_info_t copied = {};
  amdf_memory_pair_info_t egress = {};
  ASSERT_NO_FATAL_FAILURE(QueryPair(output->DeviceSite(family_.ordinal),
                                    output->DeviceSite(sdma_family_.ordinal),
                                    &produced));
  ASSERT_NO_FATAL_FAILURE(QueryPair(readback->DeviceSite(sdma_family_.ordinal),
                                    readback->DeviceSite(family_.ordinal),
                                    &copied));
  ASSERT_NO_FATAL_FAILURE(QueryPair(readback->DeviceSite(sdma_family_.ordinal),
                                    readback->HostSite(), &egress));
  ASSERT_NO_FATAL_FAILURE(CheckTransition(
      produced.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
  ASSERT_NO_FATAL_FAILURE(CheckTransition(
      copied.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
  ASSERT_NO_FATAL_FAILURE(
      CheckTransition(egress.acquire, AMDF_CACHE_OPERATION_NONE));
  ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
      produced.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM, out_flags));
  ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
      copied.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM, out_flags));
  ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
      egress.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM, out_flags));
}

template class DeviceSdmaTest<AqlDispatchTest>;
template class DeviceSdmaTest<Pm4DispatchTest>;
