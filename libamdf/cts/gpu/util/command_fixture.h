// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_UTIL_COMMAND_FIXTURE_H_
#define AMDF_CTS_GPU_UTIL_COMMAND_FIXTURE_H_

#include <deque>

#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/gpu/util/command_queue.h"
#include "libamdf/cts/gpu/util/memory.h"
#include "libamdf/cts/gpu/util/user_queue.h"

// Operation and encoding requirements for one case-owned queue family.
struct GpuQueueRequirements {
  // Engine packet representation required by the case.
  amdf_queue_command_type_t command_type;
  // Family operation roles required by the case.
  amdf_queue_roles_t roles;
  // Optional packet fields required by the encoding.
  amdf_queue_format_features_t format_features = 0;
  // Semantic cache operations required in addition to packet fields.
  amdf_cache_operations_t cache_operations = 0;
  // Range or global domains used by the cache commands.
  amdf_cache_transition_kinds_t cache_transition_kinds = 0;
  // Native publication modes the caller can use without changing the case.
  amdf_queue_publication_modes_t publication_modes =
      AMDF_QUEUE_PUBLICATION_MODE_USER;
};

// Queries one passive endpoint without acquiring a native device. Success
// publishes a match result; the family output changes only on a match.
amdf_status_t FindGpuQueueFamily(const amdf_api_t* api,
                                 amdf_endpoint_t* endpoint,
                                 const GpuQueueRequirements& requirements,
                                 amdf_queue_family_info_t* out_family,
                                 bool* out_matches);

// Borrows the corpus's cached device and owns only this case's workload.
// No helper emits cache commands or conflates ring consumption with execution.
class GpuCommandTest : public GpuDeviceFixture {
 protected:
  explicit GpuCommandTest(const GpuQueueRequirements& requirements)
      : requirements_(requirements) {}

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override;
  void TearDown() override;
  void CreateMemory(amdf_memory_access_t access, uint64_t byte_length,
                    GpuMemory** out_memory);
  // Uses the exact profile and access inputs already queried by the caller.
  void CreateMemory(amdf_memory_scope_t* scope,
                    const amdf_memory_create_info_t& create_info,
                    GpuMemory** out_memory);
  void CreateQueue(GpuUserQueue** out_queue,
                   amdf_queue_producer_mode_t producer_mode =
                       AMDF_QUEUE_PRODUCER_MODE_SINGLE,
                   const amdf_gpu_queue_scratch_t& scratch = {});
  void CreateQueue(const amdf_queue_family_info_t& family,
                   GpuUserQueue** out_queue,
                   amdf_queue_producer_mode_t producer_mode =
                       AMDF_QUEUE_PRODUCER_MODE_SINGLE,
                   const amdf_gpu_queue_scratch_t& scratch = {});
  void CreateQueue(GpuCommandQueue** out_queue);
  void CreateQueue(const amdf_queue_family_info_t& family,
                   GpuCommandQueue** out_queue);

  // Exact family chosen passively before borrowing the cached native device.
  amdf_queue_family_info_t family_ = {};

 private:
  // Requirements for the default family, selected before native activation.
  GpuQueueRequirements requirements_;
  // Stable case-owned allocations, released only after every queue succeeds.
  std::deque<GpuMemory> memories_;
  // Stable case-owned queues, sharing the same cached native device.
  std::deque<GpuUserQueue> queues_;
  // Case-owned PM4/SDMA streams using the passively admitted publication mode.
  std::deque<GpuCommandQueue> command_queues_;
};

#endif  // AMDF_CTS_GPU_UTIL_COMMAND_FIXTURE_H_
