// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_AQL_DISPATCH_FIXTURE_H_
#define AMDF_CTS_GPU_AQL_DISPATCH_FIXTURE_H_

#include <cstdint>
#include <deque>

#include "libamdf/cts/gpu/aql/executable.h"
#include "libamdf/cts/gpu/aql/queue_fixture.h"
#include "libamdf/cts/gpu/kernels/kernel.h"

// Target-selected executable publication completed before workload dispatch.
// Dataflow dependencies and result observations belong to each consuming case.
class AqlDispatchTest : public AqlQueueTest {
 protected:
  explicit AqlDispatchTest(amdf_queue_format_features_t format_features = 0)
      : AqlQueueTest(AMDF_QUEUE_ROLE_CACHE_CONTROL, format_features) {}

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override;
  void TearDown() override;

  // Backs every physical scratch slot and retains it through queue destruction.
  // Each dispatch's private requirement fits the configured per-workitem limit.
  void CreateFixedScratchQueue(uint32_t maximum_private_segment_byte_length,
                               GpuUserQueue** out_queue);

  // Allocates case-owned executable storage and a private setup signal, then
  // waits for code publication and ring consumption. Advances the packet index
  // at reservation and publishes the descriptor address only after completion.
  // All backing remains owned by the case through successful queue destruction.
  // Distinct images use distinct receipt prefixes to retain both identities.
  void PublishKernel(GpuUserQueue& queue, const kernels::Kernel& kernel,
                     const char* property_prefix, uint64_t* next_packet_index,
                     uint64_t* out_descriptor_address);

 private:
  // Stable image owners released only after every case-owned queue is removed.
  std::deque<aql::Executable> executables_;
};

#endif  // AMDF_CTS_GPU_AQL_DISPATCH_FIXTURE_H_
