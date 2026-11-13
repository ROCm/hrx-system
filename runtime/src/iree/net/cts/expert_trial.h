// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CTS_EXPERT_TRIAL_H_
#define IREE_NET_CTS_EXPERT_TRIAL_H_

#include "iree/net/cts/collective_transport.h"

namespace iree::net::cts {

// Uniform input, rotating empty/unequal producers, or an idle hot receiver.
enum class ExpertTraffic { kUniform, kUneven, kHotspot };

struct ExpertTrialOptions {
  // Independent application poll owners, each owning an equal expert shard.
  uint32_t rank_count = 4;
  // Maximum input tokens per rank; actual counts may be smaller or zero.
  uint32_t token_capacity = 8;
  // Total experts, evenly divisible across ranks.
  uint32_t expert_count = 32;
  // Distinct experts per token; several may reside on the same destination.
  uint32_t top_k = 6;
  // Per-token activation bytes, a positive multiple of four.
  size_t dispatch_bytes = 256;
  // Separate per-token scale bytes, a multiple of four; zero is permitted.
  size_t scale_bytes = 16;
  // Per-token returned contribution bytes, a multiple of four and at least
  // dispatch_bytes. These model wire geometry, not a floating-point codec.
  size_t combine_bytes = 512;
  // Maximum per-transfer bytes; a multiple of four and at least 16.
  size_t block_size = 1024;
  // Exact source callbacks outstanding per directed peer connection.
  uint32_t window_size = 4;
  // Independent rounds retained together; consumed newest first within a batch.
  uint32_t depth = 2;
  // Consecutive rounds reusing a rank's private routes, with new payloads.
  uint32_t route_reuse_rounds = 3;
  // Input count and destination-load distribution.
  ExpertTraffic traffic = ExpertTraffic::kUneven;
  // Complete routed round trips before measurement.
  uint32_t warmup_rounds = 2;
  // Complete routed round trips in the measured phase.
  uint32_t measured_rounds = 5;
  // Explicit message or registered target placement.
  CollectiveDelivery delivery = CollectiveDelivery::kMessage;
};

struct ExpertTrialResult : CollectiveTransportResult {
  // Checked dispatch/combine round trips, not multiplied by ranks.
  uint64_t rounds = 0;
  // Input tokens across all ranks, excluding replicated destinations.
  uint64_t tokens = 0;
  // Distinct remote destinations across all input tokens.
  uint64_t remote_tokens = 0;
  // Expert contributions computed locally without network traffic.
  uint64_t local_routes = 0;
  // Remote activation bytes, excluding scales and route metadata.
  uint64_t dispatch_bytes = 0;
  // Remote scale bytes.
  uint64_t scale_bytes = 0;
  // Remote returned values, excluding inverse token indices.
  uint64_t combine_bytes = 0;
  // Route identifiers, weights and inverse token indices sent as data.
  uint64_t metadata_bytes = 0;
  // Application count/round/length headers, including empty participation.
  uint64_t header_bytes = 0;
  // Largest receiver's remote token count in a single round.
  uint64_t maximum_receiver_tokens = 0;
  // Handles consumed after a younger handle became ready in the same batch.
  uint64_t retained_handles = 0;
  // Maximum rank sum of dispatch preparation and all-input readiness times.
  double dispatch_seconds = 0;
  // Maximum rank sum of expert work, inverse transfer and final result checks.
  double combine_seconds = 0;
};

// Runs a full directed mesh with source-private routes and explicit count
// frames. Each token crosses an edge once regardless of how many of its experts
// live there. The expert uses received routes, values and scales; the caller
// checks the weighted result against its retained original routes. Empty
// producers still announce participation and may receive the entire expert
// load.
//
// A bounded batch retains all dispatch inputs until the younger round is ready,
// then computes/returns contributions in reverse round order. Input consumption
// is explicit, source callbacks join independently, and all handles join before
// batch storage reuse. This is a host ownership workload, not an MoE kernel or
// a claim about GPU/NIC overlap. Setup/registration are outside measurement;
// preparation, host transforms, checking, feedback and source joins are inside.
iree_status_t RunExpertTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const ExpertTrialOptions& options, TransferTrialMeasurement measurement,
    ExpertTrialResult* out_result);

}  // namespace iree::net::cts

#endif  // IREE_NET_CTS_EXPERT_TRIAL_H_
