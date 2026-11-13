// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CTS_PIPELINE_ROUND_TRIP_H_
#define IREE_NET_CTS_PIPELINE_ROUND_TRIP_H_

#include "iree/net/cts/collective_transport.h"

namespace iree::net::cts {

struct PipelineRoundTripOptions {
  // Application stages in a forward/backward chain.
  uint32_t rank_count = 4;
  // Forward activation bytes, a positive multiple of four.
  size_t activation_size = 8196;
  // Reverse contribution bytes, independently sized and a multiple of four.
  size_t gradient_size = 4100;
  // Maximum transfer bytes in either direction, a multiple of four.
  size_t block_size = 1024;
  // Maximum source callbacks outstanding per direction on one edge.
  uint32_t window_size = 4;
  // Microbatches admitted at rank zero but not yet returned and checked.
  uint32_t depth = 3;
  // Full round trips before measurement.
  uint32_t warmup_rounds = 2;
  // Full round trips in the measured phase.
  uint32_t measured_rounds = 11;
  // Explicit message or registered target placement.
  CollectiveDelivery delivery = CollectiveDelivery::kMessage;
};

struct PipelineRoundTripResult : CollectiveTransportResult {
  // Checked upstream results, not multiplied by stages or directions.
  uint64_t microbatches = 0;
  // Rank-zero phase entry through its first fully checked reverse result.
  double first_completion_seconds = 0;
  // Peak rank-zero microbatches admitted but not yet checked complete.
  uint64_t pipeline_high_water = 0;
};

// Sends whole activations downstream and different-sized contributions back.
// Each backward transform reads that stage's retained forward activation;
// rank zero checks the complete analytic result. All stage slots are bounded
// by depth, and activation reuse joins backward consumption plus exact source
// returns in both directions. Forward/reverse transfer progress is independent.
//
// Message inputs release transport leases after assembly, while registered
// writes land directly in the stage inputs. This measures checked host work and
// actual transport progress, not GPU compute overlap or backpropagation speed.
// First-result latency and amortized throughput have separate result fields.
iree_status_t RunPipelineRoundTrip(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const PipelineRoundTripOptions& options,
    TransferTrialMeasurement measurement, PipelineRoundTripResult* out_result);

}  // namespace iree::net::cts

#endif  // IREE_NET_CTS_PIPELINE_ROUND_TRIP_H_
