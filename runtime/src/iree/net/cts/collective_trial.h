// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CTS_COLLECTIVE_TRIAL_H_
#define IREE_NET_CTS_COLLECTIVE_TRIAL_H_

#include "iree/net/cts/collective_transport.h"

namespace iree::net::cts {

// Distinct dependency shapes, not interchangeable collective algorithms.
enum class CollectiveSchedule { kTensorAllReduce, kPipeline };

// Checked host schedules shared by CTS and fixed-work benchmarks.
struct CollectiveTrialOptions {
  // Dependency schedule and checked application operation.
  CollectiveSchedule schedule = CollectiveSchedule::kTensorAllReduce;
  // Independently polling application ranks or pipeline stages.
  uint32_t rank_count = 2;
  // Full tensor/activation bytes, divisible by sizeof(uint32_t). All-reduce
  // additionally requires equal, whole-element shards across ranks.
  size_t tensor_size = 4096;
  // Maximum bytes per transfer, divisible by sizeof(uint32_t); tails are exact.
  size_t block_size = 1024;
  // Maximum outstanding source callbacks per edge; also bounds all-reduce
  // input slots. Pipeline inputs instead hold pipeline_depth whole activations.
  uint32_t window_size = 4;
  // Maximum microbatches in flight; must be one for the all-reduce schedule.
  uint32_t pipeline_depth = 1;
  // Complete all-reduces or microbatches before measurement.
  uint32_t warmup_rounds = 2;
  // Complete all-reduces or microbatches in the measured phase.
  uint32_t measured_rounds = 17;
  // Payload ownership strategy used by every rank.
  CollectiveDelivery delivery = CollectiveDelivery::kMessage;
};

struct CollectiveTrialResult : CollectiveTransportResult {
  // Whole all-reduces completed by the rank group, not multiplied by ranks.
  uint64_t collectives = 0;
  // Pipeline microbatches checked at the final stage and observed by rank zero.
  uint64_t microbatches = 0;
  // Pipeline phase start through rank zero observing its first final result.
  double first_completion_seconds = 0;
  // Maximum pipeline microbatches admitted but not yet observed complete.
  uint64_t pipeline_high_water = 0;
};

// Runs a ring reduce-scatter/all-gather or pipeline over actual connections.
// Each rank owns an application poll thread and bounded payload storage. In the
// all-reduce schedule, incoming blocks are consumed before forwarding; outgoing
// sources join before that tensor storage is modified. Every rank checks its
// entire reduced tensor after each round. Subsequent rounds depend on that
// check.
//
// Pipeline stages receive a whole activation before computing a deterministic
// transform and sending their output. Bounded microbatch slots allow overlap;
// output reuse joins its exact source callbacks independently of peer input
// consumption. Message blocks assemble into application-owned stage inputs;
// registered blocks land directly there. The final stage checks its entire
// output and returns completion to rank zero over a control-only connection.
// First-result latency and amortized throughput have separate result fields.
//
// Registered mode places into reusable application inputs without message
// staging. Message mode consumes the transport's actual receive leases. Only
// bootstrap, phase joins and errors use cross-thread state outside the net
// APIs. Setup, warm-up and teardown are excluded from timing; host reduction,
// validation, progress feedback and every source callback are included.
// This same-process host schedule is not a GPU collective implementation or
// a measurement of multi-host topology, isolated wire latency or link
// bandwidth.
iree_status_t RunCollectiveTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const CollectiveTrialOptions& options, TransferTrialMeasurement measurement,
    CollectiveTrialResult* out_result);

}  // namespace iree::net::cts

#endif  // IREE_NET_CTS_COLLECTIVE_TRIAL_H_
