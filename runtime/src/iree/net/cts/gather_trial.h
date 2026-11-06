// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CTS_GATHER_TRIAL_H_
#define IREE_NET_CTS_GATHER_TRIAL_H_

#include "iree/net/cts/collective_transport.h"

namespace iree::net::cts {

struct GatherTrialOptions {
  // Application ranks, each contributing one distinct shard per handle.
  uint32_t rank_count = 4;
  // Bytes contributed by each rank, a positive multiple of four.
  size_t shard_size = 8196;
  // Maximum bytes per transfer, a positive multiple of four.
  size_t block_size = 1024;
  // Outstanding source callbacks per directed edge, independent of depth.
  uint32_t window_size = 4;
  // Gather handles retained together before the session storage can be reused.
  uint32_t depth = 3;
  // Complete gathered handles before measurement.
  uint32_t warmup_rounds = 2;
  // Complete gathered handles in the measured phase.
  uint32_t measured_rounds = 7;
  // Explicit message or registered target placement.
  CollectiveDelivery delivery = CollectiveDelivery::kMessage;
};

struct GatherTrialResult : CollectiveTransportResult {
  // Group-wide all-gathers checked by every rank, not multiplied by ranks.
  uint64_t gathers = 0;
  // Handles per rank checked while an older handle is still retained.
  uint64_t out_of_order_reads = 0;
  // Largest per-rank set of application handles concurrently retained.
  uint64_t handles_high_water = 0;
};

// All ranks exchange distinct shards through a real peer mesh. A bounded batch
// of gather handles shares setup registration but has disjoint payload storage.
// Newer handles are read first, checking every rank's shard while older views
// remain retained. Only actual application consumption releases target credit;
// exact source callbacks join before the next batch overwrites its inputs.
//
// Message inputs assemble in owned slots and release transport leases promptly;
// registered writes land in those same final gather inputs. No timer or shared
// peer progress state supplies readiness. Generation, checked reads, feedback
// and source joins are measured; setup, warm-up and teardown are not. This is
// a host retention/progress workload, not GPU compute-overlap qualification.
iree_status_t RunGatherTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const GatherTrialOptions& options, TransferTrialMeasurement measurement,
    GatherTrialResult* out_result);

}  // namespace iree::net::cts

#endif  // IREE_NET_CTS_GATHER_TRIAL_H_
