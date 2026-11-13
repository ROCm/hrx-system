// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CTS_DIRECT_TRANSFER_TRIAL_H_
#define IREE_NET_CTS_DIRECT_TRANSFER_TRIAL_H_

#include "iree/net/cts/transfer_trial.h"

namespace iree::net::cts {

// Physical layout of the same logical payload and descriptor batch.
enum class DirectTransferLayout { kContiguous, kPermutedPages };

// Fixed registered-placement application shared by CTS and benchmarks.
struct DirectTransferTrialOptions {
  // Connections sharing one registration and poll owner on each side.
  size_t connection_count = 1;
  // Final target bytes checked for each logical write.
  size_t record_size = 256;
  // Reusable source/target slots per independent application timeline.
  size_t window_size = 32;
  // Registered fragments per logical write; descriptors are transient.
  size_t fragment_count = 1;
  // Contiguous slices or independently remapped source and destination pages.
  DirectTransferLayout layout = DirectTransferLayout::kContiguous;
  // Records per connection and timeline before measurement.
  uint32_t warmup_records = 7;
  // Records per connection and timeline during measurement.
  uint32_t measured_records = 257;
  // Whether timeline 0 waits for independent timeline 1 progress.
  TransferConsumerMode consumer_mode = TransferConsumerMode::kInline;
  // Reporting policy for witnessed consumed coordinates.
  TransferProgressPolicy progress_policy = TransferProgressPolicy::kPollTurn;
};

struct DirectTransferTrialResult {
  // Registered setup succeeded; later failures cannot become skips.
  bool available = false;
  // Submission through consumer progress observation and all source returns.
  double elapsed_seconds = 0;
  // Records checked in final registered targets during measurement.
  uint64_t records = 0;
  // Checked payload bytes, excluding feedback and target descriptions.
  uint64_t payload_bytes = 0;
  // Exactly-once source callbacks, independently joined from peer progress.
  uint64_t source_completions = 0;
  // Consumed-frontier messages observed during measurement.
  uint64_t progress_messages = 0;
  // Observations of timeline 1 ahead while timeline 0 retains its targets.
  uint64_t independent_progress_messages = 0;
  // Maximum unobserved records on any one timeline.
  uint64_t window_high_water = 0;
  // Registered targets checked after their placement callback returned.
  uint64_t retained_records = 0;
  // Source and target ring storage, including unused page gaps.
  uint64_t payload_storage_bytes = 0;
};

// Runs two application-owned poll threads with explicit, reusable
// registrations. Each connection exchanges target descriptions through a queue
// channel and writes sequence-varying payload directly into its final peer
// slots. Placement notification, source return and consumer permission are
// independent. Slot reuse joins source return with covering consumer progress;
// callback order is never interpreted as a completed prefix. Retained
// consumption waits for a full window or phase tail and publication of the
// other timeline's progress.
//
// Permuted-page layout rotates source pages and reverses their destination
// positions each round. Each physical page has a checked, unwritten gap byte.
// Payload generation and checking use logical offsets independent of physical
// placement. This models paged-cache scatter/gather without copying through a
// contiguous staging buffer or changing registration on reuse.
//
// Setup, registration and teardown are outside measurement. Payload generation,
// checking and feedback are inside. All accepted callbacks and native accesses
// join before storage release, including on error. This same-process host
// witness does not qualify GPU mappings or device-initiated progress.
iree_status_t RunDirectTransferTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const DirectTransferTrialOptions& options,
    TransferTrialMeasurement measurement,
    DirectTransferTrialResult* out_result);

}  // namespace iree::net::cts

#endif  // IREE_NET_CTS_DIRECT_TRANSFER_TRIAL_H_
