// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CTS_TRANSFER_TRIAL_H_
#define IREE_NET_CTS_TRANSFER_TRIAL_H_

#include "iree/async/cts/util/registry.h"
#include "iree/net/cts/transport_backend.h"

namespace iree::net::cts {

// Application reporting policy, independent of native completion batching.
enum class TransferProgressPolicy {
  // Report checked records immediately from the receive callback when admitted.
  kImmediate,
  // Merge checked positions and flush before the next blocking poll.
  kPollTurn,
};

// Application receive ownership, independent of the transport's storage choice.
enum class TransferConsumerMode {
  // Check bytes within each receive callback.
  kInline,
  // Retain timeline 0 until timeline 1 has reported a whole window's progress.
  kRetainedWindow,
};

// Publication order of valid completed snapshots, not transport delivery order.
enum class TransferProgressOrder {
  // Report the latest completed snapshot without retaining an older one.
  kMonotonic,
  // Report a retained window before its saved first-message completed prefix.
  kNewestThenSaved,
};

// Fixed workload shared by transport correctness tests and benchmarks.
struct TransferTrialOptions {
  // Connections sharing one producer proactor and one consumer proactor.
  size_t connection_count = 1;
  // Checked bytes per record, excluding the queue envelope.
  size_t record_size = 64;
  // Maximum records of one timeline in each COMMAND message.
  size_t batch_size = 1;
  // Maximum submitted but unobserved records per independent timeline.
  size_t window_size = 32;
  // Nonempty borrowed SG fragments per COMMAND payload.
  size_t fragment_count = 1;
  // Records per timeline and connection completed before measurement.
  uint64_t warmup_records = 32;
  // Records per timeline and connection in the measured interval.
  uint64_t measured_records = 1024;
  // Application policy for sending completed-frontier ADVANCE messages.
  TransferProgressPolicy progress_policy = TransferProgressPolicy::kPollTurn;
  // Whether one consumer retains original receive storage across callbacks.
  TransferConsumerMode consumer_mode = TransferConsumerMode::kInline;
  // Saved-prefix publication requires retained-window consumption.
  TransferProgressOrder progress_order = TransferProgressOrder::kMonotonic;
};

// Measurements of the fixed workload, excluding setup, warm-up and teardown.
struct TransferTrialResult {
  // The requested stack created a listener; later errors are trial failures.
  bool available = false;
  // End-to-end interval through progress observation and producer source join.
  double elapsed_seconds = 0;
  // Total checked records across both timelines and all connections.
  uint64_t records = 0;
  // Total checked payload bytes, excluding protocol envelopes and feedback.
  uint64_t payload_bytes = 0;
  // Accepted producer COMMAND messages.
  uint64_t command_messages = 0;
  // Producer terminal source callbacks; must equal command_messages.
  uint64_t source_completions = 0;
  // Completed-frontier ADVANCE messages observed by the producer.
  uint64_t progress_messages = 0;
  // Maximum unobserved records on any one timeline during measurement.
  uint64_t window_high_water = 0;
  // Progress reports showing timeline 1 ahead of timeline 0.
  uint64_t independent_progress_messages = 0;
  // Older completed snapshots observed after a dominating frontier.
  uint64_t saved_progress_messages = 0;
  // Receive ownership measurements, excluding warm-up.
  struct {
    // Records checked through original views after their callbacks returned.
    uint64_t records = 0;
    // Windows consumed only after independent progress was published.
    uint64_t windows = 0;
    // Largest number of simultaneously held messages on any connection.
    uint64_t messages_high_water = 0;
    // Largest sum of held payload bytes on any connection, excluding framing.
    uint64_t bytes_high_water = 0;
  } retained;
  // Actual producer proactor capabilities after the named backend mask.
  iree_async_proactor_capabilities_t proactor_capabilities = 0;
};

// Optional outer measurement hooks, called on the producer thread. A benchmark
// uses these to exclude setup/warm-up/teardown from process CPU accounting too.
// Both hooks are supplied together or both are null. They run once each around
// the measured phase, never inside message callbacks or per-record work.
struct TransferTrialMeasurement {
  // Begins outer measurement after both warm-up ownership joins.
  void (*begin)(void* user_data) = nullptr;
  // Ends outer measurement after producer progress and source callbacks join.
  void (*end)(void* user_data) = nullptr;
  // Borrowed context valid until RunTransferTrial returns.
  void* user_data = nullptr;
};

// Runs a checked-transfer trial using real sessions and queue channels.
//
// Two application timelines per connection advance independently after payload
// validation. Each timeline is consumed in message order; that application
// contract establishes its completed prefix, not queue submission order.
// Retained-window consumption moves timeline 0's original views and leases out
// of the receive callback until a window arrives and timeline 1's covering
// progress report is admitted. The poll owner then checks and releases the
// retained messages. Phase tails release without waiting for unsubmitted work.
// Reporting coalesces only witnessed coordinates.
// Newest-then-saved publication retains the first-message completed prefix of
// each multi-message window, publishing it only after the full window frontier.
// Each phase also joins those late observations before completing.
//
// The producer runs on the caller, the consumer on a dedicated test-application
// thread. Each owns its proactor throughout polling and cleanup. Borrowed SG
// sources are immutable until every accepted send completes. Timing ends only
// after producer observation and source retirement; consumer send callbacks and
// both session deactivations are also joined before returning. Errors drain
// accepted work and are returned instead of producing a successful timing row.
//
// This same-process host-memory trial does not qualify process isolation,
// registered device memory, native execution or DMA visibility.
iree_status_t RunTransferTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const TransferTrialOptions& options, TransferTrialMeasurement measurement,
    TransferTrialResult* out_result);

}  // namespace iree::net::cts

#endif  // IREE_NET_CTS_TRANSFER_TRIAL_H_
