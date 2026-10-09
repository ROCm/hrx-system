// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/transfer_trial.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/slab.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/session.h"

namespace iree::net::cts {
namespace {

constexpr size_t kTimelineCount = 2;
using Positions = std::array<uint64_t, kTimelineCount>;

// Cross-owner communication is limited to phase/error boundaries. Workload
// progress and every session callback remain private to their poll owner.
struct TrialControl {
  // Proactors borrowed until both owners finish cleanup.
  std::array<iree_async_proactor_t*, 2> proactors = {};
  // First failure closes workload admission on both owners.
  std::atomic<bool> failed{false};
  // Both owners are intentionally draining their sessions.
  std::atomic<bool> closing{false};
  // Consumer has joined every warm-up feedback send.
  std::atomic<bool> warmup_drained{false};
  // Consumer has joined every measured feedback send.
  std::atomic<bool> measured_drained{false};
  // Serializes error collection, never acquired on successful work.
  std::mutex error_mutex;
  // Owned failure status returned after both owners join.
  iree::Status error;

  void Wake() {
    for (auto* proactor : proactors) {
      if (proactor) {
        iree_async_proactor_wake(proactor);
      }
    }
  }

  void Fail(iree_status_t status) {
    if (iree_status_is_ok(status)) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      error = iree_status_join(error.release(), status);
    }
    failed.store(true, std::memory_order_release);
    Wake();
  }

  void EndpointError(iree_status_t status) {
    // Local cancellation and peer EOF are expected only after the workload has
    // joined or failed and both owners have entered deliberate shutdown.
    if (closing.load(std::memory_order_acquire) &&
        (iree_status_is_cancelled(status) ||
         iree_status_is_out_of_range(status) ||
         iree_status_is_unavailable(status))) {
      iree_status_free(status);
    } else {
      Fail(status);
    }
  }
};

enum class PeerRole { kProducer, kConsumer };

// Original receive views owned beyond the callback by a moved lease.
struct RetainedCommand {
  // Signal metadata reread when consuming the message, not copied on receipt.
  iree_net_queue_frontier_view_t signal_frontier;
  // Original bytes that must remain valid until consumption.
  iree_const_byte_span_t payload;
  // Moved endpoint storage ownership, returned by the receiving poll owner.
  iree_async_buffer_lease_t lease;
};

// A concrete checked-transfer application, not a scheduler or HAL emulator.
struct TransferPeer {
  // Shared immutable workload dimensions.
  const TransferTrialOptions& options;
  // Phase and failure communication with the other poll owner.
  TrialControl& control;
  // Direction of this endpoint's application work.
  PeerRole role;
  // Stable patterned source, also used as the consumer's validation oracle.
  std::vector<uint8_t> payload;
  // Reusable send descriptors; only payload bytes outlive a send call.
  std::vector<iree_async_span_t> spans;
  // Session reference retained through deactivation.
  iree_net_session_t* session = nullptr;
  // Channel borrowing the session's application endpoint.
  iree_net_queue_channel_t* channel = nullptr;
  // True only after all session and endpoint callbacks have joined.
  bool deactivated = false;
  // Last admitted COMMAND (producer) or ADVANCE (consumer) coordinates.
  Positions submitted = {};
  // Checked (consumer) or observed completed (producer) coordinates.
  Positions observed = {};
  // Target records per timeline in the current producer phase.
  uint64_t goal = 0;
  // Round-robin next producer timeline, preventing window starvation.
  size_t next_timeline = 0;
  // Accepted message count, independent of callback order.
  uint64_t sends = 0;
  // Terminal callbacks for accepted sends.
  uint64_t completions = 0;
  // Number of observed ADVANCE messages.
  uint64_t progress_messages = 0;
  // Maximum submitted-minus-observed records on any timeline.
  uint64_t window_high_water = 0;
  // Observations of independent progress while timeline 0 remains behind.
  uint64_t independent_progress_messages = 0;
  // Saved completed prefix waiting for publication after a newer snapshot.
  Positions saved_progress = {};
  // Older progress reports observed after a dominating snapshot.
  uint64_t saved_progress_messages = 0;
  // Bounded application ownership for deferred timeline-0 consumption.
  struct {
    // Descriptors reserved at setup for at most one record window.
    std::vector<RetainedCommand> commands;
    // Last received timeline-0 coordinate, not a completion witness.
    uint64_t received = 0;
    // Payload bytes currently held by the commands.
    uint64_t bytes = 0;
    // Records checked outside their receive callbacks in the measured phase.
    uint64_t records = 0;
    // Completed retained windows in the measured phase.
    uint64_t windows = 0;
    // Largest number of held messages in the measured phase.
    uint64_t messages_high_water = 0;
    // Largest retained payload footprint in the measured phase.
    uint64_t bytes_high_water = 0;
  } retained;

  TransferPeer(const TransferTrialOptions& options, TrialControl& control,
               PeerRole role)
      : options(options),
        control(control),
        role(role),
        payload(options.record_size * options.batch_size),
        spans(options.fragment_count) {
    for (size_t i = 0; i < payload.size(); ++i) {
      payload[i] = static_cast<uint8_t>((i * 131 + i / 251 + 17) & 0xFF);
    }
    if (role == PeerRole::kConsumer &&
        options.consumer_mode == TransferConsumerMode::kRetainedWindow) {
      retained.commands.reserve(options.window_size);
    }
  }

  ~TransferPeer() {
    ReleaseRetained();
    iree_net_queue_channel_free(channel);
    iree_net_session_release(session);
  }

  void ReleaseRetained() {
    for (auto& command : retained.commands) {
      iree_async_buffer_lease_release(&command.lease);
    }
    retained.commands.clear();
    retained.bytes = 0;
  }

  iree_status_t Consume(iree_async_frontier_entry_t entry,
                        iree_const_byte_span_t bytes) {
    const uint64_t count = bytes.data_length / options.record_size;
    if (entry.axis >= kTimelineCount ||
        entry.epoch != observed[entry.axis] + count ||
        memcmp(bytes.data, payload.data(), bytes.data_length) != 0) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "checked-transfer bytes or timeline differ");
    }
    observed[entry.axis] = entry.epoch;
    return iree_ok_status();
  }

  void ConsumeRetained() {
    if (retained.commands.empty() || saved_progress != Positions{}) {
      return;
    }
    const uint64_t phase_end =
        observed[0] < options.warmup_records
            ? options.warmup_records
            : options.warmup_records + options.measured_records;
    const uint64_t window_end =
        observed[0] +
        std::min<uint64_t>(options.window_size, phase_end - observed[0]);
    if (retained.received != window_end || submitted[1] < window_end) {
      return;
    }
    // Timeline 1's earlier ADVANCE is already captured by the transport. The
    // producer can observe that independent progress before this completion.
    iree_status_t status = iree_ok_status();
    for (size_t i = 0;
         i < retained.commands.size() && iree_status_is_ok(status); ++i) {
      const auto& command = retained.commands[i];
      auto entry =
          iree_net_queue_frontier_view_get(&command.signal_frontier, 0);
      if (entry.axis != 0) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "retained timeline metadata changed");
      } else {
        status = Consume(entry, command.payload);
      }
      if (iree_status_is_ok(status) && i == 0 && retained.commands.size() > 1 &&
          options.progress_order == TransferProgressOrder::kNewestThenSaved) {
        saved_progress = observed;
      }
    }
    if (iree_status_is_ok(status)) {
      retained.records += retained.bytes / options.record_size;
      ++retained.windows;
    }
    ReleaseRetained();
    control.Fail(status);
  }

  static void OnSend(void* user_data, iree_status_t status, iree_host_size_t) {
    auto& peer = *static_cast<TransferPeer*>(user_data);
    ++peer.completions;
    peer.control.Fail(status);
  }

  static void OnError(void* user_data, iree_status_t status) {
    static_cast<TransferPeer*>(user_data)->control.EndpointError(status);
  }

  bool SendProgress(const Positions& positions) {
    auto budget = iree_net_queue_channel_query_send_budget(channel);
    const uint8_t count =
        static_cast<uint8_t>((positions[0] != 0) + (positions[1] != 0));
    if (!budget.slots ||
        budget.bytes < IREE_NET_QUEUE_MESSAGE_HEADER_SIZE +
                           count * IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE) {
      return false;
    }
    struct Progress {
      // Owner retaining each accepted send through its terminal callback.
      TransferPeer* peer;
      // Completed snapshot copied synchronously by the admitted builder.
      const Positions* positions;
    } progress{this, &positions};
    iree_net_queue_channel_send_params_t params = {
        .signal_frontier_count = count,
        .build =
            +[](void* user_data,
                const iree_net_queue_message_builder_t* builder) {
              auto& progress = *static_cast<Progress*>(user_data);
              size_t index = 0;
              for (size_t axis = 0; axis < kTimelineCount; ++axis) {
                if ((*progress.positions)[axis]) {
                  iree_net_queue_frontier_builder_set(
                      &builder->signal_frontier, index++,
                      {axis, (*progress.positions)[axis]});
                }
              }
              ++progress.peer->sends;
              return iree_ok_status();
            },
        .build_user_data = &progress,
        .completion_callback = {OnSend, this}};
    iree_status_t status =
        iree_net_queue_channel_send_advance(channel, &params);
    const bool accepted = iree_status_is_ok(status);
    if (iree_status_is_resource_exhausted(status)) {
      // Advisory admission may race. A later source completion retries flush.
      iree_status_free(status);
    } else {
      control.Fail(status);
    }
    return accepted;
  }

  // Feedback captures coordinates, never receive storage. A saved older prefix
  // follows the full window that dominates it, even across admission pressure.
  void FlushProgress() {
    if (!channel) {
      return;
    }
    if (observed != submitted) {
      if (!SendProgress(observed)) {
        return;
      }
      submitted = observed;
    }
    if (saved_progress != Positions{} && SendProgress(saved_progress)) {
      saved_progress = {};
    }
  }

  void Pump() {
    if (!channel) {
      return;
    }
    if (role == PeerRole::kConsumer) {
      FlushProgress();
      if (!retained.commands.empty()) {
        ConsumeRetained();
        if (!control.failed.load(std::memory_order_acquire)) {
          FlushProgress();
        }
      }
      return;
    }
    size_t stalled_timelines = 0;
    while (stalled_timelines < kTimelineCount &&
           !control.failed.load(std::memory_order_acquire)) {
      const size_t axis = next_timeline;
      next_timeline = (next_timeline + 1) % kTimelineCount;
      const uint64_t count = std::min<uint64_t>(
          {options.batch_size, goal - submitted[axis],
           options.window_size - (submitted[axis] - observed[axis])});
      if (!count) {
        ++stalled_timelines;
        continue;
      }
      const size_t length = count * options.record_size;
      auto budget = iree_net_queue_channel_query_send_budget(channel);
      if (!budget.slots ||
          budget.bytes < IREE_NET_QUEUE_MESSAGE_HEADER_SIZE +
                             IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE + length) {
        return;
      }
      stalled_timelines = 0;
      for (size_t i = 0; i < spans.size(); ++i) {
        const size_t begin = length * i / spans.size();
        const size_t end = length * (i + 1) / spans.size();
        spans[i] =
            iree_async_span_from_ptr(payload.data() + begin, end - begin);
      }
      struct Command {
        // Owner whose admission state is updated synchronously by the builder.
        TransferPeer* peer;
        // Independent application timeline receiving this batch.
        size_t axis;
        // Number of records in the admitted payload.
        uint64_t count;
      } command{this, axis, count};
      iree_net_queue_channel_send_params_t params = {
          .signal_frontier_count = 1,
          .build =
              +[](void* user_data,
                  const iree_net_queue_message_builder_t* builder) {
                auto& command = *static_cast<Command*>(user_data);
                auto& peer = *command.peer;
                peer.submitted[command.axis] += command.count;
                iree_net_queue_frontier_builder_set(
                    &builder->signal_frontier, 0,
                    {command.axis, peer.submitted[command.axis]});
                ++peer.sends;
                peer.window_high_water = std::max(
                    peer.window_high_water,
                    peer.submitted[command.axis] - peer.observed[command.axis]);
                return iree_ok_status();
              },
          .build_user_data = &command,
          .payload = iree_async_span_list_make(spans.data(), spans.size()),
          .completion_callback = {OnSend, this}};
      iree_status_t status = iree_net_queue_channel_send_command(
          channel, IREE_NET_QUEUE_ID_NONE, &params);
      if (iree_status_is_resource_exhausted(status)) {
        iree_status_free(status);
        return;
      }
      control.Fail(status);
    }
  }

  static iree_status_t OnCommand(
      void* user_data, uint32_t queue_id,
      const iree_net_queue_frontier_view_t* wait_frontier,
      const iree_net_queue_frontier_view_t* signal_frontier,
      iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
    auto& peer = *static_cast<TransferPeer*>(user_data);
    if (peer.role != PeerRole::kConsumer ||
        queue_id != IREE_NET_QUEUE_ID_NONE || wait_frontier->count != 0 ||
        signal_frontier->count != 1) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unexpected checked-transfer command envelope");
    }
    auto entry = iree_net_queue_frontier_view_get(signal_frontier, 0);
    if (entry.axis >= kTimelineCount || payload.data_length == 0 ||
        payload.data_length > peer.payload.size() ||
        payload.data_length % peer.options.record_size != 0) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "invalid checked-transfer payload extent");
    }
    if (entry.axis == 0 &&
        peer.options.consumer_mode == TransferConsumerMode::kRetainedWindow) {
      auto& retained = peer.retained;
      const uint64_t count = payload.data_length / peer.options.record_size;
      if (entry.epoch != retained.received + count ||
          entry.epoch - peer.observed[0] > peer.options.window_size ||
          retained.commands.size() == peer.options.window_size) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "retained command exceeds its window");
      }
      retained.commands.push_back({*signal_frontier, payload, *lease});
      *lease = {};
      retained.received = entry.epoch;
      retained.bytes += payload.data_length;
      retained.messages_high_water = std::max<uint64_t>(
          retained.messages_high_water, retained.commands.size());
      retained.bytes_high_water =
          std::max(retained.bytes_high_water, retained.bytes);
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(peer.Consume(entry, payload));
    if (peer.options.progress_policy == TransferProgressPolicy::kImmediate) {
      peer.FlushProgress();
    }
    return iree_ok_status();
  }

  static iree_status_t OnAdvance(
      void* user_data, const iree_net_queue_frontier_view_t* signal_frontier,
      iree_const_byte_span_t payload, iree_async_buffer_lease_t*) {
    auto& peer = *static_cast<TransferPeer*>(user_data);
    if (peer.role != PeerRole::kProducer || payload.data_length != 0) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unexpected checked-transfer progress envelope");
    }
    bool has_older_coordinate = false;
    for (size_t i = 0; i < signal_frontier->count; ++i) {
      auto entry = iree_net_queue_frontier_view_get(signal_frontier, i);
      if (entry.axis >= kTimelineCount ||
          entry.epoch > peer.submitted[entry.axis]) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "progress exceeds submitted timeline");
      }
      has_older_coordinate |= entry.epoch < peer.observed[entry.axis];
      peer.observed[entry.axis] =
          std::max(peer.observed[entry.axis], entry.epoch);
    }
    ++peer.progress_messages;
    if (has_older_coordinate) {
      ++peer.saved_progress_messages;
    }
    if (peer.observed[1] > peer.observed[0]) {
      ++peer.independent_progress_messages;
    }
    return iree_ok_status();
  }

  static void OnEndpoint(void* user_data, iree_status_t status,
                         iree_net_message_endpoint_t endpoint) {
    auto& peer = *static_cast<TransferPeer*>(user_data);
    if (iree_status_is_ok(status)) {
      status = iree_net_queue_channel_allocate(
          endpoint, {OnCommand, OnAdvance, OnError, &peer},
          iree_allocator_system(), &peer.channel);
    }
    if (iree_status_is_ok(status)) {
      iree_net_queue_channel_attach(peer.channel);
      status = iree_net_message_endpoint_activate(endpoint);
    }
    peer.control.Fail(status);
  }

  iree_net_session_callbacks_t Callbacks() {
    return {
        +[](void* user_data, iree_net_session_t* session,
            const iree_net_bootstrap_peer_info_view_t*,
            iree_net_bootstrap_capabilities_t) {
          auto& peer = *static_cast<TransferPeer*>(user_data);
          peer.control.Fail(
              iree_net_session_open_endpoint(session, {OnEndpoint, &peer}));
        },
        +[](void*, iree_net_session_t*, iree_net_control_data_flags_t,
            iree_const_byte_span_t,
            iree_async_buffer_lease_t*) -> iree_status_t {
          return iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "unexpected trial control data");
        },
        +[](void* user_data, iree_net_session_t*, uint32_t) {
          static_cast<TransferPeer*>(user_data)->control.Fail(iree_make_status(
              IREE_STATUS_DATA_LOSS, "unexpected trial GOAWAY"));
        },
        +[](void* user_data, iree_net_session_t*, iree_status_t status) {
          OnError(user_data, status);
        },
        +[](void* user_data, iree_net_session_t*) {
          static_cast<TransferPeer*>(user_data)->deactivated = true;
        },
        this,
    };
  }

  bool Done(uint64_t target) const {
    // The fixed producer shape gives one saved prefix per multi-message
    // window. Join its observation too, not just the earlier dominating report.
    uint64_t saved_report_count = 0;
    if (role == PeerRole::kProducer &&
        options.progress_order == TransferProgressOrder::kNewestThenSaved) {
      auto count = [&](uint64_t records) {
        return (options.window_size > options.batch_size
                    ? records / options.window_size
                    : 0) +
               (records % options.window_size > options.batch_size ? 1 : 0);
      };
      saved_report_count = count(options.warmup_records);
      if (target > options.warmup_records) {
        saved_report_count += count(target - options.warmup_records);
      }
    }
    return channel && observed == Positions{target, target} &&
           submitted == observed && sends == completions &&
           retained.commands.empty() && saved_progress == Positions{} &&
           saved_progress_messages == saved_report_count;
  }
};

// Owns the real sessions and callback targets on one side of the trial.
struct TrialSide {
  // Workload dimensions shared with each peer.
  const TransferTrialOptions& options;
  // Cross-owner phase/error communication.
  TrialControl& control;
  // One proactor owned by this role throughout polling.
  iree_async_proactor_t* proactor = nullptr;
  // Bootstrap receive storage for factories using the generic pool.
  iree_async_slab_t* slab = nullptr;
  // Registered slab region, released on the poll owner before it exits.
  iree_async_region_t* region = nullptr;
  // Generic factory receive pool.
  iree_async_buffer_pool_t* pool = nullptr;
  // Consumer listener, absent on the producer.
  iree_net_listener_t* listener = nullptr;
  // Listener stop callback has joined all accepts.
  bool listener_stopped = false;
  // Stable peer storage allocated before connections begin.
  std::vector<std::unique_ptr<TransferPeer>> peers;
  // Next consumer slot assigned by an accept callback.
  size_t accepted_count = 0;
  // Unadopted accepted connections awaiting their deactivation callbacks.
  size_t pending_rejection_count = 0;

  TrialSide(const TransferTrialOptions& options, TrialControl& control,
            PeerRole role)
      : options(options), control(control) {
    for (size_t i = 0; i < options.connection_count; ++i) {
      peers.push_back(std::make_unique<TransferPeer>(options, control, role));
    }
  }

  ~TrialSide() {
    peers.clear();
    iree_net_listener_free(listener);
    ReleasePool();
    iree_async_proactor_release(proactor);
  }

  void ReleasePool() {
    iree_async_buffer_pool_release(pool);
    pool = nullptr;
    iree_async_region_release(region);
    region = nullptr;
    iree_async_slab_release(slab);
    slab = nullptr;
  }

  iree_status_t Initialize(
      const iree::async::cts::ProactorFactory& create_proactor,
      iree_async_proactor_threading_mode_t threading_mode) {
    auto proactor_options = iree_async_proactor_options_default();
    proactor_options.threading_mode = threading_mode;
    auto created = create_proactor(proactor_options);
    if (!created.ok()) {
      return std::move(created).status().release();
    }
    proactor = *created;
    iree_async_slab_options_t slab_options = {.buffer_size = 64 * 1024,
                                              .buffer_count = 16};
    IREE_RETURN_IF_ERROR(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab));
    IREE_RETURN_IF_ERROR(iree_async_proactor_register_slab(
        proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));
    return iree_async_buffer_pool_create(region, iree_allocator_system(),
                                         &pool);
  }

  static iree_net_session_options_t SessionOptions() {
    auto options = iree_net_session_options_default();
    options.local_peer.application_endpoint_count = 1;
    return options;
  }

  static void OnAccept(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto& side = *static_cast<TrialSide*>(user_data);
    if (iree_status_is_ok(status)) {
      if (side.accepted_count == side.peers.size()) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "unexpected extra trial connection");
      } else {
        auto& peer = *side.peers[side.accepted_count++];
        auto options = SessionOptions();
        status =
            iree_net_session_accept(connection, &options, peer.Callbacks(),
                                    iree_allocator_system(), &peer.session);
      }
    }
    const bool needs_deactivation = connection && !iree_status_is_ok(status);
    side.control.Fail(status);
    if (needs_deactivation) {
      struct Rejection {
        // Poll owner retained until this connection's deactivation joins.
        TrialSide* side;
        // Accepted connection reference transferred to this cleanup callback.
        iree_net_connection_t* connection;
      };
      auto* rejection = new Rejection{&side, connection};
      ++side.pending_rejection_count;
      iree_net_connection_deactivate(
          connection, {+[](void* user_data) {
                         auto* rejection = static_cast<Rejection*>(user_data);
                         iree_net_connection_release(rejection->connection);
                         --rejection->side->pending_rejection_count;
                         delete rejection;
                       },
                       rejection});
    } else {
      iree_net_connection_release(connection);
    }
  }

  void Poll() {
    control.Fail(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                          /*out_completed_count=*/nullptr));
  }

  bool Done(uint64_t target) const {
    return std::all_of(peers.begin(), peers.end(),
                       [&](const auto& peer) { return peer->Done(target); });
  }

  void RunProducer(uint64_t goal) {
    for (auto& peer : peers) {
      peer->goal = goal;
    }
    while (!Done(goal) && !control.failed.load(std::memory_order_acquire)) {
      for (auto& peer : peers) {
        peer->Pump();
      }
      if (!Done(goal) && !control.failed.load(std::memory_order_acquire)) {
        Poll();
      }
    }
  }

  void RunConsumer() {
    while (!control.closing.load(std::memory_order_acquire) &&
           !control.failed.load(std::memory_order_acquire)) {
      for (auto& peer : peers) {
        peer->Pump();
      }
      if (!control.warmup_drained.load(std::memory_order_relaxed) &&
          Done(options.warmup_records)) {
        for (auto& peer : peers) {
          peer->retained.records = 0;
          peer->retained.windows = 0;
          peer->retained.messages_high_water = 0;
          peer->retained.bytes_high_water = 0;
        }
        control.warmup_drained.store(true, std::memory_order_release);
        iree_async_proactor_wake(control.proactors[0]);
      }
      if (!control.measured_drained.load(std::memory_order_relaxed) &&
          Done(options.warmup_records + options.measured_records)) {
        control.measured_drained.store(true, std::memory_order_release);
        iree_async_proactor_wake(control.proactors[0]);
      }
      if (!control.closing.load(std::memory_order_acquire) &&
          !control.failed.load(std::memory_order_acquire)) {
        Poll();
      }
    }
  }

  void Shutdown() {
    if (listener) {
      control.Fail(iree_net_listener_stop(
          listener, {+[](void* user_data) {
                       static_cast<TrialSide*>(user_data)->listener_stopped =
                           true;
                     },
                     this}));
      // Accept callbacks can publish a session until this barrier joins them.
      while (!listener_stopped) {
        Poll();
      }
    }
    for (auto& peer : peers) {
      if (peer->session) {
        iree_net_session_deactivate(peer->session);
      }
    }
    while (pending_rejection_count ||
           !std::all_of(peers.begin(), peers.end(), [](const auto& peer) {
             return !peer->session || peer->deactivated;
           })) {
      Poll();
    }
    peers.clear();
    iree_net_listener_free(listener);
    listener = nullptr;
    ReleasePool();
    iree_async_proactor_end_polling(proactor);
  }

  void Accumulate(TransferTrialResult* result) const {
    for (const auto& peer : peers) {
      result->command_messages += peer->sends;
      result->source_completions += peer->completions;
      result->progress_messages += peer->progress_messages;
      result->independent_progress_messages +=
          peer->independent_progress_messages;
      result->saved_progress_messages += peer->saved_progress_messages;
      result->window_high_water =
          std::max(result->window_high_water, peer->window_high_water);
      result->retained.records += peer->retained.records;
      result->retained.windows += peer->retained.windows;
      result->retained.messages_high_water =
          std::max(result->retained.messages_high_water,
                   peer->retained.messages_high_water);
      result->retained.bytes_high_water = std::max(
          result->retained.bytes_high_water, peer->retained.bytes_high_water);
    }
  }
};

}  // namespace

iree_status_t RunTransferTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const TransferTrialOptions& options, TransferTrialResult* out_result,
    TransferTrialMeasurement measurement) {
  *out_result = {};
  if (!options.connection_count || !options.record_size ||
      !options.batch_size || !options.window_size || !options.fragment_count ||
      options.fragment_count > options.record_size || !options.warmup_records ||
      !options.measured_records ||
      options.batch_size > SIZE_MAX / options.record_size ||
      (options.progress_order == TransferProgressOrder::kNewestThenSaved &&
       options.consumer_mode != TransferConsumerMode::kRetainedWindow) ||
      options.measured_records > UINT64_MAX - options.warmup_records ||
      options.connection_count > UINT64_MAX / kTimelineCount /
                                     options.measured_records /
                                     options.record_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid checked-transfer trial dimensions");
  }
  TrialControl control;
  TrialSide producer(options, control, PeerRole::kProducer);
  TrialSide consumer(options, control, PeerRole::kConsumer);
  IREE_RETURN_IF_ERROR(producer.Initialize(
      create_proactor, IREE_ASYNC_PROACTOR_THREADING_SAME_THREAD));
  IREE_RETURN_IF_ERROR(consumer.Initialize(
      create_proactor, IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD));
  control.proactors = {producer.proactor, consumer.proactor};

  iree_net_transport_factory_t* factory_ptr = nullptr;
  IREE_RETURN_IF_ERROR(
      transport.create_factory(iree_allocator_system(), &factory_ptr));
  std::unique_ptr<iree_net_transport_factory_t,
                  decltype(&iree_net_transport_factory_release)>
      factory(factory_ptr, iree_net_transport_factory_release);
  std::string address;
  IREE_RETURN_IF_ERROR(transport.make_bind_address(&address));
  IREE_RETURN_IF_ERROR(iree_net_transport_factory_create_listener(
      factory.get(), iree_make_string_view(address.data(), address.size()),
      consumer.proactor, consumer.pool, {TrialSide::OnAccept, &consumer},
      iree_allocator_system(), &consumer.listener));
  out_result->available = true;
  std::array<char, 1024> address_storage = {};
  iree_string_view_t bound_address = iree_string_view_empty();
  iree_status_t status = iree_net_listener_query_bound_address(
      consumer.listener, address_storage.size(), address_storage.data(),
      &bound_address);
  const auto session_options = TrialSide::SessionOptions();
  for (auto& peer : producer.peers) {
    if (iree_status_is_ok(status)) {
      status = iree_net_session_connect(
          factory.get(), bound_address, producer.proactor, producer.pool,
          &session_options, peer->Callbacks(), iree_allocator_system(),
          &peer->session);
    }
  }
  control.Fail(status);
  TransferTrialResult consumer_result;
  std::thread consumer_thread([&] {
    consumer.RunConsumer();
    // A failure wakes the producer, which publishes closing before both owners
    // retire callbacks. The consumer must not exit its poll-owner lifetime yet.
    while (!control.closing.load(std::memory_order_acquire)) {
      consumer.Poll();
    }
    consumer.Accumulate(&consumer_result);
    consumer.Shutdown();
  });

  producer.RunProducer(options.warmup_records);
  while (!control.warmup_drained.load(std::memory_order_acquire) &&
         !control.failed.load(std::memory_order_acquire)) {
    producer.Poll();
  }
  TransferTrialResult warmup;
  producer.Accumulate(&warmup);
  for (auto& peer : producer.peers) {
    peer->window_high_water = 0;
  }
  if (measurement.begin) {
    measurement.begin(measurement.user_data);
  }
  const auto start = std::chrono::steady_clock::now();
  producer.RunProducer(options.warmup_records + options.measured_records);
  const auto end = std::chrono::steady_clock::now();
  if (measurement.end) {
    measurement.end(measurement.user_data);
  }
  TransferTrialResult result;
  result.available = true;
  producer.Accumulate(&result);
  result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
  result.command_messages -= warmup.command_messages;
  result.source_completions -= warmup.source_completions;
  result.progress_messages -= warmup.progress_messages;
  result.independent_progress_messages -= warmup.independent_progress_messages;
  result.saved_progress_messages -= warmup.saved_progress_messages;
  result.records =
      options.measured_records * kTimelineCount * options.connection_count;
  result.payload_bytes = result.records * options.record_size;
  result.proactor_capabilities =
      iree_async_proactor_query_capabilities(producer.proactor);
  while (!control.measured_drained.load(std::memory_order_acquire) &&
         !control.failed.load(std::memory_order_acquire)) {
    producer.Poll();
  }

  control.closing.store(true, std::memory_order_release);
  control.Wake();
  producer.Shutdown();
  consumer_thread.join();
  result.retained = consumer_result.retained;
  if (!control.failed.load(std::memory_order_acquire)) {
    *out_result = result;
  }
  return control.error.release();
}

}  // namespace iree::net::cts
