// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/direct_transfer_trial.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/connection.h"
#include "iree/net/direct_endpoint.h"

namespace iree::net::cts {
namespace {

using Positions = std::array<uint32_t, 2>;
enum class Role { kProducer, kConsumer };

// Shared immutable page geometry, prepared outside the measured interval.
struct StorageLayout {
  struct Fragment {
    // Byte position in the logical record, independent of physical placement.
    size_t position;
    // Payload bytes in this fragment, excluding unused storage gaps.
    size_t length;
  };
  // Nonzero for permuted pages, including one unwritten gap byte per page.
  size_t page_stride;
  // Physical bytes reserved for one logical source or target record.
  size_t record_extent;
  // Logical partition reused by both application owners and all connections.
  std::vector<Fragment> fragments;

  StorageLayout(const DirectTransferTrialOptions& options, size_t page_stride,
                size_t record_extent)
      : page_stride(page_stride), record_extent(record_extent) {
    size_t position = 0;
    for (size_t i = 0; i < options.fragment_count; ++i) {
      size_t length =
          (options.record_size - position) / (options.fragment_count - i);
      fragments.push_back({position, length});
      position += length;
    }
  }
  size_t Offset(size_t index, uint32_t epoch, Role role) const {
    if (!page_stride) {
      return fragments[index].position;
    }
    size_t page = (index + epoch % fragments.size()) % fragments.size();
    if (role == Role::kConsumer) {
      page = fragments.size() - 1 - page;
    }
    return page * page_stride;
  }
};

// Only phase/error communication crosses owners. Transfer observations and
// source-slot retirement are private to the application's corresponding owner.
struct Control {
  enum Flag : uint32_t {
    kFailed = 1u << 0,
    kClosing = 1u << 1,
    kWarmupDrained = 1u << 2,
    kMeasuredDrained = 1u << 3,
  };
  // Borrowed executors remain alive through both owner joins.
  std::array<iree_async_proactor_t*, 2> proactors = {};
  // Cross-owner phase witnesses; no per-record atomic state.
  std::atomic<uint32_t> flags{0};
  // Serializes failure collection only.
  std::mutex mutex;
  // Joined failures returned after native and callback retirement.
  iree::Status error;

  bool Has(Flag flag) const {
    return (flags.load(std::memory_order_acquire) & flag) != 0;
  }
  void Publish(Flag flag) {
    flags.fetch_or(flag, std::memory_order_release);
    for (auto* proactor : proactors) {
      iree_async_proactor_wake(proactor);
    }
  }
  void Fail(iree_status_t status) {
    if (iree_status_is_ok(status)) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      error = iree_status_join(error.release(), status);
    }
    Publish(kFailed);
  }
  void EndpointError(iree_status_t status) {
    if (Has(kClosing) && (iree_status_is_cancelled(status) ||
                          iree_status_is_unavailable(status) ||
                          iree_status_is_out_of_range(status))) {
      iree_status_free(status);
    } else {
      Fail(status);
    }
  }
};

struct Peer {
  enum Flag : uint32_t {
    kConnectPending = 1u << 0,
    kMessageReady = 1u << 1,
    kDirectReady = 1u << 2,
    kTargetAvailable = 1u << 3,
    kGrantSent = 1u << 4,
    kDrained = 1u << 5,
  };
  enum class SlotState { kFree, kSourcePending, kArrived, kConsumed };
  struct Slot {
    // Stable callback owner, valid until connection retirement.
    Peer* peer = nullptr;
    // Application epoch occupying this source or target slot.
    uint32_t epoch = 0;
    // Producer source return or consumer placement/consumption state.
    SlotState state = SlotState::kFree;
  };

  // Immutable workload geometry shared with both application owners.
  const DirectTransferTrialOptions& options;
  // Shared logical partition and physical source/target page geometry.
  const StorageLayout& layout;
  // Cross-owner phase and terminal failure channel.
  Control& control;
  // Direction of this connection's application payload.
  Role role;
  // Local disjoint range in the side's shared registration.
  iree_async_span_t storage;
  // Local peer identity, sent in the target grant; accept order is immaterial.
  size_t identity;
  // Identity received with the target description, used in payload patterns.
  size_t target_identity = 0;
  // Caller-owned outbound operation, stable through its terminal callback.
  iree_net_transport_connect_operation_t connect_operation;
  // Owned connection, released only after deactivation.
  iree_net_connection_t* connection = nullptr;
  // Borrowed endpoints sharing the connection's native lifetime.
  struct {
    // Message endpoint carrying target grants and consumed frontiers.
    iree_net_message_endpoint_t message = {};
    // Direct endpoint carrying only registered payload placements.
    iree_net_direct_endpoint_t direct = {};
  } endpoints;
  // Queue protocol over the borrowed message endpoint.
  iree_net_queue_channel_t* channel = nullptr;
  // Imported target facts, not ownership or revocation state.
  iree_net_direct_target_t target = {};
  // Reusable serialized target description, produced once after setup.
  std::vector<uint8_t> description;
  // Temporary descriptors overwritten immediately after accepted submission.
  std::vector<iree_net_direct_write_entry_t> entries;
  // Bounded application source/target slots, not native completion tracking.
  std::array<std::vector<Slot>, 2> slots;
  // Owner-local connection and endpoint setup/retirement state.
  uint32_t flags = 0;
  // Highest admitted write or published consumed coordinate, by role.
  Positions submitted = {};
  // Consumer checked prefix or producer observed consumed prefix.
  Positions observed = {};
  // Consumer placement prefix, independent of consumption.
  Positions arrived = {};
  // Current producer phase limit.
  uint32_t goal = 0;
  // Fair next axis when endpoint admission is exhausted.
  size_t next_axis = 0;
  // Accepted logical writes, independent of native segmentation.
  uint64_t writes = 0;
  // Terminal source returns, independent of callback order.
  uint64_t source_completions = 0;
  // Accepted feedback/grant sends that must join before storage release.
  uint64_t message_sends = 0;
  // Terminal message source returns.
  uint64_t message_completions = 0;
  // Observed consumed-frontier messages.
  uint64_t progress_messages = 0;
  // Observed timeline 1 progress while timeline 0 remains behind.
  uint64_t independent_progress_messages = 0;
  // Largest admitted-minus-observed record count on one timeline.
  uint64_t window_high_water = 0;
  // Records checked after placement callback return.
  uint64_t retained_records = 0;

  Peer(const DirectTransferTrialOptions& options, const StorageLayout& layout,
       Control& control, Role role, iree_async_span_t storage, size_t identity)
      : options(options),
        layout(layout),
        control(control),
        role(role),
        storage(storage),
        identity(identity),
        entries(options.fragment_count) {
    iree_net_transport_connect_operation_initialize(&connect_operation);
    for (auto& axis : slots) {
      axis.resize(options.window_size);
      for (auto& slot : axis) {
        slot.peer = this;
      }
    }
  }
  ~Peer() {
    iree_net_queue_channel_free(channel);
    iree_net_connection_release(connection);
    iree_net_transport_connect_operation_deinitialize(&connect_operation);
  }
  bool Has(Flag flag) const { return (flags & flag) != 0; }
  size_t Offset(size_t axis, uint32_t epoch) const {
    return (axis * options.window_size + (epoch - 1) % options.window_size) *
           layout.record_extent;
  }
  Slot& Record(size_t axis, uint32_t epoch) {
    return slots[axis][(epoch - 1) % options.window_size];
  }
  uint8_t* Bytes(size_t axis, uint32_t epoch) {
    return iree_async_span_ptr(storage) + Offset(axis, epoch);
  }
  static uint8_t Pattern(size_t identity, size_t axis, uint32_t epoch,
                         size_t offset) {
    return static_cast<uint8_t>(identity * 71 + axis * 37 + epoch * 17 +
                                (epoch >> 8) * 13 + offset * 131 +
                                offset / 251);
  }
  iree_status_t Consume(size_t axis, uint32_t epoch) {
    const auto* bytes = Bytes(axis, epoch);
    for (size_t index = 0; index < layout.fragments.size(); ++index) {
      const auto& fragment = layout.fragments[index];
      size_t offset = layout.Offset(index, epoch, Role::kConsumer);
      for (size_t i = 0; i < fragment.length; ++i) {
        if (bytes[offset + i] !=
            Pattern(identity, axis, epoch, fragment.position + i)) {
          return iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "registered target record changed");
        }
      }
      if (layout.page_stride &&
          bytes[offset + layout.page_stride - 1] != 0xA5) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "registered target page gap changed");
      }
    }
    Record(axis, epoch).state = SlotState::kConsumed;
    while (observed[axis] < arrived[axis]) {
      auto& next = Record(axis, observed[axis] + 1);
      if (next.state != SlotState::kConsumed ||
          next.epoch != observed[axis] + 1) {
        break;
      }
      ++observed[axis];
    }
    return iree_ok_status();
  }
  static void OnError(void* user_data, iree_status_t status) {
    static_cast<Peer*>(user_data)->control.EndpointError(status);
  }
  static void OnMessageSent(void* user_data, iree_status_t status,
                            iree_host_size_t) {
    auto& peer = *static_cast<Peer*>(user_data);
    ++peer.message_completions;
    peer.control.Fail(status);
  }
  static void OnSourceReturned(void* user_data, iree_status_t status,
                               iree_host_size_t transferred) {
    auto& slot = *static_cast<Slot*>(user_data);
    auto& peer = *slot.peer;
    ++peer.source_completions;
    slot.state = SlotState::kFree;
    if (iree_status_is_ok(status) && transferred != peer.options.record_size) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "direct source completion byte count differs");
    }
    peer.control.EndpointError(status);
  }
  static iree_status_t OnGrant(void* user_data, uint32_t queue_id,
                               const iree_net_queue_frontier_view_t* waits,
                               const iree_net_queue_frontier_view_t* signals,
                               iree_const_byte_span_t payload,
                               iree_async_buffer_lease_t*) {
    auto& peer = *static_cast<Peer*>(user_data);
    if (peer.role != Role::kProducer || !peer.description.empty() ||
        queue_id != IREE_NET_QUEUE_ID_NONE || waits->count || signals->count ||
        payload.data_length < 8) {
      return iree_make_status(IREE_STATUS_DATA_LOSS, "unexpected target grant");
    }
    peer.target_identity = iree_unaligned_load_le_u64(payload.data);
    if (peer.target_identity >= peer.options.connection_count) {
      return iree_make_status(IREE_STATUS_DATA_LOSS, "target identity differs");
    }
    // Message and direct readiness are independent. Keep setup metadata until
    // this owner's direct-ready callback publishes the usable borrowed view.
    peer.description.assign(payload.data + 8,
                            payload.data + payload.data_length);
    return iree_ok_status();
  }
  static iree_status_t OnProgress(
      void* user_data, const iree_net_queue_frontier_view_t* frontier,
      iree_const_byte_span_t payload, iree_async_buffer_lease_t*) {
    auto& peer = *static_cast<Peer*>(user_data);
    if (peer.role != Role::kProducer || payload.data_length) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unexpected target progress");
    }
    for (uint8_t i = 0; i < frontier->count; ++i) {
      auto entry = iree_net_queue_frontier_view_get(frontier, i);
      if (entry.axis >= 2 || entry.epoch > peer.submitted[entry.axis]) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "unwitnessed consumed coordinate");
      }
      peer.observed[entry.axis] = std::max(peer.observed[entry.axis],
                                           static_cast<uint32_t>(entry.epoch));
    }
    ++peer.progress_messages;
    if (peer.observed[1] > peer.observed[0]) {
      ++peer.independent_progress_messages;
    }
    return iree_ok_status();
  }
  void PublishProgress() {
    if (observed == submitted ||
        !iree_net_queue_channel_query_send_budget(channel).slots) {
      return;
    }
    iree_net_queue_channel_send_params_t params = {};
    for (auto epoch : observed) {
      params.signal_frontier_count += epoch != 0;
    }
    params.build = +[](void* user_data,
                       const iree_net_queue_message_builder_t* builder) {
      auto& peer = *static_cast<Peer*>(user_data);
      size_t index = 0;
      for (size_t axis = 0; axis < 2; ++axis) {
        if (peer.observed[axis]) {
          iree_net_queue_frontier_builder_set(
              &builder->signal_frontier, index++, {axis, peer.observed[axis]});
        }
      }
      return iree_ok_status();
    };
    params.build_user_data = this;
    params.completion_callback = {OnMessageSent, this};
    iree_status_t status =
        iree_net_queue_channel_send_advance(channel, &params);
    if (!iree_status_is_ok(status)) {
      control.Fail(status);
      return;
    }
    submitted = observed;
    ++message_sends;
  }
  static iree_status_t OnPlacement(void* user_data, uint32_t cookie) {
    auto& peer = *static_cast<Peer*>(user_data);
    size_t axis = cookie & 1;
    uint32_t epoch = cookie >> 1;
    if (peer.role != Role::kConsumer || epoch <= peer.observed[axis] ||
        epoch - peer.observed[axis] > peer.options.window_size ||
        peer.Record(axis, epoch).epoch == epoch) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "placement escaped its application window");
    }
    auto& record = peer.Record(axis, epoch);
    record.epoch = epoch;
    record.state = SlotState::kArrived;
    while (peer.arrived[axis] - peer.observed[axis] <
               peer.options.window_size &&
           peer.Record(axis, peer.arrived[axis] + 1).epoch ==
               peer.arrived[axis] + 1) {
      ++peer.arrived[axis];
    }
    if (axis == 1 ||
        peer.options.consumer_mode == TransferConsumerMode::kInline) {
      IREE_RETURN_IF_ERROR(peer.Consume(axis, epoch));
      if (peer.options.progress_policy == TransferProgressPolicy::kImmediate) {
        peer.PublishProgress();
      }
    }
    return iree_ok_status();
  }
  void ConsumeRetained() {
    if (options.consumer_mode != TransferConsumerMode::kRetainedWindow) {
      return;
    }
    uint32_t phase_end =
        observed[0] < options.warmup_records
            ? options.warmup_records
            : options.warmup_records + options.measured_records;
    uint32_t end = observed[0] + std::min<uint32_t>(options.window_size,
                                                    phase_end - observed[0]);
    if (arrived[0] < end || submitted[1] < end) {
      return;
    }
    iree_status_t status = iree_ok_status();
    for (uint32_t epoch = observed[0] + 1;
         epoch <= end && iree_status_is_ok(status); ++epoch) {
      status = Consume(0, epoch);
      ++retained_records;
    }
    control.Fail(status);
  }
  void GrantTarget() {
    if (Has(kGrantSent)) {
      return;
    }
    if (description.empty()) {
      iree_host_size_t length = 0;
      iree_status_t status = iree_net_direct_endpoint_export_target(
          endpoints.direct, storage, IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE,
          iree_byte_span_empty(), &length);
      if (!iree_status_is_out_of_range(status)) {
        if (iree_status_is_ok(status)) {
          status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                    "empty target description is not usable");
        }
        control.Fail(status);
        return;
      }
      iree_status_free(status);
      description.resize(length);
      status = iree_net_direct_endpoint_export_target(
          endpoints.direct, storage, IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE,
          iree_make_byte_span(description.data(), description.size()), &length);
      if (!iree_status_is_ok(status)) {
        control.Fail(status);
        return;
      }
    }
    if (!iree_net_queue_channel_query_send_budget(channel).slots) {
      return;
    }
    iree_net_queue_channel_send_params_t params = {
        .generated_payload_length = 8 + description.size(),
        .build =
            +[](void* user_data,
                const iree_net_queue_message_builder_t* builder) {
              auto& peer = *static_cast<Peer*>(user_data);
              iree_unaligned_store_le_u64(builder->generated_payload.data,
                                          peer.identity);
              memcpy(builder->generated_payload.data + 8,
                     peer.description.data(), peer.description.size());
              return iree_ok_status();
            },
        .build_user_data = this,
        .completion_callback = {OnMessageSent, this}};
    iree_status_t status = iree_net_queue_channel_send_command(
        channel, IREE_NET_QUEUE_ID_NONE, &params);
    if (iree_status_is_ok(status)) {
      flags |= kGrantSent;
      ++message_sends;
    }
    control.Fail(status);
  }
  void Pump() {
    if (control.Has(Control::kFailed) || !Has(kMessageReady) ||
        !Has(kDirectReady)) {
      return;
    }
    if (role == Role::kConsumer) {
      GrantTarget();
      PublishProgress();
      ConsumeRetained();
      PublishProgress();
      return;
    }
    if (!Has(kTargetAvailable)) {
      if (description.empty()) {
        return;
      }
      iree_status_t status = iree_net_direct_endpoint_import_target(
          endpoints.direct,
          iree_make_const_byte_span(description.data(), description.size()),
          &target);
      if (iree_status_is_ok(status) && target.length != storage.length) {
        status =
            iree_make_status(IREE_STATUS_DATA_LOSS, "target geometry differs");
      }
      if (!iree_status_is_ok(status)) {
        control.Fail(status);
        return;
      }
      flags |= kTargetAvailable;
    }
    size_t stalled_axes = 0;
    while (
        stalled_axes < 2 && !control.Has(Control::kFailed) &&
        iree_net_direct_endpoint_query_write_budget(endpoints.direct).slots) {
      size_t axis = next_axis;
      next_axis ^= 1;
      uint32_t epoch = submitted[axis] + 1;
      auto& slot = Record(axis, epoch);
      if (epoch > goal || epoch - observed[axis] > options.window_size ||
          slot.state != SlotState::kFree) {
        ++stalled_axes;
        continue;
      }
      stalled_axes = 0;
      auto* bytes = Bytes(axis, epoch);
      for (size_t index = 0; index < entries.size(); ++index) {
        const auto& fragment = layout.fragments[index];
        size_t source_offset = layout.Offset(index, epoch, Role::kProducer);
        size_t target_offset = layout.Offset(index, epoch, Role::kConsumer);
        for (size_t i = 0; i < fragment.length; ++i) {
          bytes[source_offset + i] =
              Pattern(target_identity, axis, epoch, fragment.position + i);
        }
        entries[index] = {
            iree_async_span_make(
                storage.region,
                storage.offset + Offset(axis, epoch) + source_offset,
                fragment.length),
            &target, Offset(axis, epoch) + target_offset};
      }
      iree_net_direct_write_params_t params = {};
      params.flags = IREE_NET_DIRECT_WRITE_FLAG_NOTIFY;
      params.notification_cookie = (epoch << 1) | axis;
      params.entry_count = entries.size();
      params.entries = entries.data();
      params.completion_callback = {OnSourceReturned, &slot};
      iree_status_t status =
          iree_net_direct_endpoint_write(endpoints.direct, &params);
      if (iree_status_is_ok(status)) {
        slot.epoch = epoch;
        slot.state = SlotState::kSourcePending;
        submitted[axis] = epoch;
        ++writes;
        window_high_water = std::max<uint64_t>(
            window_high_water, submitted[axis] - observed[axis]);
        // Accepted operations captured these descriptors, not their addresses.
        std::fill(entries.begin(), entries.end(),
                  iree_net_direct_write_entry_t{});
      }
      control.Fail(status);
    }
  }
  static void OnMessageReady(void* user_data, iree_status_t status,
                             iree_net_message_endpoint_t endpoint) {
    auto& peer = *static_cast<Peer*>(user_data);
    if (iree_status_is_ok(status)) {
      peer.endpoints.message = endpoint;
      status = iree_net_queue_channel_allocate(
          endpoint, {OnGrant, OnProgress, OnError, &peer},
          iree_allocator_system(), &peer.channel);
    }
    if (iree_status_is_ok(status)) {
      iree_net_queue_channel_attach(peer.channel);
      status = iree_net_message_endpoint_activate(endpoint);
    }
    if (iree_status_is_ok(status)) {
      peer.flags |= kMessageReady;
    }
    peer.control.EndpointError(status);
  }
  static void OnDirectReady(void* user_data, iree_status_t status,
                            iree_net_direct_endpoint_t endpoint) {
    auto& peer = *static_cast<Peer*>(user_data);
    if (iree_status_is_ok(status)) {
      peer.endpoints.direct = endpoint;
      iree_net_direct_endpoint_set_callbacks(endpoint,
                                             {OnPlacement, OnError, &peer});
      status = iree_net_direct_endpoint_activate(endpoint);
    }
    if (iree_status_is_ok(status)) {
      peer.flags |= kDirectReady;
    }
    peer.control.EndpointError(status);
  }
  static void Connected(void* user_data, iree_status_t status,
                        iree_net_connection_t* connection) {
    auto& peer = *static_cast<Peer*>(user_data);
    peer.flags &= ~kConnectPending;
    peer.connection = connection;
    if (iree_status_is_ok(status) && !peer.control.Has(Control::kClosing)) {
      status = iree_net_connection_open_endpoint(connection,
                                                 {OnMessageReady, &peer});
      if (iree_status_is_ok(status)) {
        status = iree_net_connection_open_direct_endpoint(
            connection, {OnDirectReady, &peer});
      }
    }
    peer.control.EndpointError(status);
  }
  bool Done(uint32_t epoch) const {
    return observed == Positions{epoch, epoch} && submitted == observed &&
           writes == source_completions && message_sends == message_completions;
  }
};

struct Side {
  // Immutable application geometry.
  const DirectTransferTrialOptions& options;
  // Shared page partition and per-record storage extent.
  const StorageLayout& layout;
  // Shared phase/error channel, not data-path progress.
  Control& control;
  // Producer or consumer lifetime owner.
  Role role;
  // Owned executor, polled only by this side's application thread.
  iree_async_proactor_t* proactor = nullptr;
  // Owned final source/target storage, independent of connection retirement.
  iree_async_slab_t* slab = nullptr;
  // Explicit registration reused by every connection on this side.
  iree_async_region_t* region = nullptr;
  // Factory sharing the registration's native context.
  iree_net_transport_factory_t* factory = nullptr;
  // Consumer's bound admission owner.
  iree_net_listener_t* listener = nullptr;
  // Listener's final callback joined its unpublished handshakes.
  bool listener_stopped = false;
  // Stable per-connection application owners allocated before connect.
  std::vector<std::unique_ptr<Peer>> peers;
  // Consumer slots assigned without assuming connection request order.
  size_t accepted_count = 0;
  // Unexpected accepted connections still owing a local deactivation join.
  size_t pending_rejections = 0;

  Side(const DirectTransferTrialOptions& options, const StorageLayout& layout,
       Control& control, Role role)
      : options(options), layout(layout), control(control), role(role) {}
  ~Side() {
    peers.clear();
    iree_net_transport_factory_release(factory);
    iree_async_region_release(region);
    iree_async_slab_release(slab);
    iree_async_proactor_release(proactor);
  }
  iree_status_t Initialize(
      const TransportBackend& transport,
      const iree::async::cts::ProactorFactory& create_proactor) {
    auto proactor_options = iree_async_proactor_options_default();
    proactor_options.threading_mode =
        role == Role::kProducer ? IREE_ASYNC_PROACTOR_THREADING_SAME_THREAD
                                : IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
    auto created = create_proactor(proactor_options);
    if (!created.ok()) {
      return std::move(created).status().release();
    }
    proactor = *created;
    auto slab_options = iree_async_slab_options_default();
    slab_options.buffer_size = 2 * options.window_size * layout.record_extent;
    slab_options.buffer_count = options.connection_count;
    IREE_RETURN_IF_ERROR(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab));
    IREE_RETURN_IF_ERROR(transport.create_registered_factory(
        slab, iree_allocator_system(), &factory, &region));
    for (size_t i = 0; i < options.connection_count; ++i) {
      peers.push_back(std::make_unique<Peer>(
          options, layout, control, role,
          iree_async_span_make(region, i * slab_options.buffer_size,
                               slab_options.buffer_size),
          i));
      if (layout.page_stride) {
        memset(iree_async_span_ptr(peers.back()->storage), 0xA5,
               slab_options.buffer_size);
      }
    }
    return iree_ok_status();
  }
  static void Accepted(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto& side = *static_cast<Side*>(user_data);
    if (iree_status_is_ok(status) && side.accepted_count < side.peers.size()) {
      Peer::Connected(side.peers[side.accepted_count++].get(), status,
                      connection);
      return;
    }
    if (iree_status_is_ok(status)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "unexpected extra connection");
    }
    side.control.EndpointError(status);
    if (connection) {
      struct Rejection {
        // Poll owner retained through this independent connection join.
        Side* side;
        // Connection reference transferred by the unexpected accept callback.
        iree_net_connection_t* connection;
      };
      ++side.pending_rejections;
      auto* rejection = new Rejection{&side, connection};
      iree_net_connection_deactivate(
          connection, {+[](void* user_data) {
                         auto* value = static_cast<Rejection*>(user_data);
                         --value->side->pending_rejections;
                         iree_net_connection_release(value->connection);
                         delete value;
                       },
                       rejection});
    }
  }
  void Poll() {
    control.Fail(
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
  }
  bool Done(uint32_t epoch) const {
    return std::all_of(peers.begin(), peers.end(),
                       [&](const auto& peer) { return peer->Done(epoch); });
  }
  void RunProducer(uint32_t goal) {
    for (auto& peer : peers) {
      peer->goal = goal;
    }
    while (!Done(goal) && !control.Has(Control::kFailed)) {
      for (auto& peer : peers) {
        peer->Pump();
      }
      if (!Done(goal) && !control.Has(Control::kFailed)) {
        Poll();
      }
    }
  }
  void RunConsumer() {
    while (!control.Has(Control::kClosing) && !control.Has(Control::kFailed)) {
      for (auto& peer : peers) {
        peer->Pump();
      }
      if (!control.Has(Control::kWarmupDrained) &&
          Done(options.warmup_records)) {
        for (auto& peer : peers) {
          peer->retained_records = 0;
        }
        control.Publish(Control::kWarmupDrained);
      }
      if (!control.Has(Control::kMeasuredDrained) &&
          Done(options.warmup_records + options.measured_records)) {
        control.Publish(Control::kMeasuredDrained);
      }
      if (!control.Has(Control::kClosing) && !control.Has(Control::kFailed)) {
        Poll();
      }
    }
  }
  void Shutdown() {
    if (listener) {
      control.Fail(iree_net_listener_stop(
          listener, {+[](void* user_data) {
                       static_cast<Side*>(user_data)->listener_stopped = true;
                     },
                     this}));
      while (!listener_stopped) {
        Poll();
      }
      iree_net_listener_free(listener);
      listener = nullptr;
    }
    for (auto& peer : peers) {
      iree_net_transport_connect_operation_cancel(&peer->connect_operation);
    }
    while (std::any_of(peers.begin(), peers.end(), [](const auto& peer) {
      return peer->Has(Peer::kConnectPending);
    })) {
      Poll();
    }
    for (auto& peer : peers) {
      if (peer->connection) {
        iree_net_connection_deactivate(
            peer->connection, {+[](void* user_data) {
                                 static_cast<Peer*>(user_data)->flags |=
                                     Peer::kDrained;
                               },
                               peer.get()});
      }
    }
    while (pending_rejections ||
           std::any_of(peers.begin(), peers.end(), [](const auto& peer) {
             return peer->connection && !peer->Has(Peer::kDrained);
           })) {
      Poll();
    }
    iree_async_proactor_end_polling(proactor);
  }
  void Accumulate(DirectTransferTrialResult* result) const {
    for (const auto& peer : peers) {
      result->records += peer->writes;
      result->source_completions += peer->source_completions;
      result->progress_messages += peer->progress_messages;
      result->independent_progress_messages +=
          peer->independent_progress_messages;
      result->window_high_water =
          std::max(result->window_high_water, peer->window_high_water);
      result->retained_records += peer->retained_records;
    }
  }
};

}  // namespace

iree_status_t RunDirectTransferTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const DirectTransferTrialOptions& options,
    DirectTransferTrialResult* out_result,
    TransferTrialMeasurement measurement) {
  *out_result = {};
  if (!transport.create_registered_factory) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "transport has no registered-target CTS setup");
  }
  if (!options.connection_count || !options.record_size ||
      !options.window_size || !options.fragment_count ||
      options.fragment_count > options.record_size || !options.warmup_records ||
      !options.measured_records ||
      options.measured_records > (UINT32_MAX >> 1) ||
      options.warmup_records > (UINT32_MAX >> 1) - options.measured_records ||
      options.window_size > UINT32_MAX ||
      options.record_size >
          SIZE_MAX / 4 / options.window_size / options.connection_count ||
      options.record_size > UINT64_MAX / 2 / options.measured_records /
                                options.connection_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid registered-transfer dimensions");
  }
  size_t page_stride = 0;
  size_t record_extent = options.record_size;
  if (options.layout == DirectTransferLayout::kPermutedPages) {
    page_stride = options.record_size / options.fragment_count +
                  (options.record_size % options.fragment_count != 0) + 1;
    if (page_stride > SIZE_MAX / 4 / options.window_size /
                          options.connection_count / options.fragment_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "registered page storage is too large");
    }
    record_extent = page_stride * options.fragment_count;
  }
  StorageLayout layout(options, page_stride, record_extent);
  Control control;
  Side producer(options, layout, control, Role::kProducer);
  Side consumer(options, layout, control, Role::kConsumer);
  IREE_RETURN_IF_ERROR(producer.Initialize(transport, create_proactor));
  IREE_RETURN_IF_ERROR(consumer.Initialize(transport, create_proactor));
  out_result->available = true;
  control.proactors = {producer.proactor, consumer.proactor};
  std::string address;
  IREE_RETURN_IF_ERROR(transport.make_bind_address(&address));
  IREE_RETURN_IF_ERROR(iree_net_transport_factory_create_listener(
      consumer.factory, iree_make_cstring_view(address.c_str()),
      consumer.proactor, nullptr, {Side::Accepted, &consumer},
      iree_allocator_system(), &consumer.listener));
  std::array<char, 1024> address_storage;
  iree_string_view_t bound_address;
  iree_status_t status = iree_net_listener_query_bound_address(
      consumer.listener, address_storage.size(), address_storage.data(),
      &bound_address);
  for (size_t i = 0; i < producer.peers.size() && iree_status_is_ok(status);
       ++i) {
    auto& peer = *producer.peers[i];
    peer.flags |= Peer::kConnectPending;
    status = iree_net_transport_factory_connect(
        producer.factory, bound_address, producer.proactor, nullptr,
        {Peer::Connected, &peer}, &peer.connect_operation);
    if (!iree_status_is_ok(status)) {
      peer.flags &= ~Peer::kConnectPending;
    }
  }
  control.Fail(status);
  DirectTransferTrialResult consumer_result;
  std::thread consumer_thread([&] {
    consumer.RunConsumer();
    while (!control.Has(Control::kClosing)) {
      consumer.Poll();
    }
    consumer.Shutdown();
    consumer.Accumulate(&consumer_result);
  });
  producer.RunProducer(options.warmup_records);
  while (!control.Has(Control::kWarmupDrained) &&
         !control.Has(Control::kFailed)) {
    producer.Poll();
  }
  DirectTransferTrialResult warmup;
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
  DirectTransferTrialResult result;
  producer.Accumulate(&result);
  result.available = true;
  result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
  result.records -= warmup.records;
  result.source_completions -= warmup.source_completions;
  result.progress_messages -= warmup.progress_messages;
  result.independent_progress_messages -= warmup.independent_progress_messages;
  result.payload_bytes = result.records * options.record_size;
  result.payload_storage_bytes =
      4 * options.connection_count * options.window_size * layout.record_extent;
  while (!control.Has(Control::kMeasuredDrained) &&
         !control.Has(Control::kFailed)) {
    producer.Poll();
  }
  control.Publish(Control::kClosing);
  producer.Shutdown();
  consumer_thread.join();
  result.retained_records = consumer_result.retained_records;
  if (!control.Has(Control::kFailed)) {
    *out_result = result;
  }
  return control.error.release();
}

}  // namespace iree::net::cts
