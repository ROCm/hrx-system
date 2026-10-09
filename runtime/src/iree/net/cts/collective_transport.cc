// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/collective_transport.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include "iree/base/alignment.h"

namespace iree::net::cts {

void CollectiveControl::Wake() {
  for (auto* proactor : proactors) {
    iree_async_proactor_wake(proactor);
  }
}

void CollectiveControl::Fail(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    error = iree_status_join(error.release(), status);
    failed.store(true, std::memory_order_release);
  }
  condition.notify_all();
  Wake();
}

void CollectiveControl::EndpointError(iree_status_t status) {
  if (closing.load(std::memory_order_acquire) &&
      (iree_status_is_cancelled(status) || iree_status_is_unavailable(status) ||
       iree_status_is_out_of_range(status))) {
    // All ranks are retiring; these are terminal peer-close receipts.
    iree_status_free(status);
  } else {
    Fail(status);
  }
}

void CollectiveControl::Arrive() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++arrivals;
  }
  condition.notify_one();
}

void CollectiveControl::Wait() {
  std::unique_lock<std::mutex> lock(mutex);
  condition.wait(lock, [&] {
    return arrivals == proactors.size() ||
           failed.load(std::memory_order_acquire);
  });
}

void CollectiveControl::Start(uint32_t value) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    arrivals = 0;
    phase.store(value, std::memory_order_release);
  }
  Wake();
}

CollectiveLink::CollectiveLink(const CollectiveLinkOptions& options,
                               CollectiveControl& control, Direction direction,
                               LinkPurpose purpose, iree_async_span_t storage)
    : options(options),
      control(control),
      direction(direction),
      purpose(purpose),
      storage(storage),
      sources(purpose == LinkPurpose::kData && direction == Direction::kOutgoing
                  ? options.window_size
                  : 0),
      inputs(purpose == LinkPurpose::kData && direction == Direction::kIncoming
                 ? storage.length / options.block_size
                 : 0) {
  iree_net_transport_connect_operation_initialize(&connect_operation);
  for (auto& source : sources) {
    source.link = this;
  }
}

CollectiveLink::~CollectiveLink() {
  for (auto& input : inputs) {
    iree_async_buffer_lease_release(&input.lease);
  }
  iree_net_queue_channel_free(channel);
  iree_net_connection_release(connection);
  iree_net_transport_connect_operation_deinitialize(&connect_operation);
}

bool CollectiveLink::Has(Flag flag) const { return (flags & flag) != 0; }

bool CollectiveLink::Registered() const {
  return purpose == LinkPurpose::kData &&
         options.delivery == CollectiveDelivery::kRegistered;
}

bool CollectiveLink::Ready() const {
  return Has(kMessageReady) &&
         (!Registered() ||
          (Has(kDirectReady) &&
           Has(direction == Direction::kIncoming ? kGrantSent : kTargetReady)));
}

void CollectiveLink::OnError(void* value, iree_status_t status) {
  static_cast<CollectiveLink*>(value)->control.EndpointError(status);
}

void CollectiveLink::OnControlSent(void* value, iree_status_t status, size_t) {
  auto& link = *static_cast<CollectiveLink*>(value);
  ++link.control_completions;
  link.control.EndpointError(status);
}

void CollectiveLink::OnSourceReturned(void* value, iree_status_t status,
                                      size_t transferred) {
  auto& source = *static_cast<Source*>(value);
  source.pending = false;
  if (source.pending_sources) {
    --*source.pending_sources;
  }
  ++source.link->completions;
  if (iree_status_is_ok(status) && transferred != source.expected_length) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "collective source length differs");
  }
  source.link->control.EndpointError(status);
}

iree_status_t CollectiveLink::Receive(uint32_t sequence,
                                      iree_const_byte_span_t bytes,
                                      iree_async_buffer_lease_t* lease) {
  if (!sequence || sequence <= consumed ||
      sequence - consumed > inputs.size()) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "collective input exceeds its reuse window");
  }
  auto& input = inputs[(sequence - 1) % inputs.size()];
  if (input.sequence) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "collective input overwrote an unconsumed slot");
  }
  input.sequence = sequence;
  if (lease && options.input_mode == CollectiveInputMode::kOwned) {
    // Whole-activation readiness cannot retain the transport's entire receive
    // pool. Stage message inputs in the same bounded layout as direct writes.
    uint8_t* target = iree_async_span_ptr(storage) +
                      ((sequence - 1) % inputs.size()) * options.block_size;
    memcpy(target, bytes.data, bytes.data_length);
    input.bytes = iree_make_const_byte_span(target, bytes.data_length);
  } else {
    input.bytes = bytes;
  }
  if (lease && options.input_mode == CollectiveInputMode::kLeased) {
    input.lease = *lease;
    memset(lease, 0, sizeof(*lease));
  }
  return iree_ok_status();
}

iree_status_t CollectiveLink::OnCommand(
    void* value, uint32_t queue_id, const iree_net_queue_frontier_view_t* waits,
    const iree_net_queue_frontier_view_t* signals,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  auto& link = *static_cast<CollectiveLink*>(value);
  if (queue_id == 0 && link.direction == Direction::kOutgoing &&
      link.Registered() && link.description.empty() && !waits->count &&
      !signals->count) {
    link.description.assign(payload.data, payload.data + payload.data_length);
    return iree_ok_status();
  }
  if (queue_id != 1 || link.purpose != LinkPurpose::kData ||
      link.direction != Direction::kIncoming || link.Registered() ||
      waits->count || signals->count != 1 ||
      payload.data_length > link.options.block_size) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "unexpected collective data message");
  }
  auto coordinate = iree_net_queue_frontier_view_get(signals, 0);
  if (coordinate.axis != 1 || coordinate.epoch > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "unexpected collective block coordinate");
  }
  return link.Receive(static_cast<uint32_t>(coordinate.epoch), payload, lease);
}

iree_status_t CollectiveLink::OnAdvance(
    void* value, const iree_net_queue_frontier_view_t* signals,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t*) {
  auto& link = *static_cast<CollectiveLink*>(value);
  if (link.purpose == LinkPurpose::kResult &&
      link.direction == Direction::kIncoming && signals->count == 1 &&
      !payload.data_length) {
    auto coordinate = iree_net_queue_frontier_view_get(signals, 0);
    if (coordinate.axis != 2 || coordinate.epoch > link.options.result_limit) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unexpected pipeline result coordinate");
    }
    link.result_coordinate = std::max(link.result_coordinate,
                                      static_cast<uint32_t>(coordinate.epoch));
    return iree_ok_status();
  }
  if (link.purpose != LinkPurpose::kData ||
      link.direction != Direction::kOutgoing || signals->count != 1 ||
      payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "unexpected collective consumed message");
  }
  auto coordinate = iree_net_queue_frontier_view_get(signals, 0);
  if (coordinate.axis != 1 || coordinate.epoch > link.submitted) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "collective consumption exceeds submission");
  }
  link.remote_consumed =
      std::max(link.remote_consumed, static_cast<uint32_t>(coordinate.epoch));
  return iree_ok_status();
}

iree_status_t CollectiveLink::OnPlacement(void* value, uint32_t sequence) {
  auto& link = *static_cast<CollectiveLink*>(value);
  if (link.direction != Direction::kIncoming || !sequence) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "unexpected collective placement");
  }
  size_t offset =
      ((sequence - 1) % link.inputs.size()) * link.options.block_size;
  return link.Receive(
      sequence,
      iree_make_const_byte_span(iree_async_span_ptr(link.storage) + offset,
                                link.options.block_size),
      nullptr);
}

void CollectiveLink::OnMessageReady(void* value, iree_status_t status,
                                    iree_net_message_endpoint_t endpoint) {
  auto& link = *static_cast<CollectiveLink*>(value);
  if (iree_status_is_ok(status)) {
    link.message = endpoint;
    status = iree_net_queue_channel_allocate(
        endpoint, {OnCommand, OnAdvance, OnError, &link},
        iree_allocator_system(), &link.channel);
  }
  if (iree_status_is_ok(status)) {
    iree_net_queue_channel_attach(link.channel);
    status = iree_net_message_endpoint_activate(endpoint);
  }
  if (iree_status_is_ok(status)) {
    link.flags |= kMessageReady;
  }
  link.control.EndpointError(status);
}

void CollectiveLink::OnDirectReady(void* value, iree_status_t status,
                                   iree_net_direct_endpoint_t endpoint) {
  auto& link = *static_cast<CollectiveLink*>(value);
  if (iree_status_is_ok(status)) {
    link.direct = endpoint;
    iree_net_direct_endpoint_set_callbacks(endpoint,
                                           {OnPlacement, OnError, &link});
    status = iree_net_direct_endpoint_activate(endpoint);
  }
  if (iree_status_is_ok(status)) {
    link.flags |= kDirectReady;
  }
  link.control.EndpointError(status);
}

void CollectiveLink::Connected(void* value, iree_status_t status,
                               iree_net_connection_t* connection) {
  auto& link = *static_cast<CollectiveLink*>(value);
  link.flags &= ~kConnecting;
  link.connection = connection;
  if (iree_status_is_ok(status) &&
      !link.control.closing.load(std::memory_order_acquire)) {
    status =
        iree_net_connection_open_endpoint(connection, {OnMessageReady, &link});
    if (iree_status_is_ok(status) && link.Registered()) {
      status = iree_net_connection_open_direct_endpoint(connection,
                                                        {OnDirectReady, &link});
    }
  }
  link.control.EndpointError(status);
}

void CollectiveLink::Pump() {
  if (!Has(kMessageReady)) {
    return;
  }
  if (Registered() && Has(kDirectReady)) {
    if (direction == Direction::kOutgoing && !Has(kTargetReady) &&
        !description.empty()) {
      iree_status_t status = iree_net_direct_endpoint_import_target(
          direct,
          iree_make_const_byte_span(description.data(), description.size()),
          &target);
      if (iree_status_is_ok(status) && target.length != storage.length) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "collective target geometry differs");
      }
      if (iree_status_is_ok(status)) {
        flags |= kTargetReady;
      }
      control.Fail(status);
    } else if (direction == Direction::kIncoming && !Has(kGrantSent) &&
               iree_net_queue_channel_query_send_budget(channel).slots) {
      if (description.empty()) {
        std::array<uint8_t, 256> data;
        size_t length = 0;
        iree_status_t status = iree_net_direct_endpoint_export_target(
            direct, storage, IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE,
            iree_make_byte_span(data.data(), data.size()), &length);
        if (!iree_status_is_ok(status)) {
          control.Fail(status);
          return;
        }
        description.assign(data.data(), data.data() + length);
      }
      iree_net_queue_channel_send_params_t params = {
          .generated_payload_length = description.size(),
          .build =
              +[](void* value,
                  const iree_net_queue_message_builder_t* builder) {
                auto& link = *static_cast<CollectiveLink*>(value);
                memcpy(builder->generated_payload.data, link.description.data(),
                       link.description.size());
                return iree_ok_status();
              },
          .build_user_data = this,
          .completion_callback = {OnControlSent, this}};
      iree_status_t status =
          iree_net_queue_channel_send_command(channel, 0, &params);
      if (iree_status_is_ok(status)) {
        flags |= kGrantSent;
        ++control_sends;
      }
      control.Fail(status);
    }
  }
  bool publishes = purpose == LinkPurpose::kResult
                       ? direction == Direction::kOutgoing
                       : direction == Direction::kIncoming;
  uint32_t progress =
      purpose == LinkPurpose::kResult ? result_coordinate : consumed;
  if (publishes && progress != published &&
      iree_net_queue_channel_query_send_budget(channel).slots) {
    iree_net_queue_channel_send_params_t params = {
        .signal_frontier_count = 1,
        .build =
            +[](void* value, const iree_net_queue_message_builder_t* builder) {
              auto& link = *static_cast<CollectiveLink*>(value);
              iree_net_queue_frontier_builder_set(
                  &builder->signal_frontier, 0,
                  link.purpose == LinkPurpose::kResult
                      ? iree_async_frontier_entry_t{2, link.result_coordinate}
                      : iree_async_frontier_entry_t{1, link.consumed});
              return iree_ok_status();
            },
        .build_user_data = this,
        .completion_callback = {OnControlSent, this}};
    iree_status_t status =
        iree_net_queue_channel_send_advance(channel, &params);
    if (iree_status_is_ok(status)) {
      published = progress;
      ++control_sends;
    }
    control.Fail(status);
  }
}

bool CollectiveLink::CanSend() const {
  if (!Ready() ||
      submitted - remote_consumed == storage.length / options.block_size ||
      sources[submitted % sources.size()].pending) {
    return false;
  }
  return Registered()
             ? iree_net_direct_endpoint_query_write_budget(direct).slots != 0
             : iree_net_queue_channel_query_send_budget(channel).slots != 0;
}

void CollectiveLink::Send(iree_async_span_t data, uint32_t* pending_sources) {
  uint32_t sequence = submitted + 1;
  auto& source = sources[submitted % sources.size()];
  source.pending = true;
  source.expected_length = data.length;
  source.pending_sources = pending_sources;
  if (pending_sources) {
    ++*pending_sources;
  }
  iree_status_t status = iree_ok_status();
  if (Registered()) {
    iree_net_direct_write_entry_t entry = {
        data, &target,
        (submitted % (storage.length / options.block_size)) *
            options.block_size};
    iree_net_direct_write_params_t params = {
        .flags = IREE_NET_DIRECT_WRITE_FLAG_NOTIFY,
        .notification_cookie = sequence,
        .entry_count = 1,
        .entries = &entry,
        .completion_callback = {OnSourceReturned, &source}};
    status = iree_net_direct_endpoint_write(direct, &params);
  } else {
    source.expected_length +=
        IREE_NET_QUEUE_MESSAGE_HEADER_SIZE + IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
    iree_net_queue_channel_send_params_t params = {
        .signal_frontier_count = 1,
        .build =
            +[](void* value, const iree_net_queue_message_builder_t* builder) {
              iree_net_queue_frontier_builder_set(
                  &builder->signal_frontier, 0,
                  {1, *static_cast<uint32_t*>(value)});
              return iree_ok_status();
            },
        .build_user_data = &sequence,
        .payload = iree_async_span_list_make(&data, 1),
        .completion_callback = {OnSourceReturned, &source}};
    status = iree_net_queue_channel_send_command(channel, 1, &params);
  }
  if (iree_status_is_ok(status)) {
    submitted = sequence;
    payload_bytes += data.length;
    high_water = std::max<uint64_t>(high_water, submitted - remote_consumed);
    source_high_water =
        std::max<uint64_t>(source_high_water, submitted - completions);
  } else {
    source.pending = false;
    if (pending_sources) {
      --*pending_sources;
    }
  }
  control.Fail(status);
}

const CollectiveLink::Input* CollectiveLink::InputAt(uint32_t sequence) const {
  const auto& input = inputs[(sequence - 1) % inputs.size()];
  return input.sequence == sequence ? &input : nullptr;
}

const CollectiveLink::Input* CollectiveLink::NextInput() const {
  return InputAt(consumed + 1);
}

bool CollectiveLink::ReadyBlocks(uint32_t first, uint32_t count,
                                 uint32_t& arrived) const {
  while (arrived < count && InputAt(first + arrived)) {
    ++arrived;
  }
  return arrived == count;
}

uint32_t CollectiveLink::Load32(uint32_t first, size_t offset) const {
  return iree_unaligned_load_le_u32(
      InputAt(first + offset / options.block_size)->bytes.data +
      offset % options.block_size);
}

void CollectiveLink::Consume() {
  auto& input = inputs[consumed % inputs.size()];
  iree_async_buffer_lease_release(&input.lease);
  input = {};
  ++consumed;
}

bool CollectiveLink::Idle() const {
  if (purpose == LinkPurpose::kResult) {
    return control_sends == control_completions &&
           (direction == Direction::kIncoming ||
            result_coordinate == published);
  }
  return submitted == completions && submitted == remote_consumed &&
         consumed == published && control_sends == control_completions;
}

struct CollectiveRank::Receiver {
  // Borrowed rank owner, alive through listener and connection retirement.
  CollectiveRank& rank;
  // Bootstrap source identity; data never uses cross-thread peer state.
  uint32_t peer;
  // Directed incoming payload connection and reverse consumption feedback.
  CollectiveLink link;
  // Setup-only listener with this edge's known source identity.
  iree_net_listener_t* listener = nullptr;
  // Bound address published before the application threads start.
  std::string address;
  // Listener retirement joined all unpublished accepts.
  bool stopped = false;
  // Unexpected accepts still awaiting asynchronous retirement.
  size_t rejections = 0;

  Receiver(CollectiveRank& rank, uint32_t peer,
           const CollectiveLinkOptions& options, LinkPurpose purpose,
           iree_async_span_t storage)
      : rank(rank),
        peer(peer),
        link(options, rank.control, Direction::kIncoming, purpose, storage) {}

  static void Accepted(void* value, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto& receiver = *static_cast<Receiver*>(value);
    if (iree_status_is_ok(status) && !receiver.link.connection) {
      CollectiveLink::Connected(&receiver.link, status, connection);
      return;
    }
    if (iree_status_is_ok(status)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "unexpected extra collective connection");
    }
    receiver.rank.control.EndpointError(status);
    if (connection) {
      struct Rejection {
        // Listener owner remains alive through native retirement.
        Receiver* receiver;
        // Transferred reference awaiting the asynchronous deactivation join.
        iree_net_connection_t* connection;
      };
      ++receiver.rejections;
      auto* rejection = new Rejection{&receiver, connection};
      iree_net_connection_deactivate(
          connection, {+[](void* value) {
                         auto* rejection = static_cast<Rejection*>(value);
                         --rejection->receiver->rejections;
                         iree_net_connection_release(rejection->connection);
                         delete rejection;
                       },
                       rejection});
    }
  }
};

struct CollectiveRank::Sender {
  // Bootstrap destination identity, never used to inspect peer application
  // state.
  uint32_t peer;
  // Directed outgoing payload connection and received consumption feedback.
  CollectiveLink link;

  Sender(CollectiveRank& rank, uint32_t peer,
         const CollectiveLinkOptions& options, LinkPurpose purpose,
         iree_async_span_t storage)
      : peer(peer),
        link(options, rank.control, Direction::kOutgoing, purpose, storage) {}
};

CollectiveRank::CollectiveRank(CollectiveControl& control, uint32_t index,
                               CollectiveDelivery delivery, size_t storage_size)
    : index(index),
      control(control),
      delivery_(delivery),
      storage_size_(storage_size) {}

CollectiveRank::~CollectiveRank() {
  senders_.clear();
  receivers_.clear();
  iree_net_transport_factory_release(factory_);
  iree_async_region_release(region_);
  iree_async_slab_release(slab_);
  iree_async_proactor_release(proactor);
}

iree_async_span_t CollectiveRank::Span(size_t offset, size_t length) const {
  return region_ ? iree_async_span_make(region_, offset, length)
                 : iree_async_span_from_ptr(
                       static_cast<uint8_t*>(iree_async_slab_base_ptr(slab_)) +
                           offset,
                       length);
}

iree_status_t CollectiveRank::Initialize(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor) {
  auto options = iree_async_proactor_options_default();
  options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
  auto created = create_proactor(options);
  if (!created.ok()) {
    return std::move(created).status().release();
  }
  proactor = *created;
  auto slab_options = iree_async_slab_options_default();
  slab_options.buffer_size = storage_size_;
  slab_options.buffer_count = 1;
  IREE_RETURN_IF_ERROR(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));
  if (delivery_ == CollectiveDelivery::kRegistered) {
    if (!transport.create_registered_factory) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "transport has no registered collective setup");
    }
    IREE_RETURN_IF_ERROR(transport.create_registered_factory(
        slab_, iree_allocator_system(), &factory_, &region_));
  } else {
    IREE_RETURN_IF_ERROR(
        transport.create_factory(iree_allocator_system(), &factory_));
  }
  CreateLinks();
  return iree_ok_status();
}

CollectiveLink* CollectiveRank::AddIncoming(
    uint32_t peer, const CollectiveLinkOptions& options, LinkPurpose purpose,
    iree_async_span_t storage) {
  receivers_.push_back(
      std::make_unique<Receiver>(*this, peer, options, purpose, storage));
  return &receivers_.back()->link;
}

CollectiveLink* CollectiveRank::AddOutgoing(
    uint32_t peer, const CollectiveLinkOptions& options, LinkPurpose purpose,
    iree_async_span_t storage) {
  senders_.push_back(
      std::make_unique<Sender>(*this, peer, options, purpose, storage));
  return &senders_.back()->link;
}

iree_status_t CollectiveRank::Listen(const TransportBackend& transport) {
  iree_status_t status = iree_ok_status();
  for (size_t i = 0; i < receivers_.size() && iree_status_is_ok(status); ++i) {
    auto& receiver = *receivers_[i];
    std::string requested;
    status = transport.make_bind_address(&requested);
    if (iree_status_is_ok(status)) {
      status = iree_net_transport_factory_create_listener(
          factory_, iree_make_string_view(requested.data(), requested.size()),
          proactor, nullptr, {Receiver::Accepted, &receiver},
          iree_allocator_system(), &receiver.listener);
    }
    if (iree_status_is_ok(status)) {
      std::array<char, 1024> data;
      iree_string_view_t bound;
      status = iree_net_listener_query_bound_address(
          receiver.listener, data.size(), data.data(), &bound);
      if (iree_status_is_ok(status)) {
        receiver.address.assign(bound.data, bound.size);
      }
    }
  }
  return status;
}

const std::string& CollectiveRank::Address(uint32_t peer) const {
  auto it =
      std::find_if(receivers_.begin(), receivers_.end(),
                   [peer](const auto& value) { return value->peer == peer; });
  IREE_ASSERT(it != receivers_.end());
  return (*it)->address;
}

void CollectiveRank::Poll() {
  control.Fail(
      iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
}

void CollectiveRank::Pump() {
  for (auto& receiver : receivers_) {
    receiver->link.Pump();
  }
  for (auto& sender : senders_) {
    sender->link.Pump();
  }
}

void CollectiveRank::Drain() {
  while (!control.failed.load(std::memory_order_acquire)) {
    Pump();
    bool idle = true;
    for (auto& receiver : receivers_) {
      idle &= receiver->link.Idle();
    }
    for (auto& sender : senders_) {
      idle &= sender->link.Idle();
    }
    if (idle) {
      break;
    }
    Poll();
  }
}

void CollectiveRank::Shutdown() {
  for (auto& receiver : receivers_) {
    if (receiver->listener) {
      control.Fail(iree_net_listener_stop(
          receiver->listener,
          {+[](void* value) { static_cast<Receiver*>(value)->stopped = true; },
           receiver.get()}));
    }
  }
  for (auto& sender : senders_) {
    iree_net_transport_connect_operation_cancel(
        &sender->link.connect_operation);
  }
  for (auto& receiver : receivers_) {
    if (receiver->listener) {
      while (!receiver->stopped) {
        Poll();
      }
      iree_net_listener_free(receiver->listener);
      receiver->listener = nullptr;
    }
  }
  for (auto& sender : senders_) {
    while (sender->link.Has(CollectiveLink::kConnecting)) {
      Poll();
    }
  }
  auto deactivate = [](CollectiveLink& link) {
    if (link.connection) {
      iree_net_connection_deactivate(
          link.connection, {+[](void* value) {
                              static_cast<CollectiveLink*>(value)->flags |=
                                  CollectiveLink::kDrained;
                            },
                            &link});
    }
  };
  for (auto& receiver : receivers_) {
    deactivate(receiver->link);
  }
  for (auto& sender : senders_) {
    deactivate(sender->link);
  }
  for (auto& receiver : receivers_) {
    while (receiver->rejections ||
           (receiver->link.connection &&
            !receiver->link.Has(CollectiveLink::kDrained))) {
      Poll();
    }
  }
  for (auto& sender : senders_) {
    while (sender->link.connection &&
           !sender->link.Has(CollectiveLink::kDrained)) {
      Poll();
    }
  }
  // Receive registrations must retire on the poll owner even after callbacks
  // join.
  senders_.clear();
  receivers_.clear();
  iree_async_proactor_end_polling(proactor);
}

void CollectiveRank::Run(
    const std::vector<std::unique_ptr<CollectiveRank>>& ranks) {
  for (auto& sender : senders_) {
    if (control.failed.load(std::memory_order_acquire)) {
      break;
    }
    auto& link = sender->link;
    const auto& address = ranks[sender->peer]->Address(index);
    link.flags |= CollectiveLink::kConnecting;
    iree_status_t status = iree_net_transport_factory_connect(
        factory_, iree_make_string_view(address.data(), address.size()),
        proactor, nullptr, {CollectiveLink::Connected, &link},
        &link.connect_operation);
    if (!iree_status_is_ok(status)) {
      link.flags &= ~CollectiveLink::kConnecting;
    }
    control.Fail(status);
  }
  while (!control.failed.load(std::memory_order_acquire)) {
    Pump();
    bool ready = true;
    for (auto& receiver : receivers_) {
      ready &= receiver->link.Ready();
    }
    for (auto& sender : senders_) {
      ready &= sender->link.Ready();
    }
    if (ready) {
      break;
    }
    Poll();
  }
  control.Arrive();
  for (uint32_t phase = 1;
       phase <= 2 && !control.failed.load(std::memory_order_acquire); ++phase) {
    while (control.phase.load(std::memory_order_acquire) < phase &&
           !control.failed.load(std::memory_order_acquire)) {
      Poll();
    }
    if (control.failed.load(std::memory_order_acquire)) {
      break;
    }
    for (auto& sender : senders_) {
      sender->link.high_water = 0;
      sender->link.source_high_water = 0;
    }
    RunPhase(phase);
    Drain();
    auto& result = transport_results[phase - 1];
    result.payload_storage_bytes = storage_size_;
    for (const auto& receiver : receivers_) {
      result.control_sends += receiver->link.control_sends;
      result.control_completions += receiver->link.control_completions;
    }
    for (const auto& sender : senders_) {
      result.control_sends += sender->link.control_sends;
      result.control_completions += sender->link.control_completions;
      result.sends += sender->link.submitted;
      result.source_completions += sender->link.completions;
      result.payload_bytes += sender->link.payload_bytes;
      result.window_high_water =
          std::max(result.window_high_water, sender->link.high_water);
      result.source_window_high_water = std::max(
          result.source_window_high_water, sender->link.source_high_water);
    }
    control.Arrive();
  }
  while (!control.closing.load(std::memory_order_acquire)) {
    Poll();
  }
  Shutdown();
}

iree_status_t RunCollectiveGroup(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    CollectiveControl& control,
    std::vector<std::unique_ptr<CollectiveRank>>& ranks,
    CollectiveTransportResult* out_result,
    TransferTrialMeasurement measurement) {
  *out_result = {};
  for (auto& rank : ranks) {
    IREE_RETURN_IF_ERROR(rank->Initialize(transport, create_proactor));
    control.proactors.push_back(rank->proactor);
  }
  for (auto& rank : ranks) {
    if (!control.failed.load(std::memory_order_acquire)) {
      control.Fail(rank->Listen(transport));
    }
  }
  out_result->available = !control.failed.load(std::memory_order_acquire);
  std::vector<std::thread> threads;
  for (auto& rank : ranks) {
    threads.emplace_back([&, rank = rank.get()] { rank->Run(ranks); });
  }
  control.Wait();
  if (!control.failed.load(std::memory_order_acquire)) {
    control.Start(1);
    control.Wait();
  }
  if (!control.failed.load(std::memory_order_acquire)) {
    if (measurement.begin) {
      measurement.begin(measurement.user_data);
    }
    const auto start = std::chrono::steady_clock::now();
    control.Start(2);
    control.Wait();
    out_result->elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (measurement.end) {
      measurement.end(measurement.user_data);
    }
  }
  control.closing.store(true, std::memory_order_release);
  control.Wake();
  for (auto& thread : threads) {
    thread.join();
  }
  if (!control.failed.load(std::memory_order_acquire)) {
    for (const auto& rank : ranks) {
      const auto& before = rank->transport_results[0];
      const auto& after = rank->transport_results[1];
      out_result->sends += after.sends - before.sends;
      out_result->source_completions +=
          after.source_completions - before.source_completions;
      out_result->control_sends += after.control_sends - before.control_sends;
      out_result->control_completions +=
          after.control_completions - before.control_completions;
      out_result->payload_bytes += after.payload_bytes - before.payload_bytes;
      out_result->window_high_water =
          std::max(out_result->window_high_water, after.window_high_water);
      out_result->source_window_high_water = std::max(
          out_result->source_window_high_water, after.source_window_high_water);
      out_result->payload_storage_bytes += after.payload_storage_bytes;
    }
  }
  return control.error.release();
}

}  // namespace iree::net::cts
