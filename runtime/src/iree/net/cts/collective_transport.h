// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CTS_COLLECTIVE_TRANSPORT_H_
#define IREE_NET_CTS_COLLECTIVE_TRANSPORT_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/connection.h"
#include "iree/net/cts/transfer_trial.h"
#include "iree/net/direct_endpoint.h"

namespace iree::net::cts {

// Explicit payload strategy; registered placement never falls back to messages.
enum class CollectiveDelivery { kMessage, kRegistered };

// Borrow individual message leases or assemble whole inputs in owned storage.
enum class CollectiveInputMode { kLeased, kOwned };

struct CollectiveLinkOptions {
  // Maximum transfer bytes, including exact shorter final blocks.
  size_t block_size;
  // Independent bound on outstanding source callbacks.
  uint32_t window_size;
  // Payload path used by both ends of this edge.
  CollectiveDelivery delivery;
  // Incoming message storage lifetime required by the application schedule.
  CollectiveInputMode input_mode;
  // Highest permitted final-result coordinate for a control-only edge.
  uint64_t result_limit = 0;
};

struct CollectiveTransportResult {
  // All listeners were created; later failures must not become skipped trials.
  bool available = false;
  // Measured phase wall time through every rank's output and ownership joins.
  double elapsed_seconds = 0;
  // Bytes admitted as data, excluding queue headers and consumption messages.
  uint64_t payload_bytes = 0;
  // Accepted data operations across all directed edges.
  uint64_t sends = 0;
  // Exact terminal data callbacks across all directed edges.
  uint64_t source_completions = 0;
  // Accepted consumption/final-result messages, excluding setup and warm-up.
  uint64_t control_sends = 0;
  // Terminal callbacks for those control messages, joined before phase end.
  uint64_t control_completions = 0;
  // Peak source callbacks outstanding on one edge.
  uint64_t source_window_high_water = 0;
  // Peak admitted-minus-consumed blocks on one edge.
  uint64_t window_high_water = 0;
  // Registered or host application slab bytes across all ranks.
  uint64_t payload_storage_bytes = 0;
};

// Shared bootstrap, phase and error state, never application data progress.
struct CollectiveControl {
  // Executors borrowed until every application thread joins.
  std::vector<iree_async_proactor_t*> proactors;
  // Current phase: zero for setup, one for warm-up and two for measurement.
  std::atomic<uint32_t> phase{0};
  // Terminal failure wakes both the coordinator and application poll owners.
  std::atomic<bool> failed{false};
  // Coordinator publication authorizing local retirement.
  std::atomic<bool> closing{false};
  // Protects phase arrivals and diagnostic ownership, never payload progress.
  std::mutex mutex;
  // Coordinator sleeps until all ranks arrive or a rank reports failure.
  std::condition_variable condition;
  // Number of ranks that have completed the current phase.
  size_t arrivals = 0;
  // Joined error returned after callback and native retirement.
  iree::Status error;

  void Wake();
  void Fail(iree_status_t status);
  void EndpointError(iree_status_t status);
  void Arrive();
  void Wait();
  void Start(uint32_t value);
};

enum class Direction { kIncoming, kOutgoing };
enum class LinkPurpose { kData, kResult };

// One directed data/credit edge, or a final-result control edge.
struct CollectiveLink {
  struct Source {
    // Stable callback owner, retained through connection deactivation.
    CollectiveLink* link = nullptr;
    // Exact source completion byte count, including message framing above net.
    size_t expected_length = 0;
    // Optional rank-local activation owner; remains alive through retirement.
    uint32_t* pending_sources = nullptr;
    // This slot cannot capture another source until its callback returns.
    bool pending = false;
  };
  struct Input {
    // Exact logical block occupying this slot; zero means free.
    uint32_t sequence = 0;
    // Borrowed received bytes, or the explicitly registered target view.
    iree_const_byte_span_t bytes = {};
    // Moved message storage, empty for explicit registered targets.
    iree_async_buffer_lease_t lease = {};
  };

  // Immutable workload dimensions and delivery strategy.
  const CollectiveLinkOptions options;
  // Phase and failure owner, independent of edge progress.
  CollectiveControl& control;
  // Whether this connection sends data or receives it.
  Direction direction;
  // Data edges exchange payload and credits; the result edge only completes PP.
  LinkPurpose purpose;
  // Reusable target storage owned by the rank, not this borrowed view.
  iree_async_span_t storage;
  // Caller-owned connect operation joined before destruction.
  iree_net_transport_connect_operation_t connect_operation;
  // Owned connection, released only after asynchronous deactivation.
  iree_net_connection_t* connection = nullptr;
  // Message endpoint carrying the queue protocol.
  iree_net_message_endpoint_t message = {};
  // Optional registered-placement endpoint on the same connection.
  iree_net_direct_endpoint_t direct = {};
  // Queue protocol owner, freed after the connection join.
  iree_net_queue_channel_t* channel = nullptr;
  // Borrowed peer registration facts imported once during setup.
  iree_net_direct_target_t target = {};
  // Serialized target grant, retained only for setup.
  std::vector<uint8_t> description;
  // Callback records bounded independently of native SQ geometry.
  std::vector<Source> sources;
  // Application inputs bounded by the advertised consumption window.
  std::vector<Input> inputs;
  // Setup and retirement obligations owned by this poll thread.
  enum Flag : uint32_t {
    kConnecting = 1u << 0,
    kMessageReady = 1u << 1,
    kDirectReady = 1u << 2,
    kGrantSent = 1u << 3,
    kTargetReady = 1u << 4,
    kDrained = 1u << 5,
  };
  // Current setup and retirement obligations.
  uint32_t flags = 0;
  // Last admitted data block, independent of callback order.
  uint32_t submitted = 0;
  // Receiver-consumed prefix observed over the real control channel.
  uint32_t remote_consumed = 0;
  // Application-consumed local prefix, never inferred from placement order.
  uint32_t consumed = 0;
  // Last consumed or final-result coordinate admitted to the control channel.
  uint32_t published = 0;
  // Final-stage microbatch coordinate, published or received on the result
  // edge.
  uint32_t result_coordinate = 0;
  // Exact accepted data-source returns.
  uint64_t completions = 0;
  // Accepted control messages requiring terminal callbacks.
  uint64_t control_sends = 0;
  // Joined terminal control-message callbacks.
  uint64_t control_completions = 0;
  // Actual payload bytes sent, excluding control and queue headers.
  uint64_t payload_bytes = 0;
  // Maximum admitted but not yet consumed blocks.
  uint64_t high_water = 0;
  // Maximum data-source callbacks outstanding, independent of peer consumption.
  uint64_t source_high_water = 0;

  CollectiveLink(const CollectiveLinkOptions& options,
                 CollectiveControl& control, Direction direction,
                 LinkPurpose purpose, iree_async_span_t storage);
  ~CollectiveLink();
  bool Has(Flag flag) const;
  bool Registered() const;
  bool Ready() const;
  static void OnError(void* value, iree_status_t status);
  static void OnControlSent(void* value, iree_status_t status, size_t);
  static void OnSourceReturned(void* value, iree_status_t status,
                               size_t transferred);
  iree_status_t Receive(uint32_t sequence, iree_const_byte_span_t bytes,
                        iree_async_buffer_lease_t* lease);
  static iree_status_t OnCommand(void* value, uint32_t queue_id,
                                 const iree_net_queue_frontier_view_t* waits,
                                 const iree_net_queue_frontier_view_t* signals,
                                 iree_const_byte_span_t payload,
                                 iree_async_buffer_lease_t* lease);
  static iree_status_t OnAdvance(void* value,
                                 const iree_net_queue_frontier_view_t* signals,
                                 iree_const_byte_span_t payload,
                                 iree_async_buffer_lease_t*);
  static iree_status_t OnPlacement(void* value, uint32_t sequence);
  static void OnMessageReady(void* value, iree_status_t status,
                             iree_net_message_endpoint_t endpoint);
  static void OnDirectReady(void* value, iree_status_t status,
                            iree_net_direct_endpoint_t endpoint);
  static void Connected(void* value, iree_status_t status,
                        iree_net_connection_t* connection);
  void Pump();
  bool CanSend() const;
  void Send(iree_async_span_t data, uint32_t* pending_sources = nullptr);
  const Input* InputAt(uint32_t sequence) const;
  const Input* NextInput() const;
  // Advances a rank-local placement scan without rescanning completed blocks.
  // This establishes readiness only; it neither consumes nor releases inputs.
  bool ReadyBlocks(uint32_t first, uint32_t count, uint32_t& arrived) const;
  // Reads an aligned word from a fully placed frame, including ring wrap.
  uint32_t Load32(uint32_t first, size_t offset) const;
  void Consume();
  bool Idle() const;
};

// A CTS application poll owner with one slab/registration and directed links.
// Schedules implement only storage layout and per-phase application behavior.
// Connections and receive registrations retire on this owner's polling thread.
class CollectiveRank {
 public:
  CollectiveRank(CollectiveControl& control, uint32_t index,
                 CollectiveDelivery delivery, size_t storage_size);
  virtual ~CollectiveRank();

  iree_status_t Initialize(
      const TransportBackend& transport,
      const iree::async::cts::ProactorFactory& create_proactor);
  iree_status_t Listen(const TransportBackend& transport);
  void Run(const std::vector<std::unique_ptr<CollectiveRank>>& ranks);
  iree_async_span_t Span(size_t offset, size_t length) const;
  CollectiveLink* AddIncoming(uint32_t peer,
                              const CollectiveLinkOptions& options,
                              LinkPurpose purpose, iree_async_span_t storage);
  CollectiveLink* AddOutgoing(uint32_t peer,
                              const CollectiveLinkOptions& options,
                              LinkPurpose purpose, iree_async_span_t storage);
  void Pump();
  void Poll();
  void Drain();

  // Rank identity used only for immutable connection bootstrap and the caller.
  const uint32_t index;
  // Shared phase/error coordinator; application progress travels through net.
  CollectiveControl& control;
  // Application executor borrowed by the coordinator through thread join.
  iree_async_proactor_t* proactor = nullptr;
  // Cumulative transfer counters after warm-up and measurement respectively.
  std::array<CollectiveTransportResult, 2> transport_results;

 protected:
  virtual void CreateLinks() = 0;
  virtual void RunPhase(uint32_t phase) = 0;

 private:
  struct Receiver;
  struct Sender;

  void Shutdown();
  const std::string& Address(uint32_t peer) const;

  // Explicit payload strategy, fixed before setup.
  const CollectiveDelivery delivery_;
  // Total bounded payload storage, including reusable direct targets.
  const size_t storage_size_;
  // Single reusable application allocation.
  iree_async_slab_t* slab_ = nullptr;
  // Optional registration shared by every connection on this rank.
  iree_async_region_t* region_ = nullptr;
  // Transport/session owner independent of slab lifetime.
  iree_net_transport_factory_t* factory_ = nullptr;
  // Stable listener and incoming connection owners, allocated only at setup.
  std::vector<std::unique_ptr<Receiver>> receivers_;
  // Stable outgoing connection owners, allocated only at setup.
  std::vector<std::unique_ptr<Sender>> senders_;
};

// Initializes and runs all ranks, excluding setup/warm-up/retirement from
// timing. Rank-local phase callbacks must produce checked results and retire
// application inputs; this join additionally includes all native source/control
// callbacks.
iree_status_t RunCollectiveGroup(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    CollectiveControl& control,
    std::vector<std::unique_ptr<CollectiveRank>>& ranks,
    TransferTrialMeasurement measurement,
    CollectiveTransportResult* out_result);

}  // namespace iree::net::cts

#endif  // IREE_NET_CTS_COLLECTIVE_TRANSPORT_H_
