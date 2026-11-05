// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/gather_trial.h"

#include <algorithm>

#include "iree/base/alignment.h"

namespace iree::net::cts {
namespace {

uint32_t Value(uint32_t rank, uint32_t round, size_t element) {
  return 1u + rank * 19u + round * 41u + uint32_t(element) * 73u;
}

struct Peer {
  // Incoming peer shard storage and consumed-credit sender.
  CollectiveLink* incoming = nullptr;
  // Outgoing local shards and received peer consumption credit.
  CollectiveLink* outgoing = nullptr;
  // Blocks admitted in the current session batch.
  uint32_t sent = 0;
  // Incremental placement scan for each independent gather handle.
  std::vector<uint32_t> arrived;
};

struct Rank : CollectiveRank {
  // Immutable host workload geometry.
  const GatherTrialOptions& options;
  // Exact block count per contributed shard, including a short tail.
  const uint32_t chunks;
  // One batch of padded receive blocks on one incoming edge.
  const size_t edge_size;
  // Rank-private connection and handle readiness state; self has no connection.
  std::vector<Peer> peers;
  // Warm-up and measured application ownership observations.
  std::array<GatherTrialResult, 2> results;

  Rank(CollectiveControl& control, uint32_t index,
       const GatherTrialOptions& options, uint32_t chunks, size_t edge_size,
       size_t total_size)
      : CollectiveRank(control, index, options.delivery, total_size),
        options(options),
        chunks(chunks),
        edge_size(edge_size),
        peers(options.rank_count) {
    for (auto& peer : peers) {
      peer.arrived.resize(options.depth);
    }
  }

  void CreateLinks() override {
    const CollectiveLinkOptions link_options = {
        options.block_size, options.window_size, options.delivery,
        CollectiveInputMode::kOwned};
    size_t offset = options.depth * options.shard_size;
    for (uint32_t peer = 0; peer < options.rank_count; ++peer) {
      if (peer == index) {
        continue;
      }
      auto target = Span(offset, edge_size);
      peers[peer].incoming =
          AddIncoming(peer, link_options, LinkPurpose::kData, target);
      peers[peer].outgoing =
          AddOutgoing(peer, link_options, LinkPurpose::kData, target);
      offset += edge_size;
    }
  }

  void Prepare(uint32_t first, uint32_t count) {
    for (uint32_t slot = 0; slot < count; ++slot) {
      auto* data = iree_async_span_ptr(
          Span(slot * options.shard_size, options.shard_size));
      for (size_t i = 0; i < options.shard_size / 4; ++i) {
        iree_unaligned_store_le_u32(data + i * 4,
                                    Value(index, first + slot, i));
      }
    }
    for (auto& peer : peers) {
      peer.sent = 0;
      std::fill(peer.arrived.begin(), peer.arrived.end(), 0);
    }
  }

  bool Ready(uint32_t slot) {
    bool ready = true;
    for (auto& peer : peers) {
      if (peer.incoming) {
        ready &= peer.incoming->ReadyBlocks(
            peer.incoming->consumed + slot * chunks + 1, chunks,
            peer.arrived[slot]);
      }
    }
    return ready;
  }

  void Read(uint32_t first, uint32_t slot) {
    for (uint32_t peer_index = 0;
         peer_index < options.rank_count &&
         !control.failed.load(std::memory_order_acquire);
         ++peer_index) {
      auto* link = peers[peer_index].incoming;
      uint32_t sequence = link ? link->consumed + slot * chunks + 1 : 0;
      auto* local = iree_async_span_ptr(
          Span(slot * options.shard_size, options.shard_size));
      for (uint32_t block = 0;
           block < chunks && !control.failed.load(std::memory_order_acquire);
           ++block) {
        size_t offset = size_t(block) * options.block_size;
        size_t length =
            std::min(options.block_size, options.shard_size - offset);
        if (link && !link->Registered() &&
            link->InputAt(sequence + block)->bytes.data_length != length) {
          control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                        "gather shard block length differs"));
          break;
        }
        const uint8_t* data =
            link ? link->InputAt(sequence + block)->bytes.data : local + offset;
        for (size_t i = 0; i < length / 4; ++i) {
          if (iree_unaligned_load_le_u32(data + i * 4) !=
              Value(peer_index, first + slot, offset / 4 + i)) {
            control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                          "gather result differs at rank %u "
                                          "round %u shard %u element %zu",
                                          index, first + slot, peer_index,
                                          offset / 4 + i));
            break;
          }
        }
      }
    }
  }

  void Batch(uint32_t first, uint32_t count, GatherTrialResult& result) {
    Prepare(first, count);
    result.handles_high_water =
        std::max<uint64_t>(result.handles_high_water, count);
    uint32_t unread = count;
    bool sent = false;
    while ((unread || !sent) &&
           !control.failed.load(std::memory_order_acquire)) {
      Pump();
      bool progressed = false;
      sent = true;
      for (auto& peer : peers) {
        if (!peer.outgoing) {
          continue;
        }
        while (peer.sent < count * chunks && peer.outgoing->CanSend() &&
               !control.failed.load(std::memory_order_acquire)) {
          uint32_t slot = peer.sent / chunks;
          size_t offset = (peer.sent % chunks) * options.block_size;
          peer.outgoing->Send(
              Span(slot * options.shard_size + offset,
                   std::min(options.block_size, options.shard_size - offset)));
          ++peer.sent;
          progressed = true;
        }
        sent &= peer.sent == count * chunks;
      }
      // Fetching a newer handle never marks an older handle consumed. The old
      // input storage remains live and is checked only when its own turn comes.
      while (unread && Ready(unread - 1) &&
             !control.failed.load(std::memory_order_acquire)) {
        Read(first, unread - 1);
        if (unread > 1) {
          ++result.out_of_order_reads;
        }
        --unread;
        progressed = true;
      }
      if ((unread || !sent) && !progressed &&
          !control.failed.load(std::memory_order_acquire)) {
        Poll();
      }
    }
    if (control.failed.load(std::memory_order_acquire)) {
      return;
    }
    for (auto& peer : peers) {
      if (!peer.incoming) {
        continue;
      }
      for (uint32_t block = 0; block < count * chunks; ++block) {
        peer.incoming->Consume();
      }
    }
    Drain();
  }

  void RunPhase(uint32_t phase) override {
    uint32_t first = phase == 1 ? 0 : options.warmup_rounds;
    uint32_t rounds =
        phase == 1 ? options.warmup_rounds : options.measured_rounds;
    for (uint32_t completed = 0;
         completed < rounds &&
         !control.failed.load(std::memory_order_acquire);) {
      uint32_t count = std::min(options.depth, rounds - completed);
      Batch(first + completed, count, results[phase - 1]);
      completed += count;
    }
  }
};

}  // namespace

iree_status_t RunGatherTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const GatherTrialOptions& options, TransferTrialMeasurement measurement,
    GatherTrialResult* out_result) {
  *out_result = {};
  if (options.rank_count < 2 || options.rank_count > 8 || !options.shard_size ||
      options.shard_size % 4 || !options.block_size || options.block_size % 4 ||
      !options.window_size || !options.depth || !options.warmup_rounds ||
      !options.measured_rounds) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid gather trial dimensions");
  }
  uint64_t rounds = uint64_t(options.warmup_rounds) + options.measured_rounds;
  uint64_t chunks = 1 + (options.shard_size - 1) / options.block_size;
  if (rounds > UINT32_MAX || chunks > UINT32_MAX / rounds ||
      options.depth > UINT32_MAX / chunks ||
      options.block_size >
          SIZE_MAX / options.depth / chunks / options.rank_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "gather trial storage or work extent overflow");
  }
  size_t edge_size = size_t(chunks) * options.depth * options.block_size;
  size_t total_size =
      options.depth * options.shard_size + (options.rank_count - 1) * edge_size;
  if (options.shard_size >
          UINT64_MAX / rounds / options.rank_count / (options.rank_count - 1) ||
      total_size > UINT64_MAX / options.rank_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "gather aggregate extent overflow");
  }
  CollectiveControl control;
  std::vector<std::unique_ptr<CollectiveRank>> ranks;
  for (uint32_t index = 0; index < options.rank_count; ++index) {
    ranks.push_back(std::make_unique<Rank>(control, index, options, chunks,
                                           edge_size, total_size));
  }
  iree_status_t status = RunCollectiveGroup(transport, create_proactor, control,
                                            ranks, measurement, out_result);
  if (iree_status_is_ok(status)) {
    out_result->gathers = options.measured_rounds;
    for (auto& rank : ranks) {
      const auto& result = static_cast<Rank&>(*rank).results[1];
      out_result->out_of_order_reads += result.out_of_order_reads;
      out_result->handles_high_water =
          std::max(out_result->handles_high_water, result.handles_high_water);
    }
  }
  return status;
}

}  // namespace iree::net::cts
