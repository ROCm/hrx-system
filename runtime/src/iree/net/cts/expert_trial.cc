// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/expert_trial.h"

#include <algorithm>
#include <chrono>
#include <numeric>

#include "iree/base/alignment.h"

namespace iree::net::cts {
namespace {

constexpr size_t kHeaderSize = 16;
constexpr uint32_t kDispatch = 1;
constexpr uint32_t kCombine = 2;

struct Layout {
  // Maximum bytes in a routed dispatch or inverse contribution frame.
  size_t frame_size;
  // Whole-block capacity for a batch on one directed edge.
  size_t edge_size;
  // Source frames followed by incoming placement slots for all remote peers.
  size_t transport_size;
  // Checked caller results after the transport arenas.
  size_t total_size;
};

struct Route {
  // Global expert identity; the rank shard is derived only from this ID.
  uint32_t expert;
  // Nonzero integer weight for exact, independently reproducible results.
  uint32_t weight;
};

struct Frame {
  // Source slab offset or incoming first block, according to the owning side.
  size_t position = 0;
  // Actual frame extent learned from the received header, not peer memory.
  uint32_t length = 0;
  // Actual token records; zero is a real participation announcement.
  uint32_t count = 0;
  // Contiguous placed block scan, distinct from application consumption.
  uint32_t arrived = 0;
};

struct Peer {
  // Incoming directed edge, borrowed through rank shutdown.
  CollectiveLink* incoming = nullptr;
  // Reverse directed edge, independently owned and progressed.
  CollectiveLink* outgoing = nullptr;
  // Source frame descriptors, bounded by configured round depth.
  std::vector<Frame> sends;
  // Received frame descriptors, retained until expert/caller consumption.
  std::vector<Frame> receives;
  // Next frame and byte position to admit on the outgoing edge.
  struct {
    // Number of completely admitted frames.
    uint32_t frame = 0;
    // Bytes admitted from the current frame.
    size_t offset = 0;
  } sent;
  // Frames with all blocks placed, not yet necessarily consumed.
  uint32_t ready = 0;
};

uint32_t Payload(uint32_t rank, uint32_t round, uint32_t token,
                 size_t element) {
  return 1u + rank * 17u + round * 31u + token * 43u + uint32_t(element) * 7u;
}

uint32_t Scale(uint32_t rank, uint32_t round, uint32_t token, size_t element) {
  return 1u + (rank * 3u + round * 5u + token * 7u + uint32_t(element)) % 13u;
}

uint32_t Hash(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  return value ^ (value >> 16);
}

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

struct Rank : CollectiveRank {
  // Immutable workload dimensions, shared without live application state.
  const ExpertTrialOptions& options;
  // Validated arena geometry.
  const Layout& layout;
  // Directed peer connections; the self entry has no connection.
  std::vector<Peer> peers;
  // Source-private route plans retained through inverse combine.
  std::vector<Route> routes;
  // Cached route generation for each reusable round slot.
  std::vector<uint32_t> route_epochs;
  // Destinations whose inverse contribution is still owed for each token.
  std::vector<uint32_t> remaining;
  // Warm-up and measurement schedule counters, not shared between ranks.
  std::array<ExpertTrialResult, 2> results;

  Rank(CollectiveControl& control, uint32_t index,
       const ExpertTrialOptions& options, const Layout& layout)
      : CollectiveRank(control, index, options.delivery, layout.total_size),
        options(options),
        layout(layout),
        peers(options.rank_count),
        routes(size_t(options.depth) * options.token_capacity * options.top_k),
        route_epochs(options.depth, UINT32_MAX),
        remaining(size_t(options.depth) * options.token_capacity) {
    for (auto& peer : peers) {
      peer.sends.resize(options.depth);
      peer.receives.resize(options.depth);
    }
  }

  uint32_t Count(uint32_t round) const {
    if (!options.token_capacity) {
      return 0;
    }
    if (options.traffic == ExpertTraffic::kHotspot) {
      return index + 1 == options.rank_count ? 0 : options.token_capacity;
    }
    if (options.traffic == ExpertTraffic::kUneven) {
      return (round + index) % options.rank_count == 0
                 ? 0
                 : 1 + (round * 3u + index * 5u) % options.token_capacity;
    }
    return options.token_capacity;
  }

  Route* Routes(uint32_t slot, uint32_t token) {
    return routes.data() +
           (size_t(slot) * options.token_capacity + token) * options.top_k;
  }

  uint32_t* Output(uint32_t slot, uint32_t token) {
    return reinterpret_cast<uint32_t*>(iree_async_span_ptr(
        Span(layout.transport_size +
                 (size_t(slot) * options.token_capacity + token) *
                     options.combine_bytes,
             options.combine_bytes)));
  }

  void CreateLinks() override {
    CollectiveLinkOptions link_options = {options.block_size,
                                          options.window_size, options.delivery,
                                          CollectiveInputMode::kOwned};
    size_t ordinal = 0;
    for (uint32_t peer = 0; peer < options.rank_count; ++peer) {
      if (peer == index) {
        continue;
      }
      auto target = Span((options.rank_count - 1 + ordinal) * layout.edge_size,
                         layout.edge_size);
      peers[peer].incoming =
          AddIncoming(peer, link_options, LinkPurpose::kData, target);
      peers[peer].outgoing =
          AddOutgoing(peer, link_options, LinkPurpose::kData, target);
      for (uint32_t slot = 0; slot < options.depth; ++slot) {
        peers[peer].sends[slot].position =
            ordinal * layout.edge_size + slot * layout.frame_size;
      }
      ++ordinal;
    }
  }

  void PrepareRoutes(uint32_t first, uint32_t count,
                     ExpertTrialResult& result) {
    const uint32_t shard = options.expert_count / options.rank_count;
    for (uint32_t slot = 0; slot < count; ++slot) {
      uint32_t round = first + slot;
      uint32_t epoch = round / options.route_reuse_rounds;
      if (route_epochs[slot] != epoch) {
        uint32_t extent = options.traffic == ExpertTraffic::kHotspot
                              ? shard
                              : options.expert_count;
        uint32_t stride = extent / options.top_k + 1;
        while (std::gcd(stride, extent) != 1) {
          ++stride;
        }
        for (uint32_t token = 0; token < options.token_capacity; ++token) {
          uint32_t start =
              Hash(index * 101u + epoch * 193u + token * 307u) % extent;
          for (uint32_t choice = 0; choice < options.top_k; ++choice) {
            Routes(slot, token)[choice] = {
                uint32_t((start + uint64_t(choice) * stride) % extent) +
                    (options.traffic == ExpertTraffic::kHotspot
                         ? options.expert_count - shard
                         : 0),
                1u + Hash(token * 71u + choice * 13u + epoch) % 7u};
          }
        }
        route_epochs[slot] = epoch;
      }
      result.tokens += Count(round);
      for (uint32_t token = 0; token < Count(round); ++token) {
        auto* output = Output(slot, token);
        std::fill(output, output + options.combine_bytes / 4, 0);
        uint32_t mask = 0;
        for (uint32_t choice = 0; choice < options.top_k; ++choice) {
          const auto& route = Routes(slot, token)[choice];
          uint32_t peer = route.expert / shard;
          if (peer != index) {
            mask |= 1u << peer;
            continue;
          }
          ++result.local_routes;
          for (size_t i = 0; i < options.combine_bytes / 4; ++i) {
            uint32_t value =
                Payload(index, round, token, i % (options.dispatch_bytes / 4));
            uint32_t scale =
                options.scale_bytes
                    ? Scale(index, round, token, i % (options.scale_bytes / 4))
                    : 0;
            output[i] += (value + route.expert * 3u + scale) * route.weight;
          }
        }
        remaining[size_t(slot) * options.token_capacity + token] = mask;
      }
    }
  }

  void PrepareDispatch(uint32_t first, uint32_t count,
                       ExpertTrialResult& result) {
    PrepareRoutes(first, count, result);
    for (uint32_t peer = 0; peer < options.rank_count; ++peer) {
      if (peer == index) {
        continue;
      }
      for (uint32_t slot = 0; slot < count; ++slot) {
        auto& frame = peers[peer].sends[slot];
        uint8_t* data =
            iree_async_span_ptr(Span(frame.position, layout.frame_size));
        size_t offset = kHeaderSize;
        frame.count = 0;
        for (uint32_t token = 0; token < Count(first + slot); ++token) {
          if (!(remaining[size_t(slot) * options.token_capacity + token] &
                (1u << peer))) {
            continue;
          }
          uint32_t route_count = 0;
          iree_unaligned_store_le_u32(data + offset, token);
          size_t route_offset = offset + 8;
          for (uint32_t choice = 0; choice < options.top_k; ++choice) {
            const auto& route = Routes(slot, token)[choice];
            if (route.expert / (options.expert_count / options.rank_count) !=
                peer) {
              continue;
            }
            iree_unaligned_store_le_u32(data + route_offset, route.expert);
            iree_unaligned_store_le_u32(data + route_offset + 4, route.weight);
            route_offset += 8;
            ++route_count;
          }
          iree_unaligned_store_le_u32(data + offset + 4, route_count);
          result.metadata_bytes += 8 + route_count * 8;
          offset = route_offset;
          for (size_t i = 0; i < options.dispatch_bytes / 4; ++i) {
            iree_unaligned_store_le_u32(data + offset + i * 4,
                                        Payload(index, first + slot, token, i));
          }
          offset += options.dispatch_bytes;
          for (size_t i = 0; i < options.scale_bytes / 4; ++i) {
            iree_unaligned_store_le_u32(data + offset + i * 4,
                                        Scale(index, first + slot, token, i));
          }
          offset += options.scale_bytes;
          ++frame.count;
          ++result.remote_tokens;
        }
        frame.length = offset;
        iree_unaligned_store_le_u32(data, kDispatch);
        iree_unaligned_store_le_u32(data + 4, first + slot);
        iree_unaligned_store_le_u32(data + 8, frame.count);
        iree_unaligned_store_le_u32(data + 12, frame.length);
        result.dispatch_bytes += frame.count * options.dispatch_bytes;
        result.scale_bytes += frame.count * options.scale_bytes;
        result.header_bytes += kHeaderSize;
      }
    }
  }

  void Exchange(uint32_t type, uint32_t first, uint32_t count) {
    for (auto& peer : peers) {
      if (!peer.incoming) {
        continue;
      }
      peer.sent = {};
      peer.ready = 0;
      for (auto& frame : peer.receives) {
        frame = {};
      }
      peer.receives[0].position = peer.incoming->consumed + 1;
    }
    bool done = false;
    while (!done && !control.failed.load(std::memory_order_acquire)) {
      Pump();
      done = true;
      bool progressed = false;
      for (auto& peer : peers) {
        if (!peer.incoming) {
          continue;
        }
        while (peer.sent.frame < count && peer.outgoing->CanSend() &&
               !control.failed.load(std::memory_order_acquire)) {
          const auto& frame = peer.sends[peer.sent.frame];
          size_t length =
              std::min(options.block_size, frame.length - peer.sent.offset);
          peer.outgoing->Send(Span(frame.position + peer.sent.offset, length));
          peer.sent.offset += length;
          if (peer.sent.offset == frame.length) {
            ++peer.sent.frame;
            peer.sent.offset = 0;
          }
          progressed = true;
        }
        while (peer.ready < count &&
               !control.failed.load(std::memory_order_acquire)) {
          auto& frame = peer.receives[peer.ready];
          auto& link = *peer.incoming;
          if (!link.InputAt(frame.position)) {
            break;
          }
          if (!frame.length) {
            const auto& input = *link.InputAt(frame.position);
            if (input.bytes.data_length < kHeaderSize) {
              control.Fail(iree_make_status(
                  IREE_STATUS_DATA_LOSS, "expert frame header is incomplete"));
              break;
            }
            frame.count = link.Load32(frame.position, 8);
            frame.length = link.Load32(frame.position, 12);
            uint32_t round = type == kDispatch ? first + peer.ready
                                               : first + count - 1 - peer.ready;
            if (link.Load32(frame.position, 0) != type ||
                link.Load32(frame.position, 4) != round ||
                frame.count > options.token_capacity ||
                frame.length < kHeaderSize ||
                frame.length > layout.frame_size || frame.length % 4) {
              control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                            "expert frame dimensions differ"));
              break;
            }
          }
          uint32_t blocks = 1 + (frame.length - 1) / options.block_size;
          if (!link.ReadyBlocks(frame.position, blocks, frame.arrived)) {
            break;
          }
          if (!link.Registered()) {
            for (uint32_t block = 0; block < blocks; ++block) {
              size_t expected =
                  std::min(options.block_size,
                           frame.length - size_t(block) * options.block_size);
              if (link.InputAt(frame.position + block)->bytes.data_length !=
                  expected) {
                control.Fail(
                    iree_make_status(IREE_STATUS_DATA_LOSS,
                                     "expert frame block length differs"));
                break;
              }
            }
          }
          ++peer.ready;
          if (peer.ready < count) {
            peer.receives[peer.ready].position = frame.position + blocks;
          }
          progressed = true;
        }
        done &= peer.sent.frame == count && peer.ready == count;
      }
      if (!done && !progressed &&
          !control.failed.load(std::memory_order_acquire)) {
        Poll();
      }
    }
    // Source return is independent of placement. Only this exact join permits
    // overwriting the outgoing dispatch arena with inverse combine sources.
    bool returned = false;
    while (!returned && !control.failed.load(std::memory_order_acquire)) {
      Pump();
      returned = true;
      for (auto& peer : peers) {
        if (peer.outgoing) {
          returned &= peer.outgoing->submitted == peer.outgoing->completions;
        }
      }
      if (!returned) {
        Poll();
      }
    }
  }

  void Consume(uint32_t count) {
    for (auto& peer : peers) {
      if (!peer.incoming) {
        continue;
      }
      for (uint32_t slot = 0; slot < count; ++slot) {
        for (uint32_t block = 0; block < peer.receives[slot].arrived; ++block) {
          peer.incoming->Consume();
        }
      }
    }
    Pump();
  }

  void PrepareCombine(uint32_t first, uint32_t count,
                      ExpertTrialResult& result) {
    for (uint32_t slot = 0; slot < count; ++slot) {
      uint64_t received = 0;
      for (auto& peer : peers) {
        if (peer.incoming) {
          received += peer.receives[slot].count;
        }
      }
      result.maximum_receiver_tokens =
          std::max(result.maximum_receiver_tokens, received);
    }
    result.retained_handles += count - 1;
    for (auto& peer : peers) {
      if (!peer.incoming) {
        continue;
      }
      auto& link = *peer.incoming;
      for (uint32_t slot = 0;
           slot < count && !control.failed.load(std::memory_order_acquire);
           ++slot) {
        const auto& input = peer.receives[count - 1 - slot];
        auto& output = peer.sends[slot];
        uint8_t* data =
            iree_async_span_ptr(Span(output.position, layout.frame_size));
        size_t read_offset = kHeaderSize;
        size_t write_offset = kHeaderSize;
        for (uint32_t record = 0;
             record < input.count &&
             !control.failed.load(std::memory_order_acquire);
             ++record) {
          if (input.length - read_offset < 8) {
            control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                          "expert token header is incomplete"));
            break;
          }
          uint32_t token = link.Load32(input.position, read_offset);
          uint32_t route_count = link.Load32(input.position, read_offset + 4);
          read_offset += 8;
          if (token >= options.token_capacity || !route_count ||
              route_count > options.top_k ||
              route_count * 8 + options.dispatch_bytes + options.scale_bytes >
                  input.length - read_offset) {
            control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                          "expert token extent differs"));
            break;
          }
          uint32_t weight_sum = 0;
          uint32_t expert_sum = 0;
          for (uint32_t route = 0; route < route_count; ++route) {
            uint32_t expert = link.Load32(input.position, read_offset);
            uint32_t weight = link.Load32(input.position, read_offset + 4);
            if (expert / (options.expert_count / options.rank_count) != index ||
                !weight) {
              control.Fail(
                  iree_make_status(IREE_STATUS_DATA_LOSS,
                                   "expert route delivered to wrong owner"));
              break;
            }
            weight_sum += weight;
            expert_sum += expert * 3u * weight;
            read_offset += 8;
          }
          if (control.failed.load(std::memory_order_acquire)) {
            break;
          }
          iree_unaligned_store_le_u32(data + write_offset, token);
          write_offset += 4;
          for (size_t i = 0; i < options.combine_bytes / 4; ++i) {
            uint32_t value = link.Load32(
                input.position, read_offset + (i * 4) % options.dispatch_bytes);
            uint32_t scale =
                options.scale_bytes
                    ? link.Load32(input.position,
                                  read_offset + options.dispatch_bytes +
                                      (i * 4) % options.scale_bytes)
                    : 0;
            iree_unaligned_store_le_u32(
                data + write_offset + i * 4,
                (value + scale) * weight_sum + expert_sum);
          }
          read_offset += options.dispatch_bytes + options.scale_bytes;
          write_offset += options.combine_bytes;
        }
        if (read_offset != input.length &&
            !control.failed.load(std::memory_order_acquire)) {
          control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                        "expert dispatch has unused bytes"));
        }
        output.count = input.count;
        output.length = write_offset;
        iree_unaligned_store_le_u32(data, kCombine);
        iree_unaligned_store_le_u32(data + 4, first + count - 1 - slot);
        iree_unaligned_store_le_u32(data + 8, output.count);
        iree_unaligned_store_le_u32(data + 12, output.length);
        result.combine_bytes += output.count * options.combine_bytes;
        result.metadata_bytes += output.count * 4;
        result.header_bytes += kHeaderSize;
      }
    }
  }

  void CheckCombine(uint32_t first, uint32_t count) {
    for (uint32_t peer_index = 0; peer_index < options.rank_count;
         ++peer_index) {
      auto& peer = peers[peer_index];
      if (!peer.incoming) {
        continue;
      }
      auto& link = *peer.incoming;
      for (uint32_t wire_slot = 0;
           wire_slot < count && !control.failed.load(std::memory_order_acquire);
           ++wire_slot) {
        uint32_t slot = count - 1 - wire_slot;
        const auto& frame = peer.receives[wire_slot];
        if (frame.length !=
            kHeaderSize + frame.count * (4 + options.combine_bytes)) {
          control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                        "expert combine extent differs"));
          break;
        }
        size_t offset = kHeaderSize;
        for (uint32_t record = 0;
             record < frame.count &&
             !control.failed.load(std::memory_order_acquire);
             ++record) {
          uint32_t token = link.Load32(frame.position, offset);
          offset += 4;
          if (token >= Count(first + slot) ||
              !(remaining[size_t(slot) * options.token_capacity + token] &
                (1u << peer_index))) {
            control.Fail(iree_make_status(
                IREE_STATUS_DATA_LOSS,
                "expert inverse token duplicated or unrouted"));
            break;
          }
          remaining[size_t(slot) * options.token_capacity + token] &=
              ~(1u << peer_index);
          auto* output = Output(slot, token);
          for (size_t i = 0; i < options.combine_bytes / 4; ++i) {
            output[i] += link.Load32(frame.position, offset + i * 4);
          }
          offset += options.combine_bytes;
        }
      }
    }
    for (uint32_t slot = 0;
         slot < count && !control.failed.load(std::memory_order_acquire);
         ++slot) {
      for (uint32_t token = 0; token < Count(first + slot) &&
                               !control.failed.load(std::memory_order_acquire);
           ++token) {
        if (remaining[size_t(slot) * options.token_capacity + token]) {
          control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                        "expert inverse contribution missing"));
          break;
        }
        auto* output = Output(slot, token);
        for (size_t i = 0; i < options.combine_bytes / 4; ++i) {
          uint32_t value = Payload(index, first + slot, token,
                                   i % (options.dispatch_bytes / 4));
          uint32_t scale = options.scale_bytes
                               ? Scale(index, first + slot, token,
                                       i % (options.scale_bytes / 4))
                               : 0;
          uint32_t expected = 0;
          for (uint32_t choice = 0; choice < options.top_k; ++choice) {
            const auto& route = Routes(slot, token)[choice];
            expected += (value + route.expert * 3u + scale) * route.weight;
          }
          if (output[i] != expected) {
            control.Fail(
                iree_make_status(IREE_STATUS_DATA_LOSS,
                                 "expert weighted result differs at rank %u "
                                 "round %u token %u element %zu",
                                 index, first + slot, token, i));
            break;
          }
        }
      }
    }
  }

  void RunPhase(uint32_t phase) override {
    uint32_t first = phase == 1 ? 0 : options.warmup_rounds;
    uint32_t rounds =
        phase == 1 ? options.warmup_rounds : options.measured_rounds;
    auto& result = results[phase - 1];
    for (uint32_t completed = 0;
         completed < rounds &&
         !control.failed.load(std::memory_order_acquire);) {
      uint32_t count = std::min(options.depth, rounds - completed);
      auto start = std::chrono::steady_clock::now();
      PrepareDispatch(first + completed, count, result);
      Exchange(kDispatch, first + completed, count);
      result.dispatch_seconds += Seconds(start);
      if (control.failed.load(std::memory_order_acquire)) {
        break;
      }
      start = std::chrono::steady_clock::now();
      PrepareCombine(first + completed, count, result);
      Consume(count);
      if (control.failed.load(std::memory_order_acquire)) {
        break;
      }
      Exchange(kCombine, first + completed, count);
      if (control.failed.load(std::memory_order_acquire)) {
        break;
      }
      CheckCombine(first + completed, count);
      Consume(count);
      Drain();
      result.combine_seconds += Seconds(start);
      completed += count;
    }
  }
};

}  // namespace

iree_status_t RunExpertTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const ExpertTrialOptions& options, TransferTrialMeasurement measurement,
    ExpertTrialResult* out_result) {
  *out_result = {};
  if (options.rank_count < 2 || options.rank_count > 8 ||
      !options.expert_count || options.expert_count % options.rank_count ||
      !options.top_k || options.top_k > options.expert_count ||
      (options.traffic == ExpertTraffic::kHotspot &&
       options.top_k > options.expert_count / options.rank_count) ||
      !options.dispatch_bytes || options.dispatch_bytes % 4 ||
      options.scale_bytes % 4 ||
      options.combine_bytes < options.dispatch_bytes ||
      options.combine_bytes % 4 ||
      options.scale_bytes > options.combine_bytes ||
      options.block_size < kHeaderSize || options.block_size % 4 ||
      !options.window_size || !options.depth || !options.route_reuse_rounds ||
      !options.warmup_rounds || !options.measured_rounds) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid expert trial dimensions");
  }
  if (options.dispatch_bytes > UINT32_MAX || options.scale_bytes > UINT32_MAX ||
      options.combine_bytes > UINT32_MAX || options.block_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expert frame geometry overflow");
  }
  uint64_t record_size =
      std::max<uint64_t>(8ull + uint64_t(options.top_k) * 8 +
                             options.dispatch_bytes + options.scale_bytes,
                         4ull + options.combine_bytes);
  if (record_size >
      (UINT32_MAX - kHeaderSize) / std::max(1u, options.token_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expert frame extent overflow");
  }
  uint64_t frame_size = kHeaderSize + record_size * options.token_capacity;
  uint64_t chunks = 1 + (frame_size - 1) / options.block_size;
  uint64_t rounds = uint64_t(options.warmup_rounds) + options.measured_rounds;
  if (rounds > UINT32_MAX || chunks > UINT32_MAX / rounds / 2 ||
      options.depth > UINT32_MAX / chunks) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expert trial work extent overflow");
  }
  uint64_t edge_size = chunks * options.block_size * options.depth;
  uint64_t result_size =
      uint64_t(options.depth) * options.token_capacity * options.combine_bytes;
  if (result_size > SIZE_MAX ||
      edge_size > (SIZE_MAX - result_size) / (2 * (options.rank_count - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expert trial storage extent overflow");
  }
  size_t transport_size = size_t(edge_size) * 2 * (options.rank_count - 1);
  Layout layout = {size_t(frame_size), size_t(edge_size), transport_size,
                   transport_size + size_t(result_size)};
  if (layout.total_size > UINT64_MAX / options.rank_count ||
      frame_size > UINT64_MAX / rounds / 2 / options.rank_count /
                       (options.rank_count - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expert aggregate extent overflow");
  }
  CollectiveControl control;
  std::vector<std::unique_ptr<CollectiveRank>> ranks;
  for (uint32_t index = 0; index < options.rank_count; ++index) {
    ranks.push_back(std::make_unique<Rank>(control, index, options, layout));
  }
  iree_status_t status = RunCollectiveGroup(transport, create_proactor, control,
                                            ranks, measurement, out_result);
  if (iree_status_is_ok(status)) {
    out_result->rounds = options.measured_rounds;
    for (auto& rank : ranks) {
      const auto& result = static_cast<Rank&>(*rank).results[1];
      out_result->tokens += result.tokens;
      out_result->remote_tokens += result.remote_tokens;
      out_result->local_routes += result.local_routes;
      out_result->dispatch_bytes += result.dispatch_bytes;
      out_result->scale_bytes += result.scale_bytes;
      out_result->combine_bytes += result.combine_bytes;
      out_result->metadata_bytes += result.metadata_bytes;
      out_result->header_bytes += result.header_bytes;
      out_result->retained_handles += result.retained_handles;
      out_result->maximum_receiver_tokens = std::max(
          out_result->maximum_receiver_tokens, result.maximum_receiver_tokens);
      out_result->dispatch_seconds =
          std::max(out_result->dispatch_seconds, result.dispatch_seconds);
      out_result->combine_seconds =
          std::max(out_result->combine_seconds, result.combine_seconds);
    }
  }
  return status;
}

}  // namespace iree::net::cts
