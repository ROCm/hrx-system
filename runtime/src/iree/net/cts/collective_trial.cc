// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/collective_trial.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <vector>

#include "iree/base/alignment.h"

namespace iree::net::cts {
namespace {

// Validated, immutable application storage geometry shared by all ranks.
struct StorageLayout {
  // Output tensor slots, reused only after their exact source callbacks.
  size_t output_size;
  // Application input slots, independently sized from source callback records.
  size_t target_slots;
  // Output tensors followed by padded input blocks in one reusable slab.
  size_t total_size;
};

struct Rank : CollectiveRank {
  // Shared immutable schedule dimensions.
  const CollectiveTrialOptions& options;
  // Validated output and receive storage layout.
  const StorageLayout& layout;
  // Borrowed links owned and retired by the base poll owner.
  CollectiveLink* incoming = nullptr;
  // Borrowed outgoing link, valid until asynchronous shutdown joins.
  CollectiveLink* outgoing = nullptr;
  // Schedule-specific counters captured independently of transport accounting.
  std::array<CollectiveTrialResult, 2> results;
  // Exact outstanding block sources for each pipeline output slot.
  std::vector<uint32_t> pending_sources;

  Rank(const CollectiveTrialOptions& options, const StorageLayout& layout,
       CollectiveControl& control, uint32_t index)
      : CollectiveRank(control, index, options.delivery, layout.total_size),
        options(options),
        layout(layout),
        pending_sources(options.pipeline_depth, 0) {}

  void CreateLinks() override {
    bool pipeline = options.schedule == CollectiveSchedule::kPipeline;
    CollectiveLinkOptions link_options = {
        options.block_size, options.window_size, options.delivery,
        pipeline ? CollectiveInputMode::kOwned : CollectiveInputMode::kLeased,
        static_cast<uint64_t>(options.warmup_rounds) + options.measured_rounds};
    auto target =
        Span(layout.output_size, options.block_size * layout.target_slots);
    incoming = AddIncoming(
        (index + options.rank_count - 1) % options.rank_count, link_options,
        pipeline && index == 0 ? LinkPurpose::kResult : LinkPurpose::kData,
        target);
    outgoing = AddOutgoing((index + 1) % options.rank_count, link_options,
                           pipeline && index + 1 == options.rank_count
                               ? LinkPurpose::kResult
                               : LinkPurpose::kData,
                           target);
  }

  static uint32_t Value(uint32_t rank, uint32_t round, size_t element) {
    return rank * 19u + round * 37u + static_cast<uint32_t>(element) * 13u + 1u;
  }
  void Reduce(uint32_t round) {
    auto* tensor = reinterpret_cast<uint32_t*>(
        iree_async_span_ptr(Span(0, options.tensor_size)));
    size_t elements = options.tensor_size / sizeof(uint32_t);
    for (size_t i = 0; i < elements; ++i) {
      tensor[i] = Value(index, round, i);
    }
    size_t shard_size = options.tensor_size / options.rank_count;
    size_t chunks = 1 + (shard_size - 1) / options.block_size;
    for (uint32_t step = 0; step < 2 * (options.rank_count - 1) &&
                            !control.failed.load(std::memory_order_acquire);
         ++step) {
      bool gathering = step >= options.rank_count - 1;
      uint32_t send_shard =
          gathering ? (index + options.rank_count + 1 -
                       (step - (options.rank_count - 1))) %
                          options.rank_count
                    : (index + options.rank_count - step) % options.rank_count;
      uint32_t receive_shard =
          (send_shard + options.rank_count - 1) % options.rank_count;
      size_t sent = 0;
      size_t received = 0;
      while (!control.failed.load(std::memory_order_acquire)) {
        Pump();
        while (sent < chunks && outgoing->CanSend() &&
               !control.failed.load(std::memory_order_acquire)) {
          size_t offset = sent * options.block_size;
          size_t length = std::min(options.block_size, shard_size - offset);
          outgoing->Send(Span(send_shard * shard_size + offset, length));
          ++sent;
        }
        while (received < chunks && incoming->NextInput() &&
               !control.failed.load(std::memory_order_acquire)) {
          const auto& input = *incoming->NextInput();
          size_t offset = received * options.block_size;
          size_t length = std::min(options.block_size, shard_size - offset);
          if (!incoming->Registered() && input.bytes.data_length != length) {
            control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                          "collective block length differs"));
            break;
          }
          auto* destination =
              tensor + (receive_shard * shard_size + offset) / sizeof(uint32_t);
          for (size_t i = 0; i < length / sizeof(uint32_t); ++i) {
            uint32_t value = iree_unaligned_load_le_u32(input.bytes.data +
                                                        i * sizeof(uint32_t));
            destination[i] = gathering ? value : destination[i] + value;
          }
          incoming->Consume();
          ++received;
        }
        Pump();
        if (sent == chunks && received == chunks &&
            outgoing->submitted == outgoing->completions) {
          break;
        }
        if (!control.failed.load(std::memory_order_acquire)) {
          Poll();
        }
      }
    }
    if (!control.failed.load(std::memory_order_acquire)) {
      for (size_t i = 0; i < elements; ++i) {
        uint32_t expected =
            options.rank_count * Value(0, round, i) +
            19u * options.rank_count * (options.rank_count - 1) / 2;
        if (tensor[i] != expected) {
          control.Fail(iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "all-reduce result differs at rank %u element %zu", index, i));
          break;
        }
      }
    }
  }
  void Pipeline(uint32_t first, uint32_t count, CollectiveTrialResult& result) {
    const uint32_t goal = first + count;
    const size_t chunks = 1 + (options.tensor_size - 1) / options.block_size;
    const bool last = index + 1 == options.rank_count;
    uint32_t prepared = first;
    uint32_t sent = first;
    size_t sent_blocks = 0;
    size_t arrived_blocks = 0;
    bool first_observed = false;
    const auto start = std::chrono::steady_clock::now();
    while (!control.failed.load(std::memory_order_acquire)) {
      Pump();
      if (index == 0 && !first_observed &&
          incoming->result_coordinate > first) {
        first_observed = true;
        result.first_completion_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count();
      }
      bool progressed = false;
      while (prepared < goal &&
             (last || prepared - sent < options.pipeline_depth) &&
             pending_sources[prepared % options.pipeline_depth] == 0 &&
             !control.failed.load(std::memory_order_acquire)) {
        if (index == 0) {
          if (prepared - incoming->result_coordinate ==
              options.pipeline_depth) {
            break;
          }
        } else {
          while (arrived_blocks < chunks &&
                 incoming->InputAt(incoming->consumed + arrived_blocks + 1)) {
            ++arrived_blocks;
          }
          if (arrived_blocks != chunks) {
            break;
          }
        }
        // A stage runs only after every input block is available. Its output is
        // distinct application storage, not a transport packing buffer.
        auto* output = reinterpret_cast<uint32_t*>(iree_async_span_ptr(
            Span((prepared % options.pipeline_depth) * options.tensor_size,
                 options.tensor_size)));
        for (size_t block = 0;
             block < chunks && !control.failed.load(std::memory_order_acquire);
             ++block) {
          size_t offset = block * options.block_size;
          size_t length =
              std::min(options.block_size, options.tensor_size - offset);
          const CollectiveLink::Input* input =
              index == 0 ? nullptr : incoming->NextInput();
          if (input && !incoming->Registered() &&
              input->bytes.data_length != length) {
            control.Fail(iree_make_status(IREE_STATUS_DATA_LOSS,
                                          "pipeline block length differs"));
            break;
          }
          for (size_t i = 0; i < length / sizeof(uint32_t); ++i) {
            size_t element = offset / sizeof(uint32_t) + i;
            uint32_t value = input
                                 ? iree_unaligned_load_le_u32(
                                       input->bytes.data + i * sizeof(uint32_t))
                                 : Value(0, prepared + 1, element);
            output[element] = value + index + 1;
          }
          if (input) {
            incoming->Consume();
          }
        }
        if (last && !control.failed.load(std::memory_order_acquire)) {
          for (size_t i = 0; i < options.tensor_size / sizeof(uint32_t); ++i) {
            uint32_t expected = Value(0, prepared + 1, i) +
                                static_cast<uint32_t>(
                                    static_cast<uint64_t>(options.rank_count) *
                                    (options.rank_count + 1ull) / 2);
            if (output[i] != expected) {
              control.Fail(iree_make_status(
                  IREE_STATUS_DATA_LOSS,
                  "pipeline result differs in microbatch %u element %zu",
                  prepared + 1, i));
              break;
            }
          }
        }
        ++prepared;
        arrived_blocks = 0;
        progressed = true;
        if (last) {
          outgoing->result_coordinate = prepared;
        } else if (index == 0) {
          result.pipeline_high_water =
              std::max<uint64_t>(result.pipeline_high_water,
                                 prepared - incoming->result_coordinate);
        }
      }
      while (!last && sent < prepared && outgoing->CanSend() &&
             !control.failed.load(std::memory_order_acquire)) {
        size_t offset = sent_blocks * options.block_size;
        size_t length =
            std::min(options.block_size, options.tensor_size - offset);
        size_t slot = sent % options.pipeline_depth;
        outgoing->Send(Span(slot * options.tensor_size + offset, length),
                       &pending_sources[slot]);
        if (++sent_blocks == chunks) {
          sent_blocks = 0;
          ++sent;
        }
        progressed = true;
      }
      Pump();
      if (prepared == goal && (last || sent == goal) &&
          (index != 0 || incoming->result_coordinate == goal)) {
        // Observe the first result even if one poll turn delivered the entire
        // measured phase. Source/control joins remain in the phase drain.
        if (index == 0 && !first_observed) {
          result.first_completion_seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            start)
                  .count();
        }
        break;
      }
      if (!progressed && !control.failed.load(std::memory_order_acquire)) {
        Poll();
      }
    }
  }
  void RunPhase(uint32_t phase) override {
    uint32_t round = phase == 1 ? 0 : options.warmup_rounds;
    uint32_t count =
        phase == 1 ? options.warmup_rounds : options.measured_rounds;
    if (options.schedule == CollectiveSchedule::kPipeline) {
      Pipeline(round, count, results[phase - 1]);
    } else {
      for (uint32_t i = 0;
           i < count && !control.failed.load(std::memory_order_acquire); ++i) {
        Reduce(++round);
      }
    }
  }
};

}  // namespace

iree_status_t RunCollectiveTrial(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const CollectiveTrialOptions& options, TransferTrialMeasurement measurement,
    CollectiveTrialResult* out_result) {
  *out_result = {};
  if (options.delivery == CollectiveDelivery::kRegistered &&
      !transport.create_registered_factory) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "transport has no registered collective setup");
  }
  const bool pipeline = options.schedule == CollectiveSchedule::kPipeline;
  if (options.rank_count < 2 || !options.tensor_size ||
      options.tensor_size % sizeof(uint32_t) || !options.block_size ||
      options.block_size % sizeof(uint32_t) || !options.window_size ||
      !options.pipeline_depth || !options.warmup_rounds ||
      !options.measured_rounds ||
      (!pipeline &&
       (options.pipeline_depth != 1 ||
        options.tensor_size % options.rank_count ||
        (options.tensor_size / options.rank_count) % sizeof(uint32_t)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid collective trial dimensions");
  }
  uint64_t extent =
      pipeline ? options.tensor_size : options.tensor_size / options.rank_count;
  uint64_t chunks = 1 + (extent - 1) / options.block_size;
  uint64_t rounds =
      static_cast<uint64_t>(options.warmup_rounds) + options.measured_rounds;
  uint64_t edge_count =
      static_cast<uint64_t>(options.rank_count - 1) * (pipeline ? 1 : 2);
  uint64_t steps = pipeline ? 1 : edge_count;
  if (rounds > UINT32_MAX || chunks > UINT32_MAX / rounds / steps ||
      options.tensor_size > UINT64_MAX / options.measured_rounds / edge_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "collective trial work extent overflow");
  }
  uint64_t target_slots =
      pipeline ? chunks * options.pipeline_depth : options.window_size;
  if (options.tensor_size > SIZE_MAX / options.pipeline_depth ||
      target_slots > UINT32_MAX ||
      options.block_size >
          (SIZE_MAX - options.tensor_size * options.pipeline_depth) /
              target_slots) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "collective trial storage extent overflow");
  }
  StorageLayout layout = {
      options.tensor_size * options.pipeline_depth,
      static_cast<size_t>(target_slots),
      options.tensor_size * options.pipeline_depth +
          options.block_size * static_cast<size_t>(target_slots),
  };
  if (layout.total_size > UINT64_MAX / options.rank_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "collective trial aggregate storage overflow");
  }
  CollectiveControl control;
  std::vector<std::unique_ptr<CollectiveRank>> ranks;
  for (uint32_t i = 0; i < options.rank_count; ++i) {
    ranks.push_back(std::make_unique<Rank>(options, layout, control, i));
  }
  iree_status_t status = RunCollectiveGroup(transport, create_proactor, control,
                                            ranks, measurement, out_result);
  if (iree_status_is_ok(status)) {
    if (pipeline) {
      const auto& result = static_cast<Rank&>(*ranks[0]).results[1];
      out_result->microbatches = options.measured_rounds;
      out_result->first_completion_seconds = result.first_completion_seconds;
      out_result->pipeline_high_water = result.pipeline_high_water;
    } else {
      out_result->collectives = options.measured_rounds;
    }
  }
  return status;
}

}  // namespace iree::net::cts
