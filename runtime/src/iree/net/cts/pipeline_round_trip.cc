// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/pipeline_round_trip.h"

#include <algorithm>
#include <chrono>

#include "iree/base/alignment.h"

namespace iree::net::cts {
namespace {

uint32_t Value(uint32_t round, size_t element) {
  return 1u + round * 37u + uint32_t(element) * 13u;
}

uint32_t ForwardValue(uint32_t stage, uint32_t round, size_t element) {
  return Value(round, element) + (stage + 1) * (stage + 2) / 2;
}

uint32_t BackwardValue(uint32_t stages, uint32_t round, size_t element,
                       size_t activation_elements) {
  size_t activation_element = element % activation_elements;
  uint32_t value = 2u * ForwardValue(stages - 1, round, activation_element) +
                   100u + uint32_t(element) * 7u;
  for (uint32_t stage = 0; stage + 1 < stages; ++stage) {
    value += ForwardValue(stage, round, activation_element) * (stage + 1);
  }
  return value;
}

struct Flow {
  // Full activation or contribution extent.
  size_t length;
  // Exact number of transfer blocks per microbatch.
  uint32_t chunks;
  // First source slot in the rank's shared registration.
  size_t offset;
  // Incoming whole-input slots; null at this flow's source stage.
  CollectiveLink* incoming = nullptr;
  // Outgoing directed edge; null at this flow's final consumer.
  CollectiveLink* outgoing = nullptr;
  // Exact outstanding source callbacks for each output slot.
  std::vector<uint32_t> pending;
  // First microbatch not yet computed into its source slot.
  uint32_t prepared = 0;
  // First microbatch not yet completely admitted to the outgoing edge.
  uint32_t sent = 0;
  // Blocks already admitted from the current outgoing microbatch.
  uint32_t sent_blocks = 0;
  // Placement scan for the next incoming whole-input dependency.
  uint32_t arrived_blocks = 0;
};

struct Rank : CollectiveRank {
  // Fixed host workload dimensions.
  const PipelineRoundTripOptions& options;
  // Downstream activations and their exact source lifetimes.
  Flow forward;
  // Upstream contributions and their independent source lifetimes.
  Flow backward;
  // Schedule measurements after warm-up and the measured phase.
  std::array<PipelineRoundTripResult, 2> results;

  Rank(CollectiveControl& control, uint32_t index,
       const PipelineRoundTripOptions& options, size_t total_size)
      : CollectiveRank(control, index, options.delivery, total_size),
        options(options),
        forward{
            options.activation_size,
            uint32_t(1 + (options.activation_size - 1) / options.block_size),
            0},
        backward{options.gradient_size,
                 uint32_t(1 + (options.gradient_size - 1) / options.block_size),
                 options.depth * options.activation_size} {
    forward.pending.resize(options.depth);
    backward.pending.resize(options.depth);
  }

  uint8_t* Output(const Flow& flow, uint32_t round) {
    return iree_async_span_ptr(
        Span(flow.offset + (round % options.depth) * flow.length, flow.length));
  }

  void CreateLinks() override {
    CollectiveLinkOptions link_options = {options.block_size,
                                          options.window_size, options.delivery,
                                          CollectiveInputMode::kOwned};
    size_t offset = options.depth * (forward.length + backward.length);
    auto target = Span(
        offset, size_t(forward.chunks) * options.depth * options.block_size);
    if (index) {
      forward.incoming =
          AddIncoming(index - 1, link_options, LinkPurpose::kData, target);
    }
    if (index + 1 < options.rank_count) {
      forward.outgoing =
          AddOutgoing(index + 1, link_options, LinkPurpose::kData, target);
    }
    offset += target.length;
    target = Span(offset,
                  size_t(backward.chunks) * options.depth * options.block_size);
    if (index + 1 < options.rank_count) {
      backward.incoming =
          AddIncoming(index + 1, link_options, LinkPurpose::kData, target);
    }
    if (index) {
      backward.outgoing =
          AddOutgoing(index - 1, link_options, LinkPurpose::kData, target);
    }
  }

  bool Ready(Flow& flow) {
    if (!flow.incoming) {
      return true;
    }
    auto& link = *flow.incoming;
    if (!link.ReadyBlocks(link.consumed + 1, flow.chunks,
                          flow.arrived_blocks)) {
      return false;
    }
    if (!link.Registered()) {
      for (uint32_t block = 0; block < flow.chunks; ++block) {
        size_t length =
            std::min(options.block_size,
                     flow.length - size_t(block) * options.block_size);
        if (link.InputAt(link.consumed + 1 + block)->bytes.data_length !=
            length) {
          control.Fail(
              iree_make_status(IREE_STATUS_DATA_LOSS,
                               "bidirectional pipeline block length differs"));
          return false;
        }
      }
    }
    return true;
  }

  void Consume(Flow& flow) {
    if (flow.incoming) {
      for (uint32_t block = 0; block < flow.chunks; ++block) {
        flow.incoming->Consume();
      }
    }
    flow.arrived_blocks = 0;
  }

  bool Forward(uint32_t goal, PipelineRoundTripResult& result) {
    uint32_t round = forward.prepared;
    uint32_t slot = round % options.depth;
    if (round == goal || round - backward.prepared == options.depth ||
        (forward.outgoing && round - forward.sent == options.depth) ||
        forward.pending[slot] || backward.pending[slot] || !Ready(forward)) {
      return false;
    }
    auto* output = Output(forward, round);
    for (size_t i = 0; i < forward.length / 4; ++i) {
      uint32_t value =
          forward.incoming
              ? forward.incoming->Load32(forward.incoming->consumed + 1, i * 4)
              : Value(round, i);
      value += index + 1;
      iree_unaligned_store_le_u32(output + i * 4, value);
      if (!forward.outgoing && value != ForwardValue(index, round, i)) {
        control.Fail(iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "pipeline activation differs at microbatch %u element %zu", round,
            i));
        break;
      }
    }
    if (control.failed.load(std::memory_order_acquire)) {
      return false;
    }
    Consume(forward);
    ++forward.prepared;
    if (!index) {
      result.pipeline_high_water = std::max<uint64_t>(
          result.pipeline_high_water, forward.prepared - backward.prepared);
    }
    return true;
  }

  bool Backward(PipelineRoundTripResult& result,
                std::chrono::steady_clock::time_point start) {
    uint32_t round = backward.prepared;
    if (round == forward.prepared ||
        (backward.outgoing && round - backward.sent == options.depth) ||
        backward.pending[round % options.depth] || !Ready(backward)) {
      return false;
    }
    auto* activation = Output(forward, round);
    auto* output = Output(backward, round);
    for (size_t i = 0; i < backward.length / 4; ++i) {
      size_t element = i % (forward.length / 4);
      uint32_t value = iree_unaligned_load_le_u32(activation + element * 4);
      value = backward.incoming ? backward.incoming->Load32(
                                      backward.incoming->consumed + 1, i * 4) +
                                      value * (index + 1)
                                : value * 2u + 100u + uint32_t(i) * 7u;
      iree_unaligned_store_le_u32(output + i * 4, value);
      if (!index && value != BackwardValue(options.rank_count, round, i,
                                           forward.length / 4)) {
        control.Fail(iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "pipeline reverse result differs at microbatch %u element %zu",
            round, i));
        break;
      }
    }
    if (control.failed.load(std::memory_order_acquire)) {
      return false;
    }
    Consume(backward);
    ++backward.prepared;
    if (!index && result.first_completion_seconds == 0) {
      result.first_completion_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start)
              .count();
    }
    return true;
  }

  bool Send(Flow& flow) {
    if (!flow.outgoing) {
      return false;
    }
    bool progressed = false;
    while (flow.sent < flow.prepared && flow.outgoing->CanSend() &&
           !control.failed.load(std::memory_order_acquire)) {
      uint32_t slot = flow.sent % options.depth;
      size_t offset = size_t(flow.sent_blocks) * options.block_size;
      flow.outgoing->Send(
          Span(flow.offset + slot * flow.length + offset,
               std::min(options.block_size, flow.length - offset)),
          &flow.pending[slot]);
      if (++flow.sent_blocks == flow.chunks) {
        flow.sent_blocks = 0;
        ++flow.sent;
      }
      progressed = true;
    }
    return progressed;
  }

  void RunPhase(uint32_t phase) override {
    const uint32_t first = phase == 1 ? 0 : options.warmup_rounds;
    const uint32_t count =
        phase == 1 ? options.warmup_rounds : options.measured_rounds;
    const uint32_t goal = first + count;
    forward.prepared = forward.sent = backward.prepared = backward.sent = first;
    auto& result = results[phase - 1];
    const auto start = std::chrono::steady_clock::now();
    while (!control.failed.load(std::memory_order_acquire)) {
      Pump();
      bool progressed = Forward(goal, result);
      progressed |= Send(forward);
      progressed |= Backward(result, start);
      progressed |= Send(backward);
      if (backward.prepared == goal &&
          (!forward.outgoing || forward.sent == goal) &&
          (!backward.outgoing || backward.sent == goal)) {
        break;
      }
      if (!progressed && !control.failed.load(std::memory_order_acquire)) {
        Poll();
      }
    }
  }
};

}  // namespace

iree_status_t RunPipelineRoundTrip(
    const TransportBackend& transport,
    const iree::async::cts::ProactorFactory& create_proactor,
    const PipelineRoundTripOptions& options,
    TransferTrialMeasurement measurement, PipelineRoundTripResult* out_result) {
  *out_result = {};
  if (options.rank_count < 2 || options.rank_count > 16 ||
      !options.activation_size || options.activation_size % 4 ||
      !options.gradient_size || options.gradient_size % 4 ||
      !options.block_size || options.block_size % 4 || !options.window_size ||
      !options.depth || !options.warmup_rounds || !options.measured_rounds) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid bidirectional pipeline dimensions");
  }
  uint64_t rounds = uint64_t(options.warmup_rounds) + options.measured_rounds;
  uint64_t forward_chunks =
      1 + (options.activation_size - 1) / options.block_size;
  uint64_t backward_chunks =
      1 + (options.gradient_size - 1) / options.block_size;
  uint64_t chunks = std::max(forward_chunks, backward_chunks);
  if (rounds > UINT32_MAX || chunks > UINT32_MAX / rounds ||
      options.depth > UINT32_MAX / chunks ||
      options.block_size > SIZE_MAX / options.depth / chunks / 4) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipeline storage or work extent overflow");
  }
  size_t total_size =
      options.depth *
      (options.activation_size + options.gradient_size +
       size_t(forward_chunks + backward_chunks) * options.block_size);
  if (total_size > UINT64_MAX / options.rank_count ||
      options.activation_size + options.gradient_size >
          UINT64_MAX / rounds / (options.rank_count - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pipeline aggregate extent overflow");
  }
  CollectiveControl control;
  std::vector<std::unique_ptr<CollectiveRank>> ranks;
  for (uint32_t index = 0; index < options.rank_count; ++index) {
    ranks.push_back(
        std::make_unique<Rank>(control, index, options, total_size));
  }
  iree_status_t status = RunCollectiveGroup(transport, create_proactor, control,
                                            ranks, measurement, out_result);
  if (iree_status_is_ok(status)) {
    const auto& result = static_cast<Rank&>(*ranks[0]).results[1];
    out_result->microbatches = options.measured_rounds;
    out_result->first_completion_seconds = result.first_completion_seconds;
    out_result->pipeline_high_water = result.pipeline_high_water;
  }
  return status;
}

}  // namespace iree::net::cts
