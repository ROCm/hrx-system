// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/serve/service/text/packing.h"
#include "loom/serve/service/text/schedule.h"

namespace {

// A virtual clock, not a hardware latency estimate. Execution costs are
// supplied by each test; requests cannot change an already submitted epoch.
using Tick = int64_t;
constexpr Tick kNever = std::numeric_limits<Tick>::max();
constexpr size_t kNoPredecessor = std::numeric_limits<size_t>::max();

struct Request {
  // New input after any restored prefix; zero starts an already decoding row.
  size_t input_count = 0;
  // Outputs still required, including the one selected by the final prefill.
  size_t output_count = 1;
  // Earliest arrival on the virtual clock.
  Tick arrival = 0;
  // Earlier request whose completion releases this request, if any.
  size_t predecessor = kNoPredecessor;
  // External tool time following predecessor completion.
  Tick delay = 0;
  // Consumer credit is unavailable in [pause_begin, pause_end).
  Tick pause_begin = kNever;
  // Clock value at which the consumer becomes ready again.
  Tick pause_end = kNever;
  // Cancellation time; device work already submitted still retires.
  Tick cancel = kNever;
};

struct Progress {
  // Input tokens consumed by retired device work, including cancelled work.
  size_t inputs = 0;
  // Outputs selected by retired work, including unpublished cancelled output.
  size_t outputs = 0;
  // Published output times; a cancelled in-flight result is not published.
  std::vector<Tick> published;
  // Completion or cancellation observed at a host scheduling boundary.
  Tick finish = kNever;
};

struct Traversal {
  // Submission time, before this traversal has changed request progress.
  Tick start = 0;
  // Retirement time, after which new readiness can affect selection.
  Tick finish = 0;
  // Prepared shape selected by the production catalog/packer.
  loom_serve_packing_shape_t shape = {};
  // Actual selected work in packing order, without padded slots.
  std::vector<loom_serve_packed_span_t> spans;
  // Known prompt inputs, excluding restored prefixes.
  size_t prefill_inputs = 0;
  // Ordinary decode inputs; this proxy does not invent verifier acceptance.
  size_t decode_inputs = 0;
};

// Models the packer's single-epoch caller contract, not native memory admission
// or the service's publication implementation. Each request has an available
// resident row. Explicit credit/arrival events feed trusted readiness; only
// retirement consumes work. Actual packing and shape selection are not mocked.
class Workload {
 public:
  Workload(size_t capacity, size_t chunk_size, std::vector<Request> requests)
      : requests_(std::move(requests)),
        progress_(requests_.size()),
        shapes_(loom_serve_text_default_shapes(capacity, nullptr)),
        chunk_size_(chunk_size) {
    loom_serve_text_default_shapes(capacity, shapes_.data());
  }

  template <typename Cost>
  void Run(Cost cost) {
    const size_t count = requests_.size();
    ASSERT_NE(count, 0);
    size_t work_bound = 0;
    for (size_t i = 0; i < count; ++i) {
      ASSERT_GT(requests_[i].output_count, 0);
      if (requests_[i].predecessor != kNoPredecessor) {
        ASSERT_LT(requests_[i].predecessor, i);
        // Dependency error propagation is outside this packing experiment.
        ASSERT_EQ(requests_[requests_[i].predecessor].cancel, kNever);
      }
      work_bound += requests_[i].input_count + requests_[i].output_count;
    }
    std::vector<loom_serve_ready_span_t> ready(count);
    std::vector<loom_serve_packed_span_t> spans(count), scratch(count);
    Tick now = 0;
    size_t cursor = 0;
    while (true) {
      size_t remaining = 0;
      Tick next_event = kNever;
      std::fill(ready.begin(), ready.end(), loom_serve_ready_span_t{});
      for (size_t i = 0; i < count; ++i) {
        const auto& request = requests_[i];
        auto& progress = progress_[i];
        if (progress.finish != kNever) {
          continue;
        }
        if (request.cancel <= now) {
          progress.finish = now;
          continue;
        }
        ++remaining;
        next_event = std::min(next_event, request.cancel);
        Tick arrival = request.arrival;
        if (request.predecessor != kNoPredecessor) {
          const Tick predecessor = progress_[request.predecessor].finish;
          if (predecessor == kNever) {
            continue;
          }
          arrival = std::max(arrival, predecessor + request.delay);
        }
        if (arrival > now) {
          next_event = std::min(next_event, arrival);
        } else if (request.pause_begin <= now && now < request.pause_end) {
          next_event = std::min(next_event, request.pause_end);
        } else {
          ready[i] = {progress.inputs < request.input_count
                          ? request.input_count - progress.inputs
                          : 1,
                      1};
        }
      }
      if (!remaining) {
        return;
      }
      const auto plan = loom_serve_pack_shapes(
          count, ready.data(), shapes_.size(), shapes_.data(), chunk_size_,
          cursor, spans.data(), scratch.data());
      const size_t selected = plan.span_count;
      cursor = plan.next_row;
      if (!selected) {
        ASSERT_NE(next_event, kNever) << "unreachable remaining work";
        ASSERT_GT(next_event, now);
        now = next_event;
        continue;
      }
      // Every traversal consumes at least one finite input or selects an
      // output. This is a work-conservation bound, not a wall-clock deadline.
      ASSERT_LT(traversals_.size(), work_bound);
      Traversal traversal;
      traversal.start = now;
      traversal.shape = shapes_[plan.shape_index];
      traversal.spans.assign(spans.begin(), spans.begin() + selected);
      std::vector<bool> seen(count);
      for (const auto& span : traversal.spans) {
        ASSERT_LT(span.row_index, count);
        ASSERT_FALSE(seen[span.row_index]);
        seen[span.row_index] = true;
        ASSERT_GE(span.token_count, ready[span.row_index].minimum_count);
        ASSERT_LE(span.token_count, ready[span.row_index].token_count);
        if (progress_[span.row_index].inputs <
            requests_[span.row_index].input_count) {
          traversal.prefill_inputs += span.token_count;
        } else {
          traversal.decode_inputs += span.token_count;
        }
      }
      ASSERT_LE(traversal.prefill_inputs + traversal.decode_inputs,
                traversal.shape.token_capacity);
      ASSERT_EQ(traversal.prefill_inputs + traversal.decode_inputs,
                plan.token_count);
      ASSERT_LE(selected, traversal.shape.span_capacity);
      const Tick duration = cost(traversal);
      ASSERT_GT(duration, 0);
      now += duration;
      traversal.finish = now;
      for (const auto& span : traversal.spans) {
        const auto& request = requests_[span.row_index];
        auto& progress = progress_[span.row_index];
        if (progress.inputs < request.input_count) {
          progress.inputs += span.token_count;
        }
        if (progress.inputs == request.input_count) {
          ++progress.outputs;
          if (request.cancel > now) {
            progress.published.push_back(now);
          }
          if (progress.outputs == request.output_count) {
            progress.finish = now;
          }
        }
      }
      traversals_.push_back(std::move(traversal));
    }
  }

  const std::vector<Progress>& progress() const { return progress_; }
  const std::vector<Traversal>& traversals() const { return traversals_; }

 private:
  // Immutable workload; scheduling cannot inspect future completion lengths.
  const std::vector<Request> requests_;
  // Caller-owned state, advanced only after simulated device retirement.
  std::vector<Progress> progress_;
  // Production catalog; every traversal uses the same prepared variants.
  std::vector<loom_serve_packing_shape_t> shapes_;
  // Existing per-row expansion limit, not an aggregate token budget.
  size_t chunk_size_;
  // Complete selected-work history, including work cancelled in flight.
  std::vector<Traversal> traversals_;
};

TEST(PackingWorkloadTest, LateShortPromptMatchesTheMixedServingPartition) {
  std::vector<Request> requests(8, {.output_count = 8});
  requests.push_back({.input_count = 32768});
  requests.push_back({.input_count = 24, .output_count = 4, .arrival = 10});
  Workload workload(8192, 8192, std::move(requests));
  ASSERT_NO_FATAL_FAILURE(workload.Run([](const Traversal&) { return 10; }));
  const size_t long_spans[] = {8184, 8160, 8183, 8183, 58};
  const size_t short_spans[] = {0, 24, 1, 1, 1};
  ASSERT_GE(workload.traversals().size(), IREE_ARRAYSIZE(long_spans));
  for (size_t i = 0; i < IREE_ARRAYSIZE(long_spans); ++i) {
    const auto& traversal = workload.traversals()[i];
    SCOPED_TRACE(i);
    size_t selected[10] = {};
    for (const auto& span : traversal.spans) {
      selected[span.row_index] = span.token_count;
    }
    for (size_t row = 0; row < 8; ++row) {
      EXPECT_EQ(selected[row], 1);
    }
    EXPECT_EQ(selected[8], long_spans[i]);
    EXPECT_EQ(selected[9], short_spans[i]);
    EXPECT_EQ(traversal.shape.token_capacity, i == 4 ? 128 : 8192);
    EXPECT_EQ(traversal.shape.span_capacity, 16);
    EXPECT_EQ(traversal.decode_inputs, i < 2 ? 8 : 9);
  }
  EXPECT_EQ(workload.progress()[8].finish, 50);
  EXPECT_EQ(workload.progress()[9].published,
            (std::vector<Tick>{20, 30, 40, 50}));
}

TEST(PackingWorkloadTest, ArrivalCannotChangeAnExecutingTraversal) {
  for (Tick arrival : {0, 1, 9, 10, 11}) {
    SCOPED_TRACE(arrival);
    Workload workload(
        8, 8, {{.input_count = 32}, {.input_count = 2, .arrival = arrival}});
    ASSERT_NO_FATAL_FAILURE(workload.Run([](const Traversal&) { return 10; }));
    const auto& first = workload.traversals().front();
    EXPECT_EQ(first.spans.size(), arrival == 0 ? 2 : 1);
    EXPECT_EQ(workload.progress()[1].finish, arrival <= 10 ? 20 : 30);
    for (const auto& traversal : workload.traversals()) {
      for (const auto& span : traversal.spans) {
        if (span.row_index == 1) {
          EXPECT_GE(traversal.start, arrival);
        }
      }
    }
  }
}

TEST(PackingWorkloadTest, ToolDependentTurnsStartAfterCompletionAndDelay) {
  Workload workload(
      8, 8,
      {{.input_count = 5, .output_count = 3},
       {.input_count = 3, .output_count = 2, .predecessor = 0, .delay = 7},
       {.input_count = 32, .output_count = 6, .arrival = 3}});
  ASSERT_NO_FATAL_FAILURE(workload.Run([](const Traversal&) { return 10; }));
  const Tick ready = workload.progress()[0].finish + 7;
  for (const auto& traversal : workload.traversals()) {
    for (const auto& span : traversal.spans) {
      if (span.row_index == 1) {
        EXPECT_GE(traversal.start, ready);
      }
    }
  }
  EXPECT_EQ(workload.progress()[1].published.size(), 2);
  // A tool-only gap has no device traversal and does not get rounded into
  // fictitious periodic batches.
  Workload idle(
      8, 8,
      {{.input_count = 1}, {.input_count = 1, .predecessor = 0, .delay = 37}});
  ASSERT_NO_FATAL_FAILURE(idle.Run([](const Traversal&) { return 10; }));
  ASSERT_EQ(idle.traversals().size(), 2);
  EXPECT_EQ(idle.traversals()[1].start, 47);
  EXPECT_EQ(idle.progress()[1].finish, 57);
}

TEST(PackingWorkloadTest, PerRowChunksDoNotBoundTheSharedTraversal) {
  for (size_t prompts : {1u, 2u}) {
    SCOPED_TRACE(prompts);
    std::vector<Request> requests(8, {.output_count = 8});
    for (size_t i = 0; i < prompts; ++i) {
      requests.push_back({.input_count = 32768});
    }
    Workload workload(32768, 8192, std::move(requests));
    ASSERT_NO_FATAL_FAILURE(workload.Run([](const Traversal&) { return 10; }));
    for (size_t i = 0; i < 4; ++i) {
      const auto& traversal = workload.traversals()[i];
      EXPECT_EQ(traversal.prefill_inputs, 8192 * prompts);
      EXPECT_EQ(traversal.decode_inputs, 8);
      EXPECT_EQ(traversal.shape.token_capacity, prompts == 1 ? 16384 : 32768);
      EXPECT_EQ(traversal.shape.span_capacity, 16);
    }
    for (size_t i = 0; i < prompts; ++i) {
      EXPECT_EQ(workload.progress()[8 + i].finish, 40);
    }
  }
}

TEST(PackingWorkloadTest, DecodeLifetimeDeterminesAdditionalTraversalWork) {
  const std::vector<std::vector<size_t>> decoder_outputs = {
      {}, {1, 2, 3, 4, 5, 6, 7, 8}, {8, 16, 32, 64, 128, 256, 384, 512}};
  const size_t expected_traversals[][2] = {{65, 5}, {65, 8}, {512, 512}};
  const size_t capacities[] = {512, 8192};
  for (size_t cohort = 0; cohort < decoder_outputs.size(); ++cohort) {
    for (size_t shape = 0; shape < IREE_ARRAYSIZE(capacities); ++shape) {
      SCOPED_TRACE(cohort);
      SCOPED_TRACE(capacities[shape]);
      std::vector<Request> requests;
      size_t expected_decode = 3;
      for (size_t outputs : decoder_outputs[cohort]) {
        requests.push_back({.output_count = outputs});
        expected_decode += outputs;
      }
      requests.push_back({.input_count = 32768});
      requests.push_back({.input_count = 24, .output_count = 4, .arrival = 10});
      Workload workload(capacities[shape], capacities[shape],
                        std::move(requests));
      ASSERT_NO_FATAL_FAILURE(
          workload.Run([](const Traversal&) { return 10; }));
      size_t prefill_traversals = 0, prefill_inputs = 0, decode_inputs = 0;
      for (const auto& traversal : workload.traversals()) {
        prefill_traversals += traversal.prefill_inputs != 0;
        prefill_inputs += traversal.prefill_inputs;
        decode_inputs += traversal.decode_inputs;
      }
      EXPECT_EQ(prefill_inputs, 32768 + 24);
      EXPECT_EQ(decode_inputs, expected_decode);
      EXPECT_EQ(prefill_traversals, shape == 0 ? 65 : 5);
      EXPECT_EQ(workload.traversals().size(),
                expected_traversals[cohort][shape]);
    }
  }
  // Long-lived unequal decoders let smaller prefills replace decode-only
  // traversals. When they finish early or are absent, smaller chunks add
  // traversals. Neither traversal count nor this virtual clock establishes
  // execution latency or physical weight traffic.
}

TEST(PackingWorkloadTest, CreditPausesOnlyItsRowAndCancellationRetiresWork) {
  Workload workload(8, 8,
                    {{.output_count = 4, .pause_begin = 10, .pause_end = 45},
                     {.output_count = 8},
                     {.input_count = 20, .cancel = 15},
                     {.input_count = 5, .arrival = 30, .cancel = 20}});
  ASSERT_NO_FATAL_FAILURE(workload.Run([](const Traversal&) { return 10; }));
  EXPECT_EQ(workload.progress()[0].published,
            (std::vector<Tick>{10, 60, 70, 80}));
  EXPECT_EQ(workload.progress()[1].published,
            (std::vector<Tick>{10, 20, 30, 40, 50, 60, 70, 80}));
  EXPECT_EQ(workload.progress()[2].inputs, 13);
  EXPECT_EQ(workload.progress()[2].finish, 20);
  EXPECT_TRUE(workload.progress()[2].published.empty());
  EXPECT_EQ(workload.progress()[3].inputs, 0);
  for (const auto& traversal : workload.traversals()) {
    for (const auto& span : traversal.spans) {
      if (span.row_index == 0) {
        EXPECT_TRUE(traversal.start < 10 || traversal.start >= 45);
      } else if (span.row_index == 2) {
        EXPECT_LT(traversal.start, 15);
      }
    }
  }
  Workload cancelled_output(8, 8, {{.input_count = 1, .cancel = 5}});
  ASSERT_NO_FATAL_FAILURE(
      cancelled_output.Run([](const Traversal&) { return 10; }));
  EXPECT_EQ(cancelled_output.progress()[0].inputs, 1);
  EXPECT_EQ(cancelled_output.progress()[0].outputs, 1);
  EXPECT_TRUE(cancelled_output.progress()[0].published.empty());
  EXPECT_EQ(cancelled_output.progress()[0].finish, 10);
}

TEST(PackingWorkloadTest, RaggedArrivalsConserveEveryInputAndOutput) {
  for (size_t capacity : {1u, 3u, 8u, 31u, 64u}) {
    for (size_t chunk : {1u, 7u, 64u}) {
      for (size_t count : {1u, 5u, 17u, 65u}) {
        SCOPED_TRACE(capacity);
        SCOPED_TRACE(chunk);
        SCOPED_TRACE(count);
        std::vector<Request> requests(count);
        size_t inputs = 0, decode_inputs = 0;
        for (size_t i = 0; i < count; ++i) {
          requests[i] = {
              .input_count = (i * 37) % 131,
              .output_count = 1 + (i * 11) % 19,
              .arrival = static_cast<Tick>((i * 13) % 41),
          };
          inputs += requests[i].input_count;
          decode_inputs +=
              requests[i].output_count - (requests[i].input_count != 0);
        }
        Workload workload(capacity, chunk, requests);
        ASSERT_NO_FATAL_FAILURE(workload.Run([](const Traversal& traversal) {
          return static_cast<Tick>(1 + traversal.shape.token_capacity);
        }));
        size_t consumed_inputs = 0, consumed_decode = 0;
        for (const auto& traversal : workload.traversals()) {
          consumed_inputs += traversal.prefill_inputs;
          consumed_decode += traversal.decode_inputs;
        }
        EXPECT_EQ(consumed_inputs, inputs);
        EXPECT_EQ(consumed_decode, decode_inputs);
        for (size_t i = 0; i < count; ++i) {
          EXPECT_EQ(workload.progress()[i].inputs, requests[i].input_count);
          EXPECT_EQ(workload.progress()[i].outputs, requests[i].output_count);
          EXPECT_EQ(workload.progress()[i].published.size(),
                    requests[i].output_count);
          EXPECT_GT(workload.progress()[i].published.front(),
                    requests[i].arrival);
          EXPECT_NE(workload.progress()[i].finish, kNever);
        }
      }
    }
  }
}

}  // namespace
