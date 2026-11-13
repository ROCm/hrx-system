// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/net/cts/gather_trial.h"

namespace iree::net::cts {
namespace {

void RunBenchmark(::benchmark::State& state,
                  const iree::async::cts::ProactorFactory& create_proactor,
                  CollectiveDelivery delivery) {
  GatherTrialOptions options;
  options.rank_count = state.range(0);
  options.shard_size = state.range(1);
  options.block_size = state.range(2);
  options.window_size = state.range(3);
  options.depth = state.range(4);
  options.measured_rounds = state.range(5);
  options.warmup_rounds = options.depth;
  options.delivery = delivery;
  const TransferTrialMeasurement measurement = {
      +[](void* value) {
        static_cast<::benchmark::State*>(value)->ResumeTiming();
      },
      +[](void* value) {
        static_cast<::benchmark::State*>(value)->PauseTiming();
      },
      &state};
  GatherTrialResult total;
  for (auto _ : state) {
    state.PauseTiming();
    GatherTrialResult result;
    iree::Status status(RunGatherTrial(GetTransportBackend(), create_proactor,
                                       options, measurement, &result));
    state.ResumeTiming();
    if (!status.ok()) {
      if (!result.available && iree_status_is_unavailable(status.get())) {
        state.SkipWithMessage(status.ToString());
      } else {
        state.SkipWithError(status.ToString());
      }
      break;
    }
    state.SetIterationTime(result.elapsed_seconds);
    total.elapsed_seconds += result.elapsed_seconds;
    total.gathers += result.gathers;
    total.payload_bytes += result.payload_bytes;
    total.sends += result.sends;
    total.source_completions += result.source_completions;
    total.control_sends += result.control_sends;
    total.control_completions += result.control_completions;
    total.out_of_order_reads += result.out_of_order_reads;
    total.handles_high_water =
        std::max(total.handles_high_water, result.handles_high_water);
    total.source_window_high_water = std::max(total.source_window_high_water,
                                              result.source_window_high_water);
    total.payload_storage_bytes = result.payload_storage_bytes;
  }
  if (state.skipped()) {
    return;
  }
  using Counter = ::benchmark::Counter;
  state.SetItemsProcessed(total.gathers);
  state.SetBytesProcessed(total.payload_bytes);
  state.counters["amortized_us_per_gather"] =
      total.elapsed_seconds * 1e6 / total.gathers;
  state.counters["data_sends"] = Counter(total.sends, Counter::kAvgIterations);
  state.counters["source_completions"] =
      Counter(total.source_completions, Counter::kAvgIterations);
  state.counters["control_sends"] =
      Counter(total.control_sends, Counter::kAvgIterations);
  state.counters["control_completions"] =
      Counter(total.control_completions, Counter::kAvgIterations);
  state.counters["out_of_order_reads"] =
      Counter(total.out_of_order_reads, Counter::kAvgIterations);
  state.counters["handles_high_water"] = total.handles_high_water;
  state.counters["source_window_high_water"] = total.source_window_high_water;
  state.counters["payload_storage_bytes"] = total.payload_storage_bytes;
}

class GatherBenchmarks {
 public:
  static void RegisterBenchmarks(
      const char* proactor_name,
      const iree::async::cts::ProactorFactory& create_proactor) {
    for (auto delivery :
         {CollectiveDelivery::kMessage, CollectiveDelivery::kRegistered}) {
      if (delivery == CollectiveDelivery::kRegistered &&
          !GetTransportBackend().create_registered_factory) {
        continue;
      }
      std::string name =
          std::string("RetainedGather/") + GetTransportBackend().name + "/" +
          proactor_name + "/same_process/" +
          (delivery == CollectiveDelivery::kMessage ? "message" : "registered");
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [create_proactor, delivery](::benchmark::State& state) {
            RunBenchmark(state, create_proactor, delivery);
          })
          ->ArgNames({"ranks", "shard_bytes", "block_bytes", "window", "depth",
                      "gathers"})
          ->Args({2, 8192, 4096, 1, 1, 16})
          ->Args({2, 8192, 4096, 4, 4, 16})
          ->Args({4, 1048576, 4096, 32, 2, 4})
          ->Args({4, 1048576, 65536, 32, 2, 4})
          ->Args({8, 65540, 65536, 7, 3, 7})
          ->Iterations(1)
          ->UseManualTime()
          ->MeasureProcessCPUTime()
          ->Unit(::benchmark::kMicrosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(GatherBenchmarks);

}  // namespace
}  // namespace iree::net::cts
