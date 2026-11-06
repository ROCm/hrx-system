// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/net/cts/collective_trial.h"

namespace iree::net::cts {
namespace {

void RunBenchmark(::benchmark::State& state,
                  const iree::async::cts::ProactorFactory& create_proactor,
                  CollectiveDelivery delivery, CollectiveSchedule schedule) {
  CollectiveTrialOptions options;
  options.rank_count = state.range(0);
  options.tensor_size = state.range(1);
  options.block_size = state.range(2);
  options.window_size = state.range(3);
  options.measured_rounds = state.range(4);
  options.delivery = delivery;
  options.schedule = schedule;
  if (schedule == CollectiveSchedule::kPipeline) {
    options.pipeline_depth = state.range(5);
  }
  CollectiveTrialResult total;
  const TransferTrialMeasurement measurement = {
      +[](void* value) {
        static_cast<::benchmark::State*>(value)->ResumeTiming();
      },
      +[](void* value) {
        static_cast<::benchmark::State*>(value)->PauseTiming();
      },
      &state,
  };
  for (auto _ : state) {
    state.PauseTiming();
    CollectiveTrialResult result;
    iree::Status status(RunCollectiveTrial(
        GetTransportBackend(), create_proactor, options, measurement, &result));
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
    total.collectives += result.collectives;
    total.microbatches += result.microbatches;
    total.first_completion_seconds += result.first_completion_seconds;
    total.pipeline_high_water =
        std::max(total.pipeline_high_water, result.pipeline_high_water);
    total.payload_bytes += result.payload_bytes;
    total.sends += result.sends;
    total.source_completions += result.source_completions;
    total.control_sends += result.control_sends;
    total.control_completions += result.control_completions;
    total.source_window_high_water = std::max(total.source_window_high_water,
                                              result.source_window_high_water);
    total.window_high_water =
        std::max(total.window_high_water, result.window_high_water);
    total.payload_storage_bytes = result.payload_storage_bytes;
  }
  if (state.skipped()) {
    return;
  }
  state.SetItemsProcessed(total.collectives + total.microbatches);
  state.SetBytesProcessed(total.payload_bytes);
  using Counter = ::benchmark::Counter;
  if (schedule == CollectiveSchedule::kPipeline) {
    state.counters["amortized_us_per_microbatch"] =
        total.elapsed_seconds * 1e6 / total.microbatches;
    state.counters["first_completion_us"] =
        Counter(total.first_completion_seconds * 1e6, Counter::kAvgIterations);
    state.counters["pipeline_high_water"] = total.pipeline_high_water;
  } else {
    state.counters["amortized_us_per_collective"] =
        total.elapsed_seconds * 1e6 / total.collectives;
  }
  state.counters["data_sends"] = Counter(total.sends, Counter::kAvgIterations);
  state.counters["source_completions"] =
      Counter(total.source_completions, Counter::kAvgIterations);
  state.counters["control_sends"] =
      Counter(total.control_sends, Counter::kAvgIterations);
  state.counters["control_completions"] =
      Counter(total.control_completions, Counter::kAvgIterations);
  state.counters["window_high_water"] = total.window_high_water;
  state.counters["source_window_high_water"] = total.source_window_high_water;
  state.counters["payload_storage_bytes"] = total.payload_storage_bytes;
}

class CollectiveBenchmarks {
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
          std::string("TensorAllReduce/") + GetTransportBackend().name + "/" +
          proactor_name + "/same_process/" +
          (delivery == CollectiveDelivery::kMessage ? "message" : "registered");
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [create_proactor, delivery](::benchmark::State& state) {
            RunBenchmark(state, create_proactor, delivery,
                         CollectiveSchedule::kTensorAllReduce);
          })
          ->ArgNames(
              {"ranks", "tensor_bytes", "block_bytes", "window", "rounds"})
          ->Args({2, 8192, 4096, 1, 32})
          ->Args({4, 8192, 2048, 1, 16})
          ->Args({4, 8192, 128, 16, 8})
          ->Args({4, 1048576, 4096, 32, 4})
          ->Args({4, 1048576, 65536, 16, 4})
          ->Args({4, 1048576, 262144, 4, 4})
          ->Args({4, 1048592, 65536, 4, 4})
          ->Args({8, 65536, 8192, 1, 4})
          ->Iterations(1)
          ->UseManualTime()
          ->MeasureProcessCPUTime()
          ->Unit(::benchmark::kMicrosecond);
      name =
          std::string("Pipeline/") + GetTransportBackend().name + "/" +
          proactor_name + "/same_process/" +
          (delivery == CollectiveDelivery::kMessage ? "message" : "registered");
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [create_proactor, delivery](::benchmark::State& state) {
            RunBenchmark(state, create_proactor, delivery,
                         CollectiveSchedule::kPipeline);
          })
          ->ArgNames({"stages", "activation_bytes", "block_bytes", "window",
                      "microbatches", "depth"})
          ->Args({2, 8192, 4096, 1, 32, 1})
          ->Args({4, 8192, 4096, 4, 32, 1})
          ->Args({4, 8192, 4096, 4, 32, 4})
          ->Args({4, 1048576, 4096, 32, 16, 4})
          ->Args({4, 1048576, 65536, 32, 16, 4})
          ->Args({4, 1048576, 262144, 32, 16, 4})
          ->Args({4, 1048580, 65536, 7, 11, 3})
          ->Args({8, 65536, 8192, 8, 16, 8})
          ->Iterations(1)
          ->UseManualTime()
          ->MeasureProcessCPUTime()
          ->Unit(::benchmark::kMicrosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(CollectiveBenchmarks);

}  // namespace
}  // namespace iree::net::cts
