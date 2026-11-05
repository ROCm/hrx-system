// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/net/cts/expert_trial.h"

namespace iree::net::cts {
namespace {

void RunBenchmark(::benchmark::State& state,
                  const iree::async::cts::ProactorFactory& create_proactor,
                  CollectiveDelivery delivery, ExpertTraffic traffic) {
  ExpertTrialOptions options;
  options.rank_count = state.range(0);
  options.token_capacity = state.range(1);
  options.dispatch_bytes = state.range(2);
  options.scale_bytes = state.range(3);
  options.combine_bytes = state.range(4);
  options.block_size = state.range(5);
  options.window_size = state.range(6);
  options.depth = state.range(7);
  options.measured_rounds = state.range(8);
  options.warmup_rounds = options.depth;
  options.expert_count = 384;
  options.traffic = traffic;
  options.delivery = delivery;
  const TransferTrialMeasurement measurement = {
      +[](void* value) {
        static_cast<::benchmark::State*>(value)->ResumeTiming();
      },
      +[](void* value) {
        static_cast<::benchmark::State*>(value)->PauseTiming();
      },
      &state};
  ExpertTrialResult total;
  for (auto _ : state) {
    state.PauseTiming();
    ExpertTrialResult result;
    iree::Status status(RunExpertTrial(GetTransportBackend(), create_proactor,
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
    total.rounds += result.rounds;
    total.tokens += result.tokens;
    total.remote_tokens += result.remote_tokens;
    total.local_routes += result.local_routes;
    total.payload_bytes += result.payload_bytes;
    total.dispatch_bytes += result.dispatch_bytes;
    total.scale_bytes += result.scale_bytes;
    total.combine_bytes += result.combine_bytes;
    total.metadata_bytes += result.metadata_bytes;
    total.header_bytes += result.header_bytes;
    total.sends += result.sends;
    total.source_completions += result.source_completions;
    total.control_sends += result.control_sends;
    total.control_completions += result.control_completions;
    total.retained_handles += result.retained_handles;
    total.dispatch_seconds += result.dispatch_seconds;
    total.combine_seconds += result.combine_seconds;
    total.maximum_receiver_tokens =
        std::max(total.maximum_receiver_tokens, result.maximum_receiver_tokens);
    total.source_window_high_water = std::max(total.source_window_high_water,
                                              result.source_window_high_water);
    total.payload_storage_bytes = result.payload_storage_bytes;
  }
  if (state.skipped()) {
    return;
  }
  state.SetItemsProcessed(total.rounds);
  state.SetBytesProcessed(total.dispatch_bytes + total.scale_bytes +
                          total.combine_bytes);
  using Counter = ::benchmark::Counter;
  state.counters["amortized_us_per_round_trip"] =
      total.elapsed_seconds * 1e6 / total.rounds;
  state.counters["max_rank_dispatch_us_per_round"] =
      total.dispatch_seconds * 1e6 / total.rounds;
  state.counters["max_rank_combine_us_per_round"] =
      total.combine_seconds * 1e6 / total.rounds;
  state.counters["tokens"] = Counter(total.tokens, Counter::kAvgIterations);
  state.counters["remote_tokens"] =
      Counter(total.remote_tokens, Counter::kAvgIterations);
  state.counters["local_routes"] =
      Counter(total.local_routes, Counter::kAvgIterations);
  state.counters["activation_bytes"] =
      Counter(total.dispatch_bytes, Counter::kAvgIterations);
  state.counters["scale_bytes"] =
      Counter(total.scale_bytes, Counter::kAvgIterations);
  state.counters["return_bytes"] =
      Counter(total.combine_bytes, Counter::kAvgIterations);
  state.counters["route_bytes"] =
      Counter(total.metadata_bytes, Counter::kAvgIterations);
  state.counters["application_header_bytes"] =
      Counter(total.header_bytes, Counter::kAvgIterations);
  state.counters["data_sends"] = Counter(total.sends, Counter::kAvgIterations);
  state.counters["source_completions"] =
      Counter(total.source_completions, Counter::kAvgIterations);
  state.counters["control_sends"] =
      Counter(total.control_sends, Counter::kAvgIterations);
  state.counters["control_completions"] =
      Counter(total.control_completions, Counter::kAvgIterations);
  state.counters["retained_handles"] =
      Counter(total.retained_handles, Counter::kAvgIterations);
  state.counters["max_receiver_tokens"] = total.maximum_receiver_tokens;
  state.counters["source_window_high_water"] = total.source_window_high_water;
  state.counters["payload_storage_bytes"] = total.payload_storage_bytes;
}

class ExpertBenchmarks {
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
      for (auto traffic : {ExpertTraffic::kUniform, ExpertTraffic::kUneven,
                           ExpertTraffic::kHotspot}) {
        std::string name =
            std::string("ExpertRoundTrip/") + GetTransportBackend().name + "/" +
            proactor_name + "/same_process/" +
            (delivery == CollectiveDelivery::kMessage ? "message/"
                                                      : "registered/") +
            (traffic == ExpertTraffic::kUniform  ? "uniform"
             : traffic == ExpertTraffic::kUneven ? "uneven"
                                                 : "hotspot");
        auto* benchmark = ::benchmark::RegisterBenchmark(
            name.c_str(),
            [create_proactor, delivery, traffic](::benchmark::State& state) {
              RunBenchmark(state, create_proactor, delivery, traffic);
            });
        benchmark
            ->ArgNames({"ranks", "tokens", "dispatch_bytes", "scale_bytes",
                        "combine_bytes", "block_bytes", "window", "depth",
                        "rounds"})
            ->Args({4, 1, 5120, 160, 10240, 4096, 4, 1, 8})
            ->Args({4, 8, 5120, 160, 10240, 4096, 16, 2, 6});
        if (traffic == ExpertTraffic::kUniform) {
          benchmark->Args({4, 128, 2560, 160, 10240, 4096, 32, 2, 2})
              ->Args({4, 128, 2560, 160, 10240, 65536, 32, 2, 2})
              ->Args({2, 512, 5120, 160, 10240, 65536, 32, 1, 2})
              ->Args({8, 8, 5120, 160, 10240, 65536, 8, 2, 2});
        }
        benchmark->Iterations(1)
            ->UseManualTime()
            ->MeasureProcessCPUTime()
            ->Unit(::benchmark::kMicrosecond);
      }
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(ExpertBenchmarks);

}  // namespace
}  // namespace iree::net::cts
