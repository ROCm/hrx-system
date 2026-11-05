// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/net/cts/direct_transfer_trial.h"

namespace iree::net::cts {
namespace {

void RunBenchmark(::benchmark::State& state,
                  const iree::async::cts::ProactorFactory& create_proactor,
                  TransferConsumerMode consumer_mode,
                  TransferProgressPolicy progress_policy,
                  DirectTransferLayout layout, uint32_t warmup_records) {
  DirectTransferTrialOptions options;
  options.connection_count = state.range(0);
  options.record_size = state.range(1);
  options.window_size = state.range(2);
  options.fragment_count = state.range(3);
  options.measured_records = state.range(4);
  options.warmup_records = warmup_records;
  options.consumer_mode = consumer_mode;
  options.progress_policy = progress_policy;
  options.layout = layout;
  DirectTransferTrialResult total;
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
    DirectTransferTrialResult result;
    iree::Status status(RunDirectTransferTrial(
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
    total.records += result.records;
    total.payload_bytes += result.payload_bytes;
    total.source_completions += result.source_completions;
    total.progress_messages += result.progress_messages;
    total.independent_progress_messages += result.independent_progress_messages;
    total.retained_records += result.retained_records;
    total.payload_storage_bytes = result.payload_storage_bytes;
    total.window_high_water =
        std::max(total.window_high_water, result.window_high_water);
  }
  if (state.skipped()) {
    return;
  }
  state.SetItemsProcessed(total.records);
  state.SetBytesProcessed(total.payload_bytes);
  using Counter = ::benchmark::Counter;
  state.counters["source_completions"] =
      Counter(total.source_completions, Counter::kAvgIterations);
  state.counters["progress_messages"] =
      Counter(total.progress_messages, Counter::kAvgIterations);
  state.counters["independent_progress_messages"] =
      Counter(total.independent_progress_messages, Counter::kAvgIterations);
  state.counters["retained_records"] =
      Counter(total.retained_records, Counter::kAvgIterations);
  state.counters["window_high_water"] = total.window_high_water;
  state.counters["records_per_progress"] =
      static_cast<double>(total.records) / total.progress_messages;
  state.counters["amortized_ns_per_record"] =
      total.elapsed_seconds * 1e9 / total.records;
  state.counters["payload_storage_bytes"] = total.payload_storage_bytes;
}

class DirectTransferBenchmarks {
 public:
  static void RegisterBenchmarks(
      const char* proactor_name,
      const iree::async::cts::ProactorFactory& create_proactor) {
    for (auto mode : {TransferConsumerMode::kInline,
                      TransferConsumerMode::kRetainedWindow}) {
      const char* consumer_name =
          mode == TransferConsumerMode::kInline ? "inline" : "retained_window";
      for (auto policy : {TransferProgressPolicy::kImmediate,
                          TransferProgressPolicy::kPollTurn}) {
        const char* policy_name = policy == TransferProgressPolicy::kImmediate
                                      ? "immediate"
                                      : "poll_turn";
        std::string name = std::string("CheckedDirectTransfer/") +
                           GetTransportBackend().name + "/" + proactor_name +
                           "/same_process/" + consumer_name + "/" + policy_name;
        ::benchmark::RegisterBenchmark(
            name.c_str(),
            [create_proactor, mode, policy](::benchmark::State& state) {
              RunBenchmark(state, create_proactor, mode, policy,
                           DirectTransferLayout::kContiguous, 256);
            })
            ->ArgNames({"peers", "bytes", "window", "sg", "records"})
            ->Args({1, 64, 1, 1, 512})
            ->Args({1, 64, 63, 1, 1024})
            ->Args({1, 64, 64, 1, 1024})
            ->Args({1, 64, 65, 1, 1024})
            ->Args({1, 64, 128, 1, 1024})
            ->Args({16, 64, 128, 1, 1024})
            ->Args({1, 4096, 128, 4, 512})
            ->Args({1, 4099, 65, 129, 257})
            ->Iterations(1)
            ->UseManualTime()
            ->MeasureProcessCPUTime()
            ->Unit(::benchmark::kMicrosecond);
      }
    }
    for (auto layout : {DirectTransferLayout::kContiguous,
                        DirectTransferLayout::kPermutedPages}) {
      std::string name =
          std::string("CheckedPagedTransfer/") + GetTransportBackend().name +
          "/" + proactor_name + "/same_process/" +
          (layout == DirectTransferLayout::kContiguous ? "contiguous"
                                                       : "permuted");
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [create_proactor, layout](::benchmark::State& state) {
            RunBenchmark(state, create_proactor,
                         TransferConsumerMode::kRetainedWindow,
                         TransferProgressPolicy::kPollTurn, layout, 3);
          })
          ->ArgNames({"peers", "bytes", "window", "sg", "records"})
          ->Args({1, 16384, 1, 4, 32})
          ->Args({1, 262144, 4, 64, 17})
          ->Args({1, 266240, 4, 65, 17})
          ->Args({1, 528384, 4, 129, 17})
          ->Args({1, 1048576, 4, 16, 17})
          ->Args({4, 65536, 4, 16, 17})
          ->Iterations(1)
          ->UseManualTime()
          ->MeasureProcessCPUTime()
          ->Unit(::benchmark::kMicrosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(DirectTransferBenchmarks);

}  // namespace
}  // namespace iree::net::cts
