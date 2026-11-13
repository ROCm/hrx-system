// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/net/cts/transfer_trial.h"

namespace iree::net::cts {
namespace {

void RunBenchmark(::benchmark::State& state,
                  const iree::async::cts::ProactorFactory& create_proactor,
                  TransferProgressPolicy policy,
                  TransferConsumerMode consumer_mode) {
  TransferTrialOptions options;
  options.connection_count = state.range(0);
  options.record_size = state.range(1);
  options.batch_size = state.range(2);
  options.window_size = state.range(3);
  options.fragment_count = state.range(4);
  options.measured_records = state.range(5);
  options.warmup_records = 256;
  options.progress_policy = policy;
  options.consumer_mode = consumer_mode;
  uint64_t records = 0;
  uint64_t bytes = 0;
  uint64_t commands = 0;
  uint64_t completions = 0;
  uint64_t progress = 0;
  uint64_t window_high_water = 0;
  uint64_t independent_progress = 0;
  uint64_t retained_messages_high_water = 0;
  uint64_t retained_bytes_high_water = 0;
  double elapsed_seconds = 0;
  iree_async_proactor_capabilities_t capabilities = 0;
  const TransferTrialMeasurement measurement = {
      +[](void* user_data) {
        static_cast<::benchmark::State*>(user_data)->ResumeTiming();
      },
      +[](void* user_data) {
        static_cast<::benchmark::State*>(user_data)->PauseTiming();
      },
      &state,
  };
  for (auto _ : state) {
    state.PauseTiming();
    TransferTrialResult result;
    iree::Status status(RunTransferTrial(GetTransportBackend(), create_proactor,
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
    records += result.records;
    bytes += result.payload_bytes;
    commands += result.command_messages;
    completions += result.source_completions;
    progress += result.progress_messages;
    window_high_water = std::max(window_high_water, result.window_high_water);
    independent_progress += result.independent_progress_messages;
    retained_messages_high_water = std::max(
        retained_messages_high_water, result.retained.messages_high_water);
    retained_bytes_high_water =
        std::max(retained_bytes_high_water, result.retained.bytes_high_water);
    elapsed_seconds += result.elapsed_seconds;
    capabilities = result.proactor_capabilities;
  }
  if (state.skipped()) {
    return;
  }
  state.SetItemsProcessed(records);
  state.SetBytesProcessed(bytes);
  using Counter = ::benchmark::Counter;
  state.counters["command_messages"] =
      Counter(commands, Counter::kAvgIterations);
  state.counters["source_completions"] =
      Counter(completions, Counter::kAvgIterations);
  state.counters["progress_messages"] =
      Counter(progress, Counter::kAvgIterations);
  state.counters["window_high_water"] = window_high_water;
  state.counters["independent_progress_messages"] =
      Counter(independent_progress, Counter::kAvgIterations);
  state.counters["retained_messages_high_water"] = retained_messages_high_water;
  state.counters["retained_bytes_high_water"] = retained_bytes_high_water;
  state.counters["proactor_capabilities"] = capabilities;
  state.counters["records_per_progress"] =
      static_cast<double>(records) / progress;
  state.counters["amortized_ns_per_record"] = elapsed_seconds * 1e9 / records;
  state.counters["payload_storage_bytes"] =
      2 * options.connection_count * options.record_size * options.batch_size;
}

class TransferBenchmarks {
 public:
  static void RegisterBenchmarks(
      const char* proactor_name,
      const iree::async::cts::ProactorFactory& create_proactor) {
    for (auto consumer_mode : {TransferConsumerMode::kInline,
                               TransferConsumerMode::kRetainedWindow}) {
      const char* consumer_name = consumer_mode == TransferConsumerMode::kInline
                                      ? "inline"
                                      : "retained_window";
      for (auto policy : {TransferProgressPolicy::kImmediate,
                          TransferProgressPolicy::kPollTurn}) {
        const char* policy_name = policy == TransferProgressPolicy::kImmediate
                                      ? "immediate"
                                      : "poll_turn";
        std::string name = std::string("CheckedTransfer/") +
                           GetTransportBackend().name + "/" + proactor_name +
                           "/same_process/" + consumer_name + "/" + policy_name;
        ::benchmark::RegisterBenchmark(
            name.c_str(),
            [create_proactor, policy,
             consumer_mode](::benchmark::State& state) {
              RunBenchmark(state, create_proactor, policy, consumer_mode);
            })
            ->ArgNames({"peers", "bytes", "batch", "window", "sg", "records"})
            ->Args({1, 64, 32, 128, 1, 2048})
            ->Args({1, 64, 1, 1, 1, 2048})
            ->Args({1, 64, 1, 128, 1, 2048})
            ->Args({16, 64, 1, 128, 1, 2048})
            ->Args({16, 64, 32, 128, 1, 2048})
            ->Args({1, 4096, 16, 128, 4, 1024})
            ->Iterations(1)
            ->UseManualTime()
            ->MeasureProcessCPUTime()
            ->Unit(::benchmark::kMicrosecond);
      }
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(TransferBenchmarks);

}  // namespace
}  // namespace iree::net::cts
