// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/xdna/driver.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"

namespace {

constexpr iree_device_size_t kBufferLength = 16 * sizeof(uint32_t);
constexpr uint64_t kWarmupIterationCount = 8;
constexpr uint64_t kStreamingDepth = 8;

enum class ProgramSequence {
  kMultiplyOnly,
  kAlternating,
};

bool HandleStatus(benchmark::State& state, iree_status_t status,
                  const char* message) {
  if (iree_status_is_ok(status)) {
    return true;
  }
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  state.SkipWithError(message);
  return false;
}

class XdnaQueueBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    available_ = false;
    completion_value_ = 0;
    stream_ordinal_ = 0;
    iree_status_t status = Initialize();
    if (!HandleStatus(state, status, "XDNA HAL device setup failed")) {
      return;
    }
    available_ = true;
  }

  void TearDown(benchmark::State& state) override {
    (void)state;
    iree_hal_semaphore_release(gate_);
    iree_hal_semaphore_release(completion_);
    for (auto*& buffer : buffers_) {
      iree_hal_buffer_release(buffer);
      buffer = nullptr;
    }
    for (auto*& executable : executables_) {
      iree_hal_executable_release(executable);
      executable = nullptr;
    }
    queue_ = nullptr;
    iree_hal_device_release(device_);
    iree_async_proactor_pool_release(pool_);
    iree_hal_driver_release(driver_);
    gate_ = nullptr;
    completion_ = nullptr;
    device_ = nullptr;
    pool_ = nullptr;
    driver_ = nullptr;
  }

 protected:
  iree_status_t Initialize() {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_xdna_driver_create(iree_allocator_system(), &driver_));
    iree_host_size_t device_count = 0;
    iree_hal_device_info_t* device_infos = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
        driver_, iree_allocator_system(), &device_count, &device_infos));
    iree_allocator_free(iree_allocator_system(), device_infos);
    if (!device_count) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no native XDNA endpoint");
    }

    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
        1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    iree_hal_device_create_params_t params =
        iree_hal_device_create_params_default();
    params.proactor_pool = pool_;
    params.event_sink = iree_hal_device_event_sink_stderr();
    IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
        driver_, &params, iree_allocator_system(), &device_));
    // The selected proactor entry keeps the runner alive with the device.
    iree_async_proactor_pool_release(pool_);
    pool_ = nullptr;
    queue_ = iree_hal_device_queue(device_, /*family_ordinal=*/0,
                                   /*queue_ordinal=*/0);
    if (!queue_) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "XDNA HAL device has no queue 0");
    }

    IREE_RETURN_IF_ERROR(
        LoadExecutable(iree_hal_amd_xdna_test_mul_i32_create(),
                       iree_hal_amd_xdna_test_mul_i32_npu4_create(),
                       IREE_SV("mul_i32"), &executables_[0], &functions_[0]));
    for (auto*& buffer : buffers_) {
      IREE_RETURN_IF_ERROR(MakeBuffer(&buffer));
    }
    std::array<uint32_t, 16> lhs;
    std::array<uint32_t, 16> rhs;
    for (iree_host_size_t i = 0; i < lhs.size(); ++i) {
      lhs[i] = static_cast<uint32_t>(i + 17);
      rhs[i] = static_cast<uint32_t>(i * 3 + 5);
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_map_write(buffers_[0], 0, lhs.data(), sizeof(lhs)));
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_map_write(buffers_[1], 0, rhs.data(), sizeof(rhs)));

    IREE_RETURN_IF_ERROR(MakeSemaphore(&completion_));
    return MakeSemaphore(&gate_);
  }

  iree_status_t LoadExecutable(const iree_file_toc_t* halo_toc,
                               const iree_file_toc_t* npu4_toc,
                               iree_string_view_t function_name,
                               iree_hal_executable_t** out_executable,
                               iree_hal_executable_function_t* out_function) {
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    if (targets->target_count != 1) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "expected one XDNA executable target, got %zu",
                              targets->target_count);
    }
    const bool halo = iree_string_view_equal(
        targets->targets[0].target_key, IREE_SV("amd.xdna.strix_halo.17f0_11"));
    const iree_file_toc_t* toc = halo ? halo_toc : npu4_toc;
    std::vector<uint8_t> bytes(toc[0].data, toc[0].data + toc[0].size);
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes.data(), bytes.size());
    IREE_RETURN_IF_ERROR(iree_hal_executable_load(
        iree_hal_device_queue_family(device_, 0), &targets->targets[0], &params,
        out_executable));
    return iree_hal_executable_lookup_function_by_name(
        *out_executable, function_name, out_function);
  }

  iree_status_t MakeBuffer(iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING;
    return iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, kBufferLength, out_buffer);
  }

  iree_status_t MakeSemaphore(iree_hal_semaphore_t** out_semaphore) {
    return iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore);
  }

  iree_status_t Dispatch(iree_hal_semaphore_list_t waits,
                         uint64_t completion_value,
                         iree_host_size_t program_ordinal = 0) {
    const iree_hal_buffer_ref_t bindings[] = {
        iree_hal_make_buffer_ref(buffers_[0], 0, kBufferLength),
        iree_hal_make_buffer_ref(buffers_[1], 0, kBufferLength),
        iree_hal_make_buffer_ref(buffers_[2], 0, kBufferLength),
    };
    const iree_hal_semaphore_list_t signals = {1, &completion_,
                                               &completion_value};
    return iree_hal_queue_dispatch(
        queue_, waits, signals, executables_[program_ordinal],
        functions_[program_ordinal],
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), {IREE_ARRAYSIZE(bindings), bindings},
        /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE);
  }

  iree_status_t Wait(uint64_t value) {
    return iree_hal_semaphore_wait(completion_, value, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  bool WarmReady(benchmark::State& state) {
    for (uint64_t i = 0; i < kWarmupIterationCount; ++i) {
      uint64_t value = ++completion_value_;
      if (!HandleStatus(state, Dispatch({}, value),
                        "ready dispatch warmup failed") ||
          !HandleStatus(state, Wait(value),
                        "ready dispatch warmup wait failed")) {
        return false;
      }
    }
    return true;
  }

  bool WarmUnresolved(benchmark::State& state) {
    for (uint64_t i = 0; i < kWarmupIterationCount; ++i) {
      uint64_t value = ++completion_value_;
      const iree_hal_semaphore_list_t waits = {1, &gate_, &value};
      if (!HandleStatus(state, Dispatch(waits, value),
                        "unresolved dispatch warmup failed") ||
          !HandleStatus(state,
                        iree_hal_semaphore_signal(gate_, value,
                                                  /*frontier=*/nullptr),
                        "unresolved dispatch warmup signal failed") ||
          !HandleStatus(state, Wait(value),
                        "unresolved dispatch warmup wait failed")) {
        return false;
      }
    }
    return true;
  }

  iree_host_size_t SelectProgram(ProgramSequence sequence,
                                 uint64_t dispatch_ordinal) const {
    return sequence == ProgramSequence::kAlternating ? dispatch_ordinal % 2 : 0;
  }

  bool SubmitStreamingBatch(benchmark::State& state, ProgramSequence sequence,
                            uint64_t* out_terminal_value) {
    uint64_t previous_value = completion_value_;
    for (uint64_t i = 0; i < kStreamingDepth; ++i) {
      const uint64_t value = ++completion_value_;
      const iree_hal_semaphore_list_t waits = {1, &completion_,
                                               &previous_value};
      if (!HandleStatus(
              state,
              Dispatch(waits, value, SelectProgram(sequence, stream_ordinal_)),
              "streaming dispatch failed")) {
        return false;
      }
      previous_value = value;
      ++stream_ordinal_;
    }
    *out_terminal_value = previous_value;
    return true;
  }

  bool WarmStreaming(benchmark::State& state, ProgramSequence sequence) {
    if (sequence == ProgramSequence::kAlternating && !executables_[1] &&
        !HandleStatus(state,
                      LoadExecutable(
                          iree_hal_amd_xdna_test_add_i32_create(),
                          iree_hal_amd_xdna_test_add_i32_npu4_create(),
                          IREE_SV("add_i32"), &executables_[1], &functions_[1]),
                      "alternate XDNA executable setup failed")) {
      return false;
    }
    uint64_t terminal_value = 0;
    return SubmitStreamingBatch(state, sequence, &terminal_value) &&
           HandleStatus(state, Wait(terminal_value),
                        "streaming dispatch warmup wait failed");
  }

  // True when setup produced a complete runnable fixture.
  bool available_ = false;
  // Driver owning native discovery and device creation.
  iree_hal_driver_t* driver_ = nullptr;
  // Construction-time proactor pool, released after device selection.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Device dominating native queue and allocation state.
  iree_hal_device_t* device_ = nullptr;
  // Borrowed provisioned queue used by every measured dispatch.
  iree_hal_queue_t* queue_ = nullptr;
  // Independently loaded multiply and add executables.
  std::array<iree_hal_executable_t*, 2> executables_ = {};
  // Reflected function tokens corresponding to |executables_|.
  std::array<iree_hal_executable_function_t, 2> functions_ = {
      iree_hal_executable_function_invalid(),
      iree_hal_executable_function_invalid(),
  };
  // Two inputs and one output reused across all iterations.
  std::array<iree_hal_buffer_t*, 3> buffers_ = {};
  // Timeline marking checked native retirement.
  iree_hal_semaphore_t* completion_ = nullptr;
  // Host-signaled timeline used to force the unresolved-wait route.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Next shared gate and completion timeline value.
  uint64_t completion_value_ = 0;
  // Next program position in streaming benchmark sequences.
  uint64_t stream_ordinal_ = 0;
};

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   ReadySubmitOnly)(benchmark::State& state) {
  if (!available_ || !WarmReady(state)) {
    return;
  }
  for (auto _ : state) {
    uint64_t value = ++completion_value_;
    if (!HandleStatus(state, Dispatch({}, value), "ready dispatch failed")) {
      break;
    }
    state.PauseTiming();
    iree_status_t status = Wait(value);
    state.ResumeTiming();
    if (!HandleStatus(state, status, "ready dispatch wait failed")) {
      break;
    }
  }
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark, ReadyEndToEnd)(benchmark::State& state) {
  if (!available_ || !WarmReady(state)) {
    return;
  }
  for (auto _ : state) {
    const uint64_t value = ++completion_value_;
    if (!HandleStatus(state, Dispatch({}, value), "ready dispatch failed") ||
        !HandleStatus(state, Wait(value), "ready dispatch wait failed")) {
      break;
    }
  }
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   UnresolvedSubmitOnly)(benchmark::State& state) {
  if (!available_ || !WarmUnresolved(state)) {
    return;
  }
  for (auto _ : state) {
    uint64_t value = ++completion_value_;
    const iree_hal_semaphore_list_t waits = {1, &gate_, &value};
    if (!HandleStatus(state, Dispatch(waits, value),
                      "unresolved dispatch failed")) {
      break;
    }
    state.PauseTiming();
    iree_status_t status =
        iree_hal_semaphore_signal(gate_, value, /*frontier=*/nullptr);
    if (iree_status_is_ok(status)) {
      status = Wait(value);
    }
    state.ResumeTiming();
    if (!HandleStatus(state, status, "unresolved dispatch completion failed")) {
      break;
    }
  }
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   StreamingSubmitOnly)(benchmark::State& state) {
  if (!available_ || !WarmStreaming(state, ProgramSequence::kMultiplyOnly)) {
    return;
  }
  for (auto _ : state) {
    uint64_t terminal_value = 0;
    if (!SubmitStreamingBatch(state, ProgramSequence::kMultiplyOnly,
                              &terminal_value)) {
      break;
    }
    state.PauseTiming();
    iree_status_t status = Wait(terminal_value);
    state.ResumeTiming();
    if (!HandleStatus(state, status, "streaming dispatch wait failed")) {
      break;
    }
  }
  state.SetItemsProcessed(state.iterations() * kStreamingDepth);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   StreamingEndToEnd)(benchmark::State& state) {
  if (!available_ || !WarmStreaming(state, ProgramSequence::kMultiplyOnly)) {
    return;
  }
  for (auto _ : state) {
    uint64_t terminal_value = 0;
    if (!SubmitStreamingBatch(state, ProgramSequence::kMultiplyOnly,
                              &terminal_value) ||
        !HandleStatus(state, Wait(terminal_value),
                      "streaming dispatch wait failed")) {
      break;
    }
  }
  state.SetItemsProcessed(state.iterations() * kStreamingDepth);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   SwitchingSubmitOnly)(benchmark::State& state) {
  if (!available_ || !WarmStreaming(state, ProgramSequence::kAlternating)) {
    return;
  }
  for (auto _ : state) {
    uint64_t terminal_value = 0;
    if (!SubmitStreamingBatch(state, ProgramSequence::kAlternating,
                              &terminal_value)) {
      break;
    }
    state.PauseTiming();
    iree_status_t status = Wait(terminal_value);
    state.ResumeTiming();
    if (!HandleStatus(state, status, "switching dispatch wait failed")) {
      break;
    }
  }
  state.SetItemsProcessed(state.iterations() * kStreamingDepth);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   SwitchingEndToEnd)(benchmark::State& state) {
  if (!available_ || !WarmStreaming(state, ProgramSequence::kAlternating)) {
    return;
  }
  for (auto _ : state) {
    uint64_t terminal_value = 0;
    if (!SubmitStreamingBatch(state, ProgramSequence::kAlternating,
                              &terminal_value) ||
        !HandleStatus(state, Wait(terminal_value),
                      "switching dispatch wait failed")) {
      break;
    }
  }
  state.SetItemsProcessed(state.iterations() * kStreamingDepth);
}

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, ReadySubmitOnly)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, ReadyEndToEnd)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, UnresolvedSubmitOnly)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, StreamingSubmitOnly)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, StreamingEndToEnd)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, SwitchingSubmitOnly)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, SwitchingEndToEnd)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
