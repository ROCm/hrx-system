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
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"

namespace {

constexpr iree_device_size_t kBufferLength = 16 * sizeof(uint32_t);
constexpr uint64_t kWarmupIterationCount = 8;

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
    iree_status_t status = Initialize();
    if (!HandleStatus(state, status, "XDNA HAL device setup failed")) {
      available_ = false;
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
    iree_hal_executable_release(executable_);
    queue_ = nullptr;
    iree_hal_device_release(device_);
    iree_async_proactor_pool_release(pool_);
    iree_hal_driver_release(driver_);
    gate_ = nullptr;
    completion_ = nullptr;
    executable_ = nullptr;
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

    IREE_RETURN_IF_ERROR(LoadExecutable());
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

  iree_status_t LoadExecutable() {
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    if (targets->target_count != 1) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "expected one XDNA executable target, got %zu",
                              targets->target_count);
    }
    const bool halo = iree_string_view_equal(
        targets->targets[0].target_key, IREE_SV("amd.xdna.strix_halo.17f0_11"));
    const iree_file_toc_t* toc =
        halo ? iree_hal_amd_xdna_test_mul_i32_create()
             : iree_hal_amd_xdna_test_mul_i32_npu4_create();
    std::vector<uint8_t> bytes(toc[0].data, toc[0].data + toc[0].size);
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes.data(), bytes.size());
    IREE_RETURN_IF_ERROR(
        iree_hal_executable_load(iree_hal_device_queue_family(device_, 0),
                                 &targets->targets[0], &params, &executable_));
    return iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("mul_i32"), &function_);
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
                         uint64_t completion_value) {
    const iree_hal_buffer_ref_t bindings[] = {
        iree_hal_make_buffer_ref(buffers_[0], 0, kBufferLength),
        iree_hal_make_buffer_ref(buffers_[1], 0, kBufferLength),
        iree_hal_make_buffer_ref(buffers_[2], 0, kBufferLength),
    };
    const iree_hal_semaphore_list_t signals = {1, &completion_,
                                               &completion_value};
    return iree_hal_queue_dispatch(
        queue_, waits, signals, executable_, function_,
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), {IREE_ARRAYSIZE(bindings), bindings},
        IREE_HAL_DISPATCH_FLAG_NONE);
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
  // Loaded compiler-produced multiply executable.
  iree_hal_executable_t* executable_ = nullptr;
  // Reflected multiply function token.
  iree_hal_executable_function_t function_ =
      iree_hal_executable_function_invalid();
  // Two inputs and one output reused across all iterations.
  std::array<iree_hal_buffer_t*, 3> buffers_ = {};
  // Timeline marking checked native retirement.
  iree_hal_semaphore_t* completion_ = nullptr;
  // Host-signaled timeline used to force the unresolved-wait route.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Next shared gate and completion timeline value.
  uint64_t completion_value_ = 0;
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

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, ReadySubmitOnly)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, ReadyEndToEnd)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, UnresolvedSubmitOnly)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace

BENCHMARK_MAIN();
