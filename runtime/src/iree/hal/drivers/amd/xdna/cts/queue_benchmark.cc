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

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/driver.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/slab_pool.h"

namespace {

constexpr iree_device_size_t kBufferLength = 16 * sizeof(uint32_t);
constexpr uint64_t kWarmupIterationCount = 8;

enum class ProgramSequence {
  kMultiplyOnly,
  kAlternating,
};

enum class SubmissionMode {
  kDirect,
  kRecorded,
};

enum class Measurement {
  kSubmitOnly,
  kEndToEnd,
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

// Collection runs after the timed region. The driver still validates the native
// result protocol and emits every record; this consumer discards the payload.
struct CounterSink {
  // Stack-owned sink retained through the fixture's profiling session.
  iree_hal_resource_t resource;

  iree_hal_profile_sink_t* base() {
    return reinterpret_cast<iree_hal_profile_sink_t*>(this);
  }
  CounterSink() {
    static const iree_hal_profile_sink_vtable_t vtable = {
        +[](iree_hal_profile_sink_t*) {},
        +[](iree_hal_profile_sink_t*,
            const iree_hal_profile_chunk_metadata_t*) {
          return iree_ok_status();
        },
        +[](iree_hal_profile_sink_t*, const iree_hal_profile_chunk_metadata_t*,
            iree_host_size_t,
            const iree_const_byte_span_t*) { return iree_ok_status(); },
        +[](iree_hal_profile_sink_t*, const iree_hal_profile_chunk_metadata_t*,
            iree_status_code_t) { return iree_ok_status(); },
    };
    iree_hal_resource_initialize(&vtable, &resource);
  }
};

class XdnaQueueBenchmark : public benchmark::Fixture {
 public:
  void SetUp(benchmark::State& state) override {
    available_ = false;
    completion_value_ = 0;
    iree_status_t status = Initialize();
    if (!HandleStatus(state, status, "XDNA HAL device setup failed")) {
      return;
    }
    available_ = true;
  }

  void TearDown(benchmark::State& state) override {
    if (profiling_active_) {
      HandleStatus(state, iree_hal_device_profiling_end(device_),
                   "XDNA timer capture failed");
      profiling_active_ = false;
    }
    iree_hal_command_buffer_release(command_buffer_);
    command_buffer_ = nullptr;
    iree_hal_semaphore_release(gate_);
    iree_hal_semaphore_release(completion_);
    for (auto*& buffer : buffers_) {
      iree_hal_buffer_release(buffer);
      buffer = nullptr;
    }
    iree_hal_pool_release(memory_pool_);
    memory_pool_ = nullptr;
    iree_hal_device_group_release(device_group_);
    device_group_ = nullptr;
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
    IREE_RETURN_IF_ERROR(iree_hal_amd_driver_create(
        AMDF_NATIVE_LIFETIME_PROCESS, iree_allocator_system(), &driver_));
    iree_host_size_t device_count = 0;
    iree_hal_device_info_t* device_infos = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
        driver_, IREE_SV("xdna"), iree_allocator_system(), &device_count,
        &device_infos));
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
    IREE_RETURN_IF_ERROR(iree_hal_driver_create_device_by_uri(
        driver_, IREE_SV("xdna"), &params, iree_allocator_system(), &device_));
    // The selected proactor entry keeps the runner alive with the device.
    iree_async_proactor_pool_release(pool_);
    pool_ = nullptr;
    queue_ = iree_hal_device_queue(device_, /*family_ordinal=*/0,
                                   /*queue_ordinal=*/0);
    if (!queue_) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "XDNA HAL device has no queue 0");
    }

    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    iree_status_t status = iree_hal_device_group_create_from_device(
        device_, tracker, iree_allocator_system(), &device_group_);
    iree_async_frontier_tracker_release(tracker);
    IREE_RETURN_IF_ERROR(status);
    const iree_hal_pool_family_access_t family_access = {
        .family = iree_hal_queue_family(queue_),
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family_access,
        .host =
            {
                .access = IREE_HAL_MEMORY_ACCESS_ALL,
                .modes = IREE_HAL_MAPPING_MODE_SCOPED,
            },
    };
    iree_hal_slab_pool_options_t pool_options;
    iree_hal_slab_pool_options_initialize(&pool_options);
    IREE_RETURN_IF_ERROR(
        iree_hal_slab_pool_create(device_group_, scope, &pool_options,
                                  iree_allocator_system(), &memory_pool_));

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
    return iree_hal_pool_allocate_buffer(
        memory_pool_, iree_hal_make_pool_reservation_request(kBufferLength, 1),
        iree_infinite_timeout(), out_buffer);
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

  iree_status_t SubmitStreamingBatch(ProgramSequence sequence,
                                     iree_host_size_t dispatch_count,
                                     uint64_t* out_terminal_value) {
    uint64_t previous_value = completion_value_;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < dispatch_count && iree_status_is_ok(status); ++i) {
      const uint64_t value = ++completion_value_;
      const iree_hal_semaphore_list_t waits = {1, &completion_,
                                               &previous_value};
      status = Dispatch(waits, value, SelectProgram(sequence, i));
      previous_value = value;
    }
    *out_terminal_value = previous_value;
    return status;
  }

  iree_status_t RecordBatch(ProgramSequence sequence,
                            iree_host_size_t dispatch_count) {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue_), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, &command_buffer_));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer_));
    const iree_hal_buffer_ref_t bindings[] = {
        iree_hal_make_buffer_ref(buffers_[0], 0, kBufferLength),
        iree_hal_make_buffer_ref(buffers_[1], 0, kBufferLength),
        iree_hal_make_buffer_ref(buffers_[2], 0, kBufferLength),
    };
    const iree_hal_memory_barrier_t memory_barrier = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
    };
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .memory_barrier_count = 1,
        .memory_barriers = &memory_barrier,
    };
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < dispatch_count && iree_status_is_ok(status); ++i) {
      if (i) {
        status = iree_hal_command_buffer_barrier(command_buffer_, &barrier);
      }
      if (iree_status_is_ok(status)) {
        const iree_host_size_t program = SelectProgram(sequence, i);
        status = iree_hal_command_buffer_dispatch(
            command_buffer_, executables_[program], functions_[program],
            iree_hal_make_static_dispatch_config(1, 1, 1),
            iree_const_byte_span_empty(), {IREE_ARRAYSIZE(bindings), bindings},
            IREE_HAL_DISPATCH_FLAG_NONE);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_end(command_buffer_);
    }
    return status;
  }

  iree_status_t SubmitBatch(SubmissionMode mode, ProgramSequence sequence,
                            iree_host_size_t dispatch_count,
                            uint64_t* out_terminal_value) {
    if (mode == SubmissionMode::kDirect) {
      return SubmitStreamingBatch(sequence, dispatch_count, out_terminal_value);
    }
    uint64_t previous_value = completion_value_;
    uint64_t value = ++completion_value_;
    *out_terminal_value = value;
    return iree_hal_queue_execute(queue_, {1, &completion_, &previous_value},
                                  {1, &completion_, &value}, command_buffer_,
                                  iree_hal_buffer_binding_table_empty(),
                                  IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  }

  iree_status_t VerifyOutput(iree_hal_buffer_t* buffer,
                             ProgramSequence sequence,
                             iree_host_size_t dispatch_count) {
    std::array<uint32_t, 16> output;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_map_read(buffer, 0, output.data(), sizeof(output)));
    const iree_host_size_t terminal_program =
        SelectProgram(sequence, dispatch_count - 1);
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < output.size() && iree_status_is_ok(status); ++i) {
      const uint32_t lhs = static_cast<uint32_t>(i + 17);
      const uint32_t rhs = static_cast<uint32_t>(i * 3 + 5);
      const uint32_t expected = terminal_program ? lhs + rhs : lhs * rhs;
      if (output[i] != expected) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "output[%zu] = %u, expected %u", i, output[i],
                                  expected);
      }
    }
    return status;
  }

  void RunBatchBenchmark(benchmark::State& state, SubmissionMode mode,
                         ProgramSequence sequence, Measurement measurement,
                         bool capture_counters) {
    if (!available_) {
      return;
    }
    const auto dispatch_count = static_cast<iree_host_size_t>(state.range(0));
    if (sequence == ProgramSequence::kAlternating &&
        !HandleStatus(state,
                      LoadExecutable(
                          iree_hal_amd_xdna_test_add_i32_create(),
                          iree_hal_amd_xdna_test_add_i32_npu4_create(),
                          IREE_SV("add_i32"), &executables_[1], &functions_[1]),
                      "alternate XDNA executable setup failed")) {
      return;
    }
    if (mode == SubmissionMode::kRecorded &&
        !HandleStatus(state, RecordBatch(sequence, dispatch_count),
                      "XDNA command buffer recording failed")) {
      return;
    }
    if (capture_counters) {
      const iree_hal_device_profiling_options_t options = {
          .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
          .sink = counter_sink_.base(),
      };
      iree_status_t status = iree_hal_device_profiling_begin(device_, &options);
      if (iree_status_is_unimplemented(status) ||
          iree_status_is_unavailable(status) ||
          iree_status_is_permission_denied(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        state.SkipWithMessage("native XDNA timer capture is unavailable");
        return;
      }
      if (!HandleStatus(state, status, "XDNA timer capture setup failed")) {
        return;
      }
      profiling_active_ = true;
    }
    iree_status_t status = iree_ok_status();
    for (uint64_t i = 0; i < kWarmupIterationCount && iree_status_is_ok(status);
         ++i) {
      uint64_t value = 0;
      status = SubmitBatch(mode, sequence, dispatch_count, &value);
      if (iree_status_is_ok(status)) {
        status = Wait(value);
      }
    }
    if (!HandleStatus(state, status, "XDNA batch warmup failed") ||
        !HandleStatus(state,
                      VerifyOutput(buffers_[2], sequence, dispatch_count),
                      "XDNA batch warmup output mismatch")) {
      return;
    }
    for (auto _ : state) {
      uint64_t value = 0;
      if (!HandleStatus(state,
                        SubmitBatch(mode, sequence, dispatch_count, &value),
                        "XDNA batch submission failed")) {
        break;
      }
      if (measurement == Measurement::kSubmitOnly) {
        state.PauseTiming();
      }
      status = Wait(value);
      if (measurement == Measurement::kSubmitOnly) {
        state.ResumeTiming();
      }
      if (!HandleStatus(state, status, "XDNA batch completion failed")) {
        break;
      }
    }
    if (!state.skipped()) {
      HandleStatus(state, VerifyOutput(buffers_[2], sequence, dispatch_count),
                   "XDNA batch output mismatch");
    }
    state.SetItemsProcessed(state.iterations() * dispatch_count);
  }

  // Receives completed finite samples outside the measured queue path.
  CounterSink counter_sink_;
  // True after begin succeeds; teardown always consumes the session.
  bool profiling_active_ = false;
  // True when setup produced a complete runnable fixture.
  bool available_ = false;
  // Driver owning native discovery and device creation.
  iree_hal_driver_t* driver_ = nullptr;
  // Construction-time proactor pool, released after device selection.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Device dominating native queue and allocation state.
  iree_hal_device_t* device_ = nullptr;
  // Sealed topology used to qualify the benchmark's explicit memory pool.
  iree_hal_device_group_t* device_group_ = nullptr;
  // Family-scoped pool owning all benchmark buffers.
  iree_hal_pool_t* memory_pool_ = nullptr;
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
  // Immutable recording reused by recorded batch measurements.
  iree_hal_command_buffer_t* command_buffer_ = nullptr;
};

enum class MemoryPolicy {
  kCached,
  kFresh,
  kCapacityWait,
};

// Measures the full queue allocation lifetime with a real native dispatch. The
// inputs and recording stay resident; each output is an indirect transient
// binding. No host wait separates alloca, execute, and dealloca.
class XdnaQueueMemoryBenchmark : public XdnaQueueBenchmark {
 public:
  void TearDown(benchmark::State& state) override {
    iree_hal_pool_release(transient_pool_);
    transient_pool_ = nullptr;
    iree_hal_pool_release(backing_cache_);
    backing_cache_ = nullptr;
    XdnaQueueBenchmark::TearDown(state);
  }

 protected:
  iree_status_t InitializeMemory(MemoryPolicy policy) {
    iree_hal_fixed_block_pool_options_t options;
    iree_hal_fixed_block_pool_options_initialize(&options);
    options.block_size = kBufferLength;
    options.blocks_per_slab = 1;
    options.budget_limit = kBufferLength;
    iree_hal_slab_cache_options_t cache_options;
    iree_hal_slab_cache_options_initialize(&cache_options);
    cache_options.max_count = policy == MemoryPolicy::kFresh ? 0 : 1;
    IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_query_backing_request(
        memory_pool_, &options, &cache_options.slab));
    IREE_RETURN_IF_ERROR(
        iree_hal_slab_cache_create(memory_pool_, &cache_options,
                                   iree_allocator_system(), &backing_cache_));
    IREE_RETURN_IF_ERROR(iree_hal_fixed_block_pool_create(
        backing_cache_, &options, iree_allocator_system(), &transient_pool_));

    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue_), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 1, &command_buffer_));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer_));
    const iree_hal_buffer_ref_t bindings[] = {
        iree_hal_make_buffer_ref(buffers_[0], 0, kBufferLength),
        iree_hal_make_buffer_ref(buffers_[1], 0, kBufferLength),
        iree_hal_make_indirect_buffer_ref(0, 0, kBufferLength),
    };
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
        command_buffer_, executables_[0], functions_[0],
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), {IREE_ARRAYSIZE(bindings), bindings},
        IREE_HAL_DISPATCH_FLAG_NONE));
    return iree_hal_command_buffer_end(command_buffer_);
  }

  iree_status_t AllocateOutput(iree_hal_buffer_t** out_buffer,
                               uint64_t* out_ready_value) {
    const auto request =
        iree_hal_make_pool_reservation_request(kBufferLength, 0);
    uint64_t value = ++completion_value_;
    iree_status_t status =
        iree_hal_queue_alloca(queue_, {}, {1, &completion_, &value},
                              transient_pool_, 1, &request, out_buffer);
    if (iree_status_is_ok(status)) {
      *out_ready_value = value;
    }
    return status;
  }

  iree_status_t ReleaseOutput(iree_hal_buffer_t* buffer,
                              uint64_t* ready_value) {
    uint64_t value = ++completion_value_;
    iree_status_t status =
        iree_hal_queue_dealloca(queue_, {1, &completion_, ready_value},
                                {1, &completion_, &value}, 1, &buffer);
    if (iree_status_is_ok(status)) {
      *ready_value = value;
    }
    iree_hal_buffer_release(buffer);
    return status;
  }

  // Occupy the complete logical budget before the timed submission. Returning
  // this reservation below must unblock the new allocation without a host wait.
  iree_status_t PreparePressure(iree_hal_buffer_t** out_buffer) {
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(AllocateOutput(out_buffer, &value));
    return Wait(value);
  }

  iree_status_t SubmitLifetime(iree_hal_buffer_t* pressure_buffer,
                               bool verify_output, uint64_t* out_value) {
    iree_hal_buffer_t* output = nullptr;
    iree_status_t status = AllocateOutput(&output, out_value);
    if (iree_status_is_ok(status)) {
      const iree_hal_buffer_binding_t binding = {output, 0, kBufferLength};
      uint64_t value = ++completion_value_;
      status = iree_hal_queue_execute(
          queue_, {1, &completion_, out_value}, {1, &completion_, &value},
          command_buffer_, {1, &binding}, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
      if (iree_status_is_ok(status)) {
        *out_value = value;
      }
    }
    if (pressure_buffer) {
      // This release has no dependency on the allocation it enables and must
      // not signal a later value on the allocation's completion timeline.
      status = iree_status_join(
          status, iree_hal_queue_dealloca(queue_, {}, {}, 1, &pressure_buffer));
      iree_hal_buffer_release(pressure_buffer);
    }
    if (verify_output && iree_status_is_ok(status)) {
      status = Wait(*out_value);
      if (iree_status_is_ok(status)) {
        status = VerifyOutput(output, ProgramSequence::kMultiplyOnly, 1);
      }
    }
    if (output) {
      status = iree_status_join(status, ReleaseOutput(output, out_value));
    }
    return status;
  }

  void RunMemoryBenchmark(benchmark::State& state, MemoryPolicy policy,
                          Measurement measurement) {
    if (!available_ || !HandleStatus(state, InitializeMemory(policy),
                                     "XDNA memory benchmark setup failed")) {
      return;
    }
    // Verify the transient output before deallocation, outside measurement.
    // The remaining warmups exercise the ordinary submission-only chain.
    for (uint64_t i = 0; i < kWarmupIterationCount; ++i) {
      iree_hal_buffer_t* pressure_buffer = nullptr;
      iree_status_t status = policy == MemoryPolicy::kCapacityWait
                                 ? PreparePressure(&pressure_buffer)
                                 : iree_ok_status();
      uint64_t value = 0;
      if (iree_status_is_ok(status)) {
        status = SubmitLifetime(pressure_buffer, i == 0, &value);
      } else {
        iree_hal_buffer_release(pressure_buffer);
      }
      if (value) {
        status = iree_status_join(status, Wait(value));
      }
      if (!HandleStatus(state, status, "XDNA memory lifetime warmup failed")) {
        return;
      }
    }
    iree_hal_pool_stats_t before;
    iree_hal_pool_query_stats(memory_pool_, &before);
    for (auto _ : state) {
      iree_hal_buffer_t* pressure_buffer = nullptr;
      iree_status_t status = iree_ok_status();
      if (policy == MemoryPolicy::kCapacityWait) {
        state.PauseTiming();
        status = PreparePressure(&pressure_buffer);
        state.ResumeTiming();
      }
      uint64_t value = 0;
      if (iree_status_is_ok(status)) {
        status = SubmitLifetime(pressure_buffer, false, &value);
      } else {
        iree_hal_buffer_release(pressure_buffer);
      }
      if (measurement == Measurement::kSubmitOnly) {
        state.PauseTiming();
      }
      if (value) {
        status = iree_status_join(status, Wait(value));
      }
      if (measurement == Measurement::kSubmitOnly) {
        state.ResumeTiming();
      }
      if (!HandleStatus(state, status, "XDNA memory lifetime failed")) {
        break;
      }
    }
    iree_hal_pool_stats_t after;
    iree_hal_pool_query_stats(memory_pool_, &after);
    state.counters["native_slabs_per_iteration"] = benchmark::Counter(
        static_cast<double>(after.reserve_count - before.reserve_count),
        benchmark::Counter::kAvgIterations);
    state.SetItemsProcessed(state.iterations());
    if (!state.skipped()) {
      uint64_t value = 0;
      iree_status_t status = SubmitLifetime(nullptr, true, &value);
      if (value) {
        status = iree_status_join(status, Wait(value));
      }
      HandleStatus(state, status, "XDNA memory lifetime output mismatch");
    }
  }

  // Explicit idle retention above the native source; no hidden provider cache.
  iree_hal_pool_t* backing_cache_ = nullptr;
  // One-block child with a one-live-output budget, returned after every
  // dispatch.
  iree_hal_pool_t* transient_pool_ = nullptr;
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
  RunBatchBenchmark(state, SubmissionMode::kDirect,
                    ProgramSequence::kMultiplyOnly, Measurement::kSubmitOnly,
                    /*capture_counters=*/false);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   StreamingEndToEnd)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kDirect,
                    ProgramSequence::kMultiplyOnly, Measurement::kEndToEnd,
                    /*capture_counters=*/false);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   SwitchingSubmitOnly)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kDirect,
                    ProgramSequence::kAlternating, Measurement::kSubmitOnly,
                    /*capture_counters=*/false);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   SwitchingEndToEnd)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kDirect,
                    ProgramSequence::kAlternating, Measurement::kEndToEnd,
                    /*capture_counters=*/false);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   RecordedStreamingSubmitOnly)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kRecorded,
                    ProgramSequence::kMultiplyOnly, Measurement::kSubmitOnly,
                    /*capture_counters=*/false);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   RecordedStreamingEndToEnd)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kRecorded,
                    ProgramSequence::kMultiplyOnly, Measurement::kEndToEnd,
                    /*capture_counters=*/false);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   RecordedSwitchingSubmitOnly)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kRecorded,
                    ProgramSequence::kAlternating, Measurement::kSubmitOnly,
                    /*capture_counters=*/false);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   RecordedSwitchingEndToEnd)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kRecorded,
                    ProgramSequence::kAlternating, Measurement::kEndToEnd,
                    /*capture_counters=*/false);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   MeasuredStreamingSubmitOnly)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kDirect,
                    ProgramSequence::kMultiplyOnly, Measurement::kSubmitOnly,
                    /*capture_counters=*/true);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   MeasuredStreamingEndToEnd)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kDirect,
                    ProgramSequence::kMultiplyOnly, Measurement::kEndToEnd,
                    /*capture_counters=*/true);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark, MeasuredRecordedStreamingSubmitOnly)(
    benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kRecorded,
                    ProgramSequence::kMultiplyOnly, Measurement::kSubmitOnly,
                    /*capture_counters=*/true);
}

BENCHMARK_DEFINE_F(XdnaQueueBenchmark,
                   MeasuredRecordedStreamingEndToEnd)(benchmark::State& state) {
  RunBatchBenchmark(state, SubmissionMode::kRecorded,
                    ProgramSequence::kMultiplyOnly, Measurement::kEndToEnd,
                    /*capture_counters=*/true);
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
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(8)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, StreamingEndToEnd)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(8)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, SwitchingSubmitOnly)
    ->Arg(2)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(8)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, SwitchingEndToEnd)
    ->Arg(2)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(8)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, RecordedStreamingSubmitOnly)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(8)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, RecordedStreamingEndToEnd)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(8)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, RecordedSwitchingSubmitOnly)
    ->Arg(2)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(8)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(XdnaQueueBenchmark, RecordedSwitchingEndToEnd)
    ->Arg(2)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(8)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

// Eight warmup batches plus 32 measured batches of up to eight commands fit
// within both qualified native finite-result capacities. Fixed iterations also
// keep submit-only timing from expanding into unbounded real device work.
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, MeasuredStreamingSubmitOnly)
    ->Arg(1)
    ->Arg(8)
    ->Iterations(32)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, MeasuredStreamingEndToEnd)
    ->Arg(1)
    ->Arg(8)
    ->Iterations(32)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, MeasuredRecordedStreamingSubmitOnly)
    ->Arg(1)
    ->Arg(8)
    ->Iterations(32)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueBenchmark, MeasuredRecordedStreamingEndToEnd)
    ->Arg(1)
    ->Arg(8)
    ->Iterations(32)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(XdnaQueueMemoryBenchmark,
                   CachedSubmitOnly)(benchmark::State& state) {
  RunMemoryBenchmark(state, MemoryPolicy::kCached, Measurement::kSubmitOnly);
}
BENCHMARK_DEFINE_F(XdnaQueueMemoryBenchmark,
                   CachedEndToEnd)(benchmark::State& state) {
  RunMemoryBenchmark(state, MemoryPolicy::kCached, Measurement::kEndToEnd);
}
BENCHMARK_DEFINE_F(XdnaQueueMemoryBenchmark,
                   FreshSubmitOnly)(benchmark::State& state) {
  RunMemoryBenchmark(state, MemoryPolicy::kFresh, Measurement::kSubmitOnly);
}
BENCHMARK_DEFINE_F(XdnaQueueMemoryBenchmark,
                   FreshEndToEnd)(benchmark::State& state) {
  RunMemoryBenchmark(state, MemoryPolicy::kFresh, Measurement::kEndToEnd);
}
BENCHMARK_DEFINE_F(XdnaQueueMemoryBenchmark,
                   CapacityWaitSubmitOnly)(benchmark::State& state) {
  RunMemoryBenchmark(state, MemoryPolicy::kCapacityWait,
                     Measurement::kSubmitOnly);
}
BENCHMARK_DEFINE_F(XdnaQueueMemoryBenchmark,
                   CapacityWaitEndToEnd)(benchmark::State& state) {
  RunMemoryBenchmark(state, MemoryPolicy::kCapacityWait,
                     Measurement::kEndToEnd);
}

// These opt-in rows are excluded from the ordinary benchmark smoke filter.
// Fixed counts bound real device work even when completion is outside timing.
BENCHMARK_REGISTER_F(XdnaQueueMemoryBenchmark, CachedSubmitOnly)
    ->Iterations(128)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueMemoryBenchmark, CachedEndToEnd)
    ->Iterations(128)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueMemoryBenchmark, FreshSubmitOnly)
    ->Iterations(128)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueMemoryBenchmark, FreshEndToEnd)
    ->Iterations(128)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueMemoryBenchmark, CapacityWaitSubmitOnly)
    ->Iterations(128)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_REGISTER_F(XdnaQueueMemoryBenchmark, CapacityWaitEndToEnd)
    ->Iterations(128)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace
