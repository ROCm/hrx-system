// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "benchmark/benchmark.h"
#include "hrx_runtime_cxx.h"
#include "iree/base/tooling/flags.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"
#include "libhrx/src/libhrx/barrier.h"
#include "libhrx/src/libhrx/hrx_internal.h"
#include "libhrx/src/libhrx/mem_pool.h"

IREE_FLAG(string, hrx_benchmark_device, "amdgpu://0",
          "Exact device URI; task:// also supports fill and allocation rows.");
IREE_FLAG(bool, hrx_benchmark_active, true,
          "Request active device completion observation in both paths.");

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

void Check(hrx_status_t status) {
  if (!hrx_status_is_ok(status)) {
    std::fprintf(stderr, "%s\n", hrx::runtime::format_status(status).c_str());
    hrx_status_ignore(status);
    std::abort();
  }
}

void CheckNative(iree_status_t status) { Check(hrx_status_from_iree(status)); }

enum class Operation { kFill, kDispatch };
enum class EventOperation {
  kRecord,
  kAlternateRecord,
  kFillAndRecord,
  kTimedRecord
};

// One group, queue, and set of owners span the process. Native oracle calls use
// those exact resources; there is no second driver or separately placed buffer.
class Environment {
 public:
  Environment() {
    Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                             runtime_.for_output()));
    const hrx_device_create_params_t device = {
        .uri = FLAG_hrx_benchmark_device,
        .completion_wait_flags = FLAG_hrx_benchmark_active
                                     ? HRX_WAIT_FLAG_ACTIVE
                                     : HRX_WAIT_FLAG_NONE,
    };
    Check(hrx_device_group_create(runtime_, 1, &device, group_.for_output()));
    device_ = hrx_device_group_get_device(group_, 0);
    for (uint32_t i = 0; i < hrx_device_queue_family_count(device_); ++i) {
      hrx_queue_family_info_t family;
      Check(hrx_device_query_queue_family(device_, i, &family));
      queue_count_ += family.provisioned_queue_count;
      if (!queue_ && family.provisioned_queue_count &&
          (family.roles &
           (HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER)) ==
              (HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER)) {
        queue_ = hrx_device_get_queue(device_, i, 0);
      }
    }
    Require(queue_ != nullptr, "no combined dispatch/transfer queue");
    Check(hrx_semaphore_create(device_, 0, completion_.for_output()));
    const hrx_pool_family_access_t family = {
        .device = device_,
        .family_ordinal = hrx_queue_get_family_ordinal(queue_),
        .usage = HRX_BUFFER_USAGE_DEFAULT,
    };
    const hrx_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
        .host =
            {
                .access = HRX_MEMORY_ACCESS_ALL,
                .modes = HRX_MAPPING_MODE_SCOPED,
                .properties = HRX_HOST_MEMORY_PROPERTY_READ_CACHED |
                              HRX_HOST_MEMORY_PROPERTY_WRITE_BACK,
            },
    };
    Check(hrx_slab_pool_create(group_, scope, nullptr, source_.for_output()));
    const hrx_tlsf_pool_options_t options = {
        .slab_size = 65536,
        .alignment = 64,
    };
    hrx_slab_cache_options_t cache = {.max_count = 1};
    Check(hrx_tlsf_pool_query_backing_request(source_, &options, &cache.slab));
    Check(hrx_slab_cache_create(source_, &cache, cache_.for_output()));
    Check(hrx_tlsf_pool_create(cache_, &options, pool_.for_output()));
    hrx_semaphore_t semaphore = completion_;
    uint64_t allocated = 1;
    const hrx_semaphore_list_t signals = {&semaphore, &allocated, 1};
    const hrx_pool_reservation_request_t request = {4096, 64};
    Check(hrx_queue_alloca(queue_, nullptr, &signals, pool_, 1, &request,
                           output_.for_output()));
    tail_ = allocated;
    Wait();
    Check(hrx_device_query_queue_family_memory_scope(
        device_, family.family_ordinal, HRX_MEMORY_SITE_PROGRAM, &program_));
    PrepareBarriers(family.family_ordinal);
    label_ = std::string(FLAG_hrx_benchmark_device) +
             (FLAG_hrx_benchmark_active ? " wait=active" : " wait=blocking") +
             " linkage=static";
    if (device_->type == HRX_ACCELERATOR_GPU) {
      char architecture[64] = {};
      Check(hrx_device_get_property(device_, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                    architecture, sizeof(architecture)));
      const auto image = hrx_cts::FindAmdgpuExecutableTestImage(architecture);
      Require(image.file != nullptr, "no configured image for selected GPU");
      Check(hrx_executable_load_data(
          device_, image.file->data, image.file->size, "amdgpu",
          image.target_key.c_str(), executable_.for_output()));
      Check(hrx_executable_lookup_export_by_name(
          executable_, "hrx_store_output", &ordinal_));
      hrx_executable_export_info_t info;
      Check(hrx_executable_export_info(executable_, ordinal_, &info));
      config_ = {{1, 1, 1},
                 {info.workgroup_size[0], info.workgroup_size[1],
                  info.workgroup_size[2]},
                 0};
      label_ += std::string(" ") + architecture;
    }
  }

  ~Environment() {
    Wait();
    hrx_barrier_storage_deinitialize(&native_barriers_);
  }

  template <bool Native, Operation Op, bool Prepared = false>
  void Submit() {
    uint64_t next = tail_ + 1;
    expected_ = 0xA0000000u | static_cast<uint32_t>(next);
    if constexpr (Native) {
      auto* semaphore = completion_.get()->hal_semaphore;
      const iree_hal_semaphore_list_t waits = {1, &semaphore, &tail_};
      const iree_hal_semaphore_list_t signals = {1, &semaphore, &next};
      if constexpr (Op == Operation::kFill) {
        CheckNative(iree_hal_queue_fill(
            queue_->hal_queue, waits, signals, output_.get()->hal_buffer, 0,
            4096, &expected_, sizeof(expected_),
            Prepared ? &native_barriers_.barriers : nullptr,
            IREE_HAL_FILL_FLAG_NONE));
      } else {
        const auto binding = iree_hal_make_buffer_ref(output_.get()->hal_buffer,
                                                      0, sizeof(expected_));
        const iree_hal_dispatch_config_t config = {
            .workgroup_size = {config_.workgroup_size[0],
                               config_.workgroup_size[1],
                               config_.workgroup_size[2]},
            .workgroup_count = {1, 1, 1},
        };
        CheckNative(iree_hal_queue_dispatch(
            queue_->hal_queue, waits, signals,
            executable_.get()->hal_executable,
            iree_hal_executable_function_from_index(ordinal_), config,
            iree_make_const_byte_span(&expected_, sizeof(expected_)),
            {1, &binding}, Prepared ? &native_barriers_.barriers : nullptr,
            IREE_HAL_DISPATCH_FLAG_NONE));
      }
    } else {
      hrx_semaphore_t semaphore = completion_;
      const hrx_semaphore_list_t waits = {&semaphore, &tail_, 1};
      const hrx_semaphore_list_t signals = {&semaphore, &next, 1};
      if constexpr (Op == Operation::kFill) {
        Check(hrx_queue_fill(queue_, &waits, &signals, output_, 0, 4096,
                             &expected_, sizeof(expected_),
                             Prepared ? &barriers_ : nullptr));
      } else {
        const hrx_buffer_ref_t binding = {output_, 0, sizeof(expected_)};
        Check(hrx_queue_dispatch(
            queue_, &waits, &signals, executable_, ordinal_, &config_,
            &expected_, sizeof(expected_), &binding, 1,
            Prepared ? &barriers_ : nullptr, HRX_DISPATCH_FLAG_NONE));
      }
    }
    tail_ = next;
  }

  template <bool Native>
  void AllocateAndFree(size_t count) {
    uint64_t allocated = tail_ + 1;
    uint64_t released = allocated + 1;
    if constexpr (Native) {
      auto* semaphore = completion_.get()->hal_semaphore;
      const iree_hal_semaphore_list_t waits = {1, &semaphore, &tail_};
      const iree_hal_semaphore_list_t allocation = {1, &semaphore, &allocated};
      const iree_hal_semaphore_list_t release = {1, &semaphore, &released};
      std::array<iree_hal_pool_reservation_request_t, 16> requests;
      requests.fill({256, 16});
      CheckNative(iree_hal_queue_alloca(
          queue_->hal_queue, waits, allocation, pool_.get()->hal_pool, count,
          requests.data(), native_buffers_.data()));
      tail_ = allocated;
      CheckNative(iree_hal_queue_dealloca(queue_->hal_queue, allocation,
                                          release, count,
                                          native_buffers_.data()));
    } else {
      hrx_semaphore_t semaphore = completion_;
      const hrx_semaphore_list_t waits = {&semaphore, &tail_, 1};
      const hrx_semaphore_list_t allocation = {&semaphore, &allocated, 1};
      const hrx_semaphore_list_t release = {&semaphore, &released, 1};
      std::array<hrx_pool_reservation_request_t, 16> requests;
      requests.fill({256, 16});
      Check(hrx_queue_alloca(queue_, &waits, &allocation, pool_, count,
                             requests.data(), buffers_.data()));
      tail_ = allocated;
      Check(hrx_queue_dealloca(queue_, &allocation, &release, count,
                               buffers_.data()));
    }
    tail_ = released;
  }

  template <bool Native>
  void Release(size_t count) {
    for (size_t i = 0; i < count; ++i) {
      if constexpr (Native) {
        iree_hal_buffer_release(native_buffers_[i]);
        native_buffers_[i] = nullptr;
      } else {
        hrx_buffer_release(buffers_[i]);
        buffers_[i] = nullptr;
      }
    }
  }

  template <bool Native>
  auto QueryAtomics() {
    if constexpr (Native) {
      return iree_hal_pool_query_atomic_operations(
          pool_.get()->hal_pool, {program_.domain, program_.id});
    } else {
      return hrx_mem_pool_query_atomic_operations(pool_, program_);
    }
  }

  template <bool Native>
  auto QueryTransition() {
    if constexpr (Native) {
      return iree_hal_memory_transition_query(
          {static_cast<const iree_hal_memory_contract_t*>(table_.contract)},
          {pair_.cell_index});
    } else {
      return hrx_memory_transition_query(table_, pair_);
    }
  }

  void Wait() { Check(hrx_semaphore_wait(completion_, tail_, UINT64_MAX)); }
  void Verify() {
    void* data = nullptr;
    Check(hrx_buffer_map(output_, HRX_MAP_READ, 0, 4096, &data));
    Require(*static_cast<const uint32_t*>(data) == expected_,
            "output mismatch");
    Check(hrx_buffer_unmap(output_));
  }
  bool has_executable() const { return !!executable_; }

  template <bool AlternateProducers>
  void RecordEvents(benchmark::State& state) {
    using EventPtr =
        hrx::runtime::hrx_ptr<hrx_event_t, hrx_event_retain, hrx_event_release>;
    constexpr size_t kStreamCount = AlternateProducers ? 2 : 1;
    std::array<hrx::runtime::stream_ptr, kStreamCount> streams;
    std::array<hrx_timeline_point_t, kStreamCount> points;
    EventPtr event;
    Check(hrx_event_create(device_, HRX_EVENT_FLAG_DISABLE_TIMING,
                           event.for_output()));
    // Record actual accepted producer work before measuring snapshots. Setup
    // and final completion are outside the timed region; the measured path
    // only names an existing point, with no pending commands to flush.
    expected_ = 0xAABBCCDD;
    for (size_t i = 0; i < kStreamCount; ++i) {
      Check(hrx_stream_create(queue_, 0, streams[i].for_output()));
      Check(hrx_stream_fill_buffer(streams[i], output_, 0, sizeof(expected_),
                                   &expected_, sizeof(expected_)));
      Check(hrx_stream_synchronize(streams[i]));
      Check(hrx_stream_get_timeline_position(streams[i], &points[i]));
    }
    Check(hrx_event_record(event, streams[0]));
    size_t producer = 0;
    for (auto _ : state) {
      Check(hrx_event_record(event, streams[producer]));
      benchmark::DoNotOptimize(event.get());
      if constexpr (AlternateProducers) {
        producer ^= 1;
      }
    }
    for (size_t i = 0; i < kStreamCount; ++i) {
      hrx_timeline_point_t after;
      Check(hrx_stream_get_timeline_position(streams[i], &after));
      Require(after.semaphore == points[i].semaphore &&
                  after.value == points[i].value,
              "untimed recording added a queue operation");
    }
    Verify();
    Describe(state);
    state.SetItemsProcessed(state.iterations());
  }

  template <EventOperation Op, bool Complete>
  void EventBatches(benchmark::State& state) {
    using EventPtr =
        hrx::runtime::hrx_ptr<hrx_event_t, hrx_event_retain, hrx_event_release>;
    constexpr size_t kStreamCount =
        Op == EventOperation::kAlternateRecord ? 2 : 1;
    std::array<hrx::runtime::stream_ptr, kStreamCount> streams;
    std::array<hrx_timeline_point_t, kStreamCount> points;
    EventPtr event;
    Check(hrx_event_create(device_,
                           Op == EventOperation::kTimedRecord
                               ? HRX_EVENT_FLAG_NONE
                               : HRX_EVENT_FLAG_DISABLE_TIMING,
                           event.for_output()));
    expected_ = 0xAABBCCDD;
    for (auto& stream : streams) {
      Check(hrx_stream_create(queue_, 0, stream.for_output()));
      Check(hrx_stream_fill_buffer(stream, output_, 0, sizeof(expected_),
                                   &expected_, sizeof(expected_)));
      Check(hrx_stream_synchronize(stream));
    }
    const size_t batch = state.range(0);
    size_t producer = 0;
    const auto record = [&]() {
      if constexpr (Op == EventOperation::kFillAndRecord) {
        ++expected_;
        Check(hrx_stream_fill_buffer(streams[0], output_, 0, sizeof(expected_),
                                     &expected_, sizeof(expected_)));
      }
      Check(hrx_event_record(event, streams[producer]));
      if constexpr (Op == EventOperation::kAlternateRecord) {
        producer ^= 1;
      }
    };
    // Warm one full batch of storage before measuring. Every batch drains
    // its final point, bounding outstanding work even when a record submits
    // an operation. Only completion rows include that drain in timing.
    for (size_t i = 0; i < batch; ++i) {
      record();
    }
    Check(hrx_event_synchronize(event));
    for (size_t i = 0; i < kStreamCount; ++i) {
      Check(hrx_stream_get_timeline_position(streams[i], &points[i]));
    }
    for (auto _ : state) {
      for (size_t i = 0; i < batch; ++i) {
        record();
      }
      if constexpr (!Complete) {
        state.PauseTiming();
      }
      Check(hrx_event_synchronize(event));
      if constexpr (!Complete) {
        state.ResumeTiming();
      }
    }
    uint64_t advances = 0;
    for (size_t i = 0; i < kStreamCount; ++i) {
      Check(hrx_stream_synchronize(streams[i]));
      hrx_timeline_point_t after;
      Check(hrx_stream_get_timeline_position(streams[i], &after));
      advances += after.value - points[i].value;
    }
    if constexpr (Op == EventOperation::kTimedRecord) {
      float elapsed_ms = -1.0f;
      Check(hrx_event_elapsed_time(event, event, &elapsed_ms));
      Require(elapsed_ms == 0.0f, "same event has nonzero elapsed time");
    }
    Verify();
    Describe(state);
    const auto records = state.iterations() * batch;
    state.counters["producer_points_per_record"] =
        static_cast<double>(advances) / static_cast<double>(records);
    state.SetItemsProcessed(records);
  }

  void Describe(benchmark::State& state) {
    state.SetLabel(label_);
    hrx_pool_stats_t stats;
    hrx_mem_pool_query_stats(source_, &stats);
    state.counters["source_acquisitions"] = stats.reserve_count;
    state.counters["source_bytes"] = stats.bytes_committed;
    state.counters["provisioned_queues"] = queue_count_;
  }

 private:
  // Both benchmark paths consume the same prepared descriptors. Translation
  // happens once for the native oracle and on each public HRX enqueue.
  void PrepareBarriers(uint32_t family_ordinal) {
    hrx_memory_scope_t scope;
    Check(hrx_device_query_queue_family_memory_scope(
        device_, family_ordinal, HRX_MEMORY_SITE_QUEUE, &scope));
    table_ = hrx_buffer_transition_table(output_);
    Check(hrx_memory_transition_prepare_pair(
        table_, scope, program_, HRX_MEMORY_TRANSITION_RELEASE, &pair_));
    const auto any = hrx_memory_scope_any(scope);
    for (size_t side = 0; side < boundaries_.size(); ++side) {
      const auto action = side == 0 ? HRX_MEMORY_TRANSITION_ACQUIRE
                                    : HRX_MEMORY_TRANSITION_RELEASE;
      hrx_memory_transition_pair_t pair;
      Check(hrx_memory_transition_prepare_pair(table_, side == 0 ? any : scope,
                                               side == 0 ? scope : any, action,
                                               &pair));
      const auto transition = hrx_memory_transition_query(table_, pair);
      const auto effects = side == 0 ? transition.acquire : transition.release;
      Require(hrx_memory_effects_is_supported(effects) &&
                  !hrx_memory_effects_requires_program(effects),
              "prepared boundary requires an executable queue action");
      ranges_[side] = {
          .source_scope = access_.source_scope,
          .target_scope = access_.target_scope,
          .buffer_ref = {output_, 0, 4096},
          .recipe = hrx_memory_transition_recipe(table_, pair, action),
      };
      boundaries_[side] = {
          .source_stage_mask =
              side == 1 ? HRX_EXECUTION_STAGE_COMMAND_RETIRE : 0,
          .target_stage_mask =
              side == 0 ? HRX_EXECUTION_STAGE_COMMAND_ISSUE : 0,
          .effects = effects,
          .memory_barrier_count = 1,
          .memory_barriers = &access_,
          .buffer_barrier_count = 1,
          .buffer_barriers = &ranges_[side],
      };
      boundary_lists_[side] = {1, &boundaries_[side]};
    }
    barriers_ = {&boundary_lists_[0], &boundary_lists_[1]};
    CheckNative(hrx_barrier_storage_initialize(&barriers_, &native_barriers_));
  }

  // Service owner, released after the group and every dependent handle.
  hrx::runtime::runtime_ptr runtime_;
  // Single sealed execution domain shared by both measured paths.
  hrx::runtime::device_group_ptr group_;
  // Borrowed participant whose queues and executable use this group.
  hrx_device_t device_ = nullptr;
  // Borrowed combined compute/transfer queue; no benchmark acquisition.
  hrx_queue_t queue_ = nullptr;
  // Owned completion timeline shared by successive rows.
  hrx::runtime::semaphore_ptr completion_;
  // Explicit physical source retained through its descendants.
  hrx::runtime::mem_pool_ptr source_;
  // One returned-slab cache for all measurements.
  hrx::runtime::mem_pool_ptr cache_;
  // One steady-state suballocator shared by native and public calls.
  hrx::runtime::mem_pool_ptr pool_;
  // Persistent output also keeps a slab live during allocation rows.
  hrx::runtime::buffer_ptr output_;
  // Prepared finite store kernel; absent for CPU-only rows.
  hrx::runtime::hrx_ptr<hrx_executable_t, hrx_executable_retain,
                        hrx_executable_release>
      executable_;
  // Reflected store entry ordinal.
  uint32_t ordinal_ = 0;
  // Reflected workgroup geometry shared by both call paths.
  hrx_dispatch_config_t config_ = {};
  // Exact prepared program scope for constant-cost capability queries.
  hrx_memory_scope_t program_ = {};
  // Captured public table for the persistent output allocation.
  hrx_memory_transition_table_t table_ = {};
  // Exact queue-to-program pair prepared outside query timing.
  hrx_memory_transition_pair_t pair_ = {};
  // Shared access dependency for the read/write output at either boundary.
  const hrx_memory_barrier_t access_ = {
      .source_scope = HRX_ACCESS_SCOPE_MEMORY_WRITE,
      .target_scope =
          HRX_ACCESS_SCOPE_MEMORY_READ | HRX_ACCESS_SCOPE_MEMORY_WRITE,
  };
  // Concrete input/output ranges with their prepared resource recipes.
  std::array<hrx_buffer_barrier_t, 2> ranges_ = {};
  // Prepared acquire and release, respectively.
  std::array<hrx_barrier_t, 2> boundaries_ = {};
  // Borrowed boundary records retained for the process lifetime.
  std::array<hrx_barrier_list_t, 2> boundary_lists_ = {};
  // Public wrapper translated inside each measured explicit enqueue.
  hrx_queue_barriers_t barriers_ = {};
  // Equivalent native descriptors prepared before measurement.
  hrx_barrier_storage_t native_barriers_ = {};
  // Last accepted signal; final retirement is outside submit-only timing.
  uint64_t tail_ = 0;
  // Changing expected output checks actual execution and argument capture.
  uint32_t expected_ = 0;
  // Bounded public handle storage retained until accepted free completes.
  std::array<hrx_buffer_t, 16> buffers_ = {};
  // Equivalent native handle storage for the baseline.
  std::array<iree_hal_buffer_t*, 16> native_buffers_ = {};
  // Provisioned inventory captured once, before any benchmark work.
  size_t queue_count_ = 0;
  // Exact URI, observation policy, architecture, and linkage.
  std::string label_;
};

Environment& GetEnvironment() {
  static Environment environment;
  return environment;
}

template <bool Native, Operation Op, bool Complete, bool Prepared = false>
void Queue(benchmark::State& state) {
  auto& environment = GetEnvironment();
  if constexpr (Op == Operation::kDispatch) {
    if (!environment.has_executable()) {
      state.SkipWithMessage("dispatch rows require a GPU device URI");
      return;
    }
  }
  for (int i = 0; i < 16; ++i) {
    environment.Submit<Native, Op, Prepared>();
  }
  environment.Wait();
  environment.Verify();
  const size_t batch = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < batch; ++i) {
      environment.Submit<Native, Op, Prepared>();
    }
    if constexpr (!Complete) {
      state.PauseTiming();
    }
    environment.Wait();
    if constexpr (!Complete) {
      state.ResumeTiming();
    }
  }
  environment.Verify();
  environment.Describe(state);
  state.SetItemsProcessed(state.iterations() * batch);
}

template <bool Native, bool Complete>
void Allocation(benchmark::State& state) {
  auto& environment = GetEnvironment();
  const size_t count = state.range(0);
  environment.AllocateAndFree<Native>(count);
  environment.Wait();
  environment.Release<Native>(count);
  for (auto _ : state) {
    environment.AllocateAndFree<Native>(count);
    if constexpr (!Complete) {
      state.PauseTiming();
    }
    environment.Wait();
    environment.Release<Native>(count);
    if constexpr (!Complete) {
      state.ResumeTiming();
    }
  }
  environment.Describe(state);
  state.SetItemsProcessed(state.iterations() * count);
}

template <bool Native>
void Atomics(benchmark::State& state) {
  auto& environment = GetEnvironment();
  for (auto _ : state) {
    benchmark::DoNotOptimize(environment.QueryAtomics<Native>());
  }
  environment.Describe(state);
}

template <bool Native>
void Transition(benchmark::State& state) {
  auto& environment = GetEnvironment();
  for (auto _ : state) {
    benchmark::DoNotOptimize(environment.QueryTransition<Native>());
  }
  environment.Describe(state);
  state.SetItemsProcessed(state.iterations());
}

template <bool AlternateProducers>
void EventRecord(benchmark::State& state) {
  GetEnvironment().RecordEvents<AlternateProducers>(state);
}

template <EventOperation Op, bool Complete>
void EventBatch(benchmark::State& state) {
  GetEnvironment().EventBatches<Op, Complete>(state);
}

#define QUEUE_BENCHMARK(native, operation, complete, name)          \
  BENCHMARK_TEMPLATE(Queue, native, Operation::operation, complete) \
      ->Name(name)                                                  \
      ->Arg(1)                                                      \
      ->Arg(64)                                                     \
      ->UseRealTime()                                               \
      ->Unit(benchmark::kNanosecond)
QUEUE_BENCHMARK(true, kFill, false, "HAL/Fill/Submit");
QUEUE_BENCHMARK(false, kFill, false, "HRX/Fill/Submit");
QUEUE_BENCHMARK(true, kFill, true, "HAL/Fill/Complete");
QUEUE_BENCHMARK(false, kFill, true, "HRX/Fill/Complete");
QUEUE_BENCHMARK(true, kDispatch, false, "HAL/Dispatch/Submit");
QUEUE_BENCHMARK(false, kDispatch, false, "HRX/Dispatch/Submit");
QUEUE_BENCHMARK(true, kDispatch, true, "HAL/Dispatch/Complete");
QUEUE_BENCHMARK(false, kDispatch, true, "HRX/Dispatch/Complete");
#undef QUEUE_BENCHMARK
#define PREPARED_BENCHMARK(native, operation, name)                    \
  BENCHMARK_TEMPLATE(Queue, native, Operation::operation, false, true) \
      ->Name(name)                                                     \
      ->Arg(1)                                                         \
      ->Arg(64)                                                        \
      ->UseRealTime()                                                  \
      ->Unit(benchmark::kNanosecond)
PREPARED_BENCHMARK(true, kFill, "HAL/PreparedFill/Submit");
PREPARED_BENCHMARK(false, kFill, "HRX/PreparedFill/Submit");
PREPARED_BENCHMARK(true, kDispatch, "HAL/PreparedDispatch/Submit");
PREPARED_BENCHMARK(false, kDispatch, "HRX/PreparedDispatch/Submit");
#undef PREPARED_BENCHMARK
#define ALLOCATION_BENCHMARK(native, complete, name) \
  BENCHMARK_TEMPLATE(Allocation, native, complete)   \
      ->Name(name)                                   \
      ->Arg(1)                                       \
      ->Arg(8)                                       \
      ->Arg(16)                                      \
      ->UseRealTime()                                \
      ->Unit(benchmark::kNanosecond)
ALLOCATION_BENCHMARK(true, false, "HAL/Allocation/Submit");
ALLOCATION_BENCHMARK(false, false, "HRX/Allocation/Submit");
ALLOCATION_BENCHMARK(true, true, "HAL/Allocation/Complete");
ALLOCATION_BENCHMARK(false, true, "HRX/Allocation/Complete");
#undef ALLOCATION_BENCHMARK
BENCHMARK_TEMPLATE(Atomics, true)->Name("HAL/AtomicQuery");
BENCHMARK_TEMPLATE(Atomics, false)->Name("HRX/AtomicQuery");
BENCHMARK_TEMPLATE(Transition, true)->Name("HAL/TransitionQuery");
BENCHMARK_TEMPLATE(Transition, false)->Name("HRX/TransitionQuery");
BENCHMARK_TEMPLATE(EventRecord, false)
    ->Name("HRX/EventRecord/SameStream")
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(EventRecord, true)
    ->Name("HRX/EventRecord/AlternateStreams")
    ->UseRealTime()
    ->Unit(benchmark::kNanosecond);
#define EVENT_BENCHMARK(operation, name)                           \
  BENCHMARK_TEMPLATE(EventBatch, EventOperation::operation, false) \
      ->Name("HRX/EventBatch/" name "/Submit")                     \
      ->Arg(256)                                                   \
      ->UseRealTime()                                              \
      ->Unit(benchmark::kNanosecond);                              \
  BENCHMARK_TEMPLATE(EventBatch, EventOperation::operation, true)  \
      ->Name("HRX/EventBatch/" name "/Complete")                   \
      ->Arg(1)                                                     \
      ->UseRealTime()                                              \
      ->Unit(benchmark::kNanosecond)
EVENT_BENCHMARK(kRecord, "SameStream");
EVENT_BENCHMARK(kAlternateRecord, "AlternateStreams");
EVENT_BENCHMARK(kFillAndRecord, "FillAndRecord");
EVENT_BENCHMARK(kTimedRecord, "TimedRecord");
#undef EVENT_BENCHMARK

}  // namespace
