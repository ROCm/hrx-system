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

  ~Environment() { Wait(); }

  template <bool Native, Operation Op>
  void Submit() {
    uint64_t next = tail_ + 1;
    expected_ = 0xA0000000u | static_cast<uint32_t>(next);
    if constexpr (Native) {
      auto* semaphore = completion_.get()->hal_semaphore;
      const iree_hal_semaphore_list_t waits = {1, &semaphore, &tail_};
      const iree_hal_semaphore_list_t signals = {1, &semaphore, &next};
      if constexpr (Op == Operation::kFill) {
        CheckNative(iree_hal_queue_fill(queue_->hal_queue, waits, signals,
                                        output_.get()->hal_buffer, 0, 4096,
                                        &expected_, sizeof(expected_), nullptr,
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
            {1, &binding}, nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
      }
    } else {
      hrx_semaphore_t semaphore = completion_;
      const hrx_semaphore_list_t waits = {&semaphore, &tail_, 1};
      const hrx_semaphore_list_t signals = {&semaphore, &next, 1};
      if constexpr (Op == Operation::kFill) {
        Check(hrx_queue_fill(queue_, &waits, &signals, output_, 0, 4096,
                             &expected_, sizeof(expected_), nullptr));
      } else {
        const hrx_buffer_ref_t binding = {output_, 0, sizeof(expected_)};
        Check(hrx_queue_dispatch(queue_, &waits, &signals, executable_,
                                 ordinal_, &config_, &expected_,
                                 sizeof(expected_), &binding, 1, nullptr,
                                 HRX_DISPATCH_FLAG_NONE));
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

  void Wait() { Check(hrx_semaphore_wait(completion_, tail_, UINT64_MAX)); }
  void Verify() {
    void* data = nullptr;
    Check(hrx_buffer_map(output_, HRX_MAP_READ, 0, 4096, &data));
    Require(*static_cast<const uint32_t*>(data) == expected_,
            "output mismatch");
    Check(hrx_buffer_unmap(output_));
  }
  bool has_executable() const { return !!executable_; }
  void Describe(benchmark::State& state) {
    state.SetLabel(label_);
    hrx_pool_stats_t stats;
    hrx_mem_pool_query_stats(source_, &stats);
    state.counters["source_acquisitions"] = stats.reserve_count;
    state.counters["source_bytes"] = stats.bytes_committed;
    state.counters["provisioned_queues"] = queue_count_;
  }

 private:
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

template <bool Native, Operation Op, bool Complete>
void Queue(benchmark::State& state) {
  auto& environment = GetEnvironment();
  if constexpr (Op == Operation::kDispatch) {
    if (!environment.has_executable()) {
      state.SkipWithMessage("dispatch rows require a GPU device URI");
      return;
    }
  }
  for (int i = 0; i < 16; ++i) {
    environment.Submit<Native, Op>();
  }
  environment.Wait();
  environment.Verify();
  const size_t batch = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < batch; ++i) {
      environment.Submit<Native, Op>();
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

}  // namespace
