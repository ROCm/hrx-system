// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <initializer_list>

#include "benchmark/benchmark.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/storage/memory.h"

namespace {

// Optional allocation traffic observation, excluded from the timing families.
// Native shared-section payload is mapped by the source, not host-allocated.
struct AllocationTraffic {
  // Successful host allocation requests, including complete reallocations.
  std::atomic<uint64_t> calls{0};
  // Cumulative requested host bytes, not retained memory or physical payload.
  std::atomic<uint64_t> bytes{0};

  static iree_status_t Control(void* context, iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto* self = static_cast<AllocationTraffic*>(context);
    const auto allocator = iree_allocator_system();
    iree_status_t status =
        allocator.ctl(allocator.self, command, parameters, pointer);
    if (iree_status_is_ok(status) && command != IREE_ALLOCATOR_COMMAND_FREE) {
      self->calls.fetch_add(1, std::memory_order_relaxed);
      self->bytes.fetch_add(
          static_cast<const iree_allocator_alloc_params_t*>(parameters)
              ->byte_length,
          std::memory_order_relaxed);
    }
    return status;
  }
};

// Measures the shipping serving path over real Task virtual mappings. One
// iteration commits a complete range and joins mapping publication. Unmapping,
// optional cold disposal, payload validation, and teardown are outside timing.
// Each benchmark process uses the same 64 MiB physical-parent policy.
template <bool RetainBacking, bool ObserveAllocations>
void BM_CommitBacking(benchmark::State& state) {
  AllocationTraffic traffic;
  const iree_allocator_t allocator =
      ObserveAllocations
          ? iree_allocator_t{&traffic, AllocationTraffic::Control}
          : iree_allocator_system();
  loom_serve_host_t* host = nullptr;
  IREE_CHECK_OK(
      loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
  const loom_serve_device_options_t options = {
      .driver = loom_serve_host_task_driver(host),
      .uri = IREE_SV("task"),
  };
  loom_serve_device_group_t* group = nullptr;
  IREE_CHECK_OK(loom_serve_device_group_allocate(
      host, 1, &options, iree_allocator_system(), &group));
  loom_serve_host_release(host);
  auto* device = loom_serve_device_group_at(group, 0);
  auto* hal_group = loom_serve_device_group(device);
  auto* queue = loom_serve_device_transfer_queue(device);
  iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(queue),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
  };
  const iree_hal_pool_scope_t scope = {.family_count = 1, .families = &family};
  iree_hal_pool_t* source = nullptr;
  const iree_hal_slab_pool_options_t source_options = {};
  IREE_CHECK_OK(iree_hal_slab_pool_create(hal_group, scope, &source_options,
                                          allocator, &source));
  family.interfaces = 0;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_CHECK_OK(iree_hal_virtual_buffer_query_capabilities(
      hal_group, scope, {1, &source}, allocator, &capabilities));
  constexpr uint64_t kParentBytes = 64 * 1024 * 1024;
  loom_serve_memory_geometry_t geometry;
  IREE_CHECK_OK(loom_serve_memory_pool_query_geometry(&capabilities, source,
                                                      kParentBytes, &geometry));
  const auto weight_geometry = geometry;
  geometry.mapping_size = geometry.minimum_size * state.range(0);
  const uint64_t length = geometry.mapping_size * state.range(1);
  loom_serve_memory_pool_t* pool = nullptr;
  IREE_CHECK_OK(loom_serve_memory_pool_allocate(
      hal_group, scope, source, queue, weight_geometry, geometry,
      iree_max(kParentBytes, length), allocator, &pool));
  loom_runtime_storage_statistics_t statistics = {};
  loom_serve_virtual_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(loom_serve_virtual_buffer_allocate(
      pool, geometry, length, geometry.granularity, &statistics, &buffer));
  // Establish the same empty virtual reservation and, optionally, retained
  // physical backing before timing either transition.
  IREE_CHECK_OK(loom_serve_virtual_buffer_commit(buffer, 0, length));
  loom_serve_virtual_buffer_begin_trim(buffer);
  IREE_CHECK_OK(loom_serve_virtual_buffer_trim(buffer));
  if (!RetainBacking) {
    loom_serve_memory_pool_trim(pool, 0);
    IREE_CHECK_OK(loom_serve_memory_pool_await(pool));
  }
  uint64_t calls = 0, bytes = 0;
  for (auto _ : state) {
    uint64_t calls_before = 0, bytes_before = 0;
    if constexpr (ObserveAllocations) {
      calls_before = traffic.calls.load(std::memory_order_relaxed);
      bytes_before = traffic.bytes.load(std::memory_order_relaxed);
    }
    IREE_CHECK_OK(loom_serve_virtual_buffer_commit(buffer, 0, length));
    state.PauseTiming();
    if constexpr (ObserveAllocations) {
      calls += traffic.calls.load(std::memory_order_relaxed) - calls_before;
      bytes += traffic.bytes.load(std::memory_order_relaxed) - bytes_before;
    }
    loom_serve_virtual_buffer_begin_trim(buffer);
    IREE_CHECK_OK(loom_serve_virtual_buffer_trim(buffer));
    if (!RetainBacking) {
      loom_serve_memory_pool_trim(pool, 0);
      IREE_CHECK_OK(loom_serve_memory_pool_await(pool));
    }
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * state.range(1));
  state.counters["page_bytes"] = geometry.mapping_size;
  state.counters["mapped_bytes"] = length;
  if (ObserveAllocations) {
    state.counters["host_allocations"] =
        benchmark::Counter(calls, benchmark::Counter::kAvgIterations);
    state.counters["host_requested_bytes"] =
        benchmark::Counter(bytes, benchmark::Counter::kAvgIterations);
  }

  // Validate real first/last page data outside the measured transition.
  IREE_CHECK_OK(loom_serve_virtual_buffer_commit(buffer, 0, length));
  state.counters["native_bytes"] =
      loom_serve_memory_pool_statistics(pool).committed_bytes;
  auto* handle = loom_serve_virtual_buffer_handle(buffer);
  iree_hal_semaphore_t* ready = nullptr;
  IREE_CHECK_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
  uint64_t value = 0;
  for (uint64_t offset : {uint64_t{0}, length - geometry.mapping_size}) {
    const uint32_t expected = 0x12345678u ^ static_cast<uint32_t>(offset);
    ++value;
    IREE_CHECK_OK(iree_hal_queue_fill(queue, {}, {1, &ready, &value}, handle,
                                      offset, sizeof(expected), &expected,
                                      sizeof(expected), nullptr, 0));
    IREE_CHECK_OK(iree_hal_semaphore_wait(ready, value, iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE));
    uint32_t observed = 0;
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = handle,
                .source_offset = offset,
                .target = &observed,
                .length = sizeof(observed),
            },
    };
    ++value;
    IREE_CHECK_OK(iree_hal_queue_transfer(queue, {}, {1, &ready, &value}, 1,
                                          &download, nullptr));
    IREE_CHECK_OK(iree_hal_semaphore_wait(ready, value, iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE));
    IREE_CHECK_OK(observed == expected
                      ? iree_ok_status()
                      : iree_make_status(IREE_STATUS_DATA_LOSS,
                                         "mapped payload differs"));
  }
  iree_hal_semaphore_release(ready);
  IREE_CHECK_OK(loom_serve_virtual_buffer_free(buffer));
  loom_serve_memory_pool_trim(pool, 0);
  IREE_CHECK_OK(loom_serve_memory_pool_await(pool));
  loom_serve_memory_pool_free(pool);
  iree_hal_pool_release(source);
  IREE_CHECK_OK(loom_serve_device_group_free(group));
}

// Small first instances bound the ordinary benchmark smoke. Larger cases are
// explicitly selected for scaling studies with fixed iteration counts.
void Cases(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"mapping_granules", "pages"})
      ->Args({1, 1})
      ->Args({1, 64})
      ->Args({1, 8192})
      ->Args({16, 512})
      ->UseRealTime();
}
BENCHMARK_TEMPLATE(BM_CommitBacking, false, false)
    ->Name("CommitCold")
    ->Apply(Cases);
BENCHMARK_TEMPLATE(BM_CommitBacking, true, false)
    ->Name("RebindRetained")
    ->Apply(Cases);
BENCHMARK_TEMPLATE(BM_CommitBacking, false, true)
    ->Name("CountCold")
    ->Apply(Cases);
BENCHMARK_TEMPLATE(BM_CommitBacking, true, true)
    ->Name("CountRetained")
    ->Apply(Cases);

}  // namespace
