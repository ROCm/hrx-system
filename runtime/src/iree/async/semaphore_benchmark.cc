// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"

namespace {

// Registration order is independent of the order of timeline advancement.
enum class Order { kForward, kReverse, kInterleaved };
// Isolates acceptance, active cancellation, and completion plus tracker
// cleanup.
enum class Operation {
  kRegister,
  kCancel,
  kSignalAndCancel,
  kSignalBelowMinimum
};

// Uses the real software timeline and public timepoint APIs. Capacity is warmed
// before measurement; final draining and all unrelated setup are untimed. Each
// iteration measures at least 256 operations across independent wait sets,
// reported through items_per_second.
template <Operation operation, Order order>
void BM_Timepoints(benchmark::State& state) {
  const size_t count = static_cast<size_t>(state.range(0));
  iree_async_proactor_t* proactor = nullptr;
  IREE_CHECK_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  // Batch small wait sets to amortize clock/pause overhead without charging
  // setup to the operation under test.
  const size_t batch_count = count < 256 ? 256 / count : 1;
  std::vector<iree_async_semaphore_t*> semaphores(batch_count);
  for (auto& semaphore : semaphores) {
    IREE_CHECK_OK(iree_async_semaphore_create(
        proactor, 0, 0, iree_allocator_system(), &semaphore));
  }
  std::vector<iree_async_semaphore_timepoint_t> points(batch_count * count);
  std::vector<size_t> indices(count);
  std::iota(indices.begin(), indices.end(), size_t{0});
  for (size_t i = 0; i < count; ++i) {
    if constexpr (order == Order::kReverse) {
      indices[i] = count - i - 1;
    }
    if constexpr (order == Order::kInterleaved) {
      indices[i] = i * 73 % count;
    }
  }
  for (auto& point : points) {
    point.callback = [](void*, iree_async_semaphore_timepoint_t*,
                        iree_status_t status) { IREE_CHECK_OK(status); };
  }
  uint64_t base = 0;
  const size_t pending_offset =
      operation == Operation::kSignalBelowMinimum ? count : 0;
  auto register_points = [&] {
    for (size_t batch = 0; batch < batch_count; ++batch) {
      for (size_t index : indices) {
        IREE_CHECK_OK(iree_async_semaphore_acquire_timepoint(
            semaphores[batch], base + pending_offset + index + 1,
            &points[batch * count + index]));
      }
    }
  };
  auto drain = [&] {
    for (auto* semaphore : semaphores) {
      IREE_CHECK_OK(iree_async_semaphore_signal(
          semaphore, base + pending_offset + count, nullptr));
    }
  };
  register_points();
  drain();
  base += pending_offset + count;

  for (auto _ : state) {
    if constexpr (operation != Operation::kRegister) {
      state.PauseTiming();
      register_points();
      state.ResumeTiming();
    }
    if constexpr (operation == Operation::kRegister) {
      register_points();
    } else if constexpr (operation == Operation::kCancel) {
      for (auto& point : points) {
        benchmark::DoNotOptimize(
            iree_async_semaphore_cancel_timepoint(point.semaphore, &point));
      }
    } else {
      for (size_t batch = 0; batch < batch_count; ++batch) {
        for (size_t i = 0; i < count; ++i) {
          IREE_CHECK_OK(iree_async_semaphore_signal(semaphores[batch],
                                                    base + i + 1, nullptr));
          if constexpr (operation == Operation::kSignalAndCancel) {
            // Mirrors tracker finalization of an already-dispatched point
            // while later epochs remain pending on this same semaphore.
            benchmark::DoNotOptimize(iree_async_semaphore_cancel_timepoint(
                semaphores[batch], &points[batch * count + i]));
          }
        }
      }
    }
    state.PauseTiming();
    if constexpr (operation != Operation::kSignalAndCancel) {
      drain();
    }
    base += pending_offset + count;
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(batch_count * count));
  for (auto* semaphore : semaphores) {
    iree_async_semaphore_release(semaphore);
  }
  iree_async_proactor_release(proactor);
}

BENCHMARK_TEMPLATE(BM_Timepoints, Operation::kRegister, Order::kForward)
    ->Arg(1)
    ->Arg(8)
    ->Arg(128)
    ->Arg(2048)
    ->Iterations(200);
BENCHMARK_TEMPLATE(BM_Timepoints, Operation::kRegister, Order::kReverse)
    ->Arg(1)
    ->Arg(8)
    ->Arg(128)
    ->Arg(2048)
    ->Iterations(200);
BENCHMARK_TEMPLATE(BM_Timepoints, Operation::kRegister, Order::kInterleaved)
    ->Arg(1)
    ->Arg(8)
    ->Arg(128)
    ->Arg(2048)
    ->Iterations(200);
BENCHMARK_TEMPLATE(BM_Timepoints, Operation::kCancel, Order::kInterleaved)
    ->Arg(1)
    ->Arg(8)
    ->Arg(128)
    ->Arg(2048)
    ->Iterations(200);
BENCHMARK_TEMPLATE(BM_Timepoints, Operation::kSignalAndCancel,
                   Order::kInterleaved)
    ->Arg(1)
    ->Arg(8)
    ->Arg(128)
    ->Arg(2048)
    ->Iterations(200);

BENCHMARK_TEMPLATE(BM_Timepoints, Operation::kSignalBelowMinimum,
                   Order::kInterleaved)
    ->Arg(1)
    ->Arg(8)
    ->Arg(128)
    ->Arg(2048)
    ->Iterations(200);

}  // namespace
