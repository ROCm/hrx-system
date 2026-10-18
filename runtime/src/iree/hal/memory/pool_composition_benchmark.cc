// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace {

enum class PoolKind { kTLSF, kFixedBlock };

// Two child policies share ordinary cached native backing. Keeping an anchor
// measures acquisition from an already attached slab; leaving children empty
// measures the complete return/reacquire path through the shared cache.
class PoolComposition {
 public:
  PoolComposition(PoolKind kind, uint32_t block_count, uint32_t batch_size,
                  bool anchored, uint8_t frontier_count)
      : anchored_(anchored),
        requests_(batch_size,
                  iree_hal_make_pool_reservation_request(kBlockSize, 64)),
        reservations_(batch_size),
        infos_(batch_size) {
    const auto allocator = iree_allocator_system();
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), allocator, &proactor_));
    IREE_CHECK_OK(iree_async_notification_create(proactor_, 0, &notification_));
    IREE_CHECK_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator, &tracker_));
    iree_async_frontier_initialize(frontier(), frontier_count);
    for (uint8_t i = 0; i < frontier_count; ++i) {
      const auto axis = iree_async_axis_make_queue(1, 0, 0, i, 0);
      IREE_CHECK_OK(
          iree_async_frontier_tracker_register_axis(tracker_, axis, nullptr));
      frontier()->entries[i] = {axis, 10};
    }
    IREE_CHECK_OK(iree_hal_cpu_slab_provider_create(64, allocator, &provider_));
    IREE_CHECK_OK(iree_hal_passthrough_pool_create(
        {}, provider_, notification_, tracker_, allocator, &native_));
    const iree_device_size_t slab_size = kBlockSize * block_count;
    const iree_hal_fixed_block_pool_options_t fixed_options = {
        .block_size = kBlockSize,
        .blocks_per_slab = block_count,
        .frontier_capacity = 8,
        .alignment = 64,
    };
    const iree_hal_tlsf_pool_options_t tlsf_options = {
        .tlsf_options =
            {
                .range_length = slab_size,
                .alignment = 64,
                .initial_block_capacity = 128,
                .frontier_capacity = 8,
            },
    };
    iree_hal_slab_cache_options_t cache_options = {
        .max_count = 2,
    };
    if (kind == PoolKind::kFixedBlock) {
      IREE_CHECK_OK(iree_hal_fixed_block_pool_query_backing_request(
          native_, &fixed_options, &cache_options.slab));
    } else {
      IREE_CHECK_OK(iree_hal_tlsf_pool_query_backing_request(
          native_, &tlsf_options, &cache_options.slab));
    }
    IREE_CHECK_OK(iree_hal_slab_cache_create(native_, &cache_options, allocator,
                                             &cache_));
    for (size_t i = 0; i < children_.size(); ++i) {
      if (kind == PoolKind::kFixedBlock) {
        IREE_CHECK_OK(iree_hal_fixed_block_pool_create(
            cache_, &fixed_options, allocator, &children_[i]));
      } else {
        IREE_CHECK_OK(iree_hal_tlsf_pool_create(cache_, &tlsf_options,
                                                allocator, &children_[i]));
      }
      if (anchored_) {
        Acquire(children_[i], 1);
        anchors_[i] = reservations_[0];
      }
      Cycle(i);
    }
  }

  ~PoolComposition() {
    for (uint8_t i = 0; i < frontier()->entry_count; ++i) {
      iree_async_frontier_tracker_advance(tracker_, frontier()->entries[i].axis,
                                          frontier()->entries[i].epoch);
    }
    for (size_t i = 0; i < children_.size(); ++i) {
      if (anchored_) {
        iree_hal_pool_release_reservations(children_[i], 1, &anchors_[i],
                                           nullptr);
      }
      iree_hal_pool_release(children_[i]);
    }
    iree_hal_pool_release(cache_);
    iree_hal_pool_release(native_);
    iree_hal_slab_provider_release(provider_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_async_proactor_release(proactor_);
  }

  void Cycle(size_t child_index) {
    auto* child = children_[child_index];
    Acquire(child, requests_.size());
    benchmark::DoNotOptimize(reservations_.data());
    iree_hal_pool_release_reservations(child, requests_.size(),
                                       reservations_.data(), frontier());
  }

  uint64_t native_slabs() const {
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(native_, &stats);
    return stats.reserve_count;
  }

 private:
  void Acquire(iree_hal_pool_t* pool, iree_host_size_t count) {
    bool completed = false;
    IREE_CHECK_OK(iree_hal_pool_acquire_reservations(
        pool, count, requests_.data(), frontier(), reservations_.data(),
        infos_.data(),
        {
            .fn =
                [](void* user_data, iree_status_t status,
                   iree_hal_pool_acquire_result_t result) {
                  IREE_CHECK_OK(status);
                  if (result != IREE_HAL_POOL_ACQUIRE_OK &&
                      result != IREE_HAL_POOL_ACQUIRE_OK_FRESH) {
                    std::abort();
                  }
                  *static_cast<bool*>(user_data) = true;
                },
            .user_data = &completed,
        },
        IREE_HAL_POOL_RESERVE_FLAG_NONE));
    // This benchmark uses prepared CPU storage with inline completion.
    if (!completed) {
      std::abort();
    }
  }

  iree_async_frontier_t* frontier() {
    return reinterpret_cast<iree_async_frontier_t*>(frontier_storage_);
  }

  // Logical reservation size; slab size scales with the block-count argument.
  static constexpr iree_device_size_t kBlockSize = 4096;
  // Whether each child keeps a live reservation throughout measurement.
  bool anchored_;
  // Platform notification owner; no polling thread is created.
  iree_async_proactor_t* proactor_ = nullptr;
  // Shared capacity notification for the native leaf.
  iree_async_notification_t* notification_ = nullptr;
  // Completion table shared by all participating pools.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Native CPU storage provider.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Ordinary native allocation source.
  iree_hal_pool_t* native_ = nullptr;
  // Idle backing shared between children.
  iree_hal_pool_t* cache_ = nullptr;
  // Child policies alternated by successive measured cycles.
  std::array<iree_hal_pool_t*, 2> children_ = {};
  // Optional live reservations that prevent child backing return.
  std::array<iree_hal_pool_reservation_t, 2> anchors_ = {};
  // Stable public acquisition inputs, allocated before measurement.
  std::vector<iree_hal_pool_reservation_request_t> requests_;
  // Complete batched outputs, allocated before measurement.
  std::vector<iree_hal_pool_reservation_t> reservations_;
  // Exact per-reservation outcome storage, allocated before measurement.
  std::vector<iree_hal_pool_acquire_info_t> infos_;
  // Requester and release proof shared by each benchmark cycle.
  alignas(IREE_ASYNC_FRONTIER_ALIGNMENT) uint8_t
      frontier_storage_[sizeof(iree_async_frontier_t) +
                        4 * sizeof(iree_async_frontier_entry_t)] = {};
};

template <PoolKind kind>
void BM_AcquireRelease(benchmark::State& state) {
  const uint32_t batch_size = static_cast<uint32_t>(state.range(1));
  PoolComposition pools(kind, static_cast<uint32_t>(state.range(0)), batch_size,
                        state.range(2) != 0,
                        static_cast<uint8_t>(state.range(3)));
  size_t child_index = 0;
  for (auto _ : state) {
    pools.Cycle(child_index);
    child_index ^= 1;
  }
  state.SetItemsProcessed(state.iterations() * batch_size);
  state.counters["native_slabs"] = pools.native_slabs();
}

void AddArguments(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"blocks", "batch", "anchored", "frontier"});
  for (int64_t blocks : {64, 512, 4096}) {
    for (int64_t batch : {1, 8, 32}) {
      for (int64_t anchored : {0, 1}) {
        for (int64_t frontier : {0, 4}) {
          benchmark->Args({blocks, batch, anchored, frontier});
        }
      }
    }
  }
}

BENCHMARK_TEMPLATE(BM_AcquireRelease, PoolKind::kTLSF)->Apply(AddArguments);
BENCHMARK_TEMPLATE(BM_AcquireRelease, PoolKind::kFixedBlock)
    ->Apply(AddArguments);

}  // namespace
