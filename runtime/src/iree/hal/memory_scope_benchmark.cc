// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/memory_scope.h"

namespace {

// Six contracts model the upper end of the common 2-6 device working set. They
// share one sealed-group namespace but retain independent backing facts, as
// buffers from distinct pools do.
class TransitionTableSet {
 public:
  static constexpr iree_host_size_t kTableCount = 6;

  TransitionTableSet() {
    const iree_hal_buffer_binding_layout_t binding_layout = {};
    for (iree_host_size_t i = 0; i < kTableCount; ++i) {
      IREE_CHECK_OK(iree_hal_memory_contract_create(
          this, /*scope_count=*/8, &binding_layout, iree_allocator_system(),
          &contracts_[i]));
      for (uint32_t scope_id = 2; scope_id < 8; ++scope_id) {
        contracts_[i]->scopes[scope_id].interfaces =
            1u << IREE_HAL_BUFFER_INTERFACE_HOST;
        contracts_[i]->scopes[scope_id].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
        const iree_hal_atomic_operation_capabilities_t operations = {
            .device_scope_32 = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL,
            .system_scope_64 = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                               IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
        };
        iree_hal_memory_scope_access_set_atomic_operations(
            &contracts_[i]->scopes[scope_id], operations);
      }
      iree_hal_memory_contract_finalize_atomic_operations(contracts_[i]);
      IREE_CHECK_OK(iree_hal_memory_contract_initialize_transitions(
          contracts_[i], QueryRangedPair, nullptr));
      tables_[i] = {contracts_[i]};
    }
    IREE_CHECK_OK(iree_hal_memory_transition_prepare_pair(
        tables_[0], scope(2), scope(4), IREE_HAL_MEMORY_TRANSITION_RELEASE,
        &pair_));
  }

  ~TransitionTableSet() {
    for (iree_hal_memory_contract_t* contract : contracts_) {
      iree_hal_memory_contract_release(contract);
    }
  }

  iree_hal_memory_contract_t* contract(iree_host_size_t index) const {
    return contracts_[index];
  }

  iree_hal_memory_transition_table_t table(iree_host_size_t index) const {
    return tables_[index];
  }

  iree_hal_memory_transition_pair_t pair() const { return pair_; }

  iree_hal_memory_scope_t scope(iree_hal_memory_scope_id_t id) const {
    return {this, id};
  }

 private:
  static iree_status_t QueryRangedPair(void* user_data,
                                       iree_hal_memory_scope_id_t producer,
                                       iree_hal_memory_scope_id_t consumer,
                                       iree_hal_memory_pair_info_t* out_info) {
    out_info->flags = IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE |
                      IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN;
    out_info->release.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
    out_info->release.executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE;
    out_info->release.operation =
        IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM;
    out_info->release.range_granularity = 64;
    out_info->acquire.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
    out_info->acquire.executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE;
    out_info->acquire.operation =
        IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM;
    out_info->acquire.range_granularity = 64;
    out_info->atomic_reach.scope_32 = IREE_HAL_ATOMIC_REACH_FABRIC;
    out_info->atomic_reach.scope_64 = IREE_HAL_ATOMIC_REACH_FABRIC;
    return iree_ok_status();
  }

  // Independently allocated backing contracts in one group namespace.
  std::array<iree_hal_memory_contract_t*, kTableCount> contracts_ = {};
  // Borrowed hot-path views of the corresponding contracts.
  std::array<iree_hal_memory_transition_table_t, kTableCount> tables_ = {};
  // Prepared producer/consumer cell shared by every table.
  iree_hal_memory_transition_pair_t pair_ = {};
};

// Measures the trusted hot primitive after scopes and roles have been checked.
static void BM_QueryPreparedPair(benchmark::State& state) {
  TransitionTableSet table_set;
  const iree_hal_memory_transition_table_t table = table_set.table(0);
  const iree_hal_memory_transition_pair_t pair = table_set.pair();
  for (auto _ : state) {
    iree_hal_memory_transition_t transition =
        iree_hal_memory_transition_query(table, pair);
    benchmark::DoNotOptimize(transition);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_QueryPreparedPair);

// Includes the support checks a generic handoff recorder performs before
// accepting the direct route.
static void BM_QueryPreparedPairAndCheck(benchmark::State& state) {
  TransitionTableSet table_set;
  const iree_hal_memory_transition_table_t table = table_set.table(0);
  const iree_hal_memory_transition_pair_t pair = table_set.pair();
  for (auto _ : state) {
    iree_hal_memory_transition_t transition =
        iree_hal_memory_transition_query(table, pair);
    bool supported = iree_hal_memory_effects_is_supported(transition.release) &&
                     iree_hal_memory_effects_is_supported(transition.acquire);
    benchmark::DoNotOptimize(transition);
    benchmark::DoNotOptimize(supported);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_QueryPreparedPairAndCheck);

// Models the loop used by a graph lowering several buffers at one handoff. One
// iteration resolves and combines both sides for a complete buffer set.
static void BM_CombinePreparedSet(benchmark::State& state) {
  TransitionTableSet table_set;
  const iree_host_size_t table_count =
      static_cast<iree_host_size_t>(state.range(0));
  const iree_hal_memory_transition_pair_t pair = table_set.pair();
  for (auto _ : state) {
    iree_hal_memory_effects_t release = {};
    iree_hal_memory_effects_t acquire = {};
    bool supported = true;
    for (iree_host_size_t i = 0; i < table_count; ++i) {
      const iree_hal_memory_transition_t transition =
          iree_hal_memory_transition_query(table_set.table(i), pair);
      supported &= iree_hal_memory_effects_is_supported(transition.release) &&
                   iree_hal_memory_effects_is_supported(transition.acquire);
      release = iree_hal_memory_effects_combine(release, transition.release);
      acquire = iree_hal_memory_effects_combine(acquire, transition.acquire);
    }
    benchmark::DoNotOptimize(release);
    benchmark::DoNotOptimize(acquire);
    benchmark::DoNotOptimize(supported);
  }
  state.SetItemsProcessed(state.iterations() * table_count);
}
BENCHMARK(BM_CombinePreparedSet)
    ->ArgName("buffers")
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(6);

// Adds the immutable ranged recipes that each side records beside its logical
// buffer reference. One iteration resolves one complete handoff set.
static void BM_ResolveRangedSet(benchmark::State& state) {
  TransitionTableSet table_set;
  const iree_host_size_t table_count =
      static_cast<iree_host_size_t>(state.range(0));
  const iree_hal_memory_transition_pair_t pair = table_set.pair();
  for (auto _ : state) {
    iree_hal_memory_effects_t release = {};
    iree_hal_memory_effects_t acquire = {};
    uint32_t operation_count = 0;
    bool supported = true;
    for (iree_host_size_t i = 0; i < table_count; ++i) {
      const iree_hal_memory_transition_table_t table = table_set.table(i);
      const iree_hal_memory_transition_t transition =
          iree_hal_memory_transition_query(table, pair);
      supported &= iree_hal_memory_effects_is_supported(transition.release) &&
                   iree_hal_memory_effects_is_supported(transition.acquire);
      release = iree_hal_memory_effects_combine(release, transition.release);
      acquire = iree_hal_memory_effects_combine(acquire, transition.acquire);
      const iree_hal_memory_transition_recipe_t* release_recipe =
          iree_hal_memory_transition_recipe(table, pair,
                                            IREE_HAL_MEMORY_TRANSITION_RELEASE);
      const iree_hal_memory_transition_recipe_t* acquire_recipe =
          iree_hal_memory_transition_recipe(table, pair,
                                            IREE_HAL_MEMORY_TRANSITION_ACQUIRE);
      operation_count += release_recipe ? release_recipe->operation_count : 0;
      operation_count += acquire_recipe ? acquire_recipe->operation_count : 0;
    }
    benchmark::DoNotOptimize(release);
    benchmark::DoNotOptimize(acquire);
    benchmark::DoNotOptimize(operation_count);
    benchmark::DoNotOptimize(supported);
  }
  state.SetItemsProcessed(state.iterations() * table_count);
}
BENCHMARK(BM_ResolveRangedSet)
    ->ArgName("buffers")
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(6);

// Pair preparation is a checked recording boundary and is intentionally kept
// separate from the trusted query measurements above.
static void BM_PreparePairChecked(benchmark::State& state) {
  TransitionTableSet table_set;
  const iree_hal_memory_transition_table_t table = table_set.table(0);
  for (auto _ : state) {
    iree_hal_memory_transition_pair_t pair;
    IREE_CHECK_OK(iree_hal_memory_transition_prepare_pair(
        table, table_set.scope(2), table_set.scope(4),
        IREE_HAL_MEMORY_TRANSITION_RELEASE, &pair));
    benchmark::DoNotOptimize(pair);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_PreparePairChecked);

// Public buffer queries across independent pools, including the precomputed
// wildcard. Completion and allocation are outside the timed recording path.
static void BM_QueryAtomicOperations(benchmark::State& state) {
  TransitionTableSet table_set;
  std::array<iree_hal_buffer_t, TransitionTableSet::kTableCount> buffers = {};
  const iree_host_size_t count = state.range(0);
  const auto scope = table_set.scope(state.range(1));
  for (iree_host_size_t i = 0; i < count; ++i) {
    buffers[i].memory.contract = table_set.contract(i);
  }
  for (auto _ : state) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      const auto operations =
          iree_hal_buffer_query_atomic_operations(&buffers[i], scope);
      benchmark::DoNotOptimize(operations);
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_QueryAtomicOperations)
    ->ArgNames({"buffers", "scope"})
    ->Args({1, 2})
    ->Args({2, 2})
    ->Args({4, 2})
    ->Args({6, 2})
    ->Args({6, IREE_HAL_MEMORY_SCOPE_ANY});

}  // namespace
