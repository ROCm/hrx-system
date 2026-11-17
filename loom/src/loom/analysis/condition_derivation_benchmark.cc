// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "benchmark/benchmark.h"
#include "loom/analysis/condition_derivation.h"

namespace {

// A conjunction of ordered endpoints has a quadratic number of useful pair
// relations. Construction must scale with those results, without repeatedly
// scanning all prior results or every unrelated source edge.
class ConditionChainBenchmark {
 public:
  explicit ConditionChainBenchmark(uint32_t count) : count_(count) {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_condition_derivation_initialize(&arena_, &derivation_);
    Complete();
  }

  ~ConditionChainBenchmark() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Complete() {
    loom_condition_derivation_reset(&derivation_);
    for (uint32_t i = 0; i < count_; ++i) {
      const loom_condition_integer_relation_t relation = {
          .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE,
          .left = {LOOM_CONDITION_INTEGER_OPERAND_VALUE, i, 0},
          .right = {LOOM_CONDITION_INTEGER_OPERAND_VALUE, i + 1, 0},
      };
      IREE_CHECK_OK(loom_condition_derivation_append_integer_relation(
          &derivation_, relation));
    }
    IREE_CHECK_OK(loom_condition_derivation_complete(nullptr, &arena_,
                                                     &scratch_, &derivation_));
    benchmark::DoNotOptimize(derivation_.integer_facts.integer_relation_count);
  }

  void SetCounters(benchmark::State& state) const {
    state.counters["source_relations"] = count_;
    state.counters["result_relations"] =
        static_cast<double>(derivation_.integer_facts.integer_relation_count);
    state.counters["arena_used_bytes"] =
        static_cast<double>(arena_.used_allocation_size);
    state.counters["arena_allocated_bytes"] =
        static_cast<double>(arena_.total_allocation_size);
    state.SetItemsProcessed(state.iterations() * count_);
  }

 private:
  // Number of adjacent source relations in the ordered chain.
  uint32_t count_;
  // Shared block pool for the reusable query and its result.
  iree_arena_block_pool_t pool_ = {};
  // Arena retaining high-water construction and result capacity.
  iree_arena_allocator_t arena_ = {};
  // Lazily allocated reusable construction state.
  loom_condition_composition_scratch_t* scratch_ = nullptr;
  // Reused complete result.
  loom_condition_derivation_t derivation_ = {};
};

static void BM_CompleteConditionChain(benchmark::State& state) {
  ConditionChainBenchmark fixture(static_cast<uint32_t>(state.range(0)));
  for (auto _ : state) {
    fixture.Complete();
  }
  fixture.SetCounters(state);
}
BENCHMARK(BM_CompleteConditionChain)->Arg(8)->Arg(32)->Arg(128)->Arg(512);

}  // namespace
