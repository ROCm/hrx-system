// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks condition derivation and proof over deeply shared boolean DAGs.
// These shapes arise after inlining and canonicalization, where the same
// predicate can feed many composed guards. Traversal must scale with unique SSA
// values rather than the exponential number of producer paths.

#include <cstdint>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/condition_facts.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/util/fact_table.h"

namespace {

class ConditionFactsBenchmark {
 public:
  explicit ConditionFactsBenchmark(int64_t depth) : depth_(depth) {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t scalar_vtable_count = 0;
    const loom_op_vtable_t* const* scalar_vtables =
        loom_scalar_dialect_vtables(&scalar_vtable_count);
    IREE_CHECK_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCALAR,
                                                scalar_vtables,
                                                (uint16_t)scalar_vtable_count));
    IREE_CHECK_OK(loom_context_finalize(&context_));

    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("condition_facts"),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_CHECK_OK(
        loom_value_fact_table_initialize(&fact_table_, &analysis_arena_, 4));

    loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &lane));
    loom_value_id_t outer_bound = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &outer_bound));
    loom_value_id_t inner_bound = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &inner_bound));
    IREE_CHECK_OK(loom_value_fact_table_define(&fact_table_, outer_bound,
                                               loom_value_facts_exact_i64(8)));
    IREE_CHECK_OK(loom_value_fact_table_define(&fact_table_, inner_bound,
                                               loom_value_facts_exact_i64(16)));

    loom_op_t* outer_compare = nullptr;
    IREE_CHECK_OK(loom_scalar_cmpi_build(
        &builder_, LOOM_SCALAR_CMPI_PREDICATE_SLT, lane, outer_bound,
        LOOM_LOCATION_UNKNOWN, &outer_compare));
    derive_condition_ = loom_scalar_cmpi_result(outer_compare);
    loom_op_t* inner_compare = nullptr;
    IREE_CHECK_OK(loom_scalar_cmpi_build(
        &builder_, LOOM_SCALAR_CMPI_PREDICATE_SLT, lane, inner_bound,
        LOOM_LOCATION_UNKNOWN, &inner_compare));
    proof_condition_ = loom_scalar_cmpi_result(inner_compare);
    for (int64_t i = 0; i < depth_; ++i) {
      loom_op_t* derive_and_op = nullptr;
      IREE_CHECK_OK(loom_scalar_andi_build(
          &builder_, derive_condition_, derive_condition_,
          loom_type_scalar(LOOM_SCALAR_TYPE_I1), LOOM_LOCATION_UNKNOWN,
          &derive_and_op));
      derive_condition_ = loom_scalar_andi_result(derive_and_op);
      loom_op_t* proof_and_op = nullptr;
      IREE_CHECK_OK(
          loom_scalar_andi_build(&builder_, proof_condition_, proof_condition_,
                                 loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                 LOOM_LOCATION_UNKNOWN, &proof_and_op));
      proof_condition_ = loom_scalar_andi_result(proof_and_op);
    }

    loom_condition_query_initialize(module_, &analysis_arena_, &query_);
    loom_condition_fact_set_initialize(relation_storage_,
                                       IREE_ARRAYSIZE(relation_storage_),
                                       &condition_facts_);
    bool complete = false;
    IREE_CHECK_OK(loom_condition_facts_query(
        &query_, &fact_table_, derive_condition_, /*assumed_truth=*/true,
        &condition_facts_, &complete));
    IREE_ASSERT(complete);
  }

  ConditionFactsBenchmark(const ConditionFactsBenchmark&) = delete;
  ConditionFactsBenchmark& operator=(const ConditionFactsBenchmark&) = delete;

  ~ConditionFactsBenchmark() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void Derive() {
    bool complete = false;
    IREE_CHECK_OK(loom_condition_facts_query(
        &query_, &fact_table_, derive_condition_, /*assumed_truth=*/true,
        &condition_facts_, &complete));
    IREE_ASSERT(complete);
  }

  bool Prove() {
    bool condition = false;
    bool proven = false;
    IREE_CHECK_OK(loom_condition_fact_set_proves_condition(
        &query_, &fact_table_, &condition_facts_, proof_condition_, &condition,
        &proven));
    IREE_ASSERT(proven);
    return condition;
  }

  void SetCounters(benchmark::State& state) const {
    state.counters["unique_condition_values"] = (double)(depth_ + 1);
    state.counters["analysis_arena_used_bytes"] =
        (double)analysis_arena_.used_allocation_size;
    state.counters["analysis_arena_owned_bytes"] =
        (double)analysis_arena_.total_allocation_size;
  }

 private:
  int64_t depth_;
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t fact_table_;
  loom_value_id_t derive_condition_ = LOOM_VALUE_ID_INVALID;
  loom_value_id_t proof_condition_ = LOOM_VALUE_ID_INVALID;
  loom_condition_query_t query_;
  loom_condition_integer_relation_t relation_storage_[1];
  loom_condition_fact_set_t condition_facts_;
};

static void SharedDagDepths(::benchmark::Benchmark* benchmark) {
  benchmark->Arg(8)->Arg(32)->Arg(256)->Arg(4096);
}

static void BM_DeriveSharedBooleanDag(benchmark::State& state) {
  ConditionFactsBenchmark fixture(state.range(0));
  for (auto _ : state) {
    fixture.Derive();
  }
  fixture.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * (state.range(0) + 1));
}
BENCHMARK(BM_DeriveSharedBooleanDag)->Apply(SharedDagDepths);

static void BM_ProveSharedBooleanDag(benchmark::State& state) {
  ConditionFactsBenchmark fixture(state.range(0));
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Prove());
  }
  fixture.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * (state.range(0) + 1));
}
BENCHMARK(BM_ProveSharedBooleanDag)->Apply(SharedDagDepths);

}  // namespace
