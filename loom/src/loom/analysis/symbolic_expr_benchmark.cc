// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Conditional proofs run during rule selection after address analysis has
// prepared canonical summaries. Query cost must not scale with that inventory.

#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/fact_table.h"

namespace {

class SymbolicProofBenchmark {
 public:
  explicit SymbolicProofBenchmark(int64_t prepared_count) {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    IREE_CHECK_OK(loom_context_finalize(&context_));
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("symbolic_proof"),
                                       &pool_, nullptr, iree_allocator_system(),
                                       &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_CHECK_OK(loom_value_fact_table_initialize(&facts_, &arena_, 16));
    for (int64_t i = 0; i < prepared_count; ++i) {
      prepared_.push_back(Constant(i));
    }

    const auto zero = Constant(0);
    const auto one = Constant(1);
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_I1), &condition));
    loom_op_t* left = nullptr;
    IREE_CHECK_OK(
        loom_scf_select_build(&builder_, condition, one, zero,
                              loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                              LOOM_LOCATION_UNKNOWN, &left));
    ComputeFacts(left);
    loom_op_t* right = nullptr;
    IREE_CHECK_OK(
        loom_scf_select_build(&builder_, condition, zero, one,
                              loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                              LOOM_LOCATION_UNKNOWN, &right));
    ComputeFacts(right);
    loom_op_t* sum = nullptr;
    IREE_CHECK_OK(loom_index_add_build(
        &builder_, loom_scf_select_result(left), loom_scf_select_result(right),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &sum));
    ComputeFacts(sum);
    sum_ = loom_index_add_result(sum);
    loom_symbolic_expr_context_initialize(module_, nullptr, &facts_, &arena_,
                                          &expressions_);
    loom_symbolic_expr_constant(1, &one_);
  }

  ~SymbolicProofBenchmark() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Prepare() {
    loom_symbolic_expr_t expression = {};
    for (const auto value : prepared_) {
      IREE_CHECK_OK(
          loom_symbolic_expr_from_value(&expressions_, value, &expression));
    }
    IREE_CHECK_OK(
        loom_symbolic_expr_from_value(&expressions_, sum_, &sum_expression_));
  }

  void Prove() {
    loom_symbolic_proof_result_t result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_CHECK_OK(loom_symbolic_expr_prove_le(&expressions_, &one_,
                                              &sum_expression_, &result));
    IREE_ASSERT_EQ(result, LOOM_SYMBOLIC_PROOF_TRUE);
    benchmark::DoNotOptimize(result);
  }

  void SetCounters(benchmark::State& state) const {
    iree_arena_block_pool_statistics_t statistics = {};
    iree_arena_block_pool_query_statistics(&pool_, &statistics);
    state.counters["retained_arena_bytes"] =
        (double)arena_.used_allocation_size;
    state.counters["system_allocation_bytes"] =
        (double)(statistics.block_system_allocation_bytes +
                 statistics.oversized_allocation_bytes);
    state.counters["system_allocation_count"] =
        (double)(statistics.block_system_allocation_count +
                 statistics.oversized_allocation_count);
  }

 private:
  void RegisterDialect(loom_dialect_id_t id, const loom_op_vtable_t* const* (
                                                 *dialect)(iree_host_size_t*)) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect(&count);
    IREE_CHECK_OK(
        loom_context_register_dialect(&context_, id, vtables, (uint16_t)count));
  }

  void ComputeFacts(loom_op_t* op) {
    IREE_CHECK_OK(loom_value_fact_table_compute_op(&facts_, module_, op));
  }

  loom_value_id_t Constant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    ComputeFacts(op);
    return loom_index_constant_result(op);
  }

  // Block storage shared by persistent analysis and temporary condition arenas.
  iree_arena_block_pool_t pool_ = {};
  // Lifetime of the canonical facts and symbolic summaries.
  iree_arena_allocator_t arena_ = {};
  // Dialects used by the real source producers.
  loom_context_t context_ = {};
  // Module containing the unrelated inventory and queried select expression.
  loom_module_t* module_ = nullptr;
  // Builder for the source fixture.
  loom_builder_t builder_ = {};
  // Retained producer facts, including selector dependencies.
  loom_value_fact_table_t facts_ = {};
  // Canonical symbolic state shared by preparation and proof.
  loom_symbolic_expr_context_t expressions_ = {};
  // Values prepared before each query, outside its timed interval.
  std::vector<loom_value_id_t> prepared_;
  // Sum of complementary selects, identically one.
  loom_value_id_t sum_ = LOOM_VALUE_ID_INVALID;
  // Lower bound used for the proof.
  loom_symbolic_expr_t one_ = {};
  // Borrowed sum expression used by the proof.
  loom_symbolic_expr_t sum_expression_ = {};
};

void BM_ConditionalProofWithPreparedSummaries(benchmark::State& state) {
  SymbolicProofBenchmark fixture(state.range(0));
  for (auto _ : state) {
    state.PauseTiming();
    fixture.Prepare();
    state.ResumeTiming();
    fixture.Prove();
  }
  fixture.SetCounters(state);
}
BENCHMARK(BM_ConditionalProofWithPreparedSummaries)
    ->Arg(0)
    ->Arg(1024)
    ->Arg(16384)
    ->Iterations(128);

}  // namespace
