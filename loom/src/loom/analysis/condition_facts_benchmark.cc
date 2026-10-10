// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks condition derivation, proof, and structured-edge projection.
// Shared boolean DAGs arise after inlining and canonicalization, where the same
// predicate can feed many composed guards. Duplicate edge payloads arise when
// one source fact is forwarded into several successor arguments. Deeply nested
// structured guards must also compose in linear time. These cases must stay
// compact instead of scaling with path, payload-pair, or lexical-depth
// products.

#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/condition_edge_projection.h"
#include "loom/analysis/condition_facts.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/pass/value_facts.h"
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

    loom_condition_query_initialize(module_, /*value_domain=*/nullptr,
                                    &analysis_arena_, &query_);
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

class ConditionEdgeProjectionBenchmark {
 public:
  explicit ConditionEdgeProjectionBenchmark(int64_t payload_width)
      : payload_width_(payload_width) {
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
    iree_host_size_t scf_vtable_count = 0;
    const loom_op_vtable_t* const* scf_vtables =
        loom_scf_dialect_vtables(&scf_vtable_count);
    IREE_CHECK_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, scf_vtables, (uint16_t)scf_vtable_count));
    IREE_CHECK_OK(loom_context_finalize(&context_));

    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("edge_projection"),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module_));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_value_id_t initial_values[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
    IREE_CHECK_OK(loom_builder_define_value(&builder, i32, &initial_values[0]));
    IREE_CHECK_OK(loom_builder_define_value(&builder, i32, &initial_values[1]));
    const std::vector<loom_type_t> result_types(payload_width_, i32);
    loom_op_t* loop = nullptr;
    IREE_CHECK_OK(loom_scf_while_build(
        &builder, initial_values, IREE_ARRAYSIZE(initial_values),
        /*iter_args_types=*/nullptr, result_types.data(), result_types.size(),
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        LOOM_LOCATION_UNKNOWN, &loop));
    source_region_ = loom_scf_while_before(loop);
    loom_region_t* target_region = loom_scf_while_after(loop);
    target_block_ = loom_region_entry_block(target_region);
    const loom_value_id_t source_left =
        loom_region_entry_arg_id(source_region_, 0);
    const loom_value_id_t source_right =
        loom_region_entry_arg_id(source_region_, 1);

    sources_.reserve(payload_width_);
    for (int64_t i = 0; i < payload_width_; ++i) {
      sources_.push_back((i & 1) == 0 ? source_left : source_right);
    }

    loom_builder_ip_t saved =
        loom_builder_enter_region(&builder, loop, source_region_);
    loom_op_t* compare = nullptr;
    IREE_CHECK_OK(loom_scalar_cmpi_build(
        &builder, LOOM_SCALAR_CMPI_PREDICATE_SLT, source_left, source_right,
        LOOM_LOCATION_UNKNOWN, &compare));
    loom_op_t* condition = nullptr;
    IREE_CHECK_OK(loom_scf_condition_build(
        &builder, loom_scalar_cmpi_result(compare), sources_.data(),
        sources_.size(), LOOM_LOCATION_UNKNOWN, &condition));
    loom_builder_restore(&builder, saved);

    saved = loom_builder_enter_region(&builder, loop, target_region);
    const loom_value_id_t yielded_values[] = {
        loom_block_arg_id(target_block_, 0),
        loom_block_arg_id(target_block_, 1),
    };
    loom_op_t* yield = nullptr;
    IREE_CHECK_OK(loom_scf_yield_build(&builder, yielded_values,
                                       IREE_ARRAYSIZE(yielded_values),
                                       LOOM_LOCATION_UNKNOWN, &yield));
    loom_builder_restore(&builder, saved);

    loom_condition_edge_projection_initialize(&analysis_arena_, &projection_);
    loom_condition_query_t query;
    loom_condition_query_initialize(module_, /*value_domain=*/nullptr,
                                    &analysis_arena_, &query);
    IREE_CHECK_OK(loom_condition_facts_query_complete(
        &query, /*fact_table=*/nullptr, loom_scalar_cmpi_result(compare),
        /*assumed_truth=*/true, &projection_.source_derivation));
    Update();
  }

  ConditionEdgeProjectionBenchmark(const ConditionEdgeProjectionBenchmark&) =
      delete;
  ConditionEdgeProjectionBenchmark& operator=(
      const ConditionEdgeProjectionBenchmark&) = delete;

  ~ConditionEdgeProjectionBenchmark() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void Update() {
    IREE_CHECK_OK(loom_condition_edge_projection_update_mapping(
        &projection_, module_, source_region_, target_block_, sources_.data(),
        sources_.size()));
  }

  bool Prove() const {
    const loom_condition_integer_relation_t query = {
        .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
        .left = ValueOperand(loom_block_arg_id(target_block_, 0)),
        .right = ValueOperand(loom_block_arg_id(target_block_, 1)),
    };
    bool result = false;
    const bool proven = loom_condition_edge_projection_proves_integer_relation(
        &projection_, /*fact_table=*/nullptr, &query, &result);
    IREE_ASSERT(proven && result);
    return result;
  }

  void SetCounters(benchmark::State& state) const {
    state.counters["payload_values"] = (double)payload_width_;
    state.counters["analysis_arena_used_bytes"] =
        (double)analysis_arena_.used_allocation_size;
    state.counters["analysis_arena_owned_bytes"] =
        (double)analysis_arena_.total_allocation_size;
  }

 private:
  static loom_condition_integer_operand_t ValueOperand(
      loom_value_id_t value_id) {
    return loom_condition_integer_operand_t{
        .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
        .value_id = value_id,
    };
  }

  int64_t payload_width_;
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_region_t* source_region_ = nullptr;
  loom_block_t* target_block_ = nullptr;
  std::vector<loom_value_id_t> sources_;
  loom_condition_edge_projection_t projection_;
};

static void ProjectionWidths(::benchmark::Benchmark* benchmark) {
  benchmark->Arg(8)->Arg(64)->Arg(512)->Arg(4096);
}

static void BM_UpdateConditionEdgeProjection(benchmark::State& state) {
  ConditionEdgeProjectionBenchmark fixture(state.range(0));
  for (auto _ : state) {
    fixture.Update();
  }
  fixture.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_UpdateConditionEdgeProjection)->Apply(ProjectionWidths);

static void BM_QueryConditionEdgeProjection(benchmark::State& state) {
  ConditionEdgeProjectionBenchmark fixture(state.range(0));
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Prove());
  }
  fixture.SetCounters(state);
}
BENCHMARK(BM_QueryConditionEdgeProjection)->Apply(ProjectionWidths);

class NestedStructuredBranchFactsBenchmark {
 public:
  explicit NestedStructuredBranchFactsBenchmark(int64_t depth) : depth_(depth) {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    IREE_CHECK_OK(loom_context_finalize(&context_));

    IREE_CHECK_OK(loom_module_allocate(
        &context_, IREE_SV("nested_structured_branch_facts"), &block_pool_,
        nullptr, iree_allocator_system(), &module_));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder, IREE_SV("nested"), &name));
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name, &symbol));
    const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* function_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        /*export_symbol=*/LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
        {/*module_id=*/0, /*symbol_id=*/symbol}, &i32, /*arg_types_count=*/1,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function_op));
    function_ = loom_func_like_cast(module_, function_op);
    uint16_t argument_count = 0;
    const loom_value_id_t* arguments =
        loom_func_like_arg_ids(function_, &argument_count);
    IREE_ASSERT_EQ(argument_count, 1);
    const loom_value_id_t input = arguments[0];

    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder);
    builder.ip.parent_op = function_op;
    std::vector<loom_builder_ip_t> saved_ips;
    saved_ips.reserve(depth_);
    for (int64_t i = 0; i < depth_; ++i) {
      loom_op_t* count = nullptr;
      IREE_CHECK_OK(loom_scalar_cttzi_build(&builder, input, i32,
                                            LOOM_LOCATION_UNKNOWN, &count));
      loom_op_t* bound = nullptr;
      IREE_CHECK_OK(loom_scalar_constant_build(&builder, loom_attr_i64(i), i32,
                                               LOOM_LOCATION_UNKNOWN, &bound));
      loom_op_t* comparison = nullptr;
      IREE_CHECK_OK(
          loom_scalar_cmpi_build(&builder, LOOM_SCALAR_CMPI_PREDICATE_SGT,
                                 input, loom_scalar_constant_result(bound),
                                 LOOM_LOCATION_UNKNOWN, &comparison));
      loom_op_t* branch = nullptr;
      IREE_CHECK_OK(loom_scf_if_build(
          &builder, /*build_flags=*/0, loom_scalar_cmpi_result(comparison),
          /*result_types=*/nullptr, /*result_count=*/0,
          /*tied_results=*/nullptr, /*tied_result_count=*/0,
          LOOM_LOCATION_UNKNOWN, &branch));
      saved_ips.push_back(loom_builder_enter_region(
          &builder, branch, loom_scf_if_then_region(branch)));
    }
    loom_op_t* deepest_count = nullptr;
    IREE_CHECK_OK(loom_scalar_cttzi_build(
        &builder, input, i32, LOOM_LOCATION_UNKNOWN, &deepest_count));
    for (int64_t i = depth_; i > 0; --i) {
      loom_op_t* yield = nullptr;
      IREE_CHECK_OK(loom_scf_yield_build(&builder, /*values=*/nullptr,
                                         /*values_count=*/0,
                                         LOOM_LOCATION_UNKNOWN, &yield));
      loom_builder_restore(&builder, saved_ips[i - 1]);
    }
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&builder, /*values=*/nullptr,
                                         /*values_count=*/0,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    loom_pass_value_fact_owner_initialize(&block_pool_, &owner_);
  }

  NestedStructuredBranchFactsBenchmark(
      const NestedStructuredBranchFactsBenchmark&) = delete;
  NestedStructuredBranchFactsBenchmark& operator=(
      const NestedStructuredBranchFactsBenchmark&) = delete;

  ~NestedStructuredBranchFactsBenchmark() {
    loom_pass_value_fact_owner_deinitialize(&owner_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void Acquire(loom_pass_value_fact_scope_kind_t kind) {
    loom_pass_value_fact_owner_invalidate(&owner_);
    loom_pass_value_fact_scope_t scope =
        loom_pass_value_fact_scope_function(function_);
    scope.kind = kind;
    loom_value_fact_table_t* facts = nullptr;
    IREE_CHECK_OK(
        loom_pass_value_fact_owner_acquire(&owner_, module_, scope, &facts));
    benchmark::DoNotOptimize(facts->touched_count);
  }

  void SetCounters(benchmark::State& state) const {
    iree_arena_block_pool_statistics_t pool_statistics = {};
    iree_arena_block_pool_query_statistics(&block_pool_, &pool_statistics);
    state.counters["branch_depth"] = (double)depth_;
    state.counters["pool_system_allocation_bytes"] =
        (double)(pool_statistics.block_system_allocation_bytes +
                 pool_statistics.oversized_allocation_bytes);
    state.counters["storage_arena_used_bytes"] =
        (double)owner_.storage_arena.used_allocation_size;
    state.counters["transient_arena_used_bytes"] =
        (double)owner_.transient_arena.used_allocation_size;
  }

 private:
  void RegisterDialect(loom_dialect_id_t id, const loom_op_vtable_t* const* (
                                                 *dialect)(iree_host_size_t*)) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect(&count);
    IREE_CHECK_OK(
        loom_context_register_dialect(&context_, id, vtables, (uint16_t)count));
  }

  int64_t depth_;
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_ = {};
  loom_pass_value_fact_owner_t owner_ = {};
};

static void StructuredBranchDepths(::benchmark::Benchmark* benchmark) {
  benchmark->Arg(8)->Arg(32)->Arg(128)->Arg(512);
}

static void BM_AcquireOrdinaryNestedStructuredBranchFacts(
    benchmark::State& state) {
  NestedStructuredBranchFactsBenchmark fixture(state.range(0));
  for (auto _ : state) {
    fixture.Acquire(LOOM_PASS_VALUE_FACT_SCOPE_FUNCTION);
  }
  fixture.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_AcquireOrdinaryNestedStructuredBranchFacts)
    ->Apply(StructuredBranchDepths);

static void BM_AcquireConditionedNestedStructuredBranchFacts(
    benchmark::State& state) {
  NestedStructuredBranchFactsBenchmark fixture(state.range(0));
  for (auto _ : state) {
    fixture.Acquire(LOOM_PASS_VALUE_FACT_SCOPE_CONDITIONED_FUNCTION);
  }
  fixture.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_AcquireConditionedNestedStructuredBranchFacts)
    ->Apply(StructuredBranchDepths);

}  // namespace
