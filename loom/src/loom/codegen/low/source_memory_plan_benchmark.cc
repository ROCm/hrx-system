// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks function-owned symbolic memory analysis and ready-only source
// planning over deep, multiply reused index expressions. The matrix separates
// unique producer depth from memory access count so analysis must scale with
// the SSA graph plus its consumers instead of producer depth per consumer.

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

namespace {

class SourceMemoryPlanBenchmark {
 public:
  SourceMemoryPlanBenchmark(int64_t producer_depth, int64_t memory_access_count)
      : producer_depth_(producer_depth) {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &facts_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_CHECK_OK(loom_context_finalize(&context_));

    IREE_CHECK_OK(loom_module_allocate(
        &context_, IREE_SV("source_memory_plan_benchmark"), &block_pool_,
        nullptr, iree_allocator_system(), &module_));
    BuildFunction();
    BuildMemoryAccesses(memory_access_count);
    IREE_CHECK_OK(loom_value_fact_table_initialize(&fact_table_, &facts_arena_,
                                                   module_->values.count));
    IREE_CHECK_OK(
        loom_value_fact_table_compute(&fact_table_, module_, function_));

    AnalyzeAndPlan();
    IREE_ASSERT_EQ(last_plan_.static_byte_offset, producer_depth_ * 4);
    IREE_ASSERT_EQ(last_plan_.dynamic_term_count, 1u);
    IREE_ASSERT_EQ(last_plan_.dynamic_terms[0].index, source_index_);
  }

  SourceMemoryPlanBenchmark(const SourceMemoryPlanBenchmark&) = delete;
  SourceMemoryPlanBenchmark& operator=(const SourceMemoryPlanBenchmark&) =
      delete;

  ~SourceMemoryPlanBenchmark() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&facts_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void AnalyzeAndPlan() {
    iree_arena_allocator_t analysis_arena;
    iree_arena_initialize(&block_pool_, &analysis_arena);
    loom_local_value_domain_t value_domain = {};
    IREE_CHECK_OK(loom_local_value_domain_acquire_for_region(
        module_, loom_func_like_body(function_), &analysis_arena,
        &value_domain));
    loom_symbolic_expr_context_t expression_context = {};
    loom_symbolic_expr_context_initialize(module_, &value_domain, &fact_table_,
                                          &analysis_arena, &expression_context);
    loom_view_region_table_t view_regions = {};
    IREE_CHECK_OK(loom_view_region_table_initialize(
        &value_domain, &expression_context, &view_regions));
    IREE_CHECK_OK(loom_view_region_table_analyze(&view_regions));
    for (const loom_op_t* memory_op : memory_ops_) {
      loom_low_source_memory_access_diagnostic_t diagnostic = {};
      const bool planned = loom_low_source_memory_access_plan_build(
          &view_regions, memory_op, &last_plan_, &diagnostic);
      if (!planned) std::abort();
    }
    benchmark::DoNotOptimize(last_plan_);
    analysis_arena_used_bytes_ = analysis_arena.used_allocation_size;
    analysis_arena_owned_bytes_ = analysis_arena.total_allocation_size;
    loom_local_value_domain_release(&value_domain);
    iree_arena_deinitialize(&analysis_arena);
  }

  void SetCounters(benchmark::State& state) const {
    state.counters["producer_depth"] = (double)producer_depth_;
    state.counters["memory_access_count"] = (double)memory_ops_.size();
    state.counters["analysis_arena_used_bytes"] =
        (double)analysis_arena_used_bytes_;
    state.counters["analysis_arena_owned_bytes"] =
        (double)analysis_arena_owned_bytes_;
  }

 private:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn vtables_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtables_fn(&vtable_count);
    IREE_CHECK_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                (uint16_t)vtable_count));
  }

  void BuildFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&module_builder,
                                             IREE_SV("benchmark"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t callee = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &module_builder, 0, 0, 0, callee, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    function_ = loom_func_like_cast(module_, func_op);
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder_);
  }

  loom_value_id_t DefineBlockArgument(loom_type_t type) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        type, &value_id));
    return value_id;
  }

  loom_value_id_t BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  loom_value_id_t BuildOffsetConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  void BuildMemoryAccesses(int64_t memory_access_count) {
    const loom_value_id_t buffer = DefineBlockArgument(loom_type_buffer());
    source_index_ =
        DefineBlockArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));

    loom_op_t* layout_op = nullptr;
    IREE_CHECK_OK(loom_encoding_layout_dense_build(
        &builder_,
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &layout_op));
    const loom_value_id_t layout = loom_encoding_layout_dense_result(layout_op);
    const loom_value_id_t zero = BuildOffsetConstant(0);
    loom_type_t view_type =
        loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_static(1024 * 1024), 0);
    view_type.encoding_id = (uint16_t)layout;
    view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    loom_op_t* view_op = nullptr;
    IREE_CHECK_OK(loom_buffer_view_build(&builder_, buffer, zero, view_type,
                                         LOOM_LOCATION_UNKNOWN, &view_op));

    const loom_value_id_t one = BuildIndexConstant(1);
    loom_value_id_t dynamic_index = source_index_;
    for (int64_t i = 0; i < producer_depth_; ++i) {
      loom_op_t* add_op = nullptr;
      IREE_CHECK_OK(
          loom_index_add_build(&builder_, dynamic_index, one,
                               loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                               LOOM_LOCATION_UNKNOWN, &add_op));
      dynamic_index = loom_index_add_result(add_op);
    }

    memory_ops_.reserve((size_t)memory_access_count);
    const loom_value_id_t dynamic_indices[] = {dynamic_index};
    const int64_t static_indices[] = {INT64_MIN};
    const loom_type_t vector_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(1), 0);
    for (int64_t i = 0; i < memory_access_count; ++i) {
      loom_op_t* load_op = nullptr;
      IREE_CHECK_OK(loom_vector_load_build(
          &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
          dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
          IREE_ARRAYSIZE(static_indices), 0, 0, vector_type,
          LOOM_LOCATION_UNKNOWN, &load_op));
      memory_ops_.push_back(load_op);
    }
  }

  // Number of unique index producers shared by every memory access.
  int64_t producer_depth_;
  // Reusable storage backing the module, facts, and each analysis run.
  iree_arena_block_pool_t block_pool_;
  // Arena retaining the function fact table across analysis runs.
  iree_arena_allocator_t facts_arena_;
  // Dialect context owning the operation tables used by the fixture module.
  loom_context_t context_;
  // Immutable module analyzed by every benchmark iteration.
  loom_module_t* module_ = nullptr;
  // Function containing the shared producer chain and memory accesses.
  loom_func_like_t function_ = {};
  // Builder used only while constructing the fixture function.
  loom_builder_t builder_ = {};
  // Stable facts shared by every function-analysis lifetime.
  loom_value_fact_table_t fact_table_ = {};
  // Root index expected in every canonical source-memory plan.
  loom_value_id_t source_index_ = LOOM_VALUE_ID_INVALID;
  // Memory operations planned after each function analysis.
  std::vector<const loom_op_t*> memory_ops_;
  // Last plan retained to validate and observe benchmark results.
  loom_low_source_memory_access_plan_t last_plan_ = {};
  // Live analysis storage used by the most recent run.
  iree_host_size_t analysis_arena_used_bytes_ = 0;
  // Block-pool storage owned by the most recent run.
  iree_host_size_t analysis_arena_owned_bytes_ = 0;
};

static void SymbolicMemoryShapes(::benchmark::Benchmark* benchmark) {
  for (int64_t producer_depth : {1, 16, 40, 4096}) {
    for (int64_t memory_access_count : {1, 16, 256, 1024}) {
      benchmark->Args({producer_depth, memory_access_count});
    }
  }
}

static void BM_AnalyzeAndPlanSymbolicMemory(benchmark::State& state) {
  SourceMemoryPlanBenchmark fixture(state.range(0), state.range(1));
  for (auto _ : state) {
    fixture.AnalyzeAndPlan();
  }
  fixture.SetCounters(state);
  state.SetItemsProcessed(state.iterations() *
                          (state.range(0) + state.range(1)));
}
BENCHMARK(BM_AnalyzeAndPlanSymbolicMemory)->Apply(SymbolicMemoryShapes);

}  // namespace
