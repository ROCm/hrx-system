// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Measures publication after an ordinary feedback equation changes. Independent
// body size, feedback length, and pending loop count are separate dimensions:
// a sparse update should not revisit invariant work, and dense or batched
// updates must not pay for repeated dependency discovery.

#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/fact_propagation.h"
#include "loom/rewrite/rewriter.h"

namespace {

class LoopPublication {
 public:
  LoopPublication(int64_t independent_count, int64_t feedback_count,
                  int64_t loop_count, bool track_rewrites, int64_t depth = 1) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &rewrite_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    Register(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    Register(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    Register(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    IREE_CHECK_OK(loom_context_finalize(&context_));
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("publication"),
                                       &pool_, nullptr, iree_allocator_system(),
                                       &module_));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name;
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("loop_facts"), &name));
    loom_symbol_id_t symbol;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name, &symbol));
    const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* function = nullptr;
    IREE_CHECK_OK(loom_test_func_build(&builder, 0, 0, 0, {0, symbol}, &type, 1,
                                       nullptr, 0, nullptr, 0, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &function));
    const auto function_like = loom_func_like_cast(module_, function);
    auto* body = loom_func_like_body(function_like);
    loom_builder_enter_region(&builder, function, body);
    const auto input = loom_region_entry_arg_id(body, 0);
    zero_ = Constant(&builder, 0);
    one_ = Constant(&builder, 1);
    const auto upper = Constant(&builder, 4);
    for (int64_t i = 0; i < loop_count; ++i) {
      BuildLoop(&builder, one_, input, upper, independent_count, feedback_count,
                depth);
    }
    loom_op_t* end = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(&builder, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &end));
    loom_pass_value_fact_owner_initialize(&pool_, &owner_);
    loom_value_fact_table_t* facts = nullptr;
    IREE_CHECK_OK(loom_pass_value_fact_owner_acquire(
        &owner_, module_, loom_pass_value_fact_scope_function(function_like),
        &facts));
    loom_rewriter_initialize(&rewriter_, module_, &rewrite_arena_);
    if (track_rewrites) {
      IREE_CHECK_OK(loom_rewriter_enable_worklist(&rewriter_));
    }
    loom_rewriter_attach_value_facts(&rewriter_, facts);
    Publish();
    Publish();
    initial_rewrite_bytes_ = rewrite_arena_.used_allocation_size;
    initial_fact_bytes_ = FactBytes();
  }

  ~LoopPublication() {
    loom_rewriter_deinitialize(&rewriter_);
    loom_pass_value_fact_owner_deinitialize(&owner_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&rewrite_arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Publish() {
    increment_ = !increment_;
    for (auto* edit : edits_) {
      IREE_CHECK_OK(loom_index_constant_rewrite_value(
          &rewriter_, edit, loom_attr_i64(increment_ ? 1 : 0)));
    }
    IREE_CHECK_OK(loom_rewriter_propagate_value_facts(&rewriter_));
    while (loom_rewriter_pop(&rewriter_)) {
    }
  }

  void Counters(benchmark::State& state) const {
    state.counters["rewrite_bytes"] = rewrite_arena_.used_allocation_size;
    state.counters["rewrite_growth_bytes"] =
        rewrite_arena_.used_allocation_size - initial_rewrite_bytes_;
    state.counters["fact_bytes"] = FactBytes();
    state.counters["fact_growth_bytes"] = FactBytes() - initial_fact_bytes_;
    state.SetItemsProcessed(state.iterations() * edits_.size());
  }

 private:
  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count = 0;
    const auto* vtables = get(&count);
    IREE_CHECK_OK(loom_context_register_dialect(&context_, dialect, vtables,
                                                static_cast<uint16_t>(count)));
  }

  loom_value_id_t Constant(loom_builder_t* builder, int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  iree_host_size_t FactBytes() const {
    return owner_.storage_arena.used_allocation_size +
           owner_.transient_arena.used_allocation_size;
  }

  loom_value_id_t BuildLoop(loom_builder_t* builder, loom_value_id_t initial,
                            loom_value_id_t input, loom_value_id_t upper,
                            int64_t independent_count, int64_t feedback_count,
                            int64_t depth) {
    const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* loop = nullptr;
    IREE_CHECK_OK(loom_scf_for_build(
        builder, 0, zero_, upper, one_, &initial, 1, nullptr, nullptr, 0,
        LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
        LOOM_LOCATION_UNKNOWN, &loop));
    auto nested = *builder;
    auto* loop_body = loom_loop_like_body(loom_loop_like_cast(module_, loop));
    loom_builder_enter_region(&nested, loop, loop_body);
    loom_value_id_t invariant = input;
    for (int64_t j = 0; j < independent_count; ++j) {
      loom_op_t* add = nullptr;
      IREE_CHECK_OK(loom_index_add_build(&nested, invariant, one_, type,
                                         LOOM_LOCATION_UNKNOWN, &add));
      invariant = loom_index_add_result(add);
    }
    loom_op_t* use = nullptr;
    IREE_CHECK_OK(loom_test_use_build(&nested, &invariant, 1,
                                      LOOM_LOCATION_UNKNOWN, &use));
    loom_op_t* increment = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(&nested, loom_attr_i64(0), type,
                                            LOOM_LOCATION_UNKNOWN, &increment));
    edits_.push_back(increment);
    loom_value_id_t feedback = loom_region_entry_arg_id(loop_body, 1);
    if (depth > 1) {
      feedback = BuildLoop(&nested, feedback, input, upper, independent_count,
                           feedback_count, depth - 1);
    }
    for (int64_t j = 0; j < feedback_count; ++j) {
      loom_op_t* add = nullptr;
      IREE_CHECK_OK(loom_index_add_build(
          &nested, feedback,
          j == 0 ? loom_index_constant_result(increment) : zero_, type,
          LOOM_LOCATION_UNKNOWN, &add));
      feedback = loom_index_add_result(add);
    }
    loom_op_t* yield = nullptr;
    IREE_CHECK_OK(loom_scf_yield_build(&nested, &feedback, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield));
    return loom_op_results(loop)[0];
  }

  // Shared backing for independent module, fact, and rewrite lifetimes.
  iree_arena_block_pool_t pool_;
  // Retained queue storage reused by each mutation batch.
  iree_arena_allocator_t rewrite_arena_;
  // Dialects used by the fixture's normal inference callbacks.
  loom_context_t context_;
  // Valid source IR retained across all measured edits.
  loom_module_t* module_ = nullptr;
  // Function-scoped production facts maintained by the rewriter.
  loom_pass_value_fact_owner_t owner_;
  // Production mutation and publication boundary under measurement.
  loom_rewriter_t rewriter_;
  // Mutable feedback increments for the publication batch.
  std::vector<loom_op_t*> edits_;
  // Zero increment used on alternating publications.
  loom_value_id_t zero_;
  // Unit increment used on alternating publications.
  loom_value_id_t one_;
  // Which increment is installed in the source equations.
  bool increment_ = false;
  // Retained rewrite bytes after both semantic states have warmed.
  iree_host_size_t initial_rewrite_bytes_;
  // Retained fact bytes after both semantic states have warmed.
  iree_host_size_t initial_fact_bytes_;
};

static void BM_LoopPublication(benchmark::State& state) {
  LoopPublication fixture(state.range(0), state.range(1), state.range(2),
                          state.range(3));
  for (auto _ : state) {
    fixture.Publish();
  }
  fixture.Counters(state);
}
BENCHMARK(BM_LoopPublication)
    ->ArgNames({"independent", "feedback", "loops", "rewrites"})
    ->Apply([](benchmark::internal::Benchmark* benchmark) {
      for (int64_t rewrites : {0, 1}) {
        for (int64_t independent : {0, 32, 256, 2048}) {
          benchmark->Args({independent, 1, 1, rewrites});
        }
        for (int64_t feedback : {32, 256}) {
          for (int64_t independent : {0, 256}) {
            benchmark->Args({independent, feedback, 1, rewrites});
          }
        }
        for (int64_t loops : {16, 256}) {
          benchmark->Args({32, 1, loops, rewrites});
        }
      }
    });

static void BM_NestedLoopPublication(benchmark::State& state) {
  LoopPublication fixture(state.range(0), state.range(1), 1, state.range(2), 2);
  for (auto _ : state) {
    fixture.Publish();
  }
  fixture.Counters(state);
}
BENCHMARK(BM_NestedLoopPublication)
    ->ArgNames({"independent", "feedback", "rewrites"})
    ->Args({0, 32, 0})
    ->Args({256, 1, 0})
    ->Args({0, 32, 1})
    ->Args({256, 1, 1});

}  // namespace
