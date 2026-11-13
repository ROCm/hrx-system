// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/util/call_graph.h"

namespace {

enum class GraphShape { kFanout, kChain, kRepeated, kIndependent };

// Construction stays outside the timed loop. The graph uses the same callable
// and call interfaces as compiler-owned modules, without parsing or lowering.
class CallModule {
 public:
  CallModule(GraphShape shape, uint32_t count) {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_CHECK_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_CHECK_OK(loom_context_finalize(&context_));
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("call_graph"), &pool_,
                                       NULL, iree_allocator_system(),
                                       &module_));
    const uint32_t function_count =
        shape == GraphShape::kRepeated ? 2 : count + 1;
    std::vector<loom_op_t*> functions;
    std::vector<loom_symbol_ref_t> symbols;
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    for (uint32_t i = 0; i < function_count; ++i) {
      const std::string name = "function_" + std::to_string(i);
      loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
      IREE_CHECK_OK(loom_builder_intern_string(
          &builder, iree_make_cstring_view(name.c_str()), &name_id));
      loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
      IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
      const loom_symbol_ref_t symbol = {.module_id = 0, .symbol_id = symbol_id};
      loom_op_t* function = NULL;
      IREE_CHECK_OK(loom_test_func_build(&builder, 0, 0, 0, symbol, NULL, 0,
                                         NULL, 0, NULL, 0, NULL, 0,
                                         LOOM_LOCATION_UNKNOWN, &function));
      symbols.push_back(symbol);
      functions.push_back(function);
    }
    if (shape != GraphShape::kIndependent) {
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t caller = shape == GraphShape::kChain ? i : 0;
        const uint32_t callee = shape == GraphShape::kRepeated ? 1 : i + 1;
        const loom_func_like_t function =
            loom_func_like_cast(module_, functions[caller]);
        loom_builder_t body_builder;
        loom_builder_initialize(
            module_, &module_->arena,
            loom_region_entry_block(loom_func_like_body(function)),
            &body_builder);
        body_builder.ip.parent_op = functions[caller];
        loom_op_t* call = NULL;
        IREE_CHECK_OK(loom_test_invoke_build(&body_builder, symbols[callee],
                                             NULL, 0, NULL, 0, NULL, 0,
                                             LOOM_LOCATION_UNKNOWN, &call));
      }
    }
    IREE_CHECK_OK(loom_module_compute_uses(module_));
  }

  ~CallModule() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_module_t* module() { return module_; }

 private:
  // Shared storage for the input module and transient graph builds.
  iree_arena_block_pool_t pool_;
  // Owns the test dialect implementing the production callable interfaces.
  loom_context_t context_;
  // Input graph, retained unchanged across timed iterations.
  loom_module_t* module_ = nullptr;
};

static void BuildGraph(benchmark::State& state, GraphShape shape) {
  const uint32_t count = static_cast<uint32_t>(state.range(0));
  CallModule input(shape, count);
  iree_arena_allocator_t arena;
  iree_arena_initialize(input.module()->arena.block_pool, &arena);
  iree_host_size_t used_bytes = 0;
  iree_host_size_t allocated_bytes = 0;
  for (auto _ : state) {
    loom_call_graph_t graph;
    IREE_CHECK_OK(loom_call_graph_build(input.module(), &arena, &graph));
    benchmark::DoNotOptimize(graph.nodes);
    state.PauseTiming();
    used_bytes = arena.used_allocation_size;
    allocated_bytes = arena.total_allocation_size;
    iree_arena_reset(&arena);
    state.ResumeTiming();
  }
  state.counters["used_bytes"] = static_cast<double>(used_bytes);
  state.counters["arena_bytes"] = static_cast<double>(allocated_bytes);
  state.SetItemsProcessed(state.iterations() * count);
  state.SetComplexityN(count);
  iree_arena_deinitialize(&arena);
}

static void Scales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(8)->Arg(64)->Arg(512)->Arg(4096);
}

static void BM_CallGraphFanout(benchmark::State& state) {
  BuildGraph(state, GraphShape::kFanout);
}
BENCHMARK(BM_CallGraphFanout)->Apply(Scales)->Complexity();

static void BM_CallGraphChain(benchmark::State& state) {
  BuildGraph(state, GraphShape::kChain);
}
BENCHMARK(BM_CallGraphChain)->Apply(Scales)->Complexity();

static void BM_CallGraphRepeated(benchmark::State& state) {
  BuildGraph(state, GraphShape::kRepeated);
}
BENCHMARK(BM_CallGraphRepeated)->Apply(Scales)->Complexity();

static void BM_CallGraphIndependent(benchmark::State& state) {
  BuildGraph(state, GraphShape::kIndependent);
}
BENCHMARK(BM_CallGraphIndependent)->Apply(Scales)->Complexity();

}  // namespace

BENCHMARK_MAIN();
