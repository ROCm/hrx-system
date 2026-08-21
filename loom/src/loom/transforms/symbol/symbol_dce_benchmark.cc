// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks complete symbol DCE over provider import shapes that stress its
// distinct scaling dimensions: anchors in one import and numbers of imports.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/module/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/transforms/symbol/symbol_dce.h"

namespace {

enum class ImportShape {
  kSingleAllLive,
  kSingleAlternatingLive,
  kSingleAllDead,
  kManyAllDead,
};

class SymbolDCEBenchmark {
 public:
  SymbolDCEBenchmark() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_MODULE, loom_module_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_CHECK_OK(loom_context_finalize(&context_));
  }

  SymbolDCEBenchmark(const SymbolDCEBenchmark&) = delete;
  SymbolDCEBenchmark& operator=(const SymbolDCEBenchmark&) = delete;

  ~SymbolDCEBenchmark() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* BuildModule(int64_t anchor_count, ImportShape shape) {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(
        &context_, IREE_SV("symbol_dce_benchmark"), &block_pool_, nullptr,
        iree_allocator_system(), &module));
    loom_builder_t builder = {};
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);

    std::vector<loom_symbol_ref_t> anchors;
    anchors.reserve(anchor_count);
    for (int64_t i = 0; i < anchor_count; ++i) {
      char symbol_name[32];
      std::snprintf(symbol_name, sizeof(symbol_name), "symbol_%04d",
                    static_cast<int>(i));
      loom_symbol_ref_t anchor = AddSymbol(module, symbol_name);
      anchors.push_back(anchor);

      if (shape == ImportShape::kSingleAllLive ||
          (shape == ImportShape::kSingleAlternatingLive && i % 2 == 0)) {
        loom_op_t* definition_op = nullptr;
        IREE_CHECK_OK(loom_test_record_build(
            &builder, 0, 0, anchor, loom_named_attr_slice_empty(),
            LOOM_LOCATION_UNKNOWN, &definition_op));
      }

      if (shape == ImportShape::kManyAllDead) {
        char provider_name[32];
        std::snprintf(provider_name, sizeof(provider_name), "provider_%04d",
                      static_cast<int>(i));
        AddImport(module, &builder, provider_name, &anchors.back(), 1);
      }
    }

    if (shape != ImportShape::kManyAllDead) {
      AddImport(module, &builder, "provider", anchors.data(), anchors.size());
    }
    return module;
  }

  void Run(benchmark::State& state, ImportShape shape) {
    const int64_t anchor_count = state.range(0);
    for (auto _ : state) {
      state.PauseTiming();
      loom_module_t* module = BuildModule(anchor_count, shape);
      iree_arena_allocator_t pass_arena;
      iree_arena_initialize(&block_pool_, &pass_arena);
      const loom_pass_info_t* pass_info = loom_symbol_dce_pass_info();
      std::vector<uint8_t> statistic_storage(
          pass_info->statistic_layout->storage_size, 0);
      loom_pass_t pass = {};
      pass.info = pass_info;
      pass.instance_arena = &pass_arena;
      pass.arena = &pass_arena;
      pass.statistic_storage = statistic_storage.data();
      state.ResumeTiming();

      IREE_CHECK_OK(loom_symbol_dce_run(&pass, module));
      benchmark::DoNotOptimize(module);

      state.PauseTiming();
      loom_module_free(module);
      iree_arena_deinitialize(&pass_arena);
      state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations() * anchor_count);
    state.SetComplexityN(anchor_count);
    state.counters["anchors"] = static_cast<double>(anchor_count);
    state.counters["imports"] = shape == ImportShape::kManyAllDead
                                    ? static_cast<double>(anchor_count)
                                    : 1.0;
  }

 private:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_CHECK_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                static_cast<uint16_t>(count)));
  }

  static loom_symbol_ref_t AddSymbol(loom_module_t* module, const char* name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(
        module, iree_make_cstring_view(name), &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    return {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  static void AddImport(loom_module_t* module, loom_builder_t* builder,
                        const char* provider_name,
                        const loom_symbol_ref_t* anchors,
                        iree_host_size_t anchor_count) {
    loom_string_id_t provider_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(
        module, iree_make_cstring_view(provider_name), &provider_id));
    loom_op_t* import_op = nullptr;
    IREE_CHECK_OK(loom_module_import_build(
        builder, provider_id, loom_make_symbol_ref_array(anchors, anchor_count),
        LOOM_LOCATION_UNKNOWN, &import_op));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

static void ImportScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

static void BM_SingleImportAllLive(benchmark::State& state) {
  SymbolDCEBenchmark benchmark;
  benchmark.Run(state, ImportShape::kSingleAllLive);
}
BENCHMARK(BM_SingleImportAllLive)
    ->Apply(ImportScales)
    ->Complexity(benchmark::oN);

static void BM_SingleImportAlternatingLive(benchmark::State& state) {
  SymbolDCEBenchmark benchmark;
  benchmark.Run(state, ImportShape::kSingleAlternatingLive);
}
BENCHMARK(BM_SingleImportAlternatingLive)
    ->Apply(ImportScales)
    ->Complexity(benchmark::oN);

static void BM_SingleImportAllDead(benchmark::State& state) {
  SymbolDCEBenchmark benchmark;
  benchmark.Run(state, ImportShape::kSingleAllDead);
}
BENCHMARK(BM_SingleImportAllDead)
    ->Apply(ImportScales)
    ->Complexity(benchmark::oN);

static void BM_ManyImportsAllDead(benchmark::State& state) {
  SymbolDCEBenchmark benchmark;
  benchmark.Run(state, ImportShape::kManyAllDead);
}
BENCHMARK(BM_ManyImportsAllDead)
    ->Apply(ImportScales)
    ->Complexity(benchmark::oN);

}  // namespace
