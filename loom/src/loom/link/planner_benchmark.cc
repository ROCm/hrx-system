// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks cold module indexing and warm metadata-only link planning.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/selected_reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/link/module_index.h"
#include "loom/link/plan_projection.h"
#include "loom/link/planner.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"

namespace {

static void CheckStatus(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

class PlannerCatalogFixture {
 public:
  explicit PlannerCatalogFixture(uint32_t symbol_count)
      : symbol_count_(symbol_count) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    CheckStatus(loom_test_dialect_register(&context_));
    CheckStatus(loom_context_finalize(&context_));

    loom_module_t* module = nullptr;
    CheckStatus(loom_module_allocate(&context_, IREE_SV("planner_catalog"),
                                     &block_pool_, nullptr,
                                     iree_allocator_system(), &module));
    BuildModule(module);
    SerializeModule(module, &bytes_);
    module_ = module;

    CheckStatus(loom_module_allocate(&context_, IREE_SV("planner_requester"),
                                     &block_pool_, nullptr,
                                     iree_allocator_system(), &input_module_));
    BuildIdentityModule(input_module_, IREE_SV("requester_entry"));
    SerializeModule(input_module_, &input_bytes_);
  }

  PlannerCatalogFixture(const PlannerCatalogFixture&) = delete;
  PlannerCatalogFixture& operator=(const PlannerCatalogFixture&) = delete;

  ~PlannerCatalogFixture() {
    for (loom_module_t* module : benchmark_modules_) {
      loom_module_free(module);
    }
    loom_module_free(input_module_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_link_module_index_t* BuildIndex(loom_link_provider_role_t role) {
    loom_link_module_index_t* index = nullptr;
    CheckStatus(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index));
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/IREE_SV("catalog"),
        /*.role=*/role,
    };
    CheckStatus(loom_link_module_index_add_bytecode(
        index, iree_make_const_byte_span(bytes_.data(), bytes_.size()),
        IREE_SV("planner_catalog.loombc"), /*index_options=*/nullptr, &options,
        /*out_provider_ordinal=*/nullptr));
    return index;
  }

  loom_link_module_index_t* BuildOneInputOverlay(
      const loom_link_module_index_t* base_index) {
    loom_link_module_index_t* index = nullptr;
    CheckStatus(loom_link_module_index_allocate_overlay(
        base_index, &block_pool_, iree_allocator_system(), &index));
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/IREE_SV("requester"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    CheckStatus(loom_link_module_index_add_bytecode(
        index,
        iree_make_const_byte_span(input_bytes_.data(), input_bytes_.size()),
        IREE_SV("planner_requester.loombc"), /*index_options=*/nullptr,
        &options,
        /*out_provider_ordinal=*/nullptr));
    return index;
  }

  loom_link_module_index_t* BuildDuplicateIndex(uint32_t provider_count) {
    loom_link_module_index_t* index = nullptr;
    CheckStatus(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index));
    for (uint32_t i = 0; i < provider_count; ++i) {
      const loom_link_module_index_add_options_t options = {
          /*.provider_name=*/IREE_SV("duplicate-provider"),
          /*.role=*/i == provider_count / 2 ? LOOM_LINK_PROVIDER_ROLE_INPUT
                                            : LOOM_LINK_PROVIDER_ROLE_LIBRARY,
      };
      CheckStatus(loom_link_module_index_add_bytecode(
          index, iree_make_const_byte_span(bytes_.data(), bytes_.size()),
          IREE_SV("planner_duplicate.loombc"), /*index_options=*/nullptr,
          &options, /*out_provider_ordinal=*/nullptr));
    }
    return index;
  }

  loom_link_module_index_t* BuildUnrelatedProviderIndex(
      uint32_t library_provider_count) {
    loom_link_module_index_t* index = nullptr;
    CheckStatus(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index));
    benchmark_modules_.reserve(benchmark_modules_.size() +
                               library_provider_count);
    for (uint32_t i = 0; i < library_provider_count; ++i) {
      char name[32];
      const int name_length =
          std::snprintf(name, sizeof(name), "library_%08u", i);
      if (name_length <= 0 || name_length >= (int)sizeof(name)) {
        std::abort();
      }
      const iree_string_view_t name_view =
          iree_make_string_view(name, (iree_host_size_t)name_length);
      loom_module_t* module = nullptr;
      CheckStatus(loom_module_allocate(&context_, name_view, &block_pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
      benchmark_modules_.push_back(module);
      BuildIdentityModule(module, name_view);
      const loom_link_module_index_add_options_t library_options = {
          /*.provider_name=*/name_view,
          /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
      };
      CheckStatus(loom_link_module_index_add_materialized(
          index, module, &library_options,
          /*out_provider_ordinal=*/nullptr));
    }
    const loom_link_module_index_add_options_t input_options = {
        /*.provider_name=*/IREE_SV("requester"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    CheckStatus(loom_link_module_index_add_materialized(
        index, input_module_, &input_options,
        /*out_provider_ordinal=*/nullptr));
    return index;
  }

  std::string SymbolName(uint32_t ordinal) const {
    char name[32];
    std::snprintf(name, sizeof(name), "function_%08u", ordinal);
    return name;
  }

  uint32_t symbol_count() const { return symbol_count_; }
  iree_host_size_t byte_count() const { return bytes_.size(); }
  iree_arena_block_pool_t* block_pool() { return &block_pool_; }
  const loom_module_t* module() const { return module_; }

 private:
  void BuildIdentityModule(loom_module_t* module,
                           iree_string_view_t function_name) {
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    CheckStatus(loom_module_intern_type(module, i32_type, &i32_type));
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    CheckStatus(loom_module_intern_string(module, function_name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    CheckStatus(loom_module_add_symbol(module, name_id, &symbol_id));

    loom_builder_t module_builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &module_builder);
    const loom_symbol_ref_t symbol_ref = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    loom_op_t* function_op = nullptr;
    CheckStatus(loom_test_func_build(
        &module_builder, LOOM_TEST_FUNC_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_TEST_VISIBILITY_PUBLIC, /*cc=*/0, symbol_ref, &i32_type,
        /*arg_types_count=*/1, &i32_type, /*result_count=*/1,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_NONE,
        &function_op));
    const loom_func_like_t function = loom_func_like_cast(module, function_op);
    uint16_t argument_count = 0;
    const loom_value_id_t* arguments =
        loom_func_like_arg_ids(function, &argument_count);
    if (argument_count != 1) std::abort();
    loom_builder_t body_builder;
    loom_builder_initialize(
        module, &module->arena,
        loom_region_entry_block(loom_func_like_body(function)), &body_builder);
    loom_op_t* yield_op = nullptr;
    CheckStatus(loom_test_yield_build(&body_builder, arguments,
                                      /*values_count=*/1, LOOM_LOCATION_NONE,
                                      &yield_op));
  }

  void BuildModule(loom_module_t* module) {
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    CheckStatus(loom_module_intern_type(module, i32_type, &i32_type));

    std::vector<loom_symbol_id_t> symbol_ids(symbol_count_);
    for (uint32_t i = 0; i < symbol_count_; ++i) {
      const std::string name = SymbolName(i);
      loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
      CheckStatus(loom_module_intern_string(
          module, iree_make_string_view(name.data(), name.size()), &name_id));
      CheckStatus(loom_module_add_symbol(module, name_id, &symbol_ids[i]));
    }

    loom_builder_t module_builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &module_builder);
    for (uint32_t i = 0; i < symbol_count_; ++i) {
      const loom_symbol_ref_t symbol_ref = {
          /*.module_id=*/0,
          /*.symbol_id=*/symbol_ids[i],
      };
      loom_op_t* function_op = nullptr;
      CheckStatus(loom_test_func_build(
          &module_builder, LOOM_TEST_FUNC_BUILD_FLAG_HAS_VISIBILITY,
          LOOM_TEST_VISIBILITY_PUBLIC, /*cc=*/0, symbol_ref, &i32_type,
          /*arg_types_count=*/1, &i32_type, /*result_count=*/1,
          /*tied_results=*/nullptr, /*tied_result_count=*/0,
          /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_NONE,
          &function_op));

      const loom_func_like_t function =
          loom_func_like_cast(module, function_op);
      uint16_t argument_count = 0;
      const loom_value_id_t* arguments =
          loom_func_like_arg_ids(function, &argument_count);
      if (argument_count != 1) std::abort();

      loom_builder_t body_builder;
      loom_builder_initialize(
          module, &module->arena,
          loom_region_entry_block(loom_func_like_body(function)),
          &body_builder);
      loom_value_id_t result = arguments[0];
      if (i + 1 < symbol_count_) {
        const loom_symbol_ref_t callee_ref = {
            /*.module_id=*/0,
            /*.symbol_id=*/symbol_ids[i + 1],
        };
        loom_op_t* invoke_op = nullptr;
        CheckStatus(loom_test_invoke_build(
            &body_builder, callee_ref, arguments, /*operands_count=*/1,
            &i32_type, /*result_count=*/1, /*tied_results=*/nullptr,
            /*tied_result_count=*/0, LOOM_LOCATION_NONE, &invoke_op));
        result = loom_test_invoke_results(invoke_op).values[0];
      }
      loom_op_t* yield_op = nullptr;
      CheckStatus(loom_test_yield_build(&body_builder, &result,
                                        /*values_count=*/1, LOOM_LOCATION_NONE,
                                        &yield_op));
    }
  }

  void SerializeModule(const loom_module_t* module,
                       std::vector<uint8_t>* out_bytes) {
    iree_io_stream_t* stream = nullptr;
    CheckStatus(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    CheckStatus(loom_bytecode_write_module(module, stream, /*options=*/nullptr,
                                           &block_pool_));
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    out_bytes->resize((size_t)length);
    CheckStatus(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    CheckStatus(iree_io_stream_read(stream, out_bytes->size(),
                                    out_bytes->data(), nullptr));
    iree_io_stream_release(stream);
  }

  uint32_t symbol_count_;
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_module_t* input_module_ = nullptr;
  std::vector<loom_module_t*> benchmark_modules_;
  std::vector<uint8_t> bytes_;
  // Serialized one-symbol requester indexed by each timed overlay.
  std::vector<uint8_t> input_bytes_;
};

static void SetCounters(benchmark::State& state,
                        const PlannerCatalogFixture& fixture,
                        iree_host_size_t selected_symbol_count) {
  state.counters["bytecode_bytes"] = static_cast<double>(fixture.byte_count());
  state.counters["selected_symbols"] =
      static_cast<double>(selected_symbol_count);
  state.counters["symbols"] = static_cast<double>(fixture.symbol_count());
  state.SetComplexityN(fixture.symbol_count());
}

static void BM_ModuleIndex_Catalog(benchmark::State& state) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  for (auto _ : state) {
    loom_link_module_index_t* index =
        fixture.BuildIndex(LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    benchmark::DoNotOptimize(index);
    state.PauseTiming();
    loom_link_module_index_free(index);
    state.ResumeTiming();
  }
  SetCounters(state, fixture, /*selected_symbol_count=*/0);
}

static void BM_ModuleIndex_OverlayOneInput_Catalog(benchmark::State& state) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  loom_link_module_index_t* base_index =
      fixture.BuildIndex(LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  for (auto _ : state) {
    loom_link_module_index_t* index = fixture.BuildOneInputOverlay(base_index);
    benchmark::DoNotOptimize(index);
    state.PauseTiming();
    loom_link_module_index_free(index);
    state.ResumeTiming();
  }
  state.counters["base_providers"] = 1.0;
  SetCounters(state, fixture, /*selected_symbol_count=*/1);
  loom_link_module_index_free(base_index);
}

static void BenchmarkPlan(benchmark::State& state, loom_link_plan_mode_t mode,
                          uint32_t root_ordinal,
                          iree_host_size_t expected_symbol_count) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  loom_link_module_index_t* index = fixture.BuildIndex(
      mode == LOOM_LINK_PLAN_MERGE ? LOOM_LINK_PROVIDER_ROLE_INPUT
                                   : LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  const std::string root_name = fixture.SymbolName(root_ordinal);
  const iree_string_view_t root =
      iree_make_string_view(root_name.data(), root_name.size());
  iree_string_view_list_t root_symbols = iree_string_view_list_empty();
  if (mode == LOOM_LINK_PLAN_LINK) {
    root_symbols = (iree_string_view_list_t){
        /*.count=*/1,
        /*.values=*/&root,
    };
  }
  const loom_link_plan_options_t options = {
      /*.mode=*/mode,
      /*.root_symbols=*/root_symbols,
  };

  iree_host_size_t selected_facet_count = 0;
  for (auto _ : state) {
    loom_link_plan_t* plan = nullptr;
    CheckStatus(
        loom_link_plan_build(index, &options, iree_allocator_system(), &plan));
    if (loom_link_plan_symbol_count(plan) != expected_symbol_count) {
      std::abort();
    }
    selected_facet_count = loom_link_plan_facet_count(plan);
    benchmark::DoNotOptimize(plan);
    state.PauseTiming();
    loom_link_plan_free(plan);
    state.ResumeTiming();
  }
  state.counters["selected_facets"] = static_cast<double>(selected_facet_count);
  SetCounters(state, fixture, expected_symbol_count);
  loom_link_module_index_free(index);
}

static void BM_Plan_Merge_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkPlan(state, LOOM_LINK_PLAN_MERGE, /*root_ordinal=*/0, symbol_count);
}

static void BM_Plan_LinkLeaf_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkPlan(state, LOOM_LINK_PLAN_LINK,
                /*root_ordinal=*/symbol_count - 1,
                /*expected_symbol_count=*/1);
}

static void BM_Plan_LinkChain_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkPlan(state, LOOM_LINK_PLAN_LINK, /*root_ordinal=*/0, symbol_count);
}

static void BM_Plan_InputExport_UnrelatedLibraryProviders(
    benchmark::State& state) {
  const uint32_t library_provider_count = (uint32_t)state.range(0);
  PlannerCatalogFixture fixture(/*symbol_count=*/1);
  loom_link_module_index_t* index =
      fixture.BuildUnrelatedProviderIndex(library_provider_count);
  const loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/iree_string_view_list_empty(),
      /*.include_input_exports=*/true,
  };

  for (auto _ : state) {
    loom_link_plan_t* plan = nullptr;
    CheckStatus(
        loom_link_plan_build(index, &options, iree_allocator_system(), &plan));
    if (loom_link_plan_symbol_count(plan) != 1) {
      std::abort();
    }
    const loom_link_plan_symbol_t* root = loom_link_plan_symbol_at(plan, 0);
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, root->symbol_ordinal);
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_symbol_provider(index, symbol);
    if (!provider || provider->role != LOOM_LINK_PROVIDER_ROLE_INPUT) {
      std::abort();
    }
    benchmark::DoNotOptimize(plan);
    state.PauseTiming();
    loom_link_plan_free(plan);
    state.ResumeTiming();
  }
  state.counters["library_exports"] =
      static_cast<double>(library_provider_count);
  state.counters["providers"] = static_cast<double>(library_provider_count + 1);
  state.counters["requester_exports"] = 1.0;
  state.counters["selected_symbols"] = 1.0;
  state.SetComplexityN(library_provider_count);
  loom_link_module_index_free(index);
}

static loom_module_t* MaterializeIndexedBytecodeModule(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* indexed_module,
    loom_context_t* context, iree_arena_block_pool_t* block_pool) {
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index,
                                         indexed_module->provider_ordinal);
  if (!provider || provider->kind != LOOM_LINK_PROVIDER_BYTECODE) {
    std::abort();
  }
  loom_bytecode_read_result_t read_result = {};
  loom_module_t* module = nullptr;
  CheckStatus(loom_bytecode_read_module_ordinal(
      provider->bytecode.contents, provider->bytecode.filename, context,
      block_pool, (uint16_t)indexed_module->provider_module_ordinal,
      /*options=*/nullptr, &read_result, &module, iree_allocator_system()));
  if (read_result.error_count != 0 || module == nullptr) {
    std::abort();
  }
  return module;
}

static void BenchmarkProjection(benchmark::State& state, uint32_t root_ordinal,
                                iree_host_size_t expected_symbol_count) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  loom_link_module_index_t* index =
      fixture.BuildIndex(LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  const std::string root_name = fixture.SymbolName(root_ordinal);
  const iree_string_view_t root =
      iree_make_string_view(root_name.data(), root_name.size());
  const loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/1, /*.values=*/&root},
  };
  loom_link_plan_t* plan = nullptr;
  CheckStatus(
      loom_link_plan_build(index, &options, iree_allocator_system(), &plan));
  if (loom_link_plan_symbol_count(plan) != expected_symbol_count) {
    std::abort();
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(fixture.block_pool(), &arena);
  for (auto _ : state) {
    loom_link_plan_module_projection_t projection;
    CheckStatus(loom_link_plan_project_modules(plan, &arena, &projection));
    if (projection.modules.count != 1 ||
        projection.symbols.count != expected_symbol_count) {
      std::abort();
    }
    benchmark::DoNotOptimize(projection.symbols.values);
    state.PauseTiming();
    iree_arena_reset(&arena);
    state.ResumeTiming();
  }
  state.counters["projection_bytes"] = static_cast<double>(
      sizeof(loom_link_plan_module_selection_t) +
      expected_symbol_count * sizeof(loom_link_plan_module_symbol_t));
  SetCounters(state, fixture, expected_symbol_count);
  iree_arena_deinitialize(&arena);
  loom_link_plan_free(plan);
  loom_link_module_index_free(index);
}

static void BM_Project_LinkLeaf_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkProjection(state, /*root_ordinal=*/symbol_count - 1,
                      /*expected_symbol_count=*/1);
}

static void BM_Project_LinkChain_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkProjection(state, /*root_ordinal=*/0, symbol_count);
}

static void BenchmarkExactLink(benchmark::State& state,
                               iree_host_size_t selected_symbol_count) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  std::vector<iree_host_size_t> source_symbol_ordinals(selected_symbol_count);
  if (selected_symbol_count == 1) {
    source_symbol_ordinals[0] = fixture.symbol_count() - 1;
  } else {
    for (iree_host_size_t i = 0; i < selected_symbol_count; ++i) {
      source_symbol_ordinals[i] = i;
    }
  }
  const loom_linker_source_symbol_list_t source_symbols = {
      /*.count=*/source_symbol_ordinals.size(),
      /*.ordinals=*/source_symbol_ordinals.data(),
  };
  const loom_linker_options_t linker_options = {
      /*.module_name=*/IREE_SV("linked"),
  };

  for (auto _ : state) {
    state.PauseTiming();
    loom_linker_t* linker = nullptr;
    CheckStatus(loom_linker_allocate(fixture.module()->context, &linker_options,
                                     fixture.block_pool(),
                                     iree_allocator_system(), &linker));
    state.ResumeTiming();
    CheckStatus(loom_linker_add_module_symbols(
        linker, fixture.module(), source_symbols,
        loom_linker_source_symbol_binding_list_empty(),
        loom_linker_source_symbol_output_list_empty(),
        loom_linker_target_symbol_list_empty()));
    benchmark::DoNotOptimize(linker);
    state.PauseTiming();
    loom_linker_free(linker);
    state.ResumeTiming();
  }
  SetCounters(state, fixture, selected_symbol_count);
}

static void BM_LinkExact_LinkLeaf_Catalog(benchmark::State& state) {
  BenchmarkExactLink(state, /*selected_symbol_count=*/1);
}

static void BM_LinkExact_LinkChain_Catalog(benchmark::State& state) {
  BenchmarkExactLink(state, /*selected_symbol_count=*/state.range(0));
}

static void BM_LinkExactDense_Catalog(benchmark::State& state) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  const loom_linker_options_t linker_options = {
      /*.module_name=*/IREE_SV("linked"),
  };

  for (auto _ : state) {
    state.PauseTiming();
    loom_linker_t* linker = nullptr;
    CheckStatus(loom_linker_allocate(fixture.module()->context, &linker_options,
                                     fixture.block_pool(),
                                     iree_allocator_system(), &linker));
    state.ResumeTiming();
    CheckStatus(loom_linker_add_exact_module(
        linker, fixture.module(), loom_linker_source_symbol_output_list_empty(),
        loom_linker_target_symbol_list_empty()));
    benchmark::DoNotOptimize(linker);
    state.PauseTiming();
    loom_linker_free(linker);
    state.ResumeTiming();
  }
  SetCounters(state, fixture, fixture.symbol_count());
}

static void BenchmarkLinkMaterializeAndLink(
    benchmark::State& state, uint32_t root_ordinal,
    iree_host_size_t expected_symbol_count) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  loom_link_module_index_t* index =
      fixture.BuildIndex(LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  const std::string root_name = fixture.SymbolName(root_ordinal);
  const iree_string_view_t root =
      iree_make_string_view(root_name.data(), root_name.size());
  const loom_link_plan_options_t plan_options = {
      /*.mode=*/LOOM_LINK_PLAN_LINK,
      /*.root_symbols=*/{/*.count=*/1, /*.values=*/&root},
  };
  loom_link_plan_t* plan = nullptr;
  CheckStatus(loom_link_plan_build(index, &plan_options,
                                   iree_allocator_system(), &plan));
  iree_arena_allocator_t projection_arena;
  iree_arena_initialize(fixture.block_pool(), &projection_arena);
  loom_link_plan_module_projection_t projection = {};
  CheckStatus(
      loom_link_plan_project_modules(plan, &projection_arena, &projection));
  if (projection.modules.count != 1 ||
      projection.symbols.count != expected_symbol_count) {
    std::abort();
  }

  const loom_link_plan_module_selection_t& selection =
      projection.modules.values[0];
  std::vector<iree_host_size_t> source_symbol_ordinals(selection.symbols.count);
  for (iree_host_size_t i = 0; i < selection.symbols.count; ++i) {
    source_symbol_ordinals[i] =
        selection.symbols.values[i].source_symbol->module_symbol_ordinal;
  }
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(
          index, selection.source_module->provider_ordinal);
  if (!provider || provider->kind != LOOM_LINK_PROVIDER_BYTECODE) {
    std::abort();
  }
  const loom_bytecode_read_options_t read_options = {};
  const loom_linker_options_t linker_options = {
      /*.module_name=*/IREE_SV("linked"),
  };

  for (auto _ : state) {
    state.PauseTiming();
    loom_linker_t* linker = nullptr;
    CheckStatus(loom_linker_allocate(fixture.module()->context, &linker_options,
                                     fixture.block_pool(),
                                     iree_allocator_system(), &linker));
    state.ResumeTiming();
    loom_bytecode_read_result_t read_result = {};
    loom_module_t* selected_module = nullptr;
    CheckStatus(loom_bytecode_materialize_module_symbols(
        provider->bytecode.contents, provider->bytecode.filename,
        fixture.module()->context, fixture.block_pool(),
        &provider->bytecode.metadata,
        (uint16_t)selection.source_module->provider_module_ordinal,
        (loom_bytecode_symbol_ordinal_list_t){
            /*.count=*/source_symbol_ordinals.size(),
            /*.ordinals=*/source_symbol_ordinals.data(),
        },
        &read_options, &read_result, &selected_module,
        iree_allocator_system()));
    if (read_result.error_count != 0 || selected_module == nullptr ||
        selected_module->symbols.count != expected_symbol_count) {
      std::abort();
    }
    CheckStatus(loom_linker_add_exact_module(
        linker, selected_module, loom_linker_source_symbol_output_list_empty(),
        loom_linker_target_symbol_list_empty()));
    loom_module_free(selected_module);
    benchmark::DoNotOptimize(linker);
    state.PauseTiming();
    loom_linker_free(linker);
    state.ResumeTiming();
  }

  SetCounters(state, fixture, expected_symbol_count);
  iree_arena_deinitialize(&projection_arena);
  loom_link_plan_free(plan);
  loom_link_module_index_free(index);
}

static void BM_MaterializeAndLink_LinkLeaf_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkLinkMaterializeAndLink(state, /*root_ordinal=*/symbol_count - 1,
                                  /*expected_symbol_count=*/1);
}

static void BM_MaterializeAndLink_LinkChain_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkLinkMaterializeAndLink(state, /*root_ordinal=*/0,
                                  /*expected_symbol_count=*/symbol_count);
}

static void BM_GlobalDuplicateEnumeration(benchmark::State& state) {
  const uint32_t provider_count = (uint32_t)state.range(0);
  PlannerCatalogFixture fixture(/*symbol_count=*/1);
  loom_link_module_index_t* index = fixture.BuildDuplicateIndex(provider_count);
  const std::string symbol_name = fixture.SymbolName(/*ordinal=*/0);
  const loom_link_module_index_symbol_t* selected =
      loom_link_module_index_lookup_global(
          index, iree_make_string_view(symbol_name.data(), symbol_name.size()));
  if (!selected) std::abort();

  for (auto _ : state) {
    iree_host_size_t duplicate_count = 0;
    const loom_link_module_index_symbol_t* duplicate = selected;
    while ((duplicate = loom_link_module_index_next_global_duplicate(
                index, duplicate))) {
      benchmark::DoNotOptimize(duplicate);
      ++duplicate_count;
    }
    if (duplicate_count != provider_count - 1) std::abort();
    benchmark::DoNotOptimize(duplicate_count);
  }
  state.counters["duplicates"] = static_cast<double>(provider_count - 1);
  state.counters["providers"] = static_cast<double>(provider_count);
  state.SetComplexityN(provider_count);
  loom_link_module_index_free(index);
}

static void CatalogScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

BENCHMARK(BM_ModuleIndex_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_ModuleIndex_OverlayOneInput_Catalog)
    ->Apply(CatalogScales)
    ->Complexity(benchmark::o1);
BENCHMARK(BM_Plan_Merge_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_Plan_LinkLeaf_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_Plan_LinkChain_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_Plan_InputExport_UnrelatedLibraryProviders)
    ->Apply(CatalogScales)
    ->Complexity(benchmark::o1);
BENCHMARK(BM_Project_LinkLeaf_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_Project_LinkChain_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_LinkExact_LinkLeaf_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_LinkExact_LinkChain_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_LinkExactDense_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_MaterializeAndLink_LinkLeaf_Catalog)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_MaterializeAndLink_LinkChain_Catalog)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_GlobalDuplicateEnumeration)->Apply(CatalogScales)->Complexity();

}  // namespace

BENCHMARK_MAIN();
