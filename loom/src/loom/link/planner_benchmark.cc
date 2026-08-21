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
#include "loom/link/provider_import_sink.h"
#include "loom/link/provider_resolver.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/module/ops.h"
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
  explicit PlannerCatalogFixture(uint32_t symbol_count,
                                 bool with_provider_imports = false)
      : symbol_count_(symbol_count),
        with_provider_imports_(with_provider_imports) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    CheckStatus(loom_test_dialect_register(&context_));
    iree_host_size_t module_op_count = 0;
    const loom_op_vtable_t* const* module_ops =
        loom_module_dialect_vtables(&module_op_count);
    CheckStatus(loom_context_register_dialect(
        &context_, LOOM_DIALECT_MODULE, module_ops, (uint16_t)module_op_count));
    iree_host_size_t module_semantic_count = 0;
    const loom_op_semantics_t* module_semantics =
        loom_module_dialect_op_semantics(&module_semantic_count);
    CheckStatus(loom_context_register_dialect_semantics(
        &context_, LOOM_DIALECT_MODULE, module_semantics,
        (uint16_t)module_semantic_count));
    CheckStatus(loom_context_finalize(&context_));

    loom_module_t* module = nullptr;
    CheckStatus(loom_module_allocate(&context_, IREE_SV("planner_catalog"),
                                     &block_pool_, nullptr,
                                     iree_allocator_system(), &module));
    BuildModule(module);
    SerializeModule(module);
    module_ = module;
  }

  PlannerCatalogFixture(const PlannerCatalogFixture&) = delete;
  PlannerCatalogFixture& operator=(const PlannerCatalogFixture&) = delete;

  ~PlannerCatalogFixture() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_link_module_index_t* BuildIndex() {
    loom_link_module_index_t* index = nullptr;
    CheckStatus(loom_link_module_index_create(&context_, &block_pool_,
                                              iree_allocator_system(), &index));
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/IREE_SV("catalog"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
    };
    CheckStatus(loom_link_module_index_add_bytecode(
        index, iree_make_const_byte_span(bytes_.data(), bytes_.size()),
        IREE_SV("planner_catalog.loombc"), /*index_options=*/nullptr, &options,
        /*out_provider_ordinal=*/nullptr));
    return index;
  }

  loom_link_module_index_t* BuildDuplicateIndex(uint32_t provider_count) {
    loom_link_module_index_t* index = nullptr;
    CheckStatus(loom_link_module_index_create(&context_, &block_pool_,
                                              iree_allocator_system(), &index));
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
    if (with_provider_imports_) {
      for (uint32_t i = 0; i < symbol_count_; ++i) {
        char provider_name[32];
        const int provider_name_length = std::snprintf(
            provider_name, sizeof(provider_name), "provider_%08u", i);
        if (provider_name_length <= 0 ||
            provider_name_length >= (int)sizeof(provider_name)) {
          std::abort();
        }
        loom_string_id_t provider_id = LOOM_STRING_ID_INVALID;
        CheckStatus(loom_module_intern_string(
            module, iree_make_string_view(provider_name, provider_name_length),
            &provider_id));
        const loom_symbol_ref_t anchor = {
            /*.module_id=*/0,
            /*.symbol_id=*/symbol_ids[i],
        };
        loom_op_t* import_op = nullptr;
        CheckStatus(loom_module_import_build(
            &module_builder, provider_id,
            loom_make_symbol_ref_array(&anchor, /*count=*/1),
            LOOM_LOCATION_NONE, &import_op));
      }
    }
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

  void SerializeModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    CheckStatus(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    CheckStatus(loom_bytecode_write_module(module, stream, /*options=*/nullptr,
                                           &block_pool_));
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    bytes_.resize((size_t)length);
    CheckStatus(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    CheckStatus(
        iree_io_stream_read(stream, bytes_.size(), bytes_.data(), nullptr));
    iree_io_stream_release(stream);
  }

  uint32_t symbol_count_;
  bool with_provider_imports_;
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  std::vector<uint8_t> bytes_;
};

// One imported declaration and a same-name definition in each candidate
// provider. Every import key aliases one chosen provider so exact planning must
// inspect the full import and candidate domains but selects only one body.
class ImportedCandidateFixture {
 public:
  explicit ImportedCandidateFixture(uint32_t candidate_count)
      : candidate_count_(candidate_count) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    CheckStatus(loom_test_dialect_register(&context_));
    iree_host_size_t module_op_count = 0;
    const loom_op_vtable_t* const* module_ops =
        loom_module_dialect_vtables(&module_op_count);
    CheckStatus(loom_context_register_dialect(
        &context_, LOOM_DIALECT_MODULE, module_ops, (uint16_t)module_op_count));
    iree_host_size_t module_semantic_count = 0;
    const loom_op_semantics_t* module_semantics =
        loom_module_dialect_op_semantics(&module_semantic_count);
    CheckStatus(loom_context_register_dialect_semantics(
        &context_, LOOM_DIALECT_MODULE, module_semantics,
        (uint16_t)module_semantic_count));
    iree_host_size_t func_op_count = 0;
    const loom_op_vtable_t* const* func_ops =
        loom_func_dialect_vtables(&func_op_count);
    CheckStatus(loom_context_register_dialect(
        &context_, LOOM_DIALECT_FUNC, func_ops, (uint16_t)func_op_count));
    iree_host_size_t func_semantic_count = 0;
    const loom_op_semantics_t* func_semantics =
        loom_func_dialect_op_semantics(&func_semantic_count);
    CheckStatus(loom_context_register_dialect_semantics(
        &context_, LOOM_DIALECT_FUNC, func_semantics,
        (uint16_t)func_semantic_count));
    CheckStatus(loom_context_finalize(&context_));

    provider_keys_.reserve(candidate_count_);
    for (uint32_t i = 0; i < candidate_count_; ++i) {
      char key[32];
      const int key_length =
          std::snprintf(key, sizeof(key), "provider_%08u", i);
      if (key_length <= 0 || key_length >= (int)sizeof(key)) {
        std::abort();
      }
      provider_keys_.emplace_back(key, (size_t)key_length);
    }

    BuildImporter();
    BuildCandidate();
    SerializeModule(importer_module_, &importer_bytes_);
    SerializeModule(candidate_module_, &candidate_bytes_);
  }

  ImportedCandidateFixture(const ImportedCandidateFixture&) = delete;
  ImportedCandidateFixture& operator=(const ImportedCandidateFixture&) = delete;

  ~ImportedCandidateFixture() {
    loom_module_free(candidate_module_);
    loom_module_free(importer_module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_link_module_index_t* BuildIndex(
      iree_host_size_t* out_chosen_provider_ordinal) {
    loom_link_module_index_t* index = nullptr;
    CheckStatus(loom_link_module_index_create(&context_, &block_pool_,
                                              iree_allocator_system(), &index));
    const loom_link_module_index_add_options_t importer_options = {
        /*.provider_name=*/IREE_SV("importer"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    CheckStatus(loom_link_module_index_add_bytecode(
        index,
        iree_make_const_byte_span(importer_bytes_.data(),
                                  importer_bytes_.size()),
        IREE_SV("importer.loombc"), /*index_options=*/nullptr,
        &importer_options, /*out_provider_ordinal=*/nullptr));

    *out_chosen_provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    for (uint32_t i = 0; i < candidate_count_; ++i) {
      char provider_name[32];
      const int provider_name_length = std::snprintf(
          provider_name, sizeof(provider_name), "candidate_%08u", i);
      if (provider_name_length <= 0 ||
          provider_name_length >= (int)sizeof(provider_name)) {
        std::abort();
      }
      const loom_link_module_index_add_options_t candidate_options = {
          /*.provider_name=*/iree_make_string_view(provider_name,
                                                   provider_name_length),
          /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
      };
      CheckStatus(loom_link_module_index_add_bytecode(
          index,
          iree_make_const_byte_span(candidate_bytes_.data(),
                                    candidate_bytes_.size()),
          IREE_SV("candidate.loombc"), /*index_options=*/nullptr,
          &candidate_options, out_chosen_provider_ordinal));
    }
    return index;
  }

  std::vector<loom_link_provider_binding_t> Bindings(
      iree_host_size_t chosen_provider_ordinal) const {
    std::vector<loom_link_provider_binding_t> bindings;
    bindings.reserve(candidate_count_);
    for (const std::string& key : provider_keys_) {
      bindings.push_back({
          iree_make_string_view(key.data(), key.size()),
          chosen_provider_ordinal,
      });
    }
    return bindings;
  }

  loom_context_t* context() { return &context_; }
  iree_arena_block_pool_t* block_pool() { return &block_pool_; }

 private:
  void BuildImporter() {
    CheckStatus(loom_module_allocate(
        &context_, IREE_SV("importer"), &block_pool_, nullptr,
        iree_allocator_system(), &importer_module_));
    loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
    CheckStatus(loom_module_intern_string(
        importer_module_, IREE_SV("function_00000000"), &symbol_name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    CheckStatus(
        loom_module_add_symbol(importer_module_, symbol_name_id, &symbol_id));
    const loom_symbol_ref_t symbol_ref = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };

    loom_builder_t builder;
    loom_builder_initialize(importer_module_, &importer_module_->arena,
                            loom_module_block(importer_module_), &builder);
    for (const std::string& key : provider_keys_) {
      loom_string_id_t provider_id = LOOM_STRING_ID_INVALID;
      CheckStatus(loom_module_intern_string(
          importer_module_, iree_make_string_view(key.data(), key.size()),
          &provider_id));
      loom_op_t* import_op = nullptr;
      CheckStatus(loom_module_import_build(
          &builder, provider_id,
          loom_make_symbol_ref_array(&symbol_ref, /*count=*/1),
          LOOM_LOCATION_NONE, &import_op));
    }
    loom_op_t* declaration_op = nullptr;
    CheckStatus(loom_func_decl_build(
        &builder, LOOM_FUNC_DECL_BUILD_FLAG_HAS_VISIBILITY,
        LOOM_FUNC_VISIBILITY_PUBLIC, /*retain=*/0,
        /*import_module=*/LOOM_STRING_ID_INVALID,
        /*import_symbol=*/LOOM_STRING_ID_INVALID, /*cc=*/0, /*purity=*/0,
        /*temperature=*/0, /*inline_policy=*/0, loom_symbol_ref_null(),
        /*abi=*/0, loom_named_attr_slice_empty(),
        /*export_symbol=*/LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
        symbol_ref,
        /*arg_types=*/nullptr, /*arg_types_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_NONE,
        &declaration_op));
  }

  void BuildCandidate() {
    CheckStatus(loom_module_allocate(
        &context_, IREE_SV("candidate"), &block_pool_, nullptr,
        iree_allocator_system(), &candidate_module_));
    loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
    CheckStatus(loom_module_intern_string(
        candidate_module_, IREE_SV("function_00000000"), &symbol_name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    CheckStatus(
        loom_module_add_symbol(candidate_module_, symbol_name_id, &symbol_id));

    loom_builder_t builder;
    loom_builder_initialize(candidate_module_, &candidate_module_->arena,
                            loom_module_block(candidate_module_), &builder);
    const loom_symbol_ref_t symbol_ref = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    loom_op_t* function_op = nullptr;
    CheckStatus(loom_func_def_build(
        &builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        /*export_symbol=*/LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
        symbol_ref,
        /*arg_types=*/nullptr, /*arg_types_count=*/0,
        /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_NONE,
        &function_op));
    const loom_func_like_t function =
        loom_func_like_cast(candidate_module_, function_op);
    loom_builder_t body_builder;
    loom_builder_initialize(
        candidate_module_, &candidate_module_->arena,
        loom_region_entry_block(loom_func_like_body(function)), &body_builder);
    loom_op_t* return_op = nullptr;
    CheckStatus(loom_func_return_build(&body_builder, /*values=*/nullptr,
                                       /*values_count=*/0, LOOM_LOCATION_NONE,
                                       &return_op));
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

  uint32_t candidate_count_;
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_module_t* importer_module_ = nullptr;
  loom_module_t* candidate_module_ = nullptr;
  std::vector<std::string> provider_keys_;
  std::vector<uint8_t> importer_bytes_;
  std::vector<uint8_t> candidate_bytes_;
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
    loom_link_module_index_t* index = fixture.BuildIndex();
    benchmark::DoNotOptimize(index);
    state.PauseTiming();
    loom_link_module_index_free(index);
    state.ResumeTiming();
  }
  SetCounters(state, fixture, /*selected_symbol_count=*/0);
}

static void BM_ModuleIndex_ProviderImports(benchmark::State& state) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0),
                                /*with_provider_imports=*/true);
  for (auto _ : state) {
    loom_link_module_index_t* index = fixture.BuildIndex();
    benchmark::DoNotOptimize(index);
    state.PauseTiming();
    loom_link_module_index_free(index);
    state.ResumeTiming();
  }
  const double count = static_cast<double>(fixture.symbol_count());
  state.counters["import_anchors"] = count;
  state.counters["import_index_bytes"] = (2.0 * count + 1.0) * sizeof(uint32_t);
  SetCounters(state, fixture, /*selected_symbol_count=*/0);
}

static void BM_ProviderImportOccurrenceLookup(benchmark::State& state) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0),
                                /*with_provider_imports=*/true);
  loom_link_module_index_t* index = fixture.BuildIndex();
  const loom_link_module_index_module_t* module =
      loom_link_module_index_module_at(index, 0);
  if (!module) {
    std::abort();
  }

  for (auto _ : state) {
    iree_host_size_t occurrence_count = 0;
    uint64_t import_checksum = 0;
    for (iree_host_size_t i = 0; i < module->symbol_count; ++i) {
      const loom_link_module_index_symbol_t* symbol =
          loom_link_module_index_symbol_at(index,
                                           module->symbol_start_ordinal + i);
      const loom_link_module_index_provider_import_list_t imports =
          loom_link_module_index_symbol_provider_imports(index, symbol);
      occurrence_count += imports.count;
      if (imports.count > 0) {
        import_checksum += imports.values[0];
      }
    }
    if (occurrence_count != fixture.symbol_count()) {
      std::abort();
    }
    benchmark::DoNotOptimize(occurrence_count);
    benchmark::DoNotOptimize(import_checksum);
  }
  state.counters["import_anchors"] =
      static_cast<double>(fixture.symbol_count());
  SetCounters(state, fixture, /*selected_symbol_count=*/0);
  loom_link_module_index_free(index);
}

static void BenchmarkPlan(benchmark::State& state, loom_link_plan_mode_t mode,
                          uint32_t root_ordinal,
                          iree_host_size_t expected_symbol_count) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  loom_link_module_index_t* index = fixture.BuildIndex();
  const std::string root_name = fixture.SymbolName(root_ordinal);
  const iree_string_view_t root =
      iree_make_string_view(root_name.data(), root_name.size());
  iree_string_view_list_t root_symbols = iree_string_view_list_empty();
  if (mode == LOOM_LINK_PLAN_SELECTIVE) {
    root_symbols = (iree_string_view_list_t){
        /*.count=*/1,
        /*.values=*/&root,
    };
  }
  const loom_link_plan_options_t options = {
      /*.mode=*/mode,
      /*.root_symbols=*/root_symbols,
  };

  for (auto _ : state) {
    loom_link_plan_t* plan = nullptr;
    CheckStatus(
        loom_link_plan_build(index, &options, iree_allocator_system(), &plan));
    if (loom_link_plan_symbol_count(plan) != expected_symbol_count) {
      std::abort();
    }
    benchmark::DoNotOptimize(plan);
    state.PauseTiming();
    loom_link_plan_free(plan);
    state.ResumeTiming();
  }
  SetCounters(state, fixture, expected_symbol_count);
  loom_link_module_index_free(index);
}

static void BM_Plan_Archive_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkPlan(state, LOOM_LINK_PLAN_ARCHIVE, /*root_ordinal=*/0,
                symbol_count);
}

static void BM_Plan_SelectiveLeaf_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkPlan(state, LOOM_LINK_PLAN_SELECTIVE,
                /*root_ordinal=*/symbol_count - 1,
                /*expected_symbol_count=*/1);
}

static void BM_Plan_SelectiveChain_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkPlan(state, LOOM_LINK_PLAN_SELECTIVE, /*root_ordinal=*/0,
                symbol_count);
}

static void BM_Plan_ImportedCandidateResolution(benchmark::State& state) {
  const uint32_t candidate_count = (uint32_t)state.range(0);
  ImportedCandidateFixture fixture(candidate_count);
  iree_host_size_t chosen_provider_ordinal =
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  loom_link_module_index_t* index =
      fixture.BuildIndex(&chosen_provider_ordinal);
  std::vector<loom_link_provider_binding_t> bindings =
      fixture.Bindings(chosen_provider_ordinal);
  loom_link_provider_resolver_t resolver = {};
  CheckStatus(loom_link_provider_resolver_prepare(
      loom_link_module_index_provider_count(index), bindings.data(),
      bindings.size(), &resolver));
  const iree_string_view_t root = IREE_SV("@function_00000000");
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/1, /*.values=*/&root},
  };
  options.provider_resolver = &resolver;
  const loom_link_module_index_module_t* importer_module =
      loom_link_module_index_module_at(index, 0);
  if (!importer_module) {
    std::abort();
  }
  const iree_host_size_t declaration_ordinal =
      importer_module->symbol_start_ordinal;

  for (auto _ : state) {
    loom_link_plan_t* plan = nullptr;
    CheckStatus(
        loom_link_plan_build(index, &options, iree_allocator_system(), &plan));
    if (loom_link_plan_symbol_count(plan) != 2 ||
        !loom_link_plan_symbol_imports_resolved(plan, declaration_ordinal)) {
      std::abort();
    }
    benchmark::DoNotOptimize(plan);
    state.PauseTiming();
    loom_link_plan_free(plan);
    state.ResumeTiming();
  }

  const double candidate_bitmap_bytes = static_cast<double>(
      ((candidate_count + 1u + 63u) / 64u) * sizeof(uint64_t));
  state.counters["candidate_bitmap_bytes"] = candidate_bitmap_bytes;
  state.counters["provider_bindings"] = static_cast<double>(candidate_count);
  state.counters["providers"] = static_cast<double>(candidate_count + 1u);
  state.counters["rejected_candidates"] =
      static_cast<double>(candidate_count - 1u);
  state.counters["resolver_bytes"] = static_cast<double>(
      candidate_count * sizeof(loom_link_provider_binding_t));
  state.counters["selected_symbols"] = 2.0;
  state.SetComplexityN(candidate_count);
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

static void BM_Link_ImportedCandidateEndToEnd(benchmark::State& state) {
  const uint32_t candidate_count = (uint32_t)state.range(0);
  ImportedCandidateFixture fixture(candidate_count);
  iree_host_size_t chosen_provider_ordinal =
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  loom_link_module_index_t* index =
      fixture.BuildIndex(&chosen_provider_ordinal);
  std::vector<loom_link_provider_binding_t> bindings =
      fixture.Bindings(chosen_provider_ordinal);
  loom_link_provider_resolver_t resolver = {};
  CheckStatus(loom_link_provider_resolver_prepare(
      loom_link_module_index_provider_count(index), bindings.data(),
      bindings.size(), &resolver));
  const iree_string_view_t root = IREE_SV("@function_00000000");
  const iree_string_view_list_t roots = {
      /*.count=*/1,
      /*.values=*/&root,
  };
  loom_link_plan_options_t plan_options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/roots,
  };
  plan_options.provider_resolver = &resolver;

  iree_arena_allocator_t projection_arena;
  iree_arena_initialize(fixture.block_pool(), &projection_arena);
  for (auto _ : state) {
    loom_link_plan_t* plan = nullptr;
    CheckStatus(loom_link_plan_build(index, &plan_options,
                                     iree_allocator_system(), &plan));
    loom_link_plan_module_projection_t module_projection = {};
    CheckStatus(loom_link_plan_project_modules(plan, &projection_arena,
                                               &module_projection));
    loom_link_plan_linker_import_projection_t import_projection = {};
    CheckStatus(loom_link_plan_project_linker_imports(
        index, &module_projection, &projection_arena, &import_projection));
    if (loom_link_plan_symbol_count(plan) != 2 ||
        module_projection.modules.count != 2 ||
        import_projection.provider_imports.count != 0) {
      std::abort();
    }

    const loom_linker_options_t linker_options = {
        /*.module_name=*/IREE_SV("linked"),
        /*.provider_imports=*/
        {
            /*.count=*/import_projection.provider_imports.count,
            /*.anchor_count=*/import_projection.provider_import_anchors.count,
        },
    };
    loom_linker_t* linker = nullptr;
    CheckStatus(loom_linker_create(fixture.context(), &linker_options,
                                   fixture.block_pool(),
                                   iree_allocator_system(), &linker));
    for (iree_host_size_t i = 0; i < module_projection.modules.count; ++i) {
      const loom_link_plan_module_selection_t& selection =
          module_projection.modules.values[i];
      if (selection.symbols.count != selection.source_module->symbol_count) {
        std::abort();
      }
      loom_module_t* materialized_module = MaterializeIndexedBytecodeModule(
          index, selection.source_module, fixture.context(),
          fixture.block_pool());
      CheckStatus(loom_linker_add_exact_module(
          linker, materialized_module, import_projection.modules.values[i],
          loom_linker_target_symbol_list_empty()));
      loom_module_free(materialized_module);
    }
    CheckStatus(loom_linker_finalize_roots(linker, roots));
    loom_module_t* linked_module = nullptr;
    CheckStatus(loom_linker_finish(linker, &linked_module));
    if (linked_module == nullptr || linked_module->symbols.count != 1) {
      std::abort();
    }
    benchmark::DoNotOptimize(linked_module);

    state.PauseTiming();
    loom_module_free(linked_module);
    loom_linker_free(linker);
    loom_link_plan_free(plan);
    iree_arena_reset(&projection_arena);
    state.ResumeTiming();
  }

  state.counters["materialized_modules"] = 2.0;
  state.counters["provider_bindings"] = static_cast<double>(candidate_count);
  state.counters["providers"] = static_cast<double>(candidate_count + 1u);
  state.counters["rejected_provider_bodies"] =
      static_cast<double>(candidate_count - 1u);
  state.counters["resolver_bytes"] = static_cast<double>(
      candidate_count * sizeof(loom_link_provider_binding_t));
  state.counters["selected_symbols"] = 2.0;
  state.SetComplexityN(candidate_count);
  iree_arena_deinitialize(&projection_arena);
  loom_link_module_index_free(index);
}

static void BenchmarkProjection(benchmark::State& state, uint32_t root_ordinal,
                                iree_host_size_t expected_symbol_count) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  loom_link_module_index_t* index = fixture.BuildIndex();
  const std::string root_name = fixture.SymbolName(root_ordinal);
  const iree_string_view_t root =
      iree_make_string_view(root_name.data(), root_name.size());
  const loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
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

static void BM_Project_SelectiveLeaf_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkProjection(state, /*root_ordinal=*/symbol_count - 1,
                      /*expected_symbol_count=*/1);
}

static void BM_Project_SelectiveChain_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkProjection(state, /*root_ordinal=*/0, symbol_count);
}

static void BM_Project_RetainedProviderImports(benchmark::State& state) {
  const uint32_t provider_import_count = (uint32_t)state.range(0);
  ImportedCandidateFixture fixture(provider_import_count);
  iree_host_size_t ignored_provider_ordinal =
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  loom_link_module_index_t* index =
      fixture.BuildIndex(&ignored_provider_ordinal);
  const iree_string_view_t root = IREE_SV("@function_00000000");
  const loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/1, /*.values=*/&root},
      /*.include_exported_roots=*/false,
      /*.unresolved_policy=*/LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
  };
  loom_link_plan_t* plan = nullptr;
  CheckStatus(
      loom_link_plan_build(index, &options, iree_allocator_system(), &plan));
  if (loom_link_plan_symbol_count(plan) != 1) {
    std::abort();
  }

  iree_arena_block_pool_t projection_block_pool;
  iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                   &projection_block_pool);
  iree_arena_allocator_t projection_arena;
  iree_arena_initialize(&projection_block_pool, &projection_arena);
  for (auto _ : state) {
    loom_link_plan_module_projection_t projection;
    CheckStatus(
        loom_link_plan_project_modules(plan, &projection_arena, &projection));
    if (projection.modules.count != 1 || projection.symbols.count != 1 ||
        projection.provider_imports.count != provider_import_count ||
        projection.provider_import_anchors.count != provider_import_count) {
      std::abort();
    }
    benchmark::DoNotOptimize(projection.provider_imports.values);
    benchmark::DoNotOptimize(projection.provider_import_anchors.values);
    state.PauseTiming();
    iree_arena_reset(&projection_arena);
    state.ResumeTiming();
  }

  state.counters["import_anchors"] = static_cast<double>(provider_import_count);
  state.counters["provider_imports"] =
      static_cast<double>(provider_import_count);
  state.counters["projection_bytes"] = static_cast<double>(
      sizeof(loom_link_plan_module_selection_t) +
      sizeof(loom_link_plan_module_symbol_t) +
      provider_import_count *
          (sizeof(loom_link_plan_module_provider_import_t) + sizeof(uint32_t)));
  state.SetComplexityN(provider_import_count);
  iree_arena_deinitialize(&projection_arena);
  iree_arena_block_pool_deinitialize(&projection_block_pool);
  loom_link_plan_free(plan);
  loom_link_module_index_free(index);
}

static void BM_Project_LinkerImports(benchmark::State& state) {
  const uint32_t provider_import_count = (uint32_t)state.range(0);
  PlannerCatalogFixture fixture(provider_import_count,
                                /*with_provider_imports=*/true);
  loom_link_module_index_t* index = fixture.BuildIndex();
  const std::string root_name = fixture.SymbolName(/*ordinal=*/0);
  const iree_string_view_t root =
      iree_make_string_view(root_name.data(), root_name.size());
  const loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/1, /*.values=*/&root},
      /*.include_exported_roots=*/false,
      /*.unresolved_policy=*/LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
  };
  loom_link_plan_t* plan = nullptr;
  CheckStatus(
      loom_link_plan_build(index, &options, iree_allocator_system(), &plan));

  iree_arena_block_pool_t projection_block_pool;
  iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                   &projection_block_pool);
  iree_arena_allocator_t module_arena;
  iree_arena_initialize(&projection_block_pool, &module_arena);
  loom_link_plan_module_projection_t module_projection;
  CheckStatus(
      loom_link_plan_project_modules(plan, &module_arena, &module_projection));
  if (module_projection.modules.count != 1 ||
      module_projection.symbols.count != provider_import_count ||
      module_projection.provider_imports.count != provider_import_count) {
    std::abort();
  }

  iree_arena_allocator_t linker_arena;
  iree_arena_initialize(&projection_block_pool, &linker_arena);
  for (auto _ : state) {
    loom_link_plan_linker_import_projection_t linker_projection;
    CheckStatus(loom_link_plan_project_linker_imports(
        index, &module_projection, &linker_arena, &linker_projection));
    if (linker_projection.modules.count != 1 ||
        linker_projection.provider_imports.count != provider_import_count ||
        linker_projection.provider_import_anchors.count !=
            provider_import_count) {
      std::abort();
    }
    benchmark::DoNotOptimize(linker_projection.provider_imports.values);
    benchmark::DoNotOptimize(linker_projection.provider_import_anchors.values);
    state.PauseTiming();
    iree_arena_reset(&linker_arena);
    state.ResumeTiming();
  }

  state.counters["import_anchors"] = static_cast<double>(provider_import_count);
  state.counters["provider_imports"] =
      static_cast<double>(provider_import_count);
  state.counters["projection_bytes"] = static_cast<double>(
      sizeof(loom_linker_source_provider_import_list_t) +
      provider_import_count *
          (sizeof(loom_linker_source_provider_import_t) + sizeof(uint32_t)));
  SetCounters(state, fixture, provider_import_count);
  iree_arena_deinitialize(&linker_arena);
  iree_arena_deinitialize(&module_arena);
  iree_arena_block_pool_deinitialize(&projection_block_pool);
  loom_link_plan_free(plan);
  loom_link_module_index_free(index);
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
    CheckStatus(loom_linker_create(fixture.module()->context, &linker_options,
                                   fixture.block_pool(),
                                   iree_allocator_system(), &linker));
    state.ResumeTiming();
    CheckStatus(loom_linker_add_module_symbols(
        linker, fixture.module(), source_symbols,
        loom_linker_source_provider_import_list_empty(),
        loom_linker_target_symbol_list_empty()));
    benchmark::DoNotOptimize(linker);
    state.PauseTiming();
    loom_linker_free(linker);
    state.ResumeTiming();
  }
  SetCounters(state, fixture, selected_symbol_count);
}

static void BM_LinkExact_SelectiveLeaf_Catalog(benchmark::State& state) {
  BenchmarkExactLink(state, /*selected_symbol_count=*/1);
}

static void BM_LinkExact_SelectiveChain_Catalog(benchmark::State& state) {
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
    CheckStatus(loom_linker_create(fixture.module()->context, &linker_options,
                                   fixture.block_pool(),
                                   iree_allocator_system(), &linker));
    state.ResumeTiming();
    CheckStatus(loom_linker_add_exact_module(
        linker, fixture.module(),
        loom_linker_source_provider_import_list_empty(),
        loom_linker_target_symbol_list_empty()));
    benchmark::DoNotOptimize(linker);
    state.PauseTiming();
    loom_linker_free(linker);
    state.ResumeTiming();
  }
  SetCounters(state, fixture, fixture.symbol_count());
}

static void BM_LinkProjectedProviderImports(benchmark::State& state) {
  const uint32_t provider_import_count = (uint32_t)state.range(0);
  PlannerCatalogFixture fixture(provider_import_count);
  std::vector<std::string> provider_names(provider_import_count);
  for (uint32_t i = 0; i < provider_import_count; ++i) {
    char provider_name[32];
    const int provider_name_length =
        std::snprintf(provider_name, sizeof(provider_name), "provider_%08u", i);
    if (provider_name_length <= 0 ||
        provider_name_length >= (int)sizeof(provider_name)) {
      std::abort();
    }
    provider_names[i] = std::string(provider_name, provider_name_length);
  }

  for (auto _ : state) {
    state.PauseTiming();
    loom_module_t* target_module = nullptr;
    CheckStatus(loom_module_allocate(
        fixture.module()->context, IREE_SV("linked"), fixture.block_pool(),
        /*hints=*/nullptr, iree_allocator_system(), &target_module));
    for (uint32_t i = 0; i < provider_import_count; ++i) {
      const std::string symbol_name = fixture.SymbolName(i);
      loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
      CheckStatus(loom_module_intern_string(
          target_module,
          iree_make_string_view(symbol_name.data(), symbol_name.size()),
          &name_id));
      uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
      CheckStatus(loom_module_add_symbol(target_module, name_id, &symbol_id));
      if (symbol_id != i) {
        std::abort();
      }
    }
    iree_arena_allocator_t sink_arena;
    iree_arena_initialize(fixture.block_pool(), &sink_arena);
    loom_link_provider_import_sink_t sink;
    CheckStatus(loom_link_provider_import_sink_initialize(
        target_module, &sink_arena, provider_import_count,
        provider_import_count, &sink));
    state.ResumeTiming();

    for (uint32_t i = 0; i < provider_import_count; ++i) {
      const loom_symbol_ref_t anchor = {
          /*.module_id=*/0,
          /*.symbol_id=*/(uint16_t)i,
      };
      CheckStatus(loom_link_provider_import_sink_append(
          &sink,
          iree_make_string_view(provider_names[i].data(),
                                provider_names[i].size()),
          loom_make_symbol_ref_array(&anchor, /*count=*/1),
          /*comments=*/{}, /*leading_blank_line=*/false));
    }
    CheckStatus(loom_link_provider_import_sink_finish(&sink));
    benchmark::DoNotOptimize(target_module);

    state.PauseTiming();
    loom_module_free(target_module);
    iree_arena_deinitialize(&sink_arena);
    state.ResumeTiming();
  }
  state.counters["import_anchors"] = static_cast<double>(provider_import_count);
  state.counters["provider_imports"] =
      static_cast<double>(provider_import_count);
  state.counters["sink_bytes"] = static_cast<double>(
      provider_import_count * (sizeof(loom_link_provider_import_sink_row_t) +
                               sizeof(loom_symbol_ref_t)));
  state.SetComplexityN(provider_import_count);
}

static void BenchmarkSelectiveMaterializeAndLink(
    benchmark::State& state, uint32_t root_ordinal,
    iree_host_size_t expected_symbol_count) {
  PlannerCatalogFixture fixture((uint32_t)state.range(0));
  loom_link_module_index_t* index = fixture.BuildIndex();
  const std::string root_name = fixture.SymbolName(root_ordinal);
  const iree_string_view_t root =
      iree_make_string_view(root_name.data(), root_name.size());
  const loom_link_plan_options_t plan_options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
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
    CheckStatus(loom_linker_create(fixture.module()->context, &linker_options,
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
        linker, selected_module,
        loom_linker_source_provider_import_list_empty(),
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

static void BM_MaterializeAndLink_SelectiveLeaf_Catalog(
    benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkSelectiveMaterializeAndLink(state, /*root_ordinal=*/symbol_count - 1,
                                       /*expected_symbol_count=*/1);
}

static void BM_MaterializeAndLink_SelectiveChain_Catalog(
    benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  BenchmarkSelectiveMaterializeAndLink(state, /*root_ordinal=*/0,
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
BENCHMARK(BM_ModuleIndex_ProviderImports)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_ProviderImportOccurrenceLookup)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_Plan_Archive_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_Plan_SelectiveLeaf_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_Plan_SelectiveChain_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_Plan_ImportedCandidateResolution)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_Link_ImportedCandidateEndToEnd)
    ->Apply(CatalogScales)
    ->Complexity(benchmark::oNLogN);
BENCHMARK(BM_Project_SelectiveLeaf_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_Project_SelectiveChain_Catalog)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_Project_RetainedProviderImports)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_Project_LinkerImports)
    ->Apply(CatalogScales)
    ->Complexity(benchmark::oNLogN);
BENCHMARK(BM_LinkExact_SelectiveLeaf_Catalog)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_LinkExact_SelectiveChain_Catalog)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_LinkExactDense_Catalog)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_LinkProjectedProviderImports)->Apply(CatalogScales)->Complexity();
BENCHMARK(BM_MaterializeAndLink_SelectiveLeaf_Catalog)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_MaterializeAndLink_SelectiveChain_Catalog)
    ->Apply(CatalogScales)
    ->Complexity();
BENCHMARK(BM_GlobalDuplicateEnumeration)->Apply(CatalogScales)->Complexity();

}  // namespace

BENCHMARK_MAIN();
