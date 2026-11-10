// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks template selection with a warm index and block pool. Library
// construction and output inspection are excluded from materialization timing.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/index_materializer.h"
#include "loom/link/module_index.h"
#include "loom/ops/op_registry.h"
#include "loom/verify/verify.h"

namespace {

static void CheckStatus(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

enum class TemplateProviderContractShape {
  kPriorityOnly,
  kComplete,
};

enum class TemplateProviderRepresentation {
  kBytecode,
  kMaterialized,
};

class TemplateCatalogFixture {
 public:
  TemplateCatalogFixture(uint32_t provider_count,
                         TemplateProviderContractShape contract_shape,
                         uint32_t unrelated_symbol_count = 0,
                         uint32_t family_count = 1,
                         TemplateProviderRepresentation representation =
                             TemplateProviderRepresentation::kBytecode)
      : provider_count_(provider_count),
        contract_shape_(contract_shape),
        unrelated_symbol_count_(unrelated_symbol_count),
        family_count_(family_count) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    CheckStatus(loom_op_registry_register_all_dialects(&context_));
    CheckStatus(loom_context_finalize(&context_));

    root_module_ = ParseModule(BuildRootSource(), IREE_SV("root.loom"));
    library_module_ =
        ParseModule(BuildLibrarySource(), IREE_SV("library.loom"));
    library_bytecode_ = WriteModule(library_module_);

    CheckStatus(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index_));
    const loom_link_module_index_add_options_t root_options = {
        .provider_name = IREE_SV("root"),
        .role = LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    CheckStatus(loom_link_module_index_add_materialized(
        index_, root_module_, &root_options,
        /*out_provider_ordinal=*/nullptr));
    const loom_link_module_index_add_options_t library_options = {
        .provider_name = IREE_SV("library"),
        .role = LOOM_LINK_PROVIDER_ROLE_LIBRARY,
    };
    if (representation == TemplateProviderRepresentation::kMaterialized) {
      CheckStatus(loom_link_module_index_add_materialized(
          index_, library_module_, &library_options,
          /*out_provider_ordinal=*/nullptr));
    } else {
      CheckStatus(loom_link_module_index_add_bytecode(
          index_,
          iree_make_const_byte_span(library_bytecode_.data(),
                                    library_bytecode_.size()),
          IREE_SV("library.loombc"), /*index_options=*/nullptr,
          &library_options,
          /*out_provider_ordinal=*/nullptr));
    }

    const loom_link_module_index_module_t* library =
        loom_link_module_index_module_at(index_, 1);
    if (library == nullptr) {
      std::abort();
    }
    for (uint32_t i = 0; i < family_count_; ++i) {
      const uint32_t selected_ordinal =
          provider_count_ - 1 - ((provider_count_ - 1 - i) % family_count_);
      const std::string selected_name = ProviderName(selected_ordinal);
      const loom_link_module_index_symbol_t* selected =
          loom_link_module_index_lookup_private(
              index_, library,
              iree_make_string_view(selected_name.data(),
                                    selected_name.size()));
      if (selected == nullptr) {
        std::abort();
      }
      selected_provider_ordinals_.push_back(selected->ordinal);
    }
  }

  TemplateCatalogFixture(const TemplateCatalogFixture&) = delete;
  TemplateCatalogFixture& operator=(const TemplateCatalogFixture&) = delete;

  ~TemplateCatalogFixture() {
    loom_link_module_index_free(index_);
    loom_module_free(library_module_);
    loom_module_free(root_module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_link_index_materialization_t Materialize() {
    const iree_string_view_t root = IREE_SV("@entry");
    loom_link_plan_options_t options = {
        .mode = LOOM_LINK_PLAN_LINK,
        .root_symbols = {.count = 1, .values = &root},
        .unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ERROR,
    };
    loom_link_plan_materialization_environment_t environment = {
        .context = &context_,
        .block_pool = &block_pool_,
        .allocator = iree_allocator_system(),
    };
    loom_link_index_materialization_t materialization = {};
    CheckStatus(loom_link_index_materialize(
        index_, &options, &environment, IREE_SV("linked"), &materialization));
    return materialization;
  }

  void CheckSelection(const loom_link_plan_t* plan) const {
    for (iree_host_size_t ordinal : selected_provider_ordinals_) {
      if (!loom_link_plan_contains_symbol(plan, ordinal)) {
        std::abort();
      }
    }
  }

  uint32_t provider_count() const { return provider_count_; }
  iree_host_size_t library_byte_count() const {
    return library_bytecode_.size();
  }
  iree_arena_block_pool_statistics_t PoolStatistics() const {
    iree_arena_block_pool_statistics_t statistics = {};
    iree_arena_block_pool_query_statistics(&block_pool_, &statistics);
    return statistics;
  }

  static iree_host_size_t CountProviderBodies(const loom_module_t* module) {
    iree_host_size_t count = 0;
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      count += module->symbols.entries[i].kind == LOOM_SYMBOL_TEMPLATE_DEF;
    }
    return count;
  }

 private:
  static std::string ProviderName(uint32_t ordinal) {
    char name[32];
    const int length =
        std::snprintf(name, sizeof(name), "provider_%08u", ordinal);
    if (length <= 0 || length >= static_cast<int>(sizeof(name))) {
      std::abort();
    }
    return std::string(name, static_cast<size_t>(length));
  }

  std::string FamilyName(uint32_t ordinal) const {
    return family_count_ == 1 ? "benchmark.choose"
                              : "benchmark.choose_" + std::to_string(ordinal);
  }

  std::string BuildFamilyDeclarations() const {
    std::string source;
    for (uint32_t i = 0; i < family_count_; ++i) {
      source.append("template.decl @" + FamilyName(i) +
                    "(%value: i32) -> (i32)\n");
    }
    return source;
  }

  std::string BuildRootSource() const {
    const bool constrained =
        contract_shape_ == TemplateProviderContractShape::kComplete;
    std::string source = constrained
                             ? "target.generic<reference> @request_target "
                               "{subgroup_size = 64}\n"
                             : "";
    source.append(BuildFamilyDeclarations());
    source.append("func.def public ");
    if (constrained) {
      source.append("target(@request_target) ");
    }
    source.append("@entry(%value: i32) -> (i32)");
    if (constrained) {
      source.append(" where [range(%value, 32, 32)]");
    }
    source.append(" {\n");
    for (uint32_t i = 0; i < family_count_; ++i) {
      source.append("  %result_" + std::to_string(i) + " = template.apply<@" +
                    FamilyName(i) + ">(%value) : (i32) -> (i32)\n");
    }
    source.append("  func.return %result_" + std::to_string(family_count_ - 1) +
                  " : i32\n}\n");
    return source;
  }

  std::string BuildLibrarySource() const {
    std::string source = BuildFamilyDeclarations();
    if (contract_shape_ == TemplateProviderContractShape::kComplete) {
      source.insert(0, "target.generic<reference> @provider_target\n\n");
    }
    source.reserve(source.size() + provider_count_ * 260u +
                   unrelated_symbol_count_ * 64u);
    for (uint32_t i = 0; i < provider_count_; ++i) {
      const std::string name = ProviderName(i);
      source.append("template.def<@" + FamilyName(i % family_count_) + "> ");
      if (contract_shape_ == TemplateProviderContractShape::kComplete) {
        source.append(
            "target(@provider_target) requires [#target.subgroup.size<64>] ");
      }
      source.append("priority(");
      source.append(std::to_string(i + 1));
      source.append(") @");
      source.append(name);
      if (contract_shape_ == TemplateProviderContractShape::kComplete) {
        source.append(
            "(%value: i32) -> (i32) where [multiple_of(%value, 16)] {\n"
            "  template.return %value : i32\n"
            "}\n\n");
        continue;
      }
      source.append(
          "(%value: i32) -> (i32) {\n"
          "  template.return %value : i32\n"
          "}\n\n");
    }
    for (uint32_t i = 0; i < unrelated_symbol_count_; ++i) {
      source.append("func.def @unrelated_");
      source.append(std::to_string(i));
      source.append("() { func.return }\n");
    }
    return source;
  }

  loom_module_t* ParseModule(const std::string& source,
                             iree_string_view_t filename) {
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    CheckStatus(loom_text_parse(
        iree_make_string_view(source.data(), source.size()), filename,
        &context_, &block_pool_, &parse_options, &module));
    if (module == nullptr) {
      std::abort();
    }
    loom_verify_options_t verify_options = {};
    verify_options.sink.fn = loom_diagnostic_stderr_sink;
    verify_options.max_errors = 20;
    loom_verify_result_t verify_result = {};
    CheckStatus(loom_verify_module(module, &verify_options, &verify_result));
    if (verify_result.error_count != 0) {
      std::abort();
    }
    return module;
  }

  loom_module_t* ParseModule(const char* source, iree_string_view_t filename) {
    return ParseModule(std::string(source), filename);
  }

  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    CheckStatus(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    CheckStatus(loom_bytecode_write_module(module, stream, /*options=*/nullptr,
                                           &block_pool_));
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    CheckStatus(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    CheckStatus(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  // Providers distributed across the demanded families.
  uint32_t provider_count_;
  // Applicability requirements shared by the providers.
  TemplateProviderContractShape contract_shape_;
  // Additional library functions outside the selected closure.
  uint32_t unrelated_symbol_count_;
  // Distinct families applied by the root function.
  uint32_t family_count_;
  // Shared arena blocks reused by indexing and repeated materializations.
  iree_arena_block_pool_t block_pool_ = {};
  // Dialect registry shared by every source and product module.
  loom_context_t context_ = {};
  // Materialized input owning the demanded entry function.
  loom_module_t* root_module_ = nullptr;
  // Parsed source used to serialize the library fixture.
  loom_module_t* library_module_ = nullptr;
  // Bytecode borrowed by the warm library index.
  std::vector<uint8_t> library_bytecode_;
  // Reusable provider index over the input and bytecode library.
  loom_link_module_index_t* index_ = nullptr;
  // Highest-priority providers that every materialization must select.
  std::vector<iree_host_size_t> selected_provider_ordinals_;
};

static void RunTemplateSelectionBenchmark(
    benchmark::State& state, uint32_t provider_count,
    TemplateProviderContractShape contract_shape,
    uint32_t unrelated_symbol_count = 0, uint32_t family_count = 1,
    TemplateProviderRepresentation representation =
        TemplateProviderRepresentation::kBytecode) {
  TemplateCatalogFixture fixture(provider_count, contract_shape,
                                 unrelated_symbol_count, family_count,
                                 representation);
  // Warm reusable blocks separately from repeated oversized arena requests.
  auto warmup = fixture.Materialize();
  fixture.CheckSelection(warmup.plan);
  loom_link_index_materialization_deinitialize(&warmup);
  const auto before = fixture.PoolStatistics();
  iree_host_size_t selected_symbol_count = 0;
  iree_host_size_t selected_body_count = 0;
  iree_host_size_t module_arena_bytes = 0;
  iree_host_size_t projection_arena_bytes = 0;
  for (auto _ : state) {
    loom_link_index_materialization_t materialization = fixture.Materialize();
    benchmark::DoNotOptimize(materialization.product.module);

    state.PauseTiming();
    fixture.CheckSelection(materialization.plan);
    selected_symbol_count = loom_link_plan_symbol_count(materialization.plan);
    selected_body_count =
        fixture.CountProviderBodies(materialization.product.module);
    if (selected_body_count != family_count) {
      std::abort();
    }
    module_arena_bytes =
        materialization.product.module->arena.used_allocation_size;
    projection_arena_bytes = materialization.arena.used_allocation_size;
    loom_link_index_materialization_deinitialize(&materialization);
    state.ResumeTiming();
  }
  const auto after = fixture.PoolStatistics();
  const double iterations = static_cast<double>(state.iterations());
  state.counters["pool_system_allocations_per_link"] =
      static_cast<double>(after.block_system_allocation_count -
                          before.block_system_allocation_count) /
      iterations;
  state.counters["oversized_allocations_per_link"] =
      static_cast<double>(after.oversized_allocation_count -
                          before.oversized_allocation_count) /
      iterations;
  state.counters["oversized_bytes_per_link"] =
      static_cast<double>(after.oversized_allocation_bytes -
                          before.oversized_allocation_bytes) /
      iterations;
  state.counters["module_arena_bytes"] =
      static_cast<double>(module_arena_bytes);
  state.counters["projection_arena_bytes"] =
      static_cast<double>(projection_arena_bytes);
  state.counters["library_bytecode_bytes"] =
      static_cast<double>(fixture.library_byte_count());
  state.counters["providers"] = static_cast<double>(fixture.provider_count());
  state.counters["families"] = static_cast<double>(family_count);
  state.counters["unrelated_symbols"] =
      static_cast<double>(unrelated_symbol_count);
  state.counters["selected_provider_bodies"] =
      static_cast<double>(selected_body_count);
  state.counters["selected_symbols"] =
      static_cast<double>(selected_symbol_count);
  state.SetItemsProcessed(state.iterations() * fixture.provider_count());
  state.SetComplexityN(provider_count + unrelated_symbol_count);
}

static void BM_IndexMaterialize_SelectOneTemplate(benchmark::State& state) {
  RunTemplateSelectionBenchmark(state, static_cast<uint32_t>(state.range(0)),
                                TemplateProviderContractShape::kPriorityOnly);
}

static void BM_IndexMaterialize_SelectOneConstrainedTemplate(
    benchmark::State& state) {
  RunTemplateSelectionBenchmark(state, static_cast<uint32_t>(state.range(0)),
                                TemplateProviderContractShape::kComplete);
}

static void BM_IndexMaterialize_UnrelatedLibrarySymbols(
    benchmark::State& state) {
  RunTemplateSelectionBenchmark(state, /*provider_count=*/1,
                                TemplateProviderContractShape::kComplete,
                                static_cast<uint32_t>(state.range(0)));
}

static void BM_IndexMaterialize_SelectTemplateFamilies(
    benchmark::State& state) {
  RunTemplateSelectionBenchmark(
      state, /*provider_count=*/4096, TemplateProviderContractShape::kComplete,
      /*unrelated_symbol_count=*/0, static_cast<uint32_t>(state.range(0)));
}

static void BM_IndexMaterialize_SelectMaterializedTemplates(
    benchmark::State& state) {
  RunTemplateSelectionBenchmark(
      state, static_cast<uint32_t>(state.range(0)),
      TemplateProviderContractShape::kComplete, /*unrelated_symbol_count=*/0,
      /*family_count=*/1, TemplateProviderRepresentation::kMaterialized);
}

static void CatalogScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

BENCHMARK(BM_IndexMaterialize_SelectOneTemplate)
    ->Apply(CatalogScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_IndexMaterialize_SelectOneConstrainedTemplate)
    ->Apply(CatalogScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_IndexMaterialize_UnrelatedLibrarySymbols)
    ->Arg(0)
    ->Arg(4096)
    ->Arg(16384)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_IndexMaterialize_SelectTemplateFamilies)
    ->Arg(64)
    ->Arg(512)
    ->Arg(4096);
BENCHMARK(BM_IndexMaterialize_SelectMaterializedTemplates)->Arg(1)->Arg(4096);

}  // namespace

BENCHMARK_MAIN();
