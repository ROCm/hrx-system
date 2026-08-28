// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks warm-index template selection from large bytecode libraries.

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

class TemplateCatalogFixture {
 public:
  TemplateCatalogFixture(uint32_t provider_count,
                         TemplateProviderContractShape contract_shape)
      : provider_count_(provider_count), contract_shape_(contract_shape) {
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
        /*.provider_name=*/IREE_SV("root"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
    };
    CheckStatus(loom_link_module_index_add_materialized(
        index_, root_module_, &root_options,
        /*out_provider_ordinal=*/nullptr));
    const loom_link_module_index_add_options_t library_options = {
        /*.provider_name=*/IREE_SV("library"),
        /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
    };
    CheckStatus(loom_link_module_index_add_bytecode(
        index_,
        iree_make_const_byte_span(library_bytecode_.data(),
                                  library_bytecode_.size()),
        IREE_SV("library.loombc"), /*index_options=*/nullptr, &library_options,
        /*out_provider_ordinal=*/nullptr));

    const loom_link_module_index_module_t* library =
        loom_link_module_index_module_at(index_, 1);
    if (library == nullptr) {
      std::abort();
    }
    const std::string selected_name = ProviderName(provider_count_ - 1);
    const loom_link_module_index_symbol_t* selected =
        loom_link_module_index_lookup_private(
            index_, library,
            iree_make_string_view(selected_name.data(), selected_name.size()));
    if (selected == nullptr) {
      std::abort();
    }
    selected_provider_ordinal_ = selected->ordinal;
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
    loom_link_plan_options_t options = {};
    options.mode = LOOM_LINK_PLAN_LINK;
    options.root_symbols = {/*.count=*/1, /*.values=*/&root};
    options.unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ERROR;
    loom_link_plan_materialization_environment_t environment = {};
    environment.context = &context_;
    environment.block_pool = &block_pool_;
    environment.allocator = iree_allocator_system();
    loom_link_index_materialization_t materialization = {};
    CheckStatus(loom_link_index_materialize(
        index_, &options, &environment, IREE_SV("linked"), &materialization));
    if (!loom_link_plan_contains_symbol(materialization.plan,
                                        selected_provider_ordinal_)) {
      std::abort();
    }
    return materialization;
  }

  uint32_t provider_count() const { return provider_count_; }
  iree_host_size_t library_byte_count() const {
    return library_bytecode_.size();
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

  std::string BuildRootSource() const {
    if (contract_shape_ == TemplateProviderContractShape::kComplete) {
      return R"(
target.generic<reference> @request_target {subgroup_size = 64}

template.decl @benchmark.choose(%value: i32) -> (i32)

func.def public target(@request_target) @entry(%value: i32) -> (i32) where [range(%value, 32, 32)] {
  %result = template.apply<@benchmark.choose>(%value) : (i32) -> (i32)
  func.return %result : i32
}
)";
    }
    return R"(
template.decl @benchmark.choose(%value: i32) -> (i32)

func.def public @entry(%value: i32) -> (i32) {
  %result = template.apply<@benchmark.choose>(%value) : (i32) -> (i32)
  func.return %result : i32
}
)";
  }

  std::string BuildLibrarySource() const {
    std::string source =
        "template.decl @benchmark.choose(%value: i32) -> (i32)\n\n";
    if (contract_shape_ == TemplateProviderContractShape::kComplete) {
      source.insert(0, "target.generic<reference> @provider_target\n\n");
    }
    source.reserve(source.size() + provider_count_ * 260u);
    for (uint32_t i = 0; i < provider_count_; ++i) {
      const std::string name = ProviderName(i);
      source.append("template.def<@benchmark.choose> ");
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
            "(%value: i32) -> (i32) where [mul(%value, 16)] {\n"
            "  template.return %value : i32\n"
            "}\n\n");
        continue;
      }
      source.append(
          "(%value: i32) -> (i32) {\n"
          "  template.return %value : i32\n"
          "}\n\n");
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

  uint32_t provider_count_;
  TemplateProviderContractShape contract_shape_;
  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_module_t* root_module_ = nullptr;
  loom_module_t* library_module_ = nullptr;
  std::vector<uint8_t> library_bytecode_;
  loom_link_module_index_t* index_ = nullptr;
  iree_host_size_t selected_provider_ordinal_ =
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
};

static void RunSelectOneTemplateBenchmark(
    benchmark::State& state, TemplateProviderContractShape contract_shape) {
  TemplateCatalogFixture fixture(static_cast<uint32_t>(state.range(0)),
                                 contract_shape);
  iree_host_size_t selected_symbol_count = 0;
  for (auto _ : state) {
    loom_link_index_materialization_t materialization = fixture.Materialize();
    selected_symbol_count = loom_link_plan_symbol_count(materialization.plan);
    benchmark::DoNotOptimize(materialization.product.module);

    state.PauseTiming();
    loom_link_index_materialization_deinitialize(&materialization);
    state.ResumeTiming();
  }
  state.counters["library_bytecode_bytes"] =
      static_cast<double>(fixture.library_byte_count());
  state.counters["providers"] = static_cast<double>(fixture.provider_count());
  state.counters["selected_provider_bodies"] = 1.0;
  state.counters["selected_symbols"] =
      static_cast<double>(selected_symbol_count);
  state.SetItemsProcessed(state.iterations() * fixture.provider_count());
  state.SetComplexityN(fixture.provider_count());
}

static void BM_IndexMaterialize_SelectOneTemplate(benchmark::State& state) {
  RunSelectOneTemplateBenchmark(state,
                                TemplateProviderContractShape::kPriorityOnly);
}

static void BM_IndexMaterialize_SelectOneConstrainedTemplate(
    benchmark::State& state) {
  RunSelectOneTemplateBenchmark(state,
                                TemplateProviderContractShape::kComplete);
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

}  // namespace

BENCHMARK_MAIN();
