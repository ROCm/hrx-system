// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks repeated selective links through the public LoomC API. The
// reusable linker and frozen index own catalog-wide work while one worker-local
// workspace supplies invocation storage and each link materializes one root
// closure.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "loom/binding/c/benchmark/util/benchmark_support.h"
#include "loomc/iree.h"

namespace loomc::bench {
namespace {

static bool SkipOnError(benchmark::State& state, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return false;
  }
  const std::string message = FormatStatus(status);
  iree_status_free(status);
  state.SkipWithError(message.c_str());
  return true;
}

static std::string RootName(uint32_t ordinal) {
  char buffer[32] = {0};
  const int length =
      std::snprintf(buffer, sizeof(buffer), "@kernel_%08u", ordinal);
  if (length <= 0 || length >= (int)sizeof(buffer)) {
    std::abort();
  }
  return std::string(buffer, (size_t)length);
}

static void AppendFormatted(std::string& target, const char* format,
                            uint32_t value) {
  char buffer[128] = {0};
  const int length = std::snprintf(buffer, sizeof(buffer), format, value);
  if (length <= 0 || length >= (int)sizeof(buffer)) {
    std::abort();
  }
  target.append(buffer, (size_t)length);
}

static std::string BuildCatalogSource(uint32_t root_count,
                                      uint32_t values_per_root,
                                      std::vector<std::string>* root_names) {
  constexpr uint32_t kOperationsPerSourceLine = 128;
  root_names->clear();
  root_names->reserve(root_count);

  std::string source;
  source.reserve((size_t)root_count * (size_t)values_per_root * 64u);
  source.append(
      "func.def @shared_identity(%value: index) -> (index) {\n"
      "  func.return %value : index\n"
      "}\n\n");
  for (uint32_t root_ordinal = 0; root_ordinal < root_count; ++root_ordinal) {
    root_names->push_back(RootName(root_ordinal));
    source.append("func.def public ");
    source.append(root_names->back());
    source.append("(%arg: index) -> (index) {\n");
    source.append("  %one = index.constant 1 : index\n");
    // Newlines are lexer whitespace. Packing several operations onto each
    // physical line keeps large synthetic modules within the source-location
    // line encoding without changing their token stream or IR shape.
    for (uint32_t value_ordinal = 0; value_ordinal < values_per_root;
         ++value_ordinal) {
      if (value_ordinal % kOperationsPerSourceLine == 0) {
        source.append("  ");
      }
      AppendFormatted(source, "%%value_%u = index.add ", value_ordinal);
      if (value_ordinal == 0) {
        source.append("%arg");
      } else {
        AppendFormatted(source, "%%value_%u", value_ordinal - 1);
      }
      source.append(", %one : index");
      source.append(value_ordinal % kOperationsPerSourceLine ==
                                kOperationsPerSourceLine - 1 ||
                            value_ordinal == values_per_root - 1
                        ? "\n"
                        : " ");
    }
    source.append("  %result = func.call @shared_identity(");
    if (values_per_root == 0) {
      source.append("%arg");
    } else {
      AppendFormatted(source, "%%value_%u", values_per_root - 1);
    }
    source.append(") : (index) -> (index)\n");
    source.append("  func.return %result : index\n}\n\n");
  }
  return source;
}

class LinkCatalogFixture {
 public:
  LinkCatalogFixture(uint32_t root_count, uint32_t values_per_root,
                     loomc_source_format_t source_format)
      : root_count_(root_count),
        values_per_root_(values_per_root),
        source_format_(source_format) {}

  iree_status_t Initialize() {
    loomc_context_t* context = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_context_create(
        /*options=*/nullptr, loom_allocator(), &context)));
    context_.reset(context);

    std::string source_text =
        BuildCatalogSource(root_count_, values_per_root_, &root_names_);
    const loomc_source_options_t source_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /*.structure_size=*/sizeof(source_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
        /*.identifier=*/loomc_make_cstring_view("link_catalog.loom"),
        /*.contents=*/
        loomc_make_byte_span(source_text.data(), source_text.size()),
        /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
        /*.release=*/nullptr,
        /*.release_user_data=*/nullptr,
    };
    loomc_source_t* source = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(
        loomc_source_create(&source_options, loom_allocator(), &source)));
    SourcePtr text_source(source);

    if (source_format_ == LOOMC_SOURCE_FORMAT_TEXT) {
      source_.reset(text_source.release());
    } else if (source_format_ == LOOMC_SOURCE_FORMAT_BYTECODE) {
      loomc_workspace_t* workspace = nullptr;
      IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
          /*options=*/nullptr, loom_allocator(), &workspace)));
      WorkspacePtr workspace_ptr(workspace);

      loomc_module_t* module = nullptr;
      loomc_result_t* result = nullptr;
      iree_status_t status =
          to_iree_status(loomc_module_deserialize_from_source(
              context_.get(), workspace_ptr.get(), text_source.get(),
              /*options=*/nullptr, loom_allocator(), &module, &result));
      ModulePtr module_ptr(module);
      ResultPtr result_ptr(result);
      IREE_RETURN_IF_ERROR(status);
      IREE_RETURN_IF_ERROR(
          RequireSucceededResult(result_ptr.get(), "catalog parsing"));

      const loomc_module_serialize_options_t serialize_options = {
          /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
          /*.structure_size=*/sizeof(serialize_options),
          /*.next=*/nullptr,
          /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
          /*.identifier=*/loomc_make_cstring_view("link_catalog.loombc"),
          /*.text_presentation=*/LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT,
      };
      loomc_source_t* bytecode_source = nullptr;
      IREE_RETURN_IF_ERROR(to_iree_status(loomc_module_serialize_to_source(
          module_ptr.get(), &serialize_options, loom_allocator(),
          &bytecode_source)));
      source_.reset(bytecode_source);
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "catalog source format must be text or bytecode");
    }
    source_size_ = loomc_source_contents(source_.get()).data_length;

    loomc_linker_t* linker = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_linker_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &linker)));
    linker_.reset(linker);
    return iree_ok_status();
  }

  iree_status_t BuildIndex(LinkIndexPtr* out_index) const {
    out_index->reset();
    loomc_link_index_builder_t* builder = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &builder)));
    LinkIndexBuilderPtr builder_ptr(builder);

    const loomc_link_index_source_options_t source_options = {
        /*.provider_name=*/loomc_make_cstring_view("catalog"),
        /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_add_source(
        builder_ptr.get(), source_.get(), &source_options,
        /*out_slot=*/nullptr)));

    loomc_link_index_t* index = nullptr;
    loomc_result_t* result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_link_index_builder_finish(builder_ptr.get(), &index, &result));
    LinkIndexPtr index_ptr(index);
    ResultPtr result_ptr(result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result_ptr.get(), "link index construction"));
    const loomc_host_size_t expected_symbol_count = root_count_ + 1;
    if (loomc_link_index_symbol_count(index_ptr.get()) !=
        expected_symbol_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "link index has %" PRIhsz
                              " symbols; expected %" PRIhsz,
                              loomc_link_index_symbol_count(index_ptr.get()),
                              expected_symbol_count);
    }
    out_index->reset(index_ptr.release());
    return iree_ok_status();
  }

  iree_status_t CreateWorkspace(WorkspacePtr* out_workspace) const {
    out_workspace->reset();
    loomc_workspace_t* workspace = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
        /*options=*/nullptr, loom_allocator(), &workspace)));
    out_workspace->reset(workspace);
    return iree_ok_status();
  }

  iree_status_t Link(loomc_link_index_t* index, loomc_workspace_t* workspace,
                     uint32_t root_ordinal, ModulePtr* out_module,
                     ResultPtr* out_result) const {
    out_module->reset();
    out_result->reset();
    const std::string& root = root_names_[root_ordinal];
    const loomc_string_view_t root_symbol =
        loomc_make_string_view(root.data(), root.size());
    const loomc_link_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.link_index=*/index,
        /*.module_name=*/loomc_make_cstring_view("selected_kernel"),
        /*.root_symbols=*/&root_symbol,
        /*.root_symbol_count=*/1,
        /*.flags=*/0,
        /*.config=*/{},
    };
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    iree_status_t status = to_iree_status(loomc_link_module(
        linker_.get(), workspace, &options, &module, &result));
    ModulePtr module_ptr(module);
    ResultPtr result_ptr(result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result_ptr.get(), "selective linking"));
    if (!module_ptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "selective linking produced no module");
    }
    out_module->reset(module_ptr.release());
    out_result->reset(result_ptr.release());
    return iree_ok_status();
  }

  iree_status_t VerifySelectedModule(const loomc_module_t* module,
                                     uint32_t root_ordinal) const {
    loomc_host_size_t function_count = 0;
    loomc_result_t* result = nullptr;
    iree_status_t status = to_iree_status(loomc_module_query_functions(
        module, /*options=*/nullptr, loom_allocator(), /*function_capacity=*/0,
        /*out_functions=*/nullptr, &function_count, &result));
    ResultPtr result_ptr(result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result_ptr.get(), "linked module query"));
    if (function_count != 2) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "selective link retained %" PRIhsz
                              " functions; expected one root and one helper",
                              function_count);
    }

    loomc_module_function_t function = {};
    const std::string& root = root_names_[root_ordinal];
    if (!loomc_module_try_lookup_function(
            module, loomc_make_string_view(root.data(), root.size()),
            &function) ||
        !loomc_module_try_lookup_function(
            module, loomc_make_cstring_view("@shared_identity"), &function)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "selective link did not retain its root and shared helper");
    }
    return iree_ok_status();
  }

  uint32_t root_count() const { return root_count_; }

  uint32_t values_per_root() const { return values_per_root_; }

  size_t source_size() const { return source_size_; }

  loomc_source_format_t source_format() const { return source_format_; }

  uint64_t catalog_value_count() const {
    return (uint64_t)root_count_ * ((uint64_t)values_per_root_ + 3u) + 1u;
  }

  uint64_t catalog_operation_count() const {
    return catalog_value_count() + (uint64_t)root_count_ + 1u;
  }

  uint64_t selected_body_operation_count() const {
    return (uint64_t)values_per_root_ + 4u;
  }

  uint64_t selected_operation_count() const {
    return selected_body_operation_count() + 2u;
  }

 private:
  uint32_t root_count_ = 0;
  uint32_t values_per_root_ = 0;
  loomc_source_format_t source_format_ = LOOMC_SOURCE_FORMAT_UNKNOWN;
  size_t source_size_ = 0;
  ContextPtr context_;
  SourcePtr source_;
  LinkerPtr linker_;
  std::vector<std::string> root_names_;
};

static void SetCatalogCounters(benchmark::State& state,
                               const LinkCatalogFixture& fixture) {
  state.counters["catalog_roots"] = (double)fixture.root_count();
  state.counters["catalog_symbols"] = (double)fixture.root_count() + 1.0;
  state.counters["catalog_body_ops"] = (double)fixture.catalog_value_count();
  state.counters["catalog_ir_ops"] = (double)fixture.catalog_operation_count();
  state.counters["catalog_ssa_values"] = (double)fixture.catalog_value_count();
  state.counters["input_bytecode"] =
      fixture.source_format() == LOOMC_SOURCE_FORMAT_BYTECODE ? 1.0 : 0.0;
  state.counters["source_bytes"] = (double)fixture.source_size();
  state.counters["source_bytes/root"] =
      (double)fixture.source_size() / (double)fixture.root_count();
  state.counters["values/root"] = (double)fixture.values_per_root();
}

static void SetSelectedClosureCounters(benchmark::State& state,
                                       const LinkCatalogFixture& fixture) {
  state.counters["selected_body_ops"] =
      (double)fixture.selected_body_operation_count();
  state.counters["selected_functions"] = 2.0;
  state.counters["selected_ir_ops"] =
      (double)fixture.selected_operation_count();
}

static void SetWorkspaceCounters(benchmark::State& state,
                                 const loomc_workspace_statistics_t& before,
                                 const loomc_workspace_statistics_t& after) {
  state.counters["workspace_block_size"] = (double)after.total_block_size;
  state.counters["workspace_block_system_allocations"] =
      (double)after.block_system_allocation_count;
  state.counters["workspace_block_system_bytes"] =
      (double)after.block_system_allocation_bytes;
  state.counters["workspace_new_block_system_allocations"] =
      (double)(after.block_system_allocation_count -
               before.block_system_allocation_count);
  state.counters["workspace_new_block_system_bytes"] =
      (double)(after.block_system_allocation_bytes -
               before.block_system_allocation_bytes);
  state.counters["workspace_oversized_allocations"] =
      (double)after.oversized_allocation_count;
  state.counters["workspace_new_oversized_allocations"] =
      (double)(after.oversized_allocation_count -
               before.oversized_allocation_count);
}

static iree_status_t LinkBatch(const LinkCatalogFixture& fixture,
                               loomc_link_index_t* index,
                               loomc_workspace_t* workspace,
                               std::vector<ModulePtr>* modules,
                               std::vector<ResultPtr>* results) {
  for (size_t i = 0; i < modules->size(); ++i) {
    const uint32_t root_ordinal = (uint32_t)(i % fixture.root_count());
    IREE_RETURN_IF_ERROR(fixture.Link(index, workspace, root_ordinal,
                                      &(*modules)[i], &(*results)[i]));
    benchmark::DoNotOptimize((*modules)[i].get());
  }
  return iree_ok_status();
}

static void ReleaseBatch(std::vector<ModulePtr>* modules,
                         std::vector<ResultPtr>* results) {
  for (ResultPtr& result : *results) {
    result.reset();
  }
  for (ModulePtr& module : *modules) {
    module.reset();
  }
}

static void BM_LinkIndexBuildForFormat(benchmark::State& state,
                                       loomc_source_format_t source_format) {
  LinkCatalogFixture fixture((uint32_t)state.range(0), (uint32_t)state.range(1),
                             source_format);
  if (SkipOnError(state, fixture.Initialize())) {
    return;
  }

  int64_t index_count = 0;
  for (auto _ : state) {
    LinkIndexPtr index;
    if (SkipOnError(state, fixture.BuildIndex(&index))) {
      break;
    }
    benchmark::DoNotOptimize(index.get());
    ++index_count;
    state.PauseTiming();
    index.reset();
    state.ResumeTiming();
  }
  SetCatalogCounters(state, fixture);
  state.SetItemsProcessed(index_count * (fixture.root_count() + 1));
  state.SetBytesProcessed(index_count * (int64_t)fixture.source_size());
  state.counters["indexes/s"] =
      benchmark::Counter(index_count, benchmark::Counter::kIsRate);
}

static void BM_LinkIndexBuild(benchmark::State& state) {
  BM_LinkIndexBuildForFormat(state, LOOMC_SOURCE_FORMAT_TEXT);
}

static void BM_LinkIndexBuildBytecode(benchmark::State& state) {
  BM_LinkIndexBuildForFormat(state, LOOMC_SOURCE_FORMAT_BYTECODE);
}

static void BM_SelectiveLinkBatchForFormat(
    benchmark::State& state, loomc_source_format_t source_format) {
  LinkCatalogFixture fixture((uint32_t)state.range(0), (uint32_t)state.range(1),
                             source_format);
  if (SkipOnError(state, fixture.Initialize())) {
    return;
  }
  LinkIndexPtr index;
  if (SkipOnError(state, fixture.BuildIndex(&index))) {
    return;
  }
  WorkspacePtr workspace;
  if (SkipOnError(state, fixture.CreateWorkspace(&workspace))) {
    return;
  }

  const uint32_t links_per_batch = (uint32_t)state.range(2);
  std::vector<ModulePtr> modules(links_per_batch);
  std::vector<ResultPtr> results(links_per_batch);
  if (SkipOnError(state, LinkBatch(fixture, index.get(), workspace.get(),
                                   &modules, &results))) {
    return;
  }
  if (SkipOnError(state, fixture.VerifySelectedModule(modules[0].get(),
                                                      /*root_ordinal=*/0))) {
    return;
  }
  ReleaseBatch(&modules, &results);

  loomc_workspace_statistics_t before = {};
  loomc_workspace_query_statistics(workspace.get(), &before);
  int64_t total_links = 0;
  for (auto _ : state) {
    if (SkipOnError(state, LinkBatch(fixture, index.get(), workspace.get(),
                                     &modules, &results))) {
      break;
    }
    total_links += links_per_batch;
    state.PauseTiming();
    ReleaseBatch(&modules, &results);
    state.ResumeTiming();
  }
  loomc_workspace_statistics_t after = {};
  loomc_workspace_query_statistics(workspace.get(), &after);

  SetCatalogCounters(state, fixture);
  SetWorkspaceCounters(state, before, after);
  SetSelectedClosureCounters(state, fixture);
  state.SetItemsProcessed(total_links);
  state.counters["links/batch"] = (double)links_per_batch;
  state.counters["links/s"] =
      benchmark::Counter(total_links, benchmark::Counter::kIsRate);
}

static void BM_SelectiveLinkBatch(benchmark::State& state) {
  BM_SelectiveLinkBatchForFormat(state, LOOMC_SOURCE_FORMAT_TEXT);
}

static void BM_SelectiveLinkBatchBytecode(benchmark::State& state) {
  BM_SelectiveLinkBatchForFormat(state, LOOMC_SOURCE_FORMAT_BYTECODE);
}

static void BM_IndexOnceAndSelectiveLinkBatchForFormat(
    benchmark::State& state, loomc_source_format_t source_format) {
  LinkCatalogFixture fixture((uint32_t)state.range(0), (uint32_t)state.range(1),
                             source_format);
  if (SkipOnError(state, fixture.Initialize())) {
    return;
  }
  WorkspacePtr workspace;
  if (SkipOnError(state, fixture.CreateWorkspace(&workspace))) {
    return;
  }

  const uint32_t links_per_batch = (uint32_t)state.range(2);
  std::vector<ModulePtr> modules(links_per_batch);
  std::vector<ResultPtr> results(links_per_batch);
  loomc_workspace_statistics_t before = {};
  loomc_workspace_query_statistics(workspace.get(), &before);
  int64_t total_links = 0;
  for (auto _ : state) {
    LinkIndexPtr index;
    if (SkipOnError(state, fixture.BuildIndex(&index)) ||
        SkipOnError(state, LinkBatch(fixture, index.get(), workspace.get(),
                                     &modules, &results))) {
      break;
    }
    total_links += links_per_batch;
    state.PauseTiming();
    ReleaseBatch(&modules, &results);
    index.reset();
    state.ResumeTiming();
  }
  loomc_workspace_statistics_t after = {};
  loomc_workspace_query_statistics(workspace.get(), &after);

  SetCatalogCounters(state, fixture);
  SetWorkspaceCounters(state, before, after);
  SetSelectedClosureCounters(state, fixture);
  state.SetItemsProcessed(total_links);
  state.counters["links/batch"] = (double)links_per_batch;
  state.counters["links/s"] =
      benchmark::Counter(total_links, benchmark::Counter::kIsRate);
}

static void BM_IndexOnceAndSelectiveLinkBatch(benchmark::State& state) {
  BM_IndexOnceAndSelectiveLinkBatchForFormat(state, LOOMC_SOURCE_FORMAT_TEXT);
}

static void BM_IndexOnceAndSelectiveLinkBatchBytecode(benchmark::State& state) {
  BM_IndexOnceAndSelectiveLinkBatchForFormat(state,
                                             LOOMC_SOURCE_FORMAT_BYTECODE);
}

static void CatalogScales(benchmark::Benchmark* benchmark) {
  for (int64_t root_count : {1, 16, 64, 512}) {
    benchmark->Args({root_count, 256});
  }
  benchmark->Args({4096, 512});
}

static void SelectiveLinkScales(benchmark::Benchmark* benchmark) {
  for (int64_t root_count : {1, 16, 64, 512}) {
    for (int64_t links_per_batch : {1, 16, 64, 512}) {
      benchmark->Args({root_count, 256, links_per_batch});
    }
  }
  benchmark->Args({4096, 512, 80});
  benchmark->Args({4096, 512, 388});
}

static void BM_LinkIndexBuild_Smoke(benchmark::State& state) {
  BM_LinkIndexBuild(state);
}

static void BM_LinkIndexBuildBytecode_Smoke(benchmark::State& state) {
  BM_LinkIndexBuildBytecode(state);
}

static void BM_SelectiveLinkBatch_Smoke(benchmark::State& state) {
  BM_SelectiveLinkBatch(state);
}

static void BM_SelectiveLinkBatchBytecode_Smoke(benchmark::State& state) {
  BM_SelectiveLinkBatchBytecode(state);
}

BENCHMARK(BM_LinkIndexBuild)
    ->Apply(CatalogScales)
    ->ArgNames({"catalog_roots", "values_per_root"});
BENCHMARK(BM_LinkIndexBuildBytecode)
    ->Apply(CatalogScales)
    ->ArgNames({"catalog_roots", "values_per_root"});
BENCHMARK(BM_SelectiveLinkBatch)
    ->Apply(SelectiveLinkScales)
    ->ArgNames({"catalog_roots", "values_per_root", "links_per_batch"});
BENCHMARK(BM_SelectiveLinkBatchBytecode)
    ->Apply(SelectiveLinkScales)
    ->ArgNames({"catalog_roots", "values_per_root", "links_per_batch"});
BENCHMARK(BM_IndexOnceAndSelectiveLinkBatch)
    ->Args({512, 256, 388})
    ->Args({4096, 512, 80})
    ->Args({4096, 512, 388})
    ->ArgNames({"catalog_roots", "values_per_root", "links_per_batch"});
BENCHMARK(BM_IndexOnceAndSelectiveLinkBatchBytecode)
    ->Args({512, 256, 388})
    ->Args({4096, 512, 80})
    ->Args({4096, 512, 388})
    ->ArgNames({"catalog_roots", "values_per_root", "links_per_batch"});

BENCHMARK(BM_LinkIndexBuild_Smoke)
    ->Args({16, 16})
    ->ArgNames({"catalog_roots", "values_per_root"});
BENCHMARK(BM_LinkIndexBuildBytecode_Smoke)
    ->Args({16, 16})
    ->ArgNames({"catalog_roots", "values_per_root"});
BENCHMARK(BM_SelectiveLinkBatch_Smoke)
    ->Args({16, 16, 4})
    ->ArgNames({"catalog_roots", "values_per_root", "links_per_batch"});
BENCHMARK(BM_SelectiveLinkBatchBytecode_Smoke)
    ->Args({16, 16, 4})
    ->ArgNames({"catalog_roots", "values_per_root", "links_per_batch"});

}  // namespace
}  // namespace loomc::bench
