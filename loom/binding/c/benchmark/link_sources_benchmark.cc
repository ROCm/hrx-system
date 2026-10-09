// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Measures source-name composition across independently indexed providers.
// Input construction and indexing are untimed. Each timed link selects every
// public function, including its file locations, into a fresh output module.

#include <cstdio>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "loom/binding/c/benchmark/util/benchmark_support.h"
#include "loomc/iree.h"

namespace loomc::bench {
namespace {

static std::string SourceName(uint32_t ordinal) {
  char name[32];
  std::snprintf(name, sizeof(name), "source_%08u.loom", ordinal);
  return name;
}

class LinkSourcesFixture {
 public:
  LinkSourcesFixture(uint32_t provider_count, uint32_t sources_per_provider,
                     uint32_t source_name_stride,
                     loomc_source_format_t format) {
    loomc_context_t* context = nullptr;
    IREE_CHECK_OK(to_iree_status(
        loomc_context_create(nullptr, loom_allocator(), &context)));
    context_.reset(context);
    loomc_workspace_t* workspace = nullptr;
    IREE_CHECK_OK(to_iree_status(
        loomc_workspace_create(nullptr, loom_allocator(), &workspace)));
    workspace_.reset(workspace);
    loomc_linker_t* linker = nullptr;
    IREE_CHECK_OK(to_iree_status(loomc_linker_create(
        context_.get(), nullptr, loom_allocator(), &linker)));
    linker_.reset(linker);
    loomc_link_index_builder_t* builder = nullptr;
    IREE_CHECK_OK(to_iree_status(loomc_link_index_builder_create(
        context_.get(), nullptr, loom_allocator(), &builder)));
    LinkIndexBuilderPtr builder_ptr(builder);
    for (uint32_t provider = 0; provider < provider_count; ++provider) {
      const std::string identifier = "provider_" + std::to_string(provider);
      std::string text = "func.def @unused_" + std::to_string(provider) +
                         "(%value: index) -> (index) {\n"
                         "  func.return %value : index\n}\n";
      for (uint32_t source = 0; source < sources_per_provider; ++source) {
        const std::string location =
            " loc(\"" + SourceName(provider * source_name_stride + source) +
            "\":1:1 to 1:4)";
        text += "func.def public @function_" + std::to_string(provider) + "_" +
                std::to_string(source) +
                "(%value: index) -> (index) {\n"
                "  func.return %value : index" +
                location + "\n}" + location + "\n";
      }
      loomc_source_options_t options = {
          .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
          .structure_size = sizeof(options),
          .format = LOOMC_SOURCE_FORMAT_TEXT,
          .identifier = loomc_make_cstring_view(identifier.c_str()),
          .contents = loomc_make_byte_span(text.data(), text.size()),
          .storage = LOOMC_SOURCE_STORAGE_COPY};
      loomc_source_t* source = nullptr;
      IREE_CHECK_OK(to_iree_status(
          loomc_source_create(&options, loom_allocator(), &source)));
      SourcePtr source_ptr(source);
      if (format == LOOMC_SOURCE_FORMAT_BYTECODE) {
        loomc_module_t* module = nullptr;
        loomc_result_t* result = nullptr;
        IREE_CHECK_OK(to_iree_status(loomc_module_deserialize_from_source(
            context_.get(), workspace_.get(), source_ptr.get(), nullptr,
            loom_allocator(), &module, &result)));
        ModulePtr module_ptr(module);
        ResultPtr result_ptr(result);
        IREE_CHECK_OK(RequireSucceededResult(result, "provider parsing"));
        source_ptr = Serialize(module);
      }
      const loomc_link_index_source_options_t index_options = {
          .provider_name = loomc_make_cstring_view(identifier.c_str()),
          .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
      };
      IREE_CHECK_OK(to_iree_status(loomc_link_index_builder_add_source(
          builder, source_ptr.get(), &index_options, nullptr)));
    }
    loomc_link_index_t* index = nullptr;
    loomc_result_t* result = nullptr;
    IREE_CHECK_OK(to_iree_status(
        loomc_link_index_builder_finish(builder, &index, &result)));
    index_.reset(index);
    ResultPtr result_ptr(result);
    IREE_CHECK_OK(RequireSucceededResult(result, "provider indexing"));
  }

  ModulePtr Link() const {
    loomc_link_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        .structure_size = sizeof(options),
        .link_index = index_.get(),
        .mode = LOOMC_LINK_MODE_LINK,
        .flags = LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS};
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    IREE_CHECK_OK(to_iree_status(loomc_link_module(
        linker_.get(), workspace_.get(), &options, &module, &result)));
    ResultPtr result_ptr(result);
    IREE_CHECK_OK(RequireSucceededResult(result, "source composition"));
    return ModulePtr(module);
  }

  SourcePtr Serialize(const loomc_module_t* module) const {
    loomc_module_serialize_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        .structure_size = sizeof(options),
        .format = LOOMC_SOURCE_FORMAT_BYTECODE};
    loomc_source_t* source = nullptr;
    IREE_CHECK_OK(to_iree_status(loomc_module_serialize_to_source(
        module, &options, loom_allocator(), &source)));
    return SourcePtr(source);
  }

  loomc_workspace_statistics_t Statistics() const {
    loomc_workspace_statistics_t statistics = {};
    loomc_workspace_query_statistics(workspace_.get(), &statistics);
    return statistics;
  }

 private:
  // Immutable registry shared by every provider and link invocation.
  ContextPtr context_;
  // Reusable invocation storage; output modules have independent ownership.
  WorkspacePtr workspace_;
  // Prepared linker reused across calls.
  LinkerPtr linker_;
  // Frozen provider catalog retaining its input sources.
  LinkIndexPtr index_;
};

static void BenchmarkLinkSources(benchmark::State& state,
                                 loomc_source_format_t format) {
  const uint32_t provider_count = state.range(0);
  const uint32_t sources_per_provider = state.range(1);
  const uint32_t source_name_stride = state.range(2);
  LinkSourcesFixture fixture(provider_count, sources_per_provider,
                             source_name_stride, format);
  {
    ModulePtr module = fixture.Link();
    loomc_host_size_t function_count = 0;
    loomc_result_t* result = nullptr;
    IREE_CHECK_OK(to_iree_status(
        loomc_module_query_functions(module.get(), nullptr, loom_allocator(), 0,
                                     nullptr, &function_count, &result)));
    ResultPtr result_ptr(result);
    IREE_CHECK_OK(RequireSucceededResult(result, "linked function query"));
    if (function_count != provider_count * sources_per_provider) {
      state.SkipWithError("link did not select exactly the public functions");
      return;
    }
    SourcePtr source = fixture.Serialize(module.get());
    state.counters["output_bytes"] =
        loomc_source_contents(source.get()).data_length;
  }
  const auto before = fixture.Statistics();
  for (auto _ : state) {
    ModulePtr module = fixture.Link();
    benchmark::DoNotOptimize(module.get());
    state.PauseTiming();
    module.reset();
    state.ResumeTiming();
  }
  const auto after = fixture.Statistics();
  state.counters["workspace_bytes"] = after.total_block_size;
  state.counters["new_block_allocations"] =
      after.block_system_allocation_count -
      before.block_system_allocation_count;
  state.counters["new_oversized_allocations"] =
      after.oversized_allocation_count - before.oversized_allocation_count;
  state.SetItemsProcessed(state.iterations() * provider_count *
                          sources_per_provider);
}

static void SourceScales(benchmark::Benchmark* benchmark) {
  for (int64_t providers : {1, 16, 256, 1024, 4096, 8192}) {
    benchmark->Args({providers, 1, 1});
    benchmark->Args({providers, 1, 0});
  }
  for (int64_t providers : {1, 16, 256, 1024}) {
    benchmark->Args({providers, 16, 16});
  }
}

BENCHMARK_CAPTURE(BenchmarkLinkSources, Bytecode, LOOMC_SOURCE_FORMAT_BYTECODE)
    ->Apply(SourceScales);
BENCHMARK_CAPTURE(BenchmarkLinkSources, Text, LOOMC_SOURCE_FORMAT_TEXT)
    ->Apply(SourceScales);

}  // namespace
}  // namespace loomc::bench
