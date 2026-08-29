// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks request sealing through the public LoomC API. A frozen library
// index, immutable bytecode request, prepared linker, and warm worker-local
// workspace model one process-local JIT worker. Timed regions include request
// indexing, exact-root planning, selective materialization, serialization, and
// allocation of the independently owned output request and result.
// `//loom/src/loom/link:planner_benchmark` isolates overlay, planning, and
// materialization costs; `//loom/src/loom/format/bytecode:writer_benchmark`
// isolates bytecode serialization.

#include <cstdint>
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

using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;

static bool SkipOnError(benchmark::State& state, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return false;
  }
  const std::string message = FormatStatus(status);
  iree_status_free(status);
  state.SkipWithError(message.c_str());
  return true;
}

static std::string SymbolName(const char* prefix, uint32_t ordinal) {
  char buffer[48] = {0};
  const int length =
      std::snprintf(buffer, sizeof(buffer), "%s_%08u", prefix, ordinal);
  if (length <= 0 || length >= static_cast<int>(sizeof(buffer))) {
    std::abort();
  }
  return std::string(buffer, static_cast<size_t>(length));
}

static std::string BuildLibrarySource(uint32_t symbol_count) {
  std::string source = R"(
func.def public @library_identity(%value: index) -> (index) {
  func.return %value : index
}

)";
  source.reserve(source.size() + static_cast<size_t>(symbol_count) * 128u);
  for (uint32_t i = 1; i < symbol_count; ++i) {
    source.append("func.def public @");
    source.append(SymbolName("unused_library", i));
    source.append(
        "(%value: index) -> (index) {\n"
        "  func.return %value : index\n"
        "}\n\n");
  }
  return source;
}

static std::string BuildRequestSource(uint32_t root_count,
                                      uint32_t values_per_root,
                                      std::vector<std::string>* root_names) {
  root_names->clear();
  root_names->reserve(root_count);

  std::string source =
      "func.decl @library_identity(%value: index) -> (index)\n\n";
  source.reserve(source.size() + static_cast<size_t>(root_count) *
                                     (256u + values_per_root * 72u));
  for (uint32_t root_ordinal = 0; root_ordinal < root_count; ++root_ordinal) {
    root_names->push_back(SymbolName("request_root", root_ordinal));
    source.append("func.def public @");
    source.append(root_names->back());
    source.append("(%arg: index) -> (index) {\n");
    if (values_per_root != 0) {
      source.append("  %one = index.constant 1 : index\n");
    }
    for (uint32_t value_ordinal = 0; value_ordinal < values_per_root;
         ++value_ordinal) {
      source.append("  %value_");
      source.append(std::to_string(value_ordinal));
      source.append(" = index.add ");
      if (value_ordinal == 0) {
        source.append("%arg");
      } else {
        source.append("%value_");
        source.append(std::to_string(value_ordinal - 1));
      }
      source.append(", %one : index\n");
    }
    source.append("  %result = func.call @library_identity(");
    if (values_per_root == 0) {
      source.append("%arg");
    } else {
      source.append("%value_");
      source.append(std::to_string(values_per_root - 1));
    }
    source.append(
        ") : (index) -> (index)\n"
        "  func.return %result : index\n"
        "}\n\n");
  }
  return source;
}

class LinkRequestFixture {
 public:
  LinkRequestFixture(uint32_t library_symbol_count, uint32_t root_count,
                     uint32_t values_per_root)
      : library_symbol_count_(library_symbol_count),
        root_count_(root_count),
        values_per_root_(values_per_root) {}

  iree_status_t Initialize() {
    if (library_symbol_count_ == 0 || root_count_ == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "benchmark dimensions must be nonzero");
    }

    loomc_context_t* context = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_context_create(
        /*options=*/nullptr, loom_allocator(), &context)));
    context_.reset(context);

    loomc_workspace_t* setup_workspace = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
        /*options=*/nullptr, loom_allocator(), &setup_workspace)));
    WorkspacePtr setup_workspace_ptr(setup_workspace);

    const std::string library_text = BuildLibrarySource(library_symbol_count_);
    SourcePtr library_text_source;
    IREE_RETURN_IF_ERROR(CreateTextSource("request_library.loom", library_text,
                                          &library_text_source));
    ModulePtr library_module;
    IREE_RETURN_IF_ERROR(ParseSource(
        setup_workspace_ptr.get(), library_text_source.get(), &library_module));
    SourcePtr library_bytecode_source;
    IREE_RETURN_IF_ERROR(SerializeModule(library_module.get(),
                                         "request_library.loombc",
                                         &library_bytecode_source));
    library_source_size_ =
        loomc_source_contents(library_bytecode_source.get()).data_length;
    IREE_RETURN_IF_ERROR(BuildLibraryIndex(library_bytecode_source.get()));

    std::vector<std::string> root_names;
    const std::string request_text =
        BuildRequestSource(root_count_, values_per_root_, &root_names);
    SourcePtr request_text_source;
    IREE_RETURN_IF_ERROR(CreateTextSource("input_request.loom", request_text,
                                          &request_text_source));
    ModulePtr request_module;
    IREE_RETURN_IF_ERROR(ParseSource(
        setup_workspace_ptr.get(), request_text_source.get(), &request_module));
    SourcePtr request_bytecode_source;
    IREE_RETURN_IF_ERROR(SerializeModule(request_module.get(),
                                         "input_request.loombc",
                                         &request_bytecode_source));
    IREE_RETURN_IF_ERROR(BuildRequest(request_bytecode_source.get(), root_names,
                                      &input_request_));
    request_source_size_ =
        loomc_source_contents(loomc_request_source(input_request_.get()))
            .data_length;

    loomc_linker_t* linker = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_linker_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &linker)));
    linker_.reset(linker);

    loomc_workspace_t* workspace = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
        /*options=*/nullptr, loom_allocator(), &workspace)));
    workspace_.reset(workspace);
    return iree_ok_status();
  }

  iree_status_t Link(RequestPtr* out_request, ResultPtr* out_result) const {
    out_request->reset();
    out_result->reset();
    const loomc_link_request_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_REQUEST_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.library_index=*/library_index_.get(),
        /*.module_name=*/loomc_make_cstring_view("sealed-request"),
    };
    loomc_request_t* request = nullptr;
    loomc_result_t* result = nullptr;
    iree_status_t status = to_iree_status(loomc_link_request(
        linker_.get(), workspace_.get(), input_request_.get(), &options,
        loom_allocator(), &request, &result));
    RequestPtr request_ptr(request);
    ResultPtr result_ptr(result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result_ptr.get(), "request sealing"));
    if (request_ptr == nullptr ||
        loomc_request_root_count(request_ptr.get()) != root_count_) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "sealed request has an unexpected root count");
    }
    out_request->reset(request_ptr.release());
    out_result->reset(result_ptr.release());
    return iree_ok_status();
  }

  uint32_t library_symbol_count() const { return library_symbol_count_; }
  uint32_t root_count() const { return root_count_; }
  uint32_t values_per_root() const { return values_per_root_; }
  size_t library_source_size() const { return library_source_size_; }
  size_t request_source_size() const { return request_source_size_; }
  loomc_workspace_t* workspace() const { return workspace_.get(); }

 private:
  iree_status_t CreateTextSource(const char* identifier,
                                 const std::string& contents,
                                 SourcePtr* out_source) const {
    const loomc_source_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
        /*.identifier=*/loomc_make_cstring_view(identifier),
        /*.contents=*/loomc_make_byte_span(contents.data(), contents.size()),
        /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(
        loomc_source_create(&options, loom_allocator(), &source)));
    out_source->reset(source);
    return iree_ok_status();
  }

  iree_status_t ParseSource(loomc_workspace_t* workspace,
                            const loomc_source_t* source,
                            ModulePtr* out_module) const {
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    iree_status_t status = to_iree_status(loomc_module_deserialize_from_source(
        context_.get(), workspace, source,
        /*options=*/nullptr, loom_allocator(), &module, &result));
    ModulePtr module_ptr(module);
    ResultPtr result_ptr(result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result_ptr.get(), "benchmark source parsing"));
    out_module->reset(module_ptr.release());
    return iree_ok_status();
  }

  iree_status_t SerializeModule(const loomc_module_t* module,
                                const char* identifier,
                                SourcePtr* out_source) const {
    const loomc_module_serialize_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
        /*.identifier=*/loomc_make_cstring_view(identifier),
        /*.text_presentation=*/LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT,
    };
    loomc_source_t* source = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_module_serialize_to_source(
        module, &options, loom_allocator(), &source)));
    out_source->reset(source);
    return iree_ok_status();
  }

  iree_status_t BuildLibraryIndex(loomc_source_t* source) {
    loomc_link_index_builder_t* builder = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &builder)));
    LinkIndexBuilderPtr builder_ptr(builder);
    const loomc_link_index_source_options_t source_options = {
        /*.provider_name=*/loomc_make_cstring_view("request-library"),
        /*.role=*/LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
    };
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_add_source(
        builder_ptr.get(), source, &source_options, /*out_slot=*/nullptr)));

    loomc_link_index_t* index = nullptr;
    loomc_result_t* result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_link_index_builder_finish(builder_ptr.get(), &index, &result));
    LinkIndexPtr index_ptr(index);
    ResultPtr result_ptr(result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result_ptr.get(), "library index construction"));
    library_index_.reset(index_ptr.release());
    return iree_ok_status();
  }

  iree_status_t BuildRequest(loomc_source_t* source,
                             const std::vector<std::string>& root_names,
                             RequestPtr* out_request) const {
    loomc_link_index_builder_t* builder = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &builder)));
    LinkIndexBuilderPtr builder_ptr(builder);
    const loomc_link_index_source_options_t source_options = {
        /*.provider_name=*/loomc_make_cstring_view("input-request"),
        /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_add_source(
        builder_ptr.get(), source, &source_options, /*out_slot=*/nullptr)));

    loomc_link_index_t* index = nullptr;
    loomc_result_t* result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_link_index_builder_finish(builder_ptr.get(), &index, &result));
    LinkIndexPtr index_ptr(index);
    ResultPtr result_ptr(result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result_ptr.get(), "request index construction"));

    std::vector<loomc_request_root_t> roots;
    roots.reserve(root_names.size());
    for (const std::string& root_name : root_names) {
      loomc_link_index_symbol_t symbol = {};
      if (!loomc_link_index_lookup_global(
              index_ptr.get(),
              loomc_make_string_view(root_name.data(), root_name.size()),
              &symbol)) {
        return iree_make_status(IREE_STATUS_NOT_FOUND,
                                "request root is absent from source index");
      }
      roots.push_back({
          /*.module_ordinal=*/
          static_cast<uint32_t>(symbol.provider_module_ordinal),
          /*.symbol_ordinal=*/
          static_cast<uint32_t>(symbol.module_symbol_ordinal),
      });
    }

    loomc_request_t* request = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(
        loomc_request_create(loomc_compiled_module_product_descriptor(), source,
                             roots.data(), roots.size(), /*bindings=*/nullptr,
                             /*binding_count=*/0, loom_allocator(), &request)));
    out_request->reset(request);
    return iree_ok_status();
  }

  // Number of public symbols in the frozen library universe.
  uint32_t library_symbol_count_;
  // Number of exact request roots selected by every invocation.
  uint32_t root_count_;
  // Number of arithmetic values in each selected request root.
  uint32_t values_per_root_;
  // Serialized frozen-library byte size.
  size_t library_source_size_ = 0;
  // Serialized input-request byte size.
  size_t request_source_size_ = 0;
  // Shared representation and dialect context.
  ContextPtr context_;
  // Frozen library-only provider index reused by every invocation.
  LinkIndexPtr library_index_;
  // Prepared immutable linker reused by every invocation.
  LinkerPtr linker_;
  // Warm worker-local scratch workspace reused by every invocation.
  WorkspacePtr workspace_;
  // Immutable input bytecode request reused by every invocation.
  RequestPtr input_request_;
};

static void SetWorkspaceCounters(benchmark::State& state,
                                 const loomc_workspace_statistics_t& before,
                                 const loomc_workspace_statistics_t& after) {
  state.counters["workspace_block_size"] =
      static_cast<double>(after.total_block_size);
  state.counters["workspace_new_block_system_allocations"] =
      static_cast<double>(after.block_system_allocation_count -
                          before.block_system_allocation_count);
  state.counters["workspace_new_block_system_bytes"] =
      static_cast<double>(after.block_system_allocation_bytes -
                          before.block_system_allocation_bytes);
  state.counters["workspace_new_oversized_allocations"] = static_cast<double>(
      after.oversized_allocation_count - before.oversized_allocation_count);
}

static iree_status_t LinkBatch(const LinkRequestFixture& fixture,
                               std::vector<RequestPtr>* requests,
                               std::vector<ResultPtr>* results) {
  for (size_t i = 0; i < requests->size(); ++i) {
    IREE_RETURN_IF_ERROR(fixture.Link(&(*requests)[i], &(*results)[i]));
    benchmark::DoNotOptimize((*requests)[i].get());
  }
  return iree_ok_status();
}

static void ReleaseBatch(std::vector<RequestPtr>* requests,
                         std::vector<ResultPtr>* results) {
  for (ResultPtr& result : *results) {
    result.reset();
  }
  for (RequestPtr& request : *requests) {
    request.reset();
  }
}

static void BenchmarkLinkRequest(benchmark::State& state) {
  LinkRequestFixture fixture(static_cast<uint32_t>(state.range(0)),
                             static_cast<uint32_t>(state.range(1)),
                             static_cast<uint32_t>(state.range(2)));
  if (SkipOnError(state, fixture.Initialize())) {
    return;
  }

  const uint32_t links_per_batch = static_cast<uint32_t>(state.range(3));
  std::vector<RequestPtr> requests(links_per_batch);
  std::vector<ResultPtr> results(links_per_batch);
  if (SkipOnError(state, LinkBatch(fixture, &requests, &results))) {
    return;
  }
  const size_t output_source_size =
      loomc_source_contents(loomc_request_source(requests.front().get()))
          .data_length;
  ReleaseBatch(&requests, &results);

  loomc_workspace_statistics_t before = {};
  loomc_workspace_query_statistics(fixture.workspace(), &before);
  int64_t total_links = 0;
  for (auto _ : state) {
    if (SkipOnError(state, LinkBatch(fixture, &requests, &results))) {
      break;
    }
    total_links += links_per_batch;
    state.PauseTiming();
    ReleaseBatch(&requests, &results);
    state.ResumeTiming();
  }
  loomc_workspace_statistics_t after = {};
  loomc_workspace_query_statistics(fixture.workspace(), &after);

  state.counters["library_symbols"] =
      static_cast<double>(fixture.library_symbol_count());
  state.counters["library_source_bytes"] =
      static_cast<double>(fixture.library_source_size());
  state.counters["request_roots"] = static_cast<double>(fixture.root_count());
  state.counters["request_values/root"] =
      static_cast<double>(fixture.values_per_root());
  state.counters["request_source_bytes"] =
      static_cast<double>(fixture.request_source_size());
  state.counters["output_source_bytes"] =
      static_cast<double>(output_source_size);
  state.counters["selected_functions"] =
      static_cast<double>(fixture.root_count() + 1u);
  state.counters["links/batch"] = static_cast<double>(links_per_batch);
  state.counters["links/s"] =
      benchmark::Counter(total_links, benchmark::Counter::kIsRate);
  SetWorkspaceCounters(state, before, after);
  state.SetItemsProcessed(total_links);
  state.SetBytesProcessed(total_links *
                          static_cast<int64_t>(output_source_size));
}

static void BM_LinkRequest_LibraryUniverse(benchmark::State& state) {
  BenchmarkLinkRequest(state);
}

static void BM_LinkRequest_RootCount(benchmark::State& state) {
  BenchmarkLinkRequest(state);
}

static void BM_LinkRequest_BodySize(benchmark::State& state) {
  BenchmarkLinkRequest(state);
}

static void BM_LinkRequest_Smoke(benchmark::State& state) {
  BenchmarkLinkRequest(state);
}

static void LibraryUniverseScales(benchmark::Benchmark* benchmark) {
  for (int64_t library_symbols : {1, 16, 64, 512, 4096}) {
    benchmark->Args({library_symbols, 1, 0, 16});
  }
}

static void RootCountScales(benchmark::Benchmark* benchmark) {
  for (int64_t request_roots : {1, 16, 64, 256}) {
    benchmark->Args({4096, request_roots, 0, 16});
  }
}

static void BodySizeScales(benchmark::Benchmark* benchmark) {
  for (int64_t values_per_root : {0, 16, 256, 1024}) {
    benchmark->Args({4096, 1, values_per_root, 16});
  }
}

BENCHMARK(BM_LinkRequest_LibraryUniverse)
    ->Apply(LibraryUniverseScales)
    ->ArgNames({"library_symbols", "request_roots", "values/root",
                "links/batch"});
BENCHMARK(BM_LinkRequest_RootCount)
    ->Apply(RootCountScales)
    ->ArgNames({"library_symbols", "request_roots", "values/root",
                "links/batch"});
BENCHMARK(BM_LinkRequest_BodySize)
    ->Apply(BodySizeScales)
    ->ArgNames({"library_symbols", "request_roots", "values/root",
                "links/batch"});
BENCHMARK(BM_LinkRequest_Smoke)
    ->Args({16, 4, 4, 2})
    ->ArgNames({"library_symbols", "request_roots", "values/root",
                "links/batch"});

}  // namespace
}  // namespace loomc::bench

BENCHMARK_MAIN();
