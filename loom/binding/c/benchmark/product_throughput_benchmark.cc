// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks command-product construction through the public LoomC API. The
// frozen-index and ordinary-request paths share identical bytecode and a warm
// worker-local workspace. Timed regions include parent artifact serialization,
// request overlay construction when selected, and optional immutable child
// request serialization.

#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "loom/binding/c/benchmark/util/benchmark_support.h"
#include "loomc/iree.h"
#include "loomc/target/cmd.h"

namespace loomc::bench {
namespace {

using CmdProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
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

static std::string BuildIndependentKernelSource(
    uint32_t catalog_kernel_count, uint32_t launched_kernel_count) {
  std::string source;
  source.reserve(static_cast<size_t>(catalog_kernel_count) * 320u);
  for (uint32_t i = 0; i < catalog_kernel_count; ++i) {
    source.append("kernel.def @kernel_");
    source.append(std::to_string(i));
    source.append(R"(() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch() {
  kernel.return
}

)");
  }
  source.append("command.program.def public @root() launch() {\n");
  for (uint32_t i = 0; i < launched_kernel_count; ++i) {
    source.append("  kernel.launch @kernel_");
    source.append(std::to_string(i));
    source.append("() : ()\n");
  }
  source.append("  command.return\n}\n");
  return source;
}

struct RequestCapture {
  // Request references transferred by the active product build.
  std::vector<RequestPtr> requests;
};

enum class CommandProductInput {
  kFrozenIndex,
  kRequest,
};

enum class RequestPublication {
  kBodyBlind,
  kPublish,
};

static loomc_status_t CaptureRequest(void* user_data,
                                     loomc_request_t* request) {
  RequestCapture* capture = static_cast<RequestCapture*>(user_data);
  capture->requests.push_back(RequestPtr(request));
  return loomc_ok_status();
}

class CommandProductFixture {
 public:
  CommandProductFixture(uint32_t catalog_kernel_count,
                        uint32_t launched_kernel_count)
      : catalog_kernel_count_(catalog_kernel_count),
        launched_kernel_count_(launched_kernel_count) {}

  iree_status_t Initialize() {
    loomc_context_t* context = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_context_create(
        /*options=*/nullptr, loom_allocator(), &context)));
    context_.reset(context);

    loomc_workspace_t* source_workspace = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
        /*options=*/nullptr, loom_allocator(), &source_workspace)));
    WorkspacePtr source_workspace_ptr(source_workspace);

    const std::string source_text = BuildIndependentKernelSource(
        catalog_kernel_count_, launched_kernel_count_);
    const loomc_source_options_t source_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /*.structure_size=*/sizeof(source_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
        /*.identifier=*/loomc_make_cstring_view("command_product.loom"),
        /*.contents=*/
        loomc_make_byte_span(source_text.data(), source_text.size()),
        /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
        /*.release=*/nullptr,
        /*.release_user_data=*/nullptr,
    };
    loomc_source_t* text_source = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(
        loomc_source_create(&source_options, loom_allocator(), &text_source)));
    SourcePtr text_source_ptr(text_source);

    loomc_module_t* module = nullptr;
    loomc_result_t* parse_result = nullptr;
    iree_status_t status = to_iree_status(loomc_module_deserialize_from_source(
        context_.get(), source_workspace_ptr.get(), text_source_ptr.get(),
        /*options=*/nullptr, loom_allocator(), &module, &parse_result));
    ModulePtr module_ptr(module);
    ResultPtr parse_result_ptr(parse_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(parse_result_ptr.get(), "catalog parsing"));

    const loomc_module_serialize_options_t serialize_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        /*.structure_size=*/sizeof(serialize_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
        /*.identifier=*/loomc_make_cstring_view("command_product.loombc"),
        /*.text_presentation=*/LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT,
    };
    loomc_source_t* bytecode_source = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(
        loomc_module_serialize_to_source(module_ptr.get(), &serialize_options,
                                         loom_allocator(), &bytecode_source)));
    SourcePtr bytecode_source_ptr(bytecode_source);
    source_size_ = loomc_source_contents(bytecode_source_ptr.get()).data_length;

    loomc_link_index_builder_t* builder = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &builder)));
    LinkIndexBuilderPtr builder_ptr(builder);
    const loomc_link_index_source_options_t index_source_options = {
        /*.provider_name=*/loomc_make_cstring_view("command_product"),
        /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_add_source(
        builder_ptr.get(), bytecode_source_ptr.get(), &index_source_options,
        /*out_slot=*/nullptr)));

    loomc_link_index_t* index = nullptr;
    loomc_result_t* index_result = nullptr;
    status = to_iree_status(loomc_link_index_builder_finish(
        builder_ptr.get(), &index, &index_result));
    LinkIndexPtr index_ptr(index);
    ResultPtr index_result_ptr(index_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(index_result_ptr.get(),
                                                "link index construction"));
    loomc_link_index_symbol_t root_symbol = {};
    if (!loomc_link_index_lookup_global(
            index_ptr.get(), loomc_make_cstring_view("root"), &root_symbol)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "command root was not indexed");
    }
    root_symbol_ordinal_ = root_symbol.ordinal;
    const loomc_request_root_t request_root = {
        /*.module_ordinal=*/
        static_cast<uint32_t>(root_symbol.provider_module_ordinal),
        /*.symbol_ordinal=*/
        static_cast<uint32_t>(root_symbol.module_symbol_ordinal),
    };
    loomc_request_t* request = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_request_create(
        loomc_cmd_program_product_descriptor(), bytecode_source_ptr.get(),
        &request_root, /*root_count=*/1, /*bindings=*/nullptr,
        /*binding_count=*/0, loom_allocator(), &request)));
    request_.reset(request);
    index_.reset(index_ptr.release());

    loomc_workspace_t* workspace = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
        /*options=*/nullptr, loom_allocator(), &workspace)));
    workspace_.reset(workspace);
    return iree_ok_status();
  }

  iree_status_t Build(CommandProductInput input,
                      RequestPublication request_publication,
                      RequestCapture* capture, CmdProductPtr* out_product,
                      ResultPtr* out_result) const {
    capture->requests.clear();
    out_product->reset();
    out_result->reset();
    const loomc_request_sink_t request_sink =
        request_publication == RequestPublication::kPublish
            ? loomc_request_sink_t{
                  /*.publish=*/CaptureRequest,
                  /*.user_data=*/capture,
              }
            : loomc_request_sink_t{};
    loomc_product_t* product = nullptr;
    loomc_result_t* result = nullptr;
    iree_status_t status = iree_ok_status();
    if (input == CommandProductInput::kFrozenIndex) {
      const loomc_cmd_program_product_options_t options = {
          /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
          /*.structure_size=*/sizeof(options),
          /*.next=*/nullptr,
          /*.link_index=*/index_.get(),
          /*.root_symbol_ordinals=*/&root_symbol_ordinal_,
          /*.root_symbol_count=*/1,
          /*.flags=*/0,
          /*.config=*/{},
          /*.request_sink=*/request_sink,
      };
      status = to_iree_status(loomc_cmd_program_product_build(
          workspace_.get(), &options, loom_allocator(), &product, &result));
    } else {
      const loomc_cmd_program_request_options_t options = {
          /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_REQUEST_OPTIONS,
          /*.structure_size=*/sizeof(options),
          /*.next=*/nullptr,
          /*.library_index=*/nullptr,
          /*.config=*/{},
          /*.request_sink=*/request_sink,
      };
      status = to_iree_status(loomc_cmd_program_product_build_request(
          context_.get(), workspace_.get(), request_.get(), &options,
          loom_allocator(), &product, &result));
    }
    CmdProductPtr product_ptr(product);
    ResultPtr result_ptr(result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result_ptr.get(), "command product build"));
    if (product_ptr == nullptr ||
        loomc_cmd_program_product_program_count(product_ptr.get()) != 1 ||
        loomc_cmd_program_product_entry_requirement_count(product_ptr.get()) !=
            launched_kernel_count_ ||
        capture->requests.size() !=
            (request_publication == RequestPublication::kPublish
                 ? launched_kernel_count_
                 : 0)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "command product has an unexpected shape");
    }
    out_product->reset(product_ptr.release());
    out_result->reset(result_ptr.release());
    return iree_ok_status();
  }

  uint32_t catalog_kernel_count() const { return catalog_kernel_count_; }

  uint32_t launched_kernel_count() const { return launched_kernel_count_; }

  size_t source_size() const { return source_size_; }

 private:
  // Number of independent source-backed kernels available in the catalog.
  uint32_t catalog_kernel_count_ = 0;

  // Number of catalog kernels launched by the selected command program.
  uint32_t launched_kernel_count_ = 0;

  // Serialized input catalog size in bytes.
  size_t source_size_ = 0;

  // Public command root ordinal in |index_|.
  loomc_host_size_t root_symbol_ordinal_ = 0;

  // Reusable API context shared by the index and active workspace.
  ContextPtr context_;

  // Frozen source index reused by every command-product build.
  LinkIndexPtr index_;

  // Immutable request selecting the same command root by source-local ordinal.
  RequestPtr request_;

  // Warm mutable storage owned by this benchmark worker.
  WorkspacePtr workspace_;
};

static void RunCommandProductBenchmark(benchmark::State& state,
                                       CommandProductInput input,
                                       RequestPublication request_publication) {
  CommandProductFixture fixture(static_cast<uint32_t>(state.range(0)),
                                static_cast<uint32_t>(state.range(1)));
  if (SkipOnError(state, fixture.Initialize())) {
    return;
  }

  RequestCapture capture;
  capture.requests.reserve(fixture.launched_kernel_count());
  CmdProductPtr product;
  ResultPtr result;
  if (SkipOnError(state, fixture.Build(input, request_publication, &capture,
                                       &product, &result))) {
    return;
  }
  uint64_t request_bytes = 0;
  for (const RequestPtr& request : capture.requests) {
    loomc_source_t* source = loomc_request_source(request.get());
    request_bytes += loomc_source_contents(source).data_length;
  }
  capture.requests.clear();
  result.reset();
  product.reset();

  int64_t build_count = 0;
  for (auto _ : state) {
    if (SkipOnError(state, fixture.Build(input, request_publication, &capture,
                                         &product, &result))) {
      break;
    }
    benchmark::DoNotOptimize(product.get());
    ++build_count;
    state.PauseTiming();
    capture.requests.clear();
    result.reset();
    product.reset();
    state.ResumeTiming();
  }

  const int64_t item_count =
      build_count * static_cast<int64_t>(fixture.launched_kernel_count());
  state.SetItemsProcessed(item_count);
  state.SetComplexityN(fixture.launched_kernel_count());
  state.counters["catalog_bytes"] = static_cast<double>(fixture.source_size());
  state.counters["catalog_kernels"] =
      static_cast<double>(fixture.catalog_kernel_count());
  state.counters["launched_kernels"] =
      static_cast<double>(fixture.launched_kernel_count());
  state.counters["request_bytes"] = static_cast<double>(request_bytes);
  state.counters["requests"] =
      request_publication == RequestPublication::kPublish
          ? static_cast<double>(fixture.launched_kernel_count())
          : 0.0;
}

static void BM_CommandProductBodyBlind(benchmark::State& state) {
  RunCommandProductBenchmark(state, CommandProductInput::kFrozenIndex,
                             RequestPublication::kBodyBlind);
}

static void BM_CommandProductRequests(benchmark::State& state) {
  RunCommandProductBenchmark(state, CommandProductInput::kFrozenIndex,
                             RequestPublication::kPublish);
}

static void BM_CommandRequestBodyBlind(benchmark::State& state) {
  RunCommandProductBenchmark(state, CommandProductInput::kRequest,
                             RequestPublication::kBodyBlind);
}

static void BM_CommandRequestRequests(benchmark::State& state) {
  RunCommandProductBenchmark(state, CommandProductInput::kRequest,
                             RequestPublication::kPublish);
}

static void ProductScales(benchmark::Benchmark* benchmark) {
  benchmark->Args({1, 1})->Args({16, 16})->Args({256, 256})->Args({1024, 1024});
}

static void CatalogScales(benchmark::Benchmark* benchmark) {
  benchmark->Args({1, 1})->Args({16, 1})->Args({256, 1})->Args({1024, 1})->Args(
      {4096, 1});
}

static void BM_CommandProductBodyBlindUnrelatedCatalog(
    benchmark::State& state) {
  BM_CommandProductBodyBlind(state);
}

static void BM_CommandProductRequestsUnrelatedCatalog(benchmark::State& state) {
  BM_CommandProductRequests(state);
}

static void BM_CommandProductBodyBlind_Smoke(benchmark::State& state) {
  BM_CommandProductBodyBlind(state);
}

static void BM_CommandProductRequests_Smoke(benchmark::State& state) {
  BM_CommandProductRequests(state);
}

static void BM_CommandRequestBodyBlind_Smoke(benchmark::State& state) {
  BM_CommandRequestBodyBlind(state);
}

static void BM_CommandRequestRequests_Smoke(benchmark::State& state) {
  BM_CommandRequestRequests(state);
}

BENCHMARK(BM_CommandProductBodyBlind)
    ->Apply(ProductScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_CommandProductRequests)
    ->Apply(ProductScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_CommandRequestBodyBlind)
    ->Apply(ProductScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_CommandRequestRequests)
    ->Apply(ProductScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_CommandProductBodyBlindUnrelatedCatalog)->Apply(CatalogScales);
BENCHMARK(BM_CommandProductRequestsUnrelatedCatalog)->Apply(CatalogScales);
BENCHMARK(BM_CommandProductBodyBlind_Smoke)->Args({4, 4});
BENCHMARK(BM_CommandProductRequests_Smoke)->Args({4, 4});
BENCHMARK(BM_CommandRequestBodyBlind_Smoke)->Args({4, 4});
BENCHMARK(BM_CommandRequestRequests_Smoke)->Args({4, 4});

}  // namespace
}  // namespace loomc::bench
