// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <string>
#include <string_view>

#include "benchmark/benchmark.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"
#include "loom/binding/c/benchmark/import/cxx/q8s32_sources.h"
#include "loomc/import/cxx.h"

namespace loomc::bench {
namespace {

enum class SourceLanguage {
  kCxx,
  kLoom,
};

struct SourceToModuleInput {
  // Language accepted at the public source-to-module boundary.
  SourceLanguage language;
  // Optional facade prefix prepended to the shared no-use function.
  const char* prefix;
  // Optional embedded translation unit replacing the shared function.
  const char* source_identifier;
  // Optional embedded header supplied through the public source provider.
  const char* header_identifier;
};

static iree_status_t CreateSource(loomc_string_view_t identifier,
                                  loomc_byte_span_t contents,
                                  loomc_source_format_t format,
                                  SourcePtr* out_source) {
  out_source->reset();
  const loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = format,
      .identifier = identifier,
      .contents = contents,
      .storage = LOOMC_SOURCE_STORAGE_COPY,
      .release = nullptr,
      .release_user_data = nullptr,
  };
  loomc_source_t* source = nullptr;
  IREE_RETURN_IF_ERROR(
      to_iree_status(loomc_source_create(&options, loom_allocator(), &source)));
  out_source->reset(source);
  return iree_ok_status();
}

class SourceToModuleScenario final : public CompileScenario {
 public:
  explicit SourceToModuleScenario(const SourceToModuleInput& source,
                                  iree_host_size_t job_count = 1)
      : input_(source), job_count_(job_count) {}

  iree_host_size_t job_count() const override { return job_count_; }

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(CompileScenario::SetUp(worker_count));
    if (input_.source_identifier) {
      const auto* sources = loomc_cxx_benchmark_q8s32_sources_create();
      const size_t source_count = loomc_cxx_benchmark_q8s32_sources_size();
      EmbeddedSource source =
          FindEmbeddedSource(sources, source_count, input_.source_identifier);
      IREE_RETURN_IF_ERROR(CreateSource(source.identifier, source.contents,
                                        input_.language == SourceLanguage::kLoom
                                            ? LOOMC_SOURCE_FORMAT_TEXT
                                            : LOOMC_SOURCE_FORMAT_UNKNOWN,
                                        &source_));
      if (input_.header_identifier) {
        EmbeddedSource header =
            FindEmbeddedSource(sources, source_count, input_.header_identifier);
        IREE_RETURN_IF_ERROR(CreateSource(header.identifier, header.contents,
                                          LOOMC_SOURCE_FORMAT_UNKNOWN,
                                          &header_));
      }
    } else {
      std::string text = std::string(input_.prefix) +
                         "__bf16 scale(__bf16 value) { "
                         "return value * (__bf16)5.0f; }";
      IREE_RETURN_IF_ERROR(
          CreateSource(loomc_make_cstring_view("scale.cpp"),
                       loomc_make_byte_span(text.data(), text.size()),
                       LOOMC_SOURCE_FORMAT_UNKNOWN, &source_));
    }
    return iree_ok_status();
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    if (input_.language == SourceLanguage::kLoom) {
      iree_status_t status =
          to_iree_status(loomc_module_deserialize_from_source(
              context_.get(), workspace_at(worker_ordinal).get(), source_.get(),
              /*options=*/nullptr, loom_allocator(), &raw_module, &raw_result));
      ModulePtr module(raw_module);
      ResultPtr result(raw_result);
      IREE_RETURN_IF_ERROR(status);
      return RequireSucceededResult(result.get(), "Loom source parsing");
    }

    loomc_cxx_import_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS;
    options.structure_size = sizeof(options);
    if (header_) {
      options.source_provider = {ProvideSource, this};
    }
    iree_status_t status = to_iree_status(loomc_module_import_cxx(
        context_.get(), workspace_at(worker_ordinal).get(), source_.get(),
        &options, loom_allocator(), &raw_module, &raw_result));
    ModulePtr module(raw_module);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    return RequireSucceededResult(result.get(), "C++ import");
  }

 private:
  static loomc_status_t ProvideSource(void* user_data, loomc_string_view_t path,
                                      loomc_source_t** out_source) {
    auto& self = *static_cast<SourceToModuleScenario*>(user_data);
    *out_source = nullptr;
    const auto header_identifier = loomc_source_identifier(self.header_.get());
    // The embedded source table is intentionally flattened. Include lookup
    // still supplies an ordinary generic path, so index it by basename.
    std::string_view candidate(path.data, path.size);
    const size_t separator = candidate.find_last_of('/');
    if (separator != std::string_view::npos) {
      candidate.remove_prefix(separator + 1);
    }
    if (candidate ==
        std::string_view(header_identifier.data, header_identifier.size)) {
      loomc_source_retain(self.header_.get());
      *out_source = self.header_.get();
    }
    return loomc_ok_status();
  }

  // Static source selection chosen by benchmark registration.
  const SourceToModuleInput& input_;
  // Copied source bytes shared across invocations without cached parse state.
  SourcePtr source_;
  // Optional immutable user header returned by the source provider.
  SourcePtr header_;
  // Number of independent imports submitted in each compile-pool batch.
  iree_host_size_t job_count_ = 1;
};

std::unique_ptr<CompileScenario> CreateSourceToModuleScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)state;
  return std::make_unique<SourceToModuleScenario>(
      *static_cast<const SourceToModuleInput*>(user_data));
}

void SourceToModule(::benchmark::State& state,
                    const SourceToModuleInput* source) {
  RunCompileBenchmarkDirect(state, CreateSourceToModuleScenario, source);
}

std::unique_ptr<CompileScenario> CreateSourceToModuleThroughputScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto worker_count = static_cast<iree_host_size_t>(state.range(0));
  return std::make_unique<SourceToModuleScenario>(
      *static_cast<const SourceToModuleInput*>(user_data), worker_count);
}

void SourceToModuleThroughput(::benchmark::State& state,
                              const SourceToModuleInput* source) {
  RunCompileBenchmark(state, CreateSourceToModuleThroughputScenario, source);
}

constexpr SourceToModuleInput kNoIncludes = {SourceLanguage::kCxx, "", nullptr,
                                             nullptr};
constexpr SourceToModuleInput kStdFloat = {
    SourceLanguage::kCxx, "#include <stdfloat>\n", nullptr, nullptr};
constexpr SourceToModuleInput kNumeric = {
    SourceLanguage::kCxx, "#include <loomcxx/numeric.h>\n", nullptr, nullptr};
constexpr SourceToModuleInput kVector = {
    SourceLanguage::kCxx, "#include <loomcxx/vector.h>\n", nullptr, nullptr};
constexpr SourceToModuleInput kEncodingType = {
    SourceLanguage::kCxx, "#include <loomcxx/encoding_type.h>\n", nullptr,
    nullptr};
constexpr SourceToModuleInput kEncoding = {
    SourceLanguage::kCxx, "#include <loomcxx/encoding.h>\n", nullptr, nullptr};
constexpr SourceToModuleInput kView = {
    SourceLanguage::kCxx, "#include <loomcxx/view.h>\n", nullptr, nullptr};
constexpr SourceToModuleInput kRankTwoView = {SourceLanguage::kCxx,
                                              R"cxx(#include <loomcxx/view.h>
float load_rank_two(const float* data, unsigned rows, unsigned row,
                    unsigned column) {
  auto layout = loom::encoding::layout::dense<2>();
  auto source = loom::buffer::view<loom::type::dynamic, 32>(
      data, {rows}, layout);
  return loom::view::load(source, row, column);
}
)cxx",
                                              nullptr, nullptr};
constexpr SourceToModuleInput kRankThreeStorageView = {
    SourceLanguage::kCxx,
    R"cxx(#include <loomcxx/encoding.h>
#include <loomcxx/numeric.h>
#include <loomcxx/view.h>
using Weight = loom::type::float8_e4m3fn_t;
Weight load_rank_three(const Weight* data, unsigned rows, unsigned tiles,
                       unsigned row, unsigned tile, unsigned lane) {
  auto layout = loom::encoding::layout::dense<3>();
  auto schema = loom::encoding::define<loom::encoding::f8e4m3fn{
      .payload_elements = 16, .scale_group_elements = 16}>();
  auto storage = loom::encoding::define(layout, schema);
  auto source = loom::buffer::view<loom::type::dynamic,
                                   loom::type::dynamic, 16>(
      data, {rows, tiles}, storage);
  return loom::view::load(source, row, tile, lane);
}
)cxx",
    nullptr, nullptr};
constexpr SourceToModuleInput kPredicate = {
    SourceLanguage::kCxx, "#include <loomcxx/predicate.h>\n", nullptr, nullptr};
constexpr SourceToModuleInput kKernel = {
    SourceLanguage::kCxx, "#include <loomcxx/kernel.h>\n", nullptr, nullptr};
constexpr SourceToModuleInput kKernelPredicate = {
    SourceLanguage::kCxx,
    "#include <loomcxx/kernel.h>\n#include <loomcxx/predicate.h>\n", nullptr,
    nullptr};
constexpr SourceToModuleInput kQ8S32Providers = {
    SourceLanguage::kCxx, nullptr, "q8s32_specialization_providers.cxx",
    "q8s32_specialization.h"};
constexpr SourceToModuleInput kQ8S32AuthoredLoomProviders = {
    SourceLanguage::kLoom, nullptr, "q8s32_specialization_providers.loom",
    nullptr};

BENCHMARK_CAPTURE(SourceToModule, NoIncludes, &kNoIncludes)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, StdFloat, &kStdFloat)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, Numeric, &kNumeric)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, Vector, &kVector)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, EncodingType, &kEncodingType)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, Encoding, &kEncoding)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, View, &kView)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, RankTwoView, &kRankTwoView)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, RankThreeStorageView, &kRankThreeStorageView)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, Predicate, &kPredicate)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, Kernel, &kKernel)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, KernelPredicate, &kKernelPredicate)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, Q8S32Providers, &kQ8S32Providers)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToModule, Q8S32AuthoredLoomProviders,
                  &kQ8S32AuthoredLoomProviders)
    ->Unit(::benchmark::kMicrosecond);

BENCHMARK_CAPTURE(SourceToModuleThroughput, Q8S32Providers, &kQ8S32Providers)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->UseRealTime()
    ->Unit(::benchmark::kMicrosecond);

}  // namespace
}  // namespace loomc::bench
