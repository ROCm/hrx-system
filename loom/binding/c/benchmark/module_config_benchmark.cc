// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sstream>

#include "benchmark/benchmark.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"

namespace loomc::bench {
namespace {

enum class ConfigOperation { kApply, kRequireResolved, kSchema };

template <ConfigOperation operation>
void BenchmarkModuleConfig(benchmark::State& state) {
  const auto skip_on_error = [&](iree_status_t status) {
    if (iree_status_is_ok(status)) {
      return false;
    }
    state.SkipWithError(FormatStatus(status));
    iree_status_ignore(status);
    return true;
  };
  loomc_context_t* raw_context = nullptr;
  if (skip_on_error(to_iree_status(
          loomc_context_create(nullptr, loom_allocator(), &raw_context)))) {
    return;
  }
  ContextPtr context(raw_context);
  WorkspacePtr workspace;
  if (skip_on_error(CreateWorkspace(/*block_size=*/0, &workspace))) {
    return;
  }
  std::ostringstream source;
  std::ostringstream json;
  json << '{';
  for (int64_t i = 0; i < state.range(0); ++i) {
    source << "config.decl @parameter_" << i << " : index\n";
    if (i) {
      json << ',';
    }
    json << "\"parameter_" << i << "\":" << i;
  }
  json << '}';
  const std::string bindings = json.str();
  ModulePtr original;
  if (skip_on_error(CreateTextModule(context.get(), workspace.get(),
                                     "configuration.loom", source.str(),
                                     &original))) {
    return;
  }
  loomc_module_config_options_t options = {
      .config =
          {
              .json_object = loomc_make_cstring_view(bindings.c_str()),
              .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
          },
  };
  if constexpr (operation != ConfigOperation::kApply) {
    loomc_result_t* result = nullptr;
    auto status = to_iree_status(loomc_module_apply_config(
        original.get(), workspace.get(), &options, loom_allocator(), &result));
    ResultPtr result_owner(result);
    if (skip_on_error(status) || skip_on_error(RequireSucceededResult(
                                     result, "initial configuration"))) {
      return;
    }
    options.config.json_object = loomc_string_view_empty();
  }
  int64_t schema_bytes = 0;
  for (auto _ : state) {
    state.PauseTiming();
    ModulePtr module;
    auto status = CloneModule(original.get(), workspace.get(), &module);
    state.ResumeTiming();
    if (skip_on_error(status)) {
      break;
    }
    loomc_result_t* result = nullptr;
    if constexpr (operation == ConfigOperation::kSchema) {
      status = to_iree_status(loomc_module_query_config_schema(
          module.get(), loom_allocator(), &result));
    } else {
      status = to_iree_status(loomc_module_apply_config(
          module.get(), workspace.get(), &options, loom_allocator(), &result));
    }
    ResultPtr result_owner(result);
    if (skip_on_error(status) ||
        skip_on_error(RequireSucceededResult(result, "module configuration"))) {
      break;
    }
    if constexpr (operation == ConfigOperation::kSchema) {
      schema_bytes += loomc_byte_sequence_length(
          loomc_result_artifact_at(result, 0)->contents);
    }
    benchmark::DoNotOptimize(result);
    state.PauseTiming();
    result_owner.reset();
    module.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations());
  if constexpr (operation == ConfigOperation::kSchema) {
    state.SetBytesProcessed(schema_bytes);
  }
}

BENCHMARK_TEMPLATE(BenchmarkModuleConfig, ConfigOperation::kApply)
    ->Name("BM_ModuleApplyConfig")
    ->Arg(1)
    ->Arg(32);
BENCHMARK_TEMPLATE(BenchmarkModuleConfig, ConfigOperation::kRequireResolved)
    ->Name("BM_ModuleRequireResolvedConfig")
    ->Arg(1)
    ->Arg(32);
BENCHMARK_TEMPLATE(BenchmarkModuleConfig, ConfigOperation::kSchema)
    ->Name("BM_ModuleConfigSchema")
    ->Arg(1)
    ->Arg(32);
BENCHMARK_TEMPLATE(BenchmarkModuleConfig, ConfigOperation::kApply)
    ->Name("BM_ModuleConfigApplySmoke")
    ->Arg(1)
    ->Iterations(1);
BENCHMARK_TEMPLATE(BenchmarkModuleConfig, ConfigOperation::kRequireResolved)
    ->Name("BM_ModuleConfigRequireResolvedSmoke")
    ->Arg(1)
    ->Iterations(1);
BENCHMARK_TEMPLATE(BenchmarkModuleConfig, ConfigOperation::kSchema)
    ->Name("BM_ModuleConfigSchemaSmoke")
    ->Arg(1)
    ->Iterations(1);

}  // namespace
}  // namespace loomc::bench
