// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Measures repeated launch-config evaluation through the public LoomC API.
// Setup compiles ordinary Loom source into a VM bytecode artifact and loads it
// through the production runtime path; only warm function invocation is timed.

#include <cstdint>
#include <string>
#include <utility>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"
#include "loom/binding/c/benchmark/util/benchmark_support.h"
#include "loomc/launch_config.h"
#include "loomc/target/amdgpu.h"

namespace {

using loomc::bench::CompilerPtr;
using loomc::bench::ContextPtr;
using loomc::bench::CreateBenchmarkKernelSource;
using loomc::bench::CreateWorkspace;
using loomc::bench::DeserializeSource;
using loomc::bench::FindArtifact;
using loomc::bench::loom_allocator;
using loomc::bench::ModulePtr;
using loomc::bench::PassProgramPtr;
using loomc::bench::RequireSucceededResult;
using loomc::bench::ResultPtr;
using loomc::bench::SourcePtr;
using loomc::bench::TargetEnvironmentPtr;
using loomc::bench::to_iree_status;
using loomc::bench::WorkspacePtr;

using LaunchConfigProgramPtr =
    loomc::bench::HandlePtr<loomc_launch_config_program_t,
                            loomc_launch_config_program_release>;

class LaunchConfigBenchmarkFixture {
 public:
  iree_status_t SetUp(iree_string_view_t function_name) {
    loomc_target_environment_t* target_environment = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_amdgpu(
        loom_allocator(), &target_environment)));
    target_environment_.reset(target_environment);

    const loomc_context_target_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_environment=*/target_environment_.get(),
    };
    const loomc_context_options_t context_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        /*.structure_size=*/sizeof(context_options),
        /*.next=*/&target_options,
    };
    loomc_context_t* context = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(
        loomc_context_create(&context_options, loom_allocator(), &context)));
    context_.reset(context);

    loomc_compiler_t* compiler = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_compiler_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &compiler)));
    compiler_.reset(compiler);

    const loomc_target_pipeline_options_t pipeline_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        /*.structure_size=*/sizeof(pipeline_options),
        /*.next=*/nullptr,
        /*.identifier=*/loomc_make_cstring_view("launch-config-benchmark"),
        /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        /*.source_to_low_max_errors=*/20,
    };
    loomc_pass_program_t* pass_program = nullptr;
    loomc_result_t* pipeline_result = nullptr;
    iree_status_t status =
        to_iree_status(loomc_pass_program_create_from_target_pipeline(
            context_.get(), &pipeline_options, loom_allocator(), &pass_program,
            &pipeline_result));
    PassProgramPtr pass_program_owner(pass_program);
    ResultPtr pipeline_result_owner(pipeline_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(pipeline_result_owner.get(),
                                                "target pipeline preparation"));
    pass_program_.reset(pass_program_owner.release());

    IREE_RETURN_IF_ERROR(CreateWorkspace(/*block_size=*/0, &workspace_));
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("launch_config.loom"), &source_));
    IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(), workspace_.get(),
                                           source_.get(), &module_));

    const loomc_compile_options_t compile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(compile_options),
        /*.next=*/nullptr,
        /*.module_name=*/loomc_make_cstring_view("launch_config_benchmark"),
        /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
    };
    loomc_result_t* compile_result = nullptr;
    status = to_iree_status(loomc_compile_module(
        compiler_.get(), workspace_.get(), pass_program_.get(), module_.get(),
        &compile_options, loom_allocator(), &compile_result));
    ResultPtr compile_result_owner(compile_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(compile_result_owner.get(),
                                                "module compilation"));

    const loomc_artifact_t* artifact = FindArtifact(
        compile_result_owner.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM_BYTECODE));
    if (artifact == nullptr) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "launch-config artifact was not produced");
    }
    loomc_launch_config_program_t* program = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_launch_config_program_load(
        artifact, loom_allocator(), &program)));
    program_.reset(program);

    IREE_RETURN_IF_ERROR(
        to_iree_status(loomc_launch_config_program_lookup_function(
            program_.get(), loomc_string_view_from_iree(function_name),
            &function_)));
    IREE_RETURN_IF_ERROR(Invoke(UINT64_C(1048576), &warm_config_));
    if (warm_config_.workgroup_count.x != 4096 ||
        warm_config_.workgroup_count.y != 1 ||
        warm_config_.workgroup_count.z != 1 ||
        warm_config_.workgroup_size.x != 256 ||
        warm_config_.workgroup_size.y != 1 ||
        warm_config_.workgroup_size.z != 1 ||
        warm_config_.workgroup_cluster_size.x != 1 ||
        warm_config_.workgroup_cluster_size.y != 1 ||
        warm_config_.workgroup_cluster_size.z != 1 ||
        warm_config_.subgroup_size != 32 ||
        warm_config_.workgroup_storage_bytes != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "warm launch-config invocation produced an unexpected result");
    }
    return iree_ok_status();
  }

  iree_status_t Invoke(uint64_t element_count,
                       loomc_launch_config_t* out_config) {
    return to_iree_status(loomc_launch_config_program_invoke(
        program_.get(), function_, &element_count, /*argument_count=*/1,
        out_config));
  }

 private:
  TargetEnvironmentPtr target_environment_;
  ContextPtr context_;
  CompilerPtr compiler_;
  PassProgramPtr pass_program_;
  WorkspacePtr workspace_;
  SourcePtr source_;
  ModulePtr module_;
  LaunchConfigProgramPtr program_;
  loomc_launch_config_function_t function_ =
      loomc_launch_config_function_invalid();
  loomc_launch_config_t warm_config_ = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(loomc_launch_config_t),
  };
};

static bool SkipOnError(benchmark::State& state, iree_status_t status) {
  if (iree_status_is_ok(status)) return false;
  const iree::Status status_owner(std::move(status));
  const std::string message = status_owner.ToString();
  state.SkipWithError(message.c_str());
  return true;
}

static void RunLaunchConfigInvokeBenchmark(benchmark::State& state,
                                           const char* function_name) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state,
                  fixture.SetUp(iree_make_cstring_view(function_name)))) {
    return;
  }

  uint64_t element_count = UINT64_C(1048576);
  loomc_launch_config_t config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(config),
  };
  for (auto _ : state) {
    (void)_;
    benchmark::DoNotOptimize(element_count);
    iree_status_t status = fixture.Invoke(element_count, &config);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.PauseTiming();
      SkipOnError(state, status);
      break;
    }
    benchmark::DoNotOptimize(config);
  }
  state.SetItemsProcessed(state.iterations());
}

static void BM_LaunchConfigInvokeSmoke(benchmark::State& state,
                                       const char* function_name) {
  RunLaunchConfigInvokeBenchmark(state, function_name);
}

BENCHMARK_CAPTURE(BM_LaunchConfigInvokeSmoke, Unchecked1D, "unchecked_1d")
    ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BM_LaunchConfigInvokeSmoke, Checked1D, "checked_1d")
    ->Unit(benchmark::kNanosecond);

}  // namespace
