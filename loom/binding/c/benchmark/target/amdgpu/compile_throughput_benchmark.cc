// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "loomc/target/amdgpu.h"

namespace {

using loomc::bench::CloneModule;
using loomc::bench::CompileScenario;
using loomc::bench::CreateBenchmarkKernelSource;
using loomc::bench::CreateWorkspace;
using loomc::bench::DeserializeSource;
using loomc::bench::loom_allocator;
using loomc::bench::ModulePtr;
using loomc::bench::RequireSucceededResult;
using loomc::bench::ResultPtr;
using loomc::bench::RunCompileBenchmarkDirect;
using loomc::bench::RunCompileBenchmarkDirectCold;
using loomc::bench::SourcePtr;
using loomc::bench::TargetCompileScenario;
using loomc::bench::TargetEnvironmentPtr;
using loomc::bench::TargetProfilePtr;
using loomc::bench::to_iree_status;
using loomc::bench::ValidateArtifact;
using loomc::bench::WorkspacePtr;

struct AmdgpuBenchmarkTarget {
  // Processor key resolved by the production AMDGPU profile table.
  const char* processor;
};

constexpr AmdgpuBenchmarkTarget kGfx1100Target = {"gfx1100"};
constexpr AmdgpuBenchmarkTarget kGfx942Target = {"gfx942"};
constexpr AmdgpuBenchmarkTarget kGfx1200Target = {"gfx1200"};

class AmdgpuI32ChainScenario final : public TargetCompileScenario {
 public:
  AmdgpuI32ChainScenario(
      iree_host_size_t job_count,
      std::initializer_list<iree_host_size_t> operation_counts,
      AmdgpuBenchmarkTarget target, iree_host_size_t workspace_block_size = 0)
      : TargetCompileScenario(workspace_block_size),
        job_count_(std::max<iree_host_size_t>(job_count, 1)),
        target_(target) {
    operation_counts_.reserve(operation_counts.size());
    for (iree_host_size_t operation_count : operation_counts) {
      operation_count = std::max<iree_host_size_t>(operation_count, 1);
      operation_counts_.push_back(
          {operation_count, std::to_string(operation_count)});
    }
  }

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    TargetEnvironmentPtr target_environment;
    loomc_target_environment_t* raw_target_environment = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_amdgpu(
        loom_allocator(), &raw_target_environment)));
    target_environment.reset(raw_target_environment);

    const loomc_string_view_t processor =
        loomc_make_cstring_view(target_.processor);
    loomc_amdgpu_profile_options_t profile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        /*.structure_size=*/sizeof(profile_options),
        /*.next=*/nullptr,
        /*.identifier=*/processor,
        /*.processor=*/processor,
    };
    loomc_target_profile_t* raw_profile = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_profile_create_amdgpu(
        target_environment.get(), &profile_options, loom_allocator(),
        &raw_profile)));
    TargetProfilePtr target_profile(raw_profile);

    IREE_RETURN_IF_ERROR(SetUpTarget(
        worker_count, std::move(target_environment), std::move(target_profile),
        loomc_make_cstring_view("benchmark-amdgpu-prepared-low")));
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("i32_memory_chain.loom"), &source_));
    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    return DeserializeSource(context_.get(), template_workspace_.get(),
                             source_.get(), &template_module_);
  }

  iree_host_size_t job_count() const override { return job_count_; }

  iree_status_t WarmUp(iree_host_size_t worker_count) override {
    for (iree_host_size_t worker_ordinal = 0; worker_ordinal < worker_count;
         ++worker_ordinal) {
      for (iree_host_size_t pattern_ordinal = 0;
           pattern_ordinal < operation_counts_.size(); ++pattern_ordinal) {
        IREE_RETURN_IF_ERROR(RunJob(worker_ordinal, pattern_ordinal));
      }
    }
    return iree_ok_status();
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    const OperationCount& operation_count =
        operation_counts_[job_ordinal % operation_counts_.size()];

    ModulePtr module;
    IREE_RETURN_IF_ERROR(
        CloneModule(template_module_.get(), workspace.get(), &module));
    loomc_config_binding_t bindings[] = {
        {
            /*.key=*/loomc_make_cstring_view("@benchmark.workgroup_size"),
            /*.value=*/loomc_make_cstring_view("64"),
        },
        {
            /*.key=*/loomc_make_cstring_view("@benchmark.operation_count"),
            /*.value=*/
            loomc_make_string_view(operation_count.text.data(),
                                   operation_count.text.size()),
        },
    };
    const loomc_config_options_t config_options = {
        /*.bindings=*/bindings,
        /*.binding_count=*/IREE_ARRAYSIZE(bindings),
        /*.json_object=*/loomc_string_view_empty(),
        /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
            LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
    };
    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("amdgpu_i32_chain"),
        config_options));
    return EmitAmdgpuArtifact(workspace, module);
  }

  void SetExtraCounters(::benchmark::State& state) const override {
    iree_host_size_t minimum = operation_counts_.front().value;
    iree_host_size_t maximum = minimum;
    for (const OperationCount& operation_count : operation_counts_) {
      minimum = std::min(minimum, operation_count.value);
      maximum = std::max(maximum, operation_count.value);
    }
    state.counters["operation_count_min"] = (double)minimum;
    state.counters["operation_count_max"] = (double)maximum;
  }

 private:
  struct OperationCount {
    // Numeric operation count reported by the benchmark.
    iree_host_size_t value;

    // Stable config spelling borrowed by each invocation.
    std::string text;
  };

  iree_status_t EmitAmdgpuArtifact(WorkspacePtr& workspace, ModulePtr& module) {
    loomc_target_selection_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_selection=*/target_selection(),
    };
    loomc_amdgpu_emit_options_t amdgpu_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(amdgpu_options),
        /*.next=*/&target_options,
        /*.runtime_globals=*/LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
    };
    loomc_emit_options_t emit_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(emit_options),
        /*.next=*/&amdgpu_options,
        /*.artifact_format=*/
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        /*.identifier=*/loomc_make_cstring_view("i32_memory_chain.hsaco"),
        /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };

    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_emit_module(target_environment(), workspace.get(), module.get(),
                          &emit_options, loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "AMDGPU emission"));

    constexpr uint8_t kElfMagic[] = {0x7F, 'E', 'L', 'F'};
    int64_t artifact_bytes = 0;
    IREE_RETURN_IF_ERROR(ValidateArtifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        sizeof(kElfMagic), "AMDGPU HSACO executable", &artifact_bytes));
    const loomc_artifact_t* artifact = loomc::bench::FindArtifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
    if (std::memcmp(artifact->contents.data, kElfMagic, sizeof(kElfMagic)) !=
        0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU executable is not an ELF image");
    }

    RecordArtifactBytes(artifact_bytes);
    return iree_ok_status();
  }

  // Number of kernels compiled by each benchmark iteration.
  iree_host_size_t job_count_ = 0;

  // Cyclic sequence of arithmetic-chain sizes compiled by each iteration.
  std::vector<OperationCount> operation_counts_;

  // Immutable target row selected by this benchmark registration.
  AmdgpuBenchmarkTarget target_;

  // Compact targetless source shared by all invocations.
  SourcePtr source_;

  // Setup-only workspace retaining the parsed template module.
  WorkspacePtr template_workspace_;

  // Immutable parsed template cloned into worker workspaces.
  ModulePtr template_module_;
};

static std::unique_ptr<CompileScenario> CreateAmdgpuI32ChainScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* target = static_cast<const AmdgpuBenchmarkTarget*>(user_data);
  return std::make_unique<AmdgpuI32ChainScenario>(
      (iree_host_size_t)state.range(1),
      std::initializer_list<iree_host_size_t>{(iree_host_size_t)state.range(2)},
      *target);
}

static std::unique_ptr<CompileScenario> CreateAmdgpuI32AlternatingChainScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* target = static_cast<const AmdgpuBenchmarkTarget*>(user_data);
  return std::make_unique<AmdgpuI32ChainScenario>(
      (iree_host_size_t)state.range(1),
      std::initializer_list<iree_host_size_t>{(iree_host_size_t)state.range(2),
                                              (iree_host_size_t)state.range(3)},
      *target);
}

static std::unique_ptr<CompileScenario> CreateAmdgpuI32ChainWorkspaceScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* target = static_cast<const AmdgpuBenchmarkTarget*>(user_data);
  return std::make_unique<AmdgpuI32ChainScenario>(
      (iree_host_size_t)state.range(1),
      std::initializer_list<iree_host_size_t>{(iree_host_size_t)state.range(2)},
      *target, (iree_host_size_t)state.range(3));
}

static void BM_AmdgpuI32ChainSmoke(::benchmark::State& state,
                                   const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuI32ChainScenario, target);
}
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainSmoke, Gfx1100, &kGfx1100Target)
    ->Args({1, 1, 16})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainSmoke, Gfx942, &kGfx942Target)
    ->Args({1, 1, 16})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainSmoke, Gfx1200, &kGfx1200Target)
    ->Args({1, 1, 16})
    ->UseRealTime();

static void BM_AmdgpuI32ChainCold(::benchmark::State& state,
                                  const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirectCold(state, CreateAmdgpuI32ChainScenario, target);
}
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainCold, Gfx1100, &kGfx1100Target)
    ->Args({1, 1, 16})
    ->Args({1, 1, 1024})
    ->Iterations(1)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainCold, Gfx942, &kGfx942Target)
    ->Args({1, 1, 16})
    ->Args({1, 1, 1024})
    ->Iterations(1)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainCold, Gfx1200, &kGfx1200Target)
    ->Args({1, 1, 16})
    ->Args({1, 1, 1024})
    ->Iterations(1)
    ->UseRealTime();

static void BM_AmdgpuI32ChainWorkspaceCold(
    ::benchmark::State& state, const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirectCold(state, CreateAmdgpuI32ChainWorkspaceScenario,
                                target);
}

static void BM_AmdgpuI32ChainWorkspace(::benchmark::State& state,
                                       const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuI32ChainWorkspaceScenario,
                            target);
}

#define LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE(target_name, target)      \
  BENCHMARK_CAPTURE(BM_AmdgpuI32ChainWorkspaceCold, target_name, target)    \
      ->ArgsProduct({{1}, {1}, {1024}, {32 * 1024, 64 * 1024, 128 * 1024}}) \
      ->Iterations(1)                                                       \
      ->UseRealTime();                                                      \
  BENCHMARK_CAPTURE(BM_AmdgpuI32ChainWorkspace, target_name, target)        \
      ->ArgsProduct({{1}, {1}, {1024}, {32 * 1024, 64 * 1024, 128 * 1024}}) \
      ->UseRealTime()

LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE(Gfx1100, &kGfx1100Target);
LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE(Gfx942, &kGfx942Target);
LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE(Gfx1200, &kGfx1200Target);

#undef LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE

static void BM_AmdgpuI32ChainAlternating(::benchmark::State& state,
                                         const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuI32AlternatingChainScenario,
                            target);
}
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainAlternating, Gfx1100, &kGfx1100Target)
    ->Args({1, 2, 16, 1024})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainAlternating, Gfx942, &kGfx942Target)
    ->Args({1, 2, 16, 1024})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainAlternating, Gfx1200, &kGfx1200Target)
    ->Args({1, 2, 16, 1024})
    ->UseRealTime();

static void BM_AmdgpuI32Chain(::benchmark::State& state,
                              const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuI32ChainScenario, target);
}
#define LOOM_BENCHMARK_AMDGPU_I32_CHAIN(target_name, target) \
  BENCHMARK_CAPTURE(BM_AmdgpuI32Chain, target_name, target)  \
      ->Args({1, 16, 1})                                     \
      ->Args({1, 16, 64})                                    \
      ->Args({1, 4, 1024})                                   \
      ->UseRealTime()

LOOM_BENCHMARK_AMDGPU_I32_CHAIN(Gfx1100, &kGfx1100Target);
LOOM_BENCHMARK_AMDGPU_I32_CHAIN(Gfx942, &kGfx942Target);
LOOM_BENCHMARK_AMDGPU_I32_CHAIN(Gfx1200, &kGfx1200Target);

#undef LOOM_BENCHMARK_AMDGPU_I32_CHAIN

}  // namespace
