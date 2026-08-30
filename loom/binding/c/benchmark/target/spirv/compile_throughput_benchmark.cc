// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "benchmark/benchmark.h"
#include "loomc/target/spirv.h"

namespace {

using loomc::bench::CloneModule;
using loomc::bench::CompileScenario;
using loomc::bench::CreateBenchmarkKernelSource;
using loomc::bench::CreateWorkspace;
using loomc::bench::DeserializeSource;
using loomc::bench::loom_allocator;
using loomc::bench::ModulePtr;
using loomc::bench::ReadArtifactPrefix;
using loomc::bench::RequireSucceededResult;
using loomc::bench::ResultPtr;
using loomc::bench::RunCompileBenchmark;
using loomc::bench::RunCompileBenchmarkDirect;
using loomc::bench::SourcePtr;
using loomc::bench::TargetCompileScenario;
using loomc::bench::TargetEnvironmentPtr;
using loomc::bench::TargetProfilePtr;
using loomc::bench::to_iree_status;
using loomc::bench::ValidateArtifact;
using loomc::bench::WorkspacePtr;

constexpr uint32_t kSpirvMagic = 0x07230203u;

enum class ModuleMaterializationMode {
  kParseSource,
  kCloneTemplate,
};

class SpirvScenarioBase : public TargetCompileScenario {
 protected:
  explicit SpirvScenarioBase(iree_host_size_t workspace_block_size = 0)
      : TargetCompileScenario(workspace_block_size) {}

  iree_status_t SetUpSpirv(iree_host_size_t worker_count) {
    TargetEnvironmentPtr target_environment;
    loomc_target_environment_t* raw_target_environment = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_spirv(
        loom_allocator(), &raw_target_environment)));
    target_environment.reset(raw_target_environment);

    const loomc_spirv_limit_fact_t limit_facts[] = {
        {
            /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
            /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
            /*.value=*/256,
            /*.provenance=*/loomc_make_cstring_view("benchmark profile"),
        },
        {
            /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
            /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
            /*.value=*/256,
            /*.provenance=*/loomc_make_cstring_view("benchmark profile"),
        },
        {
            /*.limit=*/LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
            /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
            /*.value=*/32,
            /*.provenance=*/loomc_make_cstring_view("benchmark profile"),
        },
    };
    const loomc_spirv_environment_fact_t environment_facts[] = {
        {
            /*.environment=*/LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
            /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
            /*.value=*/LOOMC_SPIRV_VERSION_1_6,
            /*.provenance=*/loomc_make_cstring_view("benchmark profile"),
        },
    };
    loomc_spirv_profile_options_t profile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
        /*.structure_size=*/sizeof(profile_options),
        /*.next=*/nullptr,
        /*.identifier=*/loomc_make_cstring_view("benchmark-vulkan13"),
        /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
        /*.feature_facts=*/nullptr,
        /*.feature_fact_count=*/0,
        /*.limit_facts=*/limit_facts,
        /*.limit_fact_count=*/IREE_ARRAYSIZE(limit_facts),
        /*.environment_facts=*/environment_facts,
        /*.environment_fact_count=*/IREE_ARRAYSIZE(environment_facts),
        /*.cooperative_matrix_rows=*/nullptr,
        /*.cooperative_matrix_row_count=*/0,
        /*.cooperative_vector_rows=*/nullptr,
        /*.cooperative_vector_row_count=*/0,
    };
    loomc_target_profile_t* raw_profile = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_target_profile_create_spirv(
        target_environment.get(), &profile_options, loom_allocator(),
        &raw_profile, &raw_result));
    TargetProfilePtr target_profile(raw_profile);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "SPIR-V profile preparation"));

    return SetUpTarget(worker_count, std::move(target_environment),
                       std::move(target_profile),
                       loomc_make_cstring_view("benchmark-spirv-prepared-low"));
  }

  iree_status_t EmitSpirvArtifact(WorkspacePtr& workspace, ModulePtr& module,
                                  loomc_string_view_t identifier) {
    loomc_spirv_emit_options_t spirv_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(spirv_options),
        /*.next=*/nullptr,
    };
    loomc_emit_options_t emit_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(emit_options),
        /*.next=*/&spirv_options,
        /*.artifact_format=*/
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
        /*.identifier=*/identifier,
        /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };

    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_emit_module(target_environment(), workspace.get(), module.get(),
                          &emit_options, loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "SPIR-V emission"));

    int64_t artifact_bytes = 0;
    IREE_RETURN_IF_ERROR(ValidateArtifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV), sizeof(uint32_t),
        "SPIR-V executable", &artifact_bytes));

    const loomc_artifact_t* artifact = loomc::bench::FindArtifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV));
    uint32_t magic = 0;
    IREE_RETURN_IF_ERROR(ReadArtifactPrefix(
        artifact, iree_make_byte_span(&magic, sizeof(magic))));
    ::benchmark::DoNotOptimize(magic);
    if (magic != kSpirvMagic) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "SPIR-V executable has magic 0x%08x", magic);
    }

    RecordArtifactBytes(artifact_bytes);
    return iree_ok_status();
  }
};

class SpirvTunerFlowScenario final : public SpirvScenarioBase {
 public:
  SpirvTunerFlowScenario(iree_host_size_t job_count,
                         ModuleMaterializationMode materialization_mode)
      : job_count_(job_count), materialization_mode_(materialization_mode) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpSpirv(worker_count));
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("i32_memory_chain.loom"), &source_));
    if (materialization_mode_ == ModuleMaterializationMode::kCloneTemplate) {
      IREE_RETURN_IF_ERROR(
          CreateWorkspace(/*block_size=*/0, &template_workspace_));
      IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(),
                                             template_workspace_.get(),
                                             source_.get(), &template_module_));
    }
    return iree_ok_status();
  }

  iree_host_size_t job_count() const override { return job_count_; }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    WorkspacePtr& workspace = workspace_at(worker_ordinal);

    ModulePtr module;
    if (materialization_mode_ == ModuleMaterializationMode::kCloneTemplate) {
      IREE_RETURN_IF_ERROR(
          CloneModule(template_module_.get(), workspace.get(), &module));
    } else {
      IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(), workspace.get(),
                                             source_.get(), &module));
    }

    char workgroup_size_value[16] = {0};
    std::snprintf(workgroup_size_value, sizeof(workgroup_size_value), "%d",
                  1 + (int)(job_ordinal % 64));
    loomc_config_binding_t bindings[] = {
        {
            /*.key=*/loomc_make_cstring_view("@benchmark.workgroup_size"),
            /*.value=*/loomc_make_cstring_view(workgroup_size_value),
        },
        {
            /*.key=*/loomc_make_cstring_view("@benchmark.operation_count"),
            /*.value=*/loomc_make_cstring_view("1"),
        },
    };
    loomc_config_options_t config_options = {
        /*.bindings=*/bindings,
        /*.binding_count=*/IREE_ARRAYSIZE(bindings),
        /*.json_object=*/loomc_string_view_empty(),
        /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
    };

    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("i32_memory_chain"),
        loomc_make_cstring_view("spirv_tuner_kernel"), config_options));
    return EmitSpirvArtifact(workspace, module,
                             loomc_make_cstring_view("i32_memory_chain.spv"));
  }

 private:
  iree_host_size_t job_count_ = 0;
  ModuleMaterializationMode materialization_mode_;
  SourcePtr source_;
  WorkspacePtr template_workspace_;
  ModulePtr template_module_;
};

class SpirvI32ChainScenario final : public SpirvScenarioBase {
 public:
  SpirvI32ChainScenario(iree_host_size_t job_count,
                        iree_host_size_t operation_count,
                        ModuleMaterializationMode materialization_mode,
                        iree_host_size_t workspace_block_size = 0)
      : SpirvScenarioBase(workspace_block_size),
        job_count_(job_count),
        operation_count_(std::max<iree_host_size_t>(operation_count, 1)),
        operation_count_value_(std::to_string(operation_count_)),
        materialization_mode_(materialization_mode) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpSpirv(worker_count));
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("i32_memory_chain.loom"), &source_));
    if (materialization_mode_ == ModuleMaterializationMode::kCloneTemplate) {
      IREE_RETURN_IF_ERROR(
          CreateWorkspace(/*block_size=*/0, &template_workspace_));
      IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(),
                                             template_workspace_.get(),
                                             source_.get(), &template_module_));
    }
    return iree_ok_status();
  }

  iree_host_size_t job_count() const override { return job_count_; }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    WorkspacePtr& workspace = workspace_at(worker_ordinal);

    ModulePtr module;
    if (materialization_mode_ == ModuleMaterializationMode::kCloneTemplate) {
      IREE_RETURN_IF_ERROR(
          CloneModule(template_module_.get(), workspace.get(), &module));
    } else {
      IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(), workspace.get(),
                                             source_.get(), &module));
    }

    (void)job_ordinal;
    loomc_config_binding_t bindings[] = {
        {
            /*.key=*/loomc_make_cstring_view("@benchmark.workgroup_size"),
            /*.value=*/loomc_make_cstring_view("64"),
        },
        {
            /*.key=*/loomc_make_cstring_view("@benchmark.operation_count"),
            /*.value=*/
            loomc_make_string_view(operation_count_value_.data(),
                                   operation_count_value_.size()),
        },
    };
    loomc_config_options_t config_options = {
        /*.bindings=*/bindings,
        /*.binding_count=*/IREE_ARRAYSIZE(bindings),
        /*.json_object=*/loomc_string_view_empty(),
        /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
    };

    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("i32_memory_chain"),
        loomc_make_cstring_view("spirv_i32_chain"), config_options));
    return EmitSpirvArtifact(workspace, module,
                             loomc_make_cstring_view("i32_chain.spv"));
  }

  void SetExtraCounters(::benchmark::State& state) const override {
    state.counters["operation_count"] = (double)operation_count_;
  }

 private:
  iree_host_size_t job_count_ = 0;
  iree_host_size_t operation_count_ = 0;
  std::string operation_count_value_;
  ModuleMaterializationMode materialization_mode_;
  SourcePtr source_;
  WorkspacePtr template_workspace_;
  ModulePtr template_module_;
};

static std::unique_ptr<CompileScenario> CreateSpirvTunerFlowParseScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvTunerFlowScenario>(
      (iree_host_size_t)state.range(1),
      ModuleMaterializationMode::kParseSource);
}

static std::unique_ptr<CompileScenario> CreateSpirvTunerFlowCloneScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvTunerFlowScenario>(
      (iree_host_size_t)state.range(1),
      ModuleMaterializationMode::kCloneTemplate);
}

static std::unique_ptr<CompileScenario> CreateSpirvI32ChainParseScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvI32ChainScenario>(
      (iree_host_size_t)state.range(1), (iree_host_size_t)state.range(2),
      ModuleMaterializationMode::kParseSource);
}

static std::unique_ptr<CompileScenario> CreateSpirvI32ChainCloneScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvI32ChainScenario>(
      (iree_host_size_t)state.range(1), (iree_host_size_t)state.range(2),
      ModuleMaterializationMode::kCloneTemplate);
}

static std::unique_ptr<CompileScenario> CreateSpirvI32ChainWorkspaceScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvI32ChainScenario>(
      (iree_host_size_t)state.range(1), (iree_host_size_t)state.range(2),
      ModuleMaterializationMode::kCloneTemplate,
      (iree_host_size_t)state.range(3));
}

static void BM_SpirvTunerFlowParseSmoke(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateSpirvTunerFlowParseScenario, nullptr);
}
BENCHMARK(BM_SpirvTunerFlowParseSmoke)
    ->Args({1, 2})
    ->Args({2, 4})
    ->UseRealTime();

static void BM_SpirvTunerFlowCloneSmoke(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateSpirvTunerFlowCloneScenario, nullptr);
}
BENCHMARK(BM_SpirvTunerFlowCloneSmoke)
    ->Args({1, 2})
    ->Args({2, 4})
    ->UseRealTime();

static void BM_SpirvTunerFlowParseDirect(::benchmark::State& state) {
  RunCompileBenchmarkDirect(state, CreateSpirvTunerFlowParseScenario, nullptr);
}
BENCHMARK(BM_SpirvTunerFlowParseDirect)
    ->Args({1, 1})
    ->Args({1, 8})
    ->Args({1, 16})
    ->UseRealTime();

static void BM_SpirvTunerFlowCloneDirect(::benchmark::State& state) {
  RunCompileBenchmarkDirect(state, CreateSpirvTunerFlowCloneScenario, nullptr);
}
BENCHMARK(BM_SpirvTunerFlowCloneDirect)
    ->Args({1, 1})
    ->Args({1, 8})
    ->Args({1, 16})
    ->UseRealTime();

static void BM_SpirvTunerFlowParse(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateSpirvTunerFlowParseScenario, nullptr);
}
BENCHMARK(BM_SpirvTunerFlowParse)
    ->Args({1, 16})
    ->Args({2, 32})
    ->Args({4, 64})
    ->Args({8, 128})
    ->Args({16, 256})
    ->Args({32, 512})
    ->Args({64, 1024})
    ->Args({96, 1536})
    ->UseRealTime();

static void BM_SpirvTunerFlowClone(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateSpirvTunerFlowCloneScenario, nullptr);
}
BENCHMARK(BM_SpirvTunerFlowClone)
    ->Args({1, 16})
    ->Args({2, 32})
    ->Args({4, 64})
    ->Args({8, 128})
    ->Args({16, 256})
    ->Args({32, 512})
    ->Args({64, 1024})
    ->Args({96, 1536})
    ->UseRealTime();

static void BM_SpirvI32ChainParseSmoke(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateSpirvI32ChainParseScenario, nullptr);
}
BENCHMARK(BM_SpirvI32ChainParseSmoke)->Args({1, 2, 16})->UseRealTime();

static void BM_SpirvI32ChainCloneSmoke(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateSpirvI32ChainCloneScenario, nullptr);
}
BENCHMARK(BM_SpirvI32ChainCloneSmoke)->Args({2, 4, 64})->UseRealTime();

static void BM_SpirvI32ChainParseDirect(::benchmark::State& state) {
  RunCompileBenchmarkDirect(state, CreateSpirvI32ChainParseScenario, nullptr);
}
BENCHMARK(BM_SpirvI32ChainParseDirect)
    ->Args({1, 1, 1})
    ->Args({1, 1, 16})
    ->Args({1, 1, 64})
    ->Args({1, 1, 256})
    ->Args({1, 1, 1024})
    ->UseRealTime();

static void BM_SpirvI32ChainCloneDirect(::benchmark::State& state) {
  RunCompileBenchmarkDirect(state, CreateSpirvI32ChainCloneScenario, nullptr);
}
BENCHMARK(BM_SpirvI32ChainCloneDirect)
    ->Args({1, 1, 1})
    ->Args({1, 1, 16})
    ->Args({1, 1, 64})
    ->Args({1, 1, 256})
    ->Args({1, 1, 1024})
    ->UseRealTime();

static void BM_SpirvI32ChainParse(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateSpirvI32ChainParseScenario, nullptr);
}
BENCHMARK(BM_SpirvI32ChainParse)
    ->Args({1, 16, 1})
    ->Args({1, 16, 64})
    ->Args({1, 16, 256})
    ->Args({32, 512, 1})
    ->Args({32, 512, 64})
    ->Args({32, 512, 256})
    ->Args({96, 1536, 1})
    ->Args({96, 1536, 64})
    ->Args({96, 1536, 256})
    ->Args({96, 384, 1024})
    ->UseRealTime();

static void BM_SpirvI32ChainClone(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateSpirvI32ChainCloneScenario, nullptr);
}
BENCHMARK(BM_SpirvI32ChainClone)
    ->Args({1, 16, 1})
    ->Args({1, 16, 64})
    ->Args({1, 16, 256})
    ->Args({32, 512, 1})
    ->Args({32, 512, 64})
    ->Args({32, 512, 256})
    ->Args({96, 1536, 1})
    ->Args({96, 1536, 64})
    ->Args({96, 1536, 256})
    ->Args({96, 384, 1024})
    ->UseRealTime();

static void BM_SpirvI32ChainWorkspaceSmoke(::benchmark::State& state) {
  RunCompileBenchmarkDirect(state, CreateSpirvI32ChainWorkspaceScenario,
                            nullptr);
}
BENCHMARK(BM_SpirvI32ChainWorkspaceSmoke)
    ->ArgsProduct({{1}, {2}, {16}, {32 * 1024, 64 * 1024, 128 * 1024}})
    ->UseRealTime();

static void BM_SpirvI32ChainWorkspaceDirect(::benchmark::State& state) {
  RunCompileBenchmarkDirect(state, CreateSpirvI32ChainWorkspaceScenario,
                            nullptr);
}
BENCHMARK(BM_SpirvI32ChainWorkspaceDirect)
    ->ArgsProduct(
        {{1}, {16}, {1, 64, 1024}, {32 * 1024, 64 * 1024, 128 * 1024}})
    ->UseRealTime();

}  // namespace
