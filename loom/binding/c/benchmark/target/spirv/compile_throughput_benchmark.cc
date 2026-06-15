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
#include <sstream>
#include <string>

#include "benchmark/benchmark.h"
#include "loomc/target/spirv.h"

namespace {

using loomc::bench::CloneModule;
using loomc::bench::CompileScenario;
using loomc::bench::CreateBenchmarkKernelSource;
using loomc::bench::CreateTextSource;
using loomc::bench::CreateWorkspace;
using loomc::bench::DeserializeSource;
using loomc::bench::loom_allocator;
using loomc::bench::ModulePtr;
using loomc::bench::PassProgramPtr;
using loomc::bench::RequireSucceededResult;
using loomc::bench::ResultPtr;
using loomc::bench::RunCompileBenchmark;
using loomc::bench::RunCompileBenchmarkDirect;
using loomc::bench::SourcePtr;
using loomc::bench::TargetEnvironmentPtr;
using loomc::bench::TargetProfilePtr;
using loomc::bench::TargetSelectionPtr;
using loomc::bench::to_iree_status;
using loomc::bench::ValidateArtifact;
using loomc::bench::WorkspacePtr;

constexpr uint32_t kSpirvMagic = 0x07230203u;

enum class ModuleMaterializationMode {
  kParseSource,
  kCloneTemplate,
};

static std::string MakeSpirvI32ChainSource(iree_host_size_t operation_count) {
  operation_count = std::max<iree_host_size_t>(operation_count, 1);

  std::ostringstream builder;
  builder
      << "spirv.target<vulkan1_3> @target {abi = hal_kernel}\n"
      << "\n"
      << "config.decl @tuner.workgroup_size : %value: index where "
         "[range(%value, 1, 256)]\n"
      << "\n"
      << "kernel.def target(@target) @i32_chain() {\n"
      << "  %unit = index.constant 1 : index\n"
      << "  %workgroup_size_x = config.get @tuner.workgroup_size : index\n"
      << "  kernel.launch.config workgroups(%unit, %unit, %unit) "
         "workgroup_size(%workgroup_size_x, %unit, %unit) : index\n"
      << "} launch(%input: buffer, %output: buffer, %byte_offset: offset) {\n"
      << "  %byte_offset_aligned = index.assume %byte_offset "
         "[mul(%byte_offset, 4)] : offset\n"
      << "  %input_aligned = buffer.assume.alignment %input "
         "{minimum_alignment = 4} : buffer\n"
      << "  %output_aligned = buffer.assume.alignment %output "
         "{minimum_alignment = 4} : buffer\n"
      << "  %input_view = buffer.view %input_aligned[%byte_offset_aligned] : "
         "buffer -> view<1xi32, #dense>\n"
      << "  %loaded = view.load %input_view[0] : view<1xi32, #dense> -> "
         "i32\n"
      << "  %acc0 = scalar.addi %loaded, %loaded : i32\n";
  for (iree_host_size_t i = 1; i < operation_count; ++i) {
    builder << "  %acc" << i << " = scalar.addi %acc" << (i - 1)
            << ", %loaded : i32\n";
  }
  builder
      << "  %output_view = buffer.view %output_aligned[%byte_offset_aligned] : "
         "buffer -> view<1xi32, #dense>\n"
      << "  view.store %acc" << (operation_count - 1)
      << ", %output_view[0] : i32, view<1xi32, #dense>\n"
      << "  kernel.return\n"
      << "}\n";
  return builder.str();
}

class SpirvScenarioBase : public CompileScenario {
 protected:
  iree_status_t SetUpSpirv(iree_host_size_t worker_count) {
    IREE_RETURN_IF_ERROR(CreateTargetEnvironmentAndContext());
    IREE_RETURN_IF_ERROR(CreateTargetProfileAndSelection());
    IREE_RETURN_IF_ERROR(CreateCompilerAndTargetPipeline());
    return SetUpWorkerSlots(worker_count);
  }

  iree_status_t CompileModuleToPreparedLow(WorkspacePtr& workspace,
                                           ModulePtr& module,
                                           loomc_string_view_t module_name,
                                           loomc_config_options_t config) {
    loomc_target_selection_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_selection=*/target_selection_.get(),
    };
    loomc_compile_options_t compile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(compile_options),
        /*.next=*/&target_options,
        /*.module_name=*/module_name,
        /*.artifact_flags=*/0,
        /*.config=*/config,
    };

    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_compile_module(
        compiler_.get(), workspace.get(), pass_program_.get(), module.get(),
        &compile_options, loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(result.get(), "compilation"));
    return iree_ok_status();
  }

  iree_status_t EmitSpirvArtifact(WorkspacePtr& workspace, ModulePtr& module,
                                  loomc_string_view_t identifier) {
    loomc_target_selection_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_selection=*/target_selection_.get(),
    };
    loomc_spirv_emit_options_t spirv_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(spirv_options),
        /*.next=*/&target_options,
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
    iree_status_t status = to_iree_status(loomc_emit_module(
        target_environment_.get(), workspace.get(), module.get(), &emit_options,
        loom_allocator(), &raw_result));
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
    std::memcpy(&magic, artifact->contents.data, sizeof(magic));
    ::benchmark::DoNotOptimize(magic);
    if (magic != kSpirvMagic) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "SPIR-V executable has magic 0x%08x", magic);
    }

    RecordArtifactBytes(artifact_bytes);
    return iree_ok_status();
  }

 private:
  iree_status_t CreateTargetEnvironmentAndContext() {
    loomc_target_environment_t* target_environment = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_spirv(
        loom_allocator(), &target_environment)));
    target_environment_.reset(target_environment);

    loomc_context_target_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_environment=*/target_environment_.get(),
    };
    loomc_context_options_t context_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        /*.structure_size=*/sizeof(context_options),
        /*.next=*/&target_options,
    };
    loomc_context_t* context = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(
        loomc_context_create(&context_options, loom_allocator(), &context)));
    context_.reset(context);
    return iree_ok_status();
  }

  iree_status_t CreateTargetProfileAndSelection() {
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
        target_environment_.get(), &profile_options, loom_allocator(),
        &raw_profile, &raw_result));
    TargetProfilePtr profile(raw_profile);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "SPIR-V profile preparation"));
    target_profile_.reset(profile.release());

    loomc_target_selection_t* raw_selection = nullptr;
    IREE_RETURN_IF_ERROR(
        to_iree_status(loomc_target_selection_create_from_profile(
            target_profile_.get(), loom_allocator(), &raw_selection)));
    target_selection_.reset(raw_selection);
    return iree_ok_status();
  }

  iree_status_t CreateCompilerAndTargetPipeline() {
    loomc_compiler_t* compiler = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_compiler_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &compiler)));
    compiler_.reset(compiler);

    loomc_target_selection_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_selection=*/target_selection_.get(),
    };
    loomc_target_pipeline_options_t pipeline_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        /*.structure_size=*/sizeof(pipeline_options),
        /*.next=*/&target_options,
        /*.identifier=*/
        loomc_make_cstring_view("benchmark-spirv-prepared-low"),
        /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        /*.source_to_low_max_errors=*/20,
    };

    loomc_pass_program_t* raw_pass_program = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status =
        to_iree_status(loomc_pass_program_create_from_target_pipeline(
            context_.get(), &pipeline_options, loom_allocator(),
            &raw_pass_program, &raw_result));
    PassProgramPtr pass_program(raw_pass_program);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "target pipeline preparation"));
    pass_program_.reset(pass_program.release());
    return iree_ok_status();
  }

  TargetEnvironmentPtr target_environment_;
  TargetProfilePtr target_profile_;
  TargetSelectionPtr target_selection_;
};

class SpirvTunerFlowScenario final : public SpirvScenarioBase {
 public:
  SpirvTunerFlowScenario(iree_host_size_t job_count,
                         ModuleMaterializationMode materialization_mode)
      : job_count_(job_count), materialization_mode_(materialization_mode) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpSpirv(worker_count));
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("double_i32_at_byte_offset.loom"), &source_));
    if (materialization_mode_ == ModuleMaterializationMode::kCloneTemplate) {
      IREE_RETURN_IF_ERROR(CreateWorkspace(&template_workspace_));
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
            /*.key=*/loomc_make_cstring_view("@tuner.workgroup_size"),
            /*.value=*/loomc_make_cstring_view(workgroup_size_value),
        },
    };
    loomc_config_options_t config_options = {
        /*.bindings=*/bindings,
        /*.binding_count=*/IREE_ARRAYSIZE(bindings),
        /*.json_object=*/loomc_string_view_empty(),
        /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
            LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
    };

    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("spirv_tuner_kernel"),
        config_options));
    return EmitSpirvArtifact(
        workspace, module,
        loomc_make_cstring_view("double_i32_at_byte_offset.spv"));
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
                        ModuleMaterializationMode materialization_mode)
      : job_count_(job_count),
        operation_count_(std::max<iree_host_size_t>(operation_count, 1)),
        materialization_mode_(materialization_mode) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpSpirv(worker_count));
    source_text_ = MakeSpirvI32ChainSource(operation_count_);
    IREE_RETURN_IF_ERROR(
        CreateTextSource("i32_chain.loom", source_text_, &source_));
    if (materialization_mode_ == ModuleMaterializationMode::kCloneTemplate) {
      IREE_RETURN_IF_ERROR(CreateWorkspace(&template_workspace_));
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
            /*.key=*/loomc_make_cstring_view("@tuner.workgroup_size"),
            /*.value=*/loomc_make_cstring_view(workgroup_size_value),
        },
    };
    loomc_config_options_t config_options = {
        /*.bindings=*/bindings,
        /*.binding_count=*/IREE_ARRAYSIZE(bindings),
        /*.json_object=*/loomc_string_view_empty(),
        /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
            LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
    };

    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("spirv_i32_chain"),
        config_options));
    return EmitSpirvArtifact(workspace, module,
                             loomc_make_cstring_view("i32_chain.spv"));
  }

  void SetExtraCounters(::benchmark::State& state) const override {
    state.counters["operation_count"] = (double)operation_count_;
    state.counters["source_bytes"] = (double)source_text_.size();
  }

 private:
  iree_host_size_t job_count_ = 0;
  iree_host_size_t operation_count_ = 0;
  ModuleMaterializationMode materialization_mode_;
  std::string source_text_;
  SourcePtr source_;
  WorkspacePtr template_workspace_;
  ModulePtr template_module_;
};

static std::unique_ptr<CompileScenario> CreateSpirvTunerFlowParseScenario(
    const ::benchmark::State& state, void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvTunerFlowScenario>(
      (iree_host_size_t)state.range(1),
      ModuleMaterializationMode::kParseSource);
}

static std::unique_ptr<CompileScenario> CreateSpirvTunerFlowCloneScenario(
    const ::benchmark::State& state, void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvTunerFlowScenario>(
      (iree_host_size_t)state.range(1),
      ModuleMaterializationMode::kCloneTemplate);
}

static std::unique_ptr<CompileScenario> CreateSpirvI32ChainParseScenario(
    const ::benchmark::State& state, void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvI32ChainScenario>(
      (iree_host_size_t)state.range(1), (iree_host_size_t)state.range(2),
      ModuleMaterializationMode::kParseSource);
}

static std::unique_ptr<CompileScenario> CreateSpirvI32ChainCloneScenario(
    const ::benchmark::State& state, void* user_data) {
  (void)user_data;
  return std::make_unique<SpirvI32ChainScenario>(
      (iree_host_size_t)state.range(1), (iree_host_size_t)state.range(2),
      ModuleMaterializationMode::kCloneTemplate);
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

}  // namespace
