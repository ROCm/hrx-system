// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Measures launch-config compilation and runtime costs. Paired compile rows
// start from the same retained one-kernel IR and differ only in whether the
// compiler captures, lowers, serializes, and publishes the VM companion
// artifact. Warm rows compare the public VM path with the current retained-IR
// value-facts evaluator using the same function and complete result contract.

#include <cstdint>
#include <string>
#include <utility>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/status_cc.h"
#include "loom/analysis/kernel_launch_config.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"
#include "loom/binding/c/benchmark/util/benchmark_support.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/function_contract.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"
#include "loom/tooling/compile/pipeline.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/vm.h"

namespace {

using loomc::bench::CloneModule;
using loomc::bench::CompilerPtr;
using loomc::bench::ContextPtr;
using loomc::bench::CreateBenchmarkKernelSource;
using loomc::bench::CreateTextSource;
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
using loomc::bench::ValidateArtifact;
using loomc::bench::WorkspacePtr;

using LaunchConfigProgramPtr =
    loomc::bench::HandlePtr<loomc_vm_launch_config_program_t,
                            loomc_vm_launch_config_program_release>;

static iree_status_t CreateAmdgpuTargetEnvironment(
    TargetEnvironmentPtr* out_target_environment) {
  out_target_environment->reset();
  loomc_target_environment_t* target_environment = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_amdgpu(
      loom_allocator(), &target_environment)));
  out_target_environment->reset(target_environment);
  return iree_ok_status();
}

static iree_status_t ValidateLaunchConfigArtifact(const loomc_result_t* result,
                                                  int64_t* out_artifact_bytes) {
  return ValidateArtifact(
      result, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM_BYTECODE),
      /*minimum_data_length=*/1, "launch-config VM bytecode",
      out_artifact_bytes);
}

static iree_status_t ValidateExpectedLaunchConfig(
    const loomc_launch_config_t* config) {
  if (config->workgroup_count.x != 4096 || config->workgroup_count.y != 1 ||
      config->workgroup_count.z != 1 || config->workgroup_size.x != 256 ||
      config->workgroup_size.y != 1 || config->workgroup_size.z != 1 ||
      config->workgroup_cluster_size.x != 1 ||
      config->workgroup_cluster_size.y != 1 ||
      config->workgroup_cluster_size.z != 1 || config->subgroup_size != 32 ||
      config->workgroup_storage_bytes != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "launch-config evaluation produced an unexpected result");
  }
  return iree_ok_status();
}

static std::string BuildLaunchConfigScaleSource(
    iree_host_size_t function_count) {
  std::string source = "amdgpu.target<gfx1100> @gfx1100\n\n";
  source.reserve(source.size() + function_count * 640);
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    source.append("kernel.def target(@gfx1100) @scale_");
    source.append(std::to_string(i));
    source.append(R"((%element_count: index) {
  %c1 = index.constant 1 : index
  %rounding = index.constant 255 : index
  %workgroup_size = index.constant 256 : index
  %rounded_count = index.add %element_count, %rounding : index
  %workgroup_count = index.div %rounded_count, %workgroup_size : index
  kernel.launch.config workgroups(%workgroup_count, %c1, %c1) workgroup_size(%workgroup_size, %c1, %c1) : index
} launch() {
  kernel.return
}

)");
  }
  return source;
}

class LaunchConfigBenchmarkFixture {
 public:
  iree_status_t SetUpCompiler(loomc_string_view_t source_identifier) {
    IREE_RETURN_IF_ERROR(SetUpCompilerInfrastructure());
    IREE_RETURN_IF_ERROR(
        CreateBenchmarkKernelSource(source_identifier, &source_));
    return DeserializeTemplateModule();
  }

  iree_status_t SetUpScaleCompiler(iree_host_size_t function_count) {
    IREE_RETURN_IF_ERROR(SetUpCompilerInfrastructure());
    const std::string source_text =
        BuildLaunchConfigScaleSource(function_count);
    IREE_RETURN_IF_ERROR(
        CreateTextSource("launch_config_scale.loom", source_text, &source_));
    return DeserializeTemplateModule();
  }

  iree_status_t Compile(loomc_compile_artifact_flags_t artifact_flags,
                        ResultPtr* out_result) {
    out_result->reset();
    ModulePtr module;
    IREE_RETURN_IF_ERROR(
        CloneModule(template_module_.get(), workspace_.get(), &module));
    const loomc_compile_options_t compile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(compile_options),
        /*.next=*/nullptr,
        /*.module_name=*/loomc_make_cstring_view("launch_config_benchmark"),
        /*.artifact_flags=*/artifact_flags,
    };
    loomc_result_t* compile_result = nullptr;
    iree_status_t status = to_iree_status(loomc_compile_module(
        compiler_.get(), workspace_.get(), pass_program_.get(), module.get(),
        &compile_options, loom_allocator(), &compile_result));
    ResultPtr compile_result_owner(compile_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(compile_result_owner.get(),
                                                "module compilation"));
    out_result->reset(compile_result_owner.release());
    return iree_ok_status();
  }

  iree_status_t SetUpArtifact(loomc_string_view_t source_identifier) {
    IREE_RETURN_IF_ERROR(SetUpCompiler(source_identifier));
    return CompileArtifact();
  }

  iree_status_t SetUpScaleArtifact(iree_host_size_t function_count) {
    IREE_RETURN_IF_ERROR(SetUpScaleCompiler(function_count));
    return CompileArtifact();
  }

  iree_status_t LoadProgram(const loomc_artifact_t* artifact,
                            LaunchConfigProgramPtr* out_program) const {
    out_program->reset();
    loomc_vm_launch_config_program_t* program = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_vm_launch_config_program_load(
        artifact, loom_allocator(), &program)));
    out_program->reset(program);
    return iree_ok_status();
  }

  iree_status_t LoadProgram(LaunchConfigProgramPtr* out_program) const {
    return LoadProgram(artifact_, out_program);
  }

  iree_host_size_t artifact_byte_length() const {
    return artifact_ != nullptr
               ? loomc_byte_sequence_length(artifact_->contents)
               : 0;
  }

  iree_status_t SetUpInvocation(iree_string_view_t source_identifier,
                                iree_string_view_t function_name) {
    IREE_RETURN_IF_ERROR(
        SetUpArtifact(loomc_string_view_from_iree(source_identifier)));
    IREE_RETURN_IF_ERROR(LoadProgram(&program_));

    IREE_RETURN_IF_ERROR(LookupFunction(function_name, &function_));
    IREE_RETURN_IF_ERROR(Invoke(UINT64_C(1048576), &warm_config_));
    return ValidateExpectedLaunchConfig(&warm_config_);
  }

  iree_status_t LookupFunction(
      iree_string_view_t function_name,
      loomc_vm_launch_config_function_t* out_function) const {
    return to_iree_status(loomc_vm_launch_config_program_lookup_function(
        program_.get(), loomc_string_view_from_iree(function_name),
        out_function));
  }

  iree_status_t Invoke(uint64_t element_count,
                       loomc_launch_config_t* out_config) {
    return to_iree_status(loomc_vm_launch_config_program_invoke(
        program_.get(), function_, &element_count, /*argument_count=*/1,
        out_config));
  }

 private:
  iree_status_t SetUpCompilerInfrastructure() {
    IREE_RETURN_IF_ERROR(CreateAmdgpuTargetEnvironment(&target_environment_));

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
    return iree_ok_status();
  }

  iree_status_t DeserializeTemplateModule() {
    return DeserializeSource(context_.get(), workspace_.get(), source_.get(),
                             &template_module_);
  }

  iree_status_t CompileArtifact() {
    IREE_RETURN_IF_ERROR(
        Compile(LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG, &compile_result_));
    artifact_ = FindArtifact(
        compile_result_.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM_BYTECODE));
    if (artifact_ == nullptr) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "launch-config artifact was not produced");
    }
    return iree_ok_status();
  }

  // Immutable target provider composition shared by compilation.
  TargetEnvironmentPtr target_environment_;

  // Context owning target-aware source and type registrations.
  ContextPtr context_;

  // Prepared compiler reused across timed compilation iterations.
  CompilerPtr compiler_;

  // Immutable target pipeline reused across compilation iterations.
  PassProgramPtr pass_program_;

  // Reusable arena block pool for compilation-local module storage.
  WorkspacePtr workspace_;

  // Immutable source containing the benchmark module.
  SourcePtr source_;

  // Immutable parsed source cloned for each compilation iteration.
  ModulePtr template_module_;

  // Compile result retaining the launch-config artifact descriptor and bytes.
  ResultPtr compile_result_;

  // Launch-config artifact borrowed from |compile_result_|.
  const loomc_artifact_t* artifact_ = nullptr;

  // Loaded VM program used by warm invocation benchmarks.
  LaunchConfigProgramPtr program_;

  // Resolved exported function used by warm invocation benchmarks.
  loomc_vm_launch_config_function_t function_ =
      loomc_vm_launch_config_function_invalid();

  // Result storage used to validate setup before timing begins.
  loomc_launch_config_t warm_config_ = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(loomc_launch_config_t),
  };
};

// Owns the retained source IR and cached target contract used by the current
// value-facts launch-config evaluator. Setup mirrors the production HAL
// testbench path; only repeated evaluation is timed.
class ValueFactsLaunchConfigBenchmarkFixture {
 public:
  ValueFactsLaunchConfigBenchmarkFixture() {
    iree_arena_block_pool_initialize(/*total_block_size=*/32 * 1024,
                                     iree_allocator_system(), &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  ~ValueFactsLaunchConfigBenchmarkFixture() {
    loom_module_free(module_);
    iree_arena_deinitialize(&analysis_arena_);
    loom_target_environment_deinitialize(&target_environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t SetUp(loomc_string_view_t source_identifier) {
    IREE_RETURN_IF_ERROR(loom_target_environment_initialize(
        &loom_amdgpu_target_provider_set, &target_environment_));
    IREE_RETURN_IF_ERROR(
        loom_target_environment_initialize_low_descriptor_registry(
            &target_environment_, &low_descriptor_registry_));
    loom_amdgpu_target_identity_t target_identity = {};
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_artifact_key_parse(IREE_SV("gfx1100"), &target_identity));
    IREE_RETURN_IF_ERROR(loom_amdgpu_target_profile_initialize(
        &target_identity, &target_profile_));
    IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(&context_));
    IREE_RETURN_IF_ERROR(loom_target_environment_register_context(
        &target_environment_, &context_));
    IREE_RETURN_IF_ERROR(loom_context_finalize(&context_));

    IREE_RETURN_IF_ERROR(
        CreateBenchmarkKernelSource(source_identifier, &source_));
    const iree_const_byte_span_t contents =
        iree_const_byte_span_from_loomc(loomc_source_contents(source_.get()));
    const iree_string_view_t identifier =
        iree_string_view_from_loomc(loomc_source_identifier(source_.get()));
    const loom_text_parse_options_t parse_options = {};
    IREE_RETURN_IF_ERROR(loom_text_parse(
        iree_make_string_view(reinterpret_cast<const char*>(contents.data),
                              contents.data_length),
        identifier, &context_, &block_pool_, &parse_options, &module_));
    if (module_ == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "benchmark launch-config source did not parse");
    }

    loom_symbol_fact_table_initialize(&symbol_facts_, &analysis_arena_);
    const loom_string_id_t function_name_id =
        loom_module_lookup_string(module_, IREE_SV("unchecked_1d"));
    if (function_name_id == LOOM_STRING_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "benchmark function name was not interned");
    }
    const loom_symbol_id_t function_symbol_id =
        loom_module_find_symbol(module_, function_name_id);
    if (function_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "benchmark function was not defined");
    }
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(
        &symbol_facts_, module_, function_symbol_id, &base_facts));
    const loom_func_symbol_facts_t* function_facts =
        loom_func_symbol_facts_cast(base_facts);
    if (function_facts == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "benchmark function facts are unavailable");
    }
    bool target_valid = false;
    IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve_facts(
        module_, &symbol_facts_, function_facts,
        /*diagnostic_emitter=*/{}, &analysis_arena_, &target_valid,
        &target_facts_));
    if (!target_valid || target_facts_ == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "benchmark target contract is invalid");
    }

    IREE_RETURN_IF_ERROR(Evaluate(INT64_C(1048576), &warm_config_));
    return ValidateExpectedLaunchConfig(&warm_config_);
  }

  iree_status_t PrepareLegacyReadiness(
      loom_module_t** out_module,
      loom_compile_pipeline_result_t* out_pipeline_result) {
    *out_module = nullptr;
    *out_pipeline_result = {};

    const loom_module_t* source_modules[] = {module_};
    const iree_string_view_t root_symbols[] = {IREE_SV("unchecked_1d")};
    const loom_link_options_t link_options = {
        /*.module_name=*/IREE_SV("legacy_launch_config"),
        /*.root_symbols=*/
        {
            /*.count=*/IREE_ARRAYSIZE(root_symbols),
            /*.values=*/root_symbols,
        },
    };
    loom_module_t* prepared_module = nullptr;
    iree_status_t status = loom_link_materialized_modules(
        source_modules, IREE_ARRAYSIZE(source_modules), &link_options,
        &block_pool_, iree_allocator_system(), &prepared_module);

    loom_compile_pipeline_result_t pipeline_result = {};
    bool pipeline_result_initialized = false;
    if (iree_status_is_ok(status)) {
      const loom_target_specialization_request_t specialization = {
          /*.function_name=*/IREE_SV("unchecked_1d"),
          /*.target_profile=*/&target_profile_.base,
      };
      loom_compile_pipeline_options_t options = {};
      loom_compile_pipeline_options_initialize(&options);
      options.default_pipeline = LOOM_COMPILE_DEFAULT_PIPELINE_EXPANDED_SOURCE;
      options.target_environment = &target_environment_;
      options.target_specializations = {
          /*.values=*/&specialization,
          /*.count=*/1,
      };
      options.low_descriptor_registry = &low_descriptor_registry_;
      status = loom_compile_run_pipeline(prepared_module, &options,
                                         &block_pool_, &pipeline_result);
      pipeline_result_initialized = true;
    }
    if (iree_status_is_ok(status) && pipeline_result.pass.error_count != 0) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "legacy launch-config preparation produced pass diagnostics");
    }

    const loom_target_function_version_t* function_version = nullptr;
    if (iree_status_is_ok(status)) {
      const loom_string_id_t name_id =
          loom_module_lookup_string(prepared_module, IREE_SV("unchecked_1d"));
      const loom_symbol_id_t symbol_id =
          name_id != LOOM_STRING_ID_INVALID
              ? loom_module_find_symbol(prepared_module, name_id)
              : LOOM_SYMBOL_ID_INVALID;
      if (symbol_id != LOOM_SYMBOL_ID_INVALID) {
        function_version = loom_target_function_version_list_find(
            &pipeline_result.function_versions.list,
            loom_func_like_cast(
                prepared_module,
                prepared_module->symbols.entries[symbol_id].defining_op));
      }
      if (function_version == nullptr ||
          function_version->function_target_facts == nullptr) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "legacy launch-config preparation produced no target facts");
      }
    }

    if (iree_status_is_ok(status)) {
      *out_module = prepared_module;
      *out_pipeline_result = pipeline_result;
    } else {
      if (pipeline_result_initialized) {
        loom_compile_pipeline_result_deinitialize(&pipeline_result);
      }
      loom_module_free(prepared_module);
    }
    return status;
  }

  iree_status_t Evaluate(int64_t element_count,
                         loomc_launch_config_t* out_config) {
    const loom_kernel_launch_config_options_t options = {
        /*.function_symbol=*/IREE_SV("unchecked_1d"),
        /*.workload_arguments=*/&element_count,
        /*.workload_argument_count=*/1,
        /*.required_fields=*/
        LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT,
        /*.function_target_facts=*/target_facts_,
        /*.diagnostic_emitter=*/{},
    };
    loom_kernel_launch_config_t evaluated_config = {};
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_evaluate(
        module_, &block_pool_, &options, &evaluated_config));
    if (loom_kernel_launch_config_has_failure(evaluated_config.failure)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "value-facts launch-config evaluation failed with code %u",
          (unsigned)evaluated_config.failure);
    }

    const loomc_launch_config_t config = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
        /*.structure_size=*/sizeof(config),
        /*.next=*/nullptr,
        /*.workgroup_count=*/
        {evaluated_config.workgroup_count.x, evaluated_config.workgroup_count.y,
         evaluated_config.workgroup_count.z},
        /*.workgroup_size=*/
        {evaluated_config.workgroup_size.x, evaluated_config.workgroup_size.y,
         evaluated_config.workgroup_size.z},
        /*.workgroup_cluster_size=*/{1, 1, 1},
        /*.subgroup_size=*/evaluated_config.subgroup_size,
        /*.workgroup_storage_bytes=*/0,
    };
    *out_config = config;
    return iree_ok_status();
  }

 private:
  // Reusable allocation pool used by parsing, persistent facts, and each
  // transient evaluator invocation.
  iree_arena_block_pool_t block_pool_ = {};

  // AMDGPU target provider composition required to parse the retained module.
  loom_target_environment_t target_environment_ = {};

  // Target-low descriptor registry used by the production compile pipeline.
  loom_target_low_descriptor_registry_t low_descriptor_registry_ = {};

  // Exact target used by the production launch-preparation specialization.
  loom_amdgpu_target_profile_t target_profile_ = {};

  // Context owning the operation and target dialect registrations.
  loom_context_t context_ = {};

  // Persistent arena owning the cached symbol and target facts.
  iree_arena_allocator_t analysis_arena_ = {};

  // Dense symbol-fact cache used to resolve the function target once.
  loom_symbol_fact_table_t symbol_facts_ = {};

  // Immutable source storage retained for the parsed module lifetime.
  SourcePtr source_;

  // Parsed source module evaluated on each invocation.
  loom_module_t* module_ = nullptr;

  // Cached target facts selected for the benchmark function.
  const loom_target_facts_t* target_facts_ = nullptr;

  // Result storage used to validate setup before timing begins.
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

static void RunLaunchConfigCompileBenchmark(
    benchmark::State& state, LaunchConfigBenchmarkFixture& fixture,
    loomc_compile_artifact_flags_t artifact_flags) {
  ResultPtr warm_result;
  if (SkipOnError(state, fixture.Compile(artifact_flags, &warm_result))) {
    return;
  }
  if (iree_any_bit_set(artifact_flags,
                       LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG)) {
    int64_t warm_artifact_bytes = 0;
    if (SkipOnError(state, ValidateLaunchConfigArtifact(
                               warm_result.get(), &warm_artifact_bytes))) {
      return;
    }
  }
  warm_result.reset();

  int64_t total_artifact_bytes = 0;
  for (auto _ : state) {
    (void)_;
    ResultPtr result;
    iree_status_t status = fixture.Compile(artifact_flags, &result);
    int64_t artifact_bytes = 0;
    if (iree_status_is_ok(status) &&
        iree_any_bit_set(artifact_flags,
                         LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG)) {
      status = ValidateLaunchConfigArtifact(result.get(), &artifact_bytes);
    }
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.PauseTiming();
      SkipOnError(state, status);
      break;
    }
    total_artifact_bytes += artifact_bytes;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(total_artifact_bytes);
  state.counters["artifact_bytes"] =
      state.iterations() == 0
          ? 0.0
          : (double)total_artifact_bytes / (double)state.iterations();
}

static void BM_DeviceCompileSmoke(benchmark::State& state,
                                  const char* source_identifier) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpCompiler(
                             loomc_make_cstring_view(source_identifier)))) {
    return;
  }
  RunLaunchConfigCompileBenchmark(state, fixture, /*artifact_flags=*/0);
}
BENCHMARK_CAPTURE(BM_DeviceCompileSmoke, OneFunction,
                  "launch_config_compile.loom")
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_DeviceCompileSmoke, CallableClosure,
                  "launch_config_closure.loom")
    ->UseRealTime();

static void BM_DeviceCompileWithLaunchConfigSmoke(
    benchmark::State& state, const char* source_identifier) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpCompiler(
                             loomc_make_cstring_view(source_identifier)))) {
    return;
  }
  RunLaunchConfigCompileBenchmark(state, fixture,
                                  LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG);
}
BENCHMARK_CAPTURE(BM_DeviceCompileWithLaunchConfigSmoke, OneFunction,
                  "launch_config_compile.loom")
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_DeviceCompileWithLaunchConfigSmoke, CallableClosure,
                  "launch_config_closure.loom")
    ->UseRealTime();

static void BM_DeviceCompileScale(benchmark::State& state) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpScaleCompiler(state.range(0)))) {
    return;
  }
  RunLaunchConfigCompileBenchmark(state, fixture, /*artifact_flags=*/0);
}
BENCHMARK(BM_DeviceCompileScale)
    ->Arg(1)
    ->Arg(8)
    ->Arg(80)
    ->Arg(1024)
    ->UseRealTime();

static void BM_DeviceCompileWithLaunchConfigScale(benchmark::State& state) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpScaleCompiler(state.range(0)))) {
    return;
  }
  RunLaunchConfigCompileBenchmark(state, fixture,
                                  LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG);
}
BENCHMARK(BM_DeviceCompileWithLaunchConfigScale)
    ->Arg(1)
    ->Arg(8)
    ->Arg(80)
    ->Arg(1024)
    ->UseRealTime();

static void RunVmLaunchConfigProgramLoadBenchmark(
    benchmark::State& state, LaunchConfigBenchmarkFixture& fixture) {
  LaunchConfigProgramPtr warm_program;
  if (SkipOnError(state, fixture.LoadProgram(&warm_program))) {
    return;
  }
  warm_program.reset();

  for (auto _ : state) {
    (void)_;
    LaunchConfigProgramPtr program;
    iree_status_t status = fixture.LoadProgram(&program);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.PauseTiming();
      SkipOnError(state, status);
      break;
    }
    benchmark::DoNotOptimize(program.get());
    state.PauseTiming();
    program.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["artifact_bytes"] =
      static_cast<double>(fixture.artifact_byte_length());
}

static void BM_VmLaunchConfigProgramLoadSmoke(benchmark::State& state,
                                              const char* source_identifier) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpArtifact(
                             loomc_make_cstring_view(source_identifier)))) {
    return;
  }
  RunVmLaunchConfigProgramLoadBenchmark(state, fixture);
}
BENCHMARK_CAPTURE(BM_VmLaunchConfigProgramLoadSmoke, OneFunction,
                  "launch_config_compile.loom")
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_VmLaunchConfigProgramLoadSmoke, CallableClosure,
                  "launch_config_closure.loom")
    ->UseRealTime();

static void BM_VmLaunchConfigProgramLoadScale(benchmark::State& state) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpScaleArtifact(state.range(0)))) {
    return;
  }
  RunVmLaunchConfigProgramLoadBenchmark(state, fixture);
}
BENCHMARK(BM_VmLaunchConfigProgramLoadScale)
    ->Arg(1)
    ->Arg(8)
    ->Arg(80)
    ->Arg(1024)
    ->UseRealTime();

static void BM_VmLaunchConfigFunctionLookupSmoke(benchmark::State& state) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpInvocation(IREE_SV("launch_config.loom"),
                                                 IREE_SV("unchecked_1d")))) {
    return;
  }

  for (auto _ : state) {
    (void)_;
    loomc_vm_launch_config_function_t function =
        loomc_vm_launch_config_function_invalid();
    iree_status_t status =
        fixture.LookupFunction(IREE_SV("unchecked_1d"), &function);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.PauseTiming();
      SkipOnError(state, status);
      break;
    }
    benchmark::DoNotOptimize(function);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["artifact_bytes"] =
      static_cast<double>(fixture.artifact_byte_length());
}
BENCHMARK(BM_VmLaunchConfigFunctionLookupSmoke)->Unit(benchmark::kNanosecond);

static void RunLaunchConfigInvokeBenchmark(benchmark::State& state,
                                           const char* source_identifier,
                                           const char* function_name) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpInvocation(
                             iree_make_cstring_view(source_identifier),
                             iree_make_cstring_view(function_name)))) {
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

static void BM_VmLaunchConfigInvokeSmoke(benchmark::State& state,
                                         const char* source_identifier,
                                         const char* function_name) {
  RunLaunchConfigInvokeBenchmark(state, source_identifier, function_name);
}

BENCHMARK_CAPTURE(BM_VmLaunchConfigInvokeSmoke, Unchecked1D,
                  "launch_config.loom", "unchecked_1d")
    ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BM_VmLaunchConfigInvokeSmoke, CallableClosure,
                  "launch_config_closure.loom", "callable_1d")
    ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BM_VmLaunchConfigInvokeSmoke, Checked1D, "launch_config.loom",
                  "checked_1d")
    ->Unit(benchmark::kNanosecond);

static void BM_ValueFactsLaunchConfigInvokeSmoke(benchmark::State& state) {
  ValueFactsLaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUp(
                             loomc_make_cstring_view("launch_config.loom")))) {
    return;
  }

  int64_t element_count = INT64_C(1048576);
  loomc_launch_config_t config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(config),
  };
  for (auto _ : state) {
    (void)_;
    benchmark::DoNotOptimize(element_count);
    iree_status_t status = fixture.Evaluate(element_count, &config);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.PauseTiming();
      SkipOnError(state, status);
      break;
    }
    benchmark::DoNotOptimize(config);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ValueFactsLaunchConfigInvokeSmoke)->Unit(benchmark::kNanosecond);

static void BM_LegacyLaunchConfigReadinessSmoke(benchmark::State& state) {
  LaunchConfigBenchmarkFixture compile_fixture;
  if (SkipOnError(state, compile_fixture.SetUpCompiler(loomc_make_cstring_view(
                             "launch_config_compile.loom")))) {
    return;
  }
  ValueFactsLaunchConfigBenchmarkFixture legacy_fixture;
  if (SkipOnError(state, legacy_fixture.SetUp(loomc_make_cstring_view(
                             "launch_config_compile.loom")))) {
    return;
  }

  ResultPtr warm_device_result;
  loom_module_t* warm_launch_module = nullptr;
  loom_compile_pipeline_result_t warm_launch_pipeline_result = {};
  iree_status_t warm_status =
      compile_fixture.Compile(/*artifact_flags=*/0, &warm_device_result);
  if (iree_status_is_ok(warm_status)) {
    warm_status = legacy_fixture.PrepareLegacyReadiness(
        &warm_launch_module, &warm_launch_pipeline_result);
  }
  loom_compile_pipeline_result_deinitialize(&warm_launch_pipeline_result);
  loom_module_free(warm_launch_module);
  warm_device_result.reset();
  if (SkipOnError(state, warm_status)) {
    return;
  }

  for (auto _ : state) {
    (void)_;
    ResultPtr device_result;
    loom_module_t* launch_module = nullptr;
    loom_compile_pipeline_result_t launch_pipeline_result = {};
    iree_status_t status =
        compile_fixture.Compile(/*artifact_flags=*/0, &device_result);
    if (iree_status_is_ok(status)) {
      status = legacy_fixture.PrepareLegacyReadiness(&launch_module,
                                                     &launch_pipeline_result);
    }
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.PauseTiming();
      loom_compile_pipeline_result_deinitialize(&launch_pipeline_result);
      loom_module_free(launch_module);
      SkipOnError(state, status);
      break;
    }
    benchmark::DoNotOptimize(device_result.get());
    benchmark::DoNotOptimize(launch_module);
    state.PauseTiming();
    loom_compile_pipeline_result_deinitialize(&launch_pipeline_result);
    loom_module_free(launch_module);
    device_result.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LegacyLaunchConfigReadinessSmoke)->UseRealTime();

static void BM_VmLaunchConfigReadinessSmoke(benchmark::State& state) {
  LaunchConfigBenchmarkFixture fixture;
  if (SkipOnError(state, fixture.SetUpCompiler(loomc_make_cstring_view(
                             "launch_config_compile.loom")))) {
    return;
  }

  ResultPtr warm_result;
  LaunchConfigProgramPtr warm_program;
  iree_status_t warm_status =
      fixture.Compile(LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG, &warm_result);
  const loomc_artifact_t* warm_artifact = nullptr;
  if (iree_status_is_ok(warm_status)) {
    warm_artifact = FindArtifact(
        warm_result.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM_BYTECODE));
    if (warm_artifact == nullptr) {
      warm_status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                     "launch-config artifact was not produced");
    }
  }
  if (iree_status_is_ok(warm_status)) {
    warm_status = fixture.LoadProgram(warm_artifact, &warm_program);
  }
  loomc_vm_launch_config_function_t warm_function =
      loomc_vm_launch_config_function_invalid();
  if (iree_status_is_ok(warm_status)) {
    warm_status = to_iree_status(loomc_vm_launch_config_program_lookup_function(
        warm_program.get(), loomc_make_cstring_view("unchecked_1d"),
        &warm_function));
  }
  benchmark::DoNotOptimize(warm_function);
  warm_program.reset();
  warm_result.reset();
  if (SkipOnError(state, warm_status)) {
    return;
  }

  int64_t total_artifact_bytes = 0;
  for (auto _ : state) {
    (void)_;
    ResultPtr result;
    LaunchConfigProgramPtr program;
    loomc_vm_launch_config_function_t function =
        loomc_vm_launch_config_function_invalid();
    iree_status_t status =
        fixture.Compile(LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG, &result);
    const loomc_artifact_t* artifact = nullptr;
    if (iree_status_is_ok(status)) {
      artifact = FindArtifact(
          result.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM_BYTECODE));
      if (artifact == nullptr) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "launch-config artifact was not produced");
      }
    }
    if (iree_status_is_ok(status)) {
      status = fixture.LoadProgram(artifact, &program);
    }
    if (iree_status_is_ok(status)) {
      status = to_iree_status(loomc_vm_launch_config_program_lookup_function(
          program.get(), loomc_make_cstring_view("unchecked_1d"), &function));
    }
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.PauseTiming();
      SkipOnError(state, status);
      break;
    }
    total_artifact_bytes += loomc_byte_sequence_length(artifact->contents);
    benchmark::DoNotOptimize(program.get());
    benchmark::DoNotOptimize(function);
    state.PauseTiming();
    program.reset();
    result.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(total_artifact_bytes);
  state.counters["artifact_bytes"] =
      state.iterations() == 0
          ? 0.0
          : (double)total_artifact_bytes / (double)state.iterations();
}
BENCHMARK(BM_VmLaunchConfigReadinessSmoke)->UseRealTime();

}  // namespace
