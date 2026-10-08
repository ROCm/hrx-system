// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "benchmark/benchmark.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"
#include "loom/binding/c/benchmark/import/cxx/kernels.h"
#include "loomc/import/cxx.h"
#include "loomc/target/amdgpu.h"

namespace loomc::bench {
namespace {

struct CxxKernel {
  // Embedded source filename; one compilation unit may contain several kernels.
  const char* source;
  // Export selected for this JIT invocation.
  const char* root;
  // Concrete AMDGPU profile used for lowering and code object emission.
  const char* target;
};

enum class CxxJitPhase {
  kImport,
  kCloneHigh,
  kSourceLow,
  kPreparedLow,
  kClonePreparedLow,
  kEmitPreparedLow,
};

struct CxxJitPhaseSpec {
  // Public compiler boundary measured by one benchmark iteration.
  CxxJitPhase phase;
  // Source, export and target shared with the complete endpoint benchmark.
  const CxxKernel* kernel;
};

class CxxSourceScenarioBase : public TargetCompileScenario {
 public:
  explicit CxxSourceScenarioBase(const CxxKernel& kernel) : kernel_(kernel) {}

  iree_host_size_t job_count() const override { return 1; }

  void SetExtraCounters(::benchmark::State& state) const override {
    state.SetLabel(kernel_.target);
    state.counters["source_bytes"] = (double)source_byte_count_;
  }

 protected:
  iree_status_t SetUpSource(iree_host_size_t worker_count,
                            loomc_target_pipeline_kind_t pipeline_kind) {
    loomc_target_environment_t* raw_environment = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_amdgpu(
        loom_allocator(), &raw_environment)));
    TargetEnvironmentPtr environment(raw_environment);
    loomc_amdgpu_profile_options_t profile_options = {};
    profile_options.type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS;
    profile_options.structure_size = sizeof(profile_options);
    profile_options.identifier = loomc_make_cstring_view(kernel_.target);
    profile_options.identity.target = profile_options.identifier;
    loomc_target_profile_t* raw_profile = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_profile_create_amdgpu(
        environment.get(), &profile_options, loom_allocator(), &raw_profile)));
    IREE_RETURN_IF_ERROR(
        SetUpTarget(worker_count, std::move(environment),
                    TargetProfilePtr(raw_profile), pipeline_kind,
                    pipeline_kind == LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW
                        ? loomc_make_cstring_view("cxx-source-low")
                        : loomc_make_cstring_view("cxx-source-to-hsaco"),
                    LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG));

    const EmbeddedSource embedded =
        FindEmbeddedSource(loomc_cxx_benchmark_kernels_create(),
                           loomc_cxx_benchmark_kernels_size(), kernel_.source);
    source_byte_count_ = (int64_t)embedded.contents.data_length;
    const loomc_source_options_t source_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /*.structure_size=*/sizeof(source_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_UNKNOWN,
        /*.identifier=*/embedded.identifier,
        /*.contents=*/embedded.contents,
        /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
    };
    loomc_source_t* raw_source = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(
        loomc_source_create(&source_options, loom_allocator(), &raw_source)));
    source_.reset(raw_source);

    std::string config;
    for (int axis = 0; axis < 3; ++axis) {
      config += "config.def @" + std::string(kernel_.root) +
                ".workgroup_count." + "xyz"[axis] + " = " +
                (axis == 0 ? "3" : "1") + " : index\n";
    }
    IREE_RETURN_IF_ERROR(CreateWorkspace(/*block_size=*/0, &setup_workspace_));
    return CreateTextModule(context_.get(), setup_workspace_.get(),
                            "launch-config.loom", config, &config_);
  }

  iree_status_t ImportSource(WorkspacePtr& workspace, ModulePtr* out_module) {
    out_module->reset();
    const loomc_string_view_t root = loomc_make_cstring_view(kernel_.root);
    loomc_cxx_import_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS;
    options.structure_size = sizeof(options);
    options.flags = LOOMC_CXX_IMPORT_FLAG_APPROXIMATE_FUNCTIONS;
    options.roots = &root;
    options.root_count = 1;
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_module_import_cxx(
        context_.get(), workspace.get(), source_.get(), &options,
        loom_allocator(), &raw_module, &raw_result));
    ModulePtr module(raw_module);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(result.get(), "C++ import"));
    out_module->reset(module.release());
    return iree_ok_status();
  }

  iree_status_t CompileSource(WorkspacePtr& workspace, ModulePtr& module) {
    const loomc_string_view_t root = loomc_make_cstring_view(kernel_.root);
    return CompileModuleToTargetBoundary(workspace, module, root, root,
                                         config_.get(), 0);
  }

  iree_status_t EmitHsaco(WorkspacePtr& workspace, ModulePtr& module) {
    const loomc_string_view_t root = loomc_make_cstring_view(kernel_.root);
    const loomc_emit_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.artifact_format=*/
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        /*.identifier=*/root,
        /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_emit_module(target_environment(), workspace.get(), module.get(),
                          &options, loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "HSACO emission"));
    int64_t byte_count = 0;
    IREE_RETURN_IF_ERROR(ValidateArtifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE, options.artifact_format,
        4, "HSACO executable", &byte_count));
    const loomc_artifact_t* artifact = FindArtifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE, options.artifact_format);
    uint8_t magic[4];
    IREE_RETURN_IF_ERROR(ReadArtifactPrefix(
        artifact, iree_make_byte_span(magic, sizeof(magic))));
    if (std::memcmp(magic,
                    "\x7f"
                    "ELF",
                    4) != 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "C++ compilation did not produce an ELF image");
    }
    RecordArtifactBytes(byte_count);
    return iree_ok_status();
  }

 private:
  // Static benchmark registration selecting source, export and target.
  const CxxKernel& kernel_;
  // Immutable source text backed by the embedded corpus table.
  SourcePtr source_;
  // Setup-only workspace retaining the immutable launch config module.
  WorkspacePtr setup_workspace_;
  // Ordinary immutable config module shared by source compilations.
  ModulePtr config_;
  // Embedded source size reported independently of generated artifacts.
  int64_t source_byte_count_ = 0;
};

class CxxSourceScenario final : public CxxSourceScenarioBase {
 public:
  explicit CxxSourceScenario(const CxxKernel& kernel)
      : CxxSourceScenarioBase(kernel) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    return SetUpSource(worker_count, LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW);
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    ModulePtr module;
    IREE_RETURN_IF_ERROR(ImportSource(workspace, &module));
    IREE_RETURN_IF_ERROR(CompileSource(workspace, module));
    return EmitHsaco(workspace, module);
  }
};

class CxxJitPhaseScenario final : public CxxSourceScenarioBase {
 public:
  CxxJitPhaseScenario(CxxJitPhase phase, const CxxKernel& kernel)
      : CxxSourceScenarioBase(kernel), phase_(phase) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    const loomc_target_pipeline_kind_t pipeline_kind =
        phase_ == CxxJitPhase::kSourceLow
            ? LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW
            : LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
    IREE_RETURN_IF_ERROR(SetUpSource(worker_count, pipeline_kind));
    if (phase_ == CxxJitPhase::kImport) {
      return iree_ok_status();
    }

    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    IREE_RETURN_IF_ERROR(ImportSource(template_workspace_, &high_template_));
    if (UsesPreparedTemplate()) {
      IREE_RETURN_IF_ERROR(CloneModule(high_template_.get(),
                                       template_workspace_.get(),
                                       &prepared_template_));
      IREE_RETURN_IF_ERROR(
          CompileSource(template_workspace_, prepared_template_));
    }
    return iree_ok_status();
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    ModulePtr module;
    if (phase_ == CxxJitPhase::kImport) {
      IREE_RETURN_IF_ERROR(ImportSource(workspace, &module));
    } else {
      const loomc_module_t* template_module = UsesPreparedTemplate()
                                                  ? prepared_template_.get()
                                                  : high_template_.get();
      IREE_RETURN_IF_ERROR(
          CloneModule(template_module, workspace.get(), &module));
      if (phase_ == CxxJitPhase::kSourceLow ||
          phase_ == CxxJitPhase::kPreparedLow) {
        IREE_RETURN_IF_ERROR(CompileSource(workspace, module));
      } else if (phase_ == CxxJitPhase::kEmitPreparedLow) {
        IREE_RETURN_IF_ERROR(EmitHsaco(workspace, module));
      }
    }
    ::benchmark::DoNotOptimize(module.get());
    return iree_ok_status();
  }

 private:
  bool UsesPreparedTemplate() const {
    return phase_ == CxxJitPhase::kClonePreparedLow ||
           phase_ == CxxJitPhase::kEmitPreparedLow;
  }

  // Public compiler boundary measured by each timed invocation.
  CxxJitPhase phase_;
  // Setup-only workspace retaining immutable High and prepared Low templates.
  WorkspacePtr template_workspace_;
  // Verified imported High module cloned by source-lowering phases.
  ModulePtr high_template_;
  // Prepared Low module retaining target specialization facts for emission.
  ModulePtr prepared_template_;
};

std::unique_ptr<CompileScenario> CreateCxxSourceScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)state;
  return std::make_unique<CxxSourceScenario>(
      *static_cast<const CxxKernel*>(user_data));
}

std::unique_ptr<CompileScenario> CreateCxxJitPhaseScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)state;
  const auto* spec = static_cast<const CxxJitPhaseSpec*>(user_data);
  return std::make_unique<CxxJitPhaseScenario>(spec->phase, *spec->kernel);
}

void SourceToHsaco(::benchmark::State& state, const CxxKernel* kernel) {
  RunCompileBenchmarkDirect(state, CreateCxxSourceScenario, kernel);
}

void SourceToHsacoColdWorkspace(::benchmark::State& state,
                                const CxxKernel* kernel) {
  RunCompileBenchmarkDirectCold(state, CreateCxxSourceScenario, kernel);
}

constexpr CxxKernel kFlashAttention = {"flash_attention.cxx", "flash_attention",
                                       "gfx1151"};
constexpr CxxKernel kRmsNorm = {"llama_rms_norm.cxx", "llama_rms_norm",
                                "gfx1151"};
constexpr CxxKernel kSwiGlu = {"aiter_swiglu_f16.cxx", "aiter_swiglu_f16",
                               "gfx1151"};
constexpr CxxKernel kMxfp4 = {"mxfp_group_dot.cxx", "mxfp4_decode_dot",
                              "gfx1250"};
constexpr CxxKernel kMxfp8 = {"mxfp_group_dot.cxx", "mxfp8_decode_dot",
                              "gfx1250"};

struct CxxJitPhaseRegistration {
  // Phase supplied to the benchmark scenario.
  CxxJitPhase phase;
  // Stable component appended to the benchmark name.
  const char* name;
};

void RegisterCxxJitPhaseBenchmarks(const char* kernel_name,
                                   const CxxKernel* kernel) {
  constexpr CxxJitPhaseRegistration kPhases[] = {
      {CxxJitPhase::kImport, "Import"},
      {CxxJitPhase::kCloneHigh, "CloneHigh"},
      {CxxJitPhase::kSourceLow, "SourceLow"},
      {CxxJitPhase::kPreparedLow, "PreparedLow"},
      {CxxJitPhase::kClonePreparedLow, "ClonePreparedLow"},
      {CxxJitPhase::kEmitPreparedLow, "EmitPreparedLow"},
  };
  for (const CxxJitPhaseRegistration& registration : kPhases) {
    const CxxJitPhaseSpec spec = {
        /*.phase=*/registration.phase,
        /*.kernel=*/kernel,
    };
    const std::string name =
        std::string("CxxJitPhase/") + kernel_name + "/" + registration.name;
    auto* benchmark = ::benchmark::RegisterBenchmark(
        name.c_str(), [spec](::benchmark::State& state) {
          RunCompileBenchmarkDirect(state, CreateCxxJitPhaseScenario, &spec);
        });
    benchmark->Unit(::benchmark::kMicrosecond);
  }
}

[[maybe_unused]] const bool kCxxJitPhasesRegistered = [] {
  RegisterCxxJitPhaseBenchmarks("RmsNorm", &kRmsNorm);
  RegisterCxxJitPhaseBenchmarks("Mxfp8Gfx1250", &kMxfp8);
  return true;
}();

BENCHMARK_CAPTURE(SourceToHsaco, FlashAttention, &kFlashAttention)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, RmsNorm, &kRmsNorm)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, SwiGluF16, &kSwiGlu)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, Mxfp4Gfx1250, &kMxfp4)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, Mxfp8Gfx1250, &kMxfp8)
    ->Unit(::benchmark::kMicrosecond);

BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, RmsNorm, &kRmsNorm)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, Mxfp8Gfx1250, &kMxfp8)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);

}  // namespace
}  // namespace loomc::bench
