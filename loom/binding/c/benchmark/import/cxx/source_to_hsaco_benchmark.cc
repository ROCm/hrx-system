// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"
#include "loom/binding/c/benchmark/import/cxx/kernels.h"
#include "loomc/import/cxx.h"
#include "loomc/link.h"
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
  // Additional definitions satisfying source-owned configuration declarations.
  const char* config_definitions = nullptr;
  // Embedded sources served when preprocessing user includes.
  const char* const* include_sources = nullptr;
  // Number of entries in include_sources.
  iree_host_size_t include_source_count = 0;
};

enum class CxxJitPhase {
  kImport,
  kSourceToPreparedLow,
  kCloneHigh,
  kSourceLow,
  kPreparedLow,
  kHighToHsaco,
  kClonePreparedLow,
  kEmitPreparedLow,
};

struct CxxJitPhaseSpec {
  // Public compiler boundary measured by one benchmark iteration.
  CxxJitPhase phase;
  // Source, export and target shared with the complete endpoint benchmark.
  const CxxKernel* kernel;
};

static iree_status_t CreateEmbeddedSource(const EmbeddedSource& embedded,
                                          SourcePtr* out_source) {
  out_source->reset();
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
  out_source->reset(raw_source);
  return iree_ok_status();
}

// Immutable embedded translation unit and its include closure.
class EmbeddedCxxSourceClosure {
 public:
  iree_status_t Initialize(const char* main_source,
                           const char* const* include_sources,
                           iree_host_size_t include_source_count) {
    const auto* embedded_sources = loomc_cxx_benchmark_kernels_create();
    const size_t embedded_source_count = loomc_cxx_benchmark_kernels_size();
    const EmbeddedSource embedded = FindEmbeddedSource(
        embedded_sources, embedded_source_count, main_source);
    IREE_RETURN_IF_ERROR(CreateEmbeddedSource(embedded, &main_source_));
    source_byte_count_ = (int64_t)embedded.contents.data_length;
    include_sources_.clear();
    include_sources_.reserve(include_source_count);
    for (iree_host_size_t i = 0; i < include_source_count; ++i) {
      const EmbeddedSource include = FindEmbeddedSource(
          embedded_sources, embedded_source_count, include_sources[i]);
      SourcePtr include_source;
      IREE_RETURN_IF_ERROR(CreateEmbeddedSource(include, &include_source));
      source_byte_count_ += (int64_t)include.contents.data_length;
      include_sources_.push_back(std::move(include_source));
    }
    return iree_ok_status();
  }

  loomc_source_t* main_source() const { return main_source_.get(); }

  loomc_cxx_source_provider_t source_provider() {
    return include_sources_.empty()
               ? loomc_cxx_source_provider_t{}
               : loomc_cxx_source_provider_t{ProvideSource, this};
  }

  int64_t source_byte_count() const { return source_byte_count_; }

 private:
  static loomc_status_t ProvideSource(void* user_data, loomc_string_view_t path,
                                      loomc_source_t** out_source) {
    auto& self = *static_cast<EmbeddedCxxSourceClosure*>(user_data);
    *out_source = nullptr;
    std::string_view candidate(path.data, path.size);
    const size_t separator = candidate.find_last_of('/');
    if (separator != std::string_view::npos) {
      candidate.remove_prefix(separator + 1);
    }
    for (const SourcePtr& include_source : self.include_sources_) {
      const loomc_string_view_t identifier =
          loomc_source_identifier(include_source.get());
      if (candidate == std::string_view(identifier.data, identifier.size)) {
        loomc_source_retain(include_source.get());
        *out_source = include_source.get();
        break;
      }
    }
    return loomc_ok_status();
  }

  // Main source submitted to the importer.
  SourcePtr main_source_;
  // Immutable user includes returned through the source provider.
  std::vector<SourcePtr> include_sources_;
  // Unique embedded bytes across the source closure.
  int64_t source_byte_count_ = 0;
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

    IREE_RETURN_IF_ERROR(source_closure_.Initialize(
        kernel_.source, kernel_.include_sources, kernel_.include_source_count));
    source_byte_count_ = source_closure_.source_byte_count();

    std::string config;
    for (int axis = 0; axis < 3; ++axis) {
      config += "config.def @" + std::string(kernel_.root) +
                ".workgroup_count." + "xyz"[axis] + " = " +
                (axis == 0 ? "3" : "1") + " : index\n";
    }
    if (kernel_.config_definitions) {
      config += kernel_.config_definitions;
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
    options.source_provider = source_closure_.source_provider();
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_module_import_cxx(
        context_.get(), workspace.get(), source_closure_.main_source(),
        &options, loom_allocator(), &raw_module, &raw_result));
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
  // Embedded translation unit and immutable include sources.
  EmbeddedCxxSourceClosure source_closure_;
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
    if (phase_ == CxxJitPhase::kImport ||
        phase_ == CxxJitPhase::kSourceToPreparedLow) {
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
    if (phase_ == CxxJitPhase::kImport ||
        phase_ == CxxJitPhase::kSourceToPreparedLow) {
      IREE_RETURN_IF_ERROR(ImportSource(workspace, &module));
      if (phase_ == CxxJitPhase::kSourceToPreparedLow) {
        IREE_RETURN_IF_ERROR(CompileSource(workspace, module));
      }
    } else {
      const loomc_module_t* template_module = UsesPreparedTemplate()
                                                  ? prepared_template_.get()
                                                  : high_template_.get();
      IREE_RETURN_IF_ERROR(
          CloneModule(template_module, workspace.get(), &module));
      if (phase_ == CxxJitPhase::kSourceLow ||
          phase_ == CxxJitPhase::kPreparedLow) {
        IREE_RETURN_IF_ERROR(CompileSource(workspace, module));
      } else if (phase_ == CxxJitPhase::kHighToHsaco) {
        IREE_RETURN_IF_ERROR(CompileSource(workspace, module));
        IREE_RETURN_IF_ERROR(EmitHsaco(workspace, module));
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

struct CxxModelConfig {
  // Authored config declaration key.
  const char* key;
  // Exact value supplied to one product request.
  const char* value;
};

struct CxxModelProduct {
  // Stable benchmark and artifact label.
  const char* name;
  // Exported kernel selected from the reusable library.
  const char* root;
  // Concrete AMDGPU processor profile.
  const char* target;
  // Exact product configuration bindings.
  const CxxModelConfig* configs;
  // Number of product configuration bindings.
  iree_host_size_t config_count;
};

constexpr const char* kModelLibraryIncludes[] = {
    "iq4xs_gate_up.cxx",          "nvfp4_matrix.cxx",  "nvfp4_matrix.h",
    "nvfp4_matrix_providers.cxx", "q4k_q8_swiglu.cxx",
};

constexpr const char* kModelLibraryRoots[] = {
    "ffn_routed_gate_up_swiglu_q4k_q8",
    "nvfp4_matrix",
    "qwen38_iq4xs_gate_up_swiglu",
};

// Shared import, serialization, indexing, linking, and compilation flow for the
// one-import model-library benchmarks.
class CxxModelLibraryScenarioBase : public TargetCompileScenario {
 public:
  explicit CxxModelLibraryScenarioBase(const CxxModelProduct* product)
      : product_(product) {}

  iree_host_size_t job_count() const override { return 1; }

  void SetExtraCounters(::benchmark::State& state) const override {
    state.SetLabel(product_ != nullptr ? product_->target : "multi-target");
    state.counters["source_bytes"] =
        (double)source_closure_.source_byte_count();
    state.counters["library_bytecode_bytes"] = (double)library_bytecode_bytes_;
    state.counters["indexed_symbols"] = (double)indexed_symbol_count_;
  }

 protected:
  iree_status_t SetUpLibrary(iree_host_size_t worker_count,
                             const char* setup_target) {
    loomc_target_environment_t* raw_environment = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_amdgpu(
        loom_allocator(), &raw_environment)));
    TargetEnvironmentPtr environment(raw_environment);
    loomc_amdgpu_profile_options_t profile_options = {};
    profile_options.type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS;
    profile_options.structure_size = sizeof(profile_options);
    profile_options.identifier = loomc_make_cstring_view(setup_target);
    profile_options.identity.target = profile_options.identifier;
    loomc_target_profile_t* raw_profile = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_profile_create_amdgpu(
        environment.get(), &profile_options, loom_allocator(), &raw_profile)));
    IREE_RETURN_IF_ERROR(SetUpTarget(
        worker_count, std::move(environment), TargetProfilePtr(raw_profile),
        LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        loomc_make_cstring_view("cxx-model-library"),
        LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG));
    return source_closure_.Initialize("specializing_model_library.cxx",
                                      kModelLibraryIncludes,
                                      IREE_ARRAYSIZE(kModelLibraryIncludes));
  }

  iree_status_t ImportSerializeAndIndex(WorkspacePtr& workspace,
                                        LinkIndexPtr* out_index) {
    out_index->reset();
    std::array<loomc_string_view_t, IREE_ARRAYSIZE(kModelLibraryRoots)> roots;
    for (iree_host_size_t i = 0; i < roots.size(); ++i) {
      roots[i] = loomc_make_cstring_view(kModelLibraryRoots[i]);
    }
    loomc_cxx_import_options_t import_options = {};
    import_options.type = LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS;
    import_options.structure_size = sizeof(import_options);
    import_options.flags = LOOMC_CXX_IMPORT_FLAG_APPROXIMATE_FUNCTIONS;
    import_options.source_provider = source_closure_.source_provider();
    import_options.roots = roots.data();
    import_options.root_count = roots.size();
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_import_result = nullptr;
    iree_status_t status = to_iree_status(loomc_module_import_cxx(
        context_.get(), workspace.get(), source_closure_.main_source(),
        &import_options, loom_allocator(), &raw_module, &raw_import_result));
    ModulePtr module(raw_module);
    ResultPtr import_result(raw_import_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(import_result.get(), "model-library import"));

    const loomc_module_serialize_options_t serialize_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        /*.structure_size=*/sizeof(serialize_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
        /*.identifier=*/
        loomc_make_cstring_view("specializing_model_library.loombc"),
    };
    loomc_source_t* raw_bytecode = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_module_serialize_to_source(
        module.get(), &serialize_options, loom_allocator(), &raw_bytecode)));
    SourcePtr bytecode(raw_bytecode);
    library_bytecode_bytes_ =
        (int64_t)loomc_source_contents(bytecode.get()).data_length;

    loomc_link_index_builder_t* raw_builder = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &raw_builder)));
    LinkIndexBuilderPtr builder(raw_builder);
    IREE_RETURN_IF_ERROR(AddSourceToIndex(builder.get(), bytecode.get(),
                                          "model-library",
                                          LOOMC_LINK_PROVIDER_ROLE_LIBRARY));
    loomc_link_index_t* raw_index = nullptr;
    loomc_result_t* raw_index_result = nullptr;
    status = to_iree_status(loomc_link_index_builder_finish(
        builder.get(), &raw_index, &raw_index_result));
    LinkIndexPtr index(raw_index);
    ResultPtr index_result(raw_index_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(index_result.get(), "model-library indexing"));
    indexed_symbol_count_ = (int64_t)loomc_link_index_symbol_count(index.get());
    out_index->reset(index.release());
    return iree_ok_status();
  }

  iree_status_t PrepareLibrary() {
    WorkspacePtr workspace;
    IREE_RETURN_IF_ERROR(CreateWorkspace(/*block_size=*/0, &workspace));
    IREE_RETURN_IF_ERROR(ImportSerializeAndIndex(workspace, &library_index_));
    loomc_linker_t* raw_linker = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_linker_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &raw_linker)));
    linker_.reset(raw_linker);
    return iree_ok_status();
  }

  iree_status_t LinkProduct(WorkspacePtr& workspace, ModulePtr* out_module) {
    out_module->reset();
    const loomc_string_view_t root = loomc_make_cstring_view(product_->root);
    const loomc_link_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.link_index=*/library_index_.get(),
        /*.module_name=*/loomc_make_cstring_view(product_->name),
        /*.mode=*/LOOMC_LINK_MODE_LINK,
        /*.root_symbols=*/&root,
        /*.root_symbol_count=*/1,
        /*.flags=*/LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS,
        /*.config=*/{},
        /*.root_provider_ordinals=*/nullptr,
        /*.root_provider_count=*/0,
        /*.module_providers=*/nullptr,
        /*.module_provider_count=*/0,
    };
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_link_module(
        linker_.get(), workspace.get(), &options, &raw_module, &raw_result));
    ModulePtr module(raw_module);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "model-library linking"));
    out_module->reset(module.release());
    return iree_ok_status();
  }

  iree_status_t CompileProduct(WorkspacePtr& workspace, ModulePtr& module) {
    constexpr iree_host_size_t kMaxConfigCount = 5;
    IREE_ASSERT_LE(product_->config_count, kMaxConfigCount);
    std::array<loomc_config_binding_t, kMaxConfigCount> bindings;
    for (iree_host_size_t i = 0; i < product_->config_count; ++i) {
      bindings[i] = {
          loomc_make_cstring_view(product_->configs[i].key),
          loomc_make_cstring_view(product_->configs[i].value),
      };
    }
    const loomc_config_options_t config_options = {
        /*.bindings=*/bindings.data(),
        /*.binding_count=*/product_->config_count,
        /*.json_object=*/{},
        /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
    };
    const loomc_emit_options_t emit_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(emit_options),
        /*.next=*/nullptr,
        /*.artifact_format=*/
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        /*.identifier=*/loomc_make_cstring_view(product_->name),
        /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    const loomc_string_view_t root = loomc_make_cstring_view(product_->root);
    const loomc_compile_artifact_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.roots=*/&root,
        /*.root_count=*/1,
        /*.excluded_roots=*/nullptr,
        /*.excluded_root_count=*/0,
        /*.target_profile=*/target_profile(),
        /*.config=*/
        product_->config_count != 0 ? &config_options : nullptr,
        /*.emit_options=*/&emit_options,
        /*.artifact_flags=*/0,
    };
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_compile_artifact(
        compiler_.get(), workspace.get(), pass_program_.get(), module.get(),
        &options, loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "model-library compilation"));
    int64_t byte_count = 0;
    IREE_RETURN_IF_ERROR(ValidateArtifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        /*minimum_data_length=*/4, "HSACO executable", &byte_count));
    const loomc_artifact_t* artifact = FindArtifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
    uint8_t magic[4];
    IREE_RETURN_IF_ERROR(ReadArtifactPrefix(
        artifact, iree_make_byte_span(magic, sizeof(magic))));
    if (std::memcmp(magic,
                    "\x7f"
                    "ELF",
                    4) != 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "model product is not an ELF image");
    }
    RecordArtifactBytes(byte_count);
    return iree_ok_status();
  }

  int64_t library_bytecode_bytes() const { return library_bytecode_bytes_; }

 private:
  // Product selected by timed link-to-artifact scenarios, or NULL for import.
  const CxxModelProduct* product_;
  // One embedded translation unit and its immutable include closure.
  EmbeddedCxxSourceClosure source_closure_;
  // Frozen library index prepared outside product timing.
  LinkIndexPtr library_index_;
  // Prepared linker shared by product invocations.
  LinkerPtr linker_;
  // Serialized reusable-library size from the most recent preparation.
  int64_t library_bytecode_bytes_ = 0;
  // Frozen index breadth from the most recent preparation.
  int64_t indexed_symbol_count_ = 0;
};

class CxxModelLibraryPreparationScenario final
    : public CxxModelLibraryScenarioBase {
 public:
  CxxModelLibraryPreparationScenario()
      : CxxModelLibraryScenarioBase(/*product=*/nullptr) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    return SetUpLibrary(worker_count, "gfx942");
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    LinkIndexPtr index;
    IREE_RETURN_IF_ERROR(
        ImportSerializeAndIndex(workspace_at(worker_ordinal), &index));
    RecordArtifactBytes(library_bytecode_bytes());
    ::benchmark::DoNotOptimize(index.get());
    return iree_ok_status();
  }
};

class CxxModelLibraryProductScenario final
    : public CxxModelLibraryScenarioBase {
 public:
  explicit CxxModelLibraryProductScenario(const CxxModelProduct& product)
      : CxxModelLibraryScenarioBase(&product), product_(product) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpLibrary(worker_count, product_.target));
    return PrepareLibrary();
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    ModulePtr module;
    IREE_RETURN_IF_ERROR(LinkProduct(workspace, &module));
    IREE_RETURN_IF_ERROR(CompileProduct(workspace, module));
    ::benchmark::DoNotOptimize(module.get());
    return iree_ok_status();
  }

 private:
  // Stable product row selected by benchmark registration.
  const CxxModelProduct& product_;
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

std::unique_ptr<CompileScenario> CreateCxxModelLibraryPreparationScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)state;
  (void)user_data;
  return std::make_unique<CxxModelLibraryPreparationScenario>();
}

std::unique_ptr<CompileScenario> CreateCxxModelLibraryProductScenario(
    const ::benchmark::State& state, const void* user_data) {
  (void)state;
  return std::make_unique<CxxModelLibraryProductScenario>(
      *static_cast<const CxxModelProduct*>(user_data));
}

void SourceToHsaco(::benchmark::State& state, const CxxKernel* kernel) {
  RunCompileBenchmarkDirect(state, CreateCxxSourceScenario, kernel);
}

void SourceToHsacoColdWorkspace(::benchmark::State& state,
                                const CxxKernel* kernel) {
  RunCompileBenchmarkDirectCold(state, CreateCxxSourceScenario, kernel);
}

void CxxModelLibraryImportSerializeIndex(::benchmark::State& state) {
  RunCompileBenchmarkDirect(state, CreateCxxModelLibraryPreparationScenario,
                            /*user_data=*/nullptr);
}

void CxxModelLibraryLinkToHsaco(::benchmark::State& state,
                                const CxxModelProduct* product) {
  RunCompileBenchmarkDirect(state, CreateCxxModelLibraryProductScenario,
                            product);
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
constexpr CxxKernel kIq4XsGateUpGfx1151 = {
    "iq4xs_gate_up.cxx",
    "qwen38_iq4xs_gate_up_swiglu",
    "gfx1151",
    "config.def @qwen38.iq4xs.async_staging = 0 : i1\n"
    "config.def @qwen38.iq4xs.staging_depth = 1 : i32\n"
    "config.def @qwen38.iq4xs.packet_unroll_factor = 4 : i32\n"
    "config.def @qwen38.iq4xs.block_unroll_factor = 2 : i32\n"
    "config.def @qwen38.iq4xs.tile_unroll_factor = 1 : i32\n",
};
constexpr CxxKernel kIq4XsGateUpGfx942AsyncSingle = {
    "iq4xs_gate_up.cxx",
    "qwen38_iq4xs_gate_up_swiglu",
    "gfx942",
    "config.def @qwen38.iq4xs.async_staging = 1 : i1\n"
    "config.def @qwen38.iq4xs.staging_depth = 1 : i32\n"
    "config.def @qwen38.iq4xs.packet_unroll_factor = 4 : i32\n"
    "config.def @qwen38.iq4xs.block_unroll_factor = 2 : i32\n"
    "config.def @qwen38.iq4xs.tile_unroll_factor = 1 : i32\n",
};
constexpr CxxKernel kIq4XsGateUpGfx942AsyncDouble = {
    "iq4xs_gate_up.cxx",
    "qwen38_iq4xs_gate_up_swiglu",
    "gfx942",
    "config.def @qwen38.iq4xs.async_staging = 1 : i1\n"
    "config.def @qwen38.iq4xs.staging_depth = 2 : i32\n"
    "config.def @qwen38.iq4xs.packet_unroll_factor = 4 : i32\n"
    "config.def @qwen38.iq4xs.block_unroll_factor = 2 : i32\n"
    "config.def @qwen38.iq4xs.tile_unroll_factor = 1 : i32\n",
};
constexpr CxxKernel kQ4KQ8SwiGlu = {
    "q4k_q8_swiglu.cxx",
    "ffn_routed_gate_up_swiglu_q4k_q8",
    "gfx1250",
    "config.def @ffn_routed_gate_up.input_size = 4096 : i32\n",
};
constexpr CxxKernel kConfiguredWorkgroupStorage = {
    "configured_workgroup_storage.cxx",
    "configured_workgroup_storage",
    "gfx1151",
    "config.def @test.buffer.stage_count = 4 : i32\n",
};
constexpr const char* kNvFp4MatrixIncludes[] = {
    "nvfp4_matrix.cxx",
    "nvfp4_matrix.h",
    "nvfp4_matrix_providers.cxx",
};
constexpr CxxKernel kNvFp4Gfx1100 = {
    "nvfp4_matrix_benchmark.cxx",
    "nvfp4_matrix",
    "gfx1100",
    nullptr,
    kNvFp4MatrixIncludes,
    IREE_ARRAYSIZE(kNvFp4MatrixIncludes),
};
constexpr CxxKernel kNvFp4Gfx942 = {
    "nvfp4_matrix_benchmark.cxx",
    "nvfp4_matrix",
    "gfx942",
    nullptr,
    kNvFp4MatrixIncludes,
    IREE_ARRAYSIZE(kNvFp4MatrixIncludes),
};

constexpr CxxModelConfig kModelQ4KQ8Configs[] = {
    {"ffn_routed_gate_up.input_size", "512"},
};

constexpr CxxModelConfig kModelIq4XsSingleConfigs[] = {
    {"qwen38.iq4xs.async_staging", "true"},
    {"qwen38.iq4xs.staging_depth", "1"},
    {"qwen38.iq4xs.packet_unroll_factor", "4"},
    {"qwen38.iq4xs.block_unroll_factor", "2"},
    {"qwen38.iq4xs.tile_unroll_factor", "1"},
};

constexpr CxxModelConfig kModelIq4XsDoubleConfigs[] = {
    {"qwen38.iq4xs.async_staging", "true"},
    {"qwen38.iq4xs.staging_depth", "2"},
    {"qwen38.iq4xs.packet_unroll_factor", "4"},
    {"qwen38.iq4xs.block_unroll_factor", "2"},
    {"qwen38.iq4xs.tile_unroll_factor", "1"},
};

constexpr CxxModelProduct kModelQ4KQ8Gfx1250 = {
    "q4k_q8_gfx1250",   "ffn_routed_gate_up_swiglu_q4k_q8", "gfx1250",
    kModelQ4KQ8Configs, IREE_ARRAYSIZE(kModelQ4KQ8Configs),
};

constexpr CxxModelProduct kModelNvFp4Wave32 = {
    "nvfp4_wave32", "nvfp4_matrix", "gfx1100", nullptr, 0,
};

constexpr CxxModelProduct kModelNvFp4Wave64 = {
    "nvfp4_wave64", "nvfp4_matrix", "gfx942", nullptr, 0,
};

constexpr CxxModelProduct kModelIq4XsAsyncSingle = {
    "iq4xs_async_single",
    "qwen38_iq4xs_gate_up_swiglu",
    "gfx942",
    kModelIq4XsSingleConfigs,
    IREE_ARRAYSIZE(kModelIq4XsSingleConfigs),
};

constexpr CxxModelProduct kModelIq4XsAsyncDouble = {
    "iq4xs_async_double",
    "qwen38_iq4xs_gate_up_swiglu",
    "gfx942",
    kModelIq4XsDoubleConfigs,
    IREE_ARRAYSIZE(kModelIq4XsDoubleConfigs),
};

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
      {CxxJitPhase::kSourceToPreparedLow, "SourceToPreparedLow"},
      {CxxJitPhase::kCloneHigh, "CloneHigh"},
      {CxxJitPhase::kSourceLow, "SourceLow"},
      {CxxJitPhase::kPreparedLow, "PreparedLow"},
      {CxxJitPhase::kHighToHsaco, "HighToHsaco"},
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
  RegisterCxxJitPhaseBenchmarks("Iq4XsGateUpGfx1151", &kIq4XsGateUpGfx1151);
  RegisterCxxJitPhaseBenchmarks("Iq4XsGateUpGfx942AsyncSingle",
                                &kIq4XsGateUpGfx942AsyncSingle);
  RegisterCxxJitPhaseBenchmarks("Iq4XsGateUpGfx942AsyncDouble",
                                &kIq4XsGateUpGfx942AsyncDouble);
  RegisterCxxJitPhaseBenchmarks("Q4KQ8SwiGluGfx1250", &kQ4KQ8SwiGlu);
  RegisterCxxJitPhaseBenchmarks("ConfiguredWorkgroupStorage",
                                &kConfiguredWorkgroupStorage);
  RegisterCxxJitPhaseBenchmarks("NvFp4Gfx1100", &kNvFp4Gfx1100);
  RegisterCxxJitPhaseBenchmarks("NvFp4Gfx942", &kNvFp4Gfx942);
  return true;
}();

BENCHMARK(CxxModelLibraryImportSerializeIndex)->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(CxxModelLibraryLinkToHsaco, Q4KQ8Gfx1250, &kModelQ4KQ8Gfx1250)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(CxxModelLibraryLinkToHsaco, NvFp4Wave32, &kModelNvFp4Wave32)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(CxxModelLibraryLinkToHsaco, NvFp4Wave64, &kModelNvFp4Wave64)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(CxxModelLibraryLinkToHsaco, Iq4XsAsyncSingle,
                  &kModelIq4XsAsyncSingle)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(CxxModelLibraryLinkToHsaco, Iq4XsAsyncDouble,
                  &kModelIq4XsAsyncDouble)
    ->Unit(::benchmark::kMicrosecond);

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
BENCHMARK_CAPTURE(SourceToHsaco, Iq4XsGateUpGfx1151, &kIq4XsGateUpGfx1151)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, Iq4XsGateUpGfx942AsyncSingle,
                  &kIq4XsGateUpGfx942AsyncSingle)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, Iq4XsGateUpGfx942AsyncDouble,
                  &kIq4XsGateUpGfx942AsyncDouble)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, Q4KQ8SwiGluGfx1250, &kQ4KQ8SwiGlu)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, ConfiguredWorkgroupStorage,
                  &kConfiguredWorkgroupStorage)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, NvFp4Gfx1100, &kNvFp4Gfx1100)
    ->Unit(::benchmark::kMicrosecond);
BENCHMARK_CAPTURE(SourceToHsaco, NvFp4Gfx942, &kNvFp4Gfx942)
    ->Unit(::benchmark::kMicrosecond);

BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, RmsNorm, &kRmsNorm)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, Mxfp8Gfx1250, &kMxfp8)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, Iq4XsGateUpGfx1151,
                  &kIq4XsGateUpGfx1151)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, Iq4XsGateUpGfx942AsyncSingle,
                  &kIq4XsGateUpGfx942AsyncSingle)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, Iq4XsGateUpGfx942AsyncDouble,
                  &kIq4XsGateUpGfx942AsyncDouble)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, Q4KQ8SwiGluGfx1250, &kQ4KQ8SwiGlu)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, ConfiguredWorkgroupStorage,
                  &kConfiguredWorkgroupStorage)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, NvFp4Gfx1100, &kNvFp4Gfx1100)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);
BENCHMARK_CAPTURE(SourceToHsacoColdWorkspace, NvFp4Gfx942, &kNvFp4Gfx942)
    ->Unit(::benchmark::kMicrosecond)
    ->Iterations(1);

}  // namespace
}  // namespace loomc::bench
