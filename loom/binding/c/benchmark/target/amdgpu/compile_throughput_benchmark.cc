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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "loomc/target/amdgpu.h"

namespace {

using loomc::bench::CloneModule;
using loomc::bench::CompileScenario;
using loomc::bench::CreateBenchmarkKernelSource;
using loomc::bench::CreateTextModule;
using loomc::bench::CreateTextSource;
using loomc::bench::CreateWorkspace;
using loomc::bench::DeserializeSource;
using loomc::bench::loom_allocator;
using loomc::bench::ModulePtr;
using loomc::bench::PassProgramPtr;
using loomc::bench::PreparePassProgram;
using loomc::bench::ReadArtifactPrefix;
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
constexpr AmdgpuBenchmarkTarget kGfx1250Target = {"gfx1250"};

class AmdgpuTargetCompileScenario : public TargetCompileScenario {
 public:
  explicit AmdgpuTargetCompileScenario(
      AmdgpuBenchmarkTarget target, iree_host_size_t workspace_block_size = 0)
      : TargetCompileScenario(workspace_block_size), target_(target) {}

 protected:
  iree_status_t SetUpAmdgpuTarget(iree_host_size_t worker_count) {
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
        /*.identity=*/
        {
            /*.processor=*/processor,
        },
    };
    loomc_target_profile_t* raw_profile = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_profile_create_amdgpu(
        target_environment.get(), &profile_options, loom_allocator(),
        &raw_profile)));
    TargetProfilePtr target_profile(raw_profile);

    return SetUpTarget(
        worker_count, std::move(target_environment), std::move(target_profile),
        loomc_make_cstring_view("benchmark-amdgpu-prepared-low"));
  }

  iree_status_t EmitAmdgpuArtifact(WorkspacePtr& workspace, ModulePtr& module,
                                   loomc_string_view_t identifier) {
    loomc_amdgpu_emit_options_t amdgpu_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(amdgpu_options),
        /*.next=*/nullptr,
        /*.runtime_globals=*/LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
    };
    loomc_emit_options_t emit_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(emit_options),
        /*.next=*/&amdgpu_options,
        /*.artifact_format=*/
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
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
        RequireSucceededResult(result.get(), "AMDGPU emission"));

    constexpr uint8_t kElfMagic[] = {0x7F, 'E', 'L', 'F'};
    int64_t artifact_bytes = 0;
    IREE_RETURN_IF_ERROR(ValidateArtifact(
        result.get(), loomc_make_cstring_view(LOOMC_ARTIFACT_ROLE_KERNEL),
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        sizeof(kElfMagic), "AMDGPU HSACO executable", &artifact_bytes));
    const loomc_artifact_t* artifact = loomc::bench::FindArtifact(
        result.get(), loomc_make_cstring_view(LOOMC_ARTIFACT_ROLE_KERNEL),
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
    uint8_t magic[sizeof(kElfMagic)] = {0};
    IREE_RETURN_IF_ERROR(ReadArtifactPrefix(
        artifact, iree_make_byte_span(magic, sizeof(magic))));
    if (std::memcmp(magic, kElfMagic, sizeof(kElfMagic)) != 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU executable is not an ELF image");
    }

    RecordArtifactBytes(artifact_bytes);
    return iree_ok_status();
  }

 private:
  // Immutable target row selected by the benchmark registration.
  AmdgpuBenchmarkTarget target_;
};

enum class QwenAttentionCompilePhase {
  kParse,
  kCloneSource,
  kCanonicalizeSource,
  kCseSource,
  kUnrollSource,
  kCloneUnrolled,
  kCanonicalizeUnrolled,
  kCanonicalizeLastUnrolled,
  kCseUnrolled,
  kSourceLow,
  kPreparedLow,
  kCompileAndEmit,
};

struct QwenAttentionBenchmarkSpec {
  // Compiler boundary measured by one benchmark iteration.
  QwenAttentionCompilePhase phase;

  // Exact AMDGPU target selected for target-lowering phases.
  AmdgpuBenchmarkTarget target;
};

struct ModuleShape {
  // Number of bytes in the generic textual module representation.
  int64_t byte_count = 0;

  // Approximate operation count in the generic textual representation.
  int64_t printed_operation_count = 0;
};

static void ReplaceAll(std::string& text, const std::string& old_value,
                       const std::string& new_value) {
  size_t offset = 0;
  while ((offset = text.find(old_value, offset)) != std::string::npos) {
    text.replace(offset, old_value.size(), new_value);
    offset += new_value.size();
  }
}

static int64_t CountPrintedOperations(const std::string& text) {
  int64_t operation_count = 0;
  size_t line_start = 0;
  while (line_start < text.size()) {
    const size_t line_end = text.find('\n', line_start);
    const size_t line_length =
        (line_end == std::string::npos ? text.size() : line_end) - line_start;
    const size_t content_start = text.find_first_not_of(" \t\r", line_start);
    if (content_start < line_start + line_length) {
      const char first = text[content_start];
      const bool is_comment = first == '/' &&
                              content_start + 1 < line_start + line_length &&
                              text[content_start + 1] == '/';
      if (!is_comment && first != '}' && first != '^' && first != ')' &&
          first != ',' && first != '{') {
        if (first == '%') {
          const size_t assignment = text.find(" = ", content_start);
          if (assignment < line_start + line_length) ++operation_count;
        } else {
          const size_t token_end = text.find_first_of(" \t<({", content_start);
          const size_t bounded_token_end =
              std::min(token_end, line_start + line_length);
          const size_t period = text.find('.', content_start);
          if (period < bounded_token_end) ++operation_count;
        }
      }
    }
    if (line_end == std::string::npos) break;
    line_start = line_end + 1;
  }
  return operation_count;
}

static iree_status_t BuildQwenAttentionSource(
    const loomc_source_t* fixture_source, iree_host_size_t kernel_copy_count,
    std::string* out_text) {
  const loomc_byte_span_t contents = loomc_source_contents(fixture_source);
  const std::string fixture_text((const char*)contents.data,
                                 contents.data_length);
  const size_t kernel_start = fixture_text.find("kernel.def retain");
  if (kernel_start == std::string::npos) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen attention fixture has no retained kernel");
  }

  constexpr const char* kKernelSymbol = "qwen38_attention_prefill_f16_wmma";
  const std::string prefix = fixture_text.substr(0, kernel_start);
  const std::string kernel = fixture_text.substr(kernel_start);
  out_text->clear();
  out_text->reserve(prefix.size() + kernel.size() * kernel_copy_count);
  out_text->append(prefix);
  for (iree_host_size_t i = 0; i < kernel_copy_count; ++i) {
    std::string copy = kernel;
    ReplaceAll(copy, kKernelSymbol,
               std::string(kKernelSymbol) + "_" + std::to_string(i));
    out_text->append(copy);
  }
  return iree_ok_status();
}

static iree_status_t CaptureModuleShape(const loomc_module_t* module,
                                        ModuleShape* out_shape) {
  const loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("qwen-attention-shape.loom"),
      /*.text_presentation=*/LOOMC_MODULE_TEXT_PRESENTATION_GENERIC,
  };
  loomc_source_t* raw_source = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_module_serialize_text_to_source(
      module, &options, loom_allocator(), &raw_source)));
  SourcePtr source(raw_source);
  const loomc_byte_span_t contents = loomc_source_contents(source.get());
  const std::string text((const char*)contents.data, contents.data_length);
  out_shape->byte_count = (int64_t)contents.data_length;
  out_shape->printed_operation_count = CountPrintedOperations(text);
  return iree_ok_status();
}

static iree_status_t PrepareTargetPassProgram(
    loomc_context_t* context, loomc_target_pipeline_kind_t kind,
    loomc_string_view_t identifier, PassProgramPtr* out_pass_program) {
  const loomc_target_pipeline_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/identifier,
      /*.kind=*/kind,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* raw_pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  iree_status_t status =
      to_iree_status(loomc_pass_program_create_from_target_pipeline(
          context, &options, loom_allocator(), &raw_pass_program, &raw_result));
  PassProgramPtr pass_program(raw_pass_program);
  ResultPtr result(raw_result);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(
      RequireSucceededResult(result.get(), "target pipeline preparation"));
  out_pass_program->reset(pass_program.release());
  return iree_ok_status();
}

static iree_status_t PrepareStructuredPassProgram(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const std::string& source_text, loomc_string_view_t pipeline_symbol,
    PassProgramPtr* out_pass_program) {
  ModulePtr pipeline_module;
  IREE_RETURN_IF_ERROR(CreateTextModule(context, workspace,
                                        "qwen-attention-pipeline.loom",
                                        source_text, &pipeline_module));
  loomc_pass_program_t* raw_pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  iree_status_t status =
      to_iree_status(loomc_pass_program_create_from_module_symbol(
          pipeline_module.get(), pipeline_symbol, /*options=*/nullptr,
          loom_allocator(), &raw_pass_program, &raw_result));
  PassProgramPtr pass_program(raw_pass_program);
  ResultPtr result(raw_result);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(
      RequireSucceededResult(result.get(), "pass program preparation"));
  out_pass_program->reset(pass_program.release());
  return iree_ok_status();
}

class QwenAttentionCompileScenario final : public AmdgpuTargetCompileScenario {
 public:
  QwenAttentionCompileScenario(QwenAttentionCompilePhase phase,
                               iree_host_size_t kernel_copy_count,
                               AmdgpuBenchmarkTarget target)
      : AmdgpuTargetCompileScenario(target),
        phase_(phase),
        kernel_copy_count_(std::max<iree_host_size_t>(kernel_copy_count, 1)) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpAmdgpuTarget(worker_count));

    SourcePtr fixture_source;
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("qwen38_attention_prefill_wmma.loom"),
        &fixture_source));
    IREE_RETURN_IF_ERROR(BuildQwenAttentionSource(
        fixture_source.get(), kernel_copy_count_, &source_text_));
    kernel_symbols_.reserve(kernel_copy_count_);
    target_specializations_.reserve(kernel_copy_count_);
    for (iree_host_size_t i = 0; i < kernel_copy_count_; ++i) {
      kernel_symbols_.push_back("qwen38_attention_prefill_f16_wmma_" +
                                std::to_string(i));
    }
    for (const std::string& kernel_symbol : kernel_symbols_) {
      target_specializations_.push_back({
          /*.function_symbol=*/loomc_make_string_view(kernel_symbol.data(),
                                                      kernel_symbol.size()),
          /*.target_profile=*/target_profile(),
      });
    }
    IREE_RETURN_IF_ERROR(CreateTextSource("qwen38_attention_prefill_wmma.loom",
                                          source_text_, &source_));
    source_shape_.byte_count = (int64_t)source_text_.size();
    source_shape_.printed_operation_count =
        CountPrintedOperations(source_text_);

    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(),
                                           template_workspace_.get(),
                                           source_.get(), &source_template_));

    switch (phase_) {
      case QwenAttentionCompilePhase::kCanonicalizeSource:
        return PreparePassProgram(context_.get(),
                                  loomc_make_cstring_view("canonicalize"),
                                  &pass_program_);
      case QwenAttentionCompilePhase::kCseSource:
        return PreparePassProgram(
            context_.get(), loomc_make_cstring_view("cse"), &pass_program_);
      case QwenAttentionCompilePhase::kUnrollSource:
        return PreparePassProgram(context_.get(),
                                  loomc_make_cstring_view("unroll-scf-for"),
                                  &pass_program_);
      case QwenAttentionCompilePhase::kCloneUnrolled:
      case QwenAttentionCompilePhase::kCanonicalizeUnrolled:
      case QwenAttentionCompilePhase::kCanonicalizeLastUnrolled:
      case QwenAttentionCompilePhase::kCseUnrolled: {
        IREE_RETURN_IF_ERROR(PrepareUnrolledTemplate());
        if (phase_ == QwenAttentionCompilePhase::kCanonicalizeUnrolled) {
          return PreparePassProgram(context_.get(),
                                    loomc_make_cstring_view("canonicalize"),
                                    &pass_program_);
        }
        if (phase_ == QwenAttentionCompilePhase::kCanonicalizeLastUnrolled) {
          const std::string pipeline_text =
              "pass.pipeline<module> @qwen_attention_last pipeline {\n"
              "  for func {\n"
              "    where name(value = \"" +
              kernel_symbols_.back() +
              "\") {\n"
              "      canonicalize\n"
              "    }\n"
              "  }\n"
              "}\n";
          return PrepareStructuredPassProgram(
              context_.get(), template_workspace_.get(), pipeline_text,
              loomc_make_cstring_view("qwen_attention_last"), &pass_program_);
        }
        if (phase_ == QwenAttentionCompilePhase::kCseUnrolled) {
          return PreparePassProgram(
              context_.get(), loomc_make_cstring_view("cse"), &pass_program_);
        }
        return iree_ok_status();
      }
      case QwenAttentionCompilePhase::kSourceLow:
        return PrepareTargetPassProgram(
            context_.get(), LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
            loomc_make_cstring_view("benchmark-qwen-attention-source-low"),
            &pass_program_);
      case QwenAttentionCompilePhase::kParse:
      case QwenAttentionCompilePhase::kCloneSource:
      case QwenAttentionCompilePhase::kPreparedLow:
      case QwenAttentionCompilePhase::kCompileAndEmit:
        return iree_ok_status();
    }
    return iree_ok_status();
  }

  iree_host_size_t job_count() const override { return 1; }

  iree_host_size_t kernel_count_per_job() const override {
    return phase_ == QwenAttentionCompilePhase::kCanonicalizeLastUnrolled
               ? 1
               : kernel_copy_count_;
  }

  iree_status_t WarmUp(iree_host_size_t worker_count) override {
    for (iree_host_size_t worker_ordinal = 0; worker_ordinal < worker_count;
         ++worker_ordinal) {
      IREE_RETURN_IF_ERROR(RunPhase(worker_ordinal, /*capture_shape=*/true));
    }
    return iree_ok_status();
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    return RunPhase(worker_ordinal, /*capture_shape=*/false);
  }

  void SetExtraCounters(::benchmark::State& state) const override {
    const int64_t module_count = state.iterations();
    state.counters["modules/s"] =
        ::benchmark::Counter(module_count, ::benchmark::Counter::kIsRate);
    state.counters["kernel_copies"] = (double)kernel_copy_count_;
    state.counters["input_bytes"] = (double)input_shape_.byte_count;
    state.counters["input_printed_ops"] =
        (double)input_shape_.printed_operation_count;
    state.counters["output_bytes"] = (double)output_shape_.byte_count;
    state.counters["output_printed_ops"] =
        (double)output_shape_.printed_operation_count;
    state.counters["ir_amplification"] =
        input_shape_.printed_operation_count == 0
            ? 0.0
            : (double)output_shape_.printed_operation_count /
                  (double)input_shape_.printed_operation_count;
  }

 private:
  bool UsesUnrolledTemplate() const {
    return phase_ == QwenAttentionCompilePhase::kCloneUnrolled ||
           phase_ == QwenAttentionCompilePhase::kCanonicalizeUnrolled ||
           phase_ == QwenAttentionCompilePhase::kCanonicalizeLastUnrolled ||
           phase_ == QwenAttentionCompilePhase::kCseUnrolled;
  }

  bool UsesTargetPipeline() const {
    return phase_ == QwenAttentionCompilePhase::kSourceLow ||
           phase_ == QwenAttentionCompilePhase::kPreparedLow ||
           phase_ == QwenAttentionCompilePhase::kCompileAndEmit;
  }

  iree_status_t CompileWithPass(WorkspacePtr& workspace, ModulePtr& module,
                                const PassProgramPtr& pass_program,
                                bool specialize_target) {
    const loomc_target_specialization_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.specializations=*/target_specializations_.data(),
        /*.specialization_count=*/target_specializations_.size(),
        /*.target_bindings=*/nullptr,
        /*.target_binding_count=*/0,
    };
    const loomc_compile_options_t compile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(compile_options),
        /*.next=*/specialize_target ? &target_options : nullptr,
        /*.module_name=*/loomc_make_cstring_view("qwen_attention_benchmark"),
        /*.artifact_flags=*/0,
        /*.config_flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
        /*.config_module=*/nullptr,
    };
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_compile_module(
        compiler_.get(), workspace.get(), pass_program.get(), module.get(),
        &compile_options, loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    return RequireSucceededResult(result.get(), "Qwen attention compilation");
  }

  iree_status_t PrepareUnrolledTemplate() {
    PassProgramPtr unroll_program;
    IREE_RETURN_IF_ERROR(PreparePassProgram(
        context_.get(), loomc_make_cstring_view("unroll-scf-for"),
        &unroll_program));
    IREE_RETURN_IF_ERROR(CloneModule(source_template_.get(),
                                     template_workspace_.get(),
                                     &unrolled_template_));
    IREE_RETURN_IF_ERROR(CompileWithPass(template_workspace_,
                                         unrolled_template_, unroll_program,
                                         /*specialize_target=*/false));
    IREE_RETURN_IF_ERROR(
        CaptureModuleShape(unrolled_template_.get(), &unrolled_shape_));
    return iree_ok_status();
  }

  iree_status_t RunPhase(iree_host_size_t worker_ordinal, bool capture_shape) {
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    ModulePtr module;
    if (phase_ == QwenAttentionCompilePhase::kParse) {
      IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(), workspace.get(),
                                             source_.get(), &module));
    } else {
      const loomc_module_t* template_module = UsesUnrolledTemplate()
                                                  ? unrolled_template_.get()
                                                  : source_template_.get();
      IREE_RETURN_IF_ERROR(
          CloneModule(template_module, workspace.get(), &module));
      if (phase_ != QwenAttentionCompilePhase::kCloneSource &&
          phase_ != QwenAttentionCompilePhase::kCloneUnrolled) {
        IREE_RETURN_IF_ERROR(CompileWithPass(workspace, module, pass_program_,
                                             UsesTargetPipeline()));
      }
      if (phase_ == QwenAttentionCompilePhase::kCompileAndEmit) {
        IREE_RETURN_IF_ERROR(EmitAmdgpuArtifact(
            workspace, module,
            loomc_make_cstring_view("qwen_attention_benchmark.hsaco")));
      }
    }

    if (capture_shape) {
      input_shape_ = UsesUnrolledTemplate() ? unrolled_shape_ : source_shape_;
      IREE_RETURN_IF_ERROR(CaptureModuleShape(module.get(), &output_shape_));
    }
    ::benchmark::DoNotOptimize(module.get());
    return iree_ok_status();
  }

  // Compiler boundary measured by each timed invocation.
  QwenAttentionCompilePhase phase_;

  // Number of independently named production kernels in the input module.
  iree_host_size_t kernel_copy_count_ = 0;

  // Stable symbol storage borrowed by target specialization rows.
  std::vector<std::string> kernel_symbols_;

  // Precomputed target rows reused without timed-path allocation.
  std::vector<loomc_target_specialization_t> target_specializations_;

  // Repeated textual module retained for parse-phase invocations.
  std::string source_text_;

  // Shape of the authored repeated module.
  ModuleShape source_shape_;

  // Shape of the pre-unrolled module used by post-unroll passes.
  ModuleShape unrolled_shape_;

  // Shape entering the selected benchmark phase.
  ModuleShape input_shape_;

  // Shape leaving the selected benchmark phase.
  ModuleShape output_shape_;

  // Immutable repeated source shared by parse invocations.
  SourcePtr source_;

  // Setup-only workspace retaining parsed and pre-unrolled templates.
  WorkspacePtr template_workspace_;

  // Immutable parsed authored module cloned by most phases.
  ModulePtr source_template_;

  // Immutable pre-unrolled module cloned by post-unroll phase benchmarks.
  ModulePtr unrolled_template_;
};

enum class QwenMoeCompilePhase {
  kSourceLow,
  kPreparedLow,
  kCompileAndEmit,
};

struct QwenMoeBenchmarkSpec {
  // Compiler boundary measured by one benchmark iteration.
  QwenMoeCompilePhase phase;

  // Exact AMDGPU target selected for the kernel.
  AmdgpuBenchmarkTarget target;
};

class QwenMoeCompileScenario final : public AmdgpuTargetCompileScenario {
 public:
  QwenMoeCompileScenario(QwenMoeCompilePhase phase, iree_host_size_t input_size,
                         AmdgpuBenchmarkTarget target)
      : AmdgpuTargetCompileScenario(target),
        phase_(phase),
        input_size_(std::max<iree_host_size_t>(input_size, 512)) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpAmdgpuTarget(worker_count));
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("qwen3_moe_routed_gate_up_q4k.loom"),
        &source_));
    const loomc_byte_span_t source_contents =
        loomc_source_contents(source_.get());
    const std::string source_text((const char*)source_contents.data,
                                  source_contents.data_length);
    source_shape_.byte_count = (int64_t)source_contents.data_length;
    source_shape_.printed_operation_count = CountPrintedOperations(source_text);

    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(),
                                           template_workspace_.get(),
                                           source_.get(), &template_module_));
    std::ostringstream config_text;
    config_text << "config.def @qwen3_moe.routed_gate_up.input_size = "
                << input_size_ << " : index\n";
    IREE_RETURN_IF_ERROR(CreateTextModule(
        context_.get(), template_workspace_.get(), "qwen3_moe_config.loom",
        config_text.str(), &config_module_));

    if (phase_ == QwenMoeCompilePhase::kSourceLow) {
      return PrepareTargetPassProgram(
          context_.get(), LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
          loomc_make_cstring_view("benchmark-qwen-moe-source-low"),
          &pass_program_);
    }
    return iree_ok_status();
  }

  iree_host_size_t job_count() const override { return 1; }

  iree_status_t WarmUp(iree_host_size_t worker_count) override {
    for (iree_host_size_t worker_ordinal = 0; worker_ordinal < worker_count;
         ++worker_ordinal) {
      IREE_RETURN_IF_ERROR(RunPhase(worker_ordinal, /*capture_shape=*/true));
    }
    return iree_ok_status();
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    return RunPhase(worker_ordinal, /*capture_shape=*/false);
  }

  void SetExtraCounters(::benchmark::State& state) const override {
    state.counters["input_size"] = (double)input_size_;
    state.counters["input_bytes"] = (double)source_shape_.byte_count;
    state.counters["input_printed_ops"] =
        (double)source_shape_.printed_operation_count;
    state.counters["output_bytes"] = (double)output_shape_.byte_count;
    state.counters["output_printed_ops"] =
        (double)output_shape_.printed_operation_count;
    state.counters["ir_amplification"] =
        source_shape_.printed_operation_count == 0
            ? 0.0
            : (double)output_shape_.printed_operation_count /
                  (double)source_shape_.printed_operation_count;
  }

 private:
  iree_status_t RunPhase(iree_host_size_t worker_ordinal, bool capture_shape) {
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    ModulePtr module;
    IREE_RETURN_IF_ERROR(
        CloneModule(template_module_.get(), workspace.get(), &module));

    const loomc_target_specialization_t specialization = {
        /*.function_symbol=*/loomc_make_cstring_view(
            "qwen3_moe_routed_gate_up_swiglu_q4k_q8"),
        /*.target_profile=*/target_profile(),
    };
    const loomc_target_specialization_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.specializations=*/&specialization,
        /*.specialization_count=*/1,
        /*.target_bindings=*/nullptr,
        /*.target_binding_count=*/0,
    };
    const loomc_compile_options_t compile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(compile_options),
        /*.next=*/&target_options,
        /*.module_name=*/loomc_make_cstring_view("qwen_moe_benchmark"),
        /*.artifact_flags=*/0,
        /*.config_flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
        /*.config_module=*/config_module_.get(),
    };
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_compile_module(
        compiler_.get(), workspace.get(), pass_program_.get(), module.get(),
        &compile_options, loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "Qwen MoE compilation"));

    if (phase_ == QwenMoeCompilePhase::kCompileAndEmit) {
      IREE_RETURN_IF_ERROR(EmitAmdgpuArtifact(
          workspace, module,
          loomc_make_cstring_view("qwen_moe_benchmark.hsaco")));
    }
    if (capture_shape) {
      IREE_RETURN_IF_ERROR(CaptureModuleShape(module.get(), &output_shape_));
    }
    ::benchmark::DoNotOptimize(module.get());
    return iree_ok_status();
  }

  // Compiler boundary measured by each timed invocation.
  QwenMoeCompilePhase phase_;

  // Q4_K contraction width controlling the unrolled iteration count.
  iree_host_size_t input_size_ = 0;

  // Shape of the authored module before config materialization.
  ModuleShape source_shape_;

  // Shape of the compiled module after the selected pipeline boundary.
  ModuleShape output_shape_;

  // Immutable authored source shared by setup and parse operations.
  SourcePtr source_;

  // Setup-only workspace retaining parsed source and config modules.
  WorkspacePtr template_workspace_;

  // Immutable parsed production kernel cloned by each invocation.
  ModulePtr template_module_;

  // Immutable exact input-size config applied before each pass program.
  ModulePtr config_module_;
};

class AmdgpuI32ChainScenario final : public AmdgpuTargetCompileScenario {
 public:
  AmdgpuI32ChainScenario(
      iree_host_size_t job_count,
      std::initializer_list<iree_host_size_t> operation_counts,
      AmdgpuBenchmarkTarget target, iree_host_size_t workspace_block_size = 0)
      : AmdgpuTargetCompileScenario(target, workspace_block_size),
        job_count_(std::max<iree_host_size_t>(job_count, 1)) {
    operation_counts_.reserve(operation_counts.size());
    for (iree_host_size_t operation_count : operation_counts) {
      operation_count = std::max<iree_host_size_t>(operation_count, 1);
      operation_counts_.push_back({operation_count, {}});
    }
  }

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpAmdgpuTarget(worker_count));
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("i32_memory_chain.loom"), &source_));
    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(),
                                           template_workspace_.get(),
                                           source_.get(), &template_module_));
    for (OperationCount& operation_count : operation_counts_) {
      std::ostringstream config_text;
      config_text << "config.def @benchmark.workgroup_size = 64 : index\n"
                  << "config.def @benchmark.operation_count = "
                  << operation_count.value << " : index\n";
      IREE_RETURN_IF_ERROR(CreateTextModule(
          context_.get(), template_workspace_.get(), "i32_chain_config.loom",
          config_text.str(), &operation_count.config_module));
    }
    return iree_ok_status();
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
    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("i32_memory_chain"),
        loomc_make_cstring_view("amdgpu_i32_chain"),
        operation_count.config_module.get(),
        LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED));
    return EmitAmdgpuArtifact(
        workspace, module, loomc_make_cstring_view("i32_memory_chain.hsaco"));
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

    // Immutable typed config module shared by compile invocations.
    ModulePtr config_module;
  };

  // Number of kernels compiled by each benchmark iteration.
  iree_host_size_t job_count_ = 0;

  // Cyclic sequence of arithmetic-chain sizes compiled by each iteration.
  std::vector<OperationCount> operation_counts_;

  // Compact targetless source shared by all invocations.
  SourcePtr source_;

  // Setup-only workspace retaining the parsed template module.
  WorkspacePtr template_workspace_;

  // Immutable parsed template cloned into worker workspaces.
  ModulePtr template_module_;
};

static std::string BuildAmdgpuClusterAsyncDisjointSource(
    iree_host_size_t transfer_count) {
  constexpr uint64_t kPacketByteCount = 16;
  constexpr uint64_t kWorkgroupSize = 64;
  const uint64_t lane_stride = transfer_count * kPacketByteCount;
  const uint64_t storage_byte_count = kWorkgroupSize * lane_stride;

  std::ostringstream source;
  source << R"(
kernel.def export("cluster_async_disjoint") @cluster_async_disjoint() {
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %workgroup_size = index.constant 64 : index
  kernel.launch.config workgroups(%one, %two, %one) workgroup_size(%workgroup_size, %one, %one) cluster_size(%one, %two, %one) : index
} launch(%input: buffer) {
  %zero = index.constant 0 : offset
  %lane_stride = index.constant )"
         << lane_stride << R"( : index
  %storage_bytes = index.constant )"
         << storage_byte_count << R"( : offset
  %participants = scalar.constant 3 : i32
  %lane = kernel.workitem.id<x> : index
  %lane_base = index.mul %lane, %lane_stride : index
  %global = buffer.assume.memory_space<global> %input : buffer
  %source = buffer.view %global[%zero] : buffer -> view<16xi8>
  %scratch = buffer.alloca<workgroup> align(16) %storage_bytes : buffer
)";

  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    source << "  %slot" << i << " = index.constant " << i * kPacketByteCount
           << " : index\n"
           << "  %dest_index" << i << " = index.add %lane_base, %slot" << i
           << " : index\n"
           << "  %dest_offset" << i << " = index.cast %dest_index" << i
           << " : index to offset\n"
           << "  %dest" << i << " = buffer.view %scratch[%dest_offset" << i
           << "] : buffer -> view<16xi8>\n";
  }
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    source << "  %copy" << i
           << " = kernel.async.cluster.gather %source to %dest" << i
           << " using %participants {cache_scope = device, cache_temporal = "
              "regular} : view<16xi8> to view<16xi8>, i32 -> "
              "kernel.async.token\n";
  }

  source << "  %group = kernel.async.group ";
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    if (i != 0) source << ", ";
    source << "%copy" << i;
  }
  source << " : ";
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    if (i != 0) source << ", ";
    source << "kernel.async.token";
  }
  source << R"( -> kernel.async.group
  kernel.async.wait %group {newer_groups = 0} : kernel.async.group
  kernel.return
}
)";
  return source.str();
}

class AmdgpuClusterAsyncDisjointScenario final
    : public AmdgpuTargetCompileScenario {
 public:
  AmdgpuClusterAsyncDisjointScenario(iree_host_size_t job_count,
                                     iree_host_size_t transfer_count,
                                     AmdgpuBenchmarkTarget target)
      : AmdgpuTargetCompileScenario(target),
        job_count_(std::max<iree_host_size_t>(job_count, 1)),
        transfer_count_(std::max<iree_host_size_t>(transfer_count, 1)) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpAmdgpuTarget(worker_count));
    IREE_RETURN_IF_ERROR(CreateTextSource(
        "cluster_async_disjoint.loom",
        BuildAmdgpuClusterAsyncDisjointSource(transfer_count_), &source_));
    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    return DeserializeSource(context_.get(), template_workspace_.get(),
                             source_.get(), &template_module_);
  }

  iree_host_size_t job_count() const override { return job_count_; }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    ModulePtr module;
    IREE_RETURN_IF_ERROR(
        CloneModule(template_module_.get(), workspace.get(), &module));
    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("cluster_async_disjoint"),
        loomc_make_cstring_view("amdgpu_cluster_async_disjoint"),
        /*config_module=*/nullptr, /*config_flags=*/0));
    return EmitAmdgpuArtifact(
        workspace, module,
        loomc_make_cstring_view("cluster_async_disjoint.hsaco"));
  }

  void SetExtraCounters(::benchmark::State& state) const override {
    state.counters["transfer_count"] = (double)transfer_count_;
  }

 private:
  // Number of kernels compiled by each benchmark iteration.
  iree_host_size_t job_count_ = 0;

  // Number of pairwise-disjoint cluster transfers in each kernel.
  iree_host_size_t transfer_count_ = 0;

  // Generated targetless source shared by all invocations.
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

static std::unique_ptr<CompileScenario>
CreateAmdgpuClusterAsyncDisjointScenario(const ::benchmark::State& state,
                                         const void* user_data) {
  const auto* target = static_cast<const AmdgpuBenchmarkTarget*>(user_data);
  return std::make_unique<AmdgpuClusterAsyncDisjointScenario>(
      (iree_host_size_t)state.range(1), (iree_host_size_t)state.range(2),
      *target);
}

static std::unique_ptr<CompileScenario> CreateQwenAttentionCompileScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* spec = static_cast<const QwenAttentionBenchmarkSpec*>(user_data);
  return std::make_unique<QwenAttentionCompileScenario>(
      spec->phase, (iree_host_size_t)state.range(1), spec->target);
}

static std::unique_ptr<CompileScenario> CreateQwenMoeCompileScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* spec = static_cast<const QwenMoeBenchmarkSpec*>(user_data);
  return std::make_unique<QwenMoeCompileScenario>(
      spec->phase, (iree_host_size_t)state.range(1), spec->target);
}

constexpr QwenAttentionBenchmarkSpec kQwenAttentionParse = {
    QwenAttentionCompilePhase::kParse, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionCloneSource = {
    QwenAttentionCompilePhase::kCloneSource, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionCanonicalizeSource = {
    QwenAttentionCompilePhase::kCanonicalizeSource, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionCseSource = {
    QwenAttentionCompilePhase::kCseSource, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionUnrollSource = {
    QwenAttentionCompilePhase::kUnrollSource, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionCloneUnrolled = {
    QwenAttentionCompilePhase::kCloneUnrolled, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionCanonicalizeUnrolled = {
    QwenAttentionCompilePhase::kCanonicalizeUnrolled, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionCanonicalizeLastUnrolled = {
    QwenAttentionCompilePhase::kCanonicalizeLastUnrolled, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionCseUnrolled = {
    QwenAttentionCompilePhase::kCseUnrolled, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionSourceLow = {
    QwenAttentionCompilePhase::kSourceLow, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionPreparedLow = {
    QwenAttentionCompilePhase::kPreparedLow, kGfx1100Target};
constexpr QwenAttentionBenchmarkSpec kQwenAttentionCompileAndEmit = {
    QwenAttentionCompilePhase::kCompileAndEmit, kGfx1100Target};
constexpr QwenMoeBenchmarkSpec kQwenMoeSourceLow = {
    QwenMoeCompilePhase::kSourceLow, kGfx1100Target};
constexpr QwenMoeBenchmarkSpec kQwenMoePreparedLow = {
    QwenMoeCompilePhase::kPreparedLow, kGfx1100Target};
constexpr QwenMoeBenchmarkSpec kQwenMoeCompileAndEmit = {
    QwenMoeCompilePhase::kCompileAndEmit, kGfx1100Target};

static void BM_AmdgpuQwenAttention(::benchmark::State& state,
                                   const QwenAttentionBenchmarkSpec* spec) {
  RunCompileBenchmarkDirect(state, CreateQwenAttentionCompileScenario, spec);
}

BENCHMARK_CAPTURE(BM_AmdgpuQwenAttention, SourceLowSmokeGfx1100,
                  &kQwenAttentionSourceLow)
    ->Args({1, 1})
    ->UseRealTime();

#define LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(name, spec) \
  BENCHMARK_CAPTURE(BM_AmdgpuQwenAttention, name, spec) \
      ->Args({1, 1})                                    \
      ->Args({1, 2})                                    \
      ->Args({1, 4})                                    \
      ->Args({1, 8})                                    \
      ->UseRealTime()

LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(ParseGfx1100, &kQwenAttentionParse);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(CloneSourceGfx1100,
                                    &kQwenAttentionCloneSource);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(CanonicalizeSourceGfx1100,
                                    &kQwenAttentionCanonicalizeSource);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(CseSourceGfx1100, &kQwenAttentionCseSource);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(UnrollSourceGfx1100,
                                    &kQwenAttentionUnrollSource);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(CloneUnrolledGfx1100,
                                    &kQwenAttentionCloneUnrolled);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(CanonicalizeUnrolledGfx1100,
                                    &kQwenAttentionCanonicalizeUnrolled);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(CanonicalizeLastUnrolledGfx1100,
                                    &kQwenAttentionCanonicalizeLastUnrolled);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(CseUnrolledGfx1100,
                                    &kQwenAttentionCseUnrolled);
LOOM_BENCHMARK_QWEN_ATTENTION_PHASE(SourceLowGfx1100, &kQwenAttentionSourceLow);

#undef LOOM_BENCHMARK_QWEN_ATTENTION_PHASE

BENCHMARK_CAPTURE(BM_AmdgpuQwenAttention, PreparedLowGfx1100,
                  &kQwenAttentionPreparedLow)
    ->Args({1, 1})
    ->Args({1, 2})
    ->Args({1, 4})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuQwenAttention, CompileAndEmitGfx1100,
                  &kQwenAttentionCompileAndEmit)
    ->Args({1, 1})
    ->Args({1, 2})
    ->UseRealTime();

static void BM_AmdgpuQwenMoe(::benchmark::State& state,
                             const QwenMoeBenchmarkSpec* spec) {
  RunCompileBenchmarkDirect(state, CreateQwenMoeCompileScenario, spec);
}

#define LOOM_BENCHMARK_QWEN_MOE_PHASE(name, spec) \
  BENCHMARK_CAPTURE(BM_AmdgpuQwenMoe, name, spec) \
      ->Args({1, 1024})                           \
      ->Args({1, 2048})                           \
      ->Args({1, 4096})                           \
      ->Args({1, 8192})                           \
      ->Args({1, 16384})                          \
      ->Args({1, 32768})                          \
      ->UseRealTime()

LOOM_BENCHMARK_QWEN_MOE_PHASE(SourceLowGfx1100, &kQwenMoeSourceLow);
LOOM_BENCHMARK_QWEN_MOE_PHASE(PreparedLowGfx1100, &kQwenMoePreparedLow);

#undef LOOM_BENCHMARK_QWEN_MOE_PHASE

BENCHMARK_CAPTURE(BM_AmdgpuQwenMoe, CompileAndEmitGfx1100,
                  &kQwenMoeCompileAndEmit)
    ->Args({1, 1024})
    ->Args({1, 4096})
    ->Args({1, 16384})
    ->UseRealTime();

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

static void BM_AmdgpuClusterAsyncDisjointSmoke(
    ::benchmark::State& state, const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuClusterAsyncDisjointScenario,
                            target);
}
BENCHMARK_CAPTURE(BM_AmdgpuClusterAsyncDisjointSmoke, Gfx1250, &kGfx1250Target)
    ->Args({1, 1, 1})
    ->Args({1, 1, 4})
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

static void BM_AmdgpuClusterAsyncDisjointCold(
    ::benchmark::State& state, const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirectCold(state, CreateAmdgpuClusterAsyncDisjointScenario,
                                target);
}
BENCHMARK_CAPTURE(BM_AmdgpuClusterAsyncDisjointCold, Gfx1250, &kGfx1250Target)
    ->Args({1, 1, 1})
    ->Args({1, 1, 4})
    ->Args({1, 1, 64})
    ->Iterations(1)
    ->UseRealTime();

static void BM_AmdgpuClusterAsyncDisjoint(::benchmark::State& state,
                                          const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuClusterAsyncDisjointScenario,
                            target);
}
BENCHMARK_CAPTURE(BM_AmdgpuClusterAsyncDisjoint, Gfx1250, &kGfx1250Target)
    ->Args({1, 1, 1})
    ->Args({1, 1, 4})
    ->Args({1, 1, 16})
    ->Args({1, 1, 64})
    ->UseRealTime();

}  // namespace
