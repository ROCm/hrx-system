// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/benchmark/workload_compile_benchmark.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"

namespace loomc::bench {
namespace {

enum class AttentionCompilePhase {
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

struct AttentionBenchmarkSpec {
  // Compiler boundary measured by one benchmark iteration.
  AttentionCompilePhase phase;

  // Target profile and emitter used for target-dependent phases.
  const WorkloadCompileTarget* target;

  // Target implementation of the functional attention workload.
  CompileWorkload workload;
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

static iree_status_t BuildAttentionSource(const loomc_source_t* fixture_source,
                                          iree_host_size_t kernel_copy_count,
                                          const char* kernel_symbol,
                                          std::string* out_text) {
  const loomc_byte_span_t contents = loomc_source_contents(fixture_source);
  const std::string fixture_text((const char*)contents.data,
                                 contents.data_length);
  const size_t kernel_start = fixture_text.find("kernel.def retain");
  if (kernel_start == std::string::npos) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "attention fixture has no retained kernel");
  }

  const std::string prefix = fixture_text.substr(0, kernel_start);
  const std::string kernel = fixture_text.substr(kernel_start);
  out_text->clear();
  out_text->reserve(prefix.size() + kernel.size() * kernel_copy_count);
  out_text->append(prefix);
  for (iree_host_size_t i = 0; i < kernel_copy_count; ++i) {
    std::string copy = kernel;
    ReplaceAll(copy, kernel_symbol,
               std::string(kernel_symbol) + "_" + std::to_string(i));
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
      /*.identifier=*/loomc_make_cstring_view("attention-shape.loom"),
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
                                        "attention-pipeline.loom", source_text,
                                        &pipeline_module));
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

class AttentionCompileScenario final : public TargetCompileScenario {
 public:
  AttentionCompileScenario(AttentionCompilePhase phase,
                           iree_host_size_t kernel_copy_count,
                           const WorkloadCompileTarget& target,
                           CompileWorkload workload)
      : phase_(phase),
        kernel_copy_count_(std::max<iree_host_size_t>(kernel_copy_count, 1)),
        target_(target),
        workload_(workload) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    TargetEnvironmentPtr target_environment;
    TargetProfilePtr selected_target_profile;
    IREE_RETURN_IF_ERROR(
        target_.CreateTarget(&target_environment, &selected_target_profile));
    IREE_RETURN_IF_ERROR(SetUpTarget(
        worker_count, std::move(target_environment),
        std::move(selected_target_profile), target_.pipeline_identifier()));

    SourcePtr fixture_source;
    IREE_RETURN_IF_ERROR(
        CreateBenchmarkSource(workload_.source, &fixture_source));
    IREE_RETURN_IF_ERROR(
        BuildAttentionSource(fixture_source.get(), kernel_copy_count_,
                             workload_.function_symbol, &source_text_));
    kernel_symbols_.reserve(kernel_copy_count_);
    target_specializations_.reserve(kernel_copy_count_);
    for (iree_host_size_t i = 0; i < kernel_copy_count_; ++i) {
      kernel_symbols_.push_back(std::string(workload_.function_symbol) + "_" +
                                std::to_string(i));
    }
    if (target_profile() != nullptr) {
      for (const std::string& kernel_symbol : kernel_symbols_) {
        target_specializations_.push_back({
            /*.function_symbol=*/loomc_make_string_view(kernel_symbol.data(),
                                                        kernel_symbol.size()),
            /*.target_profile=*/target_profile(),
        });
      }
    }
    const std::string source_identifier(workload_.source.identifier.data,
                                        workload_.source.identifier.size);
    IREE_RETURN_IF_ERROR(
        CreateTextSource(source_identifier, source_text_, &source_));
    source_shape_.byte_count = (int64_t)source_text_.size();
    source_shape_.printed_operation_count =
        CountPrintedOperations(source_text_);

    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(),
                                           template_workspace_.get(),
                                           source_.get(), &source_template_));

    switch (phase_) {
      case AttentionCompilePhase::kCanonicalizeSource:
        return PreparePassProgram(context_.get(),
                                  loomc_make_cstring_view("canonicalize"),
                                  &pass_program_);
      case AttentionCompilePhase::kCseSource:
        return PreparePassProgram(
            context_.get(), loomc_make_cstring_view("cse"), &pass_program_);
      case AttentionCompilePhase::kUnrollSource:
        return PreparePassProgram(context_.get(),
                                  loomc_make_cstring_view("unroll-scf-for"),
                                  &pass_program_);
      case AttentionCompilePhase::kCloneUnrolled:
      case AttentionCompilePhase::kCanonicalizeUnrolled:
      case AttentionCompilePhase::kCanonicalizeLastUnrolled:
      case AttentionCompilePhase::kCseUnrolled: {
        IREE_RETURN_IF_ERROR(PrepareUnrolledTemplate());
        if (phase_ == AttentionCompilePhase::kCanonicalizeUnrolled) {
          return PreparePassProgram(context_.get(),
                                    loomc_make_cstring_view("canonicalize"),
                                    &pass_program_);
        }
        if (phase_ == AttentionCompilePhase::kCanonicalizeLastUnrolled) {
          const std::string pipeline_text =
              "pass.pipeline<module> @attention_last pipeline {\n"
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
              loomc_make_cstring_view("attention_last"), &pass_program_);
        }
        if (phase_ == AttentionCompilePhase::kCseUnrolled) {
          return PreparePassProgram(
              context_.get(), loomc_make_cstring_view("cse"), &pass_program_);
        }
        return iree_ok_status();
      }
      case AttentionCompilePhase::kSourceLow:
        return PrepareTargetPassProgram(
            context_.get(), LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
            loomc_make_cstring_view("benchmark-attention-source-low"),
            &pass_program_);
      case AttentionCompilePhase::kParse:
      case AttentionCompilePhase::kCloneSource:
      case AttentionCompilePhase::kPreparedLow:
      case AttentionCompilePhase::kCompileAndEmit:
        return iree_ok_status();
    }
    return iree_ok_status();
  }

  iree_host_size_t job_count() const override { return 1; }

  iree_host_size_t kernel_count_per_job() const override {
    return phase_ == AttentionCompilePhase::kCanonicalizeLastUnrolled
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
    state.SetLabel(target_.benchmark_name());
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
    return phase_ == AttentionCompilePhase::kCloneUnrolled ||
           phase_ == AttentionCompilePhase::kCanonicalizeUnrolled ||
           phase_ == AttentionCompilePhase::kCanonicalizeLastUnrolled ||
           phase_ == AttentionCompilePhase::kCseUnrolled;
  }

  bool UsesTargetPipeline() const {
    return phase_ == AttentionCompilePhase::kSourceLow ||
           phase_ == AttentionCompilePhase::kPreparedLow ||
           phase_ == AttentionCompilePhase::kCompileAndEmit;
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
        /*.next=*/specialize_target && !target_specializations_.empty()
            ? &target_options
            : nullptr,
        /*.module_name=*/loomc_make_cstring_view("attention_benchmark"),
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
    return RequireSucceededResult(result.get(), "attention compilation");
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
    if (phase_ == AttentionCompilePhase::kParse) {
      IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(), workspace.get(),
                                             source_.get(), &module));
    } else {
      const loomc_module_t* template_module = UsesUnrolledTemplate()
                                                  ? unrolled_template_.get()
                                                  : source_template_.get();
      IREE_RETURN_IF_ERROR(
          CloneModule(template_module, workspace.get(), &module));
      if (phase_ != AttentionCompilePhase::kCloneSource &&
          phase_ != AttentionCompilePhase::kCloneUnrolled) {
        IREE_RETURN_IF_ERROR(CompileWithPass(workspace, module, pass_program_,
                                             UsesTargetPipeline()));
      }
      if (phase_ == AttentionCompilePhase::kCompileAndEmit) {
        int64_t artifact_byte_count = 0;
        IREE_RETURN_IF_ERROR(target_.EmitArtifact(
            target_environment(), workspace.get(), module.get(),
            loomc_make_cstring_view(workload_.artifact_identifier),
            &artifact_byte_count));
        RecordArtifactBytes(artifact_byte_count);
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
  AttentionCompilePhase phase_;

  // Number of independently named production kernels in the input module.
  iree_host_size_t kernel_copy_count_ = 0;

  // Target profile and emitter used for target-dependent phases.
  const WorkloadCompileTarget& target_;

  // Target implementation of the functional attention workload.
  CompileWorkload workload_;

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

enum class InputScalingCompilePhase {
  kSourceLow,
  kPreparedLow,
  kCompileAndEmit,
};

struct InputScalingBenchmarkSpec {
  // Compiler boundary measured by one benchmark iteration.
  InputScalingCompilePhase phase;

  // Target profile and emitter used for target-dependent phases.
  const WorkloadCompileTarget* target;

  // Target implementation of the functional workload.
  InputScalingCompileWorkload workload;
};

class InputScalingCompileScenario final : public TargetCompileScenario {
 public:
  InputScalingCompileScenario(InputScalingCompilePhase phase,
                              iree_host_size_t input_size,
                              const WorkloadCompileTarget& target,
                              InputScalingCompileWorkload workload)
      : phase_(phase),
        input_size_(input_size),
        target_(target),
        workload_(workload) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    TargetEnvironmentPtr target_environment;
    TargetProfilePtr selected_target_profile;
    IREE_RETURN_IF_ERROR(
        target_.CreateTarget(&target_environment, &selected_target_profile));
    IREE_RETURN_IF_ERROR(SetUpTarget(
        worker_count, std::move(target_environment),
        std::move(selected_target_profile), target_.pipeline_identifier()));
    IREE_RETURN_IF_ERROR(CreateBenchmarkSource(workload_.source, &source_));
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
    config_text << "config.def @" << workload_.input_size_config_symbol << " = "
                << input_size_ << " : index\n";
    IREE_RETURN_IF_ERROR(CreateTextModule(
        context_.get(), template_workspace_.get(), "input_size_config.loom",
        config_text.str(), &config_module_));

    if (phase_ == InputScalingCompilePhase::kSourceLow) {
      return PrepareTargetPassProgram(
          context_.get(), LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
          loomc_make_cstring_view("benchmark-input-scaling-source-low"),
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
    state.SetLabel(target_.benchmark_name());
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
        /*.function_symbol=*/
        loomc_make_cstring_view(workload_.function_symbol),
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
        /*.next=*/target_profile() != nullptr ? &target_options : nullptr,
        /*.module_name=*/loomc_make_cstring_view("input_scaling_benchmark"),
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
        RequireSucceededResult(result.get(), "input-scaling compilation"));

    if (phase_ == InputScalingCompilePhase::kCompileAndEmit) {
      int64_t artifact_byte_count = 0;
      IREE_RETURN_IF_ERROR(target_.EmitArtifact(
          target_environment(), workspace.get(), module.get(),
          loomc_make_cstring_view(workload_.artifact_identifier),
          &artifact_byte_count));
      RecordArtifactBytes(artifact_byte_count);
    }
    if (capture_shape) {
      IREE_RETURN_IF_ERROR(CaptureModuleShape(module.get(), &output_shape_));
    }
    ::benchmark::DoNotOptimize(module.get());
    return iree_ok_status();
  }

  // Compiler boundary measured by each timed invocation.
  InputScalingCompilePhase phase_;

  // Contraction width controlling the target implementation's input shape.
  iree_host_size_t input_size_ = 0;

  // Target profile and emitter used for target-dependent phases.
  const WorkloadCompileTarget& target_;

  // Target implementation of the functional workload.
  InputScalingCompileWorkload workload_;

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

static std::unique_ptr<CompileScenario> CreateAttentionCompileScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* spec = static_cast<const AttentionBenchmarkSpec*>(user_data);
  return std::make_unique<AttentionCompileScenario>(
      spec->phase, (iree_host_size_t)state.range(1), *spec->target,
      spec->workload);
}

static std::unique_ptr<CompileScenario> CreateInputScalingCompileScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* spec = static_cast<const InputScalingBenchmarkSpec*>(user_data);
  return std::make_unique<InputScalingCompileScenario>(
      spec->phase, (iree_host_size_t)state.range(1), *spec->target,
      spec->workload);
}

static std::string BuildBenchmarkName(const char* workload_name,
                                      const char* phase_name,
                                      const WorkloadCompileTarget& target) {
  return std::string("BM_") + workload_name + "/" + phase_name + "/" +
         target.benchmark_name();
}

}  // namespace

void RegisterAttentionCompileBenchmarks(const WorkloadCompileTarget& target,
                                        CompileWorkload workload) {
  auto register_phase = [&](AttentionCompilePhase phase, const char* phase_name,
                            std::initializer_list<int64_t> kernel_copy_counts) {
    const AttentionBenchmarkSpec spec = {
        /*.phase=*/phase,
        /*.target=*/&target,
        /*.workload=*/workload,
    };
    const std::string name =
        BuildBenchmarkName("AttentionPrefill", phase_name, target);
    auto* registration = ::benchmark::RegisterBenchmark(
        name.c_str(), [spec](::benchmark::State& state) {
          RunCompileBenchmarkDirect(state, CreateAttentionCompileScenario,
                                    &spec);
        });
    for (int64_t kernel_copy_count : kernel_copy_counts) {
      registration->Args({1, kernel_copy_count});
    }
    registration->UseRealTime();
  };

  register_phase(AttentionCompilePhase::kParse, "Parse", {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kCloneSource, "CloneSource",
                 {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kCanonicalizeSource,
                 "CanonicalizeSource", {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kCseSource, "CseSource", {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kUnrollSource, "UnrollSource",
                 {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kCloneUnrolled, "CloneUnrolled",
                 {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kCanonicalizeUnrolled,
                 "CanonicalizeUnrolled", {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kCanonicalizeLastUnrolled,
                 "CanonicalizeLastUnrolled", {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kCseUnrolled, "CseUnrolled",
                 {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kSourceLow, "SourceLowSmoke", {1});
  register_phase(AttentionCompilePhase::kSourceLow, "SourceLow", {1, 2, 4, 8});
  register_phase(AttentionCompilePhase::kPreparedLow, "PreparedLow", {1, 2, 4});
  register_phase(AttentionCompilePhase::kCompileAndEmit, "CompileAndEmit",
                 {1, 2});
}

void RegisterInputScalingCompileBenchmarks(
    const WorkloadCompileTarget& target, const char* workload_name,
    InputScalingCompileWorkload workload) {
  auto register_phase = [&](InputScalingCompilePhase phase,
                            const char* phase_name,
                            std::initializer_list<int64_t> input_sizes) {
    const InputScalingBenchmarkSpec spec = {
        /*.phase=*/phase,
        /*.target=*/&target,
        /*.workload=*/workload,
    };
    const std::string name =
        BuildBenchmarkName(workload_name, phase_name, target);
    auto* registration = ::benchmark::RegisterBenchmark(
        name.c_str(), [spec](::benchmark::State& state) {
          RunCompileBenchmarkDirect(state, CreateInputScalingCompileScenario,
                                    &spec);
        });
    for (int64_t input_size : input_sizes) {
      registration->Args({1, input_size});
    }
    registration->UseRealTime();
  };

  register_phase(InputScalingCompilePhase::kSourceLow, "SourceLowSmoke",
                 {1024});
  register_phase(InputScalingCompilePhase::kSourceLow, "SourceLow",
                 {1024, 2048, 4096, 8192, 16384, 32768});
  register_phase(InputScalingCompilePhase::kPreparedLow, "PreparedLow",
                 {1024, 2048, 4096, 8192, 16384, 32768});
  register_phase(InputScalingCompilePhase::kCompileAndEmit, "CompileAndEmit",
                 {1024, 4096, 16384});
}

}  // namespace loomc::bench
