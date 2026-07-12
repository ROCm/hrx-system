// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cinttypes>
#include <cstdio>
#include <cstring>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/stages/ideogram4_dit_test_util.h"
#include "experimental/id4/tooling/filesystem.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/benchmark.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(
    string, id4_fixture_dir, "",
    "Directory containing a full or DiT-only Ideogram4 fixture manifest.");
IREE_FLAG(string, id4_plan_output_dir, "",
          "Optional directory receiving benchmark DiT stage plan JSON files.");
IREE_FLAG(string, dit_parameter_format, "fp8_e4m3",
          "DiT parameter format: bf16 or fp8_e4m3.");
IREE_FLAG(string, dit_weight_execution_format, "fp8_compact_rhs",
          "DiT weight execution format: bf16_resident, fp8_compact_rhs, "
          "fp8_compact_rhs_feed_forward_bf16_resident, or "
          "streaming_compact_rhs.");
IREE_FLAG(string, dit_attention_implementation, "online_wmma",
          "DiT attention implementation: streaming, materialized_wmma, "
          "blocked_wmma, or online_wmma.");
IREE_FLAG(string, dit_feed_forward_implementation, "pytorch_parity",
          "DiT feed-forward implementation: fused_product or "
          "pytorch_parity.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT FP8 e4m3 source parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT FP8 e4m3 source parameter scope.");

namespace {

enum class DitBenchmarkIssueMode {
  // Include submission and completion wait in the timed region.
  kEndToEnd,
  // Time submission only and pause timing around completion waits.
  kSubmitOnly,
};

struct DitBenchmarkContext {
  // Live HAL, executable cache, and kernel-cache context selected by flags.
  id4::test::LiveStageContext live;
  // Embedded Loom source library used during stage preparation.
  id4::test::KernelLibraryRef kernel_library;
  // Parameter provider created from standard --parameters= flags.
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  // Loaded DiT stage under benchmark.
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  // Fixture tensors used to configure requests and initialize inputs.
  id4::test::FixtureTensorSet fixture_tensors;
  // Dynamic request dimensions inferred from the fixture.
  id4_ideogram4_dit_request_config_t request = {};
  // DiT parameter source policy selected by benchmark flags.
  id4_ideogram4_dit_parameter_format_t parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  // DiT linear weight execution strategy selected by benchmark flags.
  id4_ideogram4_dit_weight_execution_format_t weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_INVALID;
  // Attention implementation selected for plan and issue benchmarks.
  id4_ideogram4_dit_attention_implementation_t attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;
  // Feed-forward implementation selected for plan and issue benchmarks.
  id4_ideogram4_dit_feed_forward_implementation_t feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT;
  // Diagnostic event counters collected by lifecycle calls.
  id4::test::StageDiagnostics diagnostics = {};
  // Diagnostics sink passed to stage lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink = {};
};

static iree_string_view_t BranchFp8ParameterScope(
    id4::test::Ideogram4DitBranch branch) {
  switch (branch) {
    case id4::test::Ideogram4DitBranch::kConditioned:
      return iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
    case id4::test::Ideogram4DitBranch::kUnconditioned:
      return iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
  }
  return iree_string_view_empty();
}

static iree_string_view_t BranchParameterScope(
    id4::test::Ideogram4DitBranch branch,
    id4::test::Ideogram4DitBranchConfig branch_config,
    id4_ideogram4_dit_parameter_format_t format) {
  switch (format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      return branch_config.parameter_scope;
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      return BranchFp8ParameterScope(branch);
    default:
      return iree_string_view_empty();
  }
}

static iree_status_t ParseDitParameterFormat(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_status_t ParseDitWeightExecutionFormat(
    id4_ideogram4_dit_weight_execution_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_weight_execution_format_parse(
      iree_make_cstring_view(FLAG_dit_weight_execution_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_weight_execution_format"));
}

static iree_status_t ParseDitAttentionImplementation(
    id4_ideogram4_dit_attention_implementation_t* out_implementation) {
  IREE_ASSERT_ARGUMENT(out_implementation);
  iree_string_view_t value =
      iree_make_cstring_view(FLAG_dit_attention_implementation);
  if (iree_string_view_equal(value, IREE_SV("streaming"))) {
    *out_implementation = ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("materialized_wmma"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("blocked_wmma"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("online_wmma"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--dit_attention_implementation must be streaming, materialized_wmma, "
      "blocked_wmma, or online_wmma");
}

static iree_string_view_t AttentionImplementationName(
    id4_ideogram4_dit_attention_implementation_t implementation) {
  switch (implementation) {
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING:
      return IREE_SV("streaming");
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA:
      return IREE_SV("materialized_wmma");
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA:
      return IREE_SV("blocked_wmma");
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA:
      return IREE_SV("online_wmma");
    default:
      return IREE_SV("invalid");
  }
}

static iree_status_t ParseDitFeedForwardImplementation(
    id4_ideogram4_dit_feed_forward_implementation_t* out_implementation) {
  IREE_ASSERT_ARGUMENT(out_implementation);
  iree_string_view_t value =
      iree_make_cstring_view(FLAG_dit_feed_forward_implementation);
  if (iree_string_view_equal(value, IREE_SV("fused_product"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("pytorch_parity"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "--dit_feed_forward_implementation must be "
                          "fused_product or pytorch_parity");
}

static iree_string_view_t FeedForwardImplementationName(
    id4_ideogram4_dit_feed_forward_implementation_t implementation) {
  switch (implementation) {
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT:
      return IREE_SV("fused_product");
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY:
      return IREE_SV("pytorch_parity");
    default:
      return IREE_SV("invalid");
  }
}

static uint64_t CeilMiB(iree_device_size_t byte_length) {
  static constexpr iree_device_size_t kMiB = 1024ull * 1024ull;
  return (uint64_t)((byte_length + kMiB - 1) / kMiB);
}

static void SetDitBenchmarkLabel(iree_benchmark_state_t* benchmark_state,
                                 const DitBenchmarkContext& context,
                                 id4::test::Ideogram4DitBranch branch,
                                 id4_pipeline_plan_statistics_t statistics) {
  const iree_string_view_t branch_name =
      id4::test::Ideogram4DitBranchName(branch);
  const iree_string_view_t parameter_format =
      id4_ideogram4_dit_parameter_format_name(context.parameter_format);
  const iree_string_view_t weight_execution_format =
      id4_ideogram4_dit_weight_execution_format_name(
          context.weight_execution_format);
  const iree_string_view_t attention_implementation =
      AttentionImplementationName(context.attention_implementation);
  const iree_string_view_t feed_forward_implementation =
      FeedForwardImplementationName(context.feed_forward_implementation);
  const uint64_t latent_width = context.request.latent_shape.dims[0];
  const uint64_t latent_height = context.request.latent_shape.dims[1];
  const uint64_t latent_token_count = latent_width * latent_height;
  char label[1024];
  std::snprintf(
      label, sizeof(label),
      "branch=%.*s params=%.*s weights=%.*s attention=%.*s ff=%.*s "
      "text_tokens=%" PRIu32 " latent_tokens=%" PRIu64 " latent=%" PRIu64
      "x%" PRIu64 " param_total=%" PRIu64 "MiB param_largest=%" PRIu64
      "MiB param_source=%" PRIu64 "MiB param_encoded_source=%" PRIu64
      "MiB param_encode_steps=%" PRIhsz " param_load_groups[total=%" PRIhsz
      ",encode=%" PRIhsz "] local_slab=%" PRIu64 "MiB local_hw=%" PRIu64
      "MiB boundary=%" PRIu64 "MiB kernels=%" PRIhsz " dispatches=%" PRIhsz
      " regions=%" PRIhsz,
      static_cast<int>(branch_name.size), branch_name.data,
      static_cast<int>(parameter_format.size), parameter_format.data,
      static_cast<int>(weight_execution_format.size),
      weight_execution_format.data,
      static_cast<int>(attention_implementation.size),
      attention_implementation.data,
      static_cast<int>(feed_forward_implementation.size),
      feed_forward_implementation.data, context.request.text_token_count,
      latent_token_count, latent_width, latent_height,
      CeilMiB(statistics.parameter_slab_byte_length),
      CeilMiB(statistics.largest_parameter_slab_byte_length),
      CeilMiB(statistics.parameter_source_byte_length),
      CeilMiB(statistics.parameter_encoded_source_byte_length),
      statistics.parameter_encode_load_step_count,
      statistics.parameter_load_group_count,
      statistics.parameter_encode_load_group_count,
      CeilMiB(statistics.memory_slab_byte_length),
      CeilMiB(statistics.memory_slab_high_water_mark),
      CeilMiB(statistics.boundary_tensor_byte_length), statistics.kernel_count,
      statistics.dispatch_count, statistics.region_count);
  iree_benchmark_set_label(benchmark_state, label);
}

static iree_status_t WritePlanJsonIfRequested(
    id4::test::Ideogram4DitBranch branch, const id4_pipeline_plan_t* plan) {
  iree_string_view_t output_dir =
      iree_make_cstring_view(FLAG_id4_plan_output_dir);
  if (iree_string_view_is_empty(output_dir)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      id4_tooling_ensure_directory(output_dir, iree_allocator_system()));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = id4_pipeline_plan_format_json(plan, &builder);
  iree_string_view_t path = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = id4_tooling_format_child_path(
        output_dir, id4::test::Ideogram4DitBranchPlanFileName(branch),
        iree_allocator_system(), &path);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        path, iree_make_const_byte_span(json.data, json.size),
        iree_allocator_system());
  }
  id4_tooling_free_path(&path, iree_allocator_system());
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t CreateDitStage(const id4::test::LiveStageContext& live,
                                    id4::test::Ideogram4DitBranch selected,
                                    id4::test::Ideogram4DitBranchConfig branch,
                                    id4_ideogram4_dit_parameter_format_t format,
                                    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = live.device_group.get();
  services.executable_cache = live.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_parameter_source_rule_list_t source_rules;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      format, *id4_ideogram4_dit_program_ideogram4_model_config(),
      BranchFp8ParameterScope(selected), iree_allocator_system(),
      &source_rules));

  id4_ideogram4_dit_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = live.kernel_cache.get();
  create_options.parameter_scope =
      BranchParameterScope(selected, branch, format);
  create_options.parameter_source_rule_count = source_rules.count;
  create_options.parameter_source_rules = source_rules.values;
  create_options.model = *id4_ideogram4_dit_program_ideogram4_model_config();
  iree_status_t status = id4_ideogram4_dit_stage_create(
      &create_options, iree_allocator_system(), out_stage);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &source_rules, iree_allocator_system());
  return status;
}

static iree_status_t LoadFixtureAndConfigureRequest(
    DitBenchmarkContext* context, id4::test::Ideogram4DitBranchConfig branch) {
  const iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  if (iree_string_view_is_empty(fixture_directory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--id4_fixture_dir is required");
  }
  IREE_RETURN_IF_ERROR(id4::test::LoadFixtureTensors(
      fixture_directory, &context->fixture_tensors));
  return id4::test::Ideogram4DitConfigureRequestFromFixture(
      context->fixture_tensors, branch, &context->request);
}

static iree_status_t CreateLoadedDitStageContext(
    id4::test::Ideogram4DitBranch branch, DitBenchmarkContext* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  const id4::test::Ideogram4DitBranchConfig branch_config =
      id4::test::Ideogram4DitBranchConfigFor(branch);
  IREE_RETURN_IF_ERROR(ParseDitParameterFormat(&out_context->parameter_format));
  IREE_RETURN_IF_ERROR(
      ParseDitWeightExecutionFormat(&out_context->weight_execution_format));
  out_context->diagnostics_sink =
      id4::test::DiagnosticsSink(&out_context->diagnostics);
  IREE_RETURN_IF_ERROR(
      ParseDitAttentionImplementation(&out_context->attention_implementation));
  IREE_RETURN_IF_ERROR(ParseDitFeedForwardImplementation(
      &out_context->feed_forward_implementation));
  IREE_RETURN_IF_ERROR(
      id4::test::CreateLiveStageContextFromFlags(&out_context->live));
  IREE_RETURN_IF_ERROR(
      LoadFixtureAndConfigureRequest(out_context, branch_config));
  IREE_RETURN_IF_ERROR(CreateDitStage(out_context->live, branch, branch_config,
                                      out_context->parameter_format,
                                      out_context->stage.out()));

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &out_context->diagnostics_sink;
  return id4_pipeline_stage_load(out_context->stage.get(), &load_options);
}

static iree_status_t AttachDitPreparationInputs(
    id4::test::Ideogram4DitBranch branch, DitBenchmarkContext* context) {
  IREE_ASSERT_ARGUMENT(context);
  const id4::test::Ideogram4DitBranchConfig branch_config =
      id4::test::Ideogram4DitBranchConfigFor(branch);
  const iree_string_view_t fp8_parameter_scope =
      BranchFp8ParameterScope(branch);
  IREE_RETURN_IF_ERROR(
      id4::test::CreateEmbeddedKernelLibrary(context->kernel_library.out()));
  if (context->parameter_format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16) {
    return id4::test::CreateParameterProviderFromFlags(
        branch_config.parameter_scope, context->parameter_provider.out());
  }
  return id4::test::CreateParameterProviderFromFlags(
      fp8_parameter_scope, context->parameter_provider.out());
}

static iree_status_t CreateLoadedDitBenchmarkContext(
    id4::test::Ideogram4DitBranch branch, DitBenchmarkContext* out_context) {
  IREE_RETURN_IF_ERROR(CreateLoadedDitStageContext(branch, out_context));
  return AttachDitPreparationInputs(branch, out_context);
}

static iree_status_t CreateDitPlan(DitBenchmarkContext* context,
                                   id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = nullptr;

  id4_ideogram4_dit_stage_plan_options_t dit_options;
  std::memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request = context->request;
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  dit_options.weight_execution_format = context->weight_execution_format;
  dit_options.attention_implementation = context->attention_implementation;
  dit_options.feed_forward_implementation =
      context->feed_forward_implementation;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_plan(context->stage.get(), &plan_options, out_plan);
}

static iree_status_t PrepareDitBundle(DitBenchmarkContext* context,
                                      const id4_pipeline_plan_t* plan,
                                      iree_hal_semaphore_t* prepare_semaphore,
                                      uint64_t signal_value,
                                      id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = nullptr;

  id4::test::SemaphoreListStorage signal;
  signal.semaphore = prepare_semaphore;
  signal.payload_value = signal_value;

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_policy = id4_pipeline_stage_parameters(
      id4_pipeline_checkpoint_parameter_source(
          context->parameter_provider.get()),
      ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT,
      /*maximum_parameter_window_byte_length=*/0);
  prepare_options.kernel_library = context->kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal.list();
  prepare_options.command_buffer_mode = context->live.command_buffer_mode;
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_prepare(context->stage.get(), plan,
                                    &prepare_options, out_bundle);
}

static iree_status_t IssueDitBundle(
    DitBenchmarkContext* context, id4_pipeline_bundle_t* bundle,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(bundle);

  id4::test::SemaphoreListStorage wait;
  wait.semaphore = wait_semaphore;
  wait.payload_value = wait_value;
  id4::test::SemaphoreListStorage signal;
  signal.semaphore = signal_semaphore;
  signal.payload_value = signal_value;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.execution_segment_submission_window = 1;
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.wait_semaphore_list = wait.list();
  issue_options.signal_semaphore_list = signal.list();
  issue_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_issue(context->stage.get(), bundle, &issue_options);
}

static iree_status_t WaitForSemaphore(iree_hal_semaphore_t* semaphore,
                                      uint64_t payload_value) {
  return iree_hal_semaphore_wait(semaphore, payload_value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static const iree_benchmark_def_t* RegisterDitBenchmark(
    iree_string_view_t name, iree_benchmark_fn_t run,
    iree_benchmark_unit_t time_unit) {
  iree_benchmark_def_t* benchmark = iree_make_function_benchmark(run);
  benchmark->flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME;
  benchmark->time_unit = time_unit;
  return iree_benchmark_register(name, benchmark);
}

#define ID4_DIT_BENCHMARK_REGISTER(name, time_unit)                 \
  static const iree_benchmark_def_t* name##_registration            \
      IREE_ATTRIBUTE_UNUSED =                                       \
          RegisterDitBenchmark(iree_make_cstring_view(#name), name, \
                               IREE_BENCHMARK_UNIT_##time_unit)

static iree_status_t RunPlanBenchmark(iree_benchmark_state_t* benchmark_state,
                                      id4::test::Ideogram4DitBranch branch) {
  DitBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedDitStageContext(branch, &context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      warm_plan;
  IREE_RETURN_IF_ERROR(CreateDitPlan(&context, warm_plan.out()));
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(warm_plan.get());
  IREE_RETURN_IF_ERROR(WritePlanJsonIfRequested(branch, warm_plan.get()));
  warm_plan.reset();

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(CreateDitPlan(&context, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count *
                           context.request.latent_shape.dims[0] *
                           context.request.latent_shape.dims[1]));
  SetDitBenchmarkLabel(benchmark_state, context, branch, statistics);
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4DitStagePlanConditionedFixture) {
  return RunPlanBenchmark(benchmark_state,
                          id4::test::Ideogram4DitBranch::kConditioned);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStagePlanConditionedFixture,
                           MICROSECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStagePlanUnconditionedFixture) {
  return RunPlanBenchmark(benchmark_state,
                          id4::test::Ideogram4DitBranch::kUnconditioned);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStagePlanUnconditionedFixture,
                           MICROSECOND);

static iree_status_t RunPrepareBenchmark(
    iree_benchmark_state_t* benchmark_state,
    id4::test::Ideogram4DitBranch branch) {
  DitBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedDitBenchmarkContext(branch, &context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateDitPlan(&context, plan.out()));
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan.get());
  IREE_RETURN_IF_ERROR(WritePlanJsonIfRequested(branch, plan.get()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  uint64_t prepare_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      warm_bundle;
  IREE_RETURN_IF_ERROR(PrepareDitBundle(&context, plan.get(),
                                        prepare_semaphore.get(), prepare_value,
                                        warm_bundle.out()));
  IREE_RETURN_IF_ERROR(
      WaitForSemaphore(prepare_semaphore.get(), prepare_value));
  warm_bundle.reset();

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    ++prepare_value;
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
        bundle;
    IREE_RETURN_IF_ERROR(PrepareDitBundle(&context, plan.get(),
                                          prepare_semaphore.get(),
                                          prepare_value, bundle.out()));
    IREE_RETURN_IF_ERROR(
        WaitForSemaphore(prepare_semaphore.get(), prepare_value));
    iree_optimization_barrier(bundle.get());
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count *
                           context.request.latent_shape.dims[0] *
                           context.request.latent_shape.dims[1]));
  SetDitBenchmarkLabel(benchmark_state, context, branch, statistics);
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4DitStagePrepareConditionedFixture) {
  return RunPrepareBenchmark(benchmark_state,
                             id4::test::Ideogram4DitBranch::kConditioned);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStagePrepareConditionedFixture,
                           MILLISECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStagePrepareUnconditionedFixture) {
  return RunPrepareBenchmark(benchmark_state,
                             id4::test::Ideogram4DitBranch::kUnconditioned);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStagePrepareUnconditionedFixture,
                           MILLISECOND);

static iree_status_t RunIssueBenchmark(iree_benchmark_state_t* benchmark_state,
                                       id4::test::Ideogram4DitBranch branch,
                                       DitBenchmarkIssueMode issue_mode) {
  DitBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedDitBenchmarkContext(branch, &context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateDitPlan(&context, plan.out()));
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan.get());
  IREE_RETURN_IF_ERROR(WritePlanJsonIfRequested(branch, plan.get()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_RETURN_IF_ERROR(PrepareDitBundle(
      &context, plan.get(), prepare_semaphore.get(), 1, bundle.out()));
  IREE_RETURN_IF_ERROR(WaitForSemaphore(prepare_semaphore.get(), 1));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_RETURN_IF_ERROR(id4::test::AllocateBoundaryBindings(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_RETURN_IF_ERROR(
      id4::test::Ideogram4DitQueueInitializedBoundaryTensorsFromFixture(
          context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
          boundary_bindings, context.fixture_tensors,
          id4::test::Ideogram4DitBranchConfigFor(branch),
          update_semaphore.get(), &update_value));
  const uint32_t sentinel_pattern = 0xA5A5A5A5u;
  IREE_RETURN_IF_ERROR(id4::test::QueueFillBoundaryTensors(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &sentinel_pattern, sizeof(sentinel_pattern), update_semaphore.get(),
      &update_value));
  if (update_value != 0) {
    IREE_RETURN_IF_ERROR(
        WaitForSemaphore(update_semaphore.get(), update_value));
  }

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  iree_hal_semaphore_t* wait_semaphore = update_semaphore.get();
  uint64_t wait_value = update_value;
  uint64_t signal_value = 1;
  IREE_RETURN_IF_ERROR(IssueDitBundle(&context, bundle.get(), boundary_bindings,
                                      wait_semaphore, wait_value,
                                      issue_semaphore.get(), signal_value));
  IREE_RETURN_IF_ERROR(WaitForSemaphore(issue_semaphore.get(), signal_value));
  wait_semaphore = issue_semaphore.get();
  wait_value = signal_value;
  context.diagnostics = {};

  uint64_t iteration_count = 0;
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_hal_begin_device_group_profiling_from_flags(
      context.live.device_group.get(), iree_allocator_system(), &profiling);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    ++signal_value;
    status = IssueDitBundle(&context, bundle.get(), boundary_bindings,
                            wait_semaphore, wait_value, issue_semaphore.get(),
                            signal_value);
    bool timing_paused = false;
    if (iree_status_is_ok(status) &&
        issue_mode == DitBenchmarkIssueMode::kSubmitOnly) {
      iree_benchmark_pause_timing(benchmark_state);
      timing_paused = true;
    }
    if (iree_status_is_ok(status)) {
      status = WaitForSemaphore(issue_semaphore.get(), signal_value);
    }
    if (timing_paused) {
      iree_benchmark_resume_timing(benchmark_state);
    }
    if (iree_status_is_ok(status)) {
      wait_semaphore = issue_semaphore.get();
      wait_value = signal_value;
      ++iteration_count;
    }
  }
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  IREE_RETURN_IF_ERROR(status);
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count *
                           context.request.latent_shape.dims[0] *
                           context.request.latent_shape.dims[1]));
  SetDitBenchmarkLabel(benchmark_state, context, branch, statistics);
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4DitStageIssueConditionedSubmitOnly) {
  return RunIssueBenchmark(benchmark_state,
                           id4::test::Ideogram4DitBranch::kConditioned,
                           DitBenchmarkIssueMode::kSubmitOnly);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStageIssueConditionedSubmitOnly,
                           MICROSECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStageIssueConditionedEndToEnd) {
  return RunIssueBenchmark(benchmark_state,
                           id4::test::Ideogram4DitBranch::kConditioned,
                           DitBenchmarkIssueMode::kEndToEnd);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStageIssueConditionedEndToEnd,
                           MILLISECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStageIssueUnconditionedSubmitOnly) {
  return RunIssueBenchmark(benchmark_state,
                           id4::test::Ideogram4DitBranch::kUnconditioned,
                           DitBenchmarkIssueMode::kSubmitOnly);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStageIssueUnconditionedSubmitOnly,
                           MICROSECOND);

IREE_BENCHMARK_FN(BM_Ideogram4DitStageIssueUnconditionedEndToEnd) {
  return RunIssueBenchmark(benchmark_state,
                           id4::test::Ideogram4DitBranch::kUnconditioned,
                           DitBenchmarkIssueMode::kEndToEnd);
}
ID4_DIT_BENCHMARK_REGISTER(BM_Ideogram4DitStageIssueUnconditionedEndToEnd,
                           MILLISECOND);

#undef ID4_DIT_BENCHMARK_REGISTER

}  // namespace
