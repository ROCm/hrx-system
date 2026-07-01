// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cinttypes>
#include <cstring>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/tooling/filesystem.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/benchmark.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(string, id4_fixture_dir, "",
          "Directory containing a Qwen3-VL fixture manifest.");
IREE_FLAG(string, id4_plan_output_dir, "",
          "Optional directory receiving benchmark Qwen3-VL stage plan JSON.");
IREE_FLAG(string, qwen_weight_execution_strategy, "hybrid_compact_rhs",
          "Qwen3-VL weight execution strategy: row_major, compact_rhs, or "
          "hybrid_compact_rhs.");
IREE_FLAG(int64_t, parameter_load_prefetch_region_distance, 0,
          "Number of future stage regions whose deferred parameter load groups "
          "may be submitted before the current region.");

namespace {

struct QwenBenchmarkShape {
  // Dynamic request token count used when planning Qwen3-VL forward.
  uint32_t token_count;
};

static constexpr QwenBenchmarkShape kQwenToken19Shape = {
    // Tiny prompt-class token count below one WMMA attention tile.
    19,
};

static constexpr QwenBenchmarkShape kQwenToken64Shape = {
    // Short prompt-class token count.
    64,
};

static constexpr QwenBenchmarkShape kQwenToken256Shape = {
    // Medium prompt-class token count.
    256,
};

static constexpr QwenBenchmarkShape kQwenToken512Shape = {
    // Long structured prompt-class token count.
    512,
};

static constexpr QwenBenchmarkShape kQwenToken1024Shape = {
    // Extended prompt-class token count near the large linear tile threshold.
    1024,
};

enum class QwenIssueTimingMode {
  // Measures queue submission while waiting for completion outside timing.
  kSubmitOnly,
  // Measures user-visible issue through completion.
  kEndToEnd,
};

enum class QwenParameterLoadMode {
  // Stage preparation submits all planned parameter loads.
  kEager,
  // Stage issue submits parameter load groups at first consumer regions.
  kDeferred,
};

struct QwenBenchmarkContext {
  // Live HAL, executable cache, and kernel-cache context selected by flags.
  id4::test::LiveStageContext live;
  // Embedded Loom source library used during stage preparation.
  id4::test::KernelLibraryRef kernel_library;
  // Parameter provider created from standard --parameters= flags.
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  // Loaded Qwen3-VL stage under benchmark.
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  // Deferred parameter load lookahead used by issue benchmarks.
  iree_host_size_t parameter_load_prefetch_region_distance = 0;
  // Fixture tensors used by fixture-backed issue benchmarks.
  id4::test::FixtureTensorSet fixture_tensors;
  // Dynamic request dimensions used by fixture-backed benchmarks.
  id4_qwen3_vl_request_config_t request = {};
  // Diagnostic event counters collected by lifecycle calls.
  id4::test::StageDiagnostics diagnostics = {};
  // Diagnostics sink passed to stage lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
};

static iree_status_t ParseQwenWeightExecutionStrategy(
    id4_qwen3_vl_weight_execution_strategy_t* out_strategy) {
  iree_status_t status = id4_qwen3_vl_weight_execution_strategy_parse(
      iree_make_cstring_view(FLAG_qwen_weight_execution_strategy),
      out_strategy);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status,
                              IREE_SV("--qwen_weight_execution_strategy"));
}

static iree_status_t ParseParameterLoadPrefetchRegionDistance(
    iree_host_size_t* out_distance) {
  IREE_ASSERT_ARGUMENT(out_distance);
  if (FLAG_parameter_load_prefetch_region_distance < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--parameter_load_prefetch_region_distance must be non-negative");
  }
  if ((uint64_t)FLAG_parameter_load_prefetch_region_distance >
      IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "--parameter_load_prefetch_region_distance exceeds host size range");
  }
  *out_distance =
      (iree_host_size_t)FLAG_parameter_load_prefetch_region_distance;
  return iree_ok_status();
}

static const QwenBenchmarkShape& QwenBenchmarkShapeFromDef(
    const iree_benchmark_def_t* benchmark_def) {
  return *static_cast<const QwenBenchmarkShape*>(benchmark_def->user_data);
}

static iree_status_t CreateQwen3VlStage(const id4::test::LiveStageContext& live,
                                        id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = live.device_group.get();
  services.executable_cache = live.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = live.kernel_cache.get();
  create_options.model = *id4_qwen3_vl_program_ideogram4_model_config();
  return id4_qwen3_vl_stage_create(&create_options, iree_allocator_system(),
                                   out_stage);
}

static iree_status_t CreateLoadedQwenStageContext(
    QwenBenchmarkContext* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  out_context->diagnostics_sink =
      id4::test::DiagnosticsSink(&out_context->diagnostics);
  IREE_RETURN_IF_ERROR(
      id4::test::CreateLiveStageContextFromFlags(&out_context->live));
  IREE_RETURN_IF_ERROR(ParseParameterLoadPrefetchRegionDistance(
      &out_context->parameter_load_prefetch_region_distance));
  IREE_RETURN_IF_ERROR(
      CreateQwen3VlStage(out_context->live, out_context->stage.out()));

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &out_context->diagnostics_sink;
  return id4_pipeline_stage_load(out_context->stage.get(), &load_options);
}

static iree_status_t AttachQwenPreparationInputs(
    QwenBenchmarkContext* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_RETURN_IF_ERROR(
      id4::test::CreateEmbeddedKernelLibrary(context->kernel_library.out()));
  return id4::test::CreateParameterProviderFromFlags(
      iree_string_view_empty(), context->parameter_provider.out());
}

static iree_status_t CreateLoadedQwenBenchmarkContext(
    QwenBenchmarkContext* out_context) {
  IREE_RETURN_IF_ERROR(CreateLoadedQwenStageContext(out_context));
  return AttachQwenPreparationInputs(out_context);
}

static iree_status_t LoadFixtureAndConfigureRequest(
    QwenBenchmarkContext* context) {
  const iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  if (iree_string_view_is_empty(fixture_directory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--id4_fixture_dir is required");
  }
  IREE_RETURN_IF_ERROR(id4::test::LoadFixtureTensors(
      fixture_directory, &context->fixture_tensors));
  return id4::test::InferRank1TensorLengthFromFixture(
      context->fixture_tensors, IREE_SV("token_ids"),
      ID4_PIPELINE_TENSOR_DTYPE_I32, &context->request.token_count);
}

static iree_status_t CreateLoadedQwenFixtureStageContext(
    QwenBenchmarkContext* out_context) {
  IREE_RETURN_IF_ERROR(CreateLoadedQwenStageContext(out_context));
  return LoadFixtureAndConfigureRequest(out_context);
}

static iree_status_t CreateLoadedQwenFixtureBenchmarkContext(
    QwenBenchmarkContext* out_context) {
  IREE_RETURN_IF_ERROR(CreateLoadedQwenFixtureStageContext(out_context));
  return AttachQwenPreparationInputs(out_context);
}

static iree_status_t CreateQwenPlan(QwenBenchmarkContext* context,
                                    id4_qwen3_vl_request_config_t request,
                                    id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = nullptr;

  id4_qwen3_vl_stage_plan_options_t qwen_options;
  std::memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request = request;
  IREE_RETURN_IF_ERROR(ParseQwenWeightExecutionStrategy(
      &qwen_options.weight_execution_strategy));

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &qwen_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_plan(context->stage.get(), &plan_options, out_plan);
}

static iree_status_t WritePlanJsonIfRequested(const id4_pipeline_plan_t* plan,
                                              uint32_t token_count) {
  iree_string_view_t output_dir =
      iree_make_cstring_view(FLAG_id4_plan_output_dir);
  if (iree_string_view_is_empty(output_dir)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      id4_tooling_ensure_directory(output_dir, iree_allocator_system()));

  char file_name_buffer[64];
  int file_name_length =
      std::snprintf(file_name_buffer, sizeof(file_name_buffer),
                    "qwen3_vl_token_%u.json", token_count);
  if (file_name_length < 0 || static_cast<iree_host_size_t>(file_name_length) >=
                                  sizeof(file_name_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL plan JSON file name overflow");
  }
  iree_string_view_t file_name = iree_make_string_view(
      file_name_buffer, static_cast<iree_host_size_t>(file_name_length));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = id4_pipeline_plan_format_json(plan, &builder);
  iree_string_view_t path = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = id4_tooling_format_child_path(output_dir, file_name,
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

static iree_status_t WriteInitialPlanJsonIfRequested(
    iree_benchmark_state_t* benchmark_state, QwenBenchmarkContext* context,
    id4_qwen3_vl_request_config_t request) {
  if (iree_string_view_is_empty(
          iree_make_cstring_view(FLAG_id4_plan_output_dir))) {
    return iree_ok_status();
  }

  iree_benchmark_pause_timing(benchmark_state);
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  iree_status_t status = CreateQwenPlan(context, request, plan.out());
  if (iree_status_is_ok(status)) {
    status = WritePlanJsonIfRequested(plan.get(), request.token_count);
  }
  iree_benchmark_resume_timing(benchmark_state);
  return status;
}

static uint64_t CeilMiB(iree_device_size_t byte_length) {
  static constexpr iree_device_size_t kMiB = 1024ull * 1024ull;
  return (uint64_t)((byte_length + kMiB - 1) / kMiB);
}

static double AverageMilliseconds(iree_duration_t duration_ns,
                                  uint64_t iteration_count) {
  if (iteration_count == 0) return 0.0;
  return (double)duration_ns / (double)iteration_count / 1000000.0;
}

static double AverageRegionDistance(iree_host_size_t distance_sum,
                                    iree_host_size_t count) {
  if (count == 0) return 0.0;
  return (double)distance_sum / (double)count;
}

static void SetQwenBenchmarkLabel(
    iree_benchmark_state_t* benchmark_state, uint32_t token_count,
    iree_host_size_t parameter_load_prefetch_region_distance,
    const id4_pipeline_plan_statistics_t& statistics,
    const id4::test::StageDiagnostics& diagnostics, uint64_t iteration_count) {
  id4_qwen3_vl_weight_execution_strategy_t weight_execution_strategy =
      ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_INVALID;
  IREE_CHECK_OK(ParseQwenWeightExecutionStrategy(&weight_execution_strategy));
  iree_string_view_t weight_execution_strategy_name =
      id4_qwen3_vl_weight_execution_strategy_name(weight_execution_strategy);
  iree_string_builder_t label_builder;
  iree_string_builder_initialize(iree_allocator_system(), &label_builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &label_builder,
      "tokens=%" PRIu32 " weights=%.*s prefetch_regions=%" PRIhsz
      " param_total=%" PRIu64 "MiB param_largest=%" PRIu64
      "MiB param_source=%" PRIu64 "MiB param_source_direct=%" PRIu64
      "MiB param_source_encoded=%" PRIu64 "MiB param_load_steps[gather=%" PRIhsz
      ",encode=%" PRIhsz "] param_load_groups[total=%" PRIhsz ",gather=%" PRIhsz
      ",encode=%" PRIhsz "] local_hw=%" PRIu64 "MiB boundary=%" PRIu64
      "MiB kernels=%" PRIhsz " dispatches=%" PRIhsz
      " load_group_submit_ms[all=%.3f,gather=%.3f,encode=%.3f,max=%.3f]"
      " load_group_submit_count[total=%" PRIhsz ",gather=%" PRIhsz
      ",encode=%" PRIhsz
      "]"
      " prefetch_groups[count=%" PRIhsz ",avg_regions=%.2f,max_regions=%" PRIhsz
      "]"
      " direct_gather_groups[count=%" PRIhsz ",requests=%" PRIhsz
      ",source=%" PRIu64 "MiB,target=%" PRIu64 "MiB,max=%" PRIu64
      "MiB]"
      " encode_windows[count=%" PRIhsz ",staging=%" PRIu64 "MiB,max=%" PRIu64
      "MiB,source=%" PRIu64 "MiB,target=%" PRIu64 "MiB,chunks=%" PRIhsz
      ",batches=%" PRIhsz ",dispatches=%" PRIhsz
      "]"
      " prepare_encode_window[count=%" PRIhsz ",staging=%" PRIu64
      "MiB,max=%" PRIu64 "MiB,source=%" PRIu64 "MiB,target=%" PRIu64
      "MiB,chunks=%" PRIhsz ",batches=%" PRIhsz ",dispatches=%" PRIhsz
      "]"
      " issue_encode_window[count=%" PRIhsz ",staging=%" PRIu64
      "MiB,max=%" PRIu64 "MiB,source=%" PRIu64 "MiB,target=%" PRIu64
      "MiB,chunks=%" PRIhsz ",batches=%" PRIhsz ",dispatches=%" PRIhsz "]",
      token_count, static_cast<int>(weight_execution_strategy_name.size),
      weight_execution_strategy_name.data,
      parameter_load_prefetch_region_distance,
      CeilMiB(statistics.parameter_slab_byte_length),
      CeilMiB(statistics.largest_parameter_slab_byte_length),
      CeilMiB(statistics.parameter_source_byte_length),
      CeilMiB(statistics.parameter_direct_source_byte_length),
      CeilMiB(statistics.parameter_encoded_source_byte_length),
      statistics.parameter_gather_load_step_count,
      statistics.parameter_encode_load_step_count,
      statistics.parameter_load_group_count,
      statistics.parameter_gather_load_group_count,
      statistics.parameter_encode_load_group_count,
      CeilMiB(statistics.memory_slab_high_water_mark),
      CeilMiB(statistics.boundary_tensor_byte_length), statistics.kernel_count,
      statistics.dispatch_count,
      AverageMilliseconds(diagnostics.parameter_load_group_submit_duration_ns,
                          iteration_count),
      AverageMilliseconds(
          diagnostics.parameter_load_group_submit_gather_duration_ns,
          iteration_count),
      AverageMilliseconds(
          diagnostics.parameter_load_group_submit_encode_duration_ns,
          iteration_count),
      (double)diagnostics.parameter_load_group_submit_max_duration_ns /
          1000000.0,
      diagnostics.parameter_load_group_submit_count,
      diagnostics.parameter_load_group_submit_gather_count,
      diagnostics.parameter_load_group_submit_encode_count,
      diagnostics.parameter_load_group_prefetch_submit_count,
      AverageRegionDistance(
          diagnostics.parameter_load_group_prefetch_region_distance_sum,
          diagnostics.parameter_load_group_prefetch_submit_count),
      diagnostics.parameter_load_group_prefetch_region_distance_max,
      diagnostics.parameter_direct_gather_group_count,
      diagnostics.parameter_direct_gather_request_count,
      CeilMiB(diagnostics.parameter_direct_gather_source_byte_length),
      CeilMiB(diagnostics.parameter_direct_gather_target_byte_length),
      CeilMiB(diagnostics.parameter_direct_gather_max_source_byte_length),
      diagnostics.parameter_encode_window_count,
      CeilMiB(diagnostics.parameter_encode_window_staging_total_byte_length),
      CeilMiB(diagnostics.parameter_encode_window_staging_max_byte_length),
      CeilMiB(diagnostics.parameter_encode_window_source_byte_length),
      CeilMiB(diagnostics.parameter_encode_window_target_byte_length),
      diagnostics.parameter_encode_window_staging_chunk_count,
      diagnostics.parameter_encode_window_source_gather_batch_count,
      diagnostics.parameter_encode_window_encoder_dispatch_count,
      diagnostics.parameter_prepare_encode_window_count,
      CeilMiB(diagnostics
                  .parameter_prepare_encode_window_staging_total_byte_length),
      CeilMiB(
          diagnostics.parameter_prepare_encode_window_staging_max_byte_length),
      CeilMiB(diagnostics.parameter_prepare_encode_window_source_byte_length),
      CeilMiB(diagnostics.parameter_prepare_encode_window_target_byte_length),
      diagnostics.parameter_prepare_encode_window_staging_chunk_count,
      diagnostics.parameter_prepare_encode_window_source_gather_batch_count,
      diagnostics.parameter_prepare_encode_window_encoder_dispatch_count,
      diagnostics.parameter_issue_encode_window_count,
      CeilMiB(
          diagnostics.parameter_issue_encode_window_staging_total_byte_length),
      CeilMiB(
          diagnostics.parameter_issue_encode_window_staging_max_byte_length),
      CeilMiB(diagnostics.parameter_issue_encode_window_source_byte_length),
      CeilMiB(diagnostics.parameter_issue_encode_window_target_byte_length),
      diagnostics.parameter_issue_encode_window_staging_chunk_count,
      diagnostics.parameter_issue_encode_window_source_gather_batch_count,
      diagnostics.parameter_issue_encode_window_encoder_dispatch_count));
  IREE_CHECK_OK(id4::test::AppendParameterLoadKindStatisticsLabel(
      &label_builder, statistics.parameter_load_kind_statistics));
  iree_benchmark_set_label(benchmark_state,
                           iree_string_builder_buffer(&label_builder));
  iree_string_builder_deinitialize(&label_builder);
}

static iree_status_t QueueFillSyntheticQwenInputs(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* fill_semaphore, uint64_t* inout_fill_value) {
  IREE_ASSERT_ARGUMENT(inout_fill_value);
  iree_hal_buffer_binding_t token_ids_binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(
      plan, boundary_bindings, IREE_SV("token_ids"), &token_ids_binding));
  iree_hal_buffer_binding_t attention_mask_binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(plan, boundary_bindings,
                                                      IREE_SV("attention_mask"),
                                                      &attention_mask_binding));
  iree_hal_buffer_binding_t token_weights_binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(plan, boundary_bindings,
                                                      IREE_SV("token_weights"),
                                                      &token_weights_binding));

  const int32_t token_id = 0;
  IREE_RETURN_IF_ERROR(id4::test::QueueFillBinding(
      device, queue_affinity, &token_ids_binding, &token_id, sizeof(token_id),
      fill_semaphore, inout_fill_value));
  const float attention_mask = 0.0f;
  IREE_RETURN_IF_ERROR(id4::test::QueueFillBinding(
      device, queue_affinity, &attention_mask_binding, &attention_mask,
      sizeof(attention_mask), fill_semaphore, inout_fill_value));
  const float token_weight = 1.0f;
  return id4::test::QueueFillBinding(
      device, queue_affinity, &token_weights_binding, &token_weight,
      sizeof(token_weight), fill_semaphore, inout_fill_value);
}

static iree_status_t PrepareQwenBundle(QwenBenchmarkContext* context,
                                       const id4_pipeline_plan_t* plan,
                                       QwenParameterLoadMode load_mode,
                                       iree_hal_semaphore_t* prepare_semaphore,
                                       uint64_t signal_value,
                                       id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = nullptr;

  const bool defers_parameter_loads =
      load_mode == QwenParameterLoadMode::kDeferred;
  id4::test::SemaphoreListStorage signal;
  iree_hal_semaphore_list_t signal_list = iree_hal_semaphore_list_empty();
  if (!defers_parameter_loads) {
    signal.semaphore = prepare_semaphore;
    signal.payload_value = signal_value;
    signal_list = signal.list();
  }

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.flags =
      defers_parameter_loads
          ? ID4_PIPELINE_STAGE_PREPARE_FLAG_DEFER_PARAMETER_LOADS_TO_ISSUE
          : 0;
  prepare_options.parameter_provider = context->parameter_provider.get();
  prepare_options.kernel_library = context->kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.command_buffer_mode = context->live.command_buffer_mode;
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_prepare(context->stage.get(), plan,
                                    &prepare_options, out_bundle);
}

static iree_status_t IssueQwenBundle(
    QwenBenchmarkContext* context, id4_pipeline_bundle_t* bundle,
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
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  id4_pipeline_parameter_slab_set_t* parameter_slabs =
      id4_pipeline_bundle_parameter_slabs(bundle);
  if (parameter_slabs &&
      id4_pipeline_parameter_slab_set_has_deferred_load_context(
          parameter_slabs)) {
    issue_options.parameter_load_prefetch_region_distance =
        context->parameter_load_prefetch_region_distance;
  }
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

static const iree_benchmark_def_t* RegisterQwenBenchmark(
    iree_string_view_t name, iree_benchmark_fn_t run,
    iree_benchmark_unit_t time_unit, const QwenBenchmarkShape* shape) {
  iree_benchmark_def_t* benchmark = iree_make_function_benchmark(run);
  benchmark->flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME;
  benchmark->time_unit = time_unit;
  benchmark->user_data = shape;
  return iree_benchmark_register(name, benchmark);
}

#define ID4_QWEN_BENCHMARK_REGISTER(name, shape, suffix, time_unit) \
  static const iree_benchmark_def_t* name##_##shape##_registration  \
      IREE_ATTRIBUTE_UNUSED =                                       \
          RegisterQwenBenchmark(IREE_SV(#name "/" suffix), name,    \
                                IREE_BENCHMARK_UNIT_##time_unit, &shape)

IREE_BENCHMARK_FN(BM_Qwen3VlStagePlan) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenStageContext(&context));
  id4_qwen3_vl_request_config_t request = {};
  request.token_count = shape.token_count;

  bool wrote_initial_plan_json = false;
  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    if (!wrote_initial_plan_json) {
      IREE_RETURN_IF_ERROR(
          WriteInitialPlanJsonIfRequested(benchmark_state, &context, request));
      wrote_initial_plan_json = true;
    }
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, request, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * shape.token_count));
  return iree_ok_status();
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePlan, kQwenToken19Shape, "token19",
                            MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePlan, kQwenToken64Shape, "token64",
                            MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePlan, kQwenToken256Shape, "token256",
                            MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePlan, kQwenToken512Shape, "token512",
                            MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePlan, kQwenToken1024Shape,
                            "token1024", MICROSECOND);

IREE_BENCHMARK_FN(BM_Qwen3VlStagePlanFixture) {
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenFixtureStageContext(&context));

  bool wrote_initial_plan_json = false;
  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    if (!wrote_initial_plan_json) {
      IREE_RETURN_IF_ERROR(WriteInitialPlanJsonIfRequested(
          benchmark_state, &context, context.request));
      wrote_initial_plan_json = true;
    }
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, context.request, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * context.request.token_count));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Qwen3VlStagePrepareCachedKernels) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenBenchmarkContext(&context));
  id4_qwen3_vl_request_config_t request = {};
  request.token_count = shape.token_count;

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, request, plan.out()));
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan.get());

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  uint64_t prepare_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      warm_bundle;
  IREE_RETURN_IF_ERROR(PrepareQwenBundle(
      &context, plan.get(), QwenParameterLoadMode::kEager,
      prepare_semaphore.get(), prepare_value, warm_bundle.out()));
  IREE_RETURN_IF_ERROR(
      WaitForSemaphore(prepare_semaphore.get(), prepare_value));
  warm_bundle.reset();
  context.diagnostics = {};

  uint64_t iteration_count = 0;
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_hal_begin_device_group_profiling_from_flags(
      context.live.device_group.get(), iree_allocator_system(), &profiling);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    ++prepare_value;
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
        bundle;
    status =
        PrepareQwenBundle(&context, plan.get(), QwenParameterLoadMode::kEager,
                          prepare_semaphore.get(), prepare_value, bundle.out());
    if (iree_status_is_ok(status)) {
      status = WaitForSemaphore(prepare_semaphore.get(), prepare_value);
    }
    if (iree_status_is_ok(status)) {
      iree_optimization_barrier(bundle.get());
      ++iteration_count;
    }
  }
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  IREE_RETURN_IF_ERROR(status);
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * shape.token_count));
  SetQwenBenchmarkLabel(benchmark_state, shape.token_count,
                        context.parameter_load_prefetch_region_distance,
                        statistics, context.diagnostics, iteration_count);
  return iree_ok_status();
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareCachedKernels,
                            kQwenToken19Shape, "token19", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareCachedKernels,
                            kQwenToken64Shape, "token64", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareCachedKernels,
                            kQwenToken256Shape, "token256", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareCachedKernels,
                            kQwenToken512Shape, "token512", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareCachedKernels,
                            kQwenToken1024Shape, "token1024", MILLISECOND);

IREE_BENCHMARK_FN(BM_Qwen3VlStagePrepareFixtureCachedKernels) {
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenFixtureBenchmarkContext(&context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, context.request, plan.out()));
  IREE_RETURN_IF_ERROR(
      WritePlanJsonIfRequested(plan.get(), context.request.token_count));
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan.get());

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  uint64_t prepare_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      warm_bundle;
  IREE_RETURN_IF_ERROR(PrepareQwenBundle(
      &context, plan.get(), QwenParameterLoadMode::kEager,
      prepare_semaphore.get(), prepare_value, warm_bundle.out()));
  IREE_RETURN_IF_ERROR(
      WaitForSemaphore(prepare_semaphore.get(), prepare_value));
  warm_bundle.reset();
  context.diagnostics = {};

  uint64_t iteration_count = 0;
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_hal_begin_device_group_profiling_from_flags(
      context.live.device_group.get(), iree_allocator_system(), &profiling);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    ++prepare_value;
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
        bundle;
    status =
        PrepareQwenBundle(&context, plan.get(), QwenParameterLoadMode::kEager,
                          prepare_semaphore.get(), prepare_value, bundle.out());
    if (iree_status_is_ok(status)) {
      status = WaitForSemaphore(prepare_semaphore.get(), prepare_value);
    }
    if (iree_status_is_ok(status)) {
      iree_optimization_barrier(bundle.get());
      ++iteration_count;
    }
  }
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  IREE_RETURN_IF_ERROR(status);
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * context.request.token_count));
  SetQwenBenchmarkLabel(benchmark_state, context.request.token_count,
                        context.parameter_load_prefetch_region_distance,
                        statistics, context.diagnostics, iteration_count);
  return iree_ok_status();
}

static iree_status_t RunIssueBenchmarkWithPreparedContext(
    iree_benchmark_state_t* benchmark_state, QwenBenchmarkContext* context,
    QwenIssueTimingMode timing_mode,
    iree_status_t (*queue_inputs)(QwenBenchmarkContext*,
                                  const id4_pipeline_plan_t*,
                                  const id4::test::BufferBindingSet&,
                                  iree_hal_semaphore_t*, uint64_t*)) {
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateQwenPlan(context, context->request, plan.out()));
  IREE_RETURN_IF_ERROR(
      WritePlanJsonIfRequested(plan.get(), context->request.token_count));
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan.get());

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_RETURN_IF_ERROR(
      PrepareQwenBundle(context, plan.get(), QwenParameterLoadMode::kEager,
                        prepare_semaphore.get(), 1, bundle.out()));
  IREE_RETURN_IF_ERROR(WaitForSemaphore(prepare_semaphore.get(), 1));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_RETURN_IF_ERROR(id4::test::AllocateBoundaryBindings(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_RETURN_IF_ERROR(queue_inputs(context, plan.get(), boundary_bindings,
                                    update_semaphore.get(), &update_value));
  const uint32_t sentinel_pattern = 0xA5A5A5A5u;
  IREE_RETURN_IF_ERROR(id4::test::QueueFillBoundaryTensors(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &sentinel_pattern, sizeof(sentinel_pattern), update_semaphore.get(),
      &update_value));
  if (update_value != 0) {
    IREE_RETURN_IF_ERROR(
        WaitForSemaphore(update_semaphore.get(), update_value));
  }
  context->diagnostics = {};

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  iree_hal_semaphore_t* wait_semaphore = update_semaphore.get();
  uint64_t wait_value = update_value;
  uint64_t signal_value = 0;
  uint64_t iteration_count = 0;
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_hal_begin_device_group_profiling_from_flags(
      context->live.device_group.get(), iree_allocator_system(), &profiling);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    ++signal_value;
    status = IssueQwenBundle(context, bundle.get(), boundary_bindings,
                             wait_semaphore, wait_value, issue_semaphore.get(),
                             signal_value);
    bool timing_paused = false;
    if (iree_status_is_ok(status) &&
        timing_mode == QwenIssueTimingMode::kSubmitOnly) {
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
      static_cast<int64_t>(iteration_count * context->request.token_count));
  SetQwenBenchmarkLabel(benchmark_state, context->request.token_count,
                        context->parameter_load_prefetch_region_distance,
                        statistics, context->diagnostics, iteration_count);
  return iree_ok_status();
}

static iree_status_t RunPrepareIssueBenchmarkWithContext(
    iree_benchmark_state_t* benchmark_state, QwenBenchmarkContext* context,
    QwenParameterLoadMode load_mode,
    iree_status_t (*queue_inputs)(QwenBenchmarkContext*,
                                  const id4_pipeline_plan_t*,
                                  const id4::test::BufferBindingSet&,
                                  iree_hal_semaphore_t*, uint64_t*)) {
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateQwenPlan(context, context->request, plan.out()));
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan.get());

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      warm_bundle;
  uint64_t prepare_value = 1;
  if (load_mode == QwenParameterLoadMode::kEager) {
    IREE_RETURN_IF_ERROR(PrepareQwenBundle(context, plan.get(), load_mode,
                                           prepare_semaphore.get(),
                                           prepare_value, warm_bundle.out()));
    IREE_RETURN_IF_ERROR(
        WaitForSemaphore(prepare_semaphore.get(), prepare_value));
    warm_bundle.reset();
  }

  id4::test::BufferBindingSet boundary_bindings;
  IREE_RETURN_IF_ERROR(id4::test::AllocateBoundaryBindings(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_RETURN_IF_ERROR(queue_inputs(context, plan.get(), boundary_bindings,
                                    update_semaphore.get(), &update_value));
  const uint32_t sentinel_pattern = 0xA5A5A5A5u;
  IREE_RETURN_IF_ERROR(id4::test::QueueFillBoundaryTensors(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
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
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  uint64_t signal_value = 0;
  if (load_mode == QwenParameterLoadMode::kDeferred) {
    IREE_RETURN_IF_ERROR(PrepareQwenBundle(context, plan.get(), load_mode,
                                           prepare_semaphore.get(),
                                           prepare_value, warm_bundle.out()));
    ++signal_value;
    IREE_RETURN_IF_ERROR(IssueQwenBundle(
        context, warm_bundle.get(), boundary_bindings, update_semaphore.get(),
        update_value, issue_semaphore.get(), signal_value));
    IREE_RETURN_IF_ERROR(WaitForSemaphore(issue_semaphore.get(), signal_value));
    warm_bundle.reset();
  }
  context->diagnostics = {};

  uint64_t iteration_count = 0;
  bool wrote_initial_plan_json = false;
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_hal_begin_device_group_profiling_from_flags(
      context->live.device_group.get(), iree_allocator_system(), &profiling);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    if (!wrote_initial_plan_json) {
      iree_benchmark_pause_timing(benchmark_state);
      status =
          WritePlanJsonIfRequested(plan.get(), context->request.token_count);
      iree_benchmark_resume_timing(benchmark_state);
      wrote_initial_plan_json = true;
    }
    if (!iree_status_is_ok(status)) break;

    ++prepare_value;
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
        bundle;
    status =
        PrepareQwenBundle(context, plan.get(), load_mode,
                          prepare_semaphore.get(), prepare_value, bundle.out());
    if (iree_status_is_ok(status)) {
      ++signal_value;
      status = IssueQwenBundle(context, bundle.get(), boundary_bindings,
                               update_semaphore.get(), update_value,
                               issue_semaphore.get(), signal_value);
    }
    if (iree_status_is_ok(status)) {
      status = WaitForSemaphore(issue_semaphore.get(), signal_value);
    }
    if (iree_status_is_ok(status) &&
        load_mode == QwenParameterLoadMode::kEager) {
      status = WaitForSemaphore(prepare_semaphore.get(), prepare_value);
    }
    if (iree_status_is_ok(status)) {
      iree_optimization_barrier(bundle.get());
      ++iteration_count;
    }
  }
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  IREE_RETURN_IF_ERROR(status);
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * context->request.token_count));
  SetQwenBenchmarkLabel(benchmark_state, context->request.token_count,
                        context->parameter_load_prefetch_region_distance,
                        statistics, context->diagnostics, iteration_count);
  return iree_ok_status();
}

static iree_status_t QueueFixtureQwenInputs(
    QwenBenchmarkContext* context, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* update_semaphore, uint64_t* inout_update_value) {
  return id4::test::QueueUpdateInitializedBoundaryTensorsFromFixture(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
      boundary_bindings, context->fixture_tensors, update_semaphore,
      inout_update_value);
}

static iree_status_t QueueSyntheticQwenInputs(
    QwenBenchmarkContext* context, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* update_semaphore, uint64_t* inout_update_value) {
  return QueueFillSyntheticQwenInputs(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
      boundary_bindings, update_semaphore, inout_update_value);
}

static iree_status_t RunFixtureIssueBenchmark(
    iree_benchmark_state_t* benchmark_state, QwenIssueTimingMode timing_mode) {
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenFixtureBenchmarkContext(&context));
  return RunIssueBenchmarkWithPreparedContext(
      benchmark_state, &context, timing_mode, QueueFixtureQwenInputs);
}

static iree_status_t RunSyntheticIssueBenchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state, QwenIssueTimingMode timing_mode) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenBenchmarkContext(&context));
  context.request.token_count = shape.token_count;
  return RunIssueBenchmarkWithPreparedContext(
      benchmark_state, &context, timing_mode, QueueSyntheticQwenInputs);
}

static iree_status_t RunFixturePrepareIssueBenchmark(
    iree_benchmark_state_t* benchmark_state) {
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenFixtureBenchmarkContext(&context));
  return RunPrepareIssueBenchmarkWithContext(benchmark_state, &context,
                                             QwenParameterLoadMode::kEager,
                                             QueueFixtureQwenInputs);
}

static iree_status_t RunSyntheticPrepareIssueBenchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenBenchmarkContext(&context));
  context.request.token_count = shape.token_count;
  return RunPrepareIssueBenchmarkWithContext(benchmark_state, &context,
                                             QwenParameterLoadMode::kEager,
                                             QueueSyntheticQwenInputs);
}

static iree_status_t RunSyntheticDeferredPrepareIssueBenchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenBenchmarkContext(&context));
  context.request.token_count = shape.token_count;
  return RunPrepareIssueBenchmarkWithContext(benchmark_state, &context,
                                             QwenParameterLoadMode::kDeferred,
                                             QueueSyntheticQwenInputs);
}

IREE_BENCHMARK_FN(BM_Qwen3VlStageIssueFixtureSubmitOnly) {
  return RunFixtureIssueBenchmark(benchmark_state,
                                  QwenIssueTimingMode::kSubmitOnly);
}

IREE_BENCHMARK_FN(BM_Qwen3VlStageIssueFixtureEndToEnd) {
  return RunFixtureIssueBenchmark(benchmark_state,
                                  QwenIssueTimingMode::kEndToEnd);
}

IREE_BENCHMARK_FN(BM_Qwen3VlStageIssueSyntheticSubmitOnly) {
  return RunSyntheticIssueBenchmark(benchmark_def, benchmark_state,
                                    QwenIssueTimingMode::kSubmitOnly);
}

IREE_BENCHMARK_FN(BM_Qwen3VlStageIssueSyntheticEndToEnd) {
  return RunSyntheticIssueBenchmark(benchmark_def, benchmark_state,
                                    QwenIssueTimingMode::kEndToEnd);
}

IREE_BENCHMARK_FN(BM_Qwen3VlStagePrepareIssueFixtureEndToEnd) {
  return RunFixturePrepareIssueBenchmark(benchmark_state);
}

IREE_BENCHMARK_FN(BM_Qwen3VlStagePrepareIssueSyntheticEndToEnd) {
  return RunSyntheticPrepareIssueBenchmark(benchmark_def, benchmark_state);
}

IREE_BENCHMARK_FN(BM_Qwen3VlStagePrepareIssueDeferredSyntheticEndToEnd) {
  return RunSyntheticDeferredPrepareIssueBenchmark(benchmark_def,
                                                   benchmark_state);
}

#define ID4_QWEN_FIXTURE_BENCHMARK_REGISTER(name, time_unit)     \
  static const iree_benchmark_def_t* name##_registration         \
      IREE_ATTRIBUTE_UNUSED =                                    \
          RegisterQwenBenchmark(IREE_SV(#name "/fixture"), name, \
                                IREE_BENCHMARK_UNIT_##time_unit, nullptr)

ID4_QWEN_FIXTURE_BENCHMARK_REGISTER(BM_Qwen3VlStagePlanFixture, MICROSECOND);
ID4_QWEN_FIXTURE_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareFixtureCachedKernels,
                                    MILLISECOND);
ID4_QWEN_FIXTURE_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueFixtureSubmitOnly,
                                    MICROSECOND);
ID4_QWEN_FIXTURE_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueFixtureEndToEnd,
                                    MILLISECOND);
ID4_QWEN_FIXTURE_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareIssueFixtureEndToEnd,
                                    MILLISECOND);

ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticSubmitOnly,
                            kQwenToken19Shape, "token19", MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticSubmitOnly,
                            kQwenToken64Shape, "token64", MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticSubmitOnly,
                            kQwenToken256Shape, "token256", MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticSubmitOnly,
                            kQwenToken512Shape, "token512", MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticSubmitOnly,
                            kQwenToken1024Shape, "token1024", MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticEndToEnd,
                            kQwenToken19Shape, "token19", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticEndToEnd,
                            kQwenToken64Shape, "token64", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticEndToEnd,
                            kQwenToken256Shape, "token256", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticEndToEnd,
                            kQwenToken512Shape, "token512", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSyntheticEndToEnd,
                            kQwenToken1024Shape, "token1024", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareIssueSyntheticEndToEnd,
                            kQwenToken19Shape, "token19", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareIssueSyntheticEndToEnd,
                            kQwenToken64Shape, "token64", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareIssueSyntheticEndToEnd,
                            kQwenToken256Shape, "token256", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareIssueSyntheticEndToEnd,
                            kQwenToken512Shape, "token512", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareIssueSyntheticEndToEnd,
                            kQwenToken1024Shape, "token1024", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(
    BM_Qwen3VlStagePrepareIssueDeferredSyntheticEndToEnd, kQwenToken19Shape,
    "token19", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(
    BM_Qwen3VlStagePrepareIssueDeferredSyntheticEndToEnd, kQwenToken64Shape,
    "token64", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(
    BM_Qwen3VlStagePrepareIssueDeferredSyntheticEndToEnd, kQwenToken256Shape,
    "token256", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(
    BM_Qwen3VlStagePrepareIssueDeferredSyntheticEndToEnd, kQwenToken512Shape,
    "token512", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(
    BM_Qwen3VlStagePrepareIssueDeferredSyntheticEndToEnd, kQwenToken1024Shape,
    "token1024", MILLISECOND);

#undef ID4_QWEN_BENCHMARK_REGISTER
#undef ID4_QWEN_FIXTURE_BENCHMARK_REGISTER

}  // namespace
