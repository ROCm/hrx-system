// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/tooling/capture.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

IREE_FLAG(string, id4_capture_dir, "",
          "Directory that receives exported ID4 boundary tensor captures.");
IREE_FLAG_LIST(string, id4_diagnostic_tap,
               "Additional diagnostic tap names to capture.");
IREE_FLAG(string, id4_fixture_dir, "",
          "Directory containing an ID4 reference fixture manifest and tensor "
          "payloads used to initialize stage inputs.");
IREE_FLAG(string, qwen_parameter_format, "bf16",
          "Qwen3-VL parameter format: bf16 or fp8_e4m3_block_scaled.");
IREE_FLAG(string, qwen_weight_execution_strategy, "hybrid_compact_rhs",
          "Qwen3-VL weight execution strategy: row_major, compact_rhs, or "
          "hybrid_compact_rhs, or streaming_compact_rhs.");
IREE_FLAG(string, qwen_attention_implementation, "auto",
          "Qwen3-VL attention implementation: auto, materialized, or wmma.");

namespace {

constexpr uint8_t kExportedBoundarySentinel = 0xA5;

static iree_status_t ParseQwenParameterFormat(
    id4_qwen3_vl_parameter_format_t* out_format) {
  iree_status_t status = id4_qwen3_vl_parameter_format_parse(
      iree_make_cstring_view(FLAG_qwen_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--qwen_parameter_format"));
}

static iree_status_t ParseQwenWeightExecutionStrategy(
    id4_qwen3_vl_weight_execution_strategy_t* out_strategy) {
  iree_status_t status = id4_qwen3_vl_weight_execution_strategy_parse(
      iree_make_cstring_view(FLAG_qwen_weight_execution_strategy),
      out_strategy);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status,
                              IREE_SV("--qwen_weight_execution_strategy"));
}

static iree_status_t ParseQwenAttentionImplementation(
    id4_qwen3_vl_attention_implementation_t* out_implementation) {
  iree_status_t status = id4_qwen3_vl_attention_implementation_parse(
      iree_make_cstring_view(FLAG_qwen_attention_implementation),
      out_implementation);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status,
                              IREE_SV("--qwen_attention_implementation"));
}

static iree_status_t CreateQwen3VlStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  IREE_RETURN_IF_ERROR(
      ParseQwenParameterFormat(&create_options.parameter_format));
  create_options.model = *id4_qwen3_vl_program_ideogram4_model_config();
  return id4_qwen3_vl_stage_create(&create_options, iree_allocator_system(),
                                   out_stage);
}

static iree_status_t FindDiagnosticTapPlan(
    const id4_pipeline_plan_t* plan, iree_string_view_t name,
    const id4_pipeline_diagnostic_tap_plan_t** out_diagnostic_tap) {
  *out_diagnostic_tap = nullptr;
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  for (iree_host_size_t i = 0; i < diagnostic_tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (diagnostic_tap && iree_string_view_equal(diagnostic_tap->name, name)) {
      *out_diagnostic_tap = diagnostic_tap;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "diagnostic tap `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

TEST(Qwen3VlStageIntegration, PrepareAndIssueForwardWithParameters) {
  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      iree_string_view_empty(), parameter_provider.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_ASSERT_OK(CreateQwen3VlStage(context, stage.out()));
  const id4_qwen3_vl_model_config_t* model_config =
      id4_qwen3_vl_program_ideogram4_model_config();

  iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  ASSERT_FALSE(iree_string_view_is_empty(fixture_directory))
      << "--id4_fixture_dir is required for Qwen integration correctness";
  id4::test::FixtureTensorSet fixture_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(fixture_directory, &fixture_tensors));
  uint32_t token_count = 0;
  IREE_ASSERT_OK(id4::test::InferRank1TensorLengthFromFixture(
      fixture_tensors, IREE_SV("token_ids"), ID4_PIPELINE_TENSOR_DTYPE_I32,
      &token_count));
  const id4::test::FixtureTensor* token_ids_fixture =
      fixture_tensors.FindTensor(IREE_SV("input"), IREE_SV("token_ids"));
  ASSERT_NE(token_ids_fixture, nullptr);
  ASSERT_EQ(token_ids_fixture->payload.size(), token_count * sizeof(int32_t));
  std::vector<int32_t> token_ids(token_count);
  std::memcpy(token_ids.data(), token_ids_fixture->payload.data(),
              token_ids_fixture->payload.size());
  const id4::test::FixtureTensor* expected_condition =
      fixture_tensors.FindTensor(IREE_SV("expected"), IREE_SV("condition"));
  ASSERT_NE(expected_condition, nullptr)
      << "fixture must provide the expected Qwen condition tensor";
  const id4::test::FixtureTensor* expected_selected_hidden_states =
      fixture_tensors.FindTensor(IREE_SV("metadata"),
                                 IREE_SV("selected_hidden_states"));
  ASSERT_NE(expected_selected_hidden_states, nullptr)
      << "fixture must provide the expected Qwen selected-hidden tensor";

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_qwen3_vl_stage_plan_options_t qwen_options;
  std::memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = token_count;
  qwen_options.request.token_ids = token_ids.data();
  IREE_ASSERT_OK(ParseQwenWeightExecutionStrategy(
      &qwen_options.weight_execution_strategy));
  IREE_ASSERT_OK(
      ParseQwenAttentionImplementation(&qwen_options.attention_implementation));

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &qwen_options;
  std::vector<std::string> diagnostic_tap_name_storage;
  const iree_flag_string_list_t extra_diagnostic_taps =
      FLAG_id4_diagnostic_tap_list();
  diagnostic_tap_name_storage.reserve(3 + model_config->selected_layer_count +
                                      extra_diagnostic_taps.count);
  diagnostic_tap_name_storage.emplace_back("qwen3_vl.embedded_hidden_states");
  diagnostic_tap_name_storage.emplace_back("selected_hidden_states");
  diagnostic_tap_name_storage.emplace_back("qwen3_vl.output.tap");
  for (uint32_t i = 0; i < model_config->selected_layer_count; ++i) {
    diagnostic_tap_name_storage.emplace_back(
        "qwen3_vl.layers." +
        std::to_string(model_config->selected_layer_ordinals[i]) + ".output");
  }
  for (iree_host_size_t i = 0; i < extra_diagnostic_taps.count; ++i) {
    iree_string_view_t tap_name = extra_diagnostic_taps.values[i];
    diagnostic_tap_name_storage.emplace_back(tap_name.data, tap_name.size);
  }
  std::vector<iree_string_view_t> diagnostic_tap_names;
  diagnostic_tap_names.reserve(diagnostic_tap_name_storage.size());
  for (const std::string& tap_name : diagnostic_tap_name_storage) {
    diagnostic_tap_names.push_back(
        iree_make_string_view(tap_name.data(), tap_name.size()));
  }
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      diagnostic_tap_names.size(),
      diagnostic_tap_names.data(),
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  id4::test::SemaphoreListStorage prepare_signal;
  prepare_signal.semaphore = prepare_semaphore.get();
  prepare_signal.payload_value = 1;

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = parameter_provider.get();
  prepare_options.kernel_library = kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal.list();
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_ASSERT_OK(id4_pipeline_stage_prepare(stage.get(), plan.get(),
                                            &prepare_options, bundle.out()));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));
  id4::test::BufferBindingSet diagnostic_tap_bindings;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &diagnostic_tap_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      fill_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, fill_semaphore.out()));
  uint64_t fill_value = 0;
  IREE_ASSERT_OK(id4::test::QueueUpdateInitializedBoundaryTensorsFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, fixture_tensors, fill_semaphore.get(), &fill_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kExportedBoundarySentinel, sizeof(kExportedBoundarySentinel),
      fill_semaphore.get(), &fill_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      diagnostic_tap_bindings, &kExportedBoundarySentinel,
      sizeof(kExportedBoundarySentinel), fill_semaphore.get(), &fill_value));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  iree_hal_semaphore_list_t issue_wait_list = iree_hal_semaphore_list_empty();
  id4::test::SemaphoreListStorage issue_wait;
  if (fill_value != 0) {
    issue_wait.semaphore = fill_semaphore.get();
    issue_wait.payload_value = fill_value;
    issue_wait_list = issue_wait.list();
  }
  id4::test::SemaphoreListStorage issue_signal;
  issue_signal.semaphore = issue_semaphore.get();
  issue_signal.payload_value = 1;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.region_submission_window = 1;
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = issue_wait_list;
  issue_options.signal_semaphore_list = issue_signal.list();
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      issue_semaphore.get(), issue_signal.payload_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree::testing::TempFilePath temp_capture_directory;
  iree_string_view_t capture_directory =
      iree_make_cstring_view(FLAG_id4_capture_dir);
  if (iree_string_view_is_empty(capture_directory)) {
    temp_capture_directory =
        iree::testing::TempFilePath("id4_qwen3_vl_capture");
    capture_directory = temp_capture_directory.path_view();
  }

  id4::test::SemaphoreListStorage capture_wait;
  capture_wait.semaphore = issue_semaphore.get();
  capture_wait.payload_value = issue_signal.payload_value;
  id4_tooling_capture_execution_options_t capture_options;
  std::memset(&capture_options, 0, sizeof(capture_options));
  capture_options.structure_size = sizeof(capture_options);
  capture_options.run_id = IREE_SV("qwen3_vl_forward_integration");
  capture_options.output_directory = capture_directory;
  capture_options.plan = plan.get();
  capture_options.device = context.device.get();
  capture_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  capture_options.boundary_binding_count = boundary_bindings.count;
  capture_options.boundary_bindings = boundary_bindings.bindings;
  capture_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  capture_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  capture_options.wait_semaphore_list = capture_wait.list();
  capture_options.host_allocator = iree_allocator_system();
  IREE_ASSERT_OK(id4_tooling_capture_execution(&capture_options));
  IREE_ASSERT_OK(id4::test::VerifyCapturedExportedBoundaryTensorsWereWritten(
      plan.get(), capture_directory, kExportedBoundarySentinel));
  IREE_ASSERT_OK(id4::test::VerifyCapturedDiagnosticTapTensorsWereWritten(
      plan.get(), capture_directory, kExportedBoundarySentinel));

  iree_hal_buffer_binding_t condition_binding = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      plan.get(), boundary_bindings, IREE_SV("condition"), &condition_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &condition_binding,
      capture_wait.list(), *expected_condition));

  const id4_pipeline_diagnostic_tap_plan_t* selected_hidden_tap = nullptr;
  IREE_ASSERT_OK(FindDiagnosticTapPlan(
      plan.get(), IREE_SV("selected_hidden_states"), &selected_hidden_tap));
  iree_hal_buffer_binding_t selected_hidden_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings, IREE_SV("selected_hidden_states"),
      &selected_hidden_binding));
  id4::test::FixtureTensor selected_hidden_comparison =
      *expected_selected_hidden_states;
  if (selected_hidden_comparison.tolerance.mode ==
      id4::test::FixtureToleranceMode::kNone) {
    selected_hidden_comparison.tolerance = expected_condition->tolerance;
  }
  IREE_ASSERT_OK(id4::test::CompareBindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &selected_hidden_binding, capture_wait.list(),
      &selected_hidden_tap->layout, selected_hidden_comparison));
}

}  // namespace
