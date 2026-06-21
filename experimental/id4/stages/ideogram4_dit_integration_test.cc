// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <limits>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/tooling/capture.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, id4_capture_dir, "",
          "Directory that receives exported ID4 DiT boundary and diagnostic "
          "tap tensor captures.");
IREE_FLAG(
    string, id4_fixture_dir, "",
    "Directory containing an ID4 DiT fixture manifest and tensor "
    "payloads used to initialize stage inputs and verify reference taps.");

namespace {

constexpr uint8_t kOutputSentinel = 0xA5;

static iree_status_t CreateIdeogram4DitStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  create_options.model = *id4_ideogram4_dit_program_ideogram4_model_config();
  return id4_ideogram4_dit_stage_create(&create_options,
                                        iree_allocator_system(), out_stage);
}

static iree_status_t FindFixtureTensor(
    const id4::test::FixtureTensorSet& fixture_tensors, iree_string_view_t role,
    iree_string_view_t name, const id4::test::FixtureTensor** out_tensor) {
  *out_tensor = fixture_tensors.FindTensor(role, name);
  if (*out_tensor) return iree_ok_status();
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "fixture tensor `%.*s` with role `%.*s` not found",
                          static_cast<int>(name.size), name.data,
                          static_cast<int>(role.size), role.data);
}

static iree_string_view_t FixtureTensorName(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.name.data(), tensor.name.size());
}

static iree_string_view_t FixtureTensorRole(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.role.data(), tensor.role.size());
}

static iree_status_t MakeProgramShape(
    id4_pipeline_tensor_shape_t tensor_shape,
    id4_pipeline_program_shape_t* out_program_shape) {
  if (tensor_shape.rank > IREE_ARRAYSIZE(out_program_shape->dims)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixture tensor rank %u exceeds program max rank",
                            tensor_shape.rank);
  }
  std::memset(out_program_shape, 0, sizeof(*out_program_shape));
  out_program_shape->rank = tensor_shape.rank;
  for (uint32_t i = 0; i < tensor_shape.rank; ++i) {
    out_program_shape->dims[i] = tensor_shape.dims[i];
  }
  return iree_ok_status();
}

static iree_status_t ConfigureRequestFromFixture(
    const id4::test::FixtureTensorSet& fixture_tensors,
    id4_ideogram4_dit_request_config_t* out_request) {
  std::memset(out_request, 0, sizeof(*out_request));

  const id4::test::FixtureTensor* latent = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                         IREE_SV("x"), &latent));
  IREE_RETURN_IF_ERROR(
      MakeProgramShape(latent->shape, &out_request->latent_shape));

  const id4::test::FixtureTensor* condition =
      fixture_tensors.FindTensor(IREE_SV("input"), IREE_SV("condition"));
  if (!condition) {
    out_request->conditioning_mode =
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED;
    return iree_ok_status();
  }
  if (condition->shape.rank != 2 ||
      condition->shape.dims[1] >
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "condition fixture tensor must be rank-2 with uint32 token count");
  }
  out_request->conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  out_request->text_token_count =
      static_cast<uint32_t>(condition->shape.dims[1]);
  return iree_ok_status();
}

static iree_status_t CompareExpectedDiagnosticTaps(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& diagnostic_tap_bindings,
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_hal_semaphore_list_t wait_list) {
  iree_host_size_t expected_count = 0;
  for (const id4::test::FixtureTensor& tensor : fixture_tensors.tensors) {
    if (!iree_string_view_equal(FixtureTensorRole(tensor),
                                IREE_SV("expected"))) {
      continue;
    }
    ++expected_count;
    iree_hal_buffer_binding_t binding = {};
    IREE_RETURN_IF_ERROR(id4::test::FindDiagnosticTapBinding(
        plan, diagnostic_tap_bindings, FixtureTensorName(tensor), &binding));
    IREE_RETURN_IF_ERROR(id4::test::CompareF32BindingWithFixtureTensor(
        device, queue_affinity, &binding, wait_list, tensor));
  }
  if (expected_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "DiT fixture contains no expected tensors");
  }
  return iree_ok_status();
}

static iree_status_t VerifyExportedBoundariesWereWritten(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_list_t wait_list) {
  iree_host_size_t exported_count = 0;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary ||
        !iree_all_bits_set(boundary->flags,
                           ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
      continue;
    }
    ++exported_count;
    std::vector<uint8_t> bytes;
    IREE_RETURN_IF_ERROR(id4::test::ReadBindingToHost(
        device, queue_affinity, &boundary_bindings.bindings[i], wait_list,
        &bytes));
    bool all_sentinel = true;
    for (uint8_t byte : bytes) {
      if (byte != kOutputSentinel) {
        all_sentinel = false;
        break;
      }
    }
    if (all_sentinel) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "exported boundary `%.*s` was not written",
                              static_cast<int>(boundary->layout.name.size),
                              boundary->layout.name.data);
    }
  }
  if (exported_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "DiT plan contains no exported boundaries");
  }
  return iree_ok_status();
}

TEST(Ideogram4DitStageIntegration, PrepareAndIssueForwardPreludeFixture) {
  iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  ASSERT_FALSE(iree_string_view_is_empty(fixture_directory))
      << "--id4_fixture_dir is required for DiT integration correctness";

  id4::test::FixtureTensorSet fixture_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(fixture_directory, &fixture_tensors));

  id4_ideogram4_dit_request_config_t request;
  IREE_ASSERT_OK(ConfigureRequestFromFixture(fixture_tensors, &request));

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
  IREE_ASSERT_OK(CreateIdeogram4DitStage(context, stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_ideogram4_dit_stage_plan_options_t dit_options;
  std::memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request = request;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
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
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  IREE_ASSERT_OK(id4::test::QueueUpdateInitializedBoundaryTensorsFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, fixture_tensors, update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      diagnostic_tap_bindings, &kOutputSentinel, sizeof(kOutputSentinel),
      update_semaphore.get(), &update_value));
  ASSERT_GT(update_value, 0u);

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  id4::test::SemaphoreListStorage issue_wait;
  issue_wait.semaphore = update_semaphore.get();
  issue_wait.payload_value = update_value;
  id4::test::SemaphoreListStorage issue_signal;
  issue_signal.semaphore = issue_semaphore.get();
  issue_signal.payload_value = 1;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = issue_wait.list();
  issue_options.signal_semaphore_list = issue_signal.list();
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));

  id4::test::SemaphoreListStorage read_wait;
  read_wait.semaphore = issue_semaphore.get();
  read_wait.payload_value = issue_signal.payload_value;
  iree_string_view_t capture_directory =
      iree_make_cstring_view(FLAG_id4_capture_dir);
  if (!iree_string_view_is_empty(capture_directory)) {
    id4_tooling_capture_execution_options_t capture_options;
    std::memset(&capture_options, 0, sizeof(capture_options));
    capture_options.structure_size = sizeof(capture_options);
    capture_options.run_id = IREE_SV("ideogram4_dit_forward_integration");
    capture_options.output_directory = capture_directory;
    capture_options.plan = plan.get();
    capture_options.device = context.device.get();
    capture_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    capture_options.boundary_binding_count = boundary_bindings.count;
    capture_options.boundary_bindings = boundary_bindings.bindings;
    capture_options.diagnostic_tap_binding_count =
        diagnostic_tap_bindings.count;
    capture_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
    capture_options.wait_semaphore_list = read_wait.list();
    capture_options.host_allocator = iree_allocator_system();
    IREE_ASSERT_OK(id4_tooling_capture_execution(&capture_options));
    IREE_ASSERT_OK(id4::test::VerifyCapturedExportedBoundaryTensorsWereWritten(
        plan.get(), capture_directory, kOutputSentinel));
    IREE_ASSERT_OK(id4::test::VerifyCapturedDiagnosticTapTensorsWereWritten(
        plan.get(), capture_directory, kOutputSentinel));
  }
  IREE_ASSERT_OK(CompareExpectedDiagnosticTaps(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      diagnostic_tap_bindings, fixture_tensors, read_wait.list()));
  IREE_ASSERT_OK(VerifyExportedBoundariesWereWritten(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, read_wait.list()));
}

}  // namespace
