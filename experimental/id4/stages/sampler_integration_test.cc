// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/sampler.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, id4_fixture_dir, "",
          "Directory containing an ID4 sampler fixture manifest and tensor "
          "payloads used to initialize stage inputs and verify outputs.");

namespace {

static iree_status_t CreateSamplerStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_sampler_denoise_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  return id4_sampler_denoise_stage_create(&create_options,
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

static iree_string_view_t FixtureTensorName(
    const id4::test::FixtureTensor& tensor) {
  return iree_make_string_view(tensor.name.data(), tensor.name.size());
}

static std::vector<float> ToF32Vector(const std::vector<uint8_t>& bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

static void ExpectF32TensorNear(const std::vector<uint8_t>& actual_bytes,
                                const id4::test::FixtureTensor& expected) {
  ASSERT_EQ(expected.dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  ASSERT_EQ(expected.tolerance.mode,
            id4::test::FixtureToleranceMode::kElementwise);
  ASSERT_EQ(actual_bytes.size(), expected.payload.size());

  std::vector<float> actual = ToF32Vector(actual_bytes);
  std::vector<float> expected_values = ToF32Vector(expected.payload);
  ASSERT_EQ(actual.size(), expected_values.size());
  for (iree_host_size_t i = 0; i < actual.size(); ++i) {
    const double tolerance = expected.tolerance.absolute_tolerance +
                             expected.tolerance.relative_tolerance *
                                 std::fabs((double)expected_values[i]);
    EXPECT_NEAR(actual[i], expected_values[i], tolerance) << "element " << i;
  }
}

TEST(SamplerDenoiseStageIntegration, PrepareAndIssueDenoiseStepFixture) {
  iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  ASSERT_FALSE(iree_string_view_is_empty(fixture_directory))
      << "--id4_fixture_dir is required for sampler integration correctness";

  id4::test::FixtureTensorSet fixture_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(fixture_directory, &fixture_tensors));

  const id4::test::FixtureTensor* cond_out = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                   IREE_SV("cond_out"), &cond_out));
  id4_pipeline_program_shape_t latent_shape;
  IREE_ASSERT_OK(MakeProgramShape(cond_out->shape, &latent_shape));

  const id4::test::FixtureTensor* expected_guided_pred = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("expected"),
                                   IREE_SV("guided_pred"),
                                   &expected_guided_pred));
  const id4::test::FixtureTensor* expected_denoised = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("expected"),
                                   IREE_SV("denoised"), &expected_denoised));

  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_ASSERT_OK(CreateSamplerStage(context, stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_sampler_denoise_stage_plan_options_t sampler_options;
  std::memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape = latent_shape;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &sampler_options;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("guided_pred"),
  };
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.kernel_library = kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
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
  IREE_ASSERT_OK(id4::test::QueueUpdateBoundaryTensorFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, IREE_SV("cond_out"), fixture_tensors,
      IREE_SV("cond_out"), update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateBoundaryTensorFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, IREE_SV("uncond_out"), fixture_tensors,
      IREE_SV("uncond_out"), update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateBoundaryTensorFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, IREE_SV("x_t"), fixture_tensors, IREE_SV("x_t"),
      update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateBoundaryTensorFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, IREE_SV("scalings"), fixture_tensors,
      IREE_SV("scalings"), update_semaphore.get(), &update_value));
  IREE_ASSERT_OK(id4::test::QueueUpdateBoundaryTensorFromFixture(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, IREE_SV("guidance"), fixture_tensors,
      IREE_SV("guidance"), update_semaphore.get(), &update_value));
  const float step_sigmas[2] = {1.0f, 0.0f};
  iree_hal_buffer_binding_t sigmas_binding = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      plan.get(), boundary_bindings, IREE_SV("sigmas"), &sigmas_binding));
  IREE_ASSERT_OK(id4::test::QueueUpdateBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &sigmas_binding,
      step_sigmas, sizeof(step_sigmas), update_semaphore.get(), &update_value));

  const uint8_t sentinel = 0xA5;
  iree_hal_buffer_binding_t denoised_binding = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      plan.get(), boundary_bindings, FixtureTensorName(*expected_denoised),
      &denoised_binding));
  IREE_ASSERT_OK(id4::test::QueueFillBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &denoised_binding,
      &sentinel, sizeof(sentinel), update_semaphore.get(), &update_value));

  iree_hal_buffer_binding_t guided_pred_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      FixtureTensorName(*expected_guided_pred), &guided_pred_binding));
  IREE_ASSERT_OK(id4::test::QueueFillBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &guided_pred_binding,
      &sentinel, sizeof(sentinel), update_semaphore.get(), &update_value));

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
  issue_options.region_submission_window = 1;
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
  std::vector<uint8_t> denoised_bytes;
  IREE_ASSERT_OK(id4::test::ReadBindingToHost(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &denoised_binding,
      read_wait.list(), &denoised_bytes));
  std::vector<uint8_t> guided_pred_bytes;
  IREE_ASSERT_OK(id4::test::ReadBindingToHost(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &guided_pred_binding,
      read_wait.list(), &guided_pred_bytes));

  ExpectF32TensorNear(guided_pred_bytes, *expected_guided_pred);
  ExpectF32TensorNear(denoised_bytes, *expected_denoised);
}

}  // namespace
