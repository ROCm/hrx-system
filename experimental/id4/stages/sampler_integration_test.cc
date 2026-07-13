// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

static id4::test::LiveStageContext& SharedLiveStageContext() {
  static id4::test::LiveStageContext context;
  static const bool initialized = []() {
    IREE_CHECK_OK(id4::test::CreateLiveStageContextFromFlags(&context));
    return true;
  }();
  (void)initialized;
  return context;
}

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

static iree_status_t CreateNoiseStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_sampler_noise_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  return id4_sampler_noise_stage_create(&create_options,
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

static std::vector<float> ToF32Vector(const std::vector<uint8_t>& bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
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
  const id4::test::FixtureTensor* uncond_out = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                   IREE_SV("uncond_out"), &uncond_out));
  const id4::test::FixtureTensor* x_t = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                   IREE_SV("x_t"), &x_t));
  id4_pipeline_program_shape_t latent_shape;
  IREE_ASSERT_OK(MakeProgramShape(cond_out->shape, &latent_shape));

  id4::test::LiveStageContext& context = SharedLiveStageContext();

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
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_policy = id4_pipeline_stage_no_parameters();
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
  const float step_values[3] = {0.25f, 0.5f, 7.0f};
  iree_hal_buffer_binding_t step_binding = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      plan.get(), boundary_bindings, IREE_SV("step"), &step_binding));
  IREE_ASSERT_OK(id4::test::QueueUpdateBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &step_binding,
      step_values, sizeof(step_values), update_semaphore.get(), &update_value));

  const uint8_t sentinel = 0xA5;
  iree_hal_buffer_binding_t x_next_binding = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      plan.get(), boundary_bindings, IREE_SV("x_next"), &x_next_binding));
  IREE_ASSERT_OK(id4::test::QueueFillBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &x_next_binding,
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
  issue_options.execution_segment_submission_window = 1;
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
  std::vector<uint8_t> x_next_bytes;
  IREE_ASSERT_OK(id4::test::ReadBindingToHost(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &x_next_binding,
      read_wait.list(), &x_next_bytes));

  const std::vector<float> conditioned_values = ToF32Vector(cond_out->payload);
  const std::vector<float> unconditioned_values =
      ToF32Vector(uncond_out->payload);
  const std::vector<float> x_values = ToF32Vector(x_t->payload);
  const std::vector<float> actual_values = ToF32Vector(x_next_bytes);
  ASSERT_EQ(actual_values.size(), conditioned_values.size());
  ASSERT_EQ(actual_values.size(), unconditioned_values.size());
  ASSERT_EQ(actual_values.size(), x_values.size());
  const float flow_delta = step_values[1] - step_values[0];
  for (iree_host_size_t i = 0; i < actual_values.size(); ++i) {
    const float guided_velocity =
        unconditioned_values[i] +
        step_values[2] * (conditioned_values[i] - unconditioned_values[i]);
    const float expected = x_values[i] + guided_velocity * flow_delta;
    EXPECT_NEAR(actual_values[i], expected, 1e-5f) << "element " << i;
  }
}

static iree_status_t RunNoiseStage(id4::test::LiveStageContext& context,
                                   id4_pipeline_program_shape_t latent_shape,
                                   uint64_t seed,
                                   std::vector<float>* out_values) {
  out_values->clear();
  id4::test::KernelLibraryRef kernel_library;
  IREE_RETURN_IF_ERROR(
      id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_RETURN_IF_ERROR(CreateNoiseStage(context, stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);
  id4_pipeline_stage_load_options_t load_options = {};
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_RETURN_IF_ERROR(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_sampler_noise_stage_plan_options_t sampler_options = {};
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape = latent_shape;
  id4_pipeline_stage_plan_options_t plan_options = {};
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &sampler_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  id4_pipeline_stage_prepare_options_t prepare_options = {};
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_policy = id4_pipeline_stage_no_parameters();
  prepare_options.kernel_library = kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &diagnostics_sink;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_RETURN_IF_ERROR(id4_pipeline_stage_prepare(
      stage.get(), plan.get(), &prepare_options, bundle.out()));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_RETURN_IF_ERROR(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));
  id4::test::BufferBindingSet diagnostic_tap_bindings;
  IREE_RETURN_IF_ERROR(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &diagnostic_tap_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  const uint32_t seed_words[] = {
      (uint32_t)seed,
      (uint32_t)(seed >> 32),
  };
  iree_hal_buffer_binding_t seed_binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(
      plan.get(), boundary_bindings, IREE_SV("seed"), &seed_binding));
  IREE_RETURN_IF_ERROR(id4::test::QueueUpdateBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &seed_binding,
      seed_words, sizeof(seed_words), update_semaphore.get(), &update_value));
  const uint8_t sentinel = 0xA5;
  iree_hal_buffer_binding_t output_binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(
      plan.get(), boundary_bindings, IREE_SV("x"), &output_binding));
  IREE_RETURN_IF_ERROR(id4::test::QueueFillBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &output_binding,
      &sentinel, sizeof(sentinel), update_semaphore.get(), &update_value));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));
  id4::test::SemaphoreListStorage issue_wait;
  issue_wait.semaphore = update_semaphore.get();
  issue_wait.payload_value = update_value;
  id4::test::SemaphoreListStorage issue_signal;
  issue_signal.semaphore = issue_semaphore.get();
  issue_signal.payload_value = 1;

  id4_pipeline_stage_issue_options_t issue_options = {};
  issue_options.structure_size = sizeof(issue_options);
  issue_options.execution_segment_submission_window = 1;
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = issue_wait.list();
  issue_options.signal_semaphore_list = issue_signal.list();
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));

  id4::test::SemaphoreListStorage read_wait;
  read_wait.semaphore = issue_semaphore.get();
  read_wait.payload_value = issue_signal.payload_value;
  std::vector<uint8_t> actual_bytes;
  IREE_RETURN_IF_ERROR(id4::test::ReadBindingToHost(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &output_binding,
      read_wait.list(), &actual_bytes));
  *out_values = ToF32Vector(actual_bytes);
  return iree_ok_status();
}

TEST(SamplerNoiseStageIntegration, MatchesReferenceSeededNoise) {
  static const float expected_values[] = {
      -0.64940690994262695f,  0.12801173329353333f,  -1.1790484189987183f,
      2.0362670421600342f,    -0.30775544047355652f, 1.1522955894470215f,
      -0.089752890169620514f, -0.59968787431716919f, 0.032890830188989639f,
      -0.30281072854995728f,  -0.45105797052383423f, -0.82258480787277222f,
      1.5867397785186768f,    -0.19672147929668427f, 0.74773162603378296f,
      1.3601166009902954f,
  };
  std::vector<float> actual_values;
  IREE_ASSERT_OK(
      RunNoiseStage(SharedLiveStageContext(),
                    id4_pipeline_program_make_shape_rank4(1, 1, 128, 1),
                    20260625, &actual_values));
  ASSERT_GE(actual_values.size(), IREE_ARRAYSIZE(expected_values));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_values); ++i) {
    EXPECT_FLOAT_EQ(actual_values[i], expected_values[i]) << "element " << i;
  }
}

TEST(SamplerNoiseStageIntegration, MatchesReferenceFullLatentMapping) {
  const struct {
    iree_host_size_t index;
    float value;
  } reference_anchors[] = {
      {0, -0.649406909942627f},     {517631, -2.22067928314209f},
      {1536, 0.09900129586458206f}, {519167, -0.053319673985242844f},
      {3072, 0.3446367681026459f},  {520703, 0.7380797266960144f},
      {4608, -1.1011921167373657f}, {522239, -0.773559033870697f},
      {6144, -1.7609195709228516f}, {523775, -0.822858989238739f},
      {7680, -0.4902384281158447f}, {524287, 0.4932446777820587f},
  };
  std::vector<float> actual_values;
  IREE_ASSERT_OK(
      RunNoiseStage(SharedLiveStageContext(),
                    id4_pipeline_program_make_shape_rank4(64, 64, 128, 1),
                    20260625, &actual_values));
  ASSERT_EQ(actual_values.size(), 524288u);
  for (const auto& anchor : reference_anchors) {
    EXPECT_FLOAT_EQ(actual_values[anchor.index], anchor.value)
        << "element " << anchor.index;
  }
}

}  // namespace
