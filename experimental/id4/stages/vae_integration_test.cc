// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/vae.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static id4_vae_model_config_t MakeSmallModelConfig() {
  return id4_vae_model_config_t{
      // Latent-to-image scale factor along the width axis.
      /*.scale_x=*/2,
      // Latent-to-image scale factor along the height axis.
      /*.scale_y=*/2,
      // Latent-to-media scale factor along the temporal axis.
      /*.scale_t=*/1,
      // Channel count in latent tensors.
      /*.latent_channel_count=*/1,
      // Channel count in decoded tensors.
      /*.decoded_channel_count=*/1,
      // Minimum latent tile width.
      /*.min_tile_size_x=*/1,
      // Minimum latent tile height.
      /*.min_tile_size_y=*/1,
      // Default latent tile width.
      /*.default_tile_size_x=*/2,
      // Default latent tile height.
      /*.default_tile_size_y=*/2,
      // Maximum legal overlap.
      /*.max_overlap=*/0.5f,
      // Supported implementation capabilities.
      /*.capabilities=*/ID4_VAE_CAPABILITY_DECODE |
          ID4_VAE_CAPABILITY_SPATIAL_TILING,
  };
}

static iree_status_t CreateVaeStage(const id4::test::LiveStageContext& context,
                                    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_vae_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  create_options.model = MakeSmallModelConfig();
  create_options.activation_format = ID4_VAE_ACTIVATION_FORMAT_F32_CANONICAL;
  return id4_vae_stage_create(&create_options, iree_allocator_system(),
                              out_stage);
}

static std::vector<float> ToF32Vector(const std::vector<uint8_t>& bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

TEST(VaeStageIntegration, PrepareAndIssueDecodeNearest) {
  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_ASSERT_OK(CreateVaeStage(context, stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_vae_stage_plan_options_t vae_options;
  std::memset(&vae_options, 0, sizeof(vae_options));
  vae_options.structure_size = sizeof(vae_options);
  vae_options.request.latent_shape =
      id4_pipeline_program_make_shape_rank4(2, 2, 1, 1);
  vae_options.request.tiling.mode = ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE;
  vae_options.request.tiling.tile_size_x = 2;
  vae_options.request.tiling.tile_size_y = 2;
  vae_options.request.tiling.overlap = 0.0f;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &vae_options;
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

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  const float latent_value = 2.0f;
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED,
      &latent_value, sizeof(latent_value), update_semaphore.get(),
      &update_value));

  iree_hal_buffer_binding_t decoded_binding = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(plan.get(), boundary_bindings,
                                                IREE_SV("media.image.decoded"),
                                                &decoded_binding));
  const uint32_t sentinel = 0xA5A5A5A5u;
  IREE_ASSERT_OK(id4::test::QueueFillBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &decoded_binding,
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
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.wait_semaphore_list = issue_wait.list();
  issue_options.signal_semaphore_list = issue_signal.list();
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));

  id4::test::SemaphoreListStorage read_wait;
  read_wait.semaphore = issue_semaphore.get();
  read_wait.payload_value = issue_signal.payload_value;
  std::vector<uint8_t> decoded_bytes;
  IREE_ASSERT_OK(id4::test::ReadBindingToHost(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &decoded_binding,
      read_wait.list(), &decoded_bytes));

  std::vector<float> decoded = ToF32Vector(decoded_bytes);
  ASSERT_EQ(decoded.size(), 16u);
  for (iree_host_size_t i = 0; i < decoded.size(); ++i) {
    EXPECT_EQ(decoded[i], latent_value) << "decoded element " << i;
  }
}

}  // namespace
