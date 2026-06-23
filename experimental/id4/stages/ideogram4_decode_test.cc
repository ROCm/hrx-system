// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_decode.h"

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static id4_vae_model_config_t MakeSmallVaeConfig() {
  return id4_vae_model_config_t{
      // Latent-to-image scale factor along the width axis.
      /*.scale_x=*/2,
      // Latent-to-image scale factor along the height axis.
      /*.scale_y=*/2,
      // Latent-to-media scale factor along the temporal axis.
      /*.scale_t=*/1,
      // Channel count in latent tensors.
      /*.latent_channel_count=*/4,
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

static id4_ideogram4_decode_model_config_t MakeSmallModelConfig() {
  return id4_ideogram4_decode_model_config_t{
      // Reusable VAE decode model configuration.
      /*.vae=*/MakeSmallVaeConfig(),
  };
}

static id4_pipeline_stage_t* CreateStage(
    iree_hal_device_group_t* device_group,
    id4_ideogram4_decode_model_config_t model) {
  id4_pipeline_stage_services_t services;
  memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_decode_stage_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.model = model;

  id4_pipeline_stage_t* stage = nullptr;
  IREE_CHECK_OK(id4_ideogram4_decode_stage_create(
      &create_options, iree_allocator_system(), &stage));
  return stage;
}

static bool ShapeEquals(id4_pipeline_tensor_shape_t actual,
                        id4_pipeline_program_shape_t expected) {
  if (actual.rank != expected.rank) return false;
  for (uint32_t i = 0; i < actual.rank; ++i) {
    if (actual.dims[i] != expected.dims[i]) return false;
  }
  return true;
}

TEST(Ideogram4DecodeStage, PlansLatentToImageDecode) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage =
      CreateStage(device_group, MakeSmallModelConfig());

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t missing_decode_options;
  memset(&missing_decode_options, 0, sizeof(missing_decode_options));
  missing_decode_options.structure_size = sizeof(missing_decode_options);
  missing_decode_options.device_index = 0;
  missing_decode_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  missing_decode_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_stage_plan(stage, &missing_decode_options, &plan));
  EXPECT_EQ(plan, nullptr);

  id4_ideogram4_decode_stage_plan_options_t decode_options;
  memset(&decode_options, 0, sizeof(decode_options));
  decode_options.structure_size = sizeof(decode_options);
  decode_options.request.diffusion_latent_shape =
      id4_pipeline_program_make_shape_rank4(2, 3, 4, 1);
  decode_options.request.vae_tiling.mode =
      ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE;
  decode_options.request.vae_tiling.tile_size_x = 2;
  decode_options.request.vae_tiling.tile_size_y = 2;
  decode_options.request.vae_tiling.overlap = 0.0f;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &decode_options;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("vae.decode.tile"),
  };
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  EXPECT_EQ(id4::test::ToString(id4_pipeline_plan_stage_name(plan)),
            ID4_IDEOGRAM4_DECODE_STAGE_NAME);
  EXPECT_GT(id4_pipeline_plan_memory_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_boundary_tensor_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_kernel_count(plan), 0u);
  EXPECT_EQ(id4_pipeline_plan_diagnostic_tap_count(plan), 1u);

  const id4_pipeline_boundary_tensor_plan_t* exported_boundary = nullptr;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary &&
        iree_all_bits_set(boundary->flags,
                          ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED) &&
        ShapeEquals(boundary->layout.shape,
                    id4_pipeline_program_make_shape_rank4(4, 6, 1, 1))) {
      exported_boundary = boundary;
      break;
    }
  }
  ASSERT_NE(exported_boundary, nullptr);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

TEST(Ideogram4DecodeStage, RejectsInvalidStaticModelConfig) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();

  id4_pipeline_stage_services_t services;
  memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_decode_model_config_t model = MakeSmallModelConfig();
  model.vae.latent_channel_count = 0;

  id4_ideogram4_decode_stage_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.model = model;

  id4_pipeline_stage_t* stage = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_decode_stage_create(
                            &create_options, iree_allocator_system(), &stage));
  EXPECT_EQ(stage, nullptr);

  iree_hal_device_group_release(device_group);
}

}  // namespace
