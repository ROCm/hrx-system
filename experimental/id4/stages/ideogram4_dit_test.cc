// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit.h"

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static id4_ideogram4_dit_model_config_t MakeModelConfig() {
  return id4_ideogram4_dit_model_config_t{
      // Number of transformer blocks in the DiT.
      /*.layer_count=*/1,
      // Channel count of each VAE latent image token.
      /*.input_channel_count=*/4,
      // Transformer hidden-state channel count.
      /*.hidden_size=*/8,
      // Feed-forward intermediate channel count.
      /*.intermediate_size=*/12,
      // Transformer attention head count.
      /*.attention_head_count=*/2,
      // AdaLN conditioning vector channel count.
      /*.adaln_size=*/4,
      // Qwen condition feature channel count.
      /*.llm_feature_count=*/52,
      // Number of image-indicator embedding rows.
      /*.image_indicator_count=*/2,
  };
}

static id4_pipeline_stage_t* CreateStage(
    iree_hal_device_group_t* device_group,
    id4_ideogram4_dit_model_config_t model) {
  id4_pipeline_stage_services_t services;
  memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_stage_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.model = model;

  id4_pipeline_stage_t* stage = nullptr;
  IREE_CHECK_OK(id4_ideogram4_dit_stage_create(
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

TEST(Ideogram4DitStage, PlansPreludeSliceFromRequestConfig) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_ideogram4_dit_model_config_t model = MakeModelConfig();
  id4_pipeline_stage_t* stage = CreateStage(device_group, model);

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t missing_dit_options;
  memset(&missing_dit_options, 0, sizeof(missing_dit_options));
  missing_dit_options.structure_size = sizeof(missing_dit_options);
  missing_dit_options.device_index = 0;
  missing_dit_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  missing_dit_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_stage_plan(stage, &missing_dit_options, &plan));
  EXPECT_EQ(plan, nullptr);

  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  id4_ideogram4_dit_stage_plan_options_t dit_options;
  memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request.latent_shape = latent_shape;
  dit_options.request.conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  EXPECT_EQ(id4::test::ToString(id4_pipeline_plan_stage_name(plan)),
            ID4_IDEOGRAM4_DIT_STAGE_NAME);
  EXPECT_GT(id4_pipeline_plan_parameter_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_memory_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_boundary_tensor_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_kernel_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_diagnostic_tap_count(plan), 0u);
  ASSERT_EQ(id4_pipeline_plan_region_count(plan), 1u);

  bool found_timestep_boundary = false;
  bool found_image_indicator_boundary = false;
  bool found_position_embedding_boundary = false;
  bool found_velocity_boundary = false;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary) continue;
    if (boundary->layout.dtype == ID4_PIPELINE_TENSOR_DTYPE_F32 &&
        ShapeEquals(boundary->layout.shape,
                    id4_pipeline_program_make_shape_rank1(1))) {
      found_timestep_boundary = true;
    }
    if (boundary->layout.dtype == ID4_PIPELINE_TENSOR_DTYPE_I32 &&
        ShapeEquals(boundary->layout.shape,
                    id4_pipeline_program_make_shape_rank2(2, 1))) {
      found_image_indicator_boundary = true;
    }
    if (boundary->layout.dtype == ID4_PIPELINE_TENSOR_DTYPE_F32 &&
        ShapeEquals(boundary->layout.shape,
                    id4_pipeline_program_make_shape_rank4(2, 2, 2, 2))) {
      found_position_embedding_boundary = true;
    }
    if (boundary->layout.dtype == ID4_PIPELINE_TENSOR_DTYPE_F32 &&
        iree_all_bits_set(boundary->flags,
                          ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
                              ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED) &&
        ShapeEquals(boundary->layout.shape, latent_shape)) {
      found_velocity_boundary = true;
    }
  }
  EXPECT_TRUE(found_timestep_boundary);
  EXPECT_TRUE(found_image_indicator_boundary);
  EXPECT_TRUE(found_position_embedding_boundary);
  EXPECT_TRUE(found_velocity_boundary);

  bool found_prelude_tap = false;
  bool found_adaln_tap = false;
  id4_pipeline_program_shape_t expected_tap_shape =
      id4_pipeline_program_make_shape_rank2(model.hidden_size, 2);
  id4_pipeline_program_shape_t expected_adaln_tap_shape =
      id4_pipeline_program_make_shape_rank1(model.adaln_size);
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_diagnostic_tap_count(plan);
       ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (!tap) continue;
    if (tap->layout.dtype != ID4_PIPELINE_TENSOR_DTYPE_F32) continue;
    if (ShapeEquals(tap->layout.shape, expected_tap_shape)) {
      found_prelude_tap = true;
    }
    if (ShapeEquals(tap->layout.shape, expected_adaln_tap_shape)) {
      found_adaln_tap = true;
    }
  }
  EXPECT_TRUE(found_prelude_tap);
  EXPECT_TRUE(found_adaln_tap);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

TEST(Ideogram4DitStage, PlansConditionedPreludeSliceFromRequestConfig) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_ideogram4_dit_model_config_t model = MakeModelConfig();
  id4_pipeline_stage_t* stage = CreateStage(device_group, model);

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_ideogram4_dit_stage_plan_options_t dit_options;
  memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request.latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 4, 1);
  dit_options.request.conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED;
  dit_options.request.text_token_count = 3;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  bool found_condition_boundary = false;
  bool found_image_indicator_boundary = false;
  bool found_position_embedding_boundary = false;
  bool found_velocity_boundary = false;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary) continue;
    if (boundary->layout.dtype == ID4_PIPELINE_TENSOR_DTYPE_F32 &&
        ShapeEquals(boundary->layout.shape,
                    id4_pipeline_program_make_shape_rank2(
                        model.llm_feature_count,
                        dit_options.request.text_token_count))) {
      found_condition_boundary = true;
    }
    if (boundary->layout.dtype == ID4_PIPELINE_TENSOR_DTYPE_I32 &&
        ShapeEquals(boundary->layout.shape,
                    id4_pipeline_program_make_shape_rank2(5, 1))) {
      found_image_indicator_boundary = true;
    }
    if (boundary->layout.dtype == ID4_PIPELINE_TENSOR_DTYPE_F32 &&
        ShapeEquals(boundary->layout.shape,
                    id4_pipeline_program_make_shape_rank4(2, 2, 2, 5))) {
      found_position_embedding_boundary = true;
    }
    if (boundary->layout.dtype == ID4_PIPELINE_TENSOR_DTYPE_F32 &&
        iree_all_bits_set(boundary->flags,
                          ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED |
                              ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED) &&
        ShapeEquals(boundary->layout.shape, dit_options.request.latent_shape)) {
      found_velocity_boundary = true;
    }
  }
  EXPECT_TRUE(found_condition_boundary);
  EXPECT_TRUE(found_image_indicator_boundary);
  EXPECT_TRUE(found_position_embedding_boundary);
  EXPECT_TRUE(found_velocity_boundary);

  bool found_prelude_tap = false;
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_diagnostic_tap_count(plan);
       ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (!tap) continue;
    if (tap->layout.dtype != ID4_PIPELINE_TENSOR_DTYPE_F32) continue;
    if (ShapeEquals(tap->layout.shape, id4_pipeline_program_make_shape_rank2(
                                           model.hidden_size, 5))) {
      found_prelude_tap = true;
    }
  }
  EXPECT_TRUE(found_prelude_tap);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

TEST(Ideogram4DitStage, RejectsInvalidStaticModelConfig) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();

  id4_pipeline_stage_services_t services;
  memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_stage_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.model = MakeModelConfig();
  create_options.model.input_channel_count = 0;

  id4_pipeline_stage_t* stage = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_dit_stage_create(
                            &create_options, iree_allocator_system(), &stage));
  EXPECT_EQ(stage, nullptr);

  iree_hal_device_group_release(device_group);
}

TEST(Ideogram4DitStage, RejectsInvalidRequestShape) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = CreateStage(device_group, MakeModelConfig());

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_ideogram4_dit_stage_plan_options_t dit_options;
  memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request.latent_shape =
      id4_pipeline_program_make_shape_rank4(1, 2, 8, 1);
  dit_options.request.conditioning_mode =
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_pipeline_stage_plan(stage, &plan_options, &plan));
  EXPECT_EQ(plan, nullptr);

  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

}  // namespace
