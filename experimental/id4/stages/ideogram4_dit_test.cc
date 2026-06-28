// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit.h"

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
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
      /*.hidden_size=*/32,
      // Feed-forward intermediate channel count.
      /*.intermediate_size=*/64,
      // Transformer attention head count.
      /*.attention_head_count=*/2,
      // AdaLN conditioning vector channel count.
      /*.adaln_size=*/4,
      // Qwen condition feature channel count.
      /*.llm_feature_count=*/208,
      // Number of image-indicator embedding rows.
      /*.image_indicator_count=*/2,
  };
}

static id4_pipeline_stage_t* CreateStage(
    iree_hal_device_group_t* device_group,
    id4_ideogram4_dit_model_config_t model,
    iree_host_size_t parameter_source_rule_count = 0,
    const id4_ideogram4_dit_parameter_source_rule_t* parameter_source_rules =
        nullptr) {
  id4_pipeline_stage_services_t services;
  memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_stage_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.parameter_scope = IREE_SV("model");
  create_options.parameter_source_rule_count = parameter_source_rule_count;
  create_options.parameter_source_rules = parameter_source_rules;
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

static const id4_pipeline_diagnostic_tap_plan_t* FindDiagnosticTap(
    const id4_pipeline_plan_t* plan, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_diagnostic_tap_count(plan);
       ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (tap && iree_string_view_equal(tap->name, name)) return tap;
  }
  return nullptr;
}

static const id4_pipeline_parameter_request_t* FindParameterSlabRequest(
    const id4_pipeline_plan_t* plan, iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_parameter_slab_count(plan);
       ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab =
        id4_pipeline_plan_parameter_slab_at(plan, i);
    if (!slab) continue;
    for (iree_host_size_t j = 0; j < slab->request_count; ++j) {
      const id4_pipeline_parameter_request_t* request = &slab->requests[j];
      if (iree_string_view_equal(request->key, key)) return request;
    }
  }
  return nullptr;
}

static const id4_pipeline_parameter_request_t* LoadStepTargetRequest(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_load_step_t* load_step) {
  if (!load_step || load_step->request_count != 1) return nullptr;
  const id4_pipeline_parameter_slab_plan_t* slab =
      id4_pipeline_plan_parameter_slab_at(plan, load_step->target_slab_index);
  if (!slab || load_step->request_offset >= slab->request_count) return nullptr;
  return &slab->requests[load_step->request_offset];
}

static const id4_pipeline_parameter_load_step_t* FindEncodedLoadStep(
    const id4_pipeline_plan_t* plan, iree_string_view_t key) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_parameter_load_step_count(plan); ++i) {
    const id4_pipeline_parameter_load_step_t* load_step =
        id4_pipeline_plan_parameter_load_step_at(plan, i);
    if (!load_step) continue;
    if (load_step->kind !=
        ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16) {
      continue;
    }
    if (load_step->source_count != 2) continue;
    if (iree_string_view_equal(load_step->sources[0].key, key)) {
      return load_step;
    }
  }
  return nullptr;
}

static void ExpectScaledFp8SourceToBf16ExecutionParameter(
    const id4_pipeline_plan_t* plan, iree_string_view_t source_scope,
    iree_string_view_t key, iree_string_view_t scale_key, uint64_t output_size,
    uint64_t input_size) {
  const id4_pipeline_parameter_load_step_t* load_step =
      FindEncodedLoadStep(plan, key);
  ASSERT_NE(load_step, nullptr);
  ASSERT_EQ(load_step->source_count, 2u);
  ASSERT_EQ(load_step->request_count, 1u);

  const id4_pipeline_parameter_load_source_t* weight_source =
      &load_step->sources[0];
  EXPECT_TRUE(
      iree_string_view_equal(weight_source->source_scope, source_scope));
  EXPECT_TRUE(iree_string_view_equal(weight_source->key, key));
  EXPECT_EQ(weight_source->dtype, ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3);
  EXPECT_TRUE(ShapeEquals(
      weight_source->shape,
      id4_pipeline_program_make_shape_rank2(output_size, input_size)));
  EXPECT_EQ(weight_source->byte_length, output_size * input_size);

  const id4_pipeline_parameter_load_source_t* scale_source =
      &load_step->sources[1];
  EXPECT_TRUE(iree_string_view_equal(scale_source->source_scope, source_scope));
  EXPECT_TRUE(iree_string_view_equal(scale_source->key, scale_key));
  EXPECT_EQ(scale_source->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  EXPECT_TRUE(ShapeEquals(scale_source->shape,
                          id4_pipeline_program_make_shape_rank1(output_size)));
  EXPECT_EQ(scale_source->byte_length, output_size * sizeof(float));

  const id4_pipeline_parameter_request_t* target_request =
      LoadStepTargetRequest(plan, load_step);
  ASSERT_NE(target_request, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target_request->key, key));
  EXPECT_EQ(target_request->span.length,
            output_size * input_size * sizeof(uint16_t));

  EXPECT_EQ(FindParameterSlabRequest(plan, scale_key), nullptr);
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
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
  dit_options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.uncond.prelude.hidden"),
      IREE_SV("ideogram4.uncond.prelude.adaln_input"),
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
            ID4_IDEOGRAM4_DIT_STAGE_NAME);
  EXPECT_GT(id4_pipeline_plan_parameter_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_memory_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_boundary_tensor_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_kernel_count(plan), 0u);
  EXPECT_EQ(id4_pipeline_plan_diagnostic_tap_count(plan), 2u);
  ASSERT_EQ(id4_pipeline_plan_region_count(plan), 1u);

  const uint32_t head_size = model.hidden_size / model.attention_head_count;
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
        ShapeEquals(
            boundary->layout.shape,
            id4_pipeline_program_make_shape_rank4(2, 2, head_size / 2, 2))) {
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
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
  dit_options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.cond.prelude.hidden"),
  };
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  const uint32_t head_size = model.hidden_size / model.attention_head_count;
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
        ShapeEquals(
            boundary->layout.shape,
            id4_pipeline_program_make_shape_rank4(2, 2, head_size / 2, 5))) {
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

TEST(Ideogram4DitStage, PlansMaterializedWmmaAttention) {
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
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED;
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  dit_options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));
  EXPECT_GT(id4_pipeline_plan_memory_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_kernel_count(plan), 0u);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

TEST(Ideogram4DitStage, PlansPyTorchParityMlpDiagnosticTaps) {
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
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED;
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  dit_options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA;

  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("ideogram4.uncond.layers.0.ffn.input"),
      IREE_SV("ideogram4.uncond.layers.0.ffn.w1_projection.output"),
      IREE_SV("ideogram4.uncond.layers.0.ffn.w3_projection.output"),
      IREE_SV("ideogram4.uncond.layers.0.ffn.hidden"),
      IREE_SV("ideogram4.uncond.layers.0.ffn.output"),
  };
  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));
  EXPECT_EQ(id4_pipeline_plan_diagnostic_tap_count(plan),
            IREE_ARRAYSIZE(diagnostic_tap_names));

  const uint32_t token_count = 2;
  const struct {
    // Semantic diagnostic tap requested from the DiT program.
    iree_string_view_t name;
    // Logical tensor shape exposed for PyTorch comparison.
    id4_pipeline_program_shape_t shape;
  } expected_taps[] = {
      {
          /*.name=*/IREE_SV("ideogram4.uncond.layers.0.ffn.input"),
          /*.shape=*/
          id4_pipeline_program_make_shape_rank2(model.hidden_size, token_count),
      },
      {
          /*.name=*/
          IREE_SV("ideogram4.uncond.layers.0.ffn.w1_projection.output"),
          /*.shape=*/
          id4_pipeline_program_make_shape_rank2(model.intermediate_size,
                                                token_count),
      },
      {
          /*.name=*/
          IREE_SV("ideogram4.uncond.layers.0.ffn.w3_projection.output"),
          /*.shape=*/
          id4_pipeline_program_make_shape_rank2(model.intermediate_size,
                                                token_count),
      },
      {
          /*.name=*/IREE_SV("ideogram4.uncond.layers.0.ffn.hidden"),
          /*.shape=*/
          id4_pipeline_program_make_shape_rank2(model.intermediate_size,
                                                token_count),
      },
      {
          /*.name=*/IREE_SV("ideogram4.uncond.layers.0.ffn.output"),
          /*.shape=*/
          id4_pipeline_program_make_shape_rank2(model.hidden_size, token_count),
      },
  };
  for (const auto& expected_tap : expected_taps) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        FindDiagnosticTap(plan, expected_tap.name);
    ASSERT_NE(tap, nullptr);
    EXPECT_EQ(tap->layout.dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
    EXPECT_TRUE(ShapeEquals(tap->layout.shape, expected_tap.shape));
  }

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

TEST(Ideogram4DitStage, PlansOfficialFp8SourcesAsBf16ExecutionParameters) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_ideogram4_dit_model_config_t model = MakeModelConfig();
  model.layer_count = 2;
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3, model, IREE_SV("fp8"),
      iree_allocator_system(), &rules));
  id4_pipeline_stage_t* stage =
      CreateStage(device_group, model, rules.count, rules.values);

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  const iree_device_size_t qkv_size = model.hidden_size * 3;
  const struct {
    // Logical upstream weight key.
    iree_string_view_t key;
    // F32 row-scale key paired with the weight source.
    iree_string_view_t scale_key;
    // Logical output rows in the weight tensor.
    uint64_t output_size;
    // Logical reduction columns in the weight tensor.
    uint64_t input_size;
  } expected_parameters[] = {
      {
          /*.key=*/IREE_SV("layers.0.attention.qkv.weight"),
          /*.scale_key=*/IREE_SV("layers.0.attention.qkv.weight_scale"),
          /*.output_size=*/qkv_size,
          /*.input_size=*/model.hidden_size,
      },
      {
          /*.key=*/IREE_SV("layers.0.attention.o.weight"),
          /*.scale_key=*/IREE_SV("layers.0.attention.o.weight_scale"),
          /*.output_size=*/model.hidden_size,
          /*.input_size=*/model.hidden_size,
      },
      {
          /*.key=*/IREE_SV("layers.1.feed_forward.w1.weight"),
          /*.scale_key=*/IREE_SV("layers.1.feed_forward.w1.weight_scale"),
          /*.output_size=*/model.intermediate_size,
          /*.input_size=*/model.hidden_size,
      },
      {
          /*.key=*/IREE_SV("layers.1.feed_forward.w3.weight"),
          /*.scale_key=*/IREE_SV("layers.1.feed_forward.w3.weight_scale"),
          /*.output_size=*/model.intermediate_size,
          /*.input_size=*/model.hidden_size,
      },
      {
          /*.key=*/IREE_SV("layers.1.feed_forward.w2.weight"),
          /*.scale_key=*/IREE_SV("layers.1.feed_forward.w2.weight_scale"),
          /*.output_size=*/model.hidden_size,
          /*.input_size=*/model.intermediate_size,
      },
      {
          /*.key=*/IREE_SV("input_proj.weight"),
          /*.scale_key=*/IREE_SV("input_proj.weight_scale"),
          /*.output_size=*/model.hidden_size,
          /*.input_size=*/model.input_channel_count,
      },
  };

  const struct {
    // Dynamic request dimensions being planned.
    id4_ideogram4_dit_request_config_t request;
    // Attention schedule used by the parity plan.
    id4_ideogram4_dit_attention_implementation_t attention_implementation;
  } requests[] = {
      {
          /*.request=*/
          {
              /*.latent_shape=*/id4_pipeline_program_make_shape_rank4(1, 2, 4,
                                                                      1),
              /*.conditioning_mode=*/
              ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED,
              /*.text_token_count=*/0,
          },
          /*.attention_implementation=*/
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING,
      },
      {
          /*.request=*/
          {
              /*.latent_shape=*/id4_pipeline_program_make_shape_rank4(3, 5, 4,
                                                                      1),
              /*.conditioning_mode=*/
              ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED,
              /*.text_token_count=*/17,
          },
          /*.attention_implementation=*/
          ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA,
      },
  };

  for (const auto& request_case : requests) {
    id4_ideogram4_dit_stage_plan_options_t dit_options;
    memset(&dit_options, 0, sizeof(dit_options));
    dit_options.structure_size = sizeof(dit_options);
    dit_options.request = request_case.request;
    dit_options.activation_format =
        ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
    dit_options.attention_implementation =
        request_case.attention_implementation;

    id4_pipeline_stage_plan_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.next = &dit_options;
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.diagnostics_sink = &diagnostics_sink;

    id4_pipeline_plan_t* plan = nullptr;
    IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));
    for (const auto& expected_parameter : expected_parameters) {
      ExpectScaledFp8SourceToBf16ExecutionParameter(
          plan, IREE_SV("fp8"), expected_parameter.key,
          expected_parameter.scale_key, expected_parameter.output_size,
          expected_parameter.input_size);
    }
    id4_pipeline_plan_release(plan);
  }

  id4_pipeline_stage_release(stage);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
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
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  dit_options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;

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
