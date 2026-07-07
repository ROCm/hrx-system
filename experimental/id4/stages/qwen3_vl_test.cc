// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl.h"

#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr uint32_t kSelectedLayerOrdinals[] = {0, 1};
static constexpr int32_t kSmallTokenIds[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

static id4_qwen3_vl_model_config_t MakeModelConfig(uint32_t layer_count) {
  return id4_qwen3_vl_model_config_t{
      // Number of decoder layers.
      /*.layer_count=*/layer_count,
      // Vocabulary row count.
      /*.vocab_size=*/32,
      // Hidden-state channel count.
      /*.hidden_size=*/32,
      // MLP intermediate channel count.
      /*.intermediate_size=*/64,
      // Number of query attention heads.
      /*.attention_head_count=*/2,
      // Number of key/value attention heads.
      /*.key_value_head_count=*/2,
      // Channel count per attention head.
      /*.head_size=*/16,
      // Maximum prompt token positions accepted by the model runner.
      /*.max_token_count=*/128,
      // Number of selected layer outputs.
      /*.selected_layer_count=*/IREE_ARRAYSIZE(kSelectedLayerOrdinals),
      // Selected layer output ordinals.
      /*.selected_layer_ordinals=*/kSelectedLayerOrdinals,
  };
}

static id4_pipeline_stage_t* CreateStage(
    iree_hal_device_group_t* device_group,
    const id4_qwen3_vl_model_config_t* model,
    id4_qwen3_vl_parameter_format_t parameter_format =
        ID4_QWEN3_VL_PARAMETER_FORMAT_BF16) {
  id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
  memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
  kernel_cache_options.structure_size = sizeof(kernel_cache_options);
  kernel_cache_options.target_processor =
      id4_pipeline_kernel_cache_default_target_processor();
  kernel_cache_options.entry_limit =
      ID4_PIPELINE_KERNEL_CACHE_INTERACTIVE_ENTRY_LIMIT;
  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_CHECK_OK(id4_pipeline_kernel_cache_create(
      &kernel_cache_options, iree_allocator_system(), &kernel_cache));

  id4_pipeline_stage_services_t services;
  memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_stage_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = kernel_cache;
  create_options.parameter_format = parameter_format;
  create_options.model = *model;

  id4_pipeline_stage_t* stage = nullptr;
  IREE_CHECK_OK(id4_qwen3_vl_stage_create(&create_options,
                                          iree_allocator_system(), &stage));
  id4_pipeline_kernel_cache_release(kernel_cache);
  return stage;
}

static id4_pipeline_stage_t* CreateSmallStage(
    iree_hal_device_group_t* device_group) {
  id4_qwen3_vl_model_config_t model = MakeModelConfig(/*layer_count=*/2);
  return CreateStage(device_group, &model);
}

TEST(Qwen3VlStage, PlansForwardStageFromRequestConfig) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = CreateSmallStage(device_group);

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t missing_qwen_options;
  memset(&missing_qwen_options, 0, sizeof(missing_qwen_options));
  missing_qwen_options.structure_size = sizeof(missing_qwen_options);
  missing_qwen_options.device_index = 0;
  missing_qwen_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  missing_qwen_options.diagnostics_sink = &diagnostics_sink;

  id4_pipeline_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_pipeline_stage_plan(stage, &missing_qwen_options, &plan));
  EXPECT_EQ(plan, nullptr);

  id4_qwen3_vl_stage_plan_options_t qwen_options;
  memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = 19;
  qwen_options.request.token_ids = kSmallTokenIds;
  qwen_options.weight_execution_strategy =
      ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR;
  qwen_options.attention_implementation =
      ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &qwen_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  EXPECT_EQ(id4::test::ToString(id4_pipeline_plan_stage_name(plan)),
            ID4_QWEN3_VL_STAGE_NAME);
  EXPECT_GT(id4_pipeline_plan_parameter_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_memory_slab_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_boundary_tensor_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_kernel_count(plan), 0u);
  EXPECT_GT(id4_pipeline_plan_region_count(plan), 1u);
  EXPECT_GT(id4_pipeline_plan_shared_tensor_count(plan), 0u);
  bool has_local_transient_region = false;
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_region_count(plan); ++i) {
    const id4_pipeline_region_plan_t* region =
        id4_pipeline_plan_region_at(plan, i);
    ASSERT_NE(region, nullptr);
    has_local_transient_region =
        has_local_transient_region ||
        (region->statistics.local_acquire_count > 0 &&
         region->statistics.local_slab_byte_length > 0);
  }
  EXPECT_TRUE(has_local_transient_region);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

TEST(Qwen3VlStage, PlansIdeogram4ForwardBoundaryContract) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  const id4_qwen3_vl_model_config_t* model =
      id4_qwen3_vl_program_ideogram4_model_config();
  id4_pipeline_stage_t* stage = CreateStage(device_group, model);

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  constexpr uint32_t kTokenCounts[] = {64, 512};
  for (uint32_t token_count : kTokenCounts) {
    std::vector<int32_t> token_ids(token_count, 0);
    id4_qwen3_vl_stage_plan_options_t qwen_options;
    memset(&qwen_options, 0, sizeof(qwen_options));
    qwen_options.structure_size = sizeof(qwen_options);
    qwen_options.request.token_count = token_count;
    qwen_options.request.token_ids = token_ids.data();
    qwen_options.weight_execution_strategy =
        ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR;
    qwen_options.attention_implementation =
        ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO;

    id4_pipeline_stage_plan_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.next = &qwen_options;
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.diagnostics_sink = &diagnostics_sink;

    id4_pipeline_plan_t* plan = nullptr;
    IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

    const id4_pipeline_boundary_tensor_plan_t* exported_boundary = nullptr;
    for (iree_host_size_t i = 0;
         i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
      const id4_pipeline_boundary_tensor_plan_t* boundary =
          id4_pipeline_plan_boundary_tensor_at(plan, i);
      if (boundary &&
          iree_all_bits_set(boundary->flags,
                            ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
        exported_boundary = boundary;
        break;
      }
    }
    ASSERT_NE(exported_boundary, nullptr);
    EXPECT_EQ(exported_boundary->layout.shape.rank, 2u);
    EXPECT_EQ(exported_boundary->layout.shape.dims[0],
              static_cast<uint64_t>(model->selected_layer_count) *
                  model->hidden_size);
    EXPECT_EQ(exported_boundary->layout.shape.dims[1], token_count);

    id4_pipeline_plan_release(plan);
  }
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

TEST(Qwen3VlStage, PlansFp8SourceWithLowerParameterPressure) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  const id4_qwen3_vl_model_config_t* model =
      id4_qwen3_vl_program_ideogram4_model_config();
  id4_pipeline_stage_t* bf16_stage =
      CreateStage(device_group, model, ID4_QWEN3_VL_PARAMETER_FORMAT_BF16);
  id4_pipeline_stage_t* fp8_stage = CreateStage(
      device_group, model, ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED);

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(bf16_stage, &load_options));
  IREE_ASSERT_OK(id4_pipeline_stage_load(fp8_stage, &load_options));

  std::vector<int32_t> token_ids(512, 0);
  auto plan_stage = [&](id4_pipeline_stage_t* stage,
                        id4_pipeline_plan_t** out_plan) {
    id4_qwen3_vl_stage_plan_options_t qwen_options;
    memset(&qwen_options, 0, sizeof(qwen_options));
    qwen_options.structure_size = sizeof(qwen_options);
    qwen_options.request.token_count = static_cast<uint32_t>(token_ids.size());
    qwen_options.request.token_ids = token_ids.data();
    qwen_options.weight_execution_strategy =
        ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS;
    qwen_options.attention_implementation =
        ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO;

    id4_pipeline_stage_plan_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.next = &qwen_options;
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.diagnostics_sink = &diagnostics_sink;
    return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
  };

  id4_pipeline_plan_t* bf16_plan = nullptr;
  IREE_ASSERT_OK(plan_stage(bf16_stage, &bf16_plan));
  id4_pipeline_plan_t* fp8_plan = nullptr;
  IREE_ASSERT_OK(plan_stage(fp8_stage, &fp8_plan));

  const id4_pipeline_plan_statistics_t bf16_statistics =
      id4_pipeline_plan_statistics(bf16_plan);
  const id4_pipeline_plan_statistics_t fp8_statistics =
      id4_pipeline_plan_statistics(fp8_plan);
  EXPECT_LT(fp8_statistics.parameter_slab_byte_length,
            bf16_statistics.parameter_slab_byte_length);
  EXPECT_LT(fp8_statistics.parameter_source_byte_length,
            bf16_statistics.parameter_source_byte_length);
  EXPECT_GT(fp8_statistics.parameter_direct_source_byte_length, 0u);
  EXPECT_GT(fp8_statistics.parameter_encoded_source_byte_length, 0u);

  id4_pipeline_plan_release(fp8_plan);
  id4_pipeline_plan_release(bf16_plan);
  id4_pipeline_stage_release(fp8_stage);
  id4_pipeline_stage_release(bf16_stage);
  iree_hal_device_group_release(device_group);
}

TEST(Qwen3VlStage, RejectsInvalidStaticModelConfig) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();

  id4_pipeline_stage_services_t services;
  memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
  memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
  kernel_cache_options.structure_size = sizeof(kernel_cache_options);
  kernel_cache_options.target_processor =
      id4_pipeline_kernel_cache_default_target_processor();
  kernel_cache_options.entry_limit =
      ID4_PIPELINE_KERNEL_CACHE_INTERACTIVE_ENTRY_LIMIT;
  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
      &kernel_cache_options, iree_allocator_system(), &kernel_cache));

  id4_qwen3_vl_stage_create_options_t create_options;
  memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = kernel_cache;
  create_options.parameter_format = ID4_QWEN3_VL_PARAMETER_FORMAT_BF16;
  create_options.model = MakeModelConfig(/*layer_count=*/0);

  id4_pipeline_stage_t* stage = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_qwen3_vl_stage_create(
                            &create_options, iree_allocator_system(), &stage));
  EXPECT_EQ(stage, nullptr);

  id4_pipeline_kernel_cache_release(kernel_cache);
  iree_hal_device_group_release(device_group);
}

}  // namespace
