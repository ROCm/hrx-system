// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl_condition.h"

#include <cstring>
#include <string>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static id4_pipeline_stage_t* CreatePlanningQwen3VlConditionStage(
    iree_hal_device_group_t* device_group) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_condition_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.source_identifier = IREE_SV("qwen3_vl_condition_f32.loom");
  options.module_name = IREE_SV("id4_qwen3_vl_condition_f32");
  options.executable_identifier = IREE_SV("id4_qwen3_vl_condition_f32.hsaco");
  options.apply_token_weights_function_name =
      IREE_SV("id4_qwen3_vl_condition_apply_token_weights_f32");
  options.normalize_token_weights_function_name =
      IREE_SV("id4_qwen3_vl_condition_normalize_token_weights_f32");
  options.hidden_row_count = 32;
  options.token_count = 64;
  options.workgroup_size_x = 256;

  id4_pipeline_stage_t* stage = nullptr;
  IREE_CHECK_OK(id4_qwen3_vl_condition_stage_create(
      &options, iree_allocator_system(), &stage));
  return stage;
}

TEST(Qwen3VlConditionStage, PlansConditionForwardRegion) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage =
      CreatePlanningQwen3VlConditionStage(device_group);

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage, &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;
  id4_pipeline_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_pipeline_stage_plan(stage, &plan_options, &plan));

  EXPECT_EQ(id4::test::ToString(id4_pipeline_plan_stage_name(plan)),
            "qwen3_vl.condition");
  EXPECT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 0u);
  EXPECT_EQ(id4_pipeline_plan_memory_slab_count(plan), 4u);

  const id4_pipeline_memory_slab_plan_t* selected_slab =
      id4_pipeline_plan_memory_slab_at(plan, 0);
  ASSERT_NE(selected_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(selected_slab->name),
            "qwen3_vl.condition.selected_hidden_states");
  EXPECT_EQ(selected_slab->binding_slot, 0u);
  EXPECT_EQ(selected_slab->byte_length, 8192u);
  EXPECT_EQ(selected_slab->high_water_mark, 8192u);

  const id4_pipeline_memory_slab_plan_t* token_weights_slab =
      id4_pipeline_plan_memory_slab_at(plan, 1);
  ASSERT_NE(token_weights_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(token_weights_slab->name),
            "qwen3_vl.condition.token_weights");
  EXPECT_EQ(token_weights_slab->binding_slot, 1u);
  EXPECT_EQ(token_weights_slab->byte_length, 256u);
  EXPECT_EQ(token_weights_slab->high_water_mark, 256u);

  const id4_pipeline_memory_slab_plan_t* condition_slab =
      id4_pipeline_plan_memory_slab_at(plan, 2);
  ASSERT_NE(condition_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(condition_slab->name),
            "qwen3_vl.condition.output");
  EXPECT_EQ(condition_slab->binding_slot, 2u);
  EXPECT_EQ(condition_slab->byte_length, 8192u);
  EXPECT_EQ(condition_slab->high_water_mark, 8192u);

  const id4_pipeline_memory_slab_plan_t* local_slab =
      id4_pipeline_plan_memory_slab_at(plan, 3);
  ASSERT_NE(local_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(local_slab->name), "qwen3_vl.condition.local");
  EXPECT_EQ(local_slab->binding_slot, 3u);
  EXPECT_EQ(local_slab->byte_length, 8u);
  EXPECT_EQ(local_slab->high_water_mark, 8u);

  EXPECT_EQ(id4_pipeline_plan_kernel_count(plan), 2u);
  const id4_pipeline_kernel_plan_t* apply_kernel =
      id4_pipeline_plan_kernel_at(plan, 0);
  ASSERT_NE(apply_kernel, nullptr);
  EXPECT_EQ(id4::test::ToString(apply_kernel->specialization_key),
            "id4_qwen3_vl_condition_f32:hidden_row_count=32:token_count=64:"
            "workgroup_size_x=256");
  EXPECT_EQ(id4::test::ToString(apply_kernel->source_identifier),
            "qwen3_vl_condition_f32.loom");
  EXPECT_EQ(id4::test::ToString(apply_kernel->module_name),
            "id4_qwen3_vl_condition_f32");
  EXPECT_EQ(id4::test::ToString(apply_kernel->executable_identifier),
            "id4_qwen3_vl_condition_f32.hsaco");
  EXPECT_EQ(id4::test::ToString(apply_kernel->function_name),
            "id4_qwen3_vl_condition_apply_token_weights_f32");
  EXPECT_EQ(apply_kernel->placement_id, 0u);
  EXPECT_EQ(apply_kernel->config_binding_count, 3u);
  ASSERT_NE(apply_kernel->config_bindings, nullptr);
  EXPECT_EQ(id4::test::ToString(apply_kernel->config_bindings[0].key),
            "id4.qwen3_vl.condition.hidden_row_count");
  EXPECT_EQ(id4::test::ToString(apply_kernel->config_bindings[0].value), "32");
  EXPECT_EQ(id4::test::ToString(apply_kernel->config_bindings[1].key),
            "id4.qwen3_vl.condition.token_count");
  EXPECT_EQ(id4::test::ToString(apply_kernel->config_bindings[1].value), "64");
  EXPECT_EQ(id4::test::ToString(apply_kernel->config_bindings[2].key),
            "id4.qwen3_vl.condition.workgroup_size_x");
  EXPECT_EQ(id4::test::ToString(apply_kernel->config_bindings[2].value), "256");

  const id4_pipeline_kernel_plan_t* normalize_kernel =
      id4_pipeline_plan_kernel_at(plan, 1);
  ASSERT_NE(normalize_kernel, nullptr);
  EXPECT_EQ(id4::test::ToString(normalize_kernel->specialization_key),
            "id4_qwen3_vl_condition_f32:hidden_row_count=32:token_count=64:"
            "workgroup_size_x=256");
  EXPECT_EQ(id4::test::ToString(normalize_kernel->function_name),
            "id4_qwen3_vl_condition_normalize_token_weights_f32");
  EXPECT_EQ(normalize_kernel->placement_id, 0u);
  EXPECT_EQ(normalize_kernel->config_binding_count, 3u);

  EXPECT_EQ(id4_pipeline_plan_region_count(plan), 1u);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(region, nullptr);
  EXPECT_EQ(id4::test::ToString(region->name), "qwen3_vl.condition.forward");
  EXPECT_EQ(region->binding_capacity, 4u);
  EXPECT_EQ(region->local_binding_slot, 3u);
  EXPECT_EQ(region->statistics.operation_count, 3u);
  EXPECT_EQ(region->statistics.dispatch_count, 2u);
  EXPECT_EQ(region->statistics.barrier_count, 1u);
  EXPECT_EQ(region->statistics.current_epoch, 1u);
  EXPECT_EQ(region->statistics.local_acquire_count, 1u);
  EXPECT_EQ(region->statistics.local_release_count, 1u);
  EXPECT_EQ(region->statistics.bound_import_count, 3u);
  EXPECT_EQ(region->statistics.local_slab_byte_length, 8u);
  EXPECT_EQ(region->statistics.local_slab_high_water_mark, 8u);

  EXPECT_EQ(id4_pipeline_plan_diagnostic_tap_count(plan), 3u);
  const id4_pipeline_diagnostic_tap_plan_t* selected_tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, 0);
  ASSERT_NE(selected_tap, nullptr);
  EXPECT_EQ(id4::test::ToString(selected_tap->name),
            "qwen3_vl.condition.selected_hidden_states.before");
  EXPECT_EQ(id4::test::ToString(selected_tap->target_name),
            "qwen3_vl.encoder.selected_hidden_states");
  const id4_pipeline_diagnostic_tap_plan_t* token_weights_tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, 1);
  ASSERT_NE(token_weights_tap, nullptr);
  EXPECT_EQ(id4::test::ToString(token_weights_tap->name),
            "qwen3_vl.condition.token_weights.before");
  EXPECT_EQ(id4::test::ToString(token_weights_tap->target_name),
            "qwen3_vl.prompt.token_weights");
  const id4_pipeline_diagnostic_tap_plan_t* condition_tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, 2);
  ASSERT_NE(condition_tap, nullptr);
  EXPECT_EQ(id4::test::ToString(condition_tap->name),
            "qwen3_vl.condition.output.after");
  EXPECT_EQ(condition_tap->after_operation_ordinal, 2u);
  EXPECT_EQ(id4::test::ToString(condition_tap->target_name),
            "qwen3_vl.encoder.condition");

  iree_string_builder_t plan_json_builder;
  iree_string_builder_initialize(iree_allocator_system(), &plan_json_builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &plan_json_builder));
  const std::string plan_json =
      id4::test::ToString(iree_string_builder_view(&plan_json_builder));
  EXPECT_NE(plan_json.find("\"qwen3_vl.condition\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"memory_slabs\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"kernels\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"regions\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"diagnostic_taps\""), std::string::npos);
  iree_string_builder_deinitialize(&plan_json_builder);

  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "stage.load"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "plan.create"));
  EXPECT_EQ(diagnostics.kernel_event_count, 0u);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

}  // namespace
