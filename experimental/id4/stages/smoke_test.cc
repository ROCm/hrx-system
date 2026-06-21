// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/smoke.h"

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

static id4_pipeline_stage_t* CreatePlanningSmokeStage(
    iree_hal_device_group_t* device_group) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_smoke_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.source_identifier = IREE_SV("smoke_configured.loom");
  options.module_name = IREE_SV("id4_smoke_configured");
  options.executable_identifier = IREE_SV("id4_smoke_configured.hsaco");
  options.function_name = IREE_SV("id4_smoke_configured");
  options.workgroups_x = 1;
  options.workgroup_size_x = 64;

  id4_pipeline_stage_t* stage = nullptr;
  IREE_CHECK_OK(
      id4_smoke_stage_create(&options, iree_allocator_system(), &stage));
  return stage;
}

TEST(SmokeStage, PlansConfiguredKernelRegion) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  id4_pipeline_stage_t* stage = CreatePlanningSmokeStage(device_group);

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

  EXPECT_EQ(id4::test::ToString(id4_pipeline_plan_stage_name(plan)), "smoke");
  EXPECT_EQ(id4_pipeline_plan_parameter_slab_count(plan), 1u);
  const id4_pipeline_parameter_slab_plan_t* parameter_slab =
      id4_pipeline_plan_parameter_slab_at(plan, 0);
  ASSERT_NE(parameter_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(parameter_slab->scope), "smoke");
  EXPECT_EQ(parameter_slab->placement_id, 0u);
  EXPECT_EQ(parameter_slab->byte_length, 16u);
  EXPECT_EQ(parameter_slab->alignment, 16u);
  EXPECT_EQ(parameter_slab->request_count, 1u);
  ASSERT_NE(parameter_slab->requests, nullptr);
  EXPECT_EQ(id4::test::ToString(parameter_slab->requests[0].key),
            "smoke.weight");
  EXPECT_EQ(parameter_slab->requests[0].span.length, 16u);

  EXPECT_EQ(id4_pipeline_plan_memory_slab_count(plan), 1u);
  const id4_pipeline_memory_slab_plan_t* memory_slab =
      id4_pipeline_plan_memory_slab_at(plan, 0);
  ASSERT_NE(memory_slab, nullptr);
  EXPECT_EQ(id4::test::ToString(memory_slab->name), "smoke.local");
  EXPECT_EQ(memory_slab->placement_id, 0u);
  EXPECT_EQ(memory_slab->binding_slot, 1u);
  EXPECT_EQ(memory_slab->byte_length, 16u);
  EXPECT_EQ(memory_slab->alignment, 16u);
  EXPECT_EQ(memory_slab->high_water_mark, 16u);

  EXPECT_EQ(id4_pipeline_plan_kernel_count(plan), 1u);
  const id4_pipeline_kernel_plan_t* kernel =
      id4_pipeline_plan_kernel_at(plan, 0);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(id4::test::ToString(kernel->specialization_key),
            "id4_smoke_configured:workgroups_x=1:workgroup_size_x=64");
  EXPECT_EQ(id4::test::ToString(kernel->source_identifier),
            "smoke_configured.loom");
  EXPECT_EQ(id4::test::ToString(kernel->module_name), "id4_smoke_configured");
  EXPECT_EQ(id4::test::ToString(kernel->executable_identifier),
            "id4_smoke_configured.hsaco");
  EXPECT_EQ(id4::test::ToString(kernel->function_name), "id4_smoke_configured");
  EXPECT_EQ(kernel->placement_id, 0u);
  EXPECT_EQ(kernel->config_binding_count, 2u);
  ASSERT_NE(kernel->config_bindings, nullptr);
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[0].key),
            "id4.smoke.workgroups_x");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[0].value), "1");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[1].key),
            "id4.smoke.workgroup_size_x");
  EXPECT_EQ(id4::test::ToString(kernel->config_bindings[1].value), "64");

  EXPECT_EQ(id4_pipeline_plan_region_count(plan), 1u);
  const id4_pipeline_region_plan_t* region =
      id4_pipeline_plan_region_at(plan, 0);
  ASSERT_NE(region, nullptr);
  EXPECT_EQ(id4::test::ToString(region->name), "smoke.region");
  EXPECT_EQ(region->binding_capacity, 3u);
  EXPECT_EQ(region->local_binding_slot, 1u);
  EXPECT_EQ(region->statistics.operation_count, 1u);
  EXPECT_EQ(region->statistics.dispatch_count, 1u);
  EXPECT_EQ(region->statistics.local_acquire_count, 1u);
  EXPECT_EQ(region->statistics.bound_import_count, 1u);

  EXPECT_EQ(id4_pipeline_plan_diagnostic_tap_count(plan), 1u);
  const id4_pipeline_diagnostic_tap_plan_t* tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, 0);
  ASSERT_NE(tap, nullptr);
  EXPECT_EQ(id4::test::ToString(tap->name), "smoke.output.after_dispatch");
  EXPECT_EQ(tap->region_id, 0u);
  EXPECT_EQ(tap->after_operation_ordinal, 0u);
  EXPECT_EQ(id4::test::ToString(tap->target_name), "smoke.output");

  iree_string_builder_t plan_json_builder;
  iree_string_builder_initialize(iree_allocator_system(), &plan_json_builder);
  IREE_ASSERT_OK(id4_pipeline_plan_format_json(plan, &plan_json_builder));
  const std::string plan_json =
      id4::test::ToString(iree_string_builder_view(&plan_json_builder));
  EXPECT_NE(plan_json.find("\"parameter_slabs\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"memory_slabs\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"kernels\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"regions\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"diagnostic_taps\""), std::string::npos);
  EXPECT_NE(plan_json.find("\"smoke.output.after_dispatch\""),
            std::string::npos);
  iree_string_builder_deinitialize(&plan_json_builder);

  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "stage.load"));
  EXPECT_TRUE(id4::test::ContainsKey(diagnostics.keys, "plan.create"));
  EXPECT_EQ(diagnostics.kernel_event_count, 0u);

  id4_pipeline_plan_release(plan);
  id4_pipeline_stage_release(stage);
  iree_hal_device_group_release(device_group);
}

}  // namespace
