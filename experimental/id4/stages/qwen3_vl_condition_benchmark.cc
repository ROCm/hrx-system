// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/qwen3_vl_condition.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gbenchmark_harness.h"

namespace {

typedef struct Qwen3VlConditionBenchmarkContext {
  // Device group retained by the loaded Qwen3-VL condition stage.
  iree_hal_device_group_t* device_group;
  // Loaded Qwen3-VL condition planning stage.
  id4_pipeline_stage_t* stage;
  // Diagnostics sink used by benchmark lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
} Qwen3VlConditionBenchmarkContext;

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

static Qwen3VlConditionBenchmarkContext CreateLoadedQwen3VlConditionStage() {
  Qwen3VlConditionBenchmarkContext context;
  std::memset(&context, 0, sizeof(context));
  context.device_group = id4::test::CreateLocalSyncDeviceGroup();
  context.stage = CreatePlanningQwen3VlConditionStage(context.device_group);
  id4_pipeline_diagnostics_sink_initialize_ignore(&context.diagnostics_sink);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &context.diagnostics_sink;
  IREE_CHECK_OK(id4_pipeline_stage_load(context.stage, &load_options));
  return context;
}

static void DestroyLoadedQwen3VlConditionStage(
    Qwen3VlConditionBenchmarkContext* context) {
  id4_pipeline_stage_release(context->stage);
  iree_hal_device_group_release(context->device_group);
  std::memset(context, 0, sizeof(*context));
}

static id4_pipeline_plan_t* CreateQwen3VlConditionPlan(
    Qwen3VlConditionBenchmarkContext* context) {
  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &context->diagnostics_sink;
  id4_pipeline_plan_t* plan = nullptr;
  IREE_CHECK_OK(id4_pipeline_stage_plan(context->stage, &plan_options, &plan));
  return plan;
}

static void BM_Qwen3VlConditionStagePlan(benchmark::State& state) {
  Qwen3VlConditionBenchmarkContext context =
      CreateLoadedQwen3VlConditionStage();
  for (auto _ : state) {
    id4_pipeline_plan_t* plan = CreateQwen3VlConditionPlan(&context);
    benchmark::DoNotOptimize(plan);
    id4_pipeline_plan_release(plan);
  }
  DestroyLoadedQwen3VlConditionStage(&context);
}
BENCHMARK(BM_Qwen3VlConditionStagePlan);

}  // namespace
