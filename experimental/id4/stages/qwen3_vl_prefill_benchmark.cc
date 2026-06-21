// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/qwen3_vl_prefill.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gbenchmark_harness.h"

namespace {

typedef struct Qwen3VlPrefillBenchmarkContext {
  // Device group retained by the loaded Qwen3-VL prefill stage.
  iree_hal_device_group_t* device_group;
  // Loaded Qwen3-VL prefill planning stage.
  id4_pipeline_stage_t* stage;
  // Diagnostics sink used by benchmark lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
} Qwen3VlPrefillBenchmarkContext;

static id4_pipeline_stage_t* CreatePlanningQwen3VlPrefillStage(
    iree_hal_device_group_t* device_group) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_prefill_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.token_count = 64;

  id4_pipeline_stage_t* stage = nullptr;
  IREE_CHECK_OK(id4_qwen3_vl_prefill_stage_create(
      &options, iree_allocator_system(), &stage));
  return stage;
}

static Qwen3VlPrefillBenchmarkContext CreateLoadedQwen3VlPrefillStage() {
  Qwen3VlPrefillBenchmarkContext context;
  std::memset(&context, 0, sizeof(context));
  context.device_group = id4::test::CreateLocalSyncDeviceGroup();
  context.stage = CreatePlanningQwen3VlPrefillStage(context.device_group);
  id4_pipeline_diagnostics_sink_initialize_ignore(&context.diagnostics_sink);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &context.diagnostics_sink;
  IREE_CHECK_OK(id4_pipeline_stage_load(context.stage, &load_options));
  return context;
}

static void DestroyLoadedQwen3VlPrefillStage(
    Qwen3VlPrefillBenchmarkContext* context) {
  id4_pipeline_stage_release(context->stage);
  iree_hal_device_group_release(context->device_group);
  std::memset(context, 0, sizeof(*context));
}

static id4_pipeline_plan_t* CreateQwen3VlPrefillPlan(
    Qwen3VlPrefillBenchmarkContext* context) {
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

static void BM_Qwen3VlPrefillStagePlan(benchmark::State& state) {
  Qwen3VlPrefillBenchmarkContext context = CreateLoadedQwen3VlPrefillStage();
  for (auto _ : state) {
    id4_pipeline_plan_t* plan = CreateQwen3VlPrefillPlan(&context);
    benchmark::DoNotOptimize(plan);
    id4_pipeline_plan_release(plan);
  }
  DestroyLoadedQwen3VlPrefillStage(&context);
}
BENCHMARK(BM_Qwen3VlPrefillStagePlan);

}  // namespace
