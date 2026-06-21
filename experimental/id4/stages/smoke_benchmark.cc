// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/smoke.h"
#include "experimental/id4/stages/smoke_test_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gbenchmark_harness.h"

namespace {

typedef struct SmokeBenchmarkContext {
  // Device group retained by the loaded smoke stage.
  iree_hal_device_group_t* device_group;
  // Fake HAL executable cache used after Loom emits HSACO.
  id4::test::SmokeExecutableCache* executable_cache;
  // Shared Loom kernel cache.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Loaded smoke stage.
  id4_pipeline_stage_t* stage;
  // Diagnostics sink used by benchmark lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
} SmokeBenchmarkContext;

static SmokeBenchmarkContext CreateLoadedSmokeStage() {
  SmokeBenchmarkContext context;
  std::memset(&context, 0, sizeof(context));
  context.device_group = id4::test::CreateLocalSyncDeviceGroup();
  IREE_CHECK_OK(id4::test::CreateExecutableCache(iree_allocator_system(),
                                                 &context.executable_cache));
  IREE_CHECK_OK(id4::test::CreateKernelCache(iree_allocator_system(),
                                             &context.kernel_cache));
  IREE_CHECK_OK(id4::test::CreateSmokeStage(
      context.device_group, context.executable_cache, context.kernel_cache,
      iree_allocator_system(), &context.stage));
  id4_pipeline_diagnostics_sink_initialize_ignore(&context.diagnostics_sink);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &context.diagnostics_sink;
  IREE_CHECK_OK(id4_pipeline_stage_load(context.stage, &load_options));
  return context;
}

static void DestroyLoadedSmokeStage(SmokeBenchmarkContext* context) {
  id4_pipeline_stage_release(context->stage);
  id4_pipeline_kernel_cache_release(context->kernel_cache);
  iree_hal_executable_cache_release(
      reinterpret_cast<iree_hal_executable_cache_t*>(
          context->executable_cache));
  iree_hal_device_group_release(context->device_group);
  std::memset(context, 0, sizeof(*context));
}

static id4_pipeline_plan_t* CreateSmokePlan(SmokeBenchmarkContext* context) {
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

static id4_pipeline_bundle_t* PrepareSmokeBundle(
    SmokeBenchmarkContext* context, const id4_pipeline_plan_t* plan,
    iree_io_parameter_provider_t* provider,
    iree_hal_semaphore_t* ready_semaphore, uint64_t ready_value) {
  iree_hal_semaphore_list_t prepare_signal_list = {
      // Number of semaphores signaled by prepare.
      /*.count=*/1,
      // Readiness semaphore for the prepared bundle.
      /*.semaphores=*/&ready_semaphore,
      // Readiness payload value for the prepared bundle.
      /*.payload_values=*/&ready_value,
  };
  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = provider;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal_list;
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  id4_pipeline_bundle_t* bundle = nullptr;
  IREE_CHECK_OK(id4_pipeline_stage_prepare(context->stage, plan,
                                           &prepare_options, &bundle));
  IREE_CHECK_OK(iree_hal_semaphore_wait(ready_semaphore, ready_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE));
  return bundle;
}

static void IssueSmokeBundle(SmokeBenchmarkContext* context,
                             id4_pipeline_bundle_t* bundle,
                             iree_hal_semaphore_t* issue_semaphore,
                             uint64_t issue_value) {
  iree_hal_semaphore_list_t issue_signal_list = {
      // Number of semaphores signaled by issue.
      /*.count=*/1,
      // Completion semaphore for this issue.
      /*.semaphores=*/&issue_semaphore,
      // Completion payload value for this issue.
      /*.payload_values=*/&issue_value,
  };
  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  issue_options.signal_semaphore_list = issue_signal_list;
  issue_options.diagnostics_sink = &context->diagnostics_sink;
  IREE_CHECK_OK(
      id4_pipeline_stage_issue(context->stage, bundle, &issue_options));
  IREE_CHECK_OK(iree_hal_semaphore_wait(issue_semaphore, issue_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE));
}

static void BM_SmokeStagePlan(benchmark::State& state) {
  SmokeBenchmarkContext context = CreateLoadedSmokeStage();
  for (auto _ : state) {
    id4_pipeline_plan_t* plan = CreateSmokePlan(&context);
    benchmark::DoNotOptimize(plan);
    id4_pipeline_plan_release(plan);
  }
  DestroyLoadedSmokeStage(&context);
}
BENCHMARK(BM_SmokeStagePlan);

static void BM_SmokeStagePrepareEndToEnd(benchmark::State& state) {
  SmokeBenchmarkContext context = CreateLoadedSmokeStage();
  id4_pipeline_plan_t* plan = CreateSmokePlan(&context);
  iree_io_parameter_provider_t* provider =
      id4::test::CreateSmokeParameterProvider();
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* ready_semaphore = id4::test::CreateSemaphore(device);
  uint64_t ready_value = 0;
  for (auto _ : state) {
    id4_pipeline_bundle_t* bundle = PrepareSmokeBundle(
        &context, plan, provider, ready_semaphore, ++ready_value);
    benchmark::DoNotOptimize(bundle);
    id4_pipeline_bundle_release(bundle);
  }
  iree_hal_semaphore_release(ready_semaphore);
  iree_io_parameter_provider_release(provider);
  id4_pipeline_plan_release(plan);
  DestroyLoadedSmokeStage(&context);
}
BENCHMARK(BM_SmokeStagePrepareEndToEnd);

static void BM_SmokeStageIssueEndToEnd(benchmark::State& state) {
  SmokeBenchmarkContext context = CreateLoadedSmokeStage();
  id4_pipeline_plan_t* plan = CreateSmokePlan(&context);
  iree_io_parameter_provider_t* provider =
      id4::test::CreateSmokeParameterProvider();
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* ready_semaphore = id4::test::CreateSemaphore(device);
  id4_pipeline_bundle_t* bundle =
      PrepareSmokeBundle(&context, plan, provider, ready_semaphore, 1);

  iree_hal_semaphore_t* issue_semaphore = id4::test::CreateSemaphore(device);
  uint64_t issue_value = 0;
  for (auto _ : state) {
    IssueSmokeBundle(&context, bundle, issue_semaphore, ++issue_value);
    benchmark::DoNotOptimize(bundle);
  }

  iree_hal_semaphore_release(issue_semaphore);
  id4_pipeline_bundle_release(bundle);
  iree_hal_semaphore_release(ready_semaphore);
  iree_io_parameter_provider_release(provider);
  id4_pipeline_plan_release(plan);
  DestroyLoadedSmokeStage(&context);
}
BENCHMARK(BM_SmokeStageIssueEndToEnd);

}  // namespace
