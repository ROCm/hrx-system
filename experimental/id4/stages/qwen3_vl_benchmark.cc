// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/stages/qwen3_vl_test_util.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gbenchmark_harness.h"

namespace {

typedef struct Qwen3VlBenchmarkContext {
  // Device group retained by the loaded Qwen3-VL stage.
  iree_hal_device_group_t* device_group;
  // Fake HAL executable cache used after Loom emits HSACO.
  id4::test::Qwen3VlExecutableCache* executable_cache;
  // Shared Loom kernel cache.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Loaded Qwen3-VL stage.
  id4_pipeline_stage_t* stage;
  // Diagnostics sink used by benchmark lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
} Qwen3VlBenchmarkContext;

static Qwen3VlBenchmarkContext CreateLoadedQwen3VlStage() {
  Qwen3VlBenchmarkContext context;
  std::memset(&context, 0, sizeof(context));
  context.device_group = id4::test::CreateLocalSyncDeviceGroup();
  IREE_CHECK_OK(id4::test::CreateQwen3VlExecutableCache(
      iree_allocator_system(), &context.executable_cache));
  IREE_CHECK_OK(id4::test::CreateKernelCache(iree_allocator_system(),
                                             &context.kernel_cache));
  IREE_CHECK_OK(id4::test::CreateQwen3VlStage(
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

static void DestroyLoadedQwen3VlStage(Qwen3VlBenchmarkContext* context) {
  id4_pipeline_stage_release(context->stage);
  id4_pipeline_kernel_cache_release(context->kernel_cache);
  iree_hal_executable_cache_release(
      reinterpret_cast<iree_hal_executable_cache_t*>(
          context->executable_cache));
  iree_hal_device_group_release(context->device_group);
  std::memset(context, 0, sizeof(*context));
}

static id4_pipeline_plan_t* CreateQwen3VlPlan(
    Qwen3VlBenchmarkContext* context) {
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

static id4_pipeline_bundle_t* PrepareQwen3VlBundle(
    Qwen3VlBenchmarkContext* context, const id4_pipeline_plan_t* plan) {
  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  id4_pipeline_bundle_t* bundle = nullptr;
  IREE_CHECK_OK(id4_pipeline_stage_prepare(context->stage, plan,
                                           &prepare_options, &bundle));
  return bundle;
}

static void FillQwen3VlSelectedHiddenStates(id4_pipeline_bundle_t* bundle) {
  iree_hal_buffer_t* selected_buffer =
      id4_qwen3_vl_stage_bundle_selected_hidden_states_buffer(bundle);
  const iree_device_size_t byte_length =
      id4_qwen3_vl_stage_bundle_condition_byte_length(bundle);
  std::vector<float> selected_values(byte_length / sizeof(float));
  for (iree_host_size_t i = 0; i < selected_values.size(); ++i) {
    selected_values[i] = static_cast<float>(i) * 0.125f - 17.0f;
  }
  IREE_CHECK_OK(iree_hal_buffer_map_write(selected_buffer, /*target_offset=*/0,
                                          selected_values.data(), byte_length));
}

static void SubmitQwen3VlBundle(Qwen3VlBenchmarkContext* context,
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
}

static void IssueQwen3VlBundleEndToEnd(Qwen3VlBenchmarkContext* context,
                                       id4_pipeline_bundle_t* bundle,
                                       iree_hal_semaphore_t* issue_semaphore,
                                       uint64_t issue_value) {
  SubmitQwen3VlBundle(context, bundle, issue_semaphore, issue_value);
  IREE_CHECK_OK(iree_hal_semaphore_wait(issue_semaphore, issue_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE));
}

static void BM_Qwen3VlStagePlan(benchmark::State& state) {
  Qwen3VlBenchmarkContext context = CreateLoadedQwen3VlStage();
  for (auto _ : state) {
    id4_pipeline_plan_t* plan = CreateQwen3VlPlan(&context);
    benchmark::DoNotOptimize(plan);
    id4_pipeline_plan_release(plan);
  }
  DestroyLoadedQwen3VlStage(&context);
}
BENCHMARK(BM_Qwen3VlStagePlan);

static void BM_Qwen3VlStagePrepareEndToEnd(benchmark::State& state) {
  Qwen3VlBenchmarkContext context = CreateLoadedQwen3VlStage();
  id4_pipeline_plan_t* plan = CreateQwen3VlPlan(&context);
  for (auto _ : state) {
    id4_pipeline_bundle_t* bundle = PrepareQwen3VlBundle(&context, plan);
    benchmark::DoNotOptimize(bundle);
    id4_pipeline_bundle_release(bundle);
  }
  id4_pipeline_plan_release(plan);
  DestroyLoadedQwen3VlStage(&context);
}
BENCHMARK(BM_Qwen3VlStagePrepareEndToEnd);

static void BM_Qwen3VlStageIssueSubmitOnly(benchmark::State& state) {
  Qwen3VlBenchmarkContext context = CreateLoadedQwen3VlStage();
  id4_pipeline_plan_t* plan = CreateQwen3VlPlan(&context);
  id4_pipeline_bundle_t* bundle = PrepareQwen3VlBundle(&context, plan);
  FillQwen3VlSelectedHiddenStates(bundle);

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* issue_semaphore = id4::test::CreateSemaphore(device);
  uint64_t issue_value = 0;
  for (auto _ : state) {
    SubmitQwen3VlBundle(&context, bundle, issue_semaphore, ++issue_value);
    benchmark::DoNotOptimize(bundle);
    state.PauseTiming();
    IREE_CHECK_OK(iree_hal_semaphore_wait(issue_semaphore, issue_value,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE));
    state.ResumeTiming();
  }

  iree_hal_semaphore_release(issue_semaphore);
  id4_pipeline_bundle_release(bundle);
  id4_pipeline_plan_release(plan);
  DestroyLoadedQwen3VlStage(&context);
}
BENCHMARK(BM_Qwen3VlStageIssueSubmitOnly);

static void BM_Qwen3VlStageIssueEndToEnd(benchmark::State& state) {
  Qwen3VlBenchmarkContext context = CreateLoadedQwen3VlStage();
  id4_pipeline_plan_t* plan = CreateQwen3VlPlan(&context);
  id4_pipeline_bundle_t* bundle = PrepareQwen3VlBundle(&context, plan);
  FillQwen3VlSelectedHiddenStates(bundle);

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(id4_pipeline_plan_device_group(plan), 0);
  iree_hal_semaphore_t* issue_semaphore = id4::test::CreateSemaphore(device);
  uint64_t issue_value = 0;
  for (auto _ : state) {
    IssueQwen3VlBundleEndToEnd(&context, bundle, issue_semaphore,
                               ++issue_value);
    benchmark::DoNotOptimize(bundle);
  }

  iree_hal_semaphore_release(issue_semaphore);
  id4_pipeline_bundle_release(bundle);
  id4_pipeline_plan_release(plan);
  DestroyLoadedQwen3VlStage(&context);
}
BENCHMARK(BM_Qwen3VlStageIssueEndToEnd);

static void BM_Qwen3VlStageFullLifecycleEndToEnd(benchmark::State& state) {
  for (auto _ : state) {
    Qwen3VlBenchmarkContext context = CreateLoadedQwen3VlStage();
    id4_pipeline_plan_t* plan = CreateQwen3VlPlan(&context);
    id4_pipeline_bundle_t* bundle = PrepareQwen3VlBundle(&context, plan);
    FillQwen3VlSelectedHiddenStates(bundle);
    iree_hal_device_t* device = iree_hal_device_group_device_at(
        id4_pipeline_plan_device_group(plan), 0);
    iree_hal_semaphore_t* issue_semaphore = id4::test::CreateSemaphore(device);
    IssueQwen3VlBundleEndToEnd(&context, bundle, issue_semaphore, 1);
    benchmark::DoNotOptimize(bundle);

    iree_hal_semaphore_release(issue_semaphore);
    id4_pipeline_bundle_release(bundle);
    id4_pipeline_plan_release(plan);
    DestroyLoadedQwen3VlStage(&context);
  }
}
BENCHMARK(BM_Qwen3VlStageFullLifecycleEndToEnd);

}  // namespace
