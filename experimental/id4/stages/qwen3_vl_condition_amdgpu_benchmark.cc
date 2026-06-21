// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl_condition.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gbenchmark_harness.h"

namespace {

std::string g_loom_source_path;
std::string g_device_uri;
std::string g_amdgpu_processor;

using BundleRef =
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>;
using ExecutableCacheRef =
    id4::test::OwningRef<iree_hal_executable_cache_t,
                         iree_hal_executable_cache_release>;
using FileContentsRef =
    id4::test::OwningRef<iree_io_file_contents_t, iree_io_file_contents_free>;
using KernelCacheRef = id4::test::OwningRef<id4_pipeline_kernel_cache_t,
                                            id4_pipeline_kernel_cache_release>;
using PlanRef =
    id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>;
using SemaphoreRef =
    id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>;
using StageRef =
    id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>;

typedef struct Qwen3VlConditionAmdgpuBenchmarkContext {
  // Live HAL device and device group used by the benchmark.
  id4::test::LiveHalDevice live_device;
  // HAL executable cache used by the ID4 kernel cache.
  ExecutableCacheRef executable_cache;
  // ID4 Loom kernel cache used by stage preparation.
  KernelCacheRef kernel_cache;
  // Loaded Qwen3-VL condition stage under benchmark.
  StageRef stage;
  // Plan produced once for the current benchmark configuration.
  PlanRef plan;
  // Diagnostics counters captured during benchmark setup and iterations.
  id4::test::StageDiagnostics diagnostics = {};
  // Diagnostics sink pointing at diagnostics.
  id4_pipeline_diagnostics_sink_t diagnostics_sink = {};
} Qwen3VlConditionAmdgpuBenchmarkContext;

static bool ArgumentHasPrefix(const char* argument, const char* prefix) {
  const iree_host_size_t prefix_length = std::strlen(prefix);
  return std::strncmp(argument, prefix, prefix_length) == 0;
}

static bool ParseBenchmarkArguments(int argc, char** argv) {
  if (id4::test::ParseStringArgument(argc, argv,
                                     "--loom_source=", &g_loom_source_path) &&
      id4::test::ParseStringArgument(argc, argv,
                                     "--device_uri=", &g_device_uri) &&
      id4::test::ParseStringArgument(
          argc, argv, "--amdgpu_processor=", &g_amdgpu_processor)) {
    return true;
  }
  std::fprintf(stderr,
               "usage: qwen3_vl_condition_amdgpu_benchmark "
               "--loom_source=<path> --device_uri=<uri> "
               "--amdgpu_processor=<processor>\n");
  return false;
}

static void RemoveBenchmarkArguments(int* argc, char** argv) {
  int target_index = 1;
  for (int source_index = 1; source_index < *argc; ++source_index) {
    if (ArgumentHasPrefix(argv[source_index], "--loom_source=") ||
        ArgumentHasPrefix(argv[source_index], "--device_uri=") ||
        ArgumentHasPrefix(argv[source_index], "--amdgpu_processor=")) {
      continue;
    }
    argv[target_index++] = argv[source_index];
  }
  *argc = target_index;
}

static void InitializeQwen3VlConditionContext(
    Qwen3VlConditionAmdgpuBenchmarkContext* context) {
  context->diagnostics_sink = id4::test::DiagnosticsSink(&context->diagnostics);
  IREE_CHECK_OK(id4::test::CreateLiveHalDevice(
      id4::test::StringView(g_device_uri), &context->live_device));
  IREE_CHECK_OK(iree_hal_executable_cache_create(
      context->live_device.device.get(),
      IREE_SV("id4-qwen3-vl-condition-benchmark"),
      context->executable_cache.out()));

  id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
  std::memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
  kernel_cache_options.structure_size = sizeof(kernel_cache_options);
  kernel_cache_options.amdgpu_processor =
      id4::test::StringView(g_amdgpu_processor);
  IREE_CHECK_OK(id4_pipeline_kernel_cache_create(&kernel_cache_options,
                                                 iree_allocator_system(),
                                                 context->kernel_cache.out()));

  FileContentsRef source_file;
  IREE_CHECK_OK(
      iree_io_file_contents_read(id4::test::StringView(g_loom_source_path),
                                 iree_allocator_system(), source_file.out()));

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context->live_device.device_group.get();
  services.executable_cache = context->executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_condition_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context->kernel_cache.get();
  create_options.source_identifier = IREE_SV("qwen3_vl_condition_f32.loom");
  create_options.source_contents = source_file.get()->const_buffer;
  create_options.module_name = IREE_SV("id4_qwen3_vl_condition_f32");
  create_options.executable_identifier =
      IREE_SV("id4_qwen3_vl_condition_f32.hsaco");
  create_options.apply_token_weights_function_name =
      IREE_SV("id4_qwen3_vl_condition_apply_token_weights_f32");
  create_options.normalize_token_weights_function_name =
      IREE_SV("id4_qwen3_vl_condition_normalize_token_weights_f32");
  create_options.hidden_row_count = 1;
  create_options.token_count = 4;
  create_options.workgroup_size_x = 256;
  IREE_CHECK_OK(id4_qwen3_vl_condition_stage_create(
      &create_options, iree_allocator_system(), context->stage.out()));

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &context->diagnostics_sink;
  IREE_CHECK_OK(id4_pipeline_stage_load(context->stage.get(), &load_options));

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &context->diagnostics_sink;
  IREE_CHECK_OK(id4_pipeline_stage_plan(context->stage.get(), &plan_options,
                                        context->plan.out()));
}

static id4_pipeline_bundle_t* PrepareQwen3VlConditionBundle(
    Qwen3VlConditionAmdgpuBenchmarkContext* context) {
  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  id4_pipeline_bundle_t* bundle = nullptr;
  IREE_CHECK_OK(id4_pipeline_stage_prepare(
      context->stage.get(), context->plan.get(), &prepare_options, &bundle));
  return bundle;
}

static void FillQwen3VlConditionInputs(
    Qwen3VlConditionAmdgpuBenchmarkContext* context,
    id4_pipeline_bundle_t* bundle) {
  std::array<float, 4> selected_hidden_states = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 4> token_weights = {0.5f, 1.0f, 1.5f, 2.0f};
  IREE_CHECK_OK(iree_hal_device_transfer_h2d(
      context->live_device.device.get(), selected_hidden_states.data(),
      id4_qwen3_vl_condition_stage_bundle_selected_hidden_states_buffer(bundle),
      /*target_offset=*/0, sizeof(selected_hidden_states),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  IREE_CHECK_OK(iree_hal_device_transfer_h2d(
      context->live_device.device.get(), token_weights.data(),
      id4_qwen3_vl_condition_stage_bundle_token_weights_buffer(bundle),
      /*target_offset=*/0, sizeof(token_weights),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
}

static void BM_Qwen3VlConditionAmdgpuPrepare(State& state) {
  Qwen3VlConditionAmdgpuBenchmarkContext context;
  InitializeQwen3VlConditionContext(&context);
  for (auto _ : state) {
    id4_pipeline_bundle_t* bundle = PrepareQwen3VlConditionBundle(&context);
    benchmark::DoNotOptimize(bundle);
    id4_pipeline_bundle_release(bundle);
  }
}
BENCHMARK(BM_Qwen3VlConditionAmdgpuPrepare);

static void BM_Qwen3VlConditionAmdgpuIssueEndToEnd(State& state) {
  Qwen3VlConditionAmdgpuBenchmarkContext context;
  InitializeQwen3VlConditionContext(&context);
  BundleRef bundle;
  bundle.reset(PrepareQwen3VlConditionBundle(&context));
  FillQwen3VlConditionInputs(&context, bundle.get());

  SemaphoreRef completion_semaphore;
  IREE_CHECK_OK(iree_hal_semaphore_create(
      context.live_device.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      completion_semaphore.out()));
  iree_hal_semaphore_t* signal_semaphores[] = {completion_semaphore.get()};
  uint64_t signal_payload_values[] = {0};
  iree_hal_semaphore_list_t signal_list = {
      // Number of final signal semaphores.
      .count = IREE_ARRAYSIZE(signal_semaphores),
      // Final signal semaphore pointer array.
      .semaphores = signal_semaphores,
      // Final signal payload values.
      .payload_values = signal_payload_values,
  };

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  issue_options.signal_semaphore_list = signal_list;
  issue_options.diagnostics_sink = &context.diagnostics_sink;

  uint64_t payload_value = 0;
  for (auto _ : state) {
    signal_payload_values[0] = ++payload_value;
    IREE_CHECK_OK(id4_pipeline_stage_issue(context.stage.get(), bundle.get(),
                                           &issue_options));
    IREE_CHECK_OK(iree_hal_semaphore_wait(
        completion_semaphore.get(), payload_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
  }
}
BENCHMARK(BM_Qwen3VlConditionAmdgpuIssueEndToEnd);

}  // namespace

int main(int argc, char** argv) {
  if (!ParseBenchmarkArguments(argc, argv)) return 1;
  RemoveBenchmarkArguments(&argc, argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
