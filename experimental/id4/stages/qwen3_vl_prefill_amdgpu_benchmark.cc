// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl_prefill.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gbenchmark_harness.h"

IREE_FLAG(string, rmsnorm_loom_source, "",
          "Path to kernels/qwen3_vl/rmsnorm.loom.");
IREE_FLAG(string, linear_loom_source, "",
          "Path to kernels/qwen3_vl/linear_bf16_f32.loom.");
IREE_FLAG(string, device_uri, "", "HAL device URI, such as amdgpu://0.");
IREE_FLAG(string, amdgpu_processor, "", "AMDGPU processor, such as gfx1100.");

namespace {

static constexpr int32_t kBenchmarkTokenCount = 1;

using BundleRef =
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>;
using ExecutableCacheRef =
    id4::test::OwningRef<iree_hal_executable_cache_t,
                         iree_hal_executable_cache_release>;
using FileContentsRef =
    id4::test::OwningRef<iree_io_file_contents_t, iree_io_file_contents_free>;
using KernelCacheRef = id4::test::OwningRef<id4_pipeline_kernel_cache_t,
                                            id4_pipeline_kernel_cache_release>;
using KernelLibraryRef =
    id4::test::OwningRef<id4_pipeline_kernel_library_t,
                         id4_pipeline_kernel_library_release>;
using ParameterProviderRef =
    id4::test::OwningRef<iree_io_parameter_provider_t,
                         iree_io_parameter_provider_release>;
using PlanRef =
    id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>;
using SemaphoreRef =
    id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>;
using StageRef =
    id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>;

typedef struct Qwen3VlPrefillAmdgpuBenchmarkContext {
  // Live HAL device and device group used by the benchmark.
  id4::test::LiveHalDevice live_device;
  // HAL executable cache used by the ID4 kernel cache.
  ExecutableCacheRef executable_cache;
  // ID4 Loom kernel cache used by stage preparation.
  KernelCacheRef kernel_cache;
  // In-memory Loom kernel library used by stage preparation.
  KernelLibraryRef kernel_library;
  // Parameter provider loaded from IREE parameter flags.
  ParameterProviderRef parameter_provider;
  // Loaded Qwen3-VL prefill stage under benchmark.
  StageRef stage;
  // Plan produced once for the current benchmark configuration.
  PlanRef plan;
  // Diagnostics counters captured during benchmark setup and iterations.
  id4::test::StageDiagnostics diagnostics = {};
  // Diagnostics sink pointing at diagnostics.
  id4_pipeline_diagnostics_sink_t diagnostics_sink = {};
  // Monotonic payload value for prepare-readiness semaphores.
  uint64_t prepare_payload_value = 0;
} Qwen3VlPrefillAmdgpuBenchmarkContext;

static bool ValidateRequiredFlags() {
  bool is_valid = true;
  if (std::strlen(FLAG_rmsnorm_loom_source) == 0) {
    std::fprintf(stderr, "--rmsnorm_loom_source is required\n");
    is_valid = false;
  }
  if (std::strlen(FLAG_linear_loom_source) == 0) {
    std::fprintf(stderr, "--linear_loom_source is required\n");
    is_valid = false;
  }
  if (std::strlen(FLAG_device_uri) == 0) {
    std::fprintf(stderr, "--device_uri is required\n");
    is_valid = false;
  }
  if (std::strlen(FLAG_amdgpu_processor) == 0) {
    std::fprintf(stderr, "--amdgpu_processor is required\n");
    is_valid = false;
  }
  return is_valid;
}

static iree_device_size_t HiddenStatesByteLength() {
  return static_cast<iree_device_size_t>(kBenchmarkTokenCount) *
         ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE * sizeof(float);
}

static void InitializeQwen3VlPrefillContext(
    Qwen3VlPrefillAmdgpuBenchmarkContext* context) {
  context->diagnostics_sink = id4::test::DiagnosticsSink(&context->diagnostics);
  IREE_CHECK_OK(id4::test::CreateLiveHalDevice(
      iree_make_cstring_view(FLAG_device_uri), &context->live_device));
  IREE_CHECK_OK(iree_hal_executable_cache_create(
      context->live_device.device.get(),
      IREE_SV("id4-qwen3-vl-prefill-benchmark"),
      context->executable_cache.out()));

  id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
  std::memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
  kernel_cache_options.structure_size = sizeof(kernel_cache_options);
  kernel_cache_options.amdgpu_processor =
      iree_make_cstring_view(FLAG_amdgpu_processor);
  IREE_CHECK_OK(id4_pipeline_kernel_cache_create(&kernel_cache_options,
                                                 iree_allocator_system(),
                                                 context->kernel_cache.out()));

  FileContentsRef rmsnorm_source_file;
  IREE_CHECK_OK(iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_rmsnorm_loom_source), iree_allocator_system(),
      rmsnorm_source_file.out()));
  FileContentsRef linear_source_file;
  IREE_CHECK_OK(iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_linear_loom_source), iree_allocator_system(),
      linear_source_file.out()));
  id4_pipeline_kernel_module_t modules[] = {
      {
          // Stable module path selected by the Qwen3-VL prefill stage.
          .module_path = IREE_SV("qwen3_vl/rmsnorm"),
          // Diagnostic source identifier for the current VFS entry.
          .source_identifier = IREE_SV("kernels/qwen3_vl/rmsnorm.loom"),
          // Source bytes loaded by the benchmark harness.
          .source_contents = rmsnorm_source_file.get()->const_buffer,
      },
      {
          // Stable module path selected by the Qwen3-VL prefill stage.
          .module_path = IREE_SV("qwen3_vl/linear_bf16_f32"),
          // Diagnostic source identifier for the current VFS entry.
          .source_identifier = IREE_SV("kernels/qwen3_vl/linear_bf16_f32.loom"),
          // Source bytes loaded by the benchmark harness.
          .source_contents = linear_source_file.get()->const_buffer,
      },
  };
  id4_pipeline_kernel_library_create_options_t library_options;
  std::memset(&library_options, 0, sizeof(library_options));
  library_options.structure_size = sizeof(library_options);
  library_options.module_count = IREE_ARRAYSIZE(modules);
  library_options.modules = modules;
  IREE_CHECK_OK(id4_pipeline_kernel_library_create(
      &library_options, iree_allocator_system(),
      context->kernel_library.out()));
  IREE_CHECK_OK(id4::test::CreateParameterProviderFromFlags(
      IREE_SV(ID4_QWEN3_VL_PREFILL_STAGE_PARAMETER_SCOPE),
      context->parameter_provider.out()));

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context->live_device.device_group.get();
  services.executable_cache = context->executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_prefill_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context->kernel_cache.get();
  create_options.token_count = kBenchmarkTokenCount;
  IREE_CHECK_OK(id4_qwen3_vl_prefill_stage_create(
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

static id4_pipeline_bundle_t* PrepareQwen3VlPrefillBundle(
    Qwen3VlPrefillAmdgpuBenchmarkContext* context,
    iree_hal_semaphore_t* readiness_semaphore,
    uint64_t readiness_payload_value) {
  iree_hal_semaphore_t* readiness_semaphores[] = {readiness_semaphore};
  uint64_t readiness_payload_values[] = {readiness_payload_value};
  iree_hal_semaphore_list_t readiness_signal_list = {
      // Number of prepare readiness semaphores.
      .count = IREE_ARRAYSIZE(readiness_semaphores),
      // Prepare readiness semaphore pointer array.
      .semaphores = readiness_semaphores,
      // Prepare readiness payload values.
      .payload_values = readiness_payload_values,
  };

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.kernel_library = context->kernel_library.get();
  prepare_options.parameter_provider = context->parameter_provider.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = readiness_signal_list;
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  id4_pipeline_bundle_t* bundle = nullptr;
  IREE_CHECK_OK(id4_pipeline_stage_prepare(
      context->stage.get(), context->plan.get(), &prepare_options, &bundle));
  return bundle;
}

static id4_pipeline_bundle_t* PrepareReadyQwen3VlPrefillBundle(
    Qwen3VlPrefillAmdgpuBenchmarkContext* context) {
  SemaphoreRef readiness_semaphore;
  IREE_CHECK_OK(iree_hal_semaphore_create(
      context->live_device.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      readiness_semaphore.out()));
  const uint64_t readiness_payload_value = ++context->prepare_payload_value;
  id4_pipeline_bundle_t* bundle = PrepareQwen3VlPrefillBundle(
      context, readiness_semaphore.get(), readiness_payload_value);
  IREE_CHECK_OK(iree_hal_semaphore_wait(
      readiness_semaphore.get(), readiness_payload_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  return bundle;
}

static void FillQwen3VlPrefillInputs(
    Qwen3VlPrefillAmdgpuBenchmarkContext* context,
    id4_pipeline_bundle_t* bundle) {
  std::vector<float> input_hidden_states(
      static_cast<size_t>(kBenchmarkTokenCount) *
          ID4_QWEN3_VL_PREFILL_STAGE_HIDDEN_SIZE,
      2.0f);
  IREE_CHECK_OK(iree_hal_device_transfer_h2d(
      context->live_device.device.get(), input_hidden_states.data(),
      id4_qwen3_vl_prefill_stage_bundle_input_buffer(bundle),
      /*target_offset=*/0, HiddenStatesByteLength(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
}

static void BM_Qwen3VlPrefillAmdgpuPrepareEndToEnd(State& state) {
  Qwen3VlPrefillAmdgpuBenchmarkContext context;
  InitializeQwen3VlPrefillContext(&context);
  SemaphoreRef readiness_semaphore;
  IREE_CHECK_OK(iree_hal_semaphore_create(
      context.live_device.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      readiness_semaphore.out()));
  for (auto _ : state) {
    const uint64_t readiness_payload_value = ++context.prepare_payload_value;
    id4_pipeline_bundle_t* bundle = PrepareQwen3VlPrefillBundle(
        &context, readiness_semaphore.get(), readiness_payload_value);
    IREE_CHECK_OK(iree_hal_semaphore_wait(
        readiness_semaphore.get(), readiness_payload_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    benchmark::DoNotOptimize(bundle);
    id4_pipeline_bundle_release(bundle);
  }
}
BENCHMARK(BM_Qwen3VlPrefillAmdgpuPrepareEndToEnd);

static void BM_Qwen3VlPrefillAmdgpuIssueEndToEnd(State& state) {
  Qwen3VlPrefillAmdgpuBenchmarkContext context;
  InitializeQwen3VlPrefillContext(&context);
  BundleRef bundle;
  bundle.reset(PrepareReadyQwen3VlPrefillBundle(&context));
  FillQwen3VlPrefillInputs(&context, bundle.get());

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
BENCHMARK(BM_Qwen3VlPrefillAmdgpuIssueEndToEnd);

}  // namespace

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "qwen3_vl_prefill_amdgpu_benchmark",
      "Benchmarks the Qwen3-VL prefill stage on AMDGPU.\n"
      "Pass --parameters=<file> to provide Qwen3-VL weights.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  if (!ValidateRequiredFlags()) return 1;
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
