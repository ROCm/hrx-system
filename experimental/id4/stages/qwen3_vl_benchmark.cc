// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/benchmark.h"

namespace {

struct QwenBenchmarkShape {
  // Dynamic request token count used when planning Qwen3-VL forward.
  uint32_t token_count;
};

static constexpr QwenBenchmarkShape kQwenSmoke64Shape = {
    // Historical reduced Qwen smoke token count.
    64,
};

static constexpr QwenBenchmarkShape kQwenIdeogram4Text451Shape = {
    // Token count from the full structured city-walk request.
    451,
};

enum class QwenIssueTimingMode {
  // Measures queue submission while waiting for completion outside timing.
  kSubmitOnly,
  // Measures user-visible issue through completion.
  kEndToEnd,
};

struct QwenBenchmarkContext {
  // Live HAL, executable cache, and kernel-cache context selected by flags.
  id4::test::LiveStageContext live;
  // Embedded Loom source library used during stage preparation.
  id4::test::KernelLibraryRef kernel_library;
  // Parameter provider created from standard --parameters= flags.
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  // Loaded Qwen3-VL stage under benchmark.
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  // Diagnostic event counters collected by lifecycle calls.
  id4::test::StageDiagnostics diagnostics = {};
  // Diagnostics sink passed to stage lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
};

static const QwenBenchmarkShape& QwenBenchmarkShapeFromDef(
    const iree_benchmark_def_t* benchmark_def) {
  return *static_cast<const QwenBenchmarkShape*>(benchmark_def->user_data);
}

static iree_status_t CreateQwen3VlStage(const id4::test::LiveStageContext& live,
                                        id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = live.device_group.get();
  services.executable_cache = live.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_qwen3_vl_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = live.kernel_cache.get();
  create_options.model = *id4_qwen3_vl_program_ideogram4_model_config();
  return id4_qwen3_vl_stage_create(&create_options, iree_allocator_system(),
                                   out_stage);
}

static iree_status_t CreateLoadedQwenStageContext(
    QwenBenchmarkContext* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  out_context->diagnostics_sink =
      id4::test::DiagnosticsSink(&out_context->diagnostics);
  IREE_RETURN_IF_ERROR(
      id4::test::CreateLiveStageContextFromFlags(&out_context->live));
  IREE_RETURN_IF_ERROR(
      CreateQwen3VlStage(out_context->live, out_context->stage.out()));

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &out_context->diagnostics_sink;
  return id4_pipeline_stage_load(out_context->stage.get(), &load_options);
}

static iree_status_t AttachQwenPreparationInputs(
    QwenBenchmarkContext* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_RETURN_IF_ERROR(
      id4::test::CreateEmbeddedKernelLibrary(context->kernel_library.out()));
  return id4::test::CreateParameterProviderFromFlags(
      iree_string_view_empty(), context->parameter_provider.out());
}

static iree_status_t CreateLoadedQwenBenchmarkContext(
    QwenBenchmarkContext* out_context) {
  IREE_RETURN_IF_ERROR(CreateLoadedQwenStageContext(out_context));
  return AttachQwenPreparationInputs(out_context);
}

static iree_status_t CreateQwenPlan(QwenBenchmarkContext* context,
                                    const QwenBenchmarkShape& shape,
                                    id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = nullptr;

  id4_qwen3_vl_stage_plan_options_t qwen_options;
  std::memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = shape.token_count;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &qwen_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_plan(context->stage.get(), &plan_options, out_plan);
}

static iree_status_t PrepareQwenBundle(QwenBenchmarkContext* context,
                                       const id4_pipeline_plan_t* plan,
                                       iree_hal_semaphore_t* prepare_semaphore,
                                       uint64_t signal_value,
                                       id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = nullptr;

  id4::test::SemaphoreListStorage signal;
  signal.semaphore = prepare_semaphore;
  signal.payload_value = signal_value;

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = context->parameter_provider.get();
  prepare_options.kernel_library = context->kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal.list();
  prepare_options.command_buffer_mode = context->live.command_buffer_mode;
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_prepare(context->stage.get(), plan,
                                    &prepare_options, out_bundle);
}

static iree_status_t IssueQwenBundle(
    QwenBenchmarkContext* context, id4_pipeline_bundle_t* bundle,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(bundle);

  id4::test::SemaphoreListStorage wait;
  wait.semaphore = wait_semaphore;
  wait.payload_value = wait_value;
  id4::test::SemaphoreListStorage signal;
  signal.semaphore = signal_semaphore;
  signal.payload_value = signal_value;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.wait_semaphore_list = wait.list();
  issue_options.signal_semaphore_list = signal.list();
  issue_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_issue(context->stage.get(), bundle, &issue_options);
}

static iree_status_t WaitForSemaphore(iree_hal_semaphore_t* semaphore,
                                      uint64_t payload_value) {
  return iree_hal_semaphore_wait(semaphore, payload_value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static const iree_benchmark_def_t* RegisterQwenBenchmark(
    iree_string_view_t name, iree_benchmark_fn_t run,
    iree_benchmark_unit_t time_unit, const QwenBenchmarkShape* shape) {
  iree_benchmark_def_t* benchmark = iree_make_function_benchmark(run);
  benchmark->flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME;
  benchmark->time_unit = time_unit;
  benchmark->user_data = shape;
  return iree_benchmark_register(name, benchmark);
}

#define ID4_QWEN_BENCHMARK_REGISTER(name, shape, suffix, time_unit) \
  static const iree_benchmark_def_t* name##_##shape##_registration  \
      IREE_ATTRIBUTE_UNUSED =                                       \
          RegisterQwenBenchmark(IREE_SV(#name "/" suffix), name,    \
                                IREE_BENCHMARK_UNIT_##time_unit, &shape)

IREE_BENCHMARK_FN(BM_Qwen3VlStagePlan) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenStageContext(&context));

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, shape, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * shape.token_count));
  return iree_ok_status();
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePlan, kQwenSmoke64Shape, "smoke64",
                            MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePlan, kQwenIdeogram4Text451Shape,
                            "id4_text451", MICROSECOND);

IREE_BENCHMARK_FN(BM_Qwen3VlStagePrepareCachedKernels) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenBenchmarkContext(&context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, shape, plan.out()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  uint64_t prepare_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      warm_bundle;
  IREE_RETURN_IF_ERROR(PrepareQwenBundle(&context, plan.get(),
                                         prepare_semaphore.get(), prepare_value,
                                         warm_bundle.out()));
  IREE_RETURN_IF_ERROR(
      WaitForSemaphore(prepare_semaphore.get(), prepare_value));
  warm_bundle.reset();

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    ++prepare_value;
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
        bundle;
    IREE_RETURN_IF_ERROR(PrepareQwenBundle(&context, plan.get(),
                                           prepare_semaphore.get(),
                                           prepare_value, bundle.out()));
    IREE_RETURN_IF_ERROR(
        WaitForSemaphore(prepare_semaphore.get(), prepare_value));
    iree_optimization_barrier(bundle.get());
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * shape.token_count));
  return iree_ok_status();
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareCachedKernels,
                            kQwenSmoke64Shape, "smoke64", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareCachedKernels,
                            kQwenIdeogram4Text451Shape, "id4_text451",
                            MILLISECOND);

static iree_status_t RunIssueBenchmark(iree_benchmark_state_t* benchmark_state,
                                       const QwenBenchmarkShape& shape,
                                       QwenIssueTimingMode timing_mode) {
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenBenchmarkContext(&context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, shape, plan.out()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_RETURN_IF_ERROR(PrepareQwenBundle(
      &context, plan.get(), prepare_semaphore.get(), 1, bundle.out()));
  IREE_RETURN_IF_ERROR(WaitForSemaphore(prepare_semaphore.get(), 1));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_RETURN_IF_ERROR(id4::test::AllocateBoundaryBindings(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      fill_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, fill_semaphore.out()));
  uint64_t fill_value = 0;
  const uint32_t zero_pattern = 0;
  IREE_RETURN_IF_ERROR(id4::test::QueueFillBoundaryTensors(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED,
      &zero_pattern, sizeof(zero_pattern), fill_semaphore.get(), &fill_value));
  if (fill_value != 0) {
    IREE_RETURN_IF_ERROR(WaitForSemaphore(fill_semaphore.get(), fill_value));
  }

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  iree_hal_semaphore_t* wait_semaphore = fill_semaphore.get();
  uint64_t wait_value = fill_value;
  uint64_t signal_value = 0;
  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    ++signal_value;
    IREE_RETURN_IF_ERROR(IssueQwenBundle(
        &context, bundle.get(), boundary_bindings, wait_semaphore, wait_value,
        issue_semaphore.get(), signal_value));
    if (timing_mode == QwenIssueTimingMode::kSubmitOnly) {
      iree_benchmark_pause_timing(benchmark_state);
    }
    iree_status_t status =
        WaitForSemaphore(issue_semaphore.get(), signal_value);
    if (timing_mode == QwenIssueTimingMode::kSubmitOnly) {
      iree_benchmark_resume_timing(benchmark_state);
    }
    IREE_RETURN_IF_ERROR(status);
    wait_semaphore = issue_semaphore.get();
    wait_value = signal_value;
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * shape.token_count));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Qwen3VlStageIssueSubmitOnly) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  return RunIssueBenchmark(benchmark_state, shape,
                           QwenIssueTimingMode::kSubmitOnly);
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSubmitOnly, kQwenSmoke64Shape,
                            "smoke64", MICROSECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSubmitOnly,
                            kQwenIdeogram4Text451Shape, "id4_text451",
                            MICROSECOND);

IREE_BENCHMARK_FN(BM_Qwen3VlStageIssueEndToEnd) {
  const QwenBenchmarkShape& shape = QwenBenchmarkShapeFromDef(benchmark_def);
  return RunIssueBenchmark(benchmark_state, shape,
                           QwenIssueTimingMode::kEndToEnd);
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueEndToEnd, kQwenSmoke64Shape,
                            "smoke64", MILLISECOND);
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueEndToEnd,
                            kQwenIdeogram4Text451Shape, "id4_text451",
                            MILLISECOND);

#undef ID4_QWEN_BENCHMARK_REGISTER

}  // namespace
