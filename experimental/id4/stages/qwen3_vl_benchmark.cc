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

static constexpr uint32_t kBenchmarkTokenCount = 64;

struct SemaphoreListStorage {
  // Semaphore carried by this single-entry list.
  iree_hal_semaphore_t* semaphore = nullptr;
  // Payload value paired with the semaphore.
  uint64_t payload_value = 0;

  iree_hal_semaphore_list_t list() {
    return iree_hal_semaphore_list_t{
        // One semaphore is carried by this stack-backed list.
        /*.count=*/1,
        // Stack-backed semaphore pointer array.
        /*.semaphores=*/&semaphore,
        // Stack-backed payload value array.
        /*.payload_values=*/&payload_value,
    };
  }
};

struct BoundaryBindingSet {
  // Number of boundary bindings allocated from the plan.
  iree_host_size_t count = 0;
  // Owned HAL buffers backing each boundary binding.
  iree_hal_buffer_t** buffers = nullptr;
  // Binding table entries in plan boundary tensor order.
  iree_hal_buffer_binding_t* bindings = nullptr;

  ~BoundaryBindingSet() { reset(); }

  void reset() {
    if (buffers) {
      for (iree_host_size_t i = 0; i < count; ++i) {
        iree_hal_buffer_release(buffers[i]);
      }
    }
    iree_allocator_free(iree_allocator_system(), buffers);
    iree_allocator_free(iree_allocator_system(), bindings);
    count = 0;
    buffers = nullptr;
    bindings = nullptr;
  }
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
                                    id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = nullptr;

  id4_qwen3_vl_stage_plan_options_t qwen_options;
  std::memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = kBenchmarkTokenCount;

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

  SemaphoreListStorage signal;
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

static iree_status_t AllocateBoundaryBindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, BoundaryBindingSet* out_binding_set) {
  IREE_ASSERT_ARGUMENT(out_binding_set);
  out_binding_set->reset();

  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  if (boundary_count == 0) return iree_ok_status();

  iree_status_t status = iree_allocator_malloc_array(
      iree_allocator_system(), boundary_count,
      sizeof(out_binding_set->buffers[0]),
      reinterpret_cast<void**>(&out_binding_set->buffers));
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->buffers, 0,
                boundary_count * sizeof(out_binding_set->buffers[0]));
    status = iree_allocator_malloc_array(
        iree_allocator_system(), boundary_count,
        sizeof(out_binding_set->bindings[0]),
        reinterpret_cast<void**>(&out_binding_set->bindings));
  }
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->bindings, 0,
                boundary_count * sizeof(out_binding_set->bindings[0]));
    out_binding_set->count = boundary_count;
  }

  for (iree_host_size_t i = 0;
       i < out_binding_set->count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "missing boundary tensor plan %" PRIhsz, i);
      break;
    }
    iree_hal_buffer_params_t params;
    std::memset(&params, 0, sizeof(params));
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
    params.queue_affinity = queue_affinity;
    params.min_alignment =
        boundary->layout.alignment ? boundary->layout.alignment : 1;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, boundary->layout.byte_length,
        &out_binding_set->buffers[i]);
    if (iree_status_is_ok(status)) {
      out_binding_set->bindings[i] = iree_hal_buffer_binding_t{
          // Boundary buffer supplied in plan order.
          /*.buffer=*/out_binding_set->buffers[i],
          // Boundary buffers are allocated as exact standalone allocations.
          /*.offset=*/0,
          // Full planned tensor byte range.
          /*.length=*/boundary->layout.byte_length,
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    out_binding_set->reset();
  }
  return status;
}

static iree_status_t QueueFillInitializedBoundaryTensors(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BoundaryBindingSet& binding_set,
    iree_hal_semaphore_t* fill_semaphore, uint64_t* out_fill_value) {
  IREE_ASSERT_ARGUMENT(out_fill_value);
  const uint32_t zero_pattern = 0;
  for (iree_host_size_t i = 0; i < binding_set.count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (!boundary ||
        !iree_all_bits_set(boundary->flags,
                           ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED)) {
      continue;
    }

    iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
    SemaphoreListStorage wait;
    wait.semaphore = fill_semaphore;
    wait.payload_value = *out_fill_value;
    if (wait.payload_value != 0) {
      wait_list = wait.list();
    }
    SemaphoreListStorage signal;
    signal.semaphore = fill_semaphore;
    signal.payload_value = wait.payload_value + 1;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
        device, queue_affinity, wait_list, signal.list(),
        binding_set.bindings[i].buffer, binding_set.bindings[i].offset,
        binding_set.bindings[i].length, &zero_pattern, sizeof(zero_pattern),
        IREE_HAL_FILL_FLAG_NONE));
    *out_fill_value = signal.payload_value;
  }
  return iree_ok_status();
}

static iree_status_t IssueQwenBundle(
    QwenBenchmarkContext* context, id4_pipeline_bundle_t* bundle,
    const BoundaryBindingSet& boundary_bindings,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(bundle);

  SemaphoreListStorage wait;
  wait.semaphore = wait_semaphore;
  wait.payload_value = wait_value;
  SemaphoreListStorage signal;
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
    iree_benchmark_unit_t time_unit) {
  iree_benchmark_def_t* benchmark = iree_make_function_benchmark(run);
  benchmark->flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME;
  benchmark->time_unit = time_unit;
  return iree_benchmark_register(name, benchmark);
}

#define ID4_QWEN_BENCHMARK_REGISTER(name, time_unit)                 \
  static const iree_benchmark_def_t* name##_registration             \
      IREE_ATTRIBUTE_UNUSED =                                        \
          RegisterQwenBenchmark(iree_make_cstring_view(#name), name, \
                                IREE_BENCHMARK_UNIT_##time_unit)

IREE_BENCHMARK_FN(BM_Qwen3VlStagePlan) {
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenStageContext(&context));

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * kBenchmarkTokenCount));
  return iree_ok_status();
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePlan, MICROSECOND);

IREE_BENCHMARK_FN(BM_Qwen3VlStagePrepareCachedKernels) {
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenBenchmarkContext(&context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, plan.out()));

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
      static_cast<int64_t>(iteration_count * kBenchmarkTokenCount));
  return iree_ok_status();
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStagePrepareCachedKernels, MILLISECOND);

static iree_status_t RunIssueBenchmark(iree_benchmark_state_t* benchmark_state,
                                       bool submit_only) {
  QwenBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedQwenBenchmarkContext(&context));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateQwenPlan(&context, plan.out()));

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

  BoundaryBindingSet boundary_bindings;
  IREE_RETURN_IF_ERROR(AllocateBoundaryBindings(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      fill_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, fill_semaphore.out()));
  uint64_t fill_value = 0;
  IREE_RETURN_IF_ERROR(QueueFillInitializedBoundaryTensors(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      boundary_bindings, fill_semaphore.get(), &fill_value));
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
    if (submit_only) iree_benchmark_pause_timing(benchmark_state);
    iree_status_t status =
        WaitForSemaphore(issue_semaphore.get(), signal_value);
    if (submit_only) iree_benchmark_resume_timing(benchmark_state);
    IREE_RETURN_IF_ERROR(status);
    wait_semaphore = issue_semaphore.get();
    wait_value = signal_value;
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * kBenchmarkTokenCount));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Qwen3VlStageIssueSubmitOnly) {
  return RunIssueBenchmark(benchmark_state, /*submit_only=*/true);
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueSubmitOnly, MICROSECOND);

IREE_BENCHMARK_FN(BM_Qwen3VlStageIssueEndToEnd) {
  return RunIssueBenchmark(benchmark_state, /*submit_only=*/false);
}
ID4_QWEN_BENCHMARK_REGISTER(BM_Qwen3VlStageIssueEndToEnd, MILLISECOND);

#undef ID4_QWEN_BENCHMARK_REGISTER

}  // namespace
