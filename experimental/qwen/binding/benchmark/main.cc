// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "benchmark/benchmark.h"
#include "experimental/qwen/runtime/model.h"
#include "experimental/qwen/runtime/program.h"
#include "experimental/qwen/runtime/request.h"
#include "experimental/qwen/tooling/layer_data.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/tooling/device_util.h"

#define QWEN_LAYER_INDEX 0
#define QWEN_LAYER_TOKEN_COUNT 512

IREE_FLAG(string, input, "",
          "Raw pre-attention layer_inp: exactly 512x2048 little-endian F32 "
          "values.");
IREE_FLAG(string, expected, "",
          "Raw complete post-attention/post-MoE residual l_out in the same "
          "F32 layout; ffn_moe_out is not a complete layer output.");
IREE_FLAG(float, atol, 0.05f, "Absolute output comparison tolerance.");
IREE_FLAG(float, rtol, 0.02f, "Relative output comparison tolerance.");

namespace {

typedef struct QwenBenchmarkTimepoint {
  // Timeline semaphore carrying this timepoint.
  iree_hal_semaphore_t* semaphore;
  // Monotonically increasing timeline value.
  uint64_t value;
} QwenBenchmarkTimepoint;

typedef struct QwenLayerBenchmarkEnvironment {
  // Allocator used for all process-global host objects.
  iree_allocator_t host_allocator;
  // Mapped input and correctness witness.
  qwen_tooling_layer_data_t layer_data;
  // Standard IREE device, parameter, and profiling configuration.
  qwen_tooling_runtime_context_t runtime_context;
  // Caller-owned timeline serializing model, request, reset, and issue work.
  iree_hal_semaphore_t* timeline;
  // Last allocated value on |timeline|.
  uint64_t timeline_value;
  // Resident fixed Qwen model.
  qwen_model_t* model;
  // Reusable complete layer-0 program.
  qwen_program_t* program;
  // Persistent layer request state.
  qwen_request_t* request;
  // Host output storage read after observed issue completion.
  void* output_buffer;
  // First or joined terminal error from setup, a row, or profiling.
  iree_status_t terminal_status;
} QwenLayerBenchmarkEnvironment;

static iree_hal_semaphore_list_t QwenBenchmarkTimepointList(
    QwenBenchmarkTimepoint* timepoint) {
  iree_hal_semaphore_list_t list = {
      /*.count=*/1,
      /*.semaphores=*/&timepoint->semaphore,
      /*.payload_values=*/&timepoint->value,
  };
  return list;
}

static QwenBenchmarkTimepoint QwenBenchmarkNextTimepoint(
    QwenLayerBenchmarkEnvironment* environment) {
  QwenBenchmarkTimepoint timepoint = {
      /*.semaphore=*/environment->timeline,
      /*.value=*/++environment->timeline_value,
  };
  return timepoint;
}

static QwenBenchmarkTimepoint QwenBenchmarkCurrentTimepoint(
    QwenLayerBenchmarkEnvironment* environment) {
  QwenBenchmarkTimepoint timepoint = {
      /*.semaphore=*/environment->timeline,
      /*.value=*/environment->timeline_value,
  };
  return timepoint;
}

static void QwenBenchmarkRecordFailure(
    QwenLayerBenchmarkEnvironment* environment, iree_status_t status,
    benchmark::State* benchmark_state) {
  if (iree_status_is_ok(status)) return;
  environment->terminal_status =
      iree_status_join(environment->terminal_status, status);
  if (benchmark_state) {
    benchmark_state->SkipWithError(
        "Qwen layer execution failed; the process exits nonzero with details");
  }
}

static iree_status_t QwenBenchmarkResetRequest(
    QwenLayerBenchmarkEnvironment* environment) {
  QwenBenchmarkTimepoint wait_timepoint =
      QwenBenchmarkCurrentTimepoint(environment);
  QwenBenchmarkTimepoint signal_timepoint =
      QwenBenchmarkNextTimepoint(environment);
  IREE_RETURN_IF_ERROR(qwen_request_reset_hidden_state(
      environment->request,
      qwen_tooling_layer_data_input(&environment->layer_data),
      QwenBenchmarkTimepointList(&wait_timepoint),
      QwenBenchmarkTimepointList(&signal_timepoint)));
  return iree_hal_semaphore_wait(environment->timeline, signal_timepoint.value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t QwenBenchmarkIssueAndWait(
    QwenLayerBenchmarkEnvironment* environment, iree_time_t* out_elapsed_time) {
  QwenBenchmarkTimepoint wait_timepoint =
      QwenBenchmarkCurrentTimepoint(environment);
  QwenBenchmarkTimepoint signal_timepoint =
      QwenBenchmarkNextTimepoint(environment);

  const iree_time_t start_time = iree_time_now();
  IREE_RETURN_IF_ERROR(
      qwen_program_issue(environment->program, environment->request,
                         QwenBenchmarkTimepointList(&wait_timepoint),
                         QwenBenchmarkTimepointList(&signal_timepoint)));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(
      environment->timeline, signal_timepoint.value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  *out_elapsed_time = iree_time_now() - start_time;
  return iree_ok_status();
}

static iree_status_t QwenBenchmarkReadAndValidate(
    QwenLayerBenchmarkEnvironment* environment) {
  iree_byte_span_t output_data = iree_make_byte_span(
      environment->output_buffer, environment->layer_data.byte_length);
  IREE_RETURN_IF_ERROR(
      qwen_request_read_hidden_state(environment->request, output_data));

  qwen_tooling_layer_comparison_t comparison;
  return qwen_tooling_layer_data_compare(
      &environment->layer_data,
      iree_make_const_byte_span(output_data.data, output_data.data_length),
      FLAG_atol, FLAG_rtol, &comparison);
}

static iree_status_t QwenBenchmarkEnvironmentInitialize(
    QwenLayerBenchmarkEnvironment* environment) {
  environment->host_allocator = iree_allocator_system();
  environment->terminal_status = iree_ok_status();

  iree_status_t status = qwen_tooling_layer_data_initialize(
      iree_make_cstring_view(FLAG_input), iree_make_cstring_view(FLAG_expected),
      QWEN_LAYER_TOKEN_COUNT, environment->host_allocator,
      &environment->layer_data);
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        environment->host_allocator, &environment->runtime_context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        qwen_tooling_runtime_context_device(&environment->runtime_context),
        IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &environment->timeline);
  }

  QwenBenchmarkTimepoint model_ready = QwenBenchmarkNextTimepoint(environment);
  if (iree_status_is_ok(status)) {
    qwen_model_options_t model_options;
    qwen_model_options_initialize(&model_options);
    model_options.device_group = environment->runtime_context.device_group;
    qwen_parameter_source_t parameter_source = {
        /*.index=*/environment->runtime_context.parameter_index,
        /*.provider=*/environment->runtime_context.parameter_provider,
        /*.scope=*/iree_string_view_empty(),
    };
    status = qwen_model_load(&model_options, &parameter_source,
                             iree_hal_semaphore_list_empty(),
                             QwenBenchmarkTimepointList(&model_ready),
                             environment->host_allocator, &environment->model);
  }

  // Host-side program preparation overlaps the asynchronous model gather.
  if (iree_status_is_ok(status)) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_LAYER;
    program_options.layer_index = QWEN_LAYER_INDEX;
    program_options.token_count = QWEN_LAYER_TOKEN_COUNT;
    program_options.context_capacity = QWEN_LAYER_TOKEN_COUNT;
    program_options.command_buffer_mode =
        environment->runtime_context.command_buffer_mode;
    status = qwen_program_prepare(environment->model, &program_options,
                                  environment->host_allocator,
                                  &environment->program);
  }

  QwenBenchmarkTimepoint request_ready =
      QwenBenchmarkNextTimepoint(environment);
  if (iree_status_is_ok(status)) {
    qwen_request_options_t request_options;
    qwen_request_options_initialize(&request_options);
    request_options.token_count = QWEN_LAYER_TOKEN_COUNT;
    request_options.context_capacity = QWEN_LAYER_TOKEN_COUNT;
    status =
        qwen_request_create(environment->model, &request_options,
                            QwenBenchmarkTimepointList(&model_ready),
                            QwenBenchmarkTimepointList(&request_ready),
                            environment->host_allocator, &environment->request);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(environment->host_allocator,
                                   environment->layer_data.byte_length,
                                   &environment->output_buffer);
  }

  // Warm with a real complete layer issue and validate it before registration.
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkResetRequest(environment);
  }
  iree_time_t warmup_time = 0;
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkIssueAndWait(environment, &warmup_time);
  }
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkReadAndValidate(environment);
  }
  return status;
}

static void QwenBenchmarkEnvironmentDeinitialize(
    QwenLayerBenchmarkEnvironment* environment) {
  qwen_request_release(environment->request);
  qwen_program_release(environment->program);
  qwen_model_release(environment->model);
  iree_hal_semaphore_release(environment->timeline);
  iree_allocator_free(environment->host_allocator, environment->output_buffer);
  qwen_tooling_runtime_context_deinitialize(&environment->runtime_context);
  qwen_tooling_layer_data_deinitialize(&environment->layer_data);
}

static void QwenLayer0Prefill512(QwenLayerBenchmarkEnvironment* environment,
                                 benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    (void)_;
    iree_status_t status = QwenBenchmarkResetRequest(environment);

    // Profiling excludes reset and warmup and surrounds the measured issue.
    iree_hal_profiling_from_flags_t* profiling = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_begin_device_group_profiling_from_flags(
          environment->runtime_context.device_group,
          environment->host_allocator, &profiling);
    }

    iree_time_t elapsed_time = 0;
    if (iree_status_is_ok(status)) {
      status = QwenBenchmarkIssueAndWait(environment, &elapsed_time);
    }
    if (iree_status_is_ok(status)) {
      benchmark_state.SetIterationTime((double)elapsed_time / 1000000000.0);
    }
    if (profiling) {
      status = iree_status_join(status,
                                iree_hal_end_profiling_from_flags(profiling));
    }
    if (iree_status_is_ok(status)) {
      status = QwenBenchmarkReadAndValidate(environment);
    }
    if (!iree_status_is_ok(status)) {
      QwenBenchmarkRecordFailure(environment, status, &benchmark_state);
      break;
    }
  }

  if (iree_status_is_ok(environment->terminal_status)) {
    const qwen_model_statistics_t model_statistics =
        qwen_model_statistics(environment->model);
    benchmark_state.SetItemsProcessed(benchmark_state.iterations() *
                                      QWEN_LAYER_TOKEN_COUNT);
    benchmark_state.counters["dispatches"] =
        (double)qwen_program_dispatch_count(environment->program);
    benchmark_state.counters["encoded_parameter_bytes"] =
        (double)model_statistics.encoded_parameter_bytes;
    benchmark_state.counters["model_bytes"] =
        (double)model_statistics.allocation_bytes;
    benchmark_state.counters["persistent_bytes"] =
        (double)qwen_request_persistent_byte_length(environment->request);
    benchmark_state.counters["submissions"] = 1;
    benchmark_state.counters["tokens"] = QWEN_LAYER_TOKEN_COUNT;
    benchmark_state.counters["transient_bytes"] =
        (double)qwen_program_transient_byte_length(environment->program);
  }
}

}  // namespace

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "qwen-layer-benchmark",
      "Benchmarks one complete resident Qwen layer-0 prefill-512 issue.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    benchmark::Shutdown();
    return EXIT_FAILURE;
  }

  QwenLayerBenchmarkEnvironment environment = {};
  environment.host_allocator = iree_allocator_system();
  environment.terminal_status = iree_ok_status();
  iree_status_t status = iree_ok_status();
  if (FLAG_atol < 0.0f || FLAG_rtol < 0.0f) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--atol and --rtol must be nonnegative");
  }
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkEnvironmentInitialize(&environment);
  }
  QwenBenchmarkRecordFailure(&environment, status, /*benchmark_state=*/nullptr);

  if (iree_status_is_ok(environment.terminal_status)) {
    benchmark::RegisterBenchmark(
        "Qwen/Layer0/Prefill/512",
        [&environment](benchmark::State& benchmark_state) {
          QwenLayer0Prefill512(&environment, benchmark_state);
        })
        ->Iterations(1)
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);
    benchmark::RunSpecifiedBenchmarks();
  }
  benchmark::Shutdown();
  QwenBenchmarkEnvironmentDeinitialize(&environment);

  if (!iree_status_is_ok(environment.terminal_status)) {
    iree_status_fprint(stderr, environment.terminal_status);
    iree_status_free(environment.terminal_status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
