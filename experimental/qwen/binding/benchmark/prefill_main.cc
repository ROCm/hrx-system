// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "benchmark/benchmark.h"
#include "experimental/qwen/runtime/model.h"
#include "experimental/qwen/runtime/model_shape.h"
#include "experimental/qwen/runtime/program.h"
#include "experimental/qwen/runtime/request.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/types.h"
#include "iree/tooling/device_util.h"

#define QWEN_PREFILL_FIXTURE_TOKEN_COUNT 512

IREE_FLAG(string, tokens, "",
          "Raw prefill token IDs: exactly 512 little-endian I32 values; the "
          "selected prefill shape consumes its leading values.");
IREE_FLAG(int32_t, prefill_token_count, QWEN_PREFILL_FIXTURE_TOKEN_COUNT,
          "Leading fixture token count to prefill in [1, 512].");
IREE_FLAG(int32_t, expected_prefill_token, IREE_TOKENIZER_TOKEN_ID_INVALID,
          "Required token expected from the configured prefill shape.");
IREE_FLAG(int32_t, expected_decode_token, IREE_TOKENIZER_TOKEN_ID_INVALID,
          "Expected token selected after appending the prefill-selected token. "
          "Providing this enables the exact one-token decode row.");

namespace {

typedef struct QwenBenchmarkTimepoint {
  // Timeline semaphore carrying this timepoint.
  iree_hal_semaphore_t* semaphore;
  // Monotonically increasing timeline value.
  uint64_t value;
} QwenBenchmarkTimepoint;

typedef struct QwenPrefillBenchmarkEnvironment {
  // Allocator used for all process-global host objects.
  iree_allocator_t host_allocator;
  // Standard IREE device, parameter, and profiling configuration.
  qwen_tooling_runtime_context_t runtime_context;
  // Caller-owned timeline serializing model, request, reset, and issue work.
  iree_hal_semaphore_t* timeline;
  // Last allocated value on |timeline|.
  uint64_t timeline_value;
  // Externally published completion of the asynchronous model gather.
  QwenBenchmarkTimepoint model_ready;
  // Leading fixture token count consumed by the configured prefill shape.
  iree_host_size_t prefill_token_count;
  // Externally supplied token oracle for the configured prefill shape.
  iree_tokenizer_token_id_t expected_prefill_token;
  // Validated token fixture retained for the process lifetime.
  iree_tokenizer_token_id_t token_ids[QWEN_PREFILL_FIXTURE_TOKEN_COUNT];
  // Resident fixed Qwen model.
  qwen_model_t* model;
  // Reusable full-model prefill program.
  qwen_program_t* prefill_program;
  // Exact-count one-token decode program, when requested.
  qwen_program_t* decode_program;
  // Persistent full-model request state.
  qwen_request_t* request;
  // First or joined terminal error from setup, a row, or profiling.
  iree_status_t terminal_status;
} QwenPrefillBenchmarkEnvironment;

typedef enum QwenBenchmarkInputKind {
  QWEN_BENCHMARK_INPUT_KIND_PREFILL = 0,
  QWEN_BENCHMARK_INPUT_KIND_DECODE = 1,
} QwenBenchmarkInputKind;

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
    QwenPrefillBenchmarkEnvironment* environment) {
  QwenBenchmarkTimepoint timepoint = {
      /*.semaphore=*/environment->timeline,
      /*.value=*/++environment->timeline_value,
  };
  return timepoint;
}

static QwenBenchmarkTimepoint QwenBenchmarkCurrentTimepoint(
    QwenPrefillBenchmarkEnvironment* environment) {
  QwenBenchmarkTimepoint timepoint = {
      /*.semaphore=*/environment->timeline,
      /*.value=*/environment->timeline_value,
  };
  return timepoint;
}

// Transient, non-sanctioned containment for an AMDGPU async file-action
// teardown defect. A host preparation failure must not release the tooling
// runtime while the model gather remains active. Keep this wait tool-only and
// delete it when asynchronous device teardown is safe.
static void qwen_wait_for_model_ready_bringup_workaround(
    QwenPrefillBenchmarkEnvironment* environment) {
  if (!environment->model) return;
  environment->terminal_status = iree_status_join(
      environment->terminal_status,
      iree_hal_semaphore_wait(
          environment->model_ready.semaphore, environment->model_ready.value,
          iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

static void QwenBenchmarkRecordFailure(
    QwenPrefillBenchmarkEnvironment* environment, iree_status_t status,
    benchmark::State* benchmark_state) {
  if (iree_status_is_ok(status)) return;
  environment->terminal_status =
      iree_status_join(environment->terminal_status, status);
  if (benchmark_state) {
    benchmark_state->SkipWithError(
        "Qwen prefill failed; the process exits nonzero with details");
  }
}

static iree_status_t QwenBenchmarkLoadTokens(
    QwenPrefillBenchmarkEnvironment* environment) {
  iree_string_view_t path = iree_make_cstring_view(FLAG_tokens);
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--tokens is required");
  }

  iree_io_file_contents_t* contents = nullptr;
  iree_status_t status =
      iree_io_file_contents_read(path, environment->host_allocator, &contents);
  const iree_host_size_t expected_byte_length =
      IREE_ARRAYSIZE(environment->token_ids) *
      sizeof(environment->token_ids[0]);
  if (iree_status_is_ok(status) &&
      contents->const_buffer.data_length != expected_byte_length) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--tokens must contain exactly %" PRIhsz
        " bytes (512 little-endian I32 values); received %" PRIhsz,
        expected_byte_length, contents->const_buffer.data_length);
  }

  if (iree_status_is_ok(status)) {
    const uint8_t* bytes = contents->const_buffer.data;
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(environment->token_ids);
         ++i) {
      const iree_host_size_t byte_offset =
          i * sizeof(environment->token_ids[0]);
      const uint32_t encoded = ((uint32_t)bytes[byte_offset + 0]) |
                               ((uint32_t)bytes[byte_offset + 1] << 8) |
                               ((uint32_t)bytes[byte_offset + 2] << 16) |
                               ((uint32_t)bytes[byte_offset + 3] << 24);
      if (encoded >= QWEN_MODEL_VOCABULARY_SIZE) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "--tokens value at index %" PRIhsz " is %" PRIu32
                             "; expected [0, %" PRIu32 ")",
                             i, encoded, (uint32_t)QWEN_MODEL_VOCABULARY_SIZE);
        break;
      }
      environment->token_ids[i] = (iree_tokenizer_token_id_t)encoded;
    }
  }

  iree_io_file_contents_free(contents);
  return status;
}

static iree_status_t QwenBenchmarkPublishTokens(
    QwenPrefillBenchmarkEnvironment* environment, iree_host_size_t context_base,
    iree_tokenizer_token_id_list_t token_ids) {
  QwenBenchmarkTimepoint wait_timepoint =
      QwenBenchmarkCurrentTimepoint(environment);
  QwenBenchmarkTimepoint signal_timepoint =
      QwenBenchmarkNextTimepoint(environment);
  IREE_RETURN_IF_ERROR(
      qwen_request_reset_tokens(environment->request, context_base, token_ids,
                                QwenBenchmarkTimepointList(&wait_timepoint),
                                QwenBenchmarkTimepointList(&signal_timepoint)));
  return iree_hal_semaphore_wait(environment->timeline, signal_timepoint.value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t QwenBenchmarkPublishPrefillTokens(
    QwenPrefillBenchmarkEnvironment* environment) {
  return QwenBenchmarkPublishTokens(
      environment, /*context_base=*/0,
      iree_tokenizer_make_token_id_list(environment->token_ids,
                                        environment->prefill_token_count));
}

static iree_status_t QwenBenchmarkPublishDecodeToken(
    QwenPrefillBenchmarkEnvironment* environment) {
  return QwenBenchmarkPublishTokens(
      environment, /*context_base=*/environment->prefill_token_count,
      iree_tokenizer_make_token_id_list(&environment->expected_prefill_token,
                                        1));
}

static iree_status_t QwenBenchmarkPublishInput(
    QwenPrefillBenchmarkEnvironment* environment,
    QwenBenchmarkInputKind input_kind) {
  return input_kind == QWEN_BENCHMARK_INPUT_KIND_PREFILL
             ? QwenBenchmarkPublishPrefillTokens(environment)
             : QwenBenchmarkPublishDecodeToken(environment);
}

static iree_status_t QwenBenchmarkIssueAndWait(
    QwenPrefillBenchmarkEnvironment* environment, qwen_program_t* program,
    iree_time_t* out_elapsed_time) {
  QwenBenchmarkTimepoint wait_timepoint =
      QwenBenchmarkCurrentTimepoint(environment);
  QwenBenchmarkTimepoint signal_timepoint =
      QwenBenchmarkNextTimepoint(environment);

  const iree_time_t start_time = iree_time_now();
  IREE_RETURN_IF_ERROR(
      qwen_program_issue(program, environment->request,
                         QwenBenchmarkTimepointList(&wait_timepoint),
                         QwenBenchmarkTimepointList(&signal_timepoint)));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(
      environment->timeline, signal_timepoint.value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  *out_elapsed_time = iree_time_now() - start_time;
  return iree_ok_status();
}

static iree_status_t QwenBenchmarkReadAndValidate(
    QwenPrefillBenchmarkEnvironment* environment,
    iree_tokenizer_token_id_t expected_token, const char* operation_name) {
  iree_tokenizer_token_id_t selected_token = IREE_TOKENIZER_TOKEN_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      qwen_request_read_selected_token(environment->request, &selected_token));
  if (selected_token != expected_token) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "%s selected token %" PRId32 "; expected %" PRId32,
                            operation_name, selected_token, expected_token);
  }
  return iree_ok_status();
}

static iree_status_t QwenBenchmarkEnvironmentInitialize(
    QwenPrefillBenchmarkEnvironment* environment) {
  environment->host_allocator = iree_allocator_system();
  environment->terminal_status = iree_ok_status();

  iree_status_t status = iree_ok_status();
  if (FLAG_prefill_token_count <= 0 ||
      FLAG_prefill_token_count > QWEN_PREFILL_FIXTURE_TOKEN_COUNT) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--prefill_token_count must be in [1, %d]; received %" PRId32,
        QWEN_PREFILL_FIXTURE_TOKEN_COUNT, FLAG_prefill_token_count);
  }
  if (iree_status_is_ok(status) &&
      (FLAG_expected_prefill_token < 0 ||
       FLAG_expected_prefill_token >= QWEN_MODEL_VOCABULARY_SIZE)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--expected_prefill_token must be in [0, %u); received %" PRId32,
        (uint32_t)QWEN_MODEL_VOCABULARY_SIZE, FLAG_expected_prefill_token);
  }
  if (iree_status_is_ok(status) &&
      (FLAG_expected_decode_token < IREE_TOKENIZER_TOKEN_ID_INVALID ||
       FLAG_expected_decode_token >= QWEN_MODEL_VOCABULARY_SIZE)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--expected_decode_token must be -1 or in [0, %u); received %" PRId32,
        (uint32_t)QWEN_MODEL_VOCABULARY_SIZE, FLAG_expected_decode_token);
  }
  if (iree_status_is_ok(status)) {
    environment->prefill_token_count =
        (iree_host_size_t)FLAG_prefill_token_count;
    environment->expected_prefill_token = FLAG_expected_prefill_token;
  }
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkLoadTokens(environment);
  }
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

  environment->model_ready = QwenBenchmarkNextTimepoint(environment);
  if (iree_status_is_ok(status)) {
    qwen_model_options_t model_options;
    qwen_model_options_initialize(&model_options);
    model_options.device_group = environment->runtime_context.device_group;
    if (environment->runtime_context.jit_worker_count != 0) {
      model_options.jit_worker_count =
          environment->runtime_context.jit_worker_count;
    }
    qwen_parameter_source_t parameter_source = {
        /*.index=*/environment->runtime_context.parameter_index,
        /*.provider=*/environment->runtime_context.parameter_provider,
        /*.scope=*/iree_string_view_empty(),
    };
    status = qwen_model_load(
        &model_options, &parameter_source, iree_hal_semaphore_list_empty(),
        QwenBenchmarkTimepointList(&environment->model_ready),
        environment->host_allocator, &environment->model);
  }

  const bool decode_is_enabled =
      FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID;
  const iree_host_size_t decode_context_class =
      decode_is_enabled
          ? qwen_program_decode_context_class(environment->prefill_token_count)
          : 0;
  const iree_host_size_t prefill_context_capacity =
      iree_host_align(environment->prefill_token_count,
                      QWEN_PROGRAM_ATTENTION_CONTEXT_ALIGNMENT);
  const iree_host_size_t request_context_capacity =
      decode_is_enabled ? decode_context_class : prefill_context_capacity;

  // Host-side program preparation overlaps the asynchronous model gather.
  if (iree_status_is_ok(status)) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_PREFILL;
    program_options.layer_index = 0;
    program_options.token_count = environment->prefill_token_count;
    program_options.context_count = environment->prefill_token_count;
    program_options.token_capacity = environment->prefill_token_count;
    program_options.context_capacity = request_context_capacity;
    program_options.command_buffer_mode =
        environment->runtime_context.command_buffer_mode;
    status = qwen_program_prepare(environment->model, &program_options,
                                  environment->host_allocator,
                                  &environment->prefill_program);
  }
  if (iree_status_is_ok(status) && decode_is_enabled) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_DECODE;
    program_options.layer_index = 0;
    program_options.token_count = 1;
    program_options.context_count = decode_context_class;
    program_options.token_capacity = environment->prefill_token_count;
    program_options.context_capacity = request_context_capacity;
    program_options.command_buffer_mode =
        environment->runtime_context.command_buffer_mode;
    status = qwen_program_prepare(environment->model, &program_options,
                                  environment->host_allocator,
                                  &environment->decode_program);
  }

  QwenBenchmarkTimepoint request_ready =
      QwenBenchmarkNextTimepoint(environment);
  if (iree_status_is_ok(status)) {
    qwen_request_options_t request_options;
    qwen_request_options_initialize(&request_options);
    request_options.token_capacity = environment->prefill_token_count;
    request_options.context_capacity = request_context_capacity;
    status = qwen_request_create(
        environment->model, &request_options,
        QwenBenchmarkTimepointList(&environment->model_ready),
        QwenBenchmarkTimepointList(&request_ready), environment->host_allocator,
        &environment->request);
  }

  // Upload the external fixture once, then warm and validate a complete issue
  // before Google Benchmark controls the repeated program issues.
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkPublishPrefillTokens(environment);
  }
  iree_time_t warmup_time = 0;
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkIssueAndWait(
        environment, environment->prefill_program, &warmup_time);
  }
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkReadAndValidate(
        environment, environment->expected_prefill_token, "full-model prefill");
  }
  return status;
}

static void QwenBenchmarkEnvironmentDeinitialize(
    QwenPrefillBenchmarkEnvironment* environment) {
  qwen_wait_for_model_ready_bringup_workaround(environment);
  qwen_request_release(environment->request);
  qwen_program_release(environment->decode_program);
  qwen_program_release(environment->prefill_program);
  qwen_model_release(environment->model);
  iree_hal_semaphore_release(environment->timeline);
  qwen_tooling_runtime_context_deinitialize(&environment->runtime_context);
}

static void QwenBenchmarkMeasureProgram(
    QwenPrefillBenchmarkEnvironment* environment, qwen_program_t* program,
    QwenBenchmarkInputKind input_kind, iree_tokenizer_token_id_t expected_token,
    const char* operation_name, iree_host_size_t logical_token_count,
    benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    (void)_;

    // Each measured row starts from the same production input state. The
    // reset is outside both profiling and the manual issue interval; generated
    // sequences instead consume the device-published continuation directly.
    iree_status_t status = QwenBenchmarkPublishInput(environment, input_kind);

    // Profiling excludes process setup and warmup and surrounds the measured
    // issue. The manual interval includes submission and user-visible
    // completion; dispatch-only timings come from the device profile.
    iree_hal_profiling_from_flags_t* profiling = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_begin_device_group_profiling_from_flags(
          environment->runtime_context.device_group,
          environment->host_allocator, &profiling);
    }

    iree_time_t elapsed_time = 0;
    if (iree_status_is_ok(status)) {
      status = QwenBenchmarkIssueAndWait(environment, program, &elapsed_time);
    }
    if (iree_status_is_ok(status)) {
      benchmark_state.SetIterationTime((double)elapsed_time / 1000000000.0);
    }
    if (profiling) {
      status = iree_status_join(status,
                                iree_hal_end_profiling_from_flags(profiling));
    }
    if (iree_status_is_ok(status)) {
      status = QwenBenchmarkReadAndValidate(environment, expected_token,
                                            operation_name);
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
                                      logical_token_count);
    benchmark_state.counters["dispatches"] =
        (double)qwen_program_dispatch_count(program);
    benchmark_state.counters["encoded_parameter_bytes"] =
        (double)model_statistics.encoded_parameter_bytes;
    benchmark_state.counters["persistent_bytes"] =
        (double)qwen_request_persistent_byte_length(environment->request);
    benchmark_state.counters["resident_bytes"] =
        (double)model_statistics.allocation_bytes;
    benchmark_state.counters["submissions"] = 1;
    benchmark_state.counters["tokens"] = (double)logical_token_count;
    benchmark_state.counters["transient_bytes"] =
        (double)qwen_program_transient_byte_length(program);
  }
}

static void QwenFullModelPrefill(QwenPrefillBenchmarkEnvironment* environment,
                                 benchmark::State& benchmark_state) {
  QwenBenchmarkMeasureProgram(
      environment, environment->prefill_program,
      QWEN_BENCHMARK_INPUT_KIND_PREFILL, environment->expected_prefill_token,
      "full-model prefill", environment->prefill_token_count, benchmark_state);
}

static void QwenFullModelDecode(QwenPrefillBenchmarkEnvironment* environment,
                                benchmark::State& benchmark_state) {
  iree_status_t status = QwenBenchmarkPublishDecodeToken(environment);
  iree_time_t warmup_time = 0;
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkIssueAndWait(environment, environment->decode_program,
                                       &warmup_time);
  }
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkReadAndValidate(
        environment, FLAG_expected_decode_token, "full-model decode");
  }
  if (!iree_status_is_ok(status)) {
    QwenBenchmarkRecordFailure(environment, status, &benchmark_state);
    return;
  }
  QwenBenchmarkMeasureProgram(environment, environment->decode_program,
                              QWEN_BENCHMARK_INPUT_KIND_DECODE,
                              FLAG_expected_decode_token, "full-model decode",
                              /*logical_token_count=*/1, benchmark_state);
}

}  // namespace

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "qwen-prefill-benchmark",
      "Benchmarks one selected resident Qwen full-model prefill shape and its "
      "optional exact-count decode issue.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    benchmark::Shutdown();
    return EXIT_FAILURE;
  }

  QwenPrefillBenchmarkEnvironment environment = {};
  environment.host_allocator = iree_allocator_system();
  environment.terminal_status = iree_ok_status();
  iree_status_t status = QwenBenchmarkEnvironmentInitialize(&environment);
  QwenBenchmarkRecordFailure(&environment, status,
                             /*benchmark_state=*/nullptr);

  if (iree_status_is_ok(environment.terminal_status)) {
    const std::string prefill_name =
        "Qwen/FullModel/Prefill/" +
        std::to_string(environment.prefill_token_count);
    benchmark::RegisterBenchmark(
        prefill_name.c_str(),
        [&environment](benchmark::State& benchmark_state) {
          QwenFullModelPrefill(&environment, benchmark_state);
        })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);
    if (FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID) {
      const std::string decode_name =
          "Qwen/FullModel/Decode/" +
          std::to_string(environment.prefill_token_count + 1);
      benchmark::RegisterBenchmark(
          decode_name.c_str(),
          [&environment](benchmark::State& benchmark_state) {
            QwenFullModelDecode(&environment, benchmark_state);
          })
          ->UseManualTime()
          ->Unit(benchmark::kMillisecond)
          ->Iterations(1);
    }
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
