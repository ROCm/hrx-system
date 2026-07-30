// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static const char* const qwen_layer_cli_usage =
    "Runs the complete Qwen layer-0 prefill-512 program and validates its "
    "output.\n"
    "\n"
    "Required flags:\n"
    "  --device=<device URI>\n"
    "  --parameters=<GGUF or parameter archive path>\n"
    "  --input=<raw F32 hidden-state path>\n"
    "  --expected=<raw F32 expected-output path>\n";

typedef struct qwen_cli_timepoint_t {
  // Timeline semaphore carrying this timepoint.
  iree_hal_semaphore_t* semaphore;
  // Monotonically increasing timeline value.
  uint64_t value;
} qwen_cli_timepoint_t;

static iree_hal_semaphore_list_t qwen_cli_timepoint_list(
    qwen_cli_timepoint_t* timepoint) {
  iree_hal_semaphore_list_t list = {
      .count = 1,
      .semaphores = &timepoint->semaphore,
      .payload_values = &timepoint->value,
  };
  return list;
}

static iree_status_t qwen_layer_cli_run(void) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();

  if (FLAG_atol < 0.0f || FLAG_rtol < 0.0f) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--atol and --rtol must be nonnegative");
  }

  qwen_tooling_layer_data_t layer_data;
  status = qwen_tooling_layer_data_initialize(
      iree_make_cstring_view(FLAG_input), iree_make_cstring_view(FLAG_expected),
      QWEN_LAYER_TOKEN_COUNT, host_allocator, &layer_data);

  qwen_tooling_runtime_context_t runtime_context;
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  } else {
    memset(&runtime_context, 0, sizeof(runtime_context));
  }

  iree_hal_semaphore_t* timeline = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        qwen_tooling_runtime_context_device(&runtime_context),
        IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline);
  }

  qwen_cli_timepoint_t model_ready = {
      .semaphore = timeline,
      .value = 1,
  };
  qwen_model_t* model = NULL;
  if (iree_status_is_ok(status)) {
    qwen_model_options_t model_options;
    qwen_model_options_initialize(&model_options);
    model_options.device_group = runtime_context.device_group;
    qwen_parameter_source_t parameter_source = {
        .index = runtime_context.parameter_index,
        .provider = runtime_context.parameter_provider,
        .scope = iree_string_view_empty(),
    };
    status = qwen_model_load(
        &model_options, &parameter_source, iree_hal_semaphore_list_empty(),
        qwen_cli_timepoint_list(&model_ready), host_allocator, &model);
  }

  // Program preparation is synchronous host work and intentionally overlaps
  // the asynchronous model gather above.
  qwen_program_t* program = NULL;
  if (iree_status_is_ok(status)) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_LAYER;
    program_options.layer_index = QWEN_LAYER_INDEX;
    program_options.token_count = QWEN_LAYER_TOKEN_COUNT;
    program_options.context_capacity = QWEN_LAYER_TOKEN_COUNT;
    program_options.command_buffer_mode = runtime_context.command_buffer_mode;
    status =
        qwen_program_prepare(model, &program_options, host_allocator, &program);
  }

  qwen_cli_timepoint_t request_ready = {
      .semaphore = timeline,
      .value = 2,
  };
  qwen_request_t* request = NULL;
  if (iree_status_is_ok(status)) {
    qwen_request_options_t request_options;
    qwen_request_options_initialize(&request_options);
    request_options.token_count = QWEN_LAYER_TOKEN_COUNT;
    request_options.context_capacity = QWEN_LAYER_TOKEN_COUNT;
    status = qwen_request_create(
        model, &request_options, qwen_cli_timepoint_list(&model_ready),
        qwen_cli_timepoint_list(&request_ready), host_allocator, &request);
  }

  qwen_cli_timepoint_t input_ready = {
      .semaphore = timeline,
      .value = 3,
  };
  if (iree_status_is_ok(status)) {
    status = qwen_request_reset_hidden_state(
        request, qwen_tooling_layer_data_input(&layer_data),
        qwen_cli_timepoint_list(&request_ready),
        qwen_cli_timepoint_list(&input_ready));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(timeline, input_ready.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }

  void* output_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, layer_data.byte_length,
                                   &output_buffer);
  }

  // Profiling surrounds only the issue and host-visible completion wait.
  iree_hal_profiling_from_flags_t* profiling = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        runtime_context.device_group, host_allocator, &profiling);
  }

  qwen_cli_timepoint_t issue_complete = {
      .semaphore = timeline,
      .value = 4,
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_issue(program, request,
                                qwen_cli_timepoint_list(&input_ready),
                                qwen_cli_timepoint_list(&issue_complete));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(timeline, issue_complete.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (profiling) {
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  }

  iree_byte_span_t output_data =
      iree_make_byte_span(output_buffer, layer_data.byte_length);
  if (iree_status_is_ok(status)) {
    status = qwen_request_read_hidden_state(request, output_data);
  }

  qwen_tooling_layer_comparison_t comparison;
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_layer_data_compare(
        &layer_data,
        iree_make_const_byte_span(output_data.data, output_data.data_length),
        FLAG_atol, FLAG_rtol, &comparison);
  }
  if (iree_status_is_ok(status)) {
    const qwen_model_statistics_t model_statistics =
        qwen_model_statistics(model);
    fprintf(stdout,
            "Qwen layer 0 prefill 512 passed: %" PRIhsz " F32 values, %" PRIhsz
            " dispatches, %" PRIu64 " resident bytes, max_abs=%g, max_rel=%g\n",
            comparison.element_count, qwen_program_dispatch_count(program),
            (uint64_t)model_statistics.allocation_bytes,
            comparison.maximum_absolute_error,
            comparison.maximum_relative_error);
  }

  iree_allocator_free(host_allocator, output_buffer);
  qwen_request_release(request);
  qwen_program_release(program);
  qwen_model_release(model);
  iree_hal_semaphore_release(timeline);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  qwen_tooling_layer_data_deinitialize(&layer_data);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen-layer-cli", qwen_layer_cli_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) {
    status = qwen_layer_cli_run();
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
