// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Independent real-tensor qualification of a source-JIT command component.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/device.h"
#include "experimental/loom_serve/weights.h"
#include "iree/base/internal/math.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, model, "", "Source catalog directory.");
IREE_FLAG(string, root, "", "Command root to qualify.");
IREE_FLAG_LIST(string, weights,
               "Checkpoint path; repeat in reflected parameter-root order.");
IREE_FLAG(string, weight_policy, "",
          "Weight preparation source; required with --weights.");
IREE_FLAG_LIST(string, input, "Raw input file; repeat in binding order.");
IREE_FLAG_LIST(string, config, "JIT specialization key=value; repeat per key.");
IREE_FLAG(string, expected, "", "Raw little-endian BF16 output reference.");
IREE_FLAG(string, actual, "",
          "Optional file overwritten with the first completed BF16 output.");
IREE_FLAG(float, atol, 0.0001220703125f, "Absolute error tolerance.");
IREE_FLAG(float, rtol, 0.0078125f, "Relative error tolerance.");

typedef struct component_check_t {
  // Shared device/queues outliving all accepted work and borrowed payloads.
  loom_serve_device_t* owner;
  // Cold source compiler, unused during execution.
  loom_serve_jit_t* jit;
  // Compiled command reflection and executable ownership.
  loom_serve_jit_stage_t* stage;
  // Retained, reusable commands after parameter loading.
  iree_hal_command_buffer_t* command;
  // Owned fixed-root references, populated even on partial loading failure.
  iree_hal_buffer_t** weights;
  // Number of fixed-root slots allocated in weights.
  iree_host_size_t weight_count;
  // Input files followed by the reference output; retained through transfers.
  iree_io_file_contents_t** files;
  // Number of file slots, including partial initialization.
  iree_host_size_t file_count;
  // Input buffers followed by output and optional reflected workspace.
  iree_hal_buffer_binding_t* bindings;
  // Number of allocated binding slots, including partial initialization.
  iree_host_size_t binding_count;
  // Reusable upload descriptors borrowing file storage.
  iree_hal_transfer_operation_t* uploads;
  // Output readback, retained until accepted feedback retires.
  uint16_t* output;
} component_check_t;

static iree_status_t component_compile(component_check_t* check,
                                       iree_allocator_t allocator) {
  const iree_flag_string_list_t values = FLAG_config_list();
  loomc_config_binding_t* bindings = NULL;
  if (values.count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, values.count, sizeof(*bindings), (void**)&bindings));
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < values.count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t key, value;
    if (iree_string_view_split(values.values[i], '=', &key, &value) < 0 ||
        iree_string_view_is_empty(key) || iree_string_view_is_empty(value)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "config must be key=value");
    } else {
      bindings[i].key = loomc_make_string_view(key.data, key.size);
      bindings[i].value = loomc_make_string_view(value.data, value.size);
    }
  }
  if (iree_status_is_ok(status)) {
    const loomc_config_options_t config = {
        bindings, values.count, {0}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
    status = loom_serve_jit_compile(
        check->jit, iree_make_cstring_view(FLAG_root), &config, &check->stage);
  }
  iree_allocator_free(allocator, bindings);
  return status;
}

static iree_status_t component_initialize(component_check_t* check,
                                          iree_allocator_t allocator) {
  if (!isfinite(FLAG_atol) || !isfinite(FLAG_rtol) || FLAG_atol < 0 ||
      FLAG_rtol < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tolerances must be finite and nonnegative");
  }
  if (FLAG_weights_list().count && !FLAG_weight_policy[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--weights requires --weight_policy");
  }
  const iree_flag_string_list_t inputs = FLAG_input_list();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, inputs.count + 1,
                                                   sizeof(*check->files),
                                                   (void**)&check->files));
  check->file_count = inputs.count + 1;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < check->file_count && iree_status_is_ok(status); ++i) {
    const iree_string_view_t path = i < inputs.count
                                        ? inputs.values[i]
                                        : iree_make_cstring_view(FLAG_expected);
    status = iree_io_file_contents_read(path, allocator, &check->files[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  const iree_host_size_t output_bytes =
      check->files[inputs.count]->const_buffer.data_length;
  if (!output_bytes || output_bytes % sizeof(uint16_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected output must be nonempty BF16 data");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, output_bytes, (void**)&check->output));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create(IREE_SV("amdgpu"), allocator, &check->owner));
  iree_hal_device_t* device = loom_serve_device_handle(check->owner);
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(check->owner);
  IREE_RETURN_IF_ERROR(loom_serve_jit_create(device, dispatch,
                                             iree_make_cstring_view(FLAG_model),
                                             NULL, allocator, &check->jit));
  IREE_RETURN_IF_ERROR(component_compile(check, allocator));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(check->stage);
  const bool workspace =
      program->requirements.transient.binding_index != UINT32_MAX;
  if (program->requirements.rebindable_binding_count !=
          check->file_count + workspace ||
      (workspace &&
       program->requirements.transient.binding_index != check->file_count) ||
      program->requirements.launch_counts.binding_index != UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "component requires ordered input/output bindings "
                            "and optional final workspace");
  }
  if (program->requirements.fixed_buffer_count) {
    const iree_flag_string_list_t checkpoints = FLAG_weights_list();
    if (checkpoints.count != program->parameter_roots.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "component requires one checkpoint path per "
                              "parameter root (%u roots, %zu paths)",
                              program->parameter_roots.count,
                              checkpoints.count);
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, program->requirements.fixed_buffer_count,
        sizeof(*check->weights), (void**)&check->weights));
    check->weight_count = program->requirements.fixed_buffer_count;
    for (uint32_t r = 0;
         r < program->parameter_roots.count && iree_status_is_ok(status); ++r) {
      const loom_cmd_program_parameter_root_t parameter_root =
          loom_cmd_program_parameter_root_at(program, r);
      const loom_serve_weight_root_t root = {
          program, parameter_root,
          &check->weights[parameter_root.fixed_buffer_index]};
      status = loom_serve_weights_load(
          device, loom_serve_device_transfer_queue(check->owner), dispatch,
          check->jit, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, 0, 1, &root,
          checkpoints.values[r], iree_make_cstring_view(FLAG_weight_policy),
          allocator);
    }
    IREE_RETURN_IF_ERROR(status);
  }
  IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
      check->stage, iree_hal_queue_family(dispatch),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, check->weights, &check->command));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, program->requirements.rebindable_binding_count,
      sizeof(*check->bindings), (void**)&check->bindings));
  check->binding_count = program->requirements.rebindable_binding_count;
  if (inputs.count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, inputs.count,
                                                     sizeof(*check->uploads),
                                                     (void**)&check->uploads));
  }
  for (iree_host_size_t i = 0;
       i < check->binding_count && iree_status_is_ok(status); ++i) {
    const iree_device_size_t length =
        i < check->file_count
            ? check->files[i]->const_buffer.data_length
            : program->requirements.transient.required_byte_length;
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.min_alignment =
        i < check->file_count
            ? 256
            : program->requirements.transient.minimum_alignment;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, length,
        &check->bindings[i].buffer);
    check->bindings[i].length = length;
    if (i < inputs.count) {
      check->uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      check->uploads[i].upload.source = check->files[i]->const_buffer.data;
      check->uploads[i].upload.target_buffer = check->bindings[i].buffer;
      check->uploads[i].upload.length = length;
    }
  }
  return status;
}

static iree_status_t component_compare(component_check_t* check,
                                       int iteration) {
  const iree_const_byte_span_t reference =
      check->files[check->file_count - 1]->const_buffer;
  const uint16_t* expected = (const uint16_t*)reference.data;
  const iree_host_size_t count = reference.data_length / sizeof(uint16_t);
  iree_host_size_t different = 0;
  iree_host_size_t outside = 0;
  double squared_error = 0;
  double squared_expected = 0;
  float maximum = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const float actual_value = iree_math_bf16_to_f32(check->output[i]);
    const float expected_value = iree_math_bf16_to_f32(expected[i]);
    const float error = fabsf(actual_value - expected_value);
    different += check->output[i] != expected[i];
    if (!isfinite(actual_value) || !isfinite(expected_value) ||
        error > FLAG_atol + FLAG_rtol * fabsf(expected_value)) {
      if (outside < 4) {
        fprintf(stderr, "Mismatch[%zu]: actual=%g expected=%g\n", i,
                actual_value, expected_value);
      }
      ++outside;
    }
    maximum = fmaxf(maximum, error);
    squared_error += (double)error * error;
    squared_expected += (double)expected_value * expected_value;
  }
  printf(
      "{\"iteration\":%d,\"elements\":%zu,\"different\":%zu,"
      "\"outside_tolerance\":%zu,\"maximum_absolute_error\":%.9g,"
      "\"relative_l2\":%.9g}\n",
      iteration, count, different, outside, maximum,
      sqrt(squared_error / fmax(squared_expected, 1e-30)));
  if (outside) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "%zu component elements exceed tolerance", outside);
  }
  return iree_ok_status();
}

static iree_status_t component_run(component_check_t* check,
                                   iree_allocator_t allocator) {
  loom_serve_execution_t* execution = loom_serve_device_execution(check->owner);
  const iree_host_size_t inputs = check->file_count - 1;
  iree_status_t status = iree_ok_status();
  for (int iteration = 0; iteration < 2 && iree_status_is_ok(status);
       ++iteration) {
    uint64_t completion = 0;
    if (inputs) {
      status = loom_serve_execution_transfer(execution, inputs, check->uploads,
                                             &completion);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_execute(
          execution, check->command,
          (iree_hal_buffer_binding_table_t){check->binding_count,
                                            check->bindings},
          &completion);
    }
    iree_hal_transfer_operation_t download = {0};
    download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    download.download.source_buffer = check->bindings[inputs].buffer;
    download.download.target = check->output;
    download.download.length = check->bindings[inputs].length;
    if (iree_status_is_ok(status)) {
      status =
          loom_serve_execution_feedback(execution, 1, &download, &completion);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_feedback_wait(execution, completion);
    }
    if (iree_status_is_ok(status) && iteration == 0 && FLAG_actual[0]) {
      status = iree_io_file_contents_write(
          iree_make_cstring_view(FLAG_actual),
          iree_make_const_byte_span(check->output,
                                    check->bindings[inputs].length),
          allocator);
    }
    if (iree_status_is_ok(status)) {
      status = component_compare(check, iteration);
    }
  }
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const iree_allocator_t allocator = iree_allocator_system();
  component_check_t check = {0};
  iree_status_t status = component_initialize(&check, allocator);
  if (iree_status_is_ok(status)) {
    status = component_run(&check, allocator);
  }
  if (check.owner) {
    status = iree_status_join(
        status,
        loom_serve_execution_drain(loom_serve_device_execution(check.owner)));
  }
  iree_hal_command_buffer_release(check.command);
  loom_serve_jit_stage_destroy(check.stage);
  for (iree_host_size_t i = 0; i < check.weight_count; ++i) {
    iree_hal_buffer_release(check.weights[i]);
  }
  for (iree_host_size_t i = 0; i < check.binding_count; ++i) {
    iree_hal_buffer_release(check.bindings[i].buffer);
  }
  for (iree_host_size_t i = 0; i < check.file_count; ++i) {
    iree_io_file_contents_free(check.files[i]);
  }
  iree_allocator_free(allocator, check.weights);
  iree_allocator_free(allocator, check.bindings);
  iree_allocator_free(allocator, check.uploads);
  iree_allocator_free(allocator, check.files);
  iree_allocator_free(allocator, check.output);
  loom_serve_jit_destroy(check.jit);
  loom_serve_device_destroy(check.owner);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
