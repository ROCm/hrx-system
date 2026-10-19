// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/control.h"

#include <string.h>

#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/json.h"
#include "loom/serve/service/text/epoch_plan.h"

iree_status_t loom_serve_text_control_initialize(
    const loom_serve_text_source_t* source, loom_serve_execution_t* execution,
    const loom_serve_text_state_t* state, iree_host_size_t shape_count,
    const loom_serve_packing_shape_t* shapes, iree_host_size_t stage_count,
    const loom_serve_stage_t* stages, iree_allocator_t host_allocator,
    loom_serve_text_control_t* out_control) {
  loom_serve_text_control_t* control = out_control;
  memset(control, 0, sizeof(*control));
  // Both dense padding and semantic packed inputs belong to the single model
  // invocation. Only the final prediction persists independently in each row.
  iree_host_size_t input_length = 0;
  for (iree_host_size_t i = 0; i < shape_count; ++i) {
    input_length =
        iree_max(input_length, 2 * shapes[i].token_capacity * sizeof(int32_t));
  }
  const bool packed =
      state->cache.capacity ||
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT);
  if (packed &&
      source->prefill_capacity > input_length / (2 * sizeof(int32_t))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source prefill capacity exceeds the epoch catalog");
  }
  for (iree_host_size_t i = 0; i < state->row_count; ++i) {
    const iree_device_size_t length =
        iree_hal_buffer_byte_length(state->rows[i].buffers[TEXT_TOKENS]);
    const iree_host_size_t minimum =
        (packed ? 1 : source->prefill_capacity) * sizeof(int32_t);
    if (length < minimum || length > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source row %zu has an invalid token extent", i);
    }
    if (!packed) {
      input_length = iree_max(input_length, (iree_host_size_t)length);
    }
  }
  const iree_host_size_t lengths[TEXT_HOST_BUFFER_COUNT] = {
      2 * LOOM_SERVE_TEXT_ROW_CAPACITY * TEXT_PLAN_FIELD_COUNT *
          sizeof(int32_t),
      input_length,
      shape_count ? iree_hal_buffer_byte_length(state->epoch.buffers[1]) : 0,
      shape_count ? iree_hal_buffer_byte_length(state->epoch.buffers[4]) : 0,
      shape_count ? iree_hal_buffer_byte_length(state->epoch.buffers[5]) : 0,
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT)
          ? state->storage.feedback_split
          : 0,
      iree_any_bit_set(state->flags, LOOM_SERVE_TEXT_STATE_FLAG_DRAFT)
          ? iree_hal_buffer_byte_length(state->draft.next_results)
          : 0,
      LOOM_SERVE_TEXT_ROW_CAPACITY * TEXT_PROGRESS_FIELD_COUNT *
          sizeof(int32_t),
      LOOM_SERVE_TEXT_ROW_CAPACITY * 4 * sizeof(int64_t),
      shape_count ? state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS]
                        .data_length
                  : 0,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < TEXT_HOST_BUFFER_COUNT && iree_status_is_ok(status); ++i) {
    status = iree_vm_buffer_create(lengths[i], iree_alignof(int64_t),
                                   host_allocator, &control->host.buffers[i]);
    if (iree_status_is_ok(status)) {
      status = iree_vm_buffer_map_write(control->host.buffers[i], 0, lengths[i],
                                        &control->host.bytes[i]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (lengths[TEXT_HOST_STATE_ORIGINS]) {
    memcpy(control->host.bytes[TEXT_HOST_STATE_ORIGINS].data,
           state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS].data,
           lengths[TEXT_HOST_STATE_ORIGINS]);
  }
  const iree_byte_span_t feedback[] = {
      control->host.bytes[TEXT_HOST_OUTPUTS],
      control->host.bytes[TEXT_HOST_RESULTS],
      control->host.bytes[TEXT_HOST_NEXT_RESULTS],
  };
  const loom_serve_module_options_t options = {
      .binding_capacity = 7,
      .stages = {stage_count, stages},
      .feedback = {IREE_ARRAYSIZE(feedback), feedback},
  };
  status = loom_serve_module_create(&source->types, execution, options,
                                    host_allocator, &control->native_module);
  IREE_RETURN_IF_ERROR(status);
  iree_vm_module_t* libraries[] = {control->native_module, NULL, NULL, NULL};
  status = loom_serve_input_module_create(
      source->environment, source->tokenizer, host_allocator, &libraries[1]);
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(source->environment, host_allocator,
                                           &libraries[2]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_tools_module_create(
        source->environment, host_allocator, &libraries[3]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_link(source->module,
                                     iree_vm_module_span_from_array(libraries),
                                     host_allocator, &control->program);
  }
  iree_vm_module_release(libraries[3]);
  iree_vm_module_release(libraries[2]);
  iree_vm_module_release(libraries[1]);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_policy_initialize(
      source->environment, control->program, source->options,
      &control->chat_policy));
  const iree_vm_signature_type_t buffer = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_REF,
      .value.ref = source->vm_types.buffer};
  const iree_vm_signature_type_t device_buffer = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_REF,
      .value.ref = source->types.buffer};
  const iree_vm_signature_type_t i32 = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR,
      .value.scalar = IREE_VM_SCALAR_TYPE_I32};
  const iree_vm_signature_type_t i64 = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR,
      .value.scalar = IREE_VM_SCALAR_TYPE_I64};
  const iree_vm_signature_type_t step_arguments[] = {
      buffer,        i32,           i32,           i32,
      device_buffer, device_buffer, device_buffer, device_buffer,
      device_buffer, device_buffer};
  const iree_vm_signature_type_t step_results[] = {i64, i32};
  const iree_vm_signature_type_t encode_arguments[] = {
      buffer, buffer, buffer, buffer, i32, i32,    i32,
      i32,    i32,    i32,    i32,    i32, buffer, buffer};
  const iree_vm_signature_type_t publish_arguments[] = {
      buffer, buffer, i32,    i32,    i32,    i32,
      i32,    i32,    buffer, buffer, buffer, buffer};
  const iree_vm_signature_type_t state_arguments[] = {buffer, i32, buffer};
  const iree_vm_signature_type_t epoch_arguments[] = {
      buffer,        i32,           i32,           i32,           i32,
      i32,           i32,           i32,           i32,           i32,
      i32,           device_buffer, device_buffer, device_buffer, device_buffer,
      device_buffer, device_buffer, device_buffer, device_buffer, device_buffer,
      device_buffer, device_buffer, device_buffer};
  const loom_serve_program_binding_t bindings[] = {
      {IREE_SVL("step"),
       {step_arguments, IREE_ARRAYSIZE(step_arguments)},
       {step_results, IREE_ARRAYSIZE(step_results)},
       &control->step},
      {IREE_SVL("encode_epoch"),
       {encode_arguments, IREE_ARRAYSIZE(encode_arguments)},
       {&i32, 1},
       &control->encode_epoch},
      {IREE_SVL("publish_epoch"),
       {publish_arguments, IREE_ARRAYSIZE(publish_arguments)},
       {NULL, 0},
       &control->publish_epoch},
      {IREE_SVL("encode_state"),
       {state_arguments, IREE_ARRAYSIZE(state_arguments)},
       {NULL, 0},
       &control->encode_state},
      {IREE_SVL("bind_request"),
       {state_arguments, IREE_ARRAYSIZE(state_arguments)},
       {NULL, 0},
       &control->bind_request},
      {IREE_SVL("epoch"),
       {epoch_arguments, IREE_ARRAYSIZE(epoch_arguments)},
       {&i64, 1},
       &control->epoch_step},
  };
  return loom_serve_program_bind(control->program, IREE_ARRAYSIZE(bindings),
                                 bindings);
}

void loom_serve_text_control_unlink(loom_serve_text_control_t* control) {
  loom_serve_text_chat_policy_deinitialize(&control->chat_policy);
  loom_serve_program_free(control->program);
  control->program = NULL;
  iree_vm_module_release(control->native_module);
  control->native_module = NULL;
}

void loom_serve_text_control_deinitialize(loom_serve_text_control_t* control) {
  for (iree_host_size_t i = 0; i < TEXT_HOST_BUFFER_COUNT; ++i) {
    iree_vm_buffer_release(control->host.buffers[i]);
  }
  memset(control, 0, sizeof(*control));
}
