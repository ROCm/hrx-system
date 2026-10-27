// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/control.h"

#include <string.h>

#include "iree/vm/sync.h"
#include "loom/serve/runtime/parameter/module.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/json.h"
#include "loom/serve/service/text/epoch_plan.h"
#include "loom/serve/service/text/tool_xml.h"

iree_status_t loom_serve_text_control_initialize(
    const loom_serve_text_source_t* source, loom_serve_execution_t* execution,
    iree_hal_queue_t* transfer, const loom_serve_text_state_t* state,
    iree_host_size_t shape_count, const loom_serve_packing_shape_t* shapes,
    iree_host_size_t stage_count, const loom_serve_stage_t* stages,
    iree_allocator_t host_allocator, loom_serve_text_control_t* out_control) {
  loom_serve_text_control_t* control = out_control;
  memset(control, 0, sizeof(*control));
  // Both dense padding and semantic packed inputs belong to the single model
  // invocation. Only the final prediction persists independently in each row.
  iree_host_size_t input_length = 0;
  iree_host_size_t span_capacity = 1;
  for (iree_host_size_t i = 0; i < shape_count; ++i) {
    input_length =
        iree_max(input_length, 2 * shapes[i].token_capacity * sizeof(int32_t));
    span_capacity = iree_max(span_capacity, shapes[i].span_capacity);
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
        iree_hal_buffer_byte_length(state->rows[i]->buffers[TEXT_TOKENS]);
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
      2 * span_capacity * TEXT_PLAN_FIELD_COUNT * sizeof(int32_t),
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
      span_capacity * TEXT_PROGRESS_FIELD_COUNT * sizeof(int32_t),
      state->row_count * 4 * sizeof(int64_t),
      shape_count ? state->storage.bytes[LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS]
                        .data_length
                  : 0,
      span_capacity * sizeof(int32_t),
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
  iree_vm_module_t* libraries[] = {control->native_module, NULL, NULL, NULL,
                                   NULL};
  status = loom_serve_input_module_create(
      source->environment, loom_serve_tokenizer_value(source->tokenizer),
      host_allocator, &libraries[1]);
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(source->environment, host_allocator,
                                           &libraries[2]);
  }
  const bool generates = source->mode == LOOM_SERVE_TEXT_MODE_GENERATE;
  if (iree_status_is_ok(status)) {
    status = loom_serve_parameters_module_create(
        source->environment, &source->types, transfer,
        source->parameter_sources, host_allocator, &libraries[3]);
  }
  if (iree_status_is_ok(status) && generates) {
    status = loom_serve_text_tool_xml_module_create(
        source->environment, host_allocator, &libraries[4]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_link_service(
        source->module, loom_serve_preparation_card(source->preparation),
        loom_serve_preparation_component(source->preparation),
        iree_vm_module_span_from_ptr(libraries, generates ? 5 : 4),
        host_allocator, &control->program);
  }
  iree_vm_module_release(libraries[4]);
  iree_vm_module_release(libraries[3]);
  iree_vm_module_release(libraries[2]);
  iree_vm_module_release(libraries[1]);
  IREE_RETURN_IF_ERROR(status);
  if (generates) {
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_policy_initialize(
        source->environment, control->program, source->options,
        &control->chat_policy));
  }
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
  const iree_vm_signature_type_t encode_results[] = {i32, buffer, i64, i64};
  const iree_vm_signature_type_t publish_arguments[] = {
      buffer, buffer, i32,    i32,    i32,    i32,
      i32,    buffer, buffer, buffer, buffer, buffer};
  const iree_vm_signature_type_t state_arguments[] = {buffer, i32, buffer};
  const iree_vm_signature_type_t resize_arguments[] = {buffer, i64};
  const iree_vm_signature_type_t resize_results[] = {buffer, buffer, buffer,
                                                     buffer, buffer, buffer};
  const iree_vm_signature_type_t epoch_arguments[] = {
      buffer,        i32,           i32,           i32,           i32,
      i32,           i32,           i32,           i32,           i32,
      i32,           device_buffer, device_buffer, device_buffer, device_buffer,
      device_buffer, device_buffer, device_buffer, device_buffer, device_buffer,
      device_buffer, device_buffer, device_buffer};
  const loom_serve_program_binding_t bindings[] = {
      {IREE_SVL("encode_epoch"),
       {encode_arguments, IREE_ARRAYSIZE(encode_arguments)},
       {encode_results, IREE_ARRAYSIZE(encode_results)},
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
      {IREE_SVL("resize_state"),
       {resize_arguments, IREE_ARRAYSIZE(resize_arguments)},
       {resize_results, IREE_ARRAYSIZE(resize_results)},
       &control->resize_state},
      {IREE_SVL("epoch"),
       {epoch_arguments, IREE_ARRAYSIZE(epoch_arguments)},
       {&i64, 1},
       &control->epoch_step},
  };
  IREE_RETURN_IF_ERROR(loom_serve_program_bind_service(
      control->program, IREE_ARRAYSIZE(bindings), bindings));
  if (generates) {
    const loom_serve_program_binding_t step = {
        IREE_SVL("step"),
        {step_arguments, IREE_ARRAYSIZE(step_arguments)},
        {step_results, IREE_ARRAYSIZE(step_results)},
        &control->step};
    return loom_serve_program_bind_service(control->program, 1, &step);
  }
  iree_vm_function_t defaults = {0};
  const iree_vm_signature_type_t capture_arguments[] = {buffer, i32, i32,
                                                        buffer};
  const iree_vm_signature_type_t options_arguments[] = {buffer, buffer, i64};
  const loom_serve_program_binding_t encoding_bindings[] = {
      {IREE_SVL("capture_epoch"),
       {capture_arguments, IREE_ARRAYSIZE(capture_arguments)},
       {&buffer, 1},
       &control->capture_epoch},
      {IREE_SVL("request_defaults"), {&buffer, 1}, {&buffer, 1}, &defaults},
      {IREE_SVL("request_options"),
       {options_arguments, IREE_ARRAYSIZE(options_arguments)},
       {&buffer, 1},
       &control->request_options},
  };
  IREE_RETURN_IF_ERROR(loom_serve_program_bind_service(
      control->program, IREE_ARRAYSIZE(encoding_bindings), encoding_bindings));
  iree_vm_variant_t argument = iree_vm_buffer_variant_from_ptr_borrowed(
      &source->vm_types, source->options);
  iree_vm_variant_t result = {0};
  status = iree_vm_invoke(loom_serve_program_invocation(control->program),
                          defaults, iree_vm_variant_span_from_ptr(&argument, 1),
                          iree_vm_variant_span_from_ptr(&result, 1));
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&source->vm_types, &result,
                                                  &control->defaults);
  }
  if (iree_status_is_ok(status) && !control->defaults) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source request defaults must not be null");
  }
  iree_vm_variant_reset(&argument);
  iree_vm_variant_reset(&result);
  return status;
}

void loom_serve_text_control_unlink(loom_serve_text_control_t* control) {
  iree_vm_buffer_release(control->defaults);
  control->defaults = NULL;
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

iree_status_t loom_serve_text_control_request_options(
    loom_serve_text_control_t* control, const loom_serve_text_source_t* source,
    iree_string_view_t body, uint64_t request_id,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_options) {
  *out_options = NULL;
  iree_vm_buffer_t* input = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                           iree_make_const_byte_span(body.data, body.size), 1,
                           host_allocator, &input));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&source->vm_types,
                                               control->defaults),
      iree_vm_buffer_variant_from_ptr_move(&source->vm_types, &input),
      iree_vm_variant_from_i64((int64_t)request_id)};
  iree_vm_variant_t result = {0};
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(control->program), control->request_options,
      iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_ptr(&result, 1));
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&source->vm_types, &result,
                                                  out_options);
  }
  if (iree_status_is_ok(status) && !*out_options) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source request options must not be null");
  }
  iree_vm_variant_reset(&result);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

iree_status_t loom_serve_text_control_capture(
    loom_serve_text_control_t* control, const loom_serve_text_source_t* source,
    iree_host_size_t shape_index, iree_host_size_t span_count,
    iree_device_size_t residual_length, iree_allocator_t host_allocator,
    loom_serve_text_capture_t** out_capture) {
  *out_capture = NULL;
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&source->vm_types,
                                               source->control_state),
      iree_vm_variant_from_i32((int32_t)shape_index),
      iree_vm_variant_from_i32((int32_t)span_count),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &source->vm_types, control->host.buffers[TEXT_HOST_PLAN]),
  };
  iree_vm_variant_t result = {0};
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(control->program), control->capture_epoch,
      iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_ptr(&result, 1));
  iree_vm_buffer_t* plan = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_ptr_from_variant_move(&source->vm_types, &result, &plan);
  }
  if (iree_status_is_ok(status) && !plan) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source capture plan must not be null");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_capture_create(plan, span_count, residual_length,
                                            host_allocator, out_capture);
  }
  iree_vm_buffer_release(plan);
  iree_vm_variant_reset(&result);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}
