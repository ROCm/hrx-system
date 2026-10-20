// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/input.h"

#include <limits.h>
#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/vm/buffer.h"
#include "iree/vm/reflection.h"
#include "loom/serve/runtime/program/input_codec.h"

typedef struct loom_serve_input_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Descriptor borrowing module-owned reference handle storage.
  iree_vm_module_descriptor_t descriptor;
  // Allocator owning the module and per-call tokenizer scratch.
  iree_allocator_t host_allocator;
  // Canonical buffer type, borrowed from the VM environment.
  iree_vm_ref_types_t types;
  // Module-local buffer, decoder, and encoder reference type ordinals.
  iree_vm_ref_type_t ref_types[3];
  // Optional immutable tokenizer, borrowed from the model owner.
  const iree_tokenizer_t* tokenizer;
} loom_serve_input_module_t;

// Native signatures establish exact reference types; null remains a source
// input error. No tokenizer state advances until all memory ranges are mapped.
static iree_status_t loom_serve_input_object_argument(
    const iree_vm_call_packet_t* call, iree_host_size_t ordinal,
    void** out_object) {
  iree_vm_ref_t argument = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, ordinal, &argument);
  if (!argument.object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input reference must not be null");
  }
  *out_object = argument.object;
  return iree_ok_status();
}

static iree_status_t loom_serve_input_size_argument(
    const iree_vm_call_packet_t* call, iree_host_size_t ordinal,
    iree_host_size_t unit_size, iree_host_size_t* out_size) {
  const uint64_t value = iree_vm_call_value_argument_load(call, ordinal);
  if (value > INT64_MAX || value > SIZE_MAX / unit_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "input byte range or capacity overflow");
  }
  *out_size = (iree_host_size_t)value * unit_size;
  return iree_ok_status();
}

static iree_status_t loom_serve_input_read_argument(
    const iree_vm_call_packet_t* call, iree_const_byte_span_t* out_bytes) {
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_object_argument(call, 0, (void**)&buffer));
  return iree_vm_buffer_map_read(buffer, 0, iree_vm_buffer_length(buffer),
                                 out_bytes);
}

static iree_status_t loom_serve_input_read_range(
    const iree_vm_call_packet_t* call, iree_host_size_t unit_size,
    iree_const_byte_span_t* out_bytes) {
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_object_argument(call, 1, (void**)&buffer));
  iree_host_size_t offset = 0;
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_serve_input_size_argument(call, 0, 1, &offset));
  IREE_RETURN_IF_ERROR(
      loom_serve_input_size_argument(call, 1, unit_size, &length));
  return iree_vm_buffer_map_read(buffer, offset, length, out_bytes);
}

static iree_status_t loom_serve_input_write_range(
    const iree_vm_call_packet_t* call, iree_host_size_t reference_ordinal,
    iree_host_size_t value_ordinal, iree_host_size_t unit_size,
    iree_byte_span_t* out_bytes) {
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_input_object_argument(call, reference_ordinal,
                                                        (void**)&buffer));
  iree_host_size_t offset = 0;
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_size_argument(call, value_ordinal, 1, &offset));
  IREE_RETURN_IF_ERROR(loom_serve_input_size_argument(call, value_ordinal + 1,
                                                      unit_size, &length));
  return iree_vm_buffer_map_write(buffer, offset, length, out_bytes);
}

// Pull only the requested prefix; all stream phases share destination credit.
static iree_status_t loom_serve_input_encode_prefix(
    const iree_tokenizer_t* tokenizer, iree_string_view_t text,
    iree_tokenizer_encode_flags_t flags, iree_host_size_t capacity,
    iree_allocator_t host_allocator, int32_t* out_tokens,
    iree_host_size_t* out_count) {
  *out_count = 0;
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  const iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_oneshot_size(text.size);
  if (!transform_size || state_size > SIZE_MAX - transform_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "prompt tokenizer storage size overflow");
  }
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, state_size + transform_size, (void**)&storage));
  iree_tokenizer_encode_state_t* state = NULL;
  iree_status_t status = iree_tokenizer_encode_state_initialize(
      tokenizer, iree_make_byte_span(storage, state_size),
      iree_make_byte_span(storage + state_size, transform_size),
      (iree_tokenizer_offset_run_list_t){0},
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START | flags, host_allocator,
      &state);
  iree_host_size_t count = 0;
  while (iree_status_is_ok(status) && text.size && count < capacity) {
    iree_host_size_t consumed = 0;
    iree_host_size_t produced = 0;
    status = iree_tokenizer_encode_state_feed(
        state, text,
        iree_tokenizer_make_token_output(out_tokens + count, NULL, NULL,
                                         capacity - count),
        &consumed, &produced);
    text = iree_string_view_substr(text, consumed, IREE_STRING_VIEW_NPOS);
    count += produced;
  }
  if (iree_status_is_ok(status) && count < capacity) {
    iree_host_size_t produced = 0;
    status = iree_tokenizer_encode_state_finalize(
        state,
        iree_tokenizer_make_token_output(out_tokens + count, NULL, NULL,
                                         capacity - count),
        &produced);
    count += produced;
  }
  if (state) {
    iree_tokenizer_encode_state_deinitialize(state);
  }
  iree_allocator_free(host_allocator, storage);
  if (iree_status_is_ok(status)) {
    *out_count = count;
  }
  return status;
}

static iree_status_t loom_serve_input_decode(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_serve_input_read_argument(call, &bytes));
  const iree_tokenizer_decode_flags_t flags =
      (iree_tokenizer_decode_flags_t)iree_vm_call_value_argument_load(call, 0);
  IREE_RETURN_IF_ERROR(loom_serve_input_validate_decode_flags(flags));
  if (bytes.data_length % sizeof(int32_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "token input must contain whole little-endian I32 IDs");
  }
  const uint64_t capacity = iree_vm_call_value_argument_load(call, 1);
  if (capacity > INT64_MAX || capacity > SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "decoded text byte capacity overflow");
  }
  const iree_host_size_t count = bytes.data_length / sizeof(int32_t);
  int32_t* tokens = NULL;
  if (count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        module->host_allocator, bytes.data_length, (void**)&tokens));
  }
  iree_status_t status =
      loom_serve_input_read_token_ids(module->tokenizer, bytes, tokens);
  iree_vm_buffer_t* output = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_create((iree_host_size_t)capacity, 1,
                                   module->host_allocator, &output);
  }
  iree_byte_span_t storage = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(output, 0, (iree_host_size_t)capacity,
                                      &storage);
  }
  iree_host_size_t length = 0;
  // Empty VM mappings have no pointer; the tokenizer accepts a non-null
  // zero-capacity span and still verifies that decoding produces no bytes.
  char empty_output = 0;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_decode(
        module->tokenizer, (iree_tokenizer_token_id_list_t){count, tokens},
        flags,
        iree_make_mutable_string_view(
            storage.data ? (char*)storage.data : &empty_output,
            storage.data_length),
        module->host_allocator, &length);
  }
  iree_allocator_free(module->host_allocator, tokens);
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t result =
        iree_vm_buffer_ref_from_ptr_move(&module->types, &output);
    iree_vm_call_ref_result_store_move(call, 0, &result);
    iree_vm_call_value_result_store(call, 0, length);
  }
  iree_vm_buffer_release(output);
  return status;
}

static iree_status_t loom_serve_input_encode(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_serve_input_read_argument(call, &bytes));
  const iree_string_view_t text =
      iree_make_string_view((const char*)bytes.data, bytes.data_length);
  const iree_tokenizer_encode_flags_t flags =
      (iree_tokenizer_encode_flags_t)iree_vm_call_value_argument_load(call, 0);
  IREE_RETURN_IF_ERROR(loom_serve_input_validate_encode_flags(flags));
  if (!iree_unicode_utf8_validate(text)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tokenizer input must be complete UTF-8");
  }
  const uint64_t capacity = iree_vm_call_value_argument_load(call, 1);
  if (capacity > SIZE_MAX / sizeof(int32_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "token output byte capacity overflow");
  }
  iree_vm_buffer_t* output = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      (iree_host_size_t)capacity * sizeof(int32_t), iree_alignof(int32_t),
      module->host_allocator, &output));
  iree_byte_span_t storage = iree_byte_span_empty();
  iree_status_t status = iree_vm_buffer_map_write(
      output, 0, iree_vm_buffer_length(output), &storage);
  iree_host_size_t count = 0;
  if (iree_status_is_ok(status) && capacity) {
    status = loom_serve_input_encode_prefix(
        module->tokenizer, text, flags, (iree_host_size_t)capacity,
        module->host_allocator, (int32_t*)storage.data, &count);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      const int32_t id = ((int32_t*)storage.data)[i];
      iree_unaligned_store_le_u32(storage.data + i * 4, (uint32_t)id);
    }
    iree_vm_ref_t result =
        iree_vm_buffer_ref_from_ptr_move(&module->types, &output);
    iree_vm_call_ref_result_store_move(call, 0, &result);
    iree_vm_call_value_result_store(call, 0, count);
  }
  iree_vm_buffer_release(output);
  return status;
}

static iree_status_t loom_serve_input_lookup(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_serve_input_read_argument(call, &bytes));
  const int32_t id = iree_tokenizer_vocab_lookup(
      iree_tokenizer_vocab(module->tokenizer),
      iree_make_string_view((const char*)bytes.data, bytes.data_length));
  iree_vm_call_value_result_store(call, 0, (uint32_t)id);
  return iree_ok_status();
}

static iree_status_t loom_serve_input_require(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  if (iree_vm_call_value_argument_load(call, 0)) {
    return iree_ok_status();
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_serve_input_read_argument(call, &bytes));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%.*s",
                          (int)iree_min(bytes.data_length, INT_MAX),
                          bytes.data);
}

static iree_status_t loom_serve_input_call_encoder_create(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  iree_host_size_t transform_capacity = 0;
  iree_host_size_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_size_argument(call, 1, 1, &transform_capacity));
  IREE_RETURN_IF_ERROR(
      loom_serve_input_size_argument(call, 2, 1, &token_capacity));
  iree_vm_ref_t result = iree_vm_ref_null();
  IREE_RETURN_IF_ERROR(loom_serve_input_encoder_create(
      module->tokenizer,
      (iree_tokenizer_encode_flags_t)iree_vm_call_value_argument_load(call, 0),
      transform_capacity, token_capacity, module->host_allocator, &result));
  iree_vm_call_ref_result_store_move(call, 0, &result);
  return iree_ok_status();
}

static iree_status_t loom_serve_input_call_encoder_reset(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  loom_serve_input_encoder_t* encoder = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_object_argument(call, 0, (void**)&encoder));
  return loom_serve_input_encoder_reset(
      encoder,
      (iree_tokenizer_encode_flags_t)iree_vm_call_value_argument_load(call, 0));
}

static iree_status_t loom_serve_input_call_encoder_feed(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  loom_serve_input_encoder_t* encoder = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_object_argument(call, 0, (void**)&encoder));
  iree_const_byte_span_t input = iree_const_byte_span_empty();
  iree_byte_span_t output = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_serve_input_read_range(call, 1, &input));
  IREE_RETURN_IF_ERROR(
      loom_serve_input_write_range(call, 2, 2, sizeof(int32_t), &output));
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_RETURN_IF_ERROR(loom_serve_input_encoder_feed(
      encoder,
      iree_make_string_view((const char*)input.data, input.data_length), output,
      &consumed, &written));
  iree_vm_call_value_result_store(call, 0, consumed);
  iree_vm_call_value_result_store(call, 1, written);
  return iree_ok_status();
}

static iree_status_t loom_serve_input_call_encoder_finalize(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  loom_serve_input_encoder_t* encoder = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_object_argument(call, 0, (void**)&encoder));
  iree_byte_span_t output = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      loom_serve_input_write_range(call, 1, 0, sizeof(int32_t), &output));
  iree_host_size_t written = 0;
  bool complete = false;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_encoder_finalize(encoder, output, &written, &complete));
  iree_vm_call_value_result_store(call, 0, written);
  iree_vm_call_value_result_store(call, 1, complete);
  return iree_ok_status();
}

static iree_status_t loom_serve_input_call_decoder_create(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  iree_host_size_t token_capacity = 0;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_size_argument(call, 1, 1, &token_capacity));
  iree_vm_ref_t result = iree_vm_ref_null();
  IREE_RETURN_IF_ERROR(loom_serve_input_decoder_create(
      module->tokenizer,
      (iree_tokenizer_decode_flags_t)iree_vm_call_value_argument_load(call, 0),
      token_capacity, module->host_allocator, &result));
  iree_vm_call_ref_result_store_move(call, 0, &result);
  return iree_ok_status();
}

static iree_status_t loom_serve_input_call_decoder_reset(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  loom_serve_input_decoder_t* decoder = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_object_argument(call, 0, (void**)&decoder));
  return loom_serve_input_decoder_reset(
      decoder,
      (iree_tokenizer_decode_flags_t)iree_vm_call_value_argument_load(call, 0));
}

static iree_status_t loom_serve_input_call_decoder_feed(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  loom_serve_input_decoder_t* decoder = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_object_argument(call, 0, (void**)&decoder));
  iree_const_byte_span_t input = iree_const_byte_span_empty();
  iree_byte_span_t output = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      loom_serve_input_read_range(call, sizeof(int32_t), &input));
  IREE_RETURN_IF_ERROR(loom_serve_input_write_range(call, 2, 2, 1, &output));
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_RETURN_IF_ERROR(loom_serve_input_decoder_feed(decoder, input, output,
                                                     &consumed, &written));
  iree_vm_call_value_result_store(call, 0, consumed);
  iree_vm_call_value_result_store(call, 1, written);
  return iree_ok_status();
}

static iree_status_t loom_serve_input_call_decoder_finalize(
    loom_serve_input_module_t* module, const iree_vm_call_packet_t* call) {
  loom_serve_input_decoder_t* decoder = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_object_argument(call, 0, (void**)&decoder));
  iree_byte_span_t output = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_serve_input_write_range(call, 1, 0, 1, &output));
  iree_host_size_t written = 0;
  bool complete = false;
  IREE_RETURN_IF_ERROR(
      loom_serve_input_decoder_finalize(decoder, output, &written, &complete));
  iree_vm_call_value_result_store(call, 0, written);
  iree_vm_call_value_result_store(call, 1, complete);
  return iree_ok_status();
}

static const iree_vm_module_signature_type_t input_lookup_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t input_lookup_results[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t input_decoder_create_arguments[] =
    {{IREE_VM_SCALAR_TYPE_I32, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_decoder_create_results[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1}};
static const iree_vm_module_signature_type_t input_require_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}, {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t input_decoder_reset_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1}, {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t input_encoder_reset_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2}, {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t input_encoder_create_arguments[] =
    {{IREE_VM_SCALAR_TYPE_I32, 0},
     {IREE_VM_SCALAR_TYPE_I64, 0},
     {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_encoder_create_results[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2}};
static const iree_vm_module_signature_type_t input_encode_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_encode_results[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t
    input_decoder_finalize_arguments[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_decoder_finalize_results[] =
    {{IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t
    input_encoder_finalize_arguments[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2},
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_encoder_finalize_results[] =
    {{IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t input_decoder_feed_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_decoder_feed_results[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_encoder_feed_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_encoder_feed_results[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};

// VM callable ordinals are structural signature order, independent of exports.
static const iree_vm_module_callable_type_declaration_t input_callables[] = {
    {.signature = {.arguments = {input_lookup_arguments, 1, 0, 1, 0},
                   .results = {input_lookup_results, 1, 1, 0, 0}}},
    {.signature = {.arguments = {input_decoder_create_arguments, 2, 2, 0, 0},
                   .results = {input_decoder_create_results, 1, 0, 1, 0}}},
    {.signature = {.arguments = {input_require_arguments, 2, 1, 1, 0}}},
    {.signature = {.arguments = {input_decoder_reset_arguments, 2, 1, 1, 0}}},
    {.signature = {.arguments = {input_encoder_reset_arguments, 2, 1, 1, 0}}},
    {.signature = {.arguments = {input_encoder_create_arguments, 3, 3, 0, 0},
                   .results = {input_encoder_create_results, 1, 0, 1, 0}}},
    {.signature = {.arguments = {input_encode_arguments, 3, 2, 1, 0},
                   .results = {input_encode_results, 2, 1, 1, 0}}},
    {.signature = {.arguments = {input_decoder_finalize_arguments, 4, 2, 2, 0},
                   .results = {input_decoder_finalize_results, 2, 2, 0, 0}}},
    {.signature = {.arguments = {input_encoder_finalize_arguments, 4, 2, 2, 0},
                   .results = {input_encoder_finalize_results, 2, 2, 0, 0}}},
    {.signature = {.arguments = {input_decoder_feed_arguments, 7, 4, 3, 0},
                   .results = {input_decoder_feed_results, 2, 2, 0, 0}}},
    {.signature = {.arguments = {input_encoder_feed_arguments, 7, 4, 3, 0},
                   .results = {input_encoder_feed_results, 2, 2, 0, 0}}},
};

typedef struct loom_serve_input_function_t {
  // Export name; rows are sorted lexically for VM lookup.
  iree_string_view_t name;
  // Ordinal of the canonical signature in input_callables.
  uint32_t callable_type;
  // Synchronous implementation; arguments are borrowed for this call.
  iree_status_t (*call)(loom_serve_input_module_t* module,
                        const iree_vm_call_packet_t* call);
  // True when the operation needs the module's immutable tokenizer.
  bool requires_tokenizer;
} loom_serve_input_function_t;

static const loom_serve_input_function_t input_functions[] = {
    {IREE_SVL("decode"), 6, loom_serve_input_decode, true},
    {IREE_SVL("decoder_create"), 1, loom_serve_input_call_decoder_create, true},
    {IREE_SVL("decoder_feed"), 9, loom_serve_input_call_decoder_feed, false},
    {IREE_SVL("decoder_finalize"), 7, loom_serve_input_call_decoder_finalize,
     false},
    {IREE_SVL("decoder_reset"), 3, loom_serve_input_call_decoder_reset, false},
    {IREE_SVL("encode"), 6, loom_serve_input_encode, true},
    {IREE_SVL("encoder_create"), 5, loom_serve_input_call_encoder_create, true},
    {IREE_SVL("encoder_feed"), 10, loom_serve_input_call_encoder_feed, false},
    {IREE_SVL("encoder_finalize"), 8, loom_serve_input_call_encoder_finalize,
     false},
    {IREE_SVL("encoder_reset"), 4, loom_serve_input_call_encoder_reset, false},
    {IREE_SVL("lookup"), 0, loom_serve_input_lookup, true},
    {IREE_SVL("require"), 2, loom_serve_input_require, false},
};

static iree_status_t loom_serve_input_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  loom_serve_input_module_t* module = (loom_serve_input_module_t*)base;
  const loom_serve_input_function_t* function =
      &input_functions[params->function_ordinal];
  if (function->requires_tokenizer && !module->tokenizer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "model has no tokenizer input capability");
  }
  IREE_RETURN_IF_ERROR(function->call(module, &params->call));
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static void loom_serve_input_module_destroy(iree_vm_module_t* base) {
  loom_serve_input_module_t* module = (loom_serve_input_module_t*)base;
  iree_allocator_free(module->host_allocator, module);
}
static void loom_serve_input_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = (iree_vm_module_export_declaration_t){
      .export_name = input_functions[ordinal].name,
      .callable_type_ordinal = input_functions[ordinal].callable_type,
      .function_ordinal = ordinal,
  };
}
static void loom_serve_input_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = input_callables[ordinal];
}
static void loom_serve_input_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("input module has no imports");
}
static void loom_serve_input_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("input module has no imports");
}
static const iree_vm_module_vtable_t input_module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_serve_input_module_destroy,
    .function_start = loom_serve_input_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_serve_input_module_query_import_group,
    .query_import = loom_serve_input_module_query_import,
    .query_export = loom_serve_input_module_query_export,
    .query_callable_type = loom_serve_input_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_input_module_create(iree_vm_environment_t* environment,
                                             const iree_tokenizer_t* tokenizer,
                                             iree_allocator_t host_allocator,
                                             iree_vm_module_t** out_module) {
  *out_module = NULL;
  const iree_vm_ref_type_table_t* codec_table =
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("input"));
  if (!codec_table) {
    IREE_RETURN_IF_ERROR(iree_vm_environment_register_ref_type_table(
        environment, &loom_serve_input_codec_type_table));
  } else if (codec_table != &loom_serve_input_codec_type_table) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "input codec type namespace has another provider");
  }
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  loom_serve_input_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->types = types;
  module->tokenizer = tokenizer;
  module->ref_types[0] = types.buffer;
  module->ref_types[1] =
      iree_vm_ref_type_storage_at(loom_serve_input_codec_type_table.types,
                                  LOOM_SERVE_INPUT_CODEC_TYPE_DECODER);
  module->ref_types[2] =
      iree_vm_ref_type_storage_at(loom_serve_input_codec_type_table.types,
                                  LOOM_SERVE_INPUT_CODEC_TYPE_ENCODER);
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("input"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {module->ref_types, IREE_ARRAYSIZE(module->ref_types)},
      .counts = {.function_count = IREE_ARRAYSIZE(input_functions),
                 .callable_type_count = IREE_ARRAYSIZE(input_callables),
                 .export_count = IREE_ARRAYSIZE(input_functions),
                 .callable_fields = {.value_count = 32, .ref_count = 18}},
  };
  iree_status_t status = iree_vm_module_initialize(
      &input_module_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
