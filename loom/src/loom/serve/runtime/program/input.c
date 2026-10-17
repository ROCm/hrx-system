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

typedef struct loom_serve_input_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Descriptor borrowing the module-owned reference type.
  iree_vm_module_descriptor_t descriptor;
  // Allocator owning the module and per-call tokenizer scratch.
  iree_allocator_t host_allocator;
  // Canonical buffer type, borrowed from the VM environment.
  iree_vm_ref_types_t types;
  // Optional immutable tokenizer, borrowed from the model owner.
  const iree_tokenizer_t* tokenizer;
} loom_serve_input_module_t;

enum loom_serve_input_function_ordinal_e {
  INPUT_FUNCTION_ENCODE = 0,
  INPUT_FUNCTION_LOOKUP = 1,
  INPUT_FUNCTION_REQUIRE = 2,
  INPUT_FUNCTION_DECODE = 3,
};

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

static iree_status_t loom_serve_input_decode(loom_serve_input_module_t* module,
                                             const iree_vm_call_packet_t* call,
                                             iree_const_byte_span_t bytes) {
  const iree_tokenizer_decode_flags_t flags =
      (iree_tokenizer_decode_flags_t)iree_vm_call_value_argument_load(call, 0);
  if (flags & ~IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported input decoding flags 0x%08x", flags);
  }
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
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(module->tokenizer);
  const iree_host_size_t vocab_capacity = iree_tokenizer_vocab_capacity(vocab);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const uint32_t id = iree_unaligned_load_le_u32(bytes.data + i * 4);
    if (id >= vocab_capacity ||
        (iree_tokenizer_vocab_token_attrs(vocab, (int32_t)id) &
         IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "token ID %" PRIu32 " is not in the vocabulary", id);
    } else {
      tokens[i] = (int32_t)id;
    }
  }
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

static iree_status_t loom_serve_input_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  loom_serve_input_module_t* module = (loom_serve_input_module_t*)base;
  const iree_vm_call_packet_t* call = &params->call;
  if (params->function_ordinal == INPUT_FUNCTION_REQUIRE &&
      iree_vm_call_value_argument_load(call, 0)) {
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    return iree_ok_status();
  }
  iree_vm_ref_t argument = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, 0, &argument);
  if (!argument.object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input buffer must not be null");
  }
  iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)argument.object;
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      buffer, 0, iree_vm_buffer_length(buffer), &bytes));
  const iree_string_view_t text =
      iree_make_string_view((const char*)bytes.data, bytes.data_length);
  if (params->function_ordinal == INPUT_FUNCTION_REQUIRE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%.*s",
                            (int)iree_min(text.size, INT_MAX), text.data);
  }
  if (!module->tokenizer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "model has no tokenizer input capability");
  }
  if (params->function_ordinal == INPUT_FUNCTION_LOOKUP) {
    const int32_t id = iree_tokenizer_vocab_lookup(
        iree_tokenizer_vocab(module->tokenizer), text);
    iree_vm_call_value_result_store(call, 0, (uint32_t)id);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    return iree_ok_status();
  }
  if (params->function_ordinal == INPUT_FUNCTION_DECODE) {
    IREE_RETURN_IF_ERROR(loom_serve_input_decode(module, call, bytes));
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    return iree_ok_status();
  }
  const iree_tokenizer_encode_flags_t flags =
      (iree_tokenizer_encode_flags_t)iree_vm_call_value_argument_load(call, 0);
  if (flags & ~(IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS |
                IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported input encoding flags 0x%08x", flags);
  }
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
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  }
  iree_vm_buffer_release(output);
  return status;
}

static const iree_vm_module_signature_type_t input_encode_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_encode_results[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t input_lookup_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t input_lookup_results[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t input_require_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}, {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};

static const iree_vm_module_callable_type_declaration_t input_callables[] = {
    {.signature = {.arguments = {input_lookup_arguments, 1, 0, 1, 0},
                   .results = {input_lookup_results, 1, 1, 0, 0}}},
    {.signature = {.arguments = {input_require_arguments, 2, 1, 1, 0}}},
    {.signature = {.arguments = {input_encode_arguments, 3, 2, 1, 0},
                   .results = {input_encode_results, 2, 1, 1, 0}}},
};
static const iree_vm_module_export_declaration_t input_exports[] = {
    {.export_name = IREE_SVL("decode"),
     .callable_type_ordinal = 2,
     .function_ordinal = INPUT_FUNCTION_DECODE},
    {.export_name = IREE_SVL("encode"),
     .callable_type_ordinal = 2,
     .function_ordinal = INPUT_FUNCTION_ENCODE},
    {.export_name = IREE_SVL("lookup"),
     .callable_type_ordinal = 0,
     .function_ordinal = INPUT_FUNCTION_LOOKUP},
    {.export_name = IREE_SVL("require"),
     .callable_type_ordinal = 1,
     .function_ordinal = INPUT_FUNCTION_REQUIRE},
};

static void loom_serve_input_module_destroy(iree_vm_module_t* base) {
  loom_serve_input_module_t* module = (loom_serve_input_module_t*)base;
  iree_allocator_free(module->host_allocator, module);
}
static void loom_serve_input_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = input_exports[ordinal];
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
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("input"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->types.buffer, 1},
      .counts = {.function_count = 4,
                 .callable_type_count = 3,
                 .export_count = IREE_ARRAYSIZE(input_exports),
                 .callable_fields = {.value_count = 5, .ref_count = 4}},
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
