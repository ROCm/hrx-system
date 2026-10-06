// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/json.h"

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/vm/buffer.h"
#include "iree/vm/reflection.h"

typedef struct json_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Descriptor borrowing the environment's reference type.
  iree_vm_module_descriptor_t descriptor;
  // Allocator for the module and request-scope member tables.
  iree_allocator_t allocator;
  // Canonical buffer type, borrowed from the environment.
  iree_vm_ref_types_t types;
} json_module_t;

typedef struct json_members_t {
  // Complete input used as the origin for all recorded offsets.
  iree_string_view_t input;
  // Binary record builder with geometric capacity growth.
  iree_string_builder_t records;
} json_members_t;

static iree_status_t json_member(void* user_data, iree_string_view_t key,
                                 iree_json_value_type_t type,
                                 iree_string_view_t value) {
  json_members_t* members = user_data;
  const uint64_t fields[] = {
      key.data ? (uint64_t)(key.data - members->input.data) : 0,
      key.size,
      (uint64_t)(value.data - members->input.data),
      value.size,
      type,
  };
  uint8_t record[sizeof(fields)];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    iree_unaligned_store_le_u64(record + i * sizeof(uint64_t), fields[i]);
  }
  return iree_string_builder_append_string(
      &members->records,
      iree_make_string_view((const char*)record, sizeof(record)));
}

static iree_status_t json_element(void* user_data, iree_host_size_t index,
                                  iree_json_value_type_t type,
                                  iree_string_view_t value) {
  return json_member(user_data, iree_string_view_empty(), type, value);
}

static iree_status_t json_read_members(json_module_t* module,
                                       const iree_vm_call_packet_t* call,
                                       iree_vm_buffer_t* input) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_map_read(input, 0, iree_vm_buffer_length(input), &bytes));
  const iree_string_view_t text =
      iree_make_string_view((const char*)bytes.data, bytes.data_length);
  if (!iree_unicode_utf8_validate(text)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JSON input must be UTF-8");
  }
  iree_string_view_t cursor = text;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (!cursor.size || (cursor.data[0] != '{' && cursor.data[0] != '[')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JSON members require an object or array");
  }
  const iree_json_value_type_t type =
      iree_json_infer_value_type(cursor.data[0]);
  iree_string_view_t value;
  IREE_RETURN_IF_ERROR(iree_json_consume_value(&cursor, &value));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing JSON input");
  }
  json_members_t members = {.input = text};
  iree_string_builder_initialize(module->allocator, &members.records);
  iree_status_t status =
      type == IREE_JSON_VALUE_TYPE_OBJECT
          ? iree_json_enumerate_object_typed(value, json_member, &members)
          : iree_json_enumerate_array_typed(value, json_element, &members);
  iree_vm_buffer_t* output = NULL;
  const iree_string_view_t records = iree_string_builder_view(&members.records);
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(records.data, records.size),
        iree_alignof(int64_t), module->allocator, &output);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t result =
        iree_vm_buffer_ref_from_ptr_move(&module->types, &output);
    iree_vm_call_ref_result_store_move(call, 0, &result);
    iree_vm_call_value_result_store(call, 0, type);
    iree_vm_call_value_result_store(call, 1,
                                    records.size / (5 * sizeof(int64_t)));
  }
  iree_vm_buffer_release(output);
  iree_string_builder_deinitialize(&members.records);
  return status;
}

static iree_status_t json_unescape(const iree_vm_call_packet_t* call,
                                   iree_vm_buffer_t* input) {
  const uint64_t offset = iree_vm_call_value_argument_load(call, 0);
  const uint64_t length = iree_vm_call_value_argument_load(call, 1);
  const uint64_t target_offset = iree_vm_call_value_argument_load(call, 2);
  iree_vm_ref_t target_ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, 1, &target_ref);
  if (!target_ref.object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JSON target must not be null");
  }
  iree_vm_buffer_t* target = (iree_vm_buffer_t*)target_ref.object;
  if (offset > SIZE_MAX || length > SIZE_MAX ||
      target_offset > iree_vm_buffer_length(target)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "JSON string range exceeds host storage");
  }
  iree_const_byte_span_t source = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(input, offset, length, &source));
  iree_byte_span_t output = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_write(
      target, target_offset, iree_vm_buffer_length(target) - target_offset,
      &output));
  iree_host_size_t written = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(
      iree_make_string_view((const char*)source.data, source.data_length),
      output.data_length, (char*)output.data, &written));
  // A zero-length VM mapping is null, which the parser treats as a size query.
  // This export always decodes into storage, including an empty destination.
  if (written > output.data_length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "JSON target needs %" PRIhsz " bytes, has %" PRIhsz,
                            written, output.data_length);
  }
  iree_vm_call_value_result_store(call, 0, written);
  return iree_ok_status();
}

static iree_status_t json_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  json_module_t* module = (json_module_t*)base;
  const iree_vm_call_packet_t* call = &params->call;
  iree_vm_ref_t input_ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, 0, &input_ref);
  if (!input_ref.object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JSON input must not be null");
  }
  iree_vm_buffer_t* input = (iree_vm_buffer_t*)input_ref.object;
  IREE_RETURN_IF_ERROR(params->function_ordinal == 0
                           ? json_read_members(module, call, input)
                           : json_unescape(call, input));
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static const iree_vm_module_signature_type_t json_members_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t json_members_results[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t json_unescape_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t json_unescape_results[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_callable_type_declaration_t json_callables[] = {
    {.signature = {.arguments = {json_members_arguments, 1, 0, 1, 0},
                   .results = {json_members_results, 3, 2, 1, 0}}},
    {.signature = {.arguments = {json_unescape_arguments, 5, 3, 2, 0},
                   .results = {json_unescape_results, 1, 1, 0, 0}}},
};
static const iree_vm_module_export_declaration_t json_exports[] = {
    {.export_name = IREE_SVL("members"),
     .callable_type_ordinal = 0,
     .function_ordinal = 0},
    {.export_name = IREE_SVL("unescape"),
     .callable_type_ordinal = 1,
     .function_ordinal = 1},
};

static void json_module_destroy(iree_vm_module_t* base) {
  json_module_t* module = (json_module_t*)base;
  iree_allocator_free(module->allocator, module);
}
static void json_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = json_exports[ordinal];
}
static void json_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = json_callables[ordinal];
}
static void json_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("json module has no imports");
}
static void json_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("json module has no imports");
}
static const iree_vm_module_vtable_t json_module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = json_module_destroy,
    .function_start = json_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = json_module_query_import_group,
    .query_import = json_module_query_import,
    .query_export = json_module_query_export,
    .query_callable_type = json_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_json_module_create(iree_vm_environment_t* environment,
                                            iree_vm_module_t** out_module,
                                            iree_allocator_t host_allocator) {
  *out_module = NULL;
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  json_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->allocator = host_allocator;
  module->types = types;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("json"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->types.buffer, 1},
      .counts = {.function_count = 2,
                 .callable_type_count = 2,
                 .export_count = 2,
                 .callable_fields = {.value_count = 6, .ref_count = 4}},
  };
  iree_status_t status = iree_vm_module_initialize(
      &json_module_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
