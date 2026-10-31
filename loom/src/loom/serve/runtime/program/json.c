// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/json.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/base/printf.h"
#include "iree/vm/buffer.h"
#include "iree/vm/reflection.h"

typedef struct loom_serve_json_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Descriptor borrowing the environment's reference type.
  iree_vm_module_descriptor_t descriptor;
  // Allocator for the module and request-scope member tables.
  iree_allocator_t host_allocator;
  // Canonical buffer type, borrowed from the environment.
  iree_vm_ref_types_t types;
} loom_serve_json_module_t;

// Copies an already validated value without insignificant bytes. Validation
// establishes closing quotes/comments and escaped-byte bounds. The output is
// at most input.size bytes and preserves every significant byte verbatim.
static iree_host_size_t loom_serve_json_compact_copy(iree_string_view_t input,
                                                     char* output) {
  iree_host_size_t written = 0;
  for (iree_host_size_t i = 0; i < input.size;) {
    const char c = input.data[i++];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      continue;
    }
    if (c == '/') {
      if (input.data[i++] == '/') {
        while (i < input.size && input.data[i] != '\n') {
          ++i;
        }
      } else {
        while (input.data[i] != '*' || input.data[i + 1] != '/') {
          ++i;
        }
        i += 2;
      }
      continue;
    }
    output[written++] = c;
    if (c == '"') {
      while (input.data[i] != '"') {
        const char byte = input.data[i++];
        output[written++] = byte;
        if (byte == '\\') {
          output[written++] = input.data[i++];
        }
      }
      output[written++] = input.data[i++];
    }
  }
  return written;
}

static iree_status_t loom_serve_json_compact(loom_serve_json_module_t* module,
                                             const iree_vm_call_packet_t* call,
                                             iree_vm_buffer_t* input) {
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_map_read(input, 0, iree_vm_buffer_length(input), &bytes));
  iree_string_view_t cursor =
      iree_make_string_view((const char*)bytes.data, bytes.data_length);
  if (!iree_unicode_utf8_validate(cursor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JSON input must be UTF-8");
  }
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  iree_string_view_t document = cursor;
  iree_string_view_t value;
  IREE_RETURN_IF_ERROR(iree_json_consume_value(&cursor, &value));
  document.size -= cursor.size;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing JSON input");
  }
  iree_string_builder_t compact;
  iree_string_builder_initialize(module->host_allocator, &compact);
  char* data = NULL;
  iree_host_size_t capacity = 0;
  iree_status_t status = iree_string_builder_reserve_for_append(
      &compact, document.size, &data, &capacity);
  iree_vm_buffer_t* output = NULL;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t length =
        loom_serve_json_compact_copy(document, data);
    status = iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                  iree_make_const_byte_span(data, length), 1,
                                  module->host_allocator, &output);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t result =
        iree_vm_buffer_ref_from_ptr_move(&module->types, &output);
    iree_vm_call_ref_result_store_move(call, 0, &result);
  }
  iree_vm_buffer_release(output);
  iree_string_builder_deinitialize(&compact);
  return status;
}

typedef struct loom_serve_json_members_t {
  // Selected input range used to resolve parser-owned pointers.
  iree_string_view_t input;
  // Byte offset of input in the caller's complete source buffer.
  uint64_t offset;
  // Binary record builder with geometric capacity growth.
  iree_string_builder_t records;
} loom_serve_json_members_t;

static iree_status_t loom_serve_json_member(void* user_data,
                                            iree_string_view_t key,
                                            iree_json_value_type_t type,
                                            iree_string_view_t value) {
  loom_serve_json_members_t* members = user_data;
  const uint64_t fields[] = {
      key.data ? members->offset + (uint64_t)(key.data - members->input.data)
               : 0,
      key.size,
      members->offset + (uint64_t)(value.data - members->input.data),
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

static iree_status_t loom_serve_json_element(void* user_data,
                                             iree_host_size_t index,
                                             iree_json_value_type_t type,
                                             iree_string_view_t value) {
  return loom_serve_json_member(user_data, iree_string_view_empty(), type,
                                value);
}

static iree_status_t loom_serve_json_read_members(
    loom_serve_json_module_t* module, const iree_vm_call_packet_t* call,
    iree_vm_buffer_t* input, uint64_t offset, uint64_t length) {
  if (offset > SIZE_MAX || length > SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "JSON member range exceeds host storage");
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(input, offset, length, &bytes));
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
  loom_serve_json_members_t members = {.input = text, .offset = offset};
  iree_string_builder_initialize(module->host_allocator, &members.records);
  iree_status_t status = type == IREE_JSON_VALUE_TYPE_OBJECT
                             ? iree_json_enumerate_object_typed(
                                   value, loom_serve_json_member, &members)
                             : iree_json_enumerate_array_typed(
                                   value, loom_serve_json_element, &members);
  iree_vm_buffer_t* output = NULL;
  const iree_string_view_t records = iree_string_builder_view(&members.records);
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(records.data, records.size),
        iree_alignof(int64_t), module->host_allocator, &output);
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

static iree_status_t loom_serve_json_unescape(const iree_vm_call_packet_t* call,
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

static iree_status_t loom_serve_json_parse_number(
    const iree_vm_call_packet_t* call, iree_vm_buffer_t* input,
    bool floating_point) {
  const uint64_t offset = iree_vm_call_value_argument_load(call, 0);
  const uint64_t length = iree_vm_call_value_argument_load(call, 1);
  if (offset > SIZE_MAX || length > SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "JSON number range exceeds host storage");
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(input, offset, length, &bytes));
  iree_string_view_t cursor =
      iree_make_string_view((const char*)bytes.data, bytes.data_length);
  iree_string_view_t number;
  IREE_RETURN_IF_ERROR(iree_json_consume_number(&cursor, &number));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing JSON number input");
  }
  uint64_t bits = 0;
  if (floating_point) {
    double value = 0;
    if (!iree_string_view_atod(number, &value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "JSON number cannot be represented as F64");
    }
    if (!isfinite(value)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "JSON number overflows F64");
    }
    if (value == 0.0) {
      // A nonzero significand must not silently become a zero-valued policy,
      // such as disabling sampling when a positive temperature underflows.
      for (iree_host_size_t i = 0;
           i < number.size && number.data[i] != 'e' && number.data[i] != 'E';
           ++i) {
        if (number.data[i] >= '1' && number.data[i] <= '9') {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "JSON number underflows F64");
        }
      }
    }
    memcpy(&bits, &value, sizeof(bits));
  } else {
    int64_t value = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_int64(number, &value));
    bits = (uint64_t)value;
  }
  iree_vm_call_value_result_store(call, 0, bits);
  return iree_ok_status();
}

static iree_status_t loom_serve_json_format_number(
    loom_serve_json_module_t* module, const iree_vm_call_packet_t* call) {
  const uint64_t bits = iree_vm_call_value_argument_load(call, 0);
  double value = 0.0;
  memcpy(&value, &bits, sizeof(value));
  const int32_t precision = (int32_t)iree_vm_call_value_argument_load(call, 1);
  if (!isfinite(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JSON numbers must be finite");
  }
  if ((uint32_t)precision > 17) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixed JSON precision must be from 0 through 17");
  }
  // Binary64 has at most 309 integral digits. The bounded fractional precision
  // leaves room for the sign, decimal point, and formatter's terminator.
  char text[1 + 309 + 1 + 17 + 1];
  const int length =
      iree_snprintf(text, sizeof(text), "%.*f", precision, value);
  iree_vm_buffer_t* output = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ, iree_make_const_byte_span(text, length),
      1, module->host_allocator, &output));
  iree_vm_ref_t result =
      iree_vm_buffer_ref_from_ptr_move(&module->types, &output);
  iree_vm_call_ref_result_store_move(call, 0, &result);
  return iree_ok_status();
}

static iree_status_t loom_serve_json_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  loom_serve_json_module_t* module = (loom_serve_json_module_t*)base;
  const iree_vm_call_packet_t* call = &params->call;
  if (params->function_ordinal == 6) {
    IREE_RETURN_IF_ERROR(loom_serve_json_format_number(module, call));
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    return iree_ok_status();
  }
  iree_vm_ref_t input_ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, 0, &input_ref);
  if (!input_ref.object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JSON input must not be null");
  }
  iree_vm_buffer_t* input = (iree_vm_buffer_t*)input_ref.object;
  switch (params->function_ordinal) {
    case 0: {
      IREE_RETURN_IF_ERROR(loom_serve_json_compact(module, call, input));
      break;
    }
    case 1: {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "loom_serve_json_members");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, iree_vm_buffer_length(input));
      iree_status_t status = loom_serve_json_read_members(
          module, call, input, 0, iree_vm_buffer_length(input));
      IREE_TRACE_ZONE_END(z0);
      IREE_RETURN_IF_ERROR(status);
      break;
    }
    case 2: {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "loom_serve_json_unescape");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(
          z0, iree_vm_call_value_argument_load(call, 1));
      iree_status_t status = loom_serve_json_unescape(call, input);
      IREE_TRACE_ZONE_END(z0);
      IREE_RETURN_IF_ERROR(status);
      break;
    }
    case 3:
    case 4: {
      IREE_RETURN_IF_ERROR(loom_serve_json_parse_number(
          call, input, params->function_ordinal == 4));
      break;
    }
    case 5: {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "loom_serve_json_members_range");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(
          z0, iree_vm_call_value_argument_load(call, 1));
      iree_status_t status = loom_serve_json_read_members(
          module, call, input, iree_vm_call_value_argument_load(call, 0),
          iree_vm_call_value_argument_load(call, 1));
      IREE_TRACE_ZONE_END(z0);
      IREE_RETURN_IF_ERROR(status);
      break;
    }
    default:
      IREE_CHECK_UNREACHABLE("invalid JSON function ordinal");
  }
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static const iree_vm_module_signature_type_t json_buffer_type[] = {
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
static const iree_vm_module_signature_type_t json_i64_result[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t json_range_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t json_float_result[] = {
    {IREE_VM_SCALAR_TYPE_F64, 0}};
static const iree_vm_module_signature_type_t json_format_arguments[] = {
    {IREE_VM_SCALAR_TYPE_F64, 0}, {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_callable_type_declaration_t json_callables[] = {
    {.signature = {.arguments = {json_buffer_type, 1, 0, 1, 0},
                   .results = {json_buffer_type, 1, 0, 1, 0}}},
    {.signature = {.arguments = {json_buffer_type, 1, 0, 1, 0},
                   .results = {json_members_results, 3, 2, 1, 0}}},
    {.signature = {.arguments = {json_format_arguments, 2, 2, 0, 0},
                   .results = {json_buffer_type, 1, 0, 1, 0}}},
    {.signature = {.arguments = {json_range_arguments, 3, 2, 1, 0},
                   .results = {json_i64_result, 1, 1, 0, 0}}},
    {.signature = {.arguments = {json_range_arguments, 3, 2, 1, 0},
                   .results = {json_float_result, 1, 1, 0, 0}}},
    {.signature = {.arguments = {json_range_arguments, 3, 2, 1, 0},
                   .results = {json_members_results, 3, 2, 1, 0}}},
    {.signature = {.arguments = {json_unescape_arguments, 5, 3, 2, 0},
                   .results = {json_i64_result, 1, 1, 0, 0}}},
};
static const iree_vm_module_export_declaration_t json_exports[] = {
    {.export_name = IREE_SVL("compact"),
     .callable_type_ordinal = 0,
     .function_ordinal = 0},
    {.export_name = IREE_SVL("format_f64"),
     .callable_type_ordinal = 2,
     .function_ordinal = 6},
    {.export_name = IREE_SVL("members"),
     .callable_type_ordinal = 1,
     .function_ordinal = 1},
    {.export_name = IREE_SVL("members_range"),
     .callable_type_ordinal = 5,
     .function_ordinal = 5},
    {.export_name = IREE_SVL("parse_f64"),
     .callable_type_ordinal = 4,
     .function_ordinal = 4},
    {.export_name = IREE_SVL("parse_i64"),
     .callable_type_ordinal = 3,
     .function_ordinal = 3},
    {.export_name = IREE_SVL("unescape"),
     .callable_type_ordinal = 6,
     .function_ordinal = 2},
};

static void loom_serve_json_module_destroy(iree_vm_module_t* base) {
  loom_serve_json_module_t* module = (loom_serve_json_module_t*)base;
  iree_allocator_free(module->host_allocator, module);
}
static void loom_serve_json_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = json_exports[ordinal];
}
static void loom_serve_json_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = json_callables[ordinal];
}
static void loom_serve_json_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("json module has no imports");
}
static void loom_serve_json_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("json module has no imports");
}
static const iree_vm_module_vtable_t json_module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_serve_json_module_destroy,
    .function_start = loom_serve_json_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_serve_json_module_query_import_group,
    .query_import = loom_serve_json_module_query_import,
    .query_export = loom_serve_json_module_query_export,
    .query_callable_type = loom_serve_json_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_json_module_create(iree_vm_environment_t* environment,
                                            iree_allocator_t host_allocator,
                                            iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  loom_serve_json_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->types = types;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("json"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->types.buffer, 1},
      .counts = {.function_count = 7,
                 .callable_type_count = IREE_ARRAYSIZE(json_callables),
                 .export_count = IREE_ARRAYSIZE(json_exports),
                 .callable_fields = {.value_count = 18, .ref_count = 11}},
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
