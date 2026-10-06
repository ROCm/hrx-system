// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/chat.h"

#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/util/json.h"

#define CHAT_TYPE(type) (1u << IREE_JSON_VALUE_TYPE_##type)
#define CHAT_STRING CHAT_TYPE(STRING)
#define CHAT_NUMBER CHAT_TYPE(NUMBER)
#define CHAT_OBJECT CHAT_TYPE(OBJECT)
#define CHAT_ARRAY CHAT_TYPE(ARRAY)
#define CHAT_NULL CHAT_TYPE(NULL)
#define CHAT_BOOL (CHAT_TYPE(TRUE) | CHAT_TYPE(FALSE))

// Typed fields preserve the distinction between JSON null and the string
// "null". This is request-boundary decoding, not an internal object model.
typedef struct chat_field_t {
  // Literal protocol key; escaped protocol keys are not accepted.
  const char* name;
  // Accepted JSON types as CHAT_TYPE bits.
  uint32_t types;
  // Borrowed value; null data denotes an absent optional field.
  iree_string_view_t value;
  // Actual JSON type, valid only when value.data is non-null.
  iree_json_value_type_t type;
} chat_field_t;

typedef enum chat_field_flag_bits_e {
  CHAT_FIELDS_IGNORE_UNKNOWN = 1u << 0,
} chat_field_flag_bits_t;
typedef uint32_t chat_field_flags_t;

typedef struct chat_fields_t {
  // Number of declared fields.
  iree_host_size_t count;
  // Mutable decoding destinations.
  chat_field_t* values;
  // Whether descriptive schema keys outside this view may pass through.
  chat_field_flags_t flags;
} chat_fields_t;

static iree_status_t chat_field_visit(void* user_data, iree_string_view_t key,
                                      iree_json_value_type_t type,
                                      iree_string_view_t value) {
  chat_fields_t* fields = user_data;
  for (iree_host_size_t i = 0; i < fields->count; ++i) {
    chat_field_t* field = &fields->values[i];
    if (!iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      continue;
    }
    if (field->value.data || !(field->types & (1u << type))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed chat field '%s'",
                              field->name);
    }
    field->value = value;
    field->type = type;
    return iree_ok_status();
  }
  if (iree_any_bit_set(fields->flags, CHAT_FIELDS_IGNORE_UNKNOWN)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported chat field '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t chat_fields_read(iree_string_view_t object,
                                      iree_host_size_t count,
                                      chat_field_t* fields,
                                      chat_field_flags_t flags) {
  chat_fields_t context = {count, fields, flags};
  return iree_json_enumerate_object_typed(object, chat_field_visit, &context);
}

static iree_status_t chat_required(const chat_field_t* field) {
  if (!field->value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing chat field '%s'", field->name);
  }
  return iree_ok_status();
}

static iree_status_t chat_append_decoded(iree_string_builder_t* output,
                                         iree_string_view_t raw) {
  char* data = NULL;
  iree_host_size_t capacity = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      output, raw.size, &data, &capacity));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(raw, capacity, data, &length));
  iree_string_builder_commit_append(output, length);
  return iree_ok_status();
}

static iree_status_t chat_quote(iree_string_builder_t* output,
                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static bool chat_is_identifier(iree_string_view_t name) {
  if (!name.size || name.size > 128) {
    return false;
  }
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    const char c = name.data[i];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) {
      return false;
    }
  }
  return true;
}

static iree_status_t chat_content_part(void* user_data, iree_host_size_t index,
                                       iree_json_value_type_t type,
                                       iree_string_view_t value) {
  (void)index;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "chat content parts must be text objects");
  }
  chat_field_t fields[] = {{"type", CHAT_STRING}, {"text", CHAT_STRING}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(chat_required(&fields[1]));
  if (!iree_string_view_equal(fields[0].value, IREE_SV("text"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only text content is supported");
  }
  return chat_append_decoded(user_data, fields[1].value);
}

static iree_status_t chat_content(const chat_field_t* field,
                                  iree_string_builder_t* output) {
  iree_string_builder_reset(output);
  if (!field->value.data || field->type == IREE_JSON_VALUE_TYPE_NULL) {
    return iree_ok_status();
  }
  if (field->type == IREE_JSON_VALUE_TYPE_STRING) {
    return chat_append_decoded(output, field->value);
  }
  return iree_json_enumerate_array_typed(field->value, chat_content_part,
                                         output);
}

static iree_status_t chat_tool_schema(void* user_data, iree_host_size_t index,
                                      iree_json_value_type_t type,
                                      iree_string_view_t value) {
  loom_serve_text_chat_t* chat = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT ||
      index >= LOOM_SERVE_TEXT_CHAT_TOOL_CAPACITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tools must contain at most %d function objects",
                            LOOM_SERVE_TEXT_CHAT_TOOL_CAPACITY);
  }
  chat_field_t fields[] = {{"type", CHAT_STRING}, {"function", CHAT_OBJECT}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(chat_required(&fields[1]));
  if (!iree_string_view_equal(fields[0].value, IREE_SV("function"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only function tools are supported");
  }
  chat_field_t function[] = {{"name", CHAT_STRING},
                             {"description", CHAT_STRING},
                             {"parameters", CHAT_OBJECT},
                             {"strict", CHAT_BOOL}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(fields[1].value, IREE_ARRAYSIZE(function), function, 0));
  IREE_RETURN_IF_ERROR(chat_required(&function[2]));
  if (function[3].value.data && function[3].type == IREE_JSON_VALUE_TYPE_TRUE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "strict constrained tool sampling is not implemented");
  }
  if (!chat_is_identifier(function[0].value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool name must be a literal ASCII identifier");
  }
  for (iree_host_size_t i = 0; i < chat->tool_count; ++i) {
    if (iree_string_view_equal(chat->tools[i].name, function[0].value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate tool name");
    }
  }
  chat_field_t schema[] = {{"type", CHAT_STRING},
                           {"properties", CHAT_OBJECT},
                           {"required", CHAT_ARRAY}};
  IREE_RETURN_IF_ERROR(chat_fields_read(function[2].value,
                                        IREE_ARRAYSIZE(schema), schema,
                                        CHAT_FIELDS_IGNORE_UNKNOWN));
  IREE_RETURN_IF_ERROR(chat_required(&schema[1]));
  if (!iree_string_view_equal(schema[0].value, IREE_SV("object"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool parameters must use an object schema");
  }
  chat->tools[chat->tool_count++] = (loom_serve_text_chat_tool_t){
      function[0].value, schema[1].value, schema[2].value};
  return iree_ok_status();
}

void loom_serve_text_chat_policy_deinitialize(
    loom_serve_text_chat_policy_t* policy) {
  iree_vm_buffer_release(policy->name_storage);
  memset(policy, 0, sizeof(*policy));
}

iree_status_t loom_serve_text_chat_policy_initialize(
    iree_vm_environment_t* environment, loom_serve_program_t* program,
    loom_serve_text_chat_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &out_policy->types));
  out_policy->invocation = loom_serve_program_invocation(program);
  iree_vm_function_t name_function = {0};
  const char* names[] = {"model_name",    "chat_begin",  "chat_message",
                         "chat_end",      "parse_tools", "render_tool",
                         "prepare_input", "text_end",    "complete_text"};
  iree_vm_function_t* functions[] = {&name_function,
                                     &out_policy->chat_begin,
                                     &out_policy->chat_message,
                                     &out_policy->chat_end,
                                     &out_policy->parse_tools,
                                     &out_policy->render_tool,
                                     &out_policy->prepare_input,
                                     &out_policy->text_end,
                                     &out_policy->complete_text};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(names) && iree_status_is_ok(status); ++i) {
    status = iree_vm_process_lookup_function(
        loom_serve_program_process(program), IREE_SV("model"),
        iree_make_cstring_view(names[i]), functions[i]);
  }
  iree_vm_variant_t result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(out_policy->invocation, name_function,
                            iree_vm_variant_span_empty(),
                            iree_vm_variant_span_from_ptr(&result, 1));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&out_policy->types, &result,
                                                  &out_policy->name_storage);
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status) && !out_policy->name_storage) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source model name is null");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(
        out_policy->name_storage, 0,
        iree_vm_buffer_length(out_policy->name_storage), &bytes);
  }
  if (iree_status_is_ok(status)) {
    out_policy->name =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    if (!out_policy->name.size ||
        !iree_unicode_utf8_validate(out_policy->name)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source model name must be nonempty UTF-8");
    }
  }
  iree_vm_variant_reset(&result);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_chat_policy_deinitialize(out_policy);
  }
  return status;
}

// Request fragments never escape this synchronous call. The returned buffer
// may alias an input; it is appended before argument references are released.
static iree_status_t chat_render(const loom_serve_text_chat_policy_t* policy,
                                 iree_vm_function_t function,
                                 iree_host_size_t input_count,
                                 const iree_string_view_t* inputs,
                                 iree_host_size_t scalar_count,
                                 const int64_t* scalars, int64_t* out_state,
                                 iree_string_builder_t* output) {
  iree_vm_variant_t arguments[10] = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span((void*)inputs[i].data, inputs[i].size),
        iree_vm_buffer_release_callback_null(), output->allocator, &buffer);
    if (iree_status_is_ok(status)) {
      arguments[i] =
          iree_vm_buffer_variant_from_ptr_move(&policy->types, &buffer);
    }
  }
  for (iree_host_size_t i = 0; i < scalar_count; ++i) {
    arguments[input_count + i] = iree_vm_variant_from_i64(scalars[i]);
  }
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(
        policy->invocation, function,
        iree_vm_variant_span_from_ptr(arguments, input_count + scalar_count),
        iree_vm_variant_span_from_ptr(results, out_state ? 2 : 1));
  }
  iree_vm_buffer_t* rendered = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[0], &rendered);
  }
  if (iree_status_is_ok(status) && !rendered) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned a null chat fragment");
  }
  int64_t state = 0;
  if (iree_status_is_ok(status) && out_state) {
    status = iree_vm_i64_from_variant(results[1], &state);
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(rendered, 0,
                                     iree_vm_buffer_length(rendered), &bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        output,
        iree_make_string_view((const char*)bytes.data, bytes.data_length));
  }
  if (iree_status_is_ok(status) && out_state) {
    *out_state = state;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

iree_status_t loom_serve_text_chat_prepare_input(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t text,
    loom_serve_text_chat_input_format_t format,
    loom_serve_text_chat_boundary_t boundary, iree_host_size_t capacity,
    int32_t* tokens, iree_host_size_t* out_count,
    iree_allocator_t host_allocator) {
  *out_count = 0;
  iree_vm_buffer_t* input = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)text.data, text.size),
      iree_vm_buffer_release_callback_null(), host_allocator, &input));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&policy->types, &input),
      iree_vm_variant_from_i32(format), iree_vm_variant_from_i32(boundary),
      iree_vm_variant_from_i64(capacity)};
  iree_vm_variant_t results[2] = {0};
  iree_status_t status =
      iree_vm_invoke(policy->invocation, policy->prepare_input,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  iree_vm_buffer_t* output = NULL;
  int64_t count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[0], &output);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &count);
  }
  if (iree_status_is_ok(status) &&
      (!output || count < 0 || (uint64_t)count > capacity ||
       (uint64_t)count > SIZE_MAX / sizeof(*tokens))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source input returned invalid token storage");
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_map_read(output, 0, count * sizeof(*tokens), &bytes);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < (iree_host_size_t)count; ++i) {
      tokens[i] = (int32_t)iree_unaligned_load_le_u32(bytes.data + i * 4);
    }
    *out_count = (iree_host_size_t)count;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

// Input wrappers borrow only for this synchronous invocation. Source owns the
// returned bytes until they are appended; no references escape into a request.
static iree_status_t chat_function_render(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t name,
    iree_string_view_t arguments, iree_host_size_t ordinal,
    iree_string_builder_t* output) {
  const iree_string_view_t inputs[] = {name, arguments};
  iree_vm_variant_t values[3] = {0};
  values[2] = iree_vm_variant_from_i64(ordinal);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(inputs) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span((void*)inputs[i].data, inputs[i].size),
        iree_vm_buffer_release_callback_null(), output->allocator, &buffer);
    if (iree_status_is_ok(status)) {
      values[i] = iree_vm_buffer_variant_from_ptr_move(&policy->types, &buffer);
    }
  }
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(policy->invocation, policy->render_tool,
                            iree_vm_variant_span_from_array(values),
                            iree_vm_variant_span_from_array(results));
  }
  iree_vm_buffer_t* rendered = NULL;
  int64_t length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy->types,
                                                      results[0], &rendered);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &length);
  }
  if (iree_status_is_ok(status) &&
      (!rendered || length < 0 || (uint64_t)length > SIZE_MAX)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source tool formatter returned invalid storage");
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_map_read(rendered, 0, (iree_host_size_t)length, &bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        output,
        iree_make_string_view((const char*)bytes.data, bytes.data_length));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(values));
  return status;
}

typedef struct chat_history_t {
  // Request being rendered.
  loom_serve_text_chat_t* chat;
  // Decoded content, reasoning and argument scratch.
  iree_string_builder_t scratch;
  // Decoded JSON argument document, stable while its values are rendered.
  iree_string_builder_t arguments;
  // Decoded reasoning content for one message.
  iree_string_builder_t reasoning;
  // Canonical tool calls for one message.
  iree_string_builder_t calls;
  // Source-owned rendering state, with no native interpretation.
  int64_t state;
} chat_history_t;

static iree_status_t chat_history_tool(void* user_data, iree_host_size_t index,
                                       iree_json_value_type_t type,
                                       iree_string_view_t value) {
  chat_history_t* history = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT || index >= 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "assistant tool_calls must contain at most 16 objects");
  }
  chat_field_t fields[] = {
      {"id", CHAT_STRING}, {"type", CHAT_STRING}, {"function", CHAT_OBJECT}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(chat_required(&fields[2]));
  if (!iree_string_view_equal(fields[1].value, IREE_SV("function"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "assistant tool_calls must be functions");
  }
  chat_field_t function[] = {{"name", CHAT_STRING}, {"arguments", CHAT_STRING}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(fields[2].value, IREE_ARRAYSIZE(function), function, 0));
  IREE_RETURN_IF_ERROR(chat_required(&function[1]));
  iree_string_builder_reset(&history->arguments);
  IREE_RETURN_IF_ERROR(
      chat_append_decoded(&history->arguments, function[1].value));
  return chat_function_render(history->chat->policy, function[0].value,
                              iree_string_builder_view(&history->arguments),
                              index, &history->calls);
}

static iree_status_t chat_history_message(void* user_data,
                                          iree_host_size_t index,
                                          iree_json_value_type_t type,
                                          iree_string_view_t value) {
  chat_history_t* history = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT || index >= 256) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "messages must contain at most 256 objects");
  }
  chat_field_t fields[] = {{"role", CHAT_STRING},
                           {"content", CHAT_STRING | CHAT_ARRAY | CHAT_NULL},
                           {"reasoning_content", CHAT_STRING},
                           {"tool_calls", CHAT_ARRAY},
                           {"tool_call_id", CHAT_STRING},
                           {"name", CHAT_STRING}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(value, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(chat_required(&fields[0]));
  IREE_RETURN_IF_ERROR(chat_content(&fields[1], &history->scratch));
  IREE_RETURN_IF_ERROR(chat_content(&fields[2], &history->reasoning));
  iree_string_builder_reset(&history->calls);
  if (fields[3].value.data) {
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
        fields[3].value, chat_history_tool, history));
  }
  const iree_string_view_t inputs[] = {
      value, fields[0].value, iree_string_builder_view(&history->scratch),
      iree_string_builder_view(&history->reasoning),
      iree_string_builder_view(&history->calls)};
  const int64_t scalars[] = {history->state, (int64_t)index,
                             (fields[2].value.data ? 1 : 0) |
                                 (fields[3].value.data ? 2 : 0) |
                                 (fields[4].value.data ? 4 : 0)};
  return chat_render(history->chat->policy, history->chat->policy->chat_message,
                     IREE_ARRAYSIZE(inputs), inputs, IREE_ARRAYSIZE(scalars),
                     scalars, &history->state, &history->chat->prompt);
}

static iree_status_t chat_request_render(iree_string_view_t body,
                                         chat_history_t* history) {
  if (!iree_unicode_utf8_validate(body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "chat JSON must be UTF-8");
  }
  iree_string_view_t cursor = body;
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "trailing chat JSON");
  }
  chat_field_t fields[] = {
      {"model", CHAT_STRING},         {"messages", CHAT_ARRAY},
      {"stream", CHAT_BOOL},          {"tools", CHAT_ARRAY},
      {"max_tokens", CHAT_NUMBER},    {"max_completion_tokens", CHAT_NUMBER},
      {"temperature", CHAT_NUMBER},   {"stream_options", CHAT_OBJECT},
      {"store", CHAT_BOOL},           {"tool_choice", CHAT_STRING},
      {"enable_thinking", CHAT_BOOL}, {"model_options", CHAT_OBJECT}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(object, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(chat_required(&fields[1]));
  IREE_RETURN_IF_ERROR(chat_required(&fields[0]));
  iree_string_builder_reset(&history->scratch);
  IREE_RETURN_IF_ERROR(chat_append_decoded(&history->scratch, fields[0].value));
  if (!iree_string_view_equal(iree_string_builder_view(&history->scratch),
                              history->chat->policy->name) ||
      fields[2].type != IREE_JSON_VALUE_TYPE_TRUE || !fields[2].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "select model %.*s and stream=true",
                            (int)history->chat->policy->name.size,
                            history->chat->policy->name.data);
  }
  if ((fields[8].value.data && fields[8].type != IREE_JSON_VALUE_TYPE_FALSE) ||
      (fields[9].value.data &&
       !iree_string_view_equal(fields[9].value, IREE_SV("auto")))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "only store=false and automatic tools are supported");
  }
  if (fields[6].value.data) {
    double temperature = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_double(fields[6].value, &temperature));
    if (temperature != 0.0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "only greedy temperature=0 is supported");
    }
  }
  if (fields[4].value.data && fields[5].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "specify one output token limit");
  }
  const iree_string_view_t limit =
      fields[4].value.data ? fields[4].value : fields[5].value;
  if (limit.data) {
    uint64_t max_tokens = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(limit, &max_tokens));
    if (!max_tokens || max_tokens > 16384) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "output limit must be in [1, 16384]");
    }
    history->chat->max_tokens = (iree_host_size_t)max_tokens;
  }
  if (fields[7].value.data) {
    chat_field_t options[] = {{"include_usage", CHAT_BOOL}};
    IREE_RETURN_IF_ERROR(
        chat_fields_read(fields[7].value, IREE_ARRAYSIZE(options), options, 0));
    history->chat->include_usage =
        options[0].value.data && options[0].type == IREE_JSON_VALUE_TYPE_TRUE;
  }
  history->chat->tool_schemas =
      fields[3].value.data ? fields[3].value : IREE_SV("[]");
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      history->chat->tool_schemas, chat_tool_schema, history->chat));
  const iree_string_view_t inputs[] = {body, history->chat->tool_schemas};
  IREE_RETURN_IF_ERROR(chat_render(history->chat->policy,
                                   history->chat->policy->chat_begin,
                                   IREE_ARRAYSIZE(inputs), inputs, 0, NULL,
                                   &history->state, &history->chat->prompt));
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      fields[1].value, chat_history_message, history));
  return chat_render(history->chat->policy, history->chat->policy->chat_end, 0,
                     NULL, 1, &history->state, NULL, &history->chat->prompt);
}

iree_status_t loom_serve_text_chat_initialize(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t body,
    iree_host_size_t default_max_tokens, iree_allocator_t host_allocator,
    loom_serve_text_chat_t* out_chat) {
  memset(out_chat, 0, sizeof(*out_chat));
  out_chat->policy = policy;
  out_chat->max_tokens = default_max_tokens;
  iree_string_builder_initialize(host_allocator, &out_chat->prompt);
  chat_history_t history = {.chat = out_chat};
  iree_string_builder_initialize(host_allocator, &history.scratch);
  iree_string_builder_initialize(host_allocator, &history.arguments);
  iree_string_builder_initialize(host_allocator, &history.reasoning);
  iree_string_builder_initialize(host_allocator, &history.calls);
  iree_status_t status = chat_request_render(body, &history);
  iree_string_builder_deinitialize(&history.calls);
  iree_string_builder_deinitialize(&history.reasoning);
  iree_string_builder_deinitialize(&history.arguments);
  iree_string_builder_deinitialize(&history.scratch);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_chat_deinitialize(out_chat);
  }
  return status;
}

void loom_serve_text_chat_deinitialize(loom_serve_text_chat_t* chat) {
  iree_string_builder_deinitialize(&chat->prompt);
  memset(chat, 0, sizeof(*chat));
}

iree_status_t loom_serve_text_chat_text_end(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t response,
    iree_host_size_t previous_end, loom_serve_text_chat_output_phase_t phase,
    iree_host_size_t* out_end, iree_allocator_t host_allocator) {
  *out_end = previous_end;
  iree_vm_buffer_t* input = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)response.data, response.size),
      iree_vm_buffer_release_callback_null(), host_allocator, &input));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&policy->types, &input),
      iree_vm_variant_from_i64(previous_end), iree_vm_variant_from_i32(phase)};
  iree_vm_variant_t result = {0};
  iree_status_t status =
      iree_vm_invoke(policy->invocation, policy->text_end,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_ptr(&result, 1));
  int64_t end = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(result, &end);
  }
  if (iree_status_is_ok(status) && (end < 0 || (uint64_t)end < previous_end ||
                                    (uint64_t)end > response.size)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned invalid text extent");
  }
  if (iree_status_is_ok(status)) {
    *out_end = (iree_host_size_t)end;
  }
  iree_vm_variant_reset(&result);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

void loom_serve_text_chat_completion_deinitialize(
    loom_serve_text_chat_completion_t* completion) {
  iree_vm_buffer_release(completion->storage);
  memset(completion, 0, sizeof(*completion));
}

// Returned buffers may alias any argument. Owned input clones make every such
// result independent of HTTP claims, mutable builders and invocation storage.
static iree_status_t chat_checkpoint(
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t prompt,
    iree_string_view_t content, iree_string_view_t tools,
    loom_serve_text_chat_completion_t* completion,
    iree_allocator_t host_allocator) {
  const iree_string_view_t inputs[] = {prompt, content, tools};
  iree_vm_variant_t arguments[3] = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(inputs) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(inputs[i].data, inputs[i].size), 1,
        host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      arguments[i] =
          iree_vm_buffer_variant_from_ptr_move(&policy->types, &buffer);
    }
  }
  iree_vm_variant_t result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(policy->invocation, policy->complete_text,
                            iree_vm_variant_span_from_array(arguments),
                            iree_vm_variant_span_from_ptr(&result, 1));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&policy->types, &result,
                                                  &completion->storage);
  }
  if (iree_status_is_ok(status) && !completion->storage) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source returned null checkpoint storage");
  }
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(completion->storage, 0,
                                     iree_vm_buffer_length(completion->storage),
                                     &bytes);
  }
  if (iree_status_is_ok(status)) {
    completion->transcript =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
  }
  iree_vm_variant_reset(&result);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

static iree_status_t chat_xml_consume(iree_string_view_t* cursor,
                                      iree_string_view_t literal) {
  *cursor = iree_string_view_trim(*cursor);
  if (!iree_string_view_consume_prefix(cursor, literal)) {
    // Bound model-generated diagnostics and keep terminal control bytes out of
    // server logs. The request still fails without publishing a partial call.
    char preview[33];
    const iree_host_size_t length = iree_min(cursor->size, sizeof(preview) - 1);
    for (iree_host_size_t i = 0; i < length; ++i) {
      const unsigned char c = (unsigned char)cursor->data[i];
      preview[i] = c >= 0x20 && c <= 0x7E ? (char)c : '.';
    }
    preview[length] = 0;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated tool call expected '%.*s'; got '%s%s'",
                            (int)literal.size, literal.data, preview,
                            cursor->size > length ? "..." : "");
  }
  return iree_ok_status();
}

static iree_status_t chat_xml_name(iree_string_view_t* cursor,
                                   iree_string_view_t prefix,
                                   iree_string_view_t* out_name) {
  IREE_RETURN_IF_ERROR(chat_xml_consume(cursor, prefix));
  const iree_host_size_t end = iree_string_view_find_char(*cursor, '>', 0);
  if (end == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unterminated generated tool name");
  }
  *out_name = iree_string_view_substr(*cursor, 0, end);
  *cursor = iree_string_view_substr(*cursor, end + 1, IREE_HOST_SIZE_MAX);
  if (!chat_is_identifier(*out_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated tool name is not an identifier");
  }
  return iree_ok_status();
}

static iree_status_t chat_parameter_value(
    const loom_serve_text_chat_tool_t* tool, iree_string_view_t name,
    iree_string_view_t value, iree_string_builder_t* arguments) {
  char name_storage[129];
  memcpy(name_storage, name.data, name.size);
  name_storage[name.size] = 0;
  chat_field_t property = {name_storage, CHAT_OBJECT};
  IREE_RETURN_IF_ERROR(chat_fields_read(tool->properties, 1, &property,
                                        CHAT_FIELDS_IGNORE_UNKNOWN));
  IREE_RETURN_IF_ERROR(chat_required(&property));
  chat_field_t schema[] = {{"type", CHAT_STRING}};
  IREE_RETURN_IF_ERROR(chat_fields_read(property.value, IREE_ARRAYSIZE(schema),
                                        schema, CHAT_FIELDS_IGNORE_UNKNOWN));
  IREE_RETURN_IF_ERROR(chat_required(&schema[0]));
  if (iree_string_view_equal(schema[0].value, IREE_SV("string"))) {
    return chat_quote(arguments, value);
  }
  value = iree_string_view_trim(value);
  if (!value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty non-string tool parameter");
  }
  const iree_json_value_type_t type = iree_json_infer_value_type(value.data[0]);
  const bool integer =
      iree_string_view_equal(schema[0].value, IREE_SV("integer"));
  const bool matches =
      (type == IREE_JSON_VALUE_TYPE_NUMBER &&
       (integer ||
        iree_string_view_equal(schema[0].value, IREE_SV("number")))) ||
      ((type == IREE_JSON_VALUE_TYPE_TRUE ||
        type == IREE_JSON_VALUE_TYPE_FALSE) &&
       iree_string_view_equal(schema[0].value, IREE_SV("boolean"))) ||
      (type == IREE_JSON_VALUE_TYPE_OBJECT &&
       iree_string_view_equal(schema[0].value, IREE_SV("object"))) ||
      (type == IREE_JSON_VALUE_TYPE_ARRAY &&
       iree_string_view_equal(schema[0].value, IREE_SV("array")));
  if (!matches) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "generated parameter '%.*s' does not match schema type '%.*s'",
        (int)name.size, name.data, (int)schema[0].value.size,
        schema[0].value.data);
  }
  iree_string_view_t cursor = value;
  iree_string_view_t parsed;
  IREE_RETURN_IF_ERROR(iree_json_consume_value(&cursor, &parsed));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing data in generated tool parameter");
  }
  if (integer) {
    int64_t number = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_int64(value, &number));
  }
  return iree_string_builder_append_string(arguments, value);
}

typedef struct chat_parameter_names_t {
  // Number of generated parameters already parsed.
  iree_host_size_t count;
  // Borrowed names in generation order, used for duplicate/required checks.
  iree_string_view_t values[32];
} chat_parameter_names_t;

static iree_status_t chat_required_parameter(void* user_data,
                                             iree_host_size_t index,
                                             iree_json_value_type_t type,
                                             iree_string_view_t value) {
  (void)index;
  const chat_parameter_names_t* names = user_data;
  if (type != IREE_JSON_VALUE_TYPE_STRING || !chat_is_identifier(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required tool parameters must be literal names");
  }
  for (iree_host_size_t i = 0; i < names->count; ++i) {
    if (iree_string_view_equal(names->values[i], value)) {
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "generated tool call omitted required parameter '%.*s'", (int)value.size,
      value.data);
}

static iree_status_t chat_parameters_parse(
    iree_string_view_t* cursor, const loom_serve_text_chat_tool_t* tool,
    iree_string_builder_t* arguments) {
  chat_parameter_names_t names = {0};
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(arguments, "{"));
  iree_status_t status = iree_ok_status();
  *cursor = iree_string_view_trim(*cursor);
  while (iree_string_view_starts_with(*cursor, IREE_SV("<parameter=")) &&
         iree_status_is_ok(status)) {
    iree_string_view_t name;
    status = chat_xml_name(cursor, IREE_SV("<parameter="), &name);
    if (iree_status_is_ok(status) &&
        names.count == IREE_ARRAYSIZE(names.values)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "too many generated parameters");
    }
    for (iree_host_size_t i = 0; i < names.count && iree_status_is_ok(status);
         ++i) {
      if (iree_string_view_equal(names.values[i], name)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "duplicate generated parameter");
      }
    }
    const iree_string_view_t end_tag = IREE_SV("</parameter>");
    const iree_host_size_t end = iree_string_view_find(*cursor, end_tag, 0);
    if (iree_status_is_ok(status) && end == IREE_STRING_VIEW_NPOS) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unterminated generated parameter");
    }
    if (iree_status_is_ok(status)) {
      iree_string_view_t value = iree_string_view_substr(*cursor, 0, end);
      // Remove the template's one framing newline, preserving string content.
      if (!iree_string_view_consume_prefix(&value, IREE_SV("\r\n"))) {
        iree_string_view_consume_prefix(&value, IREE_SV("\n"));
      }
      if (!iree_string_view_consume_suffix(&value, IREE_SV("\r\n"))) {
        iree_string_view_consume_suffix(&value, IREE_SV("\n"));
      }
      if (names.count) {
        status = iree_string_builder_append_cstring(arguments, ",");
      }
      if (iree_status_is_ok(status)) {
        status = chat_quote(arguments, name);
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(arguments, ":");
      }
      if (iree_status_is_ok(status)) {
        status = chat_parameter_value(tool, name, value, arguments);
      }
      names.values[names.count++] = name;
      *cursor = iree_string_view_trim(iree_string_view_substr(
          *cursor, end + end_tag.size, IREE_HOST_SIZE_MAX));
    }
  }
  if (iree_status_is_ok(status) && tool->required.data) {
    status = iree_json_enumerate_array_typed(tool->required,
                                             chat_required_parameter, &names);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(arguments, "}");
  }
  return status;
}

static iree_status_t chat_parse_xml(const loom_serve_text_chat_t* chat,
                                    iree_string_view_t text,
                                    iree_string_builder_t* tool_calls) {
  iree_host_size_t tool_count = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(tool_calls, "["));
  iree_string_view_t cursor = iree_string_view_trim(text);
  iree_string_builder_t arguments;
  iree_string_builder_initialize(tool_calls->allocator, &arguments);
  iree_status_t status = iree_ok_status();
  while (cursor.size && iree_status_is_ok(status)) {
    status = chat_xml_consume(&cursor, IREE_SV("<tool_call>"));
    iree_string_view_t name = iree_string_view_empty();
    if (iree_status_is_ok(status)) {
      status = chat_xml_name(&cursor, IREE_SV("<function="), &name);
    }
    const loom_serve_text_chat_tool_t* tool = NULL;
    for (iree_host_size_t i = 0; i < chat->tool_count; ++i) {
      if (iree_string_view_equal(chat->tools[i].name, name)) {
        tool = &chat->tools[i];
      }
    }
    if (iree_status_is_ok(status) && (!tool || tool_count == 16)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown or excessive generated tool call");
    }
    iree_string_builder_reset(&arguments);
    if (iree_status_is_ok(status)) {
      status = chat_parameters_parse(&cursor, tool, &arguments);
    }
    if (iree_status_is_ok(status)) {
      status = chat_xml_consume(&cursor, IREE_SV("</function>"));
    }
    if (iree_status_is_ok(status)) {
      status = chat_xml_consume(&cursor, IREE_SV("</tool_call>"));
    }
    if (iree_status_is_ok(status)) {
      const iree_host_size_t index = tool_count;
      status = iree_string_builder_append_format(
          tool_calls,
          "%s{\"type\":\"function\",\"function\":{\"name\":", index ? "," : "");
      if (iree_status_is_ok(status)) {
        status = chat_quote(tool_calls, name);
      }
      if (iree_status_is_ok(status)) {
        status =
            iree_string_builder_append_cstring(tool_calls, ",\"arguments\":");
      }
      if (iree_status_is_ok(status)) {
        status = chat_quote(tool_calls, iree_string_builder_view(&arguments));
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(tool_calls, "}}");
      }
      ++tool_count;
      cursor = iree_string_view_trim(cursor);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(tool_calls, "]");
  }
  iree_string_builder_deinitialize(&arguments);
  return status;
}

typedef struct chat_completion_tools_t {
  // Source policy used for canonical history formatting.
  const loom_serve_text_chat_policy_t* policy;
  // Request-unique namespace for transport tool IDs.
  uint64_t request_id;
  // Number of fully parsed calls.
  iree_host_size_t count;
  // Standard tool-call delta destination.
  iree_string_builder_t* output;
  // Canonical model-format call fragments.
  iree_string_builder_t rendered;
  // Decoded argument JSON for the current call.
  iree_string_builder_t arguments;
} chat_completion_tools_t;

static iree_status_t chat_completion_tool(void* user_data,
                                          iree_host_size_t index,
                                          iree_json_value_type_t type,
                                          iree_string_view_t value) {
  chat_completion_tools_t* tools = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT || index >= 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source tool calls must contain at most 16 objects");
  }
  chat_field_t call[] = {{"type", CHAT_STRING}, {"function", CHAT_OBJECT}};
  IREE_RETURN_IF_ERROR(chat_fields_read(value, IREE_ARRAYSIZE(call), call, 0));
  IREE_RETURN_IF_ERROR(chat_required(&call[1]));
  if (!iree_string_view_equal(call[0].value, IREE_SV("function"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source tool call must be a function");
  }
  chat_field_t function[] = {{"name", CHAT_STRING}, {"arguments", CHAT_STRING}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(call[1].value, IREE_ARRAYSIZE(function), function, 0));
  IREE_RETURN_IF_ERROR(chat_required(&function[0]));
  IREE_RETURN_IF_ERROR(chat_required(&function[1]));
  iree_string_builder_reset(&tools->arguments);
  IREE_RETURN_IF_ERROR(
      chat_append_decoded(&tools->arguments, function[1].value));
  IREE_RETURN_IF_ERROR(chat_function_render(
      tools->policy, function[0].value,
      iree_string_builder_view(&tools->arguments), index, &tools->rendered));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      tools->output,
      "%s{\"index\":%zu,\"id\":\"call_%" PRIu64
      "_%zu\",\"type\":\"function\",\"function\":{\"name\":",
      index ? "," : "", index, tools->request_id, index));
  // Source records preserve JSON escaping of names and arguments.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(tools->output, "\""));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(tools->output, function[0].value));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(tools->output, "\",\"arguments\":\""));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(tools->output, function[1].value));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(tools->output, "\"}}"));
  ++tools->count;
  return iree_ok_status();
}

iree_status_t loom_serve_text_chat_complete(
    const loom_serve_text_chat_t* chat, iree_string_view_t response,
    iree_host_size_t previous_end, uint64_t request_id,
    iree_string_builder_t* tool_calls,
    loom_serve_text_chat_completion_t* completion) {
  loom_serve_text_chat_completion_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_text_end(chat->policy, response, previous_end,
                                    LOOM_SERVE_TEXT_CHAT_OUTPUT_COMPLETE,
                                    &result.text_end, tool_calls->allocator));
  const iree_string_view_t content =
      iree_string_view_substr(response, 0, result.text_end);
  const iree_string_view_t inputs[] = {
      chat->tool_schemas,
      iree_string_view_substr(response, result.text_end, IREE_HOST_SIZE_MAX)};
  iree_string_builder_t parsed;
  iree_string_builder_initialize(tool_calls->allocator, &parsed);
  chat_completion_tools_t tools = {
      .policy = chat->policy, .request_id = request_id, .output = tool_calls};
  iree_string_builder_initialize(tool_calls->allocator, &tools.rendered);
  iree_string_builder_initialize(tool_calls->allocator, &tools.arguments);
  iree_status_t status =
      chat_render(chat->policy, chat->policy->parse_tools,
                  IREE_ARRAYSIZE(inputs), inputs, 0, NULL, NULL, &parsed);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(tool_calls, "[");
  }
  if (iree_status_is_ok(status)) {
    status = iree_json_enumerate_array_typed(iree_string_builder_view(&parsed),
                                             chat_completion_tool, &tools);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(tool_calls, "]");
  }
  if (iree_status_is_ok(status)) {
    result.tool_count = tools.count;
    status =
        chat_checkpoint(chat->policy, iree_string_builder_view(&chat->prompt),
                        content, iree_string_builder_view(&tools.rendered),
                        &result, tool_calls->allocator);
  }
  iree_string_builder_deinitialize(&tools.arguments);
  iree_string_builder_deinitialize(&tools.rendered);
  iree_string_builder_deinitialize(&parsed);
  if (iree_status_is_ok(status)) {
    loom_serve_text_chat_completion_deinitialize(completion);
    *completion = result;
  } else {
    loom_serve_text_chat_completion_deinitialize(&result);
  }
  return status;
}

iree_status_t loom_serve_text_chat_event(iree_string_view_t model_name,
                                         uint64_t request_id,
                                         iree_string_view_t delta,
                                         iree_string_view_t finish_reason,
                                         iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "data: {\"id\":\"loom-%" PRIu64
      "\",\"object\":\"chat.completion.chunk\",\"model\":",
      request_id));
  IREE_RETURN_IF_ERROR(chat_quote(output, model_name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"choices\":[{\"index\":0,\"delta\":%.*s,\"finish_reason\":",
      (int)delta.size, delta.data));
  if (finish_reason.size) {
    IREE_RETURN_IF_ERROR(chat_quote(output, finish_reason));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  return iree_string_builder_append_cstring(output, "}]}\n\n");
}

iree_status_t loom_serve_text_chat_usage(iree_string_view_t model_name,
                                         uint64_t request_id,
                                         iree_host_size_t input_tokens,
                                         iree_host_size_t retained_tokens,
                                         iree_host_size_t output_tokens,
                                         iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "data: {\"id\":\"loom-%" PRIu64
      "\",\"object\":\"chat.completion.chunk\",\"model\":",
      request_id));
  IREE_RETURN_IF_ERROR(chat_quote(output, model_name));
  return iree_string_builder_append_format(
      output,
      ",\"choices\":[],\"usage\":{\"prompt_tokens\":%zu,"
      "\"completion_tokens\":%zu,\"total_tokens\":%zu,"
      "\"prompt_tokens_details\":{\"cached_tokens\":%zu}}}\n\n",
      input_tokens, output_tokens, input_tokens + output_tokens,
      retained_tokens);
}

typedef struct chat_tools_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Descriptor borrowing the environment's reference types.
  iree_vm_module_descriptor_t descriptor;
  // Host allocation policy.
  iree_allocator_t allocator;
  // Canonical buffer reference identity.
  iree_vm_ref_types_t types;
} chat_tools_module_t;

static iree_status_t chat_tools_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  chat_tools_module_t* module = (chat_tools_module_t*)base;
  iree_string_view_t inputs[2];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    iree_vm_ref_t ref = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_borrow(&params->call, i, &ref);
    if (!ref.object) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tool input is null");
    }
    iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)ref.object;
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
        buffer, 0, iree_vm_buffer_length(buffer), &bytes));
    inputs[i] =
        iree_make_string_view((const char*)bytes.data, bytes.data_length);
    if (!iree_unicode_utf8_validate(inputs[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "tool input must be UTF-8");
    }
  }
  loom_serve_text_chat_t schema = {0};
  IREE_RETURN_IF_ERROR(
      iree_json_enumerate_array_typed(inputs[0], chat_tool_schema, &schema));
  iree_string_builder_t output;
  iree_string_builder_initialize(module->allocator, &output);
  iree_status_t status = chat_parse_xml(&schema, inputs[1], &output);
  iree_vm_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_string_view_t value = iree_string_builder_view(&output);
    status =
        iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_make_const_byte_span(value.data, value.size),
                             1, module->allocator, &buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t result =
        iree_vm_buffer_ref_from_ptr_move(&module->types, &buffer);
    iree_vm_call_ref_result_store_move(&params->call, 0, &result);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  }
  iree_vm_buffer_release(buffer);
  iree_string_builder_deinitialize(&output);
  return status;
}

static const iree_vm_module_signature_type_t chat_tools_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t chat_tools_results[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static void chat_tools_destroy(iree_vm_module_t* base) {
  chat_tools_module_t* module = (chat_tools_module_t*)base;
  iree_allocator_free(module->allocator, module);
}
static void chat_tools_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = (iree_vm_module_export_declaration_t){
      .export_name = IREE_SVL("parse_xml"),
      .callable_type_ordinal = 0,
      .function_ordinal = 0};
}
static void chat_tools_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = (iree_vm_module_callable_type_declaration_t){
      .signature = {.arguments = {chat_tools_arguments, 2, 0, 2, 0},
                    .results = {chat_tools_results, 1, 0, 1, 0}}};
}
static void chat_tools_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("tools module has no imports");
}
static void chat_tools_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("tools module has no imports");
}
static const iree_vm_module_vtable_t chat_tools_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = chat_tools_destroy,
    .function_start = chat_tools_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = chat_tools_query_import_group,
    .query_import = chat_tools_query_import,
    .query_export = chat_tools_query_export,
    .query_callable_type = chat_tools_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_text_chat_tools_module_create(
    iree_vm_environment_t* environment, iree_vm_module_t** out_module,
    iree_allocator_t host_allocator) {
  *out_module = NULL;
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  chat_tools_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->allocator = host_allocator;
  module->types = types;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("tools"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->types.buffer, 1},
      .counts = {.function_count = 1,
                 .callable_type_count = 1,
                 .export_count = 1,
                 .callable_fields = {.value_count = 0, .ref_count = 3}}};
  iree_status_t status = iree_vm_module_initialize(
      &chat_tools_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
