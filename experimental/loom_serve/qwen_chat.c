// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/qwen_chat.h"

#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "loom/util/json.h"

#define CHAT_TYPE(type) (1u << IREE_JSON_VALUE_TYPE_##type)
#define CHAT_STRING CHAT_TYPE(STRING)
#define CHAT_NUMBER CHAT_TYPE(NUMBER)
#define CHAT_OBJECT CHAT_TYPE(OBJECT)
#define CHAT_ARRAY CHAT_TYPE(ARRAY)
#define CHAT_NULL CHAT_TYPE(NULL)
#define CHAT_BOOL (CHAT_TYPE(TRUE) | CHAT_TYPE(FALSE))

static const char chat_assistant_prefix[] =
    "<|im_start|>assistant\n<think>\n\n</think>\n\n";
static const char chat_tools_instructions[] =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
    "following "
    "format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter"
    "\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified"
    "\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no "
    "function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls"
    "\n</IMPORTANT>";

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
  loom_serve_qwen_chat_t* chat = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT ||
      index >= LOOM_SERVE_QWEN_CHAT_TOOL_CAPACITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tools must contain at most %d function objects",
                            LOOM_SERVE_QWEN_CHAT_TOOL_CAPACITY);
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
  chat->tools[chat->tool_count++] = (loom_serve_qwen_chat_tool_t){
      function[0].value, schema[1].value, schema[2].value};
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(&chat->prompt, "\n"));
  return iree_string_builder_append_string(&chat->prompt, value);
}

typedef struct chat_argument_render_t {
  // Canonical XML destination.
  iree_string_builder_t* output;
  // Scratch used to decode string-valued arguments.
  iree_string_builder_t* scratch;
} chat_argument_render_t;

static iree_status_t chat_argument_render(void* user_data,
                                          iree_string_view_t name,
                                          iree_json_value_type_t type,
                                          iree_string_view_t value) {
  chat_argument_render_t* context = user_data;
  if (!chat_is_identifier(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool parameter name must be a literal identifier");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      context->output, "<parameter=%.*s>\n", (int)name.size, name.data));
  if (type == IREE_JSON_VALUE_TYPE_STRING) {
    iree_string_builder_reset(context->scratch);
    IREE_RETURN_IF_ERROR(chat_append_decoded(context->scratch, value));
    value = iree_string_builder_view(context->scratch);
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->output, value));
  return iree_string_builder_append_cstring(context->output,
                                            "\n</parameter>\n");
}

// Renders one function with already decoded JSON arguments. The same writer
// handles history and generated calls, keeping their checkpoint spelling equal.
static iree_status_t chat_function_render(iree_string_view_t name,
                                          iree_string_view_t arguments,
                                          chat_argument_render_t* context) {
  if (!chat_is_identifier(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool call name must be a literal identifier");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      context->output, "<tool_call>\n<function=%.*s>\n", (int)name.size,
      name.data));
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      arguments, chat_argument_render, context));
  return iree_string_builder_append_cstring(context->output,
                                            "</function>\n</tool_call>");
}

typedef enum chat_role_e {
  CHAT_ROLE_SYSTEM,
  CHAT_ROLE_USER,
  CHAT_ROLE_ASSISTANT,
  CHAT_ROLE_TOOL,
} chat_role_t;

typedef struct chat_history_t {
  // Request being rendered.
  loom_serve_qwen_chat_t* chat;
  // Decoded message content and argument scratch.
  iree_string_builder_t scratch;
  // Decoded JSON argument document, stable while its values are rendered.
  iree_string_builder_t arguments;
  // Number of user messages establishing a real query.
  iree_host_size_t user_count;
  // Previous role; consecutive tools share one user role envelope.
  chat_role_t last_role;
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
  iree_string_view_t arguments_cursor =
      iree_string_builder_view(&history->arguments);
  iree_string_view_t arguments;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&arguments_cursor, &arguments));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&arguments_cursor));
  if (arguments_cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing data in assistant tool arguments");
  }
  if (index) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(&history->chat->prompt, "\n"));
  }
  chat_argument_render_t context = {&history->chat->prompt, &history->scratch};
  return chat_function_render(function[0].value, arguments, &context);
}

static iree_status_t chat_history_message(void* user_data,
                                          iree_host_size_t index,
                                          iree_json_value_type_t type,
                                          iree_string_view_t value) {
  chat_history_t* history = user_data;
  iree_string_builder_t* output = &history->chat->prompt;
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
  const iree_string_view_t role = fields[0].value;
  const bool is_tool = iree_string_view_equal(role, IREE_SV("tool"));
  const bool is_assistant = iree_string_view_equal(role, IREE_SV("assistant"));
  if ((!is_assistant && (fields[2].value.data || fields[3].value.data)) ||
      (!is_tool && fields[4].value.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "chat fields do not match their message role");
  }
  if (!index && history->chat->tool_count &&
      !iree_string_view_equal(role, IREE_SV("system"))) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, "<|im_end|>\n"));
  }
  if (history->last_role == CHAT_ROLE_TOOL && !is_tool) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, "<|im_end|>\n"));
  }
  if (is_assistant) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        output, "<|im_start|>assistant\n<think>\n"));
    IREE_RETURN_IF_ERROR(chat_content(&fields[2], &history->scratch));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        output,
        iree_string_view_trim(iree_string_builder_view(&history->scratch))));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, "\n</think>\n\n"));
    IREE_RETURN_IF_ERROR(chat_content(&fields[1], &history->scratch));
    iree_string_view_t content =
        iree_string_view_trim(iree_string_builder_view(&history->scratch));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, content));
    if (fields[3].value.data) {
      iree_host_size_t count = 0;
      IREE_RETURN_IF_ERROR(iree_json_array_length(fields[3].value, &count));
      if (count && content.size) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(output, "\n\n"));
      }
      IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
          fields[3].value, chat_history_tool, history));
    }
    history->last_role = CHAT_ROLE_ASSISTANT;
    return iree_string_builder_append_cstring(output, "<|im_end|>\n");
  }
  IREE_RETURN_IF_ERROR(chat_content(&fields[1], &history->scratch));
  const iree_string_view_t content =
      iree_string_view_trim(iree_string_builder_view(&history->scratch));
  if (iree_string_view_equal(role, IREE_SV("system"))) {
    if (index) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "system message must be first");
    }
    history->last_role = CHAT_ROLE_SYSTEM;
    if (history->chat->tool_count) {
      if (content.size) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(output, "\n\n"));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(output, content));
      }
      return iree_string_builder_append_cstring(output, "<|im_end|>\n");
    }
    if (!content.size) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, "<|im_start|>system\n"));
  } else {
    if (is_tool) {
      IREE_RETURN_IF_ERROR(chat_required(&fields[4]));
      if (history->last_role != CHAT_ROLE_TOOL) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(output, "<|im_start|>user"));
      }
      history->last_role = CHAT_ROLE_TOOL;
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(output, "\n<tool_response>\n"));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, content));
      return iree_string_builder_append_cstring(output, "\n</tool_response>");
    }
    if (!iree_string_view_equal(role, IREE_SV("user"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported chat message role");
    }
    ++history->user_count;
    history->last_role = CHAT_ROLE_USER;
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, "<|im_start|>user\n"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, content));
  return iree_string_builder_append_cstring(output, "<|im_end|>\n");
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
      {"model", CHAT_STRING},        {"messages", CHAT_ARRAY},
      {"stream", CHAT_BOOL},         {"tools", CHAT_ARRAY},
      {"max_tokens", CHAT_NUMBER},   {"max_completion_tokens", CHAT_NUMBER},
      {"temperature", CHAT_NUMBER},  {"stream_options", CHAT_OBJECT},
      {"store", CHAT_BOOL},          {"tool_choice", CHAT_STRING},
      {"enable_thinking", CHAT_BOOL}};
  IREE_RETURN_IF_ERROR(
      chat_fields_read(object, IREE_ARRAYSIZE(fields), fields, 0));
  IREE_RETURN_IF_ERROR(chat_required(&fields[1]));
  if (!iree_string_view_equal(fields[0].value,
                              IREE_SV(LOOM_SERVE_QWEN_CHAT_MODEL)) ||
      fields[2].type != IREE_JSON_VALUE_TYPE_TRUE || !fields[2].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "select model %s and stream=true",
                            LOOM_SERVE_QWEN_CHAT_MODEL);
  }
  if ((fields[8].value.data && fields[8].type != IREE_JSON_VALUE_TYPE_FALSE) ||
      (fields[10].value.data &&
       fields[10].type != IREE_JSON_VALUE_TYPE_FALSE) ||
      (fields[9].value.data &&
       !iree_string_view_equal(fields[9].value, IREE_SV("auto")))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only store=false, enable_thinking=false and "
                            "automatic tools are supported");
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
  if (fields[3].value.data) {
    iree_host_size_t count = 0;
    IREE_RETURN_IF_ERROR(iree_json_array_length(fields[3].value, &count));
    if (count) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          &history->chat->prompt,
          "<|im_start|>system\n# Tools\n\nYou have access to the following "
          "functions:\n\n<tools>"));
      IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
          fields[3].value, chat_tool_schema, history->chat));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          &history->chat->prompt, chat_tools_instructions));
    }
  }
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      fields[1].value, chat_history_message, history));
  if (!history->user_count || history->last_role == CHAT_ROLE_ASSISTANT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "chat must contain a user query and end with user or tool input");
  }
  if (history->last_role == CHAT_ROLE_TOOL) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &history->chat->prompt, "<|im_end|>\n"));
  }
  return iree_string_builder_append_cstring(&history->chat->prompt,
                                            chat_assistant_prefix);
}

iree_status_t loom_serve_qwen_chat_initialize(
    iree_string_view_t body, iree_host_size_t default_max_tokens,
    iree_allocator_t host_allocator, loom_serve_qwen_chat_t* out_chat) {
  memset(out_chat, 0, sizeof(*out_chat));
  out_chat->max_tokens = default_max_tokens;
  iree_string_builder_initialize(host_allocator, &out_chat->prompt);
  chat_history_t history = {.chat = out_chat};
  iree_string_builder_initialize(host_allocator, &history.scratch);
  iree_string_builder_initialize(host_allocator, &history.arguments);
  iree_status_t status = chat_request_render(body, &history);
  iree_string_builder_deinitialize(&history.arguments);
  iree_string_builder_deinitialize(&history.scratch);
  if (!iree_status_is_ok(status)) {
    loom_serve_qwen_chat_deinitialize(out_chat);
  }
  return status;
}

void loom_serve_qwen_chat_deinitialize(loom_serve_qwen_chat_t* chat) {
  iree_string_builder_deinitialize(&chat->prompt);
  memset(chat, 0, sizeof(*chat));
}

iree_host_size_t loom_serve_qwen_chat_text_end(iree_string_view_t response,
                                               iree_host_size_t previous_end) {
  const iree_string_view_t marker = IREE_SV("<tool_call>");
  const iree_host_size_t found =
      iree_string_view_find(response, marker, previous_end);
  if (found != IREE_STRING_VIEW_NPOS) {
    return found;
  }
  const iree_host_size_t remaining = response.size - previous_end;
  for (iree_host_size_t count = iree_min(marker.size - 1, remaining); count;
       --count) {
    if (memcmp(response.data + response.size - count, marker.data, count) ==
        0) {
      return response.size - count;
    }
  }
  return response.size;
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
    const loom_serve_qwen_chat_tool_t* tool, iree_string_view_t name,
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
    iree_string_view_t* cursor, const loom_serve_qwen_chat_tool_t* tool,
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

iree_status_t loom_serve_qwen_chat_complete(const loom_serve_qwen_chat_t* chat,
                                            iree_string_view_t response,
                                            uint64_t request_id,
                                            iree_string_builder_t* checkpoint,
                                            iree_string_builder_t* tool_calls,
                                            iree_host_size_t* out_tool_count) {
  *out_tool_count = 0;
  const iree_host_size_t tool_offset =
      iree_string_view_find(response, IREE_SV("<tool_call>"), 0);
  const iree_string_view_t content =
      iree_string_view_trim(iree_string_view_substr(response, 0, tool_offset));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      checkpoint, iree_string_builder_view(&chat->prompt)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(checkpoint, content));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(tool_calls, "["));
  iree_string_view_t cursor =
      tool_offset == IREE_STRING_VIEW_NPOS
          ? iree_string_view_empty()
          : iree_string_view_substr(response, tool_offset, IREE_HOST_SIZE_MAX);
  iree_string_builder_t arguments;
  iree_string_builder_t scratch;
  iree_string_builder_initialize(checkpoint->allocator, &arguments);
  iree_string_builder_initialize(checkpoint->allocator, &scratch);
  chat_argument_render_t render = {checkpoint, &scratch};
  iree_status_t status = iree_ok_status();
  while (cursor.size && iree_status_is_ok(status)) {
    status = chat_xml_consume(&cursor, IREE_SV("<tool_call>"));
    iree_string_view_t name = iree_string_view_empty();
    if (iree_status_is_ok(status)) {
      status = chat_xml_name(&cursor, IREE_SV("<function="), &name);
    }
    const loom_serve_qwen_chat_tool_t* tool = NULL;
    for (iree_host_size_t i = 0; i < chat->tool_count; ++i) {
      if (iree_string_view_equal(chat->tools[i].name, name)) {
        tool = &chat->tools[i];
      }
    }
    if (iree_status_is_ok(status) && (!tool || *out_tool_count == 16)) {
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
      const iree_host_size_t index = *out_tool_count;
      status =
          iree_string_builder_append_cstring(checkpoint, index          ? "\n"
                                                         : content.size ? "\n\n"
                                                                        : "");
      if (iree_status_is_ok(status)) {
        status = chat_function_render(
            name, iree_string_builder_view(&arguments), &render);
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_format(
            tool_calls,
            "%s{\"index\":%zu,\"id\":\"call_%" PRIu64
            "_%zu\",\"type\":\"function\",\"function\":{\"name\":",
            index ? "," : "", index, request_id, index);
      }
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
      ++*out_tool_count;
      cursor = iree_string_view_trim(cursor);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(tool_calls, "]");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(checkpoint, "<|im_end|>\n");
  }
  iree_string_builder_deinitialize(&scratch);
  iree_string_builder_deinitialize(&arguments);
  return status;
}

iree_status_t loom_serve_qwen_chat_event(uint64_t request_id,
                                         iree_string_view_t delta,
                                         iree_string_view_t finish_reason,
                                         iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "data: {\"id\":\"loom-%" PRIu64
      "\",\"object\":\"chat.completion.chunk\","
      "\"model\":\"" LOOM_SERVE_QWEN_CHAT_MODEL
      "\",\"choices\":[{\"index\":0,\"delta\":%.*s,\"finish_reason\":",
      request_id, (int)delta.size, delta.data));
  if (finish_reason.size) {
    IREE_RETURN_IF_ERROR(chat_quote(output, finish_reason));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  return iree_string_builder_append_cstring(output, "}]}\n\n");
}

iree_status_t loom_serve_qwen_chat_usage(uint64_t request_id,
                                         iree_host_size_t input_tokens,
                                         iree_host_size_t retained_tokens,
                                         iree_host_size_t output_tokens,
                                         iree_string_builder_t* output) {
  return iree_string_builder_append_format(
      output,
      "data: {\"id\":\"loom-%" PRIu64
      "\",\"object\":\"chat.completion.chunk\","
      "\"model\":\"" LOOM_SERVE_QWEN_CHAT_MODEL
      "\",\"choices\":[],\"usage\":{"
      "\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu,"
      "\"prompt_tokens_details\":{\"cached_tokens\":%zu}}}\n\n",
      request_id, input_tokens, output_tokens, input_tokens + output_tokens,
      retained_tokens);
}
