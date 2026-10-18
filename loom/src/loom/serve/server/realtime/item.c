// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/item.h"

#include "loom/serve/server/realtime/settings.h"
#include "loom/serve/service/text/chat_fields.h"

static iree_status_t loom_serve_realtime_content_visit(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "conversation content requires text-part objects");
  }
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"text", 1u << IREE_JSON_VALUE_TYPE_STRING}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      value, IREE_ARRAYSIZE(rules), rules, values, &present));
  if (present != 3 ||
      (!loom_serve_realtime_string_is(values[0].text, IREE_SV("input_text")) &&
       !loom_serve_realtime_string_is(values[0].text, IREE_SV("text")))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "conversation creation currently accepts text parts");
  }
  iree_string_builder_t* output = user_data;
  iree_string_builder_t decoded;
  iree_string_builder_initialize(output->allocator, &decoded);
  iree_status_t status = loom_serve_realtime_string_decode(
      values[1].text, 1024 * 1024 - output->size, &decoded);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        output, iree_string_builder_view(&decoded));
  }
  iree_string_builder_deinitialize(&decoded);
  return status;
}

enum {
  LOOM_SERVE_REALTIME_ITEM_FIELD_TYPE,
  LOOM_SERVE_REALTIME_ITEM_FIELD_ID,
  LOOM_SERVE_REALTIME_ITEM_FIELD_ROLE,
  LOOM_SERVE_REALTIME_ITEM_FIELD_CONTENT,
  LOOM_SERVE_REALTIME_ITEM_FIELD_STATUS,
  LOOM_SERVE_REALTIME_ITEM_FIELD_OBJECT,
  LOOM_SERVE_REALTIME_ITEM_FIELD_CALL_ID,
  LOOM_SERVE_REALTIME_ITEM_FIELD_NAME,
  LOOM_SERVE_REALTIME_ITEM_FIELD_ARGUMENTS,
  LOOM_SERVE_REALTIME_ITEM_FIELD_OUTPUT,
  LOOM_SERVE_REALTIME_ITEM_FIELD_COUNT,
};

#define LOOM_REALTIME_ITEM_FIELD(name) \
  (1u << LOOM_SERVE_REALTIME_ITEM_FIELD_##name)

typedef struct loom_serve_realtime_item_rule_t {
  // Decoded public item type.
  const char* name;
  // Canonical contribution and wire representation.
  loom_serve_realtime_item_kind_t kind;
  // Required type-specific fields, in addition to type.
  uint32_t required;
  // Optional type-specific fields, in addition to common metadata.
  uint32_t optional;
} loom_serve_realtime_item_rule_t;

static const loom_serve_realtime_item_rule_t loom_serve_realtime_item_rules[] =
    {
        {"message", LOOM_SERVE_REALTIME_ITEM_MESSAGE,
         LOOM_REALTIME_ITEM_FIELD(ROLE) | LOOM_REALTIME_ITEM_FIELD(CONTENT), 0},
        {"function_call", LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL,
         LOOM_REALTIME_ITEM_FIELD(NAME) | LOOM_REALTIME_ITEM_FIELD(ARGUMENTS),
         LOOM_REALTIME_ITEM_FIELD(CALL_ID)},
        {"function_call_output", LOOM_SERVE_REALTIME_ITEM_FUNCTION_OUTPUT,
         LOOM_REALTIME_ITEM_FIELD(CALL_ID) | LOOM_REALTIME_ITEM_FIELD(OUTPUT),
         0},
};

iree_status_t loom_serve_realtime_item_input_initialize(
    iree_string_view_t object, iree_allocator_t host_allocator,
    loom_serve_realtime_item_input_t* input) {
  *input = (loom_serve_realtime_item_input_t){0};
  iree_string_builder_initialize(host_allocator, &input->id);
  iree_string_builder_initialize(host_allocator, &input->call_id);
  iree_string_builder_initialize(host_allocator, &input->text);
  iree_string_builder_t* id = &input->id;
  iree_string_builder_t* call_id = &input->call_id;
  iree_string_builder_t* text = &input->text;
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"id", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"role", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"content", 1u << IREE_JSON_VALUE_TYPE_ARRAY},
      {"status", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"object", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"call_id", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"name", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"arguments", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"output", 1u << IREE_JSON_VALUE_TYPE_STRING}};
  loom_serve_realtime_event_value_t
      values[LOOM_SERVE_REALTIME_ITEM_FIELD_COUNT];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(rules), rules, values, &present));
  const loom_serve_realtime_item_rule_t* rule = NULL;
  if (present & LOOM_REALTIME_ITEM_FIELD(TYPE)) {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(loom_serve_realtime_item_rules); ++i) {
      if (loom_serve_realtime_string_is(
              values[LOOM_SERVE_REALTIME_ITEM_FIELD_TYPE].text,
              iree_make_cstring_view(loom_serve_realtime_item_rules[i].name))) {
        rule = &loom_serve_realtime_item_rules[i];
        break;
      }
    }
  }
  const uint32_t common =
      LOOM_REALTIME_ITEM_FIELD(TYPE) | LOOM_REALTIME_ITEM_FIELD(ID) |
      LOOM_REALTIME_ITEM_FIELD(STATUS) | LOOM_REALTIME_ITEM_FIELD(OBJECT);
  if (!rule || (present & rule->required) != rule->required ||
      (present & ~(common | rule->required | rule->optional))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "item fields do not match its message or function type");
  }
  if ((present & LOOM_REALTIME_ITEM_FIELD(OBJECT)) &&
      !loom_serve_realtime_string_is(
          values[LOOM_SERVE_REALTIME_ITEM_FIELD_OBJECT].text,
          IREE_SV("realtime.item"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "item object must be realtime.item");
  }
  if (present & LOOM_REALTIME_ITEM_FIELD(STATUS)) {
    const iree_string_view_t status =
        values[LOOM_SERVE_REALTIME_ITEM_FIELD_STATUS].text;
    if (!loom_serve_realtime_string_is(status, IREE_SV("completed")) &&
        !loom_serve_realtime_string_is(status, IREE_SV("in_progress")) &&
        !loom_serve_realtime_string_is(status, IREE_SV("incomplete"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid item status");
    }
  }
  if (present & LOOM_REALTIME_ITEM_FIELD(ID)) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        values[LOOM_SERVE_REALTIME_ITEM_FIELD_ID].text, 128, id));
    if (!id->size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "item ID must not be empty");
    }
  }
  if (present & LOOM_REALTIME_ITEM_FIELD(CALL_ID)) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        values[LOOM_SERVE_REALTIME_ITEM_FIELD_CALL_ID].text, 128, call_id));
    if (!call_id->size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "call_id must not be empty");
    }
  }
  loom_serve_realtime_role_t role = LOOM_SERVE_REALTIME_ASSISTANT;
  iree_string_view_t name = iree_string_view_empty();
  if (rule->kind == LOOM_SERVE_REALTIME_ITEM_MESSAGE) {
    static const char* const roles[] = {"user", "assistant", "system"};
    iree_host_size_t index = 0;
    while (index < IREE_ARRAYSIZE(roles) &&
           !loom_serve_realtime_string_is(
               values[LOOM_SERVE_REALTIME_ITEM_FIELD_ROLE].text,
               iree_make_cstring_view(roles[index]))) {
      ++index;
    }
    if (index == IREE_ARRAYSIZE(roles)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported message role");
    }
    role = (loom_serve_realtime_role_t)index;
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
        values[LOOM_SERVE_REALTIME_ITEM_FIELD_CONTENT].text,
        loom_serve_realtime_content_visit, text));
  } else {
    const bool call = rule->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL;
    if (call) {
      name = values[LOOM_SERVE_REALTIME_ITEM_FIELD_NAME].text;
      if (!loom_serve_text_chat_is_identifier(name)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "function name requires a literal ASCII identifier");
      }
    }
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        values[call ? LOOM_SERVE_REALTIME_ITEM_FIELD_ARGUMENTS
                    : LOOM_SERVE_REALTIME_ITEM_FIELD_OUTPUT]
            .text,
        1024 * 1024, text));
  }
  input->kind = rule->kind;
  input->role = role;
  input->name = name;
  return iree_ok_status();
}

void loom_serve_realtime_item_input_deinitialize(
    loom_serve_realtime_item_input_t* input) {
  iree_string_builder_deinitialize(&input->id);
  iree_string_builder_deinitialize(&input->call_id);
  iree_string_builder_deinitialize(&input->text);
}

iree_status_t loom_serve_realtime_item_parse(
    loom_serve_realtime_conversation_t* conversation, iree_string_view_t object,
    loom_serve_realtime_item_t** out_item) {
  *out_item = NULL;
  loom_serve_realtime_item_input_t input;
  iree_status_t status = loom_serve_realtime_item_input_initialize(
      object, conversation->host_allocator, &input);
  loom_serve_realtime_item_t* item = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_conversation_reserve(
        conversation, iree_string_builder_view(&input.id), input.kind,
        input.role, false, &item);
  }
  if (iree_status_is_ok(status)) {
    status =
        input.kind == LOOM_SERVE_REALTIME_ITEM_MESSAGE
            ? loom_serve_realtime_conversation_append_text(
                  conversation, item, iree_string_builder_view(&input.text))
            : loom_serve_realtime_conversation_function(
                  conversation, item, iree_string_builder_view(&input.call_id),
                  input.name, iree_string_builder_view(&input.text));
  }
  loom_serve_realtime_item_input_deinitialize(&input);
  if (iree_status_is_ok(status)) {
    loom_serve_realtime_item_complete(item);
    *out_item = item;
  } else if (item) {
    loom_serve_realtime_conversation_remove(conversation, item);
  }
  return status;
}

#undef LOOM_REALTIME_ITEM_FIELD
