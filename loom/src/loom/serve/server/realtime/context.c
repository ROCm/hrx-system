// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/context.h"

#include <inttypes.h>

#include "loom/serve/server/realtime/input.h"
#include "loom/serve/server/realtime/item.h"
#include "loom/serve/server/realtime/settings.h"

enum {
  LOOM_SERVE_REALTIME_CONTEXT_ITEMS = 128,
  LOOM_SERVE_REALTIME_CONTEXT_BYTES = 1024 * 1024,
};

typedef struct loom_serve_realtime_context_parser_t {
  // Session identity lookup, borrowed only while accepting this command.
  loom_serve_realtime_conversation_t* conversation;
  // Owned output being constructed transactionally.
  loom_serve_realtime_context_t* context;
  // Last selected source member; adjacency preserves its original native group.
  const loom_serve_realtime_item_t* previous;
  // Decoded call IDs validated in this selected input scope, not default
  // history.
  iree_string_builder_t ids;
  // Offsets into ids for each earlier complete call.
  struct {
    // First byte of the decoded call identity.
    iree_host_size_t offset;
    // Length of that identity in bytes.
    iree_host_size_t length;
  } calls[LOOM_SERVE_REALTIME_CONTEXT_ITEMS];
  // Number of prior calls in this input scope.
  iree_host_size_t call_count;
  // Number of selected items, including omitted truncated content.
  iree_host_size_t item_count;
  // Explicit input establishes a new external call/result validation scope.
  bool validate_calls;
  // The current native message has not emitted its closing delimiter.
  bool open;
  // The current native message has an open tool_calls array.
  bool calls_open;
  // A preceding message or instruction requires a comma.
  bool comma;
} loom_serve_realtime_context_parser_t;

static iree_status_t loom_serve_realtime_context_close_message(
    loom_serve_realtime_context_parser_t* parser) {
  if (parser->open) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &parser->context->body, parser->calls_open ? "]}" : "}"));
    parser->open = parser->calls_open = false;
    parser->comma = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_context_charge(
    loom_serve_realtime_context_t* context, iree_host_size_t bytes) {
  if (bytes > LOOM_SERVE_REALTIME_CONTEXT_BYTES - context->bytes) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "response input exceeds 1 MiB of decoded content");
  }
  context->bytes += bytes;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_context_insert(
    loom_serve_realtime_context_t* context,
    loom_serve_realtime_input_item_t* input) {
  if (context->count == context->capacity) {
    const iree_host_size_t capacity =
        context->capacity ? context->capacity * 2 : 4;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        context->body.allocator, capacity * sizeof(*context->inserts),
        (void**)&context->inserts));
    context->capacity = capacity;
  }
  loom_serve_realtime_input_retain(input);
  context->inserts[context->count++] =
      (loom_serve_realtime_context_insert_t){context->body.size, input};
  return iree_ok_status();
}

static bool loom_serve_realtime_context_has_call(
    const loom_serve_realtime_context_parser_t* parser, iree_string_view_t id) {
  for (iree_host_size_t i = 0; i < parser->call_count; ++i) {
    if (iree_string_view_equal(
            id,
            iree_make_string_view(parser->ids.buffer + parser->calls[i].offset,
                                  parser->calls[i].length))) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_serve_realtime_context_append(
    loom_serve_realtime_context_parser_t* parser,
    loom_serve_realtime_item_kind_t kind, loom_serve_realtime_role_t role,
    iree_string_view_t text, iree_string_view_t call_id,
    iree_string_view_t name, loom_serve_realtime_input_item_t* input,
    bool joins) {
  loom_serve_realtime_context_t* context = parser->context;
  if (parser->validate_calls && kind != LOOM_SERVE_REALTIME_ITEM_MESSAGE) {
    const bool found = loom_serve_realtime_context_has_call(parser, call_id);
    if ((kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL && found) ||
        (kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_OUTPUT && !found)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "response input requires unique calls and "
                              "preceding complete calls for results");
    }
    if (kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL) {
      parser->calls[parser->call_count].offset = parser->ids.size;
      parser->calls[parser->call_count++].length = call_id.size;
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(&parser->ids, call_id));
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_context_charge(
      context, text.size + call_id.size + name.size));
  if (!joins) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_context_close_message(parser));
  }
  iree_string_builder_t* output = &context->body;
  if (!parser->open) {
    static const char* const roles[] = {"user", "assistant", "system"};
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "%s{\"role\":\"%s\",\"content\":", parser->comma ? "," : "",
        kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_OUTPUT ? "tool"
                                                         : roles[role]));
    if (input) {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_context_insert(context, input));
    } else {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
          output, kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL
                      ? iree_string_view_empty()
                      : text));
    }
    if (kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_OUTPUT) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(output, ",\"tool_call_id\":"));
      IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, call_id));
    }
    parser->open = true;
  }
  if (kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        output, parser->calls_open ? "," : ",\"tool_calls\":["));
    parser->calls_open = true;
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, "{\"id\":"));
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, call_id));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        output, ",\"type\":\"function\",\"function\":{\"name\":"));
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, name));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"arguments\":"));
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, text));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}}"));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_context_reference(
    loom_serve_realtime_context_parser_t* parser,
    const loom_serve_realtime_item_t* item, bool explicit_reference) {
  if (++parser->item_count > LOOM_SERVE_REALTIME_CONTEXT_ITEMS) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "response input exceeds 128 items");
  }
  const bool joins = parser->previous && parser->previous->next == item;
  parser->previous = item;
  if (!joins) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_context_close_message(parser));
  }
  if (item->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL &&
      item->status != LOOM_SERVE_REALTIME_ITEM_COMPLETED) {
    return explicit_reference
               ? iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "response input call is not complete")
               : iree_ok_status();
  }
  if (item->kind == LOOM_SERVE_REALTIME_ITEM_MESSAGE && !item->has_transcript) {
    return iree_ok_status();
  }
  return loom_serve_realtime_context_append(
      parser, item->kind, item->role, iree_string_builder_view(&item->text),
      iree_string_builder_view(&item->call_id),
      iree_string_builder_view(&item->name), item->input, joins);
}

static iree_status_t loom_serve_realtime_context_visit(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_realtime_context_parser_t* parser = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "response input requires item objects");
  }
  iree_string_view_t kind;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(value, IREE_SV("type"), &kind));
  if (loom_serve_realtime_string_is(kind, IREE_SV("item_reference"))) {
    static const loom_serve_realtime_field_rule_t rules[] = {
        {"type", 1u << IREE_JSON_VALUE_TYPE_STRING},
        {"id", 1u << IREE_JSON_VALUE_TYPE_STRING}};
    loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
    uint32_t present = 0;
    IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
        value, IREE_ARRAYSIZE(rules), rules, values, &present));
    if (present != 3) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "item_reference requires an id");
    }
    iree_string_builder_t id;
    iree_string_builder_initialize(parser->context->body.allocator, &id);
    iree_status_t status =
        loom_serve_realtime_string_decode(values[1].text, 128, &id);
    if (iree_status_is_ok(status)) {
      const loom_serve_realtime_item_t* item =
          loom_serve_realtime_conversation_find(parser->conversation,
                                                iree_string_builder_view(&id));
      status = item ? loom_serve_realtime_context_reference(parser, item, true)
                    : iree_make_status(IREE_STATUS_NOT_FOUND,
                                       "response input item was not found");
    }
    iree_string_builder_deinitialize(&id);
    return status;
  }
  if (++parser->item_count > LOOM_SERVE_REALTIME_CONTEXT_ITEMS) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "response input exceeds 128 items");
  }
  parser->previous = NULL;
  loom_serve_realtime_item_input_t input;
  iree_status_t status = loom_serve_realtime_item_input_initialize(
      value, parser->context->body.allocator, &input);
  char generated[64];
  iree_string_view_t call_id = iree_string_builder_view(&input.call_id);
  if (iree_status_is_ok(status) &&
      input.kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL && !call_id.size) {
    do {
      const int length = iree_snprintf(
          generated, sizeof(generated), "call_input_%" PRIu64 "_%" PRIu64,
          parser->conversation->session_id, ++parser->conversation->identity);
      call_id = iree_make_string_view(generated, (iree_host_size_t)length);
    } while (loom_serve_realtime_context_has_call(parser, call_id));
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_context_append(
        parser, input.kind, input.role, iree_string_builder_view(&input.text),
        call_id, input.name, NULL, false);
  }
  loom_serve_realtime_item_input_deinitialize(&input);
  return status;
}

iree_status_t loom_serve_realtime_context_initialize(
    loom_serve_realtime_conversation_t* conversation, iree_string_view_t input,
    iree_string_view_t model, iree_string_view_t instructions,
    iree_string_view_t tools, uint32_t max_tokens, uint64_t seed,
    loom_serve_realtime_context_t* context) {
  *context = (loom_serve_realtime_context_t){0};
  iree_string_builder_initialize(conversation->host_allocator, &context->body);
  iree_string_builder_t* output = &context->body;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"model\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, model));
  if (tools.size) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"tools\":"));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, tools));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"stream\":true,\"max_tokens\":%" PRIu32 ",\"seed\":%" PRIu64
      ",\"messages\":[",
      max_tokens, seed));
  if (instructions.size) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        output, "{\"role\":\"system\",\"content\":"));
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_string_write(output, instructions));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
  }
  loom_serve_realtime_context_parser_t parser = {
      .conversation = conversation,
      .context = context,
      .validate_calls = input.data != NULL,
      .comma = instructions.size != 0};
  iree_string_builder_initialize(conversation->host_allocator, &parser.ids);
  iree_status_t status = iree_ok_status();
  if (input.data) {
    status = iree_json_enumerate_array_typed(
        input, loom_serve_realtime_context_visit, &parser);
  } else {
    for (const loom_serve_realtime_group_t* group = conversation->history.first;
         group && iree_status_is_ok(status); group = group->next) {
      for (const loom_serve_realtime_item_t* item = group->first;
           item && iree_status_is_ok(status); item = item->next) {
        status = loom_serve_realtime_context_reference(&parser, item, false);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_context_close_message(&parser);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "]}");
  }
  iree_string_builder_deinitialize(&parser.ids);
  return status;
}

void loom_serve_realtime_context_deinitialize(
    loom_serve_realtime_context_t* context) {
  for (iree_host_size_t i = 0; i < context->count; ++i) {
    loom_serve_realtime_input_release(context->inserts[i].input);
  }
  iree_allocator_free(context->body.allocator, context->inserts);
  iree_string_builder_deinitialize(&context->body);
  *context = (loom_serve_realtime_context_t){0};
}

iree_status_t loom_serve_realtime_context_ready(
    const loom_serve_realtime_context_t* context, bool* out_ready) {
  *out_ready = true;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < context->count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_realtime_input_status(context->inserts[i].input);
    *out_ready &= loom_serve_realtime_input_complete(context->inserts[i].input);
  }
  return status;
}

iree_status_t loom_serve_realtime_context_take(
    loom_serve_realtime_context_t* context, iree_string_builder_t* out_body) {
  if (context->count) {
    iree_string_builder_t body;
    iree_string_builder_initialize(context->body.allocator, &body);
    iree_host_size_t offset = 0;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < context->count && iree_status_is_ok(status); ++i) {
      const loom_serve_realtime_context_insert_t* insert = &context->inserts[i];
      const iree_string_builder_t* transcript =
          loom_serve_realtime_input_transcript(insert->input);
      status = loom_serve_realtime_context_charge(context, transcript->size);
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_string(
            &body, iree_make_string_view(context->body.buffer + offset,
                                         insert->offset - offset));
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_realtime_string_write(
            &body, iree_string_builder_view(transcript));
      }
      offset = insert->offset;
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(
          &body, iree_make_string_view(context->body.buffer + offset,
                                       context->body.size - offset));
    }
    if (!iree_status_is_ok(status)) {
      iree_string_builder_deinitialize(&body);
      return status;
    }
    iree_string_builder_deinitialize(&context->body);
    context->body = body;
  }
  *out_body = context->body;
  iree_string_builder_initialize(out_body->allocator, &context->body);
  loom_serve_realtime_context_deinitialize(context);
  return iree_ok_status();
}
