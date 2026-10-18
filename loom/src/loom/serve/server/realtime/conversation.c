// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/conversation.h"

#include <inttypes.h>

static const char* const loom_serve_realtime_roles[] = {"user", "assistant",
                                                        "system"};

void loom_serve_realtime_conversation_initialize(
    uint64_t session_id, iree_allocator_t host_allocator,
    loom_serve_realtime_conversation_t* conversation) {
  *conversation = (loom_serve_realtime_conversation_t){
      .host_allocator = host_allocator, .session_id = session_id};
}

static void loom_serve_realtime_group_free(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* group) {
  if (group->previous) {
    group->previous->next = group->next;
  } else {
    conversation->first = group->next;
  }
  if (group->next) {
    group->next->previous = group->previous;
  } else {
    conversation->last = group->previous;
  }
  iree_allocator_free(conversation->host_allocator, group);
}

void loom_serve_realtime_conversation_deinitialize(
    loom_serve_realtime_conversation_t* conversation) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(conversation->items); ++i) {
    if (conversation->items[i]) {
      loom_serve_realtime_conversation_remove(conversation,
                                              conversation->items[i]);
    }
  }
  while (conversation->first) {
    loom_serve_realtime_conversation_release(conversation, conversation->first);
  }
}

static loom_serve_realtime_item_t* loom_serve_realtime_item_find(
    const loom_serve_realtime_conversation_t* conversation,
    iree_string_view_t id) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(conversation->items); ++i) {
    loom_serve_realtime_item_t* item = conversation->items[i];
    if (item &&
        iree_string_view_equal(iree_string_builder_view(&item->id), id)) {
      return item;
    }
  }
  return NULL;
}

loom_serve_realtime_item_t* loom_serve_realtime_conversation_find(
    const loom_serve_realtime_conversation_t* conversation,
    iree_string_view_t id) {
  loom_serve_realtime_item_t* item =
      loom_serve_realtime_item_find(conversation, id);
  return item && item->group ? item : NULL;
}

static iree_status_t loom_serve_realtime_item_allocate(
    loom_serve_realtime_conversation_t* conversation, iree_string_view_t id,
    loom_serve_realtime_item_kind_t kind, loom_serve_realtime_role_t role,
    bool audio, bool reserved_capacity, loom_serve_realtime_item_t** out_item) {
  *out_item = NULL;
  char generated_id[64];
  if (!id.size) {
    do {
      const int length = iree_snprintf(
          generated_id, sizeof(generated_id), "item_%" PRIu64 "_%" PRIu64,
          conversation->session_id, ++conversation->identity);
      id = iree_make_string_view(generated_id, (iree_host_size_t)length);
    } while (loom_serve_realtime_item_find(conversation, id));
  } else if (id.size > 128 || loom_serve_realtime_item_find(conversation, id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "conversation item requires a unique ID of 1-128 bytes");
  }
  if ((!reserved_capacity &&
       conversation->count == IREE_ARRAYSIZE(conversation->items)) ||
      id.size > 1024 * 1024 - conversation->bytes) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "conversation storage is full; delete earlier items");
  }
  loom_serve_realtime_item_t* item = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(conversation->host_allocator,
                                             sizeof(*item), (void**)&item));
  iree_string_builder_initialize(conversation->host_allocator, &item->id);
  iree_string_builder_initialize(conversation->host_allocator, &item->text);
  iree_string_builder_initialize(conversation->host_allocator, &item->call_id);
  iree_string_builder_initialize(conversation->host_allocator, &item->name);
  iree_status_t status = iree_string_builder_append_string(&item->id, id);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&item->id);
    iree_allocator_free(conversation->host_allocator, item);
    return status;
  }
  item->kind = kind;
  item->role = role;
  item->audio = audio;
  item->has_transcript = true;
  while (conversation->items[item->slot]) {
    ++item->slot;
  }
  conversation->items[item->slot] = item;
  conversation->count += !reserved_capacity;
  conversation->bytes += id.size;
  *out_item = item;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_conversation_reserve(
    loom_serve_realtime_conversation_t* conversation, iree_string_view_t id,
    loom_serve_realtime_item_kind_t kind, loom_serve_realtime_role_t role,
    bool audio, loom_serve_realtime_item_t** out_item) {
  return loom_serve_realtime_item_allocate(conversation, id, kind, role, audio,
                                           false, out_item);
}

static iree_status_t loom_serve_realtime_group_allocate(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t** out_group) {
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      conversation->host_allocator, sizeof(**out_group), (void**)out_group));
  (*out_group)->sequence = ++conversation->sequence;
  return iree_ok_status();
}

static void loom_serve_realtime_group_insert(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* after, loom_serve_realtime_group_t* group) {
  group->previous = after;
  group->next = after ? after->next : conversation->first;
  if (after) {
    after->next = group;
  } else {
    conversation->first = group;
  }
  if (group->next) {
    group->next->previous = group;
  } else {
    conversation->last = group;
  }
}

static void loom_serve_realtime_group_append(loom_serve_realtime_group_t* group,
                                             loom_serve_realtime_item_t* item) {
  item->group = group;
  item->previous = group->last;
  item->next = NULL;
  if (group->last) {
    group->last->next = item;
  } else {
    group->first = item;
  }
  group->last = item;
  if (item->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL) {
    group->completed_calls +=
        item->status == LOOM_SERVE_REALTIME_ITEM_COMPLETED;
  } else {
    group->message = item;
  }
}

// The insertion boundary owns validation of a client-supplied result. Creation
// sequence is not canonical order after edits, so walk the bounded preceding
// history once here; no call pointer survives this command.
static iree_status_t loom_serve_realtime_result_validate(
    const loom_serve_realtime_item_t* after,
    const loom_serve_realtime_item_t* item) {
  if (item->kind != LOOM_SERVE_REALTIME_ITEM_FUNCTION_OUTPUT) {
    return iree_ok_status();
  }
  for (const loom_serve_realtime_item_t* previous = after; previous;
       previous =
           previous->previous
               ? previous->previous
               : loom_serve_realtime_conversation_previous(previous->group)) {
    if (previous->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL &&
        previous->status == LOOM_SERVE_REALTIME_ITEM_COMPLETED &&
        iree_string_view_equal(iree_string_builder_view(&previous->call_id),
                               iree_string_builder_view(&item->call_id))) {
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "function result requires a preceding completed call");
}

iree_status_t loom_serve_realtime_conversation_insert(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* after, loom_serve_realtime_item_t* item) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_result_validate(
      !after        ? NULL
      : after->last ? after->last
                    : loom_serve_realtime_conversation_previous(after),
      item));
  loom_serve_realtime_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_group_allocate(conversation, &group));
  loom_serve_realtime_group_insert(conversation, after, group);
  loom_serve_realtime_group_append(group, item);
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_conversation_insert_after(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* after, loom_serve_realtime_item_t* item) {
  if (!after || !after->next) {
    return loom_serve_realtime_conversation_insert(
        conversation, after ? after->group : NULL, item);
  }
  if (after->group->retained) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "response members remain owned until retirement");
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_result_validate(after, item));
  loom_serve_realtime_group_t* inserted = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_group_allocate(conversation, &inserted));
  loom_serve_realtime_group_t* suffix = NULL;
  iree_status_t status =
      loom_serve_realtime_group_allocate(conversation, &suffix);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(conversation->host_allocator, inserted);
    return status;
  }
  loom_serve_realtime_group_t* prefix = after->group;
  loom_serve_realtime_item_t* member = after->next;
  after->next = NULL;
  prefix->last = after;
  while (member) {
    loom_serve_realtime_item_t* next = member->next;
    if (member->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL) {
      prefix->completed_calls -=
          member->status == LOOM_SERVE_REALTIME_ITEM_COMPLETED;
    } else {
      prefix->message = NULL;
    }
    loom_serve_realtime_group_append(suffix, member);
    member = next;
  }
  loom_serve_realtime_group_insert(conversation, prefix, inserted);
  loom_serve_realtime_group_append(inserted, item);
  loom_serve_realtime_group_insert(conversation, inserted, suffix);
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_conversation_begin(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t** out_group) {
  *out_group = NULL;
  if (conversation->count == IREE_ARRAYSIZE(conversation->items)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "conversation storage is full; delete earlier items");
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_group_allocate(conversation, out_group));
  loom_serve_realtime_group_insert(conversation, conversation->last,
                                   *out_group);
  (*out_group)->retained = true;
  ++conversation->count;
  return iree_ok_status();
}

void loom_serve_realtime_conversation_release(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* group) {
  group->retained = false;
  if (!group->first) {
    --conversation->count;
    loom_serve_realtime_group_free(conversation, group);
  }
}

iree_status_t loom_serve_realtime_conversation_add(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* group, loom_serve_realtime_item_kind_t kind,
    loom_serve_realtime_role_t role, bool audio,
    loom_serve_realtime_item_t** out_item) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_item_allocate(
      conversation, iree_string_view_empty(), kind, role, audio, !group->first,
      out_item));
  loom_serve_realtime_group_append(group, *out_item);
  return iree_ok_status();
}

loom_serve_realtime_item_t* loom_serve_realtime_conversation_previous(
    const loom_serve_realtime_group_t* group) {
  for (group = group->previous; group; group = group->previous) {
    if (group->last) {
      return group->last;
    }
  }
  return NULL;
}

loom_serve_realtime_item_t* loom_serve_realtime_conversation_last_item(
    const loom_serve_realtime_conversation_t* conversation) {
  const loom_serve_realtime_group_t* group = conversation->last;
  return !group        ? NULL
         : group->last ? group->last
                       : loom_serve_realtime_conversation_previous(group);
}

static iree_status_t loom_serve_realtime_conversation_append_string(
    loom_serve_realtime_conversation_t* conversation,
    iree_string_builder_t* storage, iree_string_view_t value) {
  if (value.size > 1024 * 1024 - conversation->bytes) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "conversation text storage is full");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(storage, value));
  conversation->bytes += value.size;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_conversation_append_text(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item, iree_string_view_t text) {
  return loom_serve_realtime_conversation_append_string(conversation,
                                                        &item->text, text);
}

static bool loom_serve_realtime_call_id_used(
    const loom_serve_realtime_conversation_t* conversation,
    iree_string_view_t id) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(conversation->items); ++i) {
    const loom_serve_realtime_item_t* item = conversation->items[i];
    if (item && item->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL &&
        iree_string_view_equal(iree_string_builder_view(&item->call_id), id)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_serve_realtime_conversation_function(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item, iree_string_view_t call_id,
    iree_string_view_t name, iree_string_view_t text) {
  char generated[64];
  if (!call_id.size && item->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL) {
    do {
      const int length = iree_snprintf(
          generated, sizeof(generated), "call_item_%" PRIu64 "_%" PRIu64,
          conversation->session_id, ++conversation->identity);
      call_id = iree_make_string_view(generated, (iree_host_size_t)length);
    } while (loom_serve_realtime_call_id_used(conversation, call_id));
  }
  if (!call_id.size || call_id.size > 128 ||
      (item->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL &&
       loom_serve_realtime_call_id_used(conversation, call_id))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "call_id requires 1-128 bytes and identifies one call");
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_append_string(
      conversation, &item->call_id, call_id));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_append_string(
      conversation, &item->name, name));
  return loom_serve_realtime_conversation_append_text(conversation, item, text);
}

void loom_serve_realtime_item_complete(loom_serve_realtime_item_t* item) {
  if (item->group && item->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL &&
      item->status != LOOM_SERVE_REALTIME_ITEM_COMPLETED) {
    ++item->group->completed_calls;
  }
  item->status = LOOM_SERVE_REALTIME_ITEM_COMPLETED;
}

void loom_serve_realtime_conversation_remove(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item) {
  loom_serve_realtime_group_t* group = item->group;
  if (group) {
    if (item->previous) {
      item->previous->next = item->next;
    } else {
      group->first = item->next;
    }
    if (item->next) {
      item->next->previous = item->previous;
    } else {
      group->last = item->previous;
    }
    if (group->message == item) {
      group->message = NULL;
    }
    if (item->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL) {
      group->completed_calls -=
          item->status == LOOM_SERVE_REALTIME_ITEM_COMPLETED;
    }
    if (group->first || !group->retained) {
      --conversation->count;
    }
    if (!group->first && !group->retained) {
      loom_serve_realtime_group_free(conversation, group);
    }
  } else {
    --conversation->count;
  }
  conversation->items[item->slot] = NULL;
  conversation->bytes -=
      item->text.size + item->id.size + item->call_id.size + item->name.size;
  iree_string_builder_deinitialize(&item->text);
  iree_string_builder_deinitialize(&item->id);
  iree_string_builder_deinitialize(&item->call_id);
  iree_string_builder_deinitialize(&item->name);
  iree_allocator_free(conversation->host_allocator, item);
}

void loom_serve_realtime_conversation_truncate(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item) {
  conversation->bytes -= item->text.size;
  iree_string_builder_reset(&item->text);
  item->has_transcript = false;
}

static iree_status_t loom_serve_realtime_message_write(
    iree_string_builder_t* output, const char* role, iree_string_view_t text,
    bool comma) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "%s{\"role\":\"%s\",\"content\":", comma ? "," : "", role));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, text));
  return iree_string_builder_append_cstring(output, "}");
}

static iree_status_t loom_serve_realtime_group_write(
    const loom_serve_realtime_group_t* group, bool comma,
    iree_string_builder_t* output) {
  const loom_serve_realtime_item_t* message = group->message;
  const bool result =
      message && message->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_OUTPUT;
  const char* role = result    ? "tool"
                     : message ? loom_serve_realtime_roles[message->role]
                               : "assistant";
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "%s{\"role\":\"%s\",\"content\":", comma ? "," : "", role));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, message && message->has_transcript
                  ? iree_string_builder_view(&message->text)
                  : iree_string_view_empty()));
  if (result) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"tool_call_id\":"));
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
        output, iree_string_builder_view(&message->call_id)));
  }
  if (group->completed_calls) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"tool_calls\":["));
    bool call_comma = false;
    iree_status_t status = iree_ok_status();
    for (const loom_serve_realtime_item_t* item = group->first;
         item && iree_status_is_ok(status); item = item->next) {
      if (item->kind != LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL ||
          item->status != LOOM_SERVE_REALTIME_ITEM_COMPLETED) {
        continue;
      }
      status = iree_string_builder_append_cstring(
          output, call_comma ? ",{\"id\":" : "{\"id\":");
      if (iree_status_is_ok(status)) {
        status = loom_serve_realtime_string_write(
            output, iree_string_builder_view(&item->call_id));
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(
            output, ",\"type\":\"function\",\"function\":{\"name\":");
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_realtime_string_write(
            output, iree_string_builder_view(&item->name));
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(output, ",\"arguments\":");
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_realtime_string_write(
            output, iree_string_builder_view(&item->text));
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(output, "}}");
      }
      call_comma = true;
    }
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "]"));
  }
  return iree_string_builder_append_cstring(output, "}");
}

iree_status_t loom_serve_realtime_conversation_request(
    const loom_serve_realtime_conversation_t* conversation,
    const loom_serve_realtime_group_t* exclude, iree_string_view_t model,
    iree_string_view_t instructions, iree_string_view_t tools,
    uint32_t max_tokens, uint64_t seed, iree_string_builder_t* output) {
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
  bool comma = false;
  if (instructions.size) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_message_write(
        output, "system", instructions, false));
    comma = true;
  }
  iree_status_t status = iree_ok_status();
  for (const loom_serve_realtime_group_t* group = conversation->first;
       group && group != exclude && iree_status_is_ok(status);
       group = group->next) {
    if ((!group->message || !group->message->has_transcript) &&
        !group->completed_calls) {
      continue;
    }
    status = loom_serve_realtime_group_write(group, comma, output);
    comma = true;
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "]}");
  }
  return status;
}
