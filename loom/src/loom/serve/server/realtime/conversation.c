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
    loom_serve_realtime_role_t role, bool audio, bool reserved_capacity,
    loom_serve_realtime_item_t** out_item) {
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
  iree_status_t status = iree_string_builder_append_string(&item->id, id);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&item->id);
    iree_allocator_free(conversation->host_allocator, item);
    return status;
  }
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
    loom_serve_realtime_role_t role, bool audio,
    loom_serve_realtime_item_t** out_item) {
  return loom_serve_realtime_item_allocate(conversation, id, role, audio, false,
                                           out_item);
}

static iree_status_t loom_serve_realtime_group_allocate(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* after,
    loom_serve_realtime_group_t** out_group) {
  *out_group = NULL;
  loom_serve_realtime_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(conversation->host_allocator,
                                             sizeof(*group), (void**)&group));
  group->sequence = ++conversation->sequence;
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
  *out_group = group;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_conversation_insert(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* after, loom_serve_realtime_item_t* item) {
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_group_allocate(conversation, after, &item->group));
  item->group->item = item;
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
  IREE_RETURN_IF_ERROR(loom_serve_realtime_group_allocate(
      conversation, conversation->last, out_group));
  (*out_group)->retained = true;
  ++conversation->count;
  return iree_ok_status();
}

void loom_serve_realtime_conversation_release(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* group) {
  group->retained = false;
  if (!group->item) {
    --conversation->count;
    loom_serve_realtime_group_free(conversation, group);
  }
}

iree_status_t loom_serve_realtime_conversation_add(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* group, loom_serve_realtime_role_t role,
    bool audio, loom_serve_realtime_item_t** out_item) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_item_allocate(
      conversation, iree_string_view_empty(), role, audio, true, out_item));
  group->item = *out_item;
  (*out_item)->group = group;
  return iree_ok_status();
}

loom_serve_realtime_item_t* loom_serve_realtime_conversation_previous(
    const loom_serve_realtime_group_t* group) {
  for (group = group->previous; group; group = group->previous) {
    if (group->item) {
      return group->item;
    }
  }
  return NULL;
}

loom_serve_realtime_item_t* loom_serve_realtime_conversation_last_item(
    const loom_serve_realtime_conversation_t* conversation) {
  const loom_serve_realtime_group_t* group = conversation->last;
  return !group        ? NULL
         : group->item ? group->item
                       : loom_serve_realtime_conversation_previous(group);
}

iree_status_t loom_serve_realtime_conversation_append_text(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item, iree_string_view_t text) {
  if (text.size > 1024 * 1024 - conversation->bytes) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "conversation transcript storage is full");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(&item->text, text));
  conversation->bytes += text.size;
  return iree_ok_status();
}

void loom_serve_realtime_conversation_remove(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item) {
  loom_serve_realtime_group_t* group = item->group;
  if (group) {
    group->item = NULL;
    if (!group->retained) {
      loom_serve_realtime_group_free(conversation, group);
      --conversation->count;
    }
  } else {
    --conversation->count;
  }
  conversation->items[item->slot] = NULL;
  conversation->bytes -= item->text.size + item->id.size;
  iree_string_builder_deinitialize(&item->text);
  iree_string_builder_deinitialize(&item->id);
  iree_allocator_free(conversation->host_allocator, item);
}

void loom_serve_realtime_conversation_truncate(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item) {
  conversation->bytes -= item->text.size;
  iree_string_builder_reset(&item->text);
  item->has_transcript = false;
}

iree_status_t loom_serve_realtime_item_write(
    const loom_serve_realtime_item_t* item, iree_string_builder_t* output) {
  static const char* const statuses[] = {"in_progress", "completed",
                                         "incomplete"};
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "{\"id\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&item->id)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"type\":\"message\",\"role\":\"%s\",\"status\":\"%s\","
      "\"content\":[{\"type\":\"%s\",\"%s\":",
      loom_serve_realtime_roles[item->role], statuses[item->status],
      item->audio
          ? (item->role == LOOM_SERVE_REALTIME_USER ? "input_audio" : "audio")
          : (item->role == LOOM_SERVE_REALTIME_USER ? "input_text" : "text"),
      item->audio ? "transcript" : "text"));
  if (item->has_transcript) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
        output, iree_string_builder_view(&item->text)));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  return iree_string_builder_append_cstring(output, "}]}");
}

static iree_status_t loom_serve_realtime_message_write(
    iree_string_builder_t* output, const char* role, iree_string_view_t text,
    bool comma) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "%s{\"role\":\"%s\",\"content\":", comma ? "," : "", role));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, text));
  return iree_string_builder_append_cstring(output, "}");
}

iree_status_t loom_serve_realtime_conversation_request(
    const loom_serve_realtime_conversation_t* conversation,
    const loom_serve_realtime_group_t* exclude, iree_string_view_t model,
    iree_string_view_t instructions, uint32_t max_tokens, uint64_t seed,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"model\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, model));
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
    const loom_serve_realtime_item_t* item = group->item;
    if (!item || !item->has_transcript) {
      continue;
    }
    status = loom_serve_realtime_message_write(
        output, loom_serve_realtime_roles[item->role],
        iree_string_builder_view(&item->text), comma);
    comma = true;
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "]}");
  }
  return status;
}
