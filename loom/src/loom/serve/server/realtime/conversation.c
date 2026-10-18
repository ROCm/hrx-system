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
    iree_allocator_t host_allocator,
    loom_serve_realtime_conversation_t* conversation) {
  *conversation =
      (loom_serve_realtime_conversation_t){.host_allocator = host_allocator};
}

void loom_serve_realtime_conversation_deinitialize(
    loom_serve_realtime_conversation_t* conversation) {
  while (conversation->first) {
    loom_serve_realtime_conversation_remove(conversation, conversation->first);
  }
}

loom_serve_realtime_item_t* loom_serve_realtime_conversation_find(
    const loom_serve_realtime_conversation_t* conversation,
    iree_string_view_t id) {
  for (loom_serve_realtime_item_t* item = conversation->first; item;
       item = item->next) {
    if (iree_string_view_equal(iree_string_builder_view(&item->id), id)) {
      return item;
    }
  }
  return NULL;
}

iree_status_t loom_serve_realtime_conversation_insert(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* after, iree_string_view_t id,
    loom_serve_realtime_role_t role, bool audio,
    loom_serve_realtime_item_t** out_item) {
  *out_item = NULL;
  if (!id.size || id.size > 128 ||
      loom_serve_realtime_conversation_find(conversation, id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "conversation item requires a unique ID of 1-128 bytes");
  }
  if (conversation->count == 128 ||
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
  item->sequence = ++conversation->sequence;
  item->next = after ? after->next : conversation->first;
  if (after) {
    after->next = item;
  } else {
    conversation->first = item;
  }
  if (!item->next) {
    conversation->last = item;
  }
  ++conversation->count;
  conversation->bytes += id.size;
  *out_item = item;
  return iree_ok_status();
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
  loom_serve_realtime_item_t* previous = NULL;
  loom_serve_realtime_item_t** link = &conversation->first;
  while (*link != item) {
    previous = *link;
    link = &previous->next;
  }
  *link = item->next;
  if (conversation->last == item) {
    conversation->last = previous;
  }
  --conversation->count;
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
    const loom_serve_realtime_item_t* exclude, iree_string_view_t model,
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
  for (const loom_serve_realtime_item_t* item = conversation->first;
       item && item != exclude && iree_status_is_ok(status);
       item = item->next) {
    if (!item->has_transcript) {
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
