// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_CONVERSATION_H_
#define LOOM_SERVE_SERVER_REALTIME_CONVERSATION_H_

#include "loom/serve/server/realtime/event.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_realtime_role_e {
  LOOM_SERVE_REALTIME_USER,
  LOOM_SERVE_REALTIME_ASSISTANT,
  LOOM_SERVE_REALTIME_SYSTEM,
} loom_serve_realtime_role_t;

typedef enum loom_serve_realtime_item_status_e {
  LOOM_SERVE_REALTIME_ITEM_IN_PROGRESS,
  LOOM_SERVE_REALTIME_ITEM_COMPLETED,
  LOOM_SERVE_REALTIME_ITEM_INCOMPLETE,
} loom_serve_realtime_item_status_t;

typedef struct loom_serve_realtime_item_t {
  // Ordered conversation link, independent of model token/cache positions.
  struct loom_serve_realtime_item_t* next;
  // Owned wire identity, unique within this conversation.
  iree_string_builder_t id;
  // Owned committed or currently generated transcript; no waveform is retained.
  iree_string_builder_t text;
  // Semantic speaker role in native chat composition.
  loom_serve_realtime_role_t role;
  // Monotonic creation identity for checkpoint dependency invalidation.
  uint64_t sequence;
  // This item represents input or output audio rather than a text-only message.
  bool audio;
  // Transcript is present in model-visible history; truncation clears it.
  bool has_transcript;
  // Semantic completion, distinguishing a full result from interrupted output.
  loom_serve_realtime_item_status_t status;
  // Generated output duration in 24 kHz wire frames, never evidence of
  // playback.
  uint64_t generated_frames;
} loom_serve_realtime_item_t;

typedef struct loom_serve_realtime_conversation_t {
  // Allocator for item records and text/identity storage.
  iree_allocator_t host_allocator;
  // Ordered canonical conversation, without model or task ownership.
  loom_serve_realtime_item_t* first;
  // Tail used for append and default previous_item_id reporting.
  loom_serve_realtime_item_t* last;
  // Number of retained items, bounded independently of text bytes.
  iree_host_size_t count;
  // Retained transcript and identity bytes across every item.
  iree_host_size_t bytes;
  // Last allocated creation identity; deleted identities are never reused.
  uint64_t sequence;
} loom_serve_realtime_conversation_t;

void loom_serve_realtime_conversation_initialize(
    iree_allocator_t host_allocator,
    loom_serve_realtime_conversation_t* conversation);
void loom_serve_realtime_conversation_deinitialize(
    loom_serve_realtime_conversation_t* conversation);
loom_serve_realtime_item_t* loom_serve_realtime_conversation_find(
    const loom_serve_realtime_conversation_t* conversation,
    iree_string_view_t id);

// Allocates an initially empty item with a copied unique ID. after is a
// retained item or NULL for the beginning. The conversation retains at most 128
// items and 1 MiB of text/identity bytes. No native model operation is
// performed.
iree_status_t loom_serve_realtime_conversation_insert(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* after, iree_string_view_t id,
    loom_serve_realtime_role_t role, bool audio,
    loom_serve_realtime_item_t** out_item);
iree_status_t loom_serve_realtime_conversation_append_text(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item, iree_string_view_t text);
// Caller first cancels any native/input references to the removed item.
void loom_serve_realtime_conversation_remove(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item);

// Applies a validated audio truncation by removing the entire transcript. The
// session validates identity and duration before cancelling dependent work.
// Client-owned playback has no trustworthy word-to-waveform alignment.
void loom_serve_realtime_conversation_truncate(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_item_t* item);

// Appends this item's standard Realtime representation. Audio is represented
// by its transcript; full recordings/generated waveforms are never accumulated.
iree_status_t loom_serve_realtime_item_write(
    const loom_serve_realtime_item_t* item, iree_string_builder_t* output);

// Composes one source-defined native text request from completed conversation
// items preceding exclude. Truncated assistant audio has no trustworthy text
// and is omitted. Source rendering/tokenization retains its own ownership.
iree_status_t loom_serve_realtime_conversation_request(
    const loom_serve_realtime_conversation_t* conversation,
    const loom_serve_realtime_item_t* exclude, iree_string_view_t model,
    iree_string_view_t instructions, uint32_t max_tokens, uint64_t seed,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_CONVERSATION_H_
