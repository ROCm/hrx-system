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

typedef struct loom_serve_realtime_group_t loom_serve_realtime_group_t;

typedef struct loom_serve_realtime_item_t {
  // Canonical native-message position; NULL while the identity is reserved.
  loom_serve_realtime_group_t* group;
  // Owned position in the bounded identity index, independent of history order.
  iree_host_size_t slot;
  // Owned wire identity, unique within this conversation.
  iree_string_builder_t id;
  // Owned committed or currently generated transcript; no waveform is retained.
  iree_string_builder_t text;
  // Semantic speaker role in native chat composition.
  loom_serve_realtime_role_t role;
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

// One native-message position, reserved before a response has a wire item.
// The response retains this position until accepted native/CPU work retires.
struct loom_serve_realtime_group_t {
  // Earlier native-message position, including retained empty responses.
  loom_serve_realtime_group_t* previous;
  // Later native-message position, independent of item publication order.
  loom_serve_realtime_group_t* next;
  // Canonical message, or NULL before the response produces content.
  loom_serve_realtime_item_t* item;
  // Monotonic creation identity for input frontiers and checkpoint
  // invalidation.
  uint64_t sequence;
  // A response still owns this position, even if it has no published content.
  bool retained;
};

typedef struct loom_serve_realtime_conversation_t {
  // Allocator for item records and text/identity storage.
  iree_allocator_t host_allocator;
  // Session namespace for server-generated item identities.
  uint64_t session_id;
  // Last generated item identity; abandoned reservations are never reused.
  uint64_t identity;
  // Ordered canonical native messages, without model or task ownership.
  loom_serve_realtime_group_t* first;
  // Append frontier, including responses that have not produced content yet.
  loom_serve_realtime_group_t* last;
  // Identity ownership includes committed items and microphone reservations.
  loom_serve_realtime_item_t* items[128];
  // Allocated items plus retained empty groups, bounded by the identity index.
  iree_host_size_t count;
  // Retained transcript and identity bytes across every item.
  iree_host_size_t bytes;
  // Last allocated creation identity; deleted identities are never reused.
  uint64_t sequence;
} loom_serve_realtime_conversation_t;

void loom_serve_realtime_conversation_initialize(
    uint64_t session_id, iree_allocator_t host_allocator,
    loom_serve_realtime_conversation_t* conversation);
void loom_serve_realtime_conversation_deinitialize(
    loom_serve_realtime_conversation_t* conversation);
loom_serve_realtime_item_t* loom_serve_realtime_conversation_find(
    const loom_serve_realtime_conversation_t* conversation,
    iree_string_view_t id);

// Reserves an actual item identity before joining canonical history. Empty id
// generates a unique server ID; explicit IDs must contain 1-128 bytes. Client
// validation distinguishes an omitted ID from an invalid explicit empty ID.
// Reservations participate in uniqueness and the 128-item/1-MiB storage bound.
// find exposes only committed items. remove releases either kind of item.
iree_status_t loom_serve_realtime_conversation_reserve(
    loom_serve_realtime_conversation_t* conversation, iree_string_view_t id,
    loom_serve_realtime_role_t role, bool audio,
    loom_serve_realtime_item_t** out_item);

// Adopts a reserved item in a new native-message position after the given
// group, or at the beginning when after is NULL. Default append uses last,
// including a response whose message has not been published yet.
iree_status_t loom_serve_realtime_conversation_insert(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* after, loom_serve_realtime_item_t* item);

// Reserves a response's append frontier and capacity for its first real item.
// The caller releases the group after all native and CPU references retire.
iree_status_t loom_serve_realtime_conversation_begin(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t** out_group);
void loom_serve_realtime_conversation_release(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* group);

// Creates the response's first real message at its already reserved position.
iree_status_t loom_serve_realtime_conversation_add(
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_group_t* group, loom_serve_realtime_role_t role,
    bool audio, loom_serve_realtime_item_t** out_item);

// Finds the wire predecessor of a reserved position, skipping empty groups.
loom_serve_realtime_item_t* loom_serve_realtime_conversation_previous(
    const loom_serve_realtime_group_t* group);
loom_serve_realtime_item_t* loom_serve_realtime_conversation_last_item(
    const loom_serve_realtime_conversation_t* conversation);

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
// groups preceding exclude. Truncated assistant audio has no trustworthy text
// and is omitted. Source rendering/tokenization retains its own ownership.
iree_status_t loom_serve_realtime_conversation_request(
    const loom_serve_realtime_conversation_t* conversation,
    const loom_serve_realtime_group_t* exclude, iree_string_view_t model,
    iree_string_view_t instructions, uint32_t max_tokens, uint64_t seed,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_CONVERSATION_H_
