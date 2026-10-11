// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_CHECKPOINT_REGISTRY_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_CHECKPOINT_REGISTRY_H_

#include "experimental/loom_serve/text/chat.h"
#include "experimental/loom_serve/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// One named immutable endpoint. The serialized model owner mutates the record;
// a preparation job may borrow its canonical history until CPU release. Native
// branches retain independent state references after their admission.
typedef struct loom_serve_text_checkpoint_entry_t {
  // Published identifier; empty until successful pin readiness.
  char name[65];
  // Owned native endpoint, including accepted but unpublished pin work.
  loom_serve_text_checkpoint_t* state;
  // Owned canonical transcript, independent of the originating session.
  loom_serve_text_chat_completion_t completion;
  // Consumed frontier captured with the pin.
  iree_host_size_t position;
  // Selected token not yet consumed at that frontier.
  int32_t token;
  // Source framing classification of the pending token.
  loom_serve_text_chat_boundary_t boundary;
  // Queued admission borrowers; deletion requires this count to be zero.
  iree_host_size_t pending;
} loom_serve_text_checkpoint_entry_t;

// Fixed host records over the model's native pin capacity. This is not a second
// device block allocator. Record pointers remain stable until deinitialization.
typedef struct loom_serve_text_checkpoint_registry_t {
  // Allocator owning entries.
  iree_allocator_t allocator;
  // Owned records, allocated once during service construction.
  loom_serve_text_checkpoint_entry_t* entries;
  // Number of allocated records, independent of resident session rows.
  iree_host_size_t capacity;
} loom_serve_text_checkpoint_registry_t;

// Allocates host records only. A zero capacity disables named endpoints.
// The initialized result may be deinitialized after success or failure.
iree_status_t loom_serve_text_checkpoint_registry_initialize(
    iree_host_size_t capacity, loom_serve_text_checkpoint_registry_t* registry,
    iree_allocator_t host_allocator);

// Ends all named ownership after queued/CPU borrowers have retired. Successful
// readiness permits native release; failed readiness leaves native handles with
// the model for actual retirement. Host transcripts are released in both cases.
// Repeated calls on an already reset registry are harmless.
void loom_serve_text_checkpoint_registry_reset(
    loom_serve_text_checkpoint_registry_t* registry,
    iree_status_code_t terminal_code);

// Releases host record storage after reset. Accepts partial initialization.
void loom_serve_text_checkpoint_registry_deinitialize(
    loom_serve_text_checkpoint_registry_t* registry);

// Returns a published entry, or NULL. The caller validates external identifiers
// and retains pending admission borrowers before returning its owner turn.
loom_serve_text_checkpoint_entry_t* loom_serve_text_checkpoint_registry_find(
    loom_serve_text_checkpoint_registry_t* registry, iree_string_view_t name);

// Returns an unused record, or NULL for ordinary capacity pressure. An accepted
// native pin occupies its record even before its name can be published.
loom_serve_text_checkpoint_entry_t*
loom_serve_text_checkpoint_registry_available(
    loom_serve_text_checkpoint_registry_t* registry);

// Submits into an available record and retains the row's canonical transcript.
// Native capacity denial leaves state NULL without acquiring host ownership.
// The owner excludes model mutation until successful ALL readiness; failure
// retains accepted native ownership for model retirement.
iree_status_t loom_serve_text_checkpoint_entry_submit_pin(
    loom_serve_text_checkpoint_entry_t* entry, loom_serve_text_row_t* row,
    const loom_serve_text_chat_completion_t* completion,
    loom_serve_execution_frontier_t* out_frontier);

// Publishes a successfully ready pin under a validated, unique identifier.
void loom_serve_text_checkpoint_entry_publish(
    loom_serve_text_checkpoint_entry_t* entry, iree_string_view_t name);

// Retains a ready endpoint and its canonical history in an available record.
// Names and queued borrowers are independent; no device work is submitted.
void loom_serve_text_checkpoint_entry_retain(
    const loom_serve_text_checkpoint_entry_t* source,
    loom_serve_text_checkpoint_entry_t* target);

// Drops an unborrowed entry at a completed model boundary. Live branches keep
// their independent references. The record becomes available immediately.
void loom_serve_text_checkpoint_entry_release(
    loom_serve_text_checkpoint_entry_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_TEXT_CHECKPOINT_REGISTRY_H_
