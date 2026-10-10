// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/checkpoint_registry.h"

#include <string.h>

iree_status_t loom_serve_text_checkpoint_registry_initialize(
    iree_host_size_t capacity, loom_serve_text_checkpoint_registry_t* registry,
    iree_allocator_t host_allocator) {
  *registry = (loom_serve_text_checkpoint_registry_t){
      .allocator = host_allocator, .capacity = capacity};
  if (!capacity) {
    return iree_ok_status();
  }
  return iree_allocator_malloc_array(host_allocator, capacity,
                                     sizeof(*registry->entries),
                                     (void**)&registry->entries);
}

void loom_serve_text_checkpoint_registry_reset(
    loom_serve_text_checkpoint_registry_t* registry,
    iree_status_code_t terminal_code) {
  for (iree_host_size_t i = 0; registry->entries && i < registry->capacity;
       ++i) {
    loom_serve_text_checkpoint_entry_t* entry = &registry->entries[i];
    if (terminal_code == IREE_STATUS_OK) {
      loom_serve_text_checkpoint_release(entry->state);
    }
    loom_serve_text_chat_completion_deinitialize(&entry->completion);
    memset(entry, 0, sizeof(*entry));
  }
  registry->host_bytes = 0;
}

void loom_serve_text_checkpoint_registry_deinitialize(
    loom_serve_text_checkpoint_registry_t* registry) {
  iree_allocator_free(registry->allocator, registry->entries);
  memset(registry, 0, sizeof(*registry));
}

loom_serve_text_checkpoint_entry_t* loom_serve_text_checkpoint_registry_find(
    loom_serve_text_checkpoint_registry_t* registry, iree_string_view_t name) {
  for (iree_host_size_t i = 0; name.size && i < registry->capacity; ++i) {
    loom_serve_text_checkpoint_entry_t* entry = &registry->entries[i];
    if (iree_string_view_equal(name, iree_make_cstring_view(entry->name))) {
      return entry;
    }
  }
  return NULL;
}

loom_serve_text_checkpoint_entry_t*
loom_serve_text_checkpoint_registry_available(
    loom_serve_text_checkpoint_registry_t* registry) {
  for (iree_host_size_t i = 0; i < registry->capacity; ++i) {
    if (!registry->entries[i].state) {
      return &registry->entries[i];
    }
  }
  return NULL;
}

iree_status_t loom_serve_text_checkpoint_entry_submit_pin(
    loom_serve_text_checkpoint_entry_t* entry, loom_serve_text_row_t* row,
    const loom_serve_text_chat_completion_t* completion,
    loom_serve_execution_frontier_t* out_frontier) {
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_submit_pin(row, &entry->state, out_frontier));
  if (entry->state) {
    entry->completion = *completion;
    iree_vm_buffer_retain(entry->completion.storage);
    entry->position = loom_serve_text_row_position(row);
    entry->token = loom_serve_text_row_token(row);
    entry->boundary = loom_serve_text_row_is_eos(row)
                          ? LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED
                          : LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN;
  }
  return iree_ok_status();
}

void loom_serve_text_checkpoint_entry_publish(
    loom_serve_text_checkpoint_entry_t* entry, iree_string_view_t name) {
  memcpy(entry->name, name.data, name.size);
  entry->name[name.size] = 0;
}

void loom_serve_text_checkpoint_registry_refresh(
    loom_serve_text_checkpoint_registry_t* registry,
    loom_serve_text_checkpoint_entry_t* entry) {
  registry->host_bytes -= entry->host_bytes;
  entry->host_bytes = loom_serve_text_checkpoint_suspended_bytes(entry->state);
  registry->host_bytes += entry->host_bytes;
}

void loom_serve_text_checkpoint_registry_release(
    loom_serve_text_checkpoint_registry_t* registry,
    loom_serve_text_checkpoint_entry_t* entry) {
  registry->host_bytes -= entry->host_bytes;
  loom_serve_text_checkpoint_release(entry->state);
  loom_serve_text_chat_completion_deinitialize(&entry->completion);
  memset(entry, 0, sizeof(*entry));
}
