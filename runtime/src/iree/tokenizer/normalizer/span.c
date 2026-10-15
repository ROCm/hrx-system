// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/span.h"

typedef struct iree_tokenizer_normalizer_span_t {
  // Base normalizer interface.
  iree_tokenizer_normalizer_t base;
  // Allocator owning this descriptor.
  iree_allocator_t allocator;
  // Owned component whose state is scoped to each input span.
  iree_tokenizer_normalizer_t* child;
} iree_tokenizer_normalizer_span_t;

typedef struct iree_tokenizer_normalizer_span_state_t {
  // Base streaming state interface.
  iree_tokenizer_normalizer_state_t base;
  // Initialized child in trailing storage, or NULL after initialization
  // failure.
  iree_tokenizer_normalizer_state_t* child;
  // The previous span has drained and the next nonempty input starts a new one.
  bool completed;
} iree_tokenizer_normalizer_span_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_span_vtable;

static iree_host_size_t iree_tokenizer_normalizer_span_header_size(void) {
  return iree_host_align(sizeof(iree_tokenizer_normalizer_span_state_t),
                         iree_max_align_t);
}

static void* iree_tokenizer_normalizer_span_child_storage(
    iree_tokenizer_normalizer_span_state_t* state) {
  return (uint8_t*)state + iree_tokenizer_normalizer_span_header_size();
}

iree_status_t iree_tokenizer_normalizer_span_allocate(
    iree_tokenizer_normalizer_t* child, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_ASSERT_ARGUMENT(child);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_span_t* normalizer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*normalizer),
                                (void**)&normalizer));
  iree_tokenizer_normalizer_initialize(
      &normalizer->base, &iree_tokenizer_normalizer_span_vtable,
      iree_tokenizer_normalizer_span_header_size() +
          iree_tokenizer_normalizer_state_size(child));
  normalizer->allocator = allocator;
  normalizer->child = child;
  *out_normalizer = &normalizer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_span_destroy(
    iree_tokenizer_normalizer_t* base_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_span_t* normalizer =
      (iree_tokenizer_normalizer_span_t*)base_normalizer;
  iree_allocator_t allocator = normalizer->allocator;
  iree_tokenizer_normalizer_free(normalizer->child);
  iree_allocator_free(allocator, normalizer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_span_state_initialize(
    const iree_tokenizer_normalizer_t* base_normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  *out_state = NULL;
  const iree_tokenizer_normalizer_span_t* normalizer =
      (const iree_tokenizer_normalizer_span_t*)base_normalizer;
  iree_tokenizer_normalizer_span_state_t* state =
      (iree_tokenizer_normalizer_span_state_t*)storage;
  state->base.normalizer = base_normalizer;
  state->child = NULL;
  state->completed = false;
  IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_initialize(
      normalizer->child, iree_tokenizer_normalizer_span_child_storage(state),
      &state->child));
  *out_state = &state->base;
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_span_state_deinitialize(
    iree_tokenizer_normalizer_state_t* base_state) {
  iree_tokenizer_normalizer_span_state_t* state =
      (iree_tokenizer_normalizer_span_state_t*)base_state;
  iree_tokenizer_normalizer_state_deinitialize(state->child);
}

static iree_status_t iree_tokenizer_normalizer_span_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_span_state_t* state =
      (iree_tokenizer_normalizer_span_state_t*)base_state;
  *out_consumed = 0;
  *out_written = 0;
  if (state->completed) {
    if (input.size == 0) {
      return iree_ok_status();
    }
    const iree_tokenizer_normalizer_span_t* normalizer =
        (const iree_tokenizer_normalizer_span_t*)base_state->normalizer;
    iree_tokenizer_normalizer_state_deinitialize(state->child);
    state->child = NULL;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_initialize(
        normalizer->child, iree_tokenizer_normalizer_span_child_storage(state),
        &state->child));
    state->completed = false;
  }
  return iree_tokenizer_normalizer_state_process(
      state->child, input, output, flags, out_consumed, out_written);
}

static iree_status_t iree_tokenizer_normalizer_span_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_span_state_t* state =
      (iree_tokenizer_normalizer_span_state_t*)base_state;
  *out_written = 0;
  if (state->completed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_finalize(
      state->child, output, out_written));
  state->completed = !iree_tokenizer_normalizer_state_has_pending(state->child);
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_span_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_span_state_t* state =
      (const iree_tokenizer_normalizer_span_state_t*)base_state;
  return !state->completed &&
         iree_tokenizer_normalizer_state_has_pending(state->child);
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_span_vtable = {
        .destroy = iree_tokenizer_normalizer_span_destroy,
        .state_initialize = iree_tokenizer_normalizer_span_state_initialize,
        .state_deinitialize = iree_tokenizer_normalizer_span_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_span_state_process,
        .state_finalize = iree_tokenizer_normalizer_span_state_finalize,
        .state_has_pending = iree_tokenizer_normalizer_span_state_has_pending,
};
