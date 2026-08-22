// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/byte_sequence.h"

#include <string.h>

IREE_API_EXPORT void iree_io_byte_sequence_retain(
    iree_io_byte_sequence_t* sequence) {
  if (IREE_LIKELY(sequence)) {
    iree_atomic_ref_count_inc(&sequence->ref_count);
  }
}

IREE_API_EXPORT void iree_io_byte_sequence_release(
    iree_io_byte_sequence_t* sequence) {
  if (IREE_LIKELY(sequence) &&
      iree_atomic_ref_count_dec(&sequence->ref_count) == 1) {
    sequence->vtable->destroy(sequence);
  }
}

IREE_API_EXPORT uint64_t
iree_io_byte_sequence_length(const iree_io_byte_sequence_t* sequence) {
  IREE_ASSERT_ARGUMENT(sequence);
  return sequence->length;
}

IREE_API_EXPORT iree_status_t iree_io_byte_sequence_enumerate(
    const iree_io_byte_sequence_t* sequence,
    iree_io_byte_sequence_segment_callback_t callback) {
  IREE_ASSERT_ARGUMENT(sequence);
  IREE_ASSERT_ARGUMENT(callback.fn);
  return sequence->vtable->enumerate(sequence, callback);
}

//===----------------------------------------------------------------------===//
// Allocator-owned span sequence
//===----------------------------------------------------------------------===//

typedef struct iree_io_owned_span_byte_sequence_t {
  // Byte sequence interface exposed to callers.
  iree_io_byte_sequence_t base;
  // Allocator owning both the wrapper and span data.
  iree_allocator_t host_allocator;
  // Contiguous immutable sequence contents.
  iree_byte_span_t span;
} iree_io_owned_span_byte_sequence_t;

static iree_io_owned_span_byte_sequence_t*
iree_io_owned_span_byte_sequence_cast(
    iree_io_byte_sequence_t* IREE_RESTRICT base_sequence) {
  return (iree_io_owned_span_byte_sequence_t*)base_sequence;
}

static const iree_io_owned_span_byte_sequence_t*
iree_io_owned_span_byte_sequence_const_cast(
    const iree_io_byte_sequence_t* IREE_RESTRICT base_sequence) {
  return (const iree_io_owned_span_byte_sequence_t*)base_sequence;
}

static void iree_io_owned_span_byte_sequence_destroy(
    iree_io_byte_sequence_t* IREE_RESTRICT base_sequence) {
  iree_io_owned_span_byte_sequence_t* sequence =
      iree_io_owned_span_byte_sequence_cast(base_sequence);
  iree_allocator_t host_allocator = sequence->host_allocator;
  iree_allocator_free(host_allocator, sequence->span.data);
  iree_allocator_free(host_allocator, sequence);
}

static iree_status_t iree_io_owned_span_byte_sequence_enumerate(
    const iree_io_byte_sequence_t* base_sequence,
    iree_io_byte_sequence_segment_callback_t callback) {
  const iree_io_owned_span_byte_sequence_t* sequence =
      iree_io_owned_span_byte_sequence_const_cast(base_sequence);
  if (sequence->span.data_length == 0) return iree_ok_status();
  return callback.fn(callback.user_data,
                     iree_const_cast_byte_span(sequence->span));
}

static const iree_io_byte_sequence_vtable_t
    iree_io_owned_span_byte_sequence_vtable = {
        .destroy = iree_io_owned_span_byte_sequence_destroy,
        .enumerate = iree_io_owned_span_byte_sequence_enumerate,
};

IREE_API_EXPORT iree_status_t iree_io_byte_sequence_create_from_span_move(
    iree_byte_span_t* inout_span, iree_allocator_t host_allocator,
    iree_io_byte_sequence_t** out_sequence) {
  IREE_ASSERT_ARGUMENT(inout_span);
  IREE_ASSERT_ARGUMENT(out_sequence);
  *out_sequence = NULL;
  if (inout_span->data_length > 0 && !inout_span->data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty byte span has NULL data");
  }

  iree_io_owned_span_byte_sequence_t* sequence = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, sizeof(*sequence), (void**)&sequence));
  iree_io_byte_sequence_initialize(&iree_io_owned_span_byte_sequence_vtable,
                                   inout_span->data_length, &sequence->base);
  sequence->host_allocator = host_allocator;
  sequence->span = *inout_span;

  *inout_span = iree_byte_span_empty();
  *out_sequence = &sequence->base;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Contiguous clone
//===----------------------------------------------------------------------===//

typedef struct iree_io_byte_sequence_clone_state_t {
  // Allocation being populated with sequence contents.
  iree_byte_span_t target_span;
  // Number of bytes populated in |target_span|.
  iree_host_size_t offset;
} iree_io_byte_sequence_clone_state_t;

static iree_status_t iree_io_byte_sequence_clone_segment(
    void* user_data, iree_const_byte_span_t segment) {
  iree_io_byte_sequence_clone_state_t* state =
      (iree_io_byte_sequence_clone_state_t*)user_data;
  IREE_ASSERT_GT(segment.data_length, 0);
  IREE_ASSERT(segment.data != NULL);
  IREE_ASSERT_LE(segment.data_length,
                 state->target_span.data_length - state->offset);
  memcpy(state->target_span.data + state->offset, segment.data,
         segment.data_length);
  state->offset += segment.data_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_byte_sequence_clone(
    const iree_io_byte_sequence_t* sequence, iree_allocator_t host_allocator,
    iree_byte_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(sequence);
  IREE_ASSERT_ARGUMENT(out_span);
  *out_span = iree_byte_span_empty();

  const uint64_t sequence_length = iree_io_byte_sequence_length(sequence);
  if (sequence_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "byte sequence length exceeds the host addressable range");
  }
  if (sequence_length == 0) return iree_ok_status();

  iree_io_byte_sequence_clone_state_t state = {
      .target_span = iree_byte_span_empty(),
      .offset = 0,
  };
  state.target_span.data_length = (iree_host_size_t)sequence_length;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, state.target_span.data_length,
      (void**)&state.target_span.data));

  iree_io_byte_sequence_segment_callback_t callback = {
      .fn = iree_io_byte_sequence_clone_segment,
      .user_data = &state,
  };
  iree_status_t status = iree_io_byte_sequence_enumerate(sequence, callback);
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(state.offset, state.target_span.data_length);
    *out_span = state.target_span;
  } else {
    iree_allocator_free(host_allocator, state.target_span.data);
  }
  return status;
}
