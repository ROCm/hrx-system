// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/byte_sequence.h"

#include <string.h>

#include "iree/base/byte_sequence.h"
#include "loomc/iree.h"

loomc_status_t loomc_byte_sequence_create_copy(
    loomc_byte_span_t contents, loomc_allocator_t allocator,
    loomc_byte_sequence_t** out_sequence) {
  if (out_sequence == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_sequence must not be NULL");
  }
  *out_sequence = NULL;
  if (contents.data == NULL && contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "contents have length but no data");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator must be valid");
  }

  iree_byte_span_t storage = iree_byte_span_empty();
  iree_allocator_t host_allocator = iree_allocator_from_loomc(allocator);
  iree_status_t status = iree_ok_status();
  if (contents.data_length != 0) {
    storage.data_length = contents.data_length;
    status = iree_allocator_malloc_uninitialized(
        host_allocator, storage.data_length, (void**)&storage.data);
    if (iree_status_is_ok(status)) {
      memcpy(storage.data, contents.data, contents.data_length);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_byte_sequence_t* sequence = NULL;
    status = iree_byte_sequence_create_from_span_move(&storage, host_allocator,
                                                      &sequence);
    *out_sequence = (loomc_byte_sequence_t*)sequence;
  }
  iree_allocator_free(host_allocator, storage.data);
  return loomc_status_from_iree(status);
}

void loomc_byte_sequence_retain(loomc_byte_sequence_t* sequence) {
  iree_byte_sequence_retain((iree_byte_sequence_t*)sequence);
}

void loomc_byte_sequence_release(loomc_byte_sequence_t* sequence) {
  iree_byte_sequence_release((iree_byte_sequence_t*)sequence);
}

uint64_t loomc_byte_sequence_length(const loomc_byte_sequence_t* sequence) {
  if (sequence == NULL) return 0;
  return iree_byte_sequence_length((const iree_byte_sequence_t*)sequence);
}

bool loomc_byte_sequence_try_get_contiguous_span(
    const loomc_byte_sequence_t* sequence, loomc_byte_span_t* out_span) {
  if (out_span == NULL) return false;
  *out_span = loomc_byte_span_empty();
  if (sequence == NULL) return false;
  iree_const_byte_span_t span = iree_const_byte_span_empty();
  const bool is_contiguous = iree_byte_sequence_try_get_contiguous_span(
      (const iree_byte_sequence_t*)sequence, &span);
  *out_span = loomc_byte_span_from_iree(span);
  return is_contiguous;
}

typedef struct loomc_byte_sequence_enumeration_t {
  // Public callback receiving converted segment views.
  loomc_byte_sequence_callback_t callback;
} loomc_byte_sequence_enumeration_t;

static iree_status_t loomc_byte_sequence_enumerate_segment(
    void* user_data, iree_const_byte_span_t segment) {
  loomc_byte_sequence_enumeration_t* enumeration =
      (loomc_byte_sequence_enumeration_t*)user_data;
  return iree_status_from_loomc(enumeration->callback.fn(
      enumeration->callback.user_data, loomc_byte_span_from_iree(segment)));
}

loomc_status_t loomc_byte_sequence_enumerate(
    const loomc_byte_sequence_t* sequence,
    loomc_byte_sequence_callback_t callback) {
  if (sequence == NULL || callback.fn == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "sequence and callback must be valid");
  }
  loomc_byte_sequence_enumeration_t enumeration = {
      .callback = callback,
  };
  return loomc_status_from_iree(iree_byte_sequence_enumerate(
      (const iree_byte_sequence_t*)sequence,
      (iree_byte_sequence_segment_callback_t){
          .fn = loomc_byte_sequence_enumerate_segment,
          .user_data = &enumeration,
      }));
}

loomc_status_t loomc_byte_sequence_clone(const loomc_byte_sequence_t* sequence,
                                         loomc_allocator_t allocator,
                                         loomc_byte_span_t* out_span) {
  if (sequence == NULL || out_span == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "sequence and out_span must not be NULL");
  }
  *out_span = loomc_byte_span_empty();
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator must be valid");
  }
  iree_byte_span_t span = iree_byte_span_empty();
  loomc_status_t status = loomc_status_from_iree(
      iree_byte_sequence_clone((const iree_byte_sequence_t*)sequence,
                               iree_allocator_from_loomc(allocator), &span));
  if (loomc_status_is_ok(status)) {
    *out_span = loomc_byte_span_from_iree(iree_const_cast_byte_span(span));
  }
  return status;
}
