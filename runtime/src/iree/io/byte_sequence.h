// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_BYTE_SEQUENCE_H_
#define IREE_IO_BYTE_SEQUENCE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_io_byte_sequence_t
//===----------------------------------------------------------------------===//

// An immutable logical sequence of bytes stored in one or more segments.
//
// A sequence has a constant byte length and may be enumerated concurrently
// from multiple threads. Enumeration visits all non-empty segments in logical
// order without allocating, copying, mapping, or locking. Segment boundaries
// are an implementation detail and callers must not depend on their count or
// sizes.
//
// NULL represents an absent sequence. An empty sequence is a non-NULL object
// with zero length whose enumeration invokes no callbacks.
typedef struct iree_io_byte_sequence_t iree_io_byte_sequence_t;

// Callback invoked for each ordered non-empty segment in a byte sequence.
//
// |segment| is valid only for the duration of the callback. The callback must
// not retain its data pointer. Returning a non-OK status immediately stops
// enumeration and transfers the status to the caller unchanged.
typedef iree_status_t(
    IREE_API_PTR* iree_io_byte_sequence_segment_callback_fn_t)(
    void* user_data, iree_const_byte_span_t segment);

typedef struct iree_io_byte_sequence_segment_callback_t {
  // Function invoked for each segment.
  iree_io_byte_sequence_segment_callback_fn_t fn;
  // Unowned state passed to the callback function.
  void* user_data;
} iree_io_byte_sequence_segment_callback_t;

// Retains |sequence| for the caller. No-op if |sequence| is NULL.
IREE_API_EXPORT void iree_io_byte_sequence_retain(
    iree_io_byte_sequence_t* sequence);

// Releases |sequence| from the caller. No-op if |sequence| is NULL.
IREE_API_EXPORT void iree_io_byte_sequence_release(
    iree_io_byte_sequence_t* sequence);

// Returns the logical byte length of |sequence|.
//
// The uint64_t length permits sequences whose segmented representation is
// larger than the host address space. Callers requiring contiguous storage
// must reject lengths greater than IREE_HOST_SIZE_MAX.
IREE_API_EXPORT uint64_t
iree_io_byte_sequence_length(const iree_io_byte_sequence_t* sequence);

// Enumerates every non-empty segment in |sequence| in logical byte order.
//
// Implementations perform no allocation, copying, mapping, or locking and
// enumerate non-NULL, non-empty segments whose lengths sum to the sequence
// length. A callback returning OK always continues enumeration. A callback
// returning a non-OK status stops enumeration and the status is returned
// unchanged.
IREE_API_EXPORT iree_status_t iree_io_byte_sequence_enumerate(
    const iree_io_byte_sequence_t* sequence,
    iree_io_byte_sequence_segment_callback_t callback);

// Creates a sequence by transferring ownership of |inout_span|.
//
// Any non-NULL span data must be an allocation owned by |host_allocator|. On
// success the span is reset to empty and the returned sequence owns both the
// data and its wrapper. On failure |inout_span| is unchanged and
// |out_sequence| is NULL. An empty span produces a real empty sequence object.
IREE_API_EXPORT iree_status_t iree_io_byte_sequence_create_from_span_move(
    iree_byte_span_t* inout_span, iree_allocator_t host_allocator,
    iree_io_byte_sequence_t** out_sequence);

// Clones |sequence| into one contiguous caller-owned allocation.
//
// The returned span is independent of |sequence| and must be freed with
// |host_allocator|. Empty sequences return an empty span without allocating.
// Lengths greater than IREE_HOST_SIZE_MAX fail with IREE_STATUS_OUT_OF_RANGE.
// On any failure |out_span| is empty.
IREE_API_EXPORT iree_status_t iree_io_byte_sequence_clone(
    const iree_io_byte_sequence_t* sequence, iree_allocator_t host_allocator,
    iree_byte_span_t* out_span);

//===----------------------------------------------------------------------===//
// iree_io_byte_sequence_t implementation details
//===----------------------------------------------------------------------===//

// Dispatch table implemented by immutable byte sequence storage types.
typedef struct iree_io_byte_sequence_vtable_t {
  // Destroys |sequence| after its final reference is released.
  void(IREE_API_PTR* destroy)(iree_io_byte_sequence_t* IREE_RESTRICT sequence);
  // Enumerates all logical segments in order and propagates the first callback
  // failure unchanged. Implementations must uphold the public segment and
  // logical-length invariants.
  iree_status_t(IREE_API_PTR* enumerate)(
      const iree_io_byte_sequence_t* sequence,
      iree_io_byte_sequence_segment_callback_t callback);
} iree_io_byte_sequence_vtable_t;

// Base structure embedded at offset zero by byte sequence implementations.
struct iree_io_byte_sequence_t {
  // Atomic reference count controlling the sequence lifetime.
  iree_atomic_ref_count_t ref_count;
  // Dispatch table for the concrete storage implementation.
  const iree_io_byte_sequence_vtable_t* vtable;
  // Constant logical length of the sequence in bytes.
  uint64_t length;
};

// Initializes an implementation-provided |out_sequence| base structure.
static inline void iree_io_byte_sequence_initialize(
    const iree_io_byte_sequence_vtable_t* vtable, uint64_t length,
    iree_io_byte_sequence_t* out_sequence) {
  iree_atomic_ref_count_init(&out_sequence->ref_count);
  out_sequence->vtable = vtable;
  out_sequence->length = length;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_BYTE_SEQUENCE_H_
