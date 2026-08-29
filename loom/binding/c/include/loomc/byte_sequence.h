// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_BYTE_SEQUENCE_H_
#define LOOMC_BYTE_SEQUENCE_H_

#include "loomc/base.h"

/// @file
/// Immutable ref-counted byte sequences.
///
/// Byte sequences preserve segmented producer storage across compiler,
/// caching, packaging, and loader boundaries. Segment boundaries are an
/// implementation detail; consumers may enumerate them without allocation or
/// request one explicit contiguous clone when their final API requires it.

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable logical sequence of bytes stored in one or more segments.
///
/// A sequence has constant contents and length. Retained handles may be shared
/// across threads and enumerated concurrently.
typedef struct loomc_byte_sequence_t loomc_byte_sequence_t;

/// Callback invoked for each ordered non-empty sequence segment.
///
/// `segment` is borrowed only for the callback duration. Returning a non-OK
/// status stops enumeration and transfers that status to the caller.
typedef loomc_status_t(LOOMC_API_PTR* loomc_byte_sequence_callback_fn_t)(
    void* user_data, loomc_byte_span_t segment);

/// Callback and caller-owned state used to enumerate a sequence.
typedef struct loomc_byte_sequence_callback_t {
  /// Function invoked for each segment.
  loomc_byte_sequence_callback_fn_t fn;

  /// Opaque value passed to `fn`.
  void* user_data;
} loomc_byte_sequence_callback_t;

/// Creates a sequence containing a copy of `contents`.
///
/// @param contents Bytes copied into immutable sequence storage.
/// @param allocator Host allocator owning the returned sequence and storage.
/// @param out_sequence Receives one retained sequence on success.
/// @return OK when the sequence was created.
///
/// Empty contents produce a real non-NULL empty sequence. The caller releases
/// the returned reference with `loomc_byte_sequence_release`.
LOOMC_API_EXPORT loomc_status_t loomc_byte_sequence_create_copy(
    loomc_byte_span_t contents, loomc_allocator_t allocator,
    loomc_byte_sequence_t** out_sequence);

/// Retains `sequence` for another owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_byte_sequence_retain(
    loomc_byte_sequence_t* sequence);

/// Releases `sequence` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_byte_sequence_release(
    loomc_byte_sequence_t* sequence);

/// Returns the logical byte length of `sequence`.
LOOMC_API_EXPORT uint64_t
loomc_byte_sequence_length(const loomc_byte_sequence_t* sequence);

/// Attempts to borrow the complete sequence as one contiguous span.
///
/// Returns true and populates `out_span` when the producer storage is already
/// contiguous. The span remains valid while `sequence` is retained. Returns
/// false and leaves `out_span` empty for segmented storage.
LOOMC_API_EXPORT bool loomc_byte_sequence_try_get_contiguous_span(
    const loomc_byte_sequence_t* sequence, loomc_byte_span_t* out_span);

/// Enumerates all non-empty segments in logical byte order.
LOOMC_API_EXPORT loomc_status_t
loomc_byte_sequence_enumerate(const loomc_byte_sequence_t* sequence,
                              loomc_byte_sequence_callback_t callback);

/// Clones `sequence` into one contiguous allocator-owned span.
///
/// The caller frees `out_span->data` with `allocator`. Empty sequences return
/// an empty span without allocating. On failure `out_span` is empty.
LOOMC_API_EXPORT loomc_status_t loomc_byte_sequence_clone(
    const loomc_byte_sequence_t* sequence, loomc_allocator_t allocator,
    loomc_byte_span_t* out_span);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_BYTE_SEQUENCE_H_
