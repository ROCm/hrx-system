// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_VEC_STREAM_H_
#define IREE_IO_VEC_STREAM_H_

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/io/stream.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_io_vec_stream_t
//===----------------------------------------------------------------------===//

// Creates an in-memory stream that grows as data is written.
// Blocks of |block_size| (16-byte aligned) are allocated each time growth is
// required and writes will be split to fit into blocks. To retrieve the data
// from the stream use iree_io_vec_stream_enumerate_blocks or seek and read it
// back.
IREE_API_EXPORT iree_status_t iree_io_vec_stream_create(
    iree_io_stream_mode_t mode, iree_host_size_t block_size,
    iree_allocator_t host_allocator, iree_io_stream_t** out_stream);

// Called for each block in stream order. Blocks may be sized under the
// requested block size if they contain partial data.
typedef iree_status_t(IREE_API_PTR* iree_io_vec_stream_callback_fn_t)(
    void* user_data, iree_const_byte_span_t block);

// Issues |callback| for each block of data in the stream.
IREE_API_EXPORT iree_status_t iree_io_vec_stream_enumerate_blocks(
    iree_io_stream_t* stream, iree_io_vec_stream_callback_fn_t callback,
    void* user_data);

// Moves all current stream contents into an immutable byte sequence.
//
// On success ownership of every allocated block transfers to |out_sequence|
// without copying their contents. The stream retains its mode, allocator, and
// block size but is reset to offset and length zero with no allocated blocks.
// Retained aliases of the stream observe the same empty state and the stream
// may immediately be reused as a builder. Moving an empty stream returns a
// non-NULL empty sequence.
//
// The operation allocates the sequence owner before modifying the stream. If
// allocation fails then |out_sequence| is NULL and the stream remains
// unchanged. The caller must have exclusive access to the stream while the
// contents are moved.
IREE_API_EXPORT iree_status_t iree_io_vec_stream_move_contents(
    iree_io_stream_t* stream, iree_byte_sequence_t** out_sequence);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_VEC_STREAM_H_
