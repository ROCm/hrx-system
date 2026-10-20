// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_STREAM_H_
#define LOOM_SERVE_SERVICE_SPEECH_STREAM_H_

#include "loom/serve/service/speech/consumer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_speech_stream_t loom_serve_speech_stream_t;

// Allocates the two owners of a bounded native result: a caller handle and a
// service consumer receipt. Neither retains a model or service pointer. The
// service ends its receipt after accepted work retires; the caller closes its
// handle independently. No asynchronous work is accepted by construction.
// Geometry and capacity are the validated generator contract supplied by the
// service; caller-facing admission is loom_serve_speech_service_try_open.
// Success transfers finite request text and returns a separate service-owned
// input reference. Failure leaves the request unchanged and all outputs empty.
iree_status_t loom_serve_speech_stream_allocate(
    loom_serve_speech_format_t format, iree_host_size_t block_capacity,
    iree_host_size_t text_capacity, uint64_t text_limit,
    loom_serve_speech_request_t* request, iree_allocator_t host_allocator,
    loom_serve_speech_stream_t** out_stream,
    loom_serve_speech_input_t** out_input,
    loom_serve_speech_consumer_t* out_consumer);

// Appends at most current byte credit to a continuing request. Arbitrary UTF-8
// byte partitions are accepted; only complete scalars reach source. Invalid
// accepted prefixes, model text-limit overflow, or closed input fail without
// accepting bytes. A zero count is ordinary backpressure. All operations are
// serialized on the serving owner and never wait for CPU or device work.
iree_status_t loom_serve_speech_stream_write(loom_serve_speech_stream_t* stream,
                                             iree_const_byte_span_t text,
                                             iree_host_size_t* out_byte_count);
iree_host_size_t loom_serve_speech_stream_write_credit(
    const loom_serve_speech_stream_t* stream);

// Publishes end-of-input. Incomplete UTF-8 fails without publishing EOF;
// otherwise repeated finish is harmless while the service remains attached.
iree_status_t loom_serve_speech_stream_finish(
    loom_serve_speech_stream_t* stream);

// Immutable output geometry, retained independently of registration lifetime.
loom_serve_speech_format_t loom_serve_speech_stream_format(
    const loom_serve_speech_stream_t* stream);

// Borrows one copied F32LE waveform block until consume or close. NULL means
// no block is published. An unread block excludes only this request from new
// source work; it never retains source feedback or another request's output.
// A final block can remain readable after its model registration is removed.
const loom_serve_speech_block_t* loom_serve_speech_stream_peek(
    const loom_serve_speech_stream_t* stream);
void loom_serve_speech_stream_consume(loom_serve_speech_stream_t* stream);

// Completion follows service detachment and consumption of final output.
bool loom_serve_speech_stream_is_complete(
    const loom_serve_speech_stream_t* stream);

// Returns an owned copy of terminal failure, or OK while healthy/pending.
// Cancellation/removal publishes status without invalidating the caller handle.
iree_status_t loom_serve_speech_stream_status(
    const loom_serve_speech_stream_t* stream);

// Consumes the caller handle immediately and suppresses further publication.
// Accepted CPU/cohort borrowers keep their storage until actual retirement.
// All caller operations are serialized on the serving owner; none waits.
void loom_serve_speech_stream_close(loom_serve_speech_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_STREAM_H_
