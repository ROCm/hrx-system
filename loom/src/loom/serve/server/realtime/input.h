// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_INPUT_H_
#define LOOM_SERVE_SERVER_REALTIME_INPUT_H_

#include "iree/task/executor.h"
#include "loom/serve/server/realtime/binding.h"
#include "loom/serve/service/recognition/request.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_input_t loom_serve_realtime_input_t;
typedef struct loom_serve_realtime_input_item_t
    loom_serve_realtime_input_item_t;

// Owns independent microphone items and one finite conversion worker. The
// registry owner serializes every call; worker publication owes observer a
// wake. PCM owners remain alive through their final release callback.
iree_status_t loom_serve_realtime_input_allocate(
    iree_task_executor_t* executor, loom_serve_registry_t* registry,
    loom_serve_realtime_binding_t binding, loom_serve_wake_callback_t observer,
    iree_allocator_t host_allocator, loom_serve_realtime_input_t** out_input);

// Allocates one item's independent conversion history. At most four items may
// remain live, including cancelled items with accepted CPU borrowers. Source
// options are copied, and native admission occurs on a later owner advance.
iree_status_t loom_serve_realtime_input_begin(
    loom_serve_realtime_input_t* input,
    const loom_serve_recognition_request_t* request,
    loom_serve_realtime_input_item_t** out_item);

// Success borrows validated, nonempty PCM16LE mono at 24 kHz until release is
// invoked exactly once on the serialized owner. The caller keeps samples and
// release data alive through that callback, including after close. Failure
// borrows nothing. Appends target an open item and preserve converter history.
iree_status_t loom_serve_realtime_input_append(
    loom_serve_realtime_input_t* input, loom_serve_realtime_input_item_t* item,
    iree_const_byte_span_t samples, void (*release)(void*), void* user_data);

// Seals the ordered PCM frontier, flushing numerical lookahead and source EOF
// without requiring another append. The caller has validated semantic commit.
void loom_serve_realtime_input_finish(loom_serve_realtime_input_item_t* item);

// Ends publication interest immediately. Accepted conversion retains the item
// and its PCM loans until release. A completed item also retires through this
// call; the caller must discard its pointer immediately.
void loom_serve_realtime_input_close(loom_serve_realtime_input_item_t* item);

// Advances bounded admission, conversion, and native input/output consumption.
// Item-local failures remain observable below; one peer's conversion or source
// failure never terminates another session.
void loom_serve_realtime_input_advance(loom_serve_realtime_input_t* input,
                                       bool stopping, bool* out_progress);

// Final transcript in source-established segment order, or NULL until all
// accepted conversion and native result ownership has retired successfully.
const iree_string_builder_t* loom_serve_realtime_input_transcript(
    const loom_serve_realtime_input_item_t* item);
// Owned failure copy. An error is terminal; caller closes its item afterward.
iree_status_t loom_serve_realtime_input_status(
    const loom_serve_realtime_input_item_t* item);
// Input duration in 24 kHz mono frames, valid with a final transcript.
uint64_t loom_serve_realtime_input_frames(
    const loom_serve_realtime_input_item_t* item);

bool loom_serve_realtime_input_pending(
    const loom_serve_realtime_input_t* input);
// Requires no items or CPU borrowers. NULL is accepted.
void loom_serve_realtime_input_free(loom_serve_realtime_input_t* input);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_INPUT_H_
