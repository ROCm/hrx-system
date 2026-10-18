// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_RECOGNITION_EVENTS_H_
#define LOOM_SERVE_SERVICE_RECOGNITION_EVENTS_H_

#include "loom/serve/service/recognition/transcript.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_recognition_event_kind_e {
  LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT = 0,
  LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY = 1,
} loom_serve_recognition_event_kind_t;

enum loom_serve_recognition_output_flag_bits_e {
  LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT =
      1u << LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT,
  LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY =
      1u << LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY,
  LOOM_SERVE_RECOGNITION_OUTPUT_ALL = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT |
                                      LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY,
};
typedef uint32_t loom_serve_recognition_output_flags_t;

// Source-established activity over actual input, independent of turn policy.
// Requested observations partition each collection's acknowledged input prefix
// in order. Lookahead remains unconsumed until its observation is established.
// Coverage is not a claim of exact speech onset, offset, or word alignment.
typedef struct loom_serve_recognition_activity_t {
  // Inclusive input sample-frame position covered by the observation.
  uint64_t begin;
  // Exclusive input sample-frame position, strictly greater than begin.
  uint64_t end;
  // Finite source probability in [0, 1]; thresholds belong to the consumer.
  float speech_probability;
} loom_serve_recognition_activity_t;

// One immutable observation borrowing its enclosing event storage. Multiple
// revisions of one segment remain distinct even after the document is updated.
typedef struct loom_serve_recognition_event_t {
  // Selects the only initialized payload member.
  loom_serve_recognition_event_kind_t kind;
  union {
    // Replacement text and optional source alignment or speaker evidence.
    loom_serve_recognition_transcript_t transcript;
    // Speech probability and its source-established input coverage.
    loom_serve_recognition_activity_t activity;
  };
} loom_serve_recognition_event_t;

typedef struct loom_serve_recognition_events_t loom_serve_recognition_events_t;

// Allocates one reusable packet and decoded-event array at cold admission.
// Both bounds come from source metadata. Parsing and queries never allocate.
iree_status_t loom_serve_recognition_events_allocate(
    iree_host_size_t event_capacity, iree_host_size_t byte_capacity,
    iree_allocator_t host_allocator,
    loom_serve_recognition_events_t** out_events);

// Releases caller-owned output after its last CPU borrower retires. NULL is
// valid; no model or device lifetime is required.
void loom_serve_recognition_events_free(
    loom_serve_recognition_events_t* events);

// Copies and checks one VM/source packet, invalidating previous views. Failure
// publishes no events. An empty buffer means no events. A nonempty packet is a
// positive U32 count followed by count records: kind (U32), payload bytes
// (U32), and payload, without padding. All fields are little-endian. Transcript
// payloads use transcript_parse's representation; activity is begin/end (U64)
// and speech probability (F32). No trailing bytes are permitted. input_begin
// and input_end bound the newly acknowledged input prefix. Requested activity
// must cover that exact interval without gaps or overlap, including empty cuts.
// allowed_outputs is the admitted request selection.
iree_status_t loom_serve_recognition_events_parse(
    loom_serve_recognition_events_t* events, iree_const_byte_span_t packet,
    uint64_t input_begin, uint64_t input_end,
    loom_serve_recognition_output_flags_t allowed_outputs);

// Direct queries of successfully checked output; no decoding or allocation.
iree_host_size_t loom_serve_recognition_events_count(
    const loom_serve_recognition_events_t* events);
// index is below events_count. The view survives until the next parse or free.
const loom_serve_recognition_event_t* loom_serve_recognition_events_at(
    const loom_serve_recognition_events_t* events, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_RECOGNITION_EVENTS_H_
