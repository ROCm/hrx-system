// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/recognition/events.h"

#include <math.h>
#include <string.h>

struct loom_serve_recognition_events_t {
  // Allocator owning this object, decoded views, and trailing packet bytes.
  iree_allocator_t host_allocator;
  // Number of decoded views reserved at cold admission.
  iree_host_size_t event_capacity;
  // Maximum retained packet bytes following the decoded-view allocation.
  iree_host_size_t byte_capacity;
  // Successfully checked events from the latest parse; zero after failure.
  iree_host_size_t count;
  // Source-ordered views whose variable payloads borrow trailing packet bytes.
  loom_serve_recognition_event_t values[];
};

iree_status_t loom_serve_recognition_events_allocate(
    iree_host_size_t event_capacity, iree_host_size_t byte_capacity,
    iree_allocator_t host_allocator,
    loom_serve_recognition_events_t** out_events) {
  *out_events = NULL;
  if (!event_capacity || byte_capacity < 4 ||
      byte_capacity > SIZE_MAX - sizeof(loom_serve_recognition_events_t) ||
      event_capacity >
          (SIZE_MAX - sizeof(loom_serve_recognition_events_t) - byte_capacity) /
              sizeof(loom_serve_recognition_event_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recognition event storage bounds are invalid");
  }
  const iree_host_size_t size =
      sizeof(loom_serve_recognition_events_t) +
      event_capacity * sizeof(loom_serve_recognition_event_t) + byte_capacity;
  loom_serve_recognition_events_t* events = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&events));
  events->host_allocator = host_allocator;
  events->event_capacity = event_capacity;
  events->byte_capacity = byte_capacity;
  *out_events = events;
  return iree_ok_status();
}

void loom_serve_recognition_events_free(
    loom_serve_recognition_events_t* events) {
  if (events) {
    iree_allocator_free(events->host_allocator, events);
  }
}

static iree_status_t loom_serve_recognition_event_parse_transcript(
    iree_const_byte_span_t packet, uint64_t input_end,
    uint64_t* activity_position, loom_serve_recognition_event_t* out_event) {
  return loom_serve_recognition_transcript_parse(packet, input_end,
                                                 &out_event->transcript);
}

static iree_status_t loom_serve_recognition_event_parse_activity(
    iree_const_byte_span_t packet, uint64_t input_end,
    uint64_t* activity_position, loom_serve_recognition_event_t* out_event) {
  if (packet.data_length != 20) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source activity payload must contain 20 bytes");
  }
  const loom_serve_recognition_activity_t activity = {
      .begin = iree_unaligned_load_le_u64(packet.data),
      .end = iree_unaligned_load_le_u64(packet.data + 8),
      .speech_probability = iree_unaligned_load_le_f32(packet.data + 16)};
  if (activity.begin >= activity.end || activity.end > input_end ||
      !isfinite(activity.speech_probability) ||
      activity.speech_probability < 0.0f ||
      activity.speech_probability > 1.0f) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "source activity interval or probability is invalid");
  }
  if (activity.begin != *activity_position) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source activity does not continue input coverage");
  }
  *activity_position = activity.end;
  out_event->activity = activity;
  return iree_ok_status();
}

iree_status_t loom_serve_recognition_events_parse(
    loom_serve_recognition_events_t* events, iree_const_byte_span_t packet,
    uint64_t input_begin, uint64_t input_end,
    loom_serve_recognition_output_flags_t allowed_outputs) {
  static iree_status_t (*const parsers[])(iree_const_byte_span_t, uint64_t,
                                          uint64_t*,
                                          loom_serve_recognition_event_t*) = {
      [LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT] =
          loom_serve_recognition_event_parse_transcript,
      [LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY] =
          loom_serve_recognition_event_parse_activity};
  events->count = 0;
  uint8_t* storage = (uint8_t*)(events->values + events->event_capacity);
  iree_host_size_t count = 0;
  iree_host_size_t offset = 0;
  if (packet.data_length) {
    if (packet.data_length < 4 || packet.data_length > events->byte_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source event packet exceeds its byte bounds");
    }
    count = iree_unaligned_load_le_u32(packet.data);
    if (!count || count > events->event_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "source event count exceeds its declared capacity");
    }
    memcpy(storage, packet.data, packet.data_length);
    offset = 4;
  }
  uint64_t activity_position = input_begin;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_host_size_t remaining = packet.data_length - offset;
    if (remaining < 8) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "source event record is truncated");
    } else {
      const uint32_t kind = iree_unaligned_load_le_u32(storage + offset);
      const iree_host_size_t length =
          iree_unaligned_load_le_u32(storage + offset + 4);
      offset += 8;
      if (kind >= IREE_ARRAYSIZE(parsers) ||
          !iree_any_bit_set(allowed_outputs, 1u << kind)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "source event type was not requested");
      } else if (length > remaining - 8) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "source event payload is truncated");
      } else {
        events->values[i].kind = (loom_serve_recognition_event_kind_t)kind;
        status =
            parsers[kind](iree_make_const_byte_span(storage + offset, length),
                          input_end, &activity_position, &events->values[i]);
        offset += length;
      }
    }
  }
  if (iree_status_is_ok(status) && offset != packet.data_length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source event packet has trailing bytes");
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(allowed_outputs,
                       LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY) &&
      activity_position != input_end) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "source activity does not cover acknowledged input");
  }
  if (iree_status_is_ok(status)) {
    events->count = count;
  }
  return status;
}

iree_host_size_t loom_serve_recognition_events_count(
    const loom_serve_recognition_events_t* events) {
  return events->count;
}

const loom_serve_recognition_event_t* loom_serve_recognition_events_at(
    const loom_serve_recognition_events_t* events, iree_host_size_t index) {
  return &events->values[index];
}
