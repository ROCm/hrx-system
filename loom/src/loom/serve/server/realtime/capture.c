// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/capture.h"

#include <string.h>

#include "iree/base/internal/base64.h"
#include "iree/base/internal/json.h"
#include "iree/base/threading/mutex.h"

typedef struct loom_serve_realtime_capture_packet_t {
  // Owner-only append link, never inspected by the decoding worker.
  struct loom_serve_realtime_capture_packet_t* next;
  // Transport receipt retained through the final decoded block publication.
  loom_serve_websocket_message_t* message;
  // Escaped JSON string borrowing the receipt above.
  iree_string_view_t audio;
  // Owned validated PCM16LE, bounded by the transport's message capacity.
  iree_byte_span_t samples;
  // Whole sample bytes already copied into reserved capture storage.
  iree_host_size_t offset;
} loom_serve_realtime_capture_packet_t;

struct loom_serve_realtime_capture_t {
  // Allocator owning this record, raw ring, and encoded packet records.
  iree_allocator_t host_allocator;
  // Borrowed shared worker pool; no task waits for source or reader progress.
  iree_task_executor_t* executor;
  // Borrowed transport retained by the session's HTTP claim.
  loom_serve_websocket_t* socket;
  // Stable owner notification target retained through the last CPU release.
  loom_serve_wake_callback_t observer;
  // Ordered accepted appends, including the packet borrowed by a worker.
  loom_serve_realtime_capture_packet_t* first;
  // Last accepted append, allowing controls/appends during a prior decode.
  loom_serve_realtime_capture_packet_t* last;
  // Immutable-capacity ring of original PCM16LE mono samples.
  uint8_t* samples;
  // Ring capacity in original 24 kHz frames.
  iree_host_size_t capacity;
  // Published original-rate endpoint, excluding accepted unpublished writes.
  uint64_t end;
  // Caller-selected earliest retained frame, before outstanding loan pinning.
  uint64_t retain;
  // Owner-only list of borrowed contiguous spans into the ring.
  loom_serve_realtime_capture_span_t* spans;
  // At least one nonempty append joined this capture's ordered frontier.
  bool appended;
  // Explicit semantic EOF; network packet exhaustion does not set this field.
  bool finished;
  // No further publication or CPU admission is permitted.
  bool closed;
  // Owned terminal failure, visible only after the failing task releases.
  iree_status_t status;
  // One reusable finite decoder/copy process, retained through final release.
  iree_task_process_t process;
  // True while an accepted process can borrow packet or ring storage.
  bool working;
  // Captured contiguous ring offset for the accepted task, in frames.
  iree_host_size_t write_offset;
  // Captured free ring extent, bounded to one finite work block.
  iree_host_size_t write_capacity;
  // Whole frames copied by the worker, published only after release.
  iree_host_size_t produced;
  // Joins process release publication with owner collection and destruction.
  iree_slim_mutex_t mutex;
  // Accepted process no longer borrows any capture or packet storage.
  bool released;
  // Owned process outcome, transferred to status after final release.
  iree_status_t work_status;
};

static void loom_serve_realtime_capture_packet_free(
    loom_serve_realtime_capture_t* capture,
    loom_serve_realtime_capture_packet_t* packet) {
  loom_serve_websocket_message_release(capture->socket, packet->message);
  iree_allocator_free(capture->host_allocator, packet->samples.data);
  iree_allocator_free(capture->host_allocator, packet);
}

static iree_status_t loom_serve_realtime_capture_decode(
    loom_serve_realtime_capture_t* capture,
    loom_serve_realtime_capture_packet_t* packet) {
  char* encoded = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      capture->host_allocator, packet->audio.size, (void**)&encoded));
  iree_host_size_t encoded_length = 0;
  iree_status_t status = iree_json_unescape_string(
      packet->audio, packet->audio.size, encoded, &encoded_length);
  const iree_string_view_t text =
      iree_make_string_view(encoded, encoded_length);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(capture->host_allocator,
                                   iree_base64_decoded_size(text),
                                   (void**)&packet->samples.data);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_base64_decode(text,
                           iree_make_byte_span(packet->samples.data,
                                               iree_base64_decoded_size(text)),
                           &packet->samples.data_length);
  }
  iree_allocator_free(capture->host_allocator, encoded);
  if (iree_status_is_ok(status) &&
      (!packet->samples.data_length ||
       packet->samples.data_length % sizeof(int16_t))) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "audio append requires whole nonempty PCM16 frames");
  }
  return status;
}

static iree_status_t loom_serve_realtime_capture_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_realtime_capture_t* capture = process->user_data;
  loom_serve_realtime_capture_packet_t* packet = capture->first;
  result->completed = true;
  result->did_work = true;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  if (!packet->samples.data) {
    status = loom_serve_realtime_capture_decode(capture, packet);
  }
  if (iree_status_is_ok(status)) {
    capture->produced = iree_min(
        capture->write_capacity,
        (packet->samples.data_length - packet->offset) / sizeof(int16_t));
    if (capture->produced > UINT64_MAX - capture->end) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "microphone frame clock exceeds uint64");
    } else {
      const iree_host_size_t bytes = capture->produced * sizeof(int16_t);
      memcpy(capture->samples + capture->write_offset * sizeof(int16_t),
             packet->samples.data + packet->offset, bytes);
      packet->offset += bytes;
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_serve_realtime_capture_completed(iree_task_process_t* process,
                                                  iree_status_t status) {
  loom_serve_realtime_capture_t* capture = process->user_data;
  iree_slim_mutex_lock(&capture->mutex);
  capture->work_status = status;
  iree_slim_mutex_unlock(&capture->mutex);
}

static void loom_serve_realtime_capture_released(iree_task_process_t* process) {
  loom_serve_realtime_capture_t* capture = process->user_data;
  iree_slim_mutex_lock(&capture->mutex);
  capture->released = true;
  capture->observer.wake(capture->observer.user_data);
  iree_slim_mutex_unlock(&capture->mutex);
}

iree_status_t loom_serve_realtime_capture_allocate(
    iree_task_executor_t* executor, loom_serve_websocket_t* socket,
    iree_host_size_t capacity, loom_serve_wake_callback_t observer,
    iree_allocator_t host_allocator,
    loom_serve_realtime_capture_t** out_capture) {
  *out_capture = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t samples_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_realtime_capture_t), &total_size,
      IREE_STRUCT_FIELD(capacity, int16_t, &samples_offset)));
  loom_serve_realtime_capture_t* capture = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, total_size, iree_alignof(loom_serve_realtime_capture_t),
      0, (void**)&capture));
  capture->host_allocator = host_allocator;
  capture->executor = executor;
  capture->socket = socket;
  capture->observer = observer;
  capture->samples = (uint8_t*)capture + samples_offset;
  capture->capacity = capacity;
  iree_slim_mutex_initialize(&capture->mutex);
  *out_capture = capture;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_capture_append(
    loom_serve_realtime_capture_t* capture,
    loom_serve_websocket_message_t* message, iree_string_view_t audio) {
  if (capture->finished || !audio.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "audio append requires a nonempty open capture");
  }
  loom_serve_realtime_capture_packet_t* packet = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(capture->host_allocator,
                                             sizeof(*packet), (void**)&packet));
  packet->message = message;
  packet->audio = audio;
  if (capture->last) {
    capture->last->next = packet;
  } else {
    capture->first = packet;
  }
  capture->last = packet;
  capture->appended = true;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_capture_finish(
    loom_serve_realtime_capture_t* capture) {
  if (!capture->appended || capture->finished) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "commit requires nonempty uncommitted audio");
  }
  capture->finished = true;
  return iree_ok_status();
}

void loom_serve_realtime_capture_advance(loom_serve_realtime_capture_t* capture,
                                         bool* out_progress) {
  if (capture->working) {
    iree_slim_mutex_lock(&capture->mutex);
    const bool released = capture->released;
    iree_slim_mutex_unlock(&capture->mutex);
    if (!released) {
      return;
    }
    capture->status = iree_status_join(capture->status, capture->work_status);
    capture->work_status = iree_ok_status();
    if (iree_status_is_ok(capture->status) && !capture->closed) {
      capture->end += capture->produced;
    }
    loom_serve_realtime_capture_packet_t* packet = capture->first;
    if (packet->offset == packet->samples.data_length) {
      capture->first = packet->next;
      if (!capture->first) {
        capture->last = NULL;
      }
      loom_serve_realtime_capture_packet_free(capture, packet);
    }
    capture->working = false;
    *out_progress = true;
  }
  if (capture->closed) {
    while (capture->first) {
      loom_serve_realtime_capture_packet_t* packet = capture->first;
      capture->first = packet->next;
      loom_serve_realtime_capture_packet_free(capture, packet);
      *out_progress = true;
    }
    capture->last = NULL;
    return;
  }
  if (!capture->first || !iree_status_is_ok(capture->status)) {
    return;
  }
  uint64_t begin = capture->retain;
  for (const loom_serve_realtime_capture_span_t* span = capture->spans; span;
       span = span->next) {
    begin = iree_min(begin, span->position);
  }
  const iree_host_size_t available =
      capture->capacity - (iree_host_size_t)(capture->end - begin);
  if (!available) {
    return;
  }
  capture->write_offset = capture->end % capture->capacity;
  capture->write_capacity = iree_min(
      4096, iree_min(available, capture->capacity - capture->write_offset));
  capture->produced = 0;
  capture->released = false;
  capture->working = true;
  iree_task_process_initialize(loom_serve_realtime_capture_run, 0, 1,
                               &capture->process);
  capture->process.user_data = capture;
  capture->process.completion_fn = loom_serve_realtime_capture_completed;
  capture->process.release_fn = loom_serve_realtime_capture_released;
  iree_task_executor_schedule_process(capture->executor, &capture->process);
  *out_progress = true;
}

iree_status_t loom_serve_realtime_capture_status(
    const loom_serve_realtime_capture_t* capture) {
  return iree_status_clone(capture->status);
}

uint64_t loom_serve_realtime_capture_end(
    const loom_serve_realtime_capture_t* capture) {
  return capture->end;
}

bool loom_serve_realtime_capture_is_complete(
    const loom_serve_realtime_capture_t* capture) {
  return capture->finished && !capture->first && !capture->working &&
         iree_status_is_ok(capture->status);
}

void loom_serve_realtime_capture_borrow(
    loom_serve_realtime_capture_t* capture, uint64_t position,
    iree_host_size_t maximum, loom_serve_realtime_capture_span_t* span) {
  const iree_host_size_t offset = position % capture->capacity;
  const iree_host_size_t count = iree_min(
      maximum, iree_min(capture->end - position, capture->capacity - offset));
  *span = (loom_serve_realtime_capture_span_t){
      .next = capture->spans,
      .capture = capture,
      .position = position,
      .samples =
          iree_make_const_byte_span(capture->samples + offset * sizeof(int16_t),
                                    count * sizeof(int16_t))};
  capture->spans = span;
}

void loom_serve_realtime_capture_release(void* user_data) {
  loom_serve_realtime_capture_span_t* span = user_data;
  loom_serve_realtime_capture_span_t** link = &span->capture->spans;
  while (*link != span) {
    link = &(*link)->next;
  }
  *link = span->next;
  *span = (loom_serve_realtime_capture_span_t){0};
}

void loom_serve_realtime_capture_discard_before(
    loom_serve_realtime_capture_t* capture, uint64_t position) {
  capture->retain = position;
}

void loom_serve_realtime_capture_close(loom_serve_realtime_capture_t* capture) {
  capture->closed = true;
}

bool loom_serve_realtime_capture_pending(
    const loom_serve_realtime_capture_t* capture) {
  return capture->first || capture->working || capture->spans;
}

void loom_serve_realtime_capture_free(loom_serve_realtime_capture_t* capture) {
  if (!capture) {
    return;
  }
  iree_status_ignore(capture->status);
  iree_slim_mutex_deinitialize(&capture->mutex);
  iree_allocator_free_aligned(capture->host_allocator, capture);
}
