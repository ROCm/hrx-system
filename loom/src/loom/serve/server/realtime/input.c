// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/input.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/serve/media/audio_resampler.h"
#include "loom/serve/service/recognition/stream.h"
#include "loom/serve/service/recognition/transcription.h"

enum {
  LOOM_SERVE_REALTIME_INPUT_ITEMS = 4,
  LOOM_SERVE_REALTIME_INPUT_FRAMES = 4096,
  LOOM_SERVE_REALTIME_CONVERSION_BYTES = 4 * 1024 * 1024,
};

typedef struct loom_serve_realtime_input_packet_t {
  // Owner-only ordered append link; workers never inspect or mutate it.
  struct loom_serve_realtime_input_packet_t* next;
  // Validated original-rate PCM, borrowed through the release callback.
  iree_const_byte_span_t pcm;
  // Owner-only callback after the final conversion borrower retires.
  void (*release)(void*);
  // Caller-owned callback data, stable through release.
  void* user_data;
  // PCM bytes already committed to this item's independent converter.
  iree_host_size_t offset;
} loom_serve_realtime_input_packet_t;

struct loom_serve_realtime_input_item_t {
  // Owner-only list link, retained through cancellation and CPU retirement.
  struct loom_serve_realtime_input_item_t* next;
  // Source options borrowing the copied strings at the end of this allocation.
  loom_serve_recognition_request_t request;
  // Native receipt; a final formatting task borrows its immutable document.
  loom_serve_recognition_stream_t* stream;
  // Ordered pending PCM loans; workers borrow only the captured head.
  loom_serve_realtime_input_packet_t* first;
  // Last packet, allowing appends while an earlier packet is being converted.
  loom_serve_realtime_input_packet_t* last;
  // Source geometry, copied after admission and immutable while work is active.
  loom_serve_recognition_format_t format;
  // Independent numerical history borrowing the owner's immutable filter.
  loom_serve_audio_resampler_t resampler;
  // Owned bounded F32LE staging, written by the worker and read after release.
  float* samples;
  // Whole staged frames available for native input admission.
  iree_host_size_t frame_count;
  // Prefix of the staged frames already accepted by recognition.
  iree_host_size_t frame_offset;
  // Total accepted PCM frames in the 24 kHz wire clock.
  uint64_t input_frames;
  // Explicit item EOF; later appends belong to another item.
  bool committed;
  // Conversion emitted every output frame through explicit EOF.
  bool converted;
  // A full staging block may have left more readable converter output.
  bool buffered;
  // All converted frames and EOF have reached the native stream.
  bool finished;
  // Caller interest ended; no new CPU or native work may be admitted.
  bool closed;
  // Final text formatting completed successfully after native completion.
  bool complete;
  // Owned terminal outcome, published only after the last CPU access retires.
  iree_status_t status;
  // Final transcript, written by a finite worker after source completion.
  iree_string_builder_t transcript;
};

typedef enum loom_serve_realtime_input_work_e {
  LOOM_SERVE_REALTIME_INPUT_CONVERT,
  LOOM_SERVE_REALTIME_INPUT_FORMAT,
} loom_serve_realtime_input_work_t;

struct loom_serve_realtime_input_t {
  // Allocator owning item records, conversion storage, and packet links.
  iree_allocator_t host_allocator;
  // Shared application workers; no worker waits for input or native execution.
  iree_task_executor_t* executor;
  // Borrowed registry accessed only by the serialized serving owner.
  loom_serve_registry_t* registry;
  // Original recognition identity; later namesakes cannot receive old input.
  loom_serve_realtime_binding_t binding;
  // Nonblocking wake target, retained until every accepted worker releases.
  loom_serve_wake_callback_t observer;
  // Bounded list of independently owned items, including retiring work.
  loom_serve_realtime_input_item_t* items;
  // Number of live item allocations, including completed results.
  iree_host_size_t count;
  // Rotating item ordinal for the single finite conversion slot.
  iree_host_size_t cursor;
  // Immutable coefficients shared by this registration's independent items.
  loom_serve_audio_resampler_filter_t filter;
  // Coefficient construction completed; a partial failed allocation is not a
  // filter.
  bool filter_ready;
  // One reusable finite process; reinitialization follows actual release.
  iree_task_process_t process;
  // Captured item, or NULL when no task borrows process or numerical storage.
  loom_serve_realtime_input_item_t* working;
  // Captured packet, independent of concurrent owner appends to its next link.
  loom_serve_realtime_input_packet_t* packet;
  // Captured semantic EOF after this packet, never inferred from packet end.
  bool finish;
  // Selected numerical conversion or final immutable-document formatting.
  loom_serve_realtime_input_work_t work;
  // Joins the final release publication with collection and destruction.
  iree_slim_mutex_t mutex;
  // Accepted process no longer borrows any input or process storage.
  bool released;
  // Owned worker failure, transferred to the item only after release.
  iree_status_t status;
};

static void loom_serve_realtime_input_packet_free(
    loom_serve_realtime_input_t* input,
    loom_serve_realtime_input_packet_t* packet) {
  packet->release(packet->user_data);
  iree_allocator_free(input->host_allocator, packet);
}

static void loom_serve_realtime_input_item_free(
    loom_serve_realtime_input_t* input,
    loom_serve_realtime_input_item_t* item) {
  while (item->first) {
    loom_serve_realtime_input_packet_t* packet = item->first;
    item->first = packet->next;
    loom_serve_realtime_input_packet_free(input, packet);
  }
  if (item->stream) {
    loom_serve_recognition_stream_close(item->stream);
  }
  loom_serve_audio_resampler_deinitialize(&item->resampler);
  iree_allocator_free(input->host_allocator, item->samples);
  iree_string_builder_deinitialize(&item->transcript);
  iree_status_ignore(item->status);
  iree_allocator_free(input->host_allocator, item);
}

static iree_status_t loom_serve_realtime_input_convert(
    loom_serve_realtime_input_t* input) {
  loom_serve_realtime_input_item_t* item = input->working;
  loom_serve_realtime_input_packet_t* packet = input->packet;
  if (!input->filter_ready) {
    loom_serve_audio_resampler_filter_deinitialize(&input->filter);
    IREE_RETURN_IF_ERROR(loom_serve_audio_resampler_filter_initialize(
        24000, item->format.sample_rate, LOOM_SERVE_REALTIME_CONVERSION_BYTES,
        input->host_allocator, &input->filter));
    input->filter_ready = true;
  }
  if (!item->resampler.samples) {
    IREE_RETURN_IF_ERROR(loom_serve_audio_resampler_initialize(
        &input->filter, item->format.channels, LOOM_SERVE_REALTIME_INPUT_FRAMES,
        LOOM_SERVE_REALTIME_CONVERSION_BYTES, input->host_allocator,
        &item->resampler));
    iree_host_size_t staging_bytes = 0;
    if (!iree_host_size_checked_mul(
            LOOM_SERVE_REALTIME_INPUT_FRAMES * sizeof(float),
            item->format.channels, &staging_bytes) ||
        staging_bytes > LOOM_SERVE_REALTIME_CONVERSION_BYTES) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "recognition geometry exceeds conversion capacity");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        input->host_allocator, staging_bytes, (void**)&item->samples));
  }
  item->frame_count = 0;
  item->frame_offset = 0;
  iree_host_size_t received = 0;
  while (item->frame_count < LOOM_SERVE_REALTIME_INPUT_FRAMES) {
    if (input->finish &&
        (!packet || packet->offset == packet->pcm.data_length)) {
      loom_serve_audio_resampler_finish(&item->resampler);
    }
    item->frame_count += loom_serve_audio_resampler_read(
        &item->resampler, LOOM_SERVE_REALTIME_INPUT_FRAMES - item->frame_count,
        item->samples + item->frame_count * item->format.channels);
    item->converted = loom_serve_audio_resampler_is_complete(&item->resampler);
    if (item->converted ||
        item->frame_count == LOOM_SERVE_REALTIME_INPUT_FRAMES ||
        received == LOOM_SERVE_REALTIME_INPUT_FRAMES || !packet ||
        packet->offset == packet->pcm.data_length) {
      break;
    }
    iree_host_size_t capacity = 0;
    float* samples =
        loom_serve_audio_resampler_reserve(&item->resampler, &capacity);
    const iree_host_size_t count = iree_min(
        (packet->pcm.data_length - packet->offset) / sizeof(int16_t),
        iree_min(capacity, LOOM_SERVE_REALTIME_INPUT_FRAMES - received));
    for (iree_host_size_t i = 0; i < count; ++i) {
      const float sample =
          (float)(int16_t)iree_unaligned_load_le_u16(
              packet->pcm.data + packet->offset + i * sizeof(int16_t)) /
          32768.0f;
      for (uint32_t channel = 0; channel < item->format.channels; ++channel) {
        samples[i * item->format.channels + channel] = sample;
      }
    }
    loom_serve_audio_resampler_commit(&item->resampler, count);
    packet->offset += count * sizeof(int16_t);
    received += count;
  }
  item->buffered = item->frame_count == LOOM_SERVE_REALTIME_INPUT_FRAMES;
  for (iree_host_size_t i = 0; i < item->frame_count * item->format.channels;
       ++i) {
    iree_unaligned_store_le_f32(item->samples + i, item->samples[i]);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_input_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_realtime_input_t* input = process->user_data;
  result->completed = true;
  result->did_work = true;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  if (input->work == LOOM_SERVE_REALTIME_INPUT_FORMAT) {
    status = loom_serve_transcription_format(
        loom_serve_recognition_stream_document(input->working->stream), true,
        &input->working->transcript);
  } else {
    status = loom_serve_realtime_input_convert(input);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_serve_realtime_input_completed(iree_task_process_t* process,
                                                iree_status_t status) {
  loom_serve_realtime_input_t* input = process->user_data;
  iree_slim_mutex_lock(&input->mutex);
  input->status = status;
  iree_slim_mutex_unlock(&input->mutex);
}

static void loom_serve_realtime_input_released(iree_task_process_t* process) {
  loom_serve_realtime_input_t* input = process->user_data;
  iree_slim_mutex_lock(&input->mutex);
  input->released = true;
  input->observer.wake(input->observer.user_data);
  iree_slim_mutex_unlock(&input->mutex);
}

iree_status_t loom_serve_realtime_input_allocate(
    iree_task_executor_t* executor, loom_serve_registry_t* registry,
    loom_serve_realtime_binding_t binding, loom_serve_wake_callback_t observer,
    iree_allocator_t host_allocator, loom_serve_realtime_input_t** out_input) {
  *out_input = NULL;
  loom_serve_realtime_input_t* input = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*input), iree_alignof(loom_serve_realtime_input_t),
      0, (void**)&input));
  input->host_allocator = host_allocator;
  input->executor = executor;
  input->registry = registry;
  input->binding = binding;
  input->observer = observer;
  iree_slim_mutex_initialize(&input->mutex);
  *out_input = input;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_input_begin(
    loom_serve_realtime_input_t* input,
    const loom_serve_recognition_request_t* request,
    loom_serve_realtime_input_item_t** out_item) {
  *out_item = NULL;
  if (input->count == LOOM_SERVE_REALTIME_INPUT_ITEMS) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "realtime input item capacity is full");
  }
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(loom_serve_realtime_input_item_t), &total_size,
                         IREE_STRUCT_FIELD(request->language.size, char, NULL),
                         IREE_STRUCT_FIELD(request->prompt.size, char, NULL)));
  loom_serve_realtime_input_item_t* item = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(input->host_allocator, total_size, (void**)&item));
  item->request = *request;
  char* storage = (char*)(item + 1);
  if (request->language.size) {
    memcpy(storage, request->language.data, request->language.size);
  }
  item->request.language =
      iree_make_string_view(storage, request->language.size);
  storage += request->language.size;
  if (request->prompt.size) {
    memcpy(storage, request->prompt.data, request->prompt.size);
  }
  item->request.prompt = iree_make_string_view(storage, request->prompt.size);
  iree_string_builder_initialize(input->host_allocator, &item->transcript);
  item->next = input->items;
  input->items = item;
  ++input->count;
  *out_item = item;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_input_append(
    loom_serve_realtime_input_t* input, loom_serve_realtime_input_item_t* item,
    iree_const_byte_span_t samples, void (*release)(void*), void* user_data) {
  loom_serve_realtime_input_packet_t* packet = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(input->host_allocator,
                                             sizeof(*packet), (void**)&packet));
  packet->pcm = samples;
  packet->release = release;
  packet->user_data = user_data;
  if (item->last) {
    item->last->next = packet;
  } else {
    item->first = packet;
  }
  item->last = packet;
  item->input_frames += samples.data_length / sizeof(int16_t);
  return iree_ok_status();
}

void loom_serve_realtime_input_finish(loom_serve_realtime_input_item_t* item) {
  item->committed = true;
}

void loom_serve_realtime_input_close(loom_serve_realtime_input_item_t* item) {
  item->closed = true;
}

static void loom_serve_realtime_input_collect(
    loom_serve_realtime_input_t* input, bool* out_progress) {
  if (!input->working) {
    return;
  }
  iree_slim_mutex_lock(&input->mutex);
  const bool released = input->released;
  iree_slim_mutex_unlock(&input->mutex);
  if (!released) {
    return;
  }
  loom_serve_realtime_input_item_t* item = input->working;
  item->status = iree_status_join(item->status, input->status);
  input->status = iree_ok_status();
  if (input->packet &&
      input->packet->offset == input->packet->pcm.data_length) {
    item->first = input->packet->next;
    if (!item->first) {
      item->last = NULL;
    }
    loom_serve_realtime_input_packet_free(input, input->packet);
  }
  if (input->work == LOOM_SERVE_REALTIME_INPUT_FORMAT) {
    item->complete = iree_status_is_ok(item->status);
    loom_serve_recognition_stream_close(item->stream);
    item->stream = NULL;
  }
  input->working = NULL;
  input->packet = NULL;
  *out_progress = true;
}

static iree_status_t loom_serve_realtime_input_item_advance(
    loom_serve_realtime_input_t* input, loom_serve_realtime_input_item_t* item,
    bool* out_progress) {
  if (!item->stream) {
    if (!loom_serve_realtime_binding_present(input->registry,
                                             &input->binding)) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "recognition registration has retired");
    }
    IREE_RETURN_IF_ERROR(loom_serve_registry_try_open_recognition(
        input->registry, input->binding.name, &item->request, &item->stream));
    if (!item->stream) {
      return iree_ok_status();
    }
    item->format = loom_serve_recognition_stream_format(item->stream);
    *out_progress = true;
  }
  IREE_RETURN_IF_ERROR(loom_serve_recognition_stream_status(item->stream));
  if (item->frame_offset < item->frame_count) {
    iree_host_size_t accepted = 0;
    IREE_RETURN_IF_ERROR(loom_serve_recognition_stream_write(
        item->stream,
        iree_make_const_byte_span(
            item->samples + item->frame_offset * item->format.channels,
            (item->frame_count - item->frame_offset) * item->format.channels *
                sizeof(float)),
        &accepted));
    item->frame_offset += accepted;
    *out_progress |= accepted != 0;
  }
  if (item->converted && !item->finished &&
      item->frame_offset == item->frame_count) {
    loom_serve_recognition_stream_finish(item->stream);
    item->finished = true;
    *out_progress = true;
  }
  // Replacement hypotheses belong to the source document. A final transcript
  // is formatted after source completion instead of pretending revisions are
  // monotonic textual deltas.
  for (iree_host_size_t i = 0;
       i < 16 && loom_serve_recognition_stream_peek(item->stream); ++i) {
    loom_serve_recognition_stream_consume(item->stream);
    *out_progress = true;
  }
  return iree_ok_status();
}

void loom_serve_realtime_input_advance(loom_serve_realtime_input_t* input,
                                       bool stopping, bool* out_progress) {
  loom_serve_realtime_input_collect(input, out_progress);
  loom_serve_realtime_input_item_t** link = &input->items;
  while (*link) {
    loom_serve_realtime_input_item_t* item = *link;
    item->closed |= stopping;
    if (item == input->working) {
      link = &item->next;
      continue;
    }
    if (item->closed) {
      *link = item->next;
      loom_serve_realtime_input_item_free(input, item);
      --input->count;
      *out_progress = true;
      continue;
    }
    if (!item->complete && iree_status_is_ok(item->status)) {
      item->status =
          loom_serve_realtime_input_item_advance(input, item, out_progress);
    }
    link = &item->next;
  }
  if (input->working || !input->count) {
    return;
  }
  for (iree_host_size_t i = 0; i < input->count; ++i) {
    const iree_host_size_t ordinal = (input->cursor + i) % input->count;
    loom_serve_realtime_input_item_t* item = input->items;
    for (iree_host_size_t j = 0; j < ordinal; ++j) {
      item = item->next;
    }
    if (item->closed || item->complete || !iree_status_is_ok(item->status) ||
        !item->stream || item->frame_offset != item->frame_count) {
      continue;
    }
    const bool format = loom_serve_recognition_stream_is_complete(item->stream);
    if (!format && (item->converted ||
                    (!item->first && !item->committed && !item->buffered))) {
      continue;
    }
    input->working = item;
    input->packet = item->first;
    input->finish = item->committed && item->first == item->last;
    input->work = format ? LOOM_SERVE_REALTIME_INPUT_FORMAT
                         : LOOM_SERVE_REALTIME_INPUT_CONVERT;
    input->released = false;
    input->cursor = (ordinal + 1) % input->count;
    iree_task_process_initialize(loom_serve_realtime_input_run, 0, 1,
                                 &input->process);
    input->process.user_data = input;
    input->process.completion_fn = loom_serve_realtime_input_completed;
    input->process.release_fn = loom_serve_realtime_input_released;
    iree_task_executor_schedule_process(input->executor, &input->process);
    *out_progress = true;
    break;
  }
}

const iree_string_builder_t* loom_serve_realtime_input_transcript(
    const loom_serve_realtime_input_item_t* item) {
  return item->complete ? &item->transcript : NULL;
}

iree_status_t loom_serve_realtime_input_status(
    const loom_serve_realtime_input_item_t* item) {
  return iree_status_clone(item->status);
}

uint64_t loom_serve_realtime_input_frames(
    const loom_serve_realtime_input_item_t* item) {
  return item->input_frames;
}

bool loom_serve_realtime_input_pending(
    const loom_serve_realtime_input_t* input) {
  return input->items != NULL;
}

void loom_serve_realtime_input_free(loom_serve_realtime_input_t* input) {
  if (!input) {
    return;
  }
  loom_serve_audio_resampler_filter_deinitialize(&input->filter);
  iree_slim_mutex_deinitialize(&input->mutex);
  iree_allocator_free_aligned(input->host_allocator, input);
}
