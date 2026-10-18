// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/generation.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/base/threading/mutex.h"
#include "loom/serve/service/speech/stream.h"

struct loom_serve_realtime_generation_t {
  // Allocator for this owner and finite worker output.
  iree_allocator_t host_allocator;
  // Shared workers for bounded waveform conversion and encoding.
  iree_task_executor_t* executor;
  // Serialized registry owner, never accessed by conversion callbacks.
  loom_serve_registry_t* registry;
  // Original text deployment identity, checked before native admission.
  loom_serve_realtime_binding_t text_binding;
  // Original speech deployment identity, checked before native admission.
  loom_serve_realtime_binding_t speech_binding;
  // Owning source-defined input, moved to native text only on acceptance.
  loom_serve_text_request_t text_request;
  // Owning continuing speech options, moved only on native acceptance.
  loom_serve_speech_request_t speech_request;
  // Native text handle retaining current spans and final semantic output.
  loom_serve_text_stream_t* text;
  // Native waveform handle, closed after any accepted conversion borrower.
  loom_serve_speech_stream_t* speech;
  // Session-owned coefficients and per-response conversion cursor.
  loom_serve_realtime_audio_t* audio;
  // This response requests audio in addition to source text semantics.
  bool speaks;
  // No new publication, native admission, or worker submission is permitted.
  bool cancelled;
  // Text completion has published EOF to continuing speech.
  bool finished;
  // The next source span within the borrowed native text block.
  iree_host_size_t span_index;
  // Prefix of that span already forwarded to continuing speech.
  iree_host_size_t byte_offset;
  // Owned bounded content prefix plus at most one incomplete UTF-8 scalar.
  iree_string_builder_t pending_text;
  // Complete UTF-8 prefix ready for a JSON text event.
  iree_host_size_t text_length;
  // Owned base64 packet, published only after the encoding worker releases.
  iree_string_builder_t packet;
  // Copied generated-frame frontier, safe to query during later CPU work.
  uint64_t frames;
  // Bounded conversion process; its storage is reused only after release.
  iree_task_process_t process;
  // Native block borrowed by the accepted worker, or NULL while idle.
  const loom_serve_speech_block_t* block;
  // First conversion task initializes a fresh per-response cursor.
  bool audio_started;
  // Worker has converted the entire captured block, including final lookahead.
  bool consumed;
  // Joins completion/release publication with collection and destruction.
  iree_slim_mutex_t mutex;
  // Nonblocking serving-owner wake, retained through release publication.
  loom_serve_wake_callback_t observer;
  // Final callback relinquished process and waveform storage.
  bool released;
  // Owned worker failure, consumed exactly once by the serving owner.
  iree_status_t status;
};

static iree_status_t loom_serve_realtime_generation_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_realtime_generation_t* generation = process->user_data;
  result->completed = true;
  result->did_work = true;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!generation->audio_started) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, loom_serve_realtime_audio_begin(
                loom_serve_speech_stream_format(generation->speech),
                generation->host_allocator, generation->audio));
    generation->audio_started = true;
  }
  const iree_status_t status = loom_serve_realtime_audio_convert(
      generation->audio, generation->block, &generation->packet,
      &generation->consumed);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_serve_realtime_generation_completed(
    iree_task_process_t* process, iree_status_t status) {
  loom_serve_realtime_generation_t* generation = process->user_data;
  iree_slim_mutex_lock(&generation->mutex);
  generation->status = status;
  iree_slim_mutex_unlock(&generation->mutex);
}

static void loom_serve_realtime_generation_released(
    iree_task_process_t* process) {
  loom_serve_realtime_generation_t* generation = process->user_data;
  iree_slim_mutex_lock(&generation->mutex);
  generation->released = true;
  generation->observer.wake(generation->observer.user_data);
  iree_slim_mutex_unlock(&generation->mutex);
}

iree_status_t loom_serve_realtime_generation_allocate(
    iree_task_executor_t* executor, loom_serve_registry_t* registry,
    loom_serve_realtime_binding_t text, loom_serve_realtime_binding_t speech,
    loom_serve_text_request_t* text_request, iree_string_view_t voice,
    uint64_t seed, loom_serve_realtime_audio_t* audio,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_realtime_generation_t** out_generation) {
  *out_generation = NULL;
  loom_serve_realtime_generation_t* generation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*generation),
      iree_alignof(loom_serve_realtime_generation_t), 0, (void**)&generation));
  generation->host_allocator = host_allocator;
  generation->executor = executor;
  generation->registry = registry;
  generation->text_binding = text;
  generation->speech_binding = speech;
  generation->audio = audio;
  generation->observer = observer;
  generation->speaks = voice.size != 0;
  iree_status_t status = iree_ok_status();
  if (generation->speaks) {
    status = loom_serve_speech_request_initialize_stream(
        voice, iree_string_view_empty(), seed, 1.0f, host_allocator,
        &generation->speech_request);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free_aligned(host_allocator, generation);
    return status;
  }
  iree_slim_mutex_initialize(&generation->mutex);
  iree_string_builder_initialize(host_allocator, &generation->packet);
  iree_string_builder_initialize(host_allocator, &generation->pending_text);
  generation->text_request = *text_request;
  *text_request = (loom_serve_text_request_t){0};
  *out_generation = generation;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_generation_collect(
    loom_serve_realtime_generation_t* generation, bool* out_progress) {
  if (!generation->block) {
    return iree_ok_status();
  }
  iree_slim_mutex_lock(&generation->mutex);
  const bool released = generation->released;
  iree_slim_mutex_unlock(&generation->mutex);
  if (!released) {
    return iree_ok_status();
  }
  generation->block = NULL;
  generation->frames = generation->audio->frames;
  if (generation->cancelled) {
    loom_serve_speech_stream_close(generation->speech);
    generation->speech = NULL;
    iree_string_builder_reset(&generation->packet);
  } else if (generation->consumed) {
    loom_serve_speech_stream_consume(generation->speech);
  }
  const iree_status_t status = generation->status;
  generation->status = iree_ok_status();
  *out_progress = true;
  return status;
}

static iree_status_t loom_serve_realtime_generation_admit(
    loom_serve_realtime_generation_t* generation, bool* out_progress) {
  if (!generation->text) {
    if (!loom_serve_realtime_binding_present(generation->registry,
                                             &generation->text_binding)) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "text registration has retired");
    }
    IREE_RETURN_IF_ERROR(loom_serve_registry_try_open_text(
        generation->registry, generation->text_binding.name,
        &generation->text_request, &generation->text));
    *out_progress |= generation->text != NULL;
  }
  if (generation->speaks && !generation->speech) {
    if (!loom_serve_realtime_binding_present(generation->registry,
                                             &generation->speech_binding)) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "speech registration has retired");
    }
    IREE_RETURN_IF_ERROR(loom_serve_registry_try_open_speech(
        generation->registry, generation->speech_binding.name,
        &generation->speech_request, &generation->speech));
    *out_progress |= generation->speech != NULL;
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_generation_forward(
    loom_serve_realtime_generation_t* generation, bool* out_progress) {
  if (!generation->text || (generation->speaks && !generation->speech) ||
      generation->text_length) {
    return iree_ok_status();
  }
  const loom_serve_text_stream_block_t* block =
      loom_serve_text_stream_peek(generation->text);
  if (block) {
    if (generation->span_index < loom_serve_text_stream_block_count(block)) {
      const loom_serve_text_stream_span_t span =
          loom_serve_text_stream_block_at(block, generation->span_index);
      const bool content = span.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT;
      iree_host_size_t count = span.text.size - generation->byte_offset;
      if (content) {
        count = iree_min(count, (iree_host_size_t)8192);
      }
      if (content && count && generation->speaks) {
        IREE_RETURN_IF_ERROR(loom_serve_speech_stream_write(
            generation->speech,
            iree_make_const_byte_span(span.text.data + generation->byte_offset,
                                      count),
            &count));
      }
      if (content && count) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
            &generation->pending_text,
            iree_make_string_view(span.text.data + generation->byte_offset,
                                  count)));
        const iree_string_view_t pending =
            iree_string_builder_view(&generation->pending_text);
        generation->text_length =
            pending.size - iree_unicode_utf8_incomplete_tail_length(
                               pending.data, pending.size);
        if (!iree_unicode_utf8_validate(
                iree_string_view_substr(pending, 0, generation->text_length))) {
          return iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "text source produced invalid UTF-8 content");
        }
      }
      generation->byte_offset += count;
      if (generation->byte_offset == span.text.size) {
        ++generation->span_index;
        generation->byte_offset = 0;
        *out_progress = true;
      }
      *out_progress |= count != 0;
    }
    if (generation->span_index == loom_serve_text_stream_block_count(block)) {
      loom_serve_text_stream_consume(generation->text);
      generation->span_index = 0;
      *out_progress = true;
    }
  }
  if (!generation->finished &&
      loom_serve_text_stream_is_complete(generation->text)) {
    if (generation->pending_text.size != generation->text_length) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "text source ended within a UTF-8 scalar");
    }
    if (generation->speech) {
      IREE_RETURN_IF_ERROR(loom_serve_speech_stream_finish(generation->speech));
    }
    generation->finished = true;
    *out_progress = true;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_generation_advance(
    loom_serve_realtime_generation_t* generation, bool can_publish,
    bool* out_progress) {
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_generation_collect(generation, out_progress));
  if (generation->cancelled) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_generation_admit(generation, out_progress));
  if (generation->text) {
    IREE_RETURN_IF_ERROR(loom_serve_text_stream_status(generation->text));
  }
  if (generation->speech) {
    IREE_RETURN_IF_ERROR(loom_serve_speech_stream_status(generation->speech));
  }
  if (!can_publish) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_generation_forward(generation, out_progress));
  if (generation->speech && !generation->block && !generation->packet.size) {
    generation->block = loom_serve_speech_stream_peek(generation->speech);
    if (generation->block) {
      generation->released = false;
      generation->consumed = false;
      iree_task_process_initialize(loom_serve_realtime_generation_run, 0, 1,
                                   &generation->process);
      generation->process.user_data = generation;
      generation->process.completion_fn =
          loom_serve_realtime_generation_completed;
      generation->process.release_fn = loom_serve_realtime_generation_released;
      iree_task_executor_schedule_process(generation->executor,
                                          &generation->process);
      *out_progress = true;
    }
  }
  return iree_ok_status();
}

iree_string_view_t loom_serve_realtime_generation_text(
    const loom_serve_realtime_generation_t* generation) {
  return iree_make_string_view(generation->pending_text.buffer,
                               generation->text_length);
}

void loom_serve_realtime_generation_consume_text(
    loom_serve_realtime_generation_t* generation) {
  const iree_host_size_t remaining =
      generation->pending_text.size - generation->text_length;
  if (remaining) {
    memmove(generation->pending_text.buffer,
            generation->pending_text.buffer + generation->text_length,
            remaining);
  }
  generation->pending_text.size = remaining;
  generation->pending_text.buffer[remaining] = 0;
  generation->text_length = 0;
}

iree_string_view_t loom_serve_realtime_generation_audio(
    const loom_serve_realtime_generation_t* generation) {
  return generation->block ? iree_string_view_empty()
                           : iree_string_builder_view(&generation->packet);
}

void loom_serve_realtime_generation_consume_audio(
    loom_serve_realtime_generation_t* generation) {
  iree_string_builder_reset(&generation->packet);
}

uint64_t loom_serve_realtime_generation_frames(
    const loom_serve_realtime_generation_t* generation) {
  return generation->frames;
}

bool loom_serve_realtime_generation_complete(
    const loom_serve_realtime_generation_t* generation) {
  return generation->finished && !generation->text_length &&
         !generation->block && !generation->packet.size &&
         (!generation->speech ||
          loom_serve_speech_stream_is_complete(generation->speech));
}

const loom_serve_text_stream_result_t* loom_serve_realtime_generation_result(
    const loom_serve_realtime_generation_t* generation) {
  return generation->text && generation->finished
             ? loom_serve_text_stream_result(generation->text)
             : NULL;
}

void loom_serve_realtime_generation_cancel(
    loom_serve_realtime_generation_t* generation) {
  generation->cancelled = true;
  iree_string_builder_reset(&generation->pending_text);
  generation->text_length = 0;
  if (generation->text) {
    loom_serve_text_stream_close(generation->text);
    generation->text = NULL;
  }
  if (!generation->block) {
    if (generation->speech) {
      loom_serve_speech_stream_close(generation->speech);
      generation->speech = NULL;
    }
    iree_string_builder_reset(&generation->packet);
  }
}

bool loom_serve_realtime_generation_pending(
    const loom_serve_realtime_generation_t* generation) {
  return generation->block != NULL;
}

void loom_serve_realtime_generation_free(
    loom_serve_realtime_generation_t* generation) {
  if (!generation) {
    return;
  }
  loom_serve_speech_request_deinitialize(&generation->speech_request);
  iree_string_builder_deinitialize(&generation->text_request.body);
  iree_string_builder_deinitialize(&generation->packet);
  iree_string_builder_deinitialize(&generation->pending_text);
  iree_status_ignore(generation->status);
  iree_slim_mutex_deinitialize(&generation->mutex);
  iree_allocator_free_aligned(generation->host_allocator, generation);
}
