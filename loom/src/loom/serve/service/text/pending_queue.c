// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/pending_queue.h"

#include <string.h>

#include "loom/serve/runtime/parameter/admission.h"

iree_status_t loom_serve_text_pending_queue_initialize(
    loom_serve_text_model_t* model, loom_serve_text_telemetry_t* telemetry,
    iree_host_size_t capacity, iree_host_size_t overtake_budget,
    iree_allocator_t host_allocator,
    loom_serve_text_pending_queue_t* out_queue) {
  *out_queue = (loom_serve_text_pending_queue_t){
      .host_allocator = host_allocator,
      .model = model,
      .telemetry = telemetry,
      .capacity = capacity,
      .admission = {.budget = overtake_budget}};
  iree_host_size_t storage_length = 0, slots_offset = 0, values_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_length,
      IREE_STRUCT_FIELD(out_queue->capacity, loom_serve_text_pending_request_t,
                        &slots_offset),
      IREE_STRUCT_FIELD(out_queue->capacity, loom_serve_text_pending_request_t*,
                        &values_offset)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, storage_length,
                                             (void**)&out_queue->storage));
  loom_serve_text_pending_request_t* slots =
      (loom_serve_text_pending_request_t*)(out_queue->storage + slots_offset);
  out_queue->values =
      (loom_serve_text_pending_request_t**)(out_queue->storage + values_offset);
  for (iree_host_size_t i = 0; i < out_queue->capacity; ++i) {
    out_queue->values[i] = &slots[i];
  }
  return iree_ok_status();
}

void loom_serve_text_pending_queue_deinitialize(
    loom_serve_text_pending_queue_t* queue) {
  iree_allocator_free(queue->host_allocator, queue->storage);
  memset(queue, 0, sizeof(*queue));
}

// Both ingress paths resolve names on the owner before accepting input. Source
// JSON validation remains on the finite preparation loan, after acceptance.
iree_status_t loom_serve_text_pending_queue_select_request(
    const loom_serve_text_pending_queue_t* queue, iree_string_view_t name,
    iree_string_view_t checkpoint_name, iree_host_size_t row_count,
    loom_serve_text_session_t* sessions,
    loom_serve_text_checkpoint_registry_t* checkpoints,
    loom_serve_text_checkpoint_entry_t** out_checkpoint) {
  *out_checkpoint = NULL;
  if (!loom_serve_text_session_name_valid(name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "session must be at most 64 ASCII identifier characters");
  }
  if (!loom_serve_text_session_name_valid(checkpoint_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "checkpoint must be at most 64 ASCII identifier characters");
  }
  if (checkpoint_name.size) {
    *out_checkpoint =
        loom_serve_text_checkpoint_registry_find(checkpoints, checkpoint_name);
    if (!*out_checkpoint) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "checkpoint is not pinned");
    }
  }
  const loom_serve_text_session_t* session =
      loom_serve_text_session_select(row_count, sessions, name, NULL);
  bool busy = session && session->request.consumer.self;
  for (iree_host_size_t i = 0; name.size && i < queue->count; ++i) {
    busy |= iree_string_view_equal(name, queue->values[i]->name);
  }
  if (busy) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "session already has an active or queued request");
  }
  return iree_ok_status();
}

void loom_serve_text_pending_queue_push(
    loom_serve_text_pending_queue_t* queue, loom_serve_text_consumer_t consumer,
    iree_string_view_t body, iree_string_view_t name,
    loom_serve_text_checkpoint_entry_t* checkpoint,
    const loom_serve_text_sequence_t* sequence,
    loom_serve_text_receipt_t receipt) {
  if (checkpoint) {
    ++checkpoint->pending;
  }
  loom_serve_text_pending_request_t* pending = queue->values[queue->count++];
  *pending = (loom_serve_text_pending_request_t){
      .consumer = consumer,
      .body = body,
      .name = name,
      .checkpoint = checkpoint,
      .receipt = receipt,
  };
  if (sequence) {
    pending->sequence = *sequence;
    pending->name = iree_make_cstring_view(pending->sequence.handle);
  }
}

// Resolves the exact completed frontier without acquiring native ownership.
loom_serve_text_input_prefix_t loom_serve_text_pending_queue_prefix(
    const loom_serve_text_pending_queue_t* queue,
    const loom_serve_text_session_t* session) {
  const loom_serve_text_pending_request_t* request = queue->prepared.request;
  const loom_serve_text_checkpoint_entry_t* pinned = request->checkpoint;
  const loom_serve_text_session_cache_entry_t* cached = queue->prepared.cached;
  const loom_serve_text_checkpoint_entry_t* endpoint = pinned ? pinned
                                                       : cached
                                                           ? &cached->endpoint
                                                           : NULL;
  if (endpoint) {
    return (loom_serve_text_input_prefix_t){.completion = &endpoint->completion,
                                            .position = endpoint->position,
                                            .token = endpoint->token,
                                            .boundary = endpoint->boundary};
  }
  if ((session->completion.storage ||
       (session->mode == LOOM_SERVE_TEXT_MODE_ENCODE &&
        loom_serve_text_session_has_history(session))) &&
      iree_string_view_equal(request->name,
                             iree_make_cstring_view(session->lifetime.name))) {
    return (loom_serve_text_input_prefix_t){
        .completion = &session->completion,
        .position = loom_serve_text_row_position(session->row),
        .token = loom_serve_text_row_token(session->row),
        .boundary = loom_serve_text_row_is_eos(session->row)
                        ? LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED
                        : LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN};
  }
  return (loom_serve_text_input_prefix_t){0};
}

iree_status_t loom_serve_text_pending_queue_prepare(
    loom_serve_text_pending_queue_t* queue, iree_string_view_t name,
    iree_host_size_t default_max_tokens, iree_host_size_t row_count,
    loom_serve_text_session_t* sessions,
    const loom_serve_text_pool_usage_t* geometry) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, name.data, name.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0,
      loom_serve_telemetry_stream_identity(queue->telemetry->stream).ordinal);
  loom_serve_text_pending_request_t* pending = queue->prepared.request;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, pending->receipt.serial);
  const iree_time_t start = iree_time_now();
  iree_status_t status = iree_ok_status();
  const bool encodes =
      loom_serve_text_model_mode(queue->model) == LOOM_SERVE_TEXT_MODE_ENCODE;
  if (!encodes && !pending->chat.policy) {
    status = loom_serve_text_chat_initialize(
        loom_serve_text_model_chat_policy(queue->model), name, pending->body,
        pending->receipt.serial, default_max_tokens, queue->host_allocator,
        &pending->chat);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_text_session_t* session = loom_serve_text_session_select(
        row_count, sessions, pending->name,
        pending->checkpoint || encodes ? NULL : &pending->chat);
    queue->prepared.resolved_session = session;
    status = loom_serve_text_pending_request_prepare(
        pending, queue->model,
        loom_serve_text_pending_queue_prefix(queue, session), geometry,
        session->tokens, queue->host_allocator);
  }
  pending->prepare_duration += iree_time_now() - start;
  IREE_RETURN_AND_END_ZONE(z0, status);
}

iree_status_t loom_serve_text_pending_queue_admit(
    loom_serve_text_pending_queue_t* queue,
    loom_serve_text_session_cache_t* cache, loom_serve_text_session_t* sessions,
    iree_string_view_t model_name) {
  loom_serve_text_pending_request_t* pending = queue->prepared.request;
  loom_serve_text_session_t* session = queue->prepared.session;
  loom_serve_text_prefix_entry_t* prefix = queue->prepared.prefix;
  const iree_host_size_t input_origin = prefix ? prefix->count : 0;
  const iree_host_size_t input_count = pending->input.count - input_origin;
  const iree_host_size_t retained_count =
      pending->input.retained_count + input_origin;
  const iree_host_size_t reservation = pending->reservation;
  const iree_host_size_t restored_bytes = queue->prepared.restored_bytes;
  loom_serve_text_pending_request_t request = *pending;
  if (session->mode == LOOM_SERVE_TEXT_MODE_ENCODE) {
    request.name = iree_make_cstring_view(request.sequence.handle);
  }
  loom_serve_text_media_request_begin(pending->media);
  const iree_host_size_t queue_rank = queue->admission.cursor;
  loom_serve_text_admission_acquired(&queue->admission);
  const iree_host_size_t overtakes = queue->admission.overtakes;
  memcpy(session->tokens, pending->input.tokens,
         pending->input.count * sizeof(*session->tokens));
  const iree_duration_t prepare_duration = pending->prepare_duration;
  loom_serve_text_session_cache_entry_t* cached = queue->prepared.cached;
  if (cached) {
    loom_serve_text_telemetry_session_end(queue->telemetry, &session->lifetime);
    session->lifetime = cached->lifetime;
    memset(&cached->lifetime, 0, sizeof(cached->lifetime));
    loom_serve_text_session_cache_release(cache, cached, queue->telemetry);
  }
  loom_serve_text_input_deinitialize(&pending->input, queue->host_allocator);
  loom_serve_text_pending_queue_remove(queue, queue->admission.cursor);
  if (!request.name.size ||
      !iree_string_view_equal(request.name,
                              iree_make_cstring_view(session->lifetime.name))) {
    loom_serve_text_telemetry_session_end(queue->telemetry, &session->lifetime);
  }
  if (request.name.size) {
    memcpy(session->lifetime.name, request.name.data, request.name.size);
  }
  session->lifetime.name[request.name.size] = 0;
  session->lifetime.serial = request.receipt.serial;
  if (!session->lifetime.id) {
    session->lifetime.id = session->lifetime.serial;
    loom_serve_text_session_record_t value = {.session = session->lifetime.id,
                                              .row = session - sessions};
    memcpy(value.name, session->lifetime.name, sizeof(value.name));
    loom_serve_text_record(queue->telemetry->stream,
                           &LOOM_SERVE_TEXT_SESSION_BEGIN,
                           session->lifetime.serial, &value, sizeof(value));
  }
  ++session->lifetime.counters.requests;
  session->request.counters.requests = 1;
  session->request.consumer = request.consumer;
  session->request.chat = request.chat;
  session->request.sequence = request.sequence;
  session->request.media = request.media;
  session->request.phase = LOOM_SERVE_TEXT_REQUEST_PREFILL;
  session->request.input_count = input_count;
  session->request.input_origin = input_origin;
  session->request.retained_count = retained_count;
  session->request.start_time = iree_time_now();
  session->request.queue_duration =
      session->request.start_time - request.receipt.arrival_time;
  session->request.reserved_tokens = reservation;
  iree_status_t status = loom_serve_text_row_bind_request(
      session->row, session->mode == LOOM_SERVE_TEXT_MODE_ENCODE
                        ? request.sequence.options
                        : request.chat.options);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_session_begin(
        session, loom_serve_text_model_tokenizer(queue->model), model_name);
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_text_pool_usage_t pool =
        loom_serve_text_model_pool_usage(queue->model);
    const loom_serve_text_admission_record_t value = {
        .session = session->lifetime.id,
        .row = session - sessions,
        .retained_tokens = retained_count,
        .appended_tokens = input_count,
        .reservation_tokens = reservation,
        .restored_bytes = restored_bytes,
        .max_tokens = request.chat.max_tokens,
        .queue_rank = queue_rank,
        .overtakes = overtakes,
        .overtake_budget = queue->admission.budget,
        .prepare_ns = prepare_duration,
        .queue_ns = session->request.queue_duration,
        .pool_reserved_tokens = pool.pending,
        .pool_resident_tokens = pool.capacity - pool.available - pool.pending};
    loom_serve_text_record(queue->telemetry->stream, &LOOM_SERVE_TEXT_ADMIT,
                           session->lifetime.serial, &value, sizeof(value));
  }
  return status;
}

void loom_serve_text_pending_queue_remove(
    loom_serve_text_pending_queue_t* queue, iree_host_size_t index) {
  loom_serve_text_pending_request_t* request = queue->values[index];
  if (request == queue->prepared.request) {
    loom_serve_admission_cancel(
        loom_serve_text_model_residency(queue->model),
        loom_serve_text_model_admission_permit(queue->model));
    memset(&queue->prepared, 0, sizeof(queue->prepared));
  } else if (queue->prepared.request && index < queue->prepared.index) {
    --queue->prepared.index;
  }
  loom_serve_text_admission_remove(&queue->admission, index);
  if (request->checkpoint) {
    --request->checkpoint->pending;
  }
  --queue->count;
  memmove(queue->values + index, queue->values + index + 1,
          (queue->count - index) * sizeof(*queue->values));
  queue->values[queue->count] = request;
}

void loom_serve_text_pending_queue_cancel(
    loom_serve_text_pending_queue_t* queue, iree_host_size_t index,
    loom_serve_text_media_projector_t* projector) {
  loom_serve_text_pending_request_t* request = queue->values[index];
  loom_serve_text_media_projector_cancel(projector, request->media);
  loom_serve_text_pending_request_cancel(request, queue->telemetry,
                                         queue->host_allocator);
  loom_serve_text_pending_queue_remove(queue, index);
}

bool loom_serve_text_pending_queue_cancel_failed(
    loom_serve_text_pending_queue_t* queue,
    const loom_serve_text_pending_request_t* borrowed,
    loom_serve_text_media_projector_t* projector) {
  bool progress = false;
  for (iree_host_size_t i = 0; i < queue->count;) {
    loom_serve_text_pending_request_t* request = queue->values[i];
    if (request != borrowed &&
        !request->consumer.vtable->is_open(request->consumer.self) &&
        loom_serve_text_media_projector_cancel(projector, request->media)) {
      loom_serve_text_pending_queue_cancel(queue, i, projector);
      progress = true;
    } else {
      ++i;
    }
  }
  return progress;
}
