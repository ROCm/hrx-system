// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/voice.h"

#include "iree/base/internal/unicode.h"

iree_status_t loom_serve_speech_voice_request_initialize(
    iree_string_view_t id, iree_string_view_t content_type,
    iree_const_byte_span_t audio, iree_allocator_t host_allocator,
    loom_serve_speech_voice_request_t* out_request) {
  *out_request = (loom_serve_speech_voice_request_t){0};
  if (!id.size || !iree_unicode_utf8_validate(id) ||
      iree_string_view_find_char(id, '\0', 0) != IREE_STRING_VIEW_NPOS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "voice ID must be nonempty UTF-8 without NUL bytes");
  }
  loom_serve_audio_source_t source;
  IREE_RETURN_IF_ERROR(
      loom_serve_audio_source_parse(content_type, audio, &source));
  iree_string_builder_initialize(host_allocator, &out_request->id);
  iree_string_builder_initialize(host_allocator, &out_request->storage);
  iree_status_t status =
      iree_string_builder_append_string(&out_request->id, id);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &out_request->storage,
        iree_make_string_view((const char*)audio.data, audio.data_length));
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t stored =
        iree_string_builder_view(&out_request->storage);
    source.data = iree_make_const_byte_span(stored.data, stored.size);
    out_request->audio = source;
  } else {
    loom_serve_speech_voice_request_deinitialize(out_request);
  }
  return status;
}

void loom_serve_speech_voice_request_deinitialize(
    loom_serve_speech_voice_request_t* request) {
  iree_string_builder_deinitialize(&request->id);
  iree_string_builder_deinitialize(&request->storage);
  *request = (loom_serve_speech_voice_request_t){0};
}

struct loom_serve_speech_voice_upload_t {
  // Allocator owning the independent completion receipt.
  iree_allocator_t host_allocator;
  // Caller ownership, consumed by close on the serialized owner.
  bool caller_open;
  // Service ownership, consumed after all accepted borrowers retire.
  bool service_open;
  // Irreversible withdrawal of publication interest.
  bool cancelled;
  // Terminal result independent of service/model lifetime.
  iree_status_t status;
};

static void loom_serve_speech_voice_upload_release(
    loom_serve_speech_voice_upload_t* upload) {
  if (!upload->caller_open && !upload->service_open) {
    iree_status_free(upload->status);
    iree_allocator_free(upload->host_allocator, upload);
  }
}

static bool loom_serve_speech_voice_upload_open(void* self) {
  const loom_serve_speech_voice_upload_t* upload = self;
  return upload->caller_open && !upload->cancelled;
}

static bool loom_serve_speech_voice_upload_credit(void* self) { return true; }

static iree_status_t loom_serve_speech_voice_upload_encode(
    void* self, const loom_serve_speech_block_t* block) {
  return iree_ok_status();
}

static iree_status_t loom_serve_speech_voice_upload_publish(
    void* self, bool* out_published) {
  *out_published = true;
  return iree_ok_status();
}

static void loom_serve_speech_voice_upload_cancel(void* self) {
  loom_serve_speech_voice_upload_t* upload = self;
  upload->cancelled = true;
}

static iree_status_t loom_serve_speech_voice_upload_end(
    void* self, loom_serve_telemetry_outcome_t outcome,
    const iree_status_t status) {
  loom_serve_speech_voice_upload_t* upload = self;
  upload->service_open = false;
  if (outcome != LOOM_SERVE_TELEMETRY_COMPLETED) {
    upload->status = iree_status_is_ok(status)
                         ? iree_status_from_code(IREE_STATUS_CANCELLED)
                         : iree_status_clone(status);
  }
  loom_serve_speech_voice_upload_release(upload);
  return iree_ok_status();
}

iree_status_t loom_serve_speech_voice_upload_allocate(
    iree_allocator_t host_allocator,
    loom_serve_speech_voice_upload_t** out_upload,
    loom_serve_speech_consumer_t* out_consumer) {
  *out_upload = NULL;
  *out_consumer = (loom_serve_speech_consumer_t){0};
  loom_serve_speech_voice_upload_t* upload = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*upload), (void**)&upload));
  upload->host_allocator = host_allocator;
  upload->caller_open = true;
  upload->service_open = true;
  *out_upload = upload;
  *out_consumer = (loom_serve_speech_consumer_t){
      .self = upload,
      .is_open = loom_serve_speech_voice_upload_open,
      .can_write = loom_serve_speech_voice_upload_credit,
      .encode = loom_serve_speech_voice_upload_encode,
      .publish = loom_serve_speech_voice_upload_publish,
      .cancel = loom_serve_speech_voice_upload_cancel,
      .end = loom_serve_speech_voice_upload_end,
  };
  return iree_ok_status();
}

bool loom_serve_speech_voice_upload_is_complete(
    const loom_serve_speech_voice_upload_t* upload) {
  return !upload->service_open;
}

iree_status_t loom_serve_speech_voice_upload_status(
    const loom_serve_speech_voice_upload_t* upload) {
  return iree_status_clone(upload->status);
}

void loom_serve_speech_voice_upload_close(
    loom_serve_speech_voice_upload_t* upload) {
  upload->caller_open = false;
  loom_serve_speech_voice_upload_release(upload);
}
