// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/voice_load.h"

#include "iree/base/threading/mutex.h"
#include "loom/model/card.h"
#include "loomc/iree.h"

typedef enum loom_serve_speech_voice_load_phase_e {
  SPEECH_VOICE_LOAD_READ,
  SPEECH_VOICE_LOAD_READING,
  SPEECH_VOICE_LOAD_UPLOAD,
} loom_serve_speech_voice_load_phase_t;

struct loom_serve_speech_voice_load_t {
  // One reusable read task, reinitialized only after its final callback joins.
  loomc_task_t task;
  // Allocator owning the loader and original recording request.
  iree_allocator_t host_allocator;
  // Borrowed shared host, alive through completion.
  loom_serve_host_t* host;
  // Owned immutable namespace retaining card and package references.
  loom_serve_sources_t* sources;
  // Borrowed service used for ordinary native upload admission.
  loom_serve_speech_service_t* service;
  // Service advancement and readiness callbacks borrowed through completion.
  loom_serve_service_t interface;
  // Nonblocking owner notification, also installed on the service.
  loom_serve_wake_callback_t observer;
  // Next declared recording; unchanged while its read task is accepted.
  iree_host_size_t index;
  // Owner-only next read/admission phase.
  loom_serve_speech_voice_load_phase_t phase;
  // Owned original bytes, moved into the service at successful admission.
  loom_serve_speech_voice_request_t request;
  // Owned receipt retained until publication or cancellation.
  loom_serve_speech_voice_upload_t* upload;
  // Terminal failure, retained while accepted ownership retires.
  iree_status_t status;
  // Read result published only by the task's final release callback.
  struct {
    // Joins the final callback with owner advancement and destruction.
    iree_slim_mutex_t mutex;
    // The task still owns borrowed loader/source storage.
    bool pending;
    // Worker outcome transferred to the owner after pending becomes false.
    iree_status_t status;
  } event;
};

static void loom_serve_speech_voice_load_execute(
    loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  loom_serve_speech_voice_load_t* load = (void*)task;
  const loom_model_selection_t* selection =
      loom_model_card_selection(load->sources->card);
  const loom_model_voice_t* voice = &selection->voices[load->index];
  loomc_source_t* contents = NULL;
  iree_status_t status = loom_serve_sources_open(
      load->sources, voice->file.path, load->host_allocator, &contents);
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t bytes = loomc_source_contents(contents);
    status = loom_serve_speech_voice_request_initialize(
        voice->name, voice->content_type,
        iree_make_const_byte_span(bytes.data, bytes.data_length),
        load->host_allocator, &load->request);
  }
  loomc_source_release(contents);
  load->event.status = status;
}

static void loom_serve_speech_voice_load_release(loomc_task_t* task) {
  loom_serve_speech_voice_load_t* load = (void*)task;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.pending = false;
  load->observer.wake(load->observer.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static const loomc_task_vtable_t voice_load_vtable = {
    .execute = loom_serve_speech_voice_load_execute,
    .destroy = loom_serve_speech_voice_load_release,
};

iree_status_t loom_serve_speech_voice_load_allocate(
    loom_serve_host_t* host, const loom_serve_sources_t* sources,
    loom_serve_speech_service_t* service, loom_serve_wake_callback_t observer,
    iree_allocator_t host_allocator,
    loom_serve_speech_voice_load_t** out_load) {
  *out_load = NULL;
  if (!sources->card ||
      !loom_model_card_selection(sources->card)->voice_count) {
    return iree_ok_status();
  }
  loom_serve_speech_voice_load_t* load = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*load), (void**)&load));
  load->host_allocator = host_allocator;
  load->host = host;
  load->service = service;
  load->interface = loom_serve_speech_service_interface(service);
  load->observer = observer;
  iree_slim_mutex_initialize(&load->event.mutex);
  iree_status_t status =
      loom_serve_sources_clone(sources, host_allocator, &load->sources);
  if (iree_status_is_ok(status)) {
    load->interface.set_observer(load->interface.self, observer);
    *out_load = load;
  } else {
    loom_serve_speech_voice_load_free(load);
  }
  return status;
}

// The final task callback establishes all source/request/status visibility.
static bool loom_serve_speech_voice_load_take_read(
    loom_serve_speech_voice_load_t* load) {
  iree_slim_mutex_lock(&load->event.mutex);
  const bool pending = load->event.pending;
  if (!pending) {
    load->status = iree_status_join(load->status, load->event.status);
    load->event.status = iree_ok_status();
    load->phase = SPEECH_VOICE_LOAD_UPLOAD;
  }
  iree_slim_mutex_unlock(&load->event.mutex);
  return !pending;
}

iree_status_t loom_serve_speech_voice_load_advance(
    loom_serve_speech_voice_load_t* load, bool cancel, bool* out_complete) {
  *out_complete = load == NULL;
  if (!load) {
    return iree_ok_status();
  }
  if (cancel && iree_status_is_ok(load->status)) {
    load->status =
        iree_make_status(IREE_STATUS_CANCELLED, "voice registration cancelled");
  }
  if (load->phase == SPEECH_VOICE_LOAD_READING) {
    if (!loom_serve_speech_voice_load_take_read(load)) {
      return iree_ok_status();
    }
  }
  if (iree_status_is_ok(load->status)) {
    const loom_model_selection_t* selection =
        loom_model_card_selection(load->sources->card);
    if (load->phase == SPEECH_VOICE_LOAD_READ) {
      loomc_task_initialize(&voice_load_vtable, &load->task);
      load->phase = SPEECH_VOICE_LOAD_READING;
      iree_slim_mutex_lock(&load->event.mutex);
      load->event.pending = true;
      iree_slim_mutex_unlock(&load->event.mutex);
      load->status = iree_status_from_loomc(loomc_task_sink_submit(
          loom_serve_host_preparation_sink(load->host), &load->task));
      if (iree_status_is_ok(load->status)) {
        return iree_ok_status();
      }
      load->phase = SPEECH_VOICE_LOAD_UPLOAD;
      load->event.pending = false;
    } else {
      if (!load->upload) {
        load->status = loom_serve_speech_service_try_upload_voice(
            load->service, &load->request, &load->upload);
      }
      if (iree_status_is_ok(load->status)) {
        bool progress = false;
        load->status =
            load->interface.advance(load->interface.self, 0, &progress);
        if (load->upload &&
            loom_serve_speech_voice_upload_is_complete(load->upload)) {
          load->status = iree_status_join(
              load->status,
              loom_serve_speech_voice_upload_status(load->upload));
          loom_serve_speech_voice_upload_close(load->upload);
          load->upload = NULL;
          ++load->index;
          load->phase = SPEECH_VOICE_LOAD_READ;
          progress = true;
          if (iree_status_is_ok(load->status) &&
              load->index == selection->voice_count) {
            *out_complete = true;
            return iree_ok_status();
          }
        }
        if (progress) {
          load->observer.wake(load->observer.user_data);
        }
      }
    }
  }
  if (!iree_status_is_ok(load->status)) {
    if (load->upload) {
      loom_serve_speech_voice_upload_close(load->upload);
      load->upload = NULL;
    }
    bool progress = false;
    load->status = iree_status_join(
        load->status,
        load->interface.advance(load->interface.self,
                                LOOM_SERVE_SERVICE_FLAG_STOPPING, &progress));
    if (progress) {
      load->observer.wake(load->observer.user_data);
    }
    *out_complete = !load->interface.is_pending(load->interface.self);
    if (*out_complete) {
      return iree_status_clone(load->status);
    }
  }
  return iree_ok_status();
}

void loom_serve_speech_voice_load_free(loom_serve_speech_voice_load_t* load) {
  if (!load) {
    return;
  }
  loom_serve_speech_voice_request_deinitialize(&load->request);
  loom_serve_sources_free(load->sources, load->host_allocator);
  iree_status_ignore(load->status);
  iree_slim_mutex_deinitialize(&load->event.mutex);
  iree_allocator_free(load->host_allocator, load);
}
