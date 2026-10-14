// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_projector.h"

#include <string.h>

#include "loom/serve/media/image_request.h"
#include "loom/serve/runtime/parameter/residency.h"

typedef enum media_projector_phase_e {
  MEDIA_PROJECTOR_IDLE = 0,
  MEDIA_PROJECTOR_COMPONENT,
  MEDIA_PROJECTOR_PREPARING,
  MEDIA_PROJECTOR_WAITING,
  MEDIA_PROJECTOR_RUNNING,
  MEDIA_PROJECTOR_DRAINING,
  MEDIA_PROJECTOR_COMPLETE,
} media_projector_phase_t;

struct loom_serve_text_media_projector_t {
  // Allocator owning this state and every projected result table.
  iree_allocator_t host_allocator;
  // Borrowed deployment component, independent of the selected request.
  loom_serve_finite_component_t* component;
  // Independent CPU/device publication for image work.
  loom_serve_text_readiness_t readiness;
  // Owner-only candidate lifecycle.
  media_projector_phase_t phase;
  // Request cancellation observed while accepted work retires.
  bool cancelled;
  // Borrowed selected request chat through terminal take.
  const loom_serve_text_chat_t* chat;
  // Borrowed ready finite model while the component remains healthy.
  loom_serve_finite_model_t* model;
  // Next image occurrence to decode and project.
  iree_host_size_t current;
  // Owned accumulating exact results and mixed-epoch views.
  loom_serve_text_projected_media_t media;
  // Owned exact image request retained through finite batch preparation.
  iree_vm_buffer_t* request;
  // Owned source-prepared request retained through finite batch preparation.
  loom_serve_finite_prepared_t prepared;
  // Whether model invocation state owns a prepared but unfinished batch.
  bool batch_prepared;
  // Accepted device frontier retained until readiness consumption.
  loom_serve_execution_frontier_t frontier;
  // Owned terminal request diagnostic transferred by take.
  iree_status_t status;
};

static void loom_serve_text_media_projector_release_request(
    loom_serve_text_media_projector_t* projector) {
  iree_vm_buffer_release(projector->prepared.bytes);
  memset(&projector->prepared, 0, sizeof(projector->prepared));
  iree_vm_buffer_release(projector->request);
  projector->request = NULL;
}

static void loom_serve_text_media_projector_finish(
    loom_serve_text_media_projector_t* projector, iree_status_t status) {
  projector->status = status;
  projector->phase = MEDIA_PROJECTOR_COMPLETE;
}

static iree_status_t loom_serve_text_media_projector_prepare(void* user_data) {
  loom_serve_text_media_projector_t* projector = user_data;
  const iree_host_size_t image_ordinal = projector->current;
  const uint32_t image_index = projector->media.image_indices[image_ordinal];
  const loom_serve_text_chat_image_t* image =
      &projector->chat->images.values[image_index];
  const loom_serve_text_media_input_span_t* span =
      &projector->media.spans[image_ordinal];
  const iree_const_byte_span_t metadata = iree_make_const_byte_span(
      span->metadata_length
          ? projector->media.input.metadata.data + span->metadata_offset
          : NULL,
      span->metadata_length);
  IREE_RETURN_IF_ERROR(loom_serve_media_image_request_create(
      image->data_url, &projector->chat->policy->images.limits, &image->info,
      metadata, span->input_count, projector->host_allocator,
      &projector->request));
  IREE_RETURN_IF_ERROR(loom_serve_finite_model_prepare_request(
      projector->model, projector->request, &projector->prepared));
  const iree_host_size_t expected_rows = span->input_count;
  IREE_RETURN_IF_ERROR(loom_serve_finite_model_prepare_batch(
      projector->model, 1, &projector->prepared, &expected_rows));
  projector->batch_prepared = true;
  loom_serve_text_media_projector_release_request(projector);
  return iree_ok_status();
}

static void loom_serve_text_media_projector_schedule(
    loom_serve_text_media_projector_t* projector) {
  projector->phase = MEDIA_PROJECTOR_PREPARING;
  loom_serve_text_readiness_schedule(&projector->readiness,
                                     loom_serve_text_media_projector_prepare,
                                     projector);
}

static void loom_serve_text_media_projector_release_batch(
    loom_serve_text_media_projector_t* projector) {
  if (projector->batch_prepared) {
    loom_serve_residency_cancel_admission(
        loom_serve_finite_model_residency(projector->model));
    loom_serve_finite_model_release_batch(projector->model);
    projector->batch_prepared = false;
  }
  loom_serve_text_media_projector_release_request(projector);
}

static void loom_serve_text_media_projector_fail_model(
    loom_serve_text_media_projector_t* projector, iree_status_t status) {
  projector->status = iree_status_clone(status);
  loom_serve_finite_component_fail(projector->component, status);
  projector->model = NULL;
  projector->phase = MEDIA_PROJECTOR_DRAINING;
}

iree_status_t loom_serve_text_media_projector_allocate(
    iree_task_executor_t* executor, iree_async_proactor_t* proactor,
    loom_serve_finite_component_t* component, iree_allocator_t host_allocator,
    loom_serve_text_media_projector_t** out_projector) {
  *out_projector = NULL;
  loom_serve_text_media_projector_t* projector = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*projector),
                                             (void**)&projector));
  *projector = (loom_serve_text_media_projector_t){
      .host_allocator = host_allocator,
      .component = component,
  };
  loom_serve_text_readiness_initialize(executor, proactor,
                                       &projector->readiness);
  *out_projector = projector;
  return iree_ok_status();
}

void loom_serve_text_media_projector_set_observer(
    loom_serve_text_media_projector_t* projector,
    loom_serve_wake_callback_t observer) {
  if (projector) {
    loom_serve_text_readiness_set_observer(&projector->readiness, observer);
  }
}

iree_status_t loom_serve_text_media_projector_begin(
    loom_serve_text_media_projector_t* projector,
    const loom_serve_text_chat_t* chat, const loom_serve_text_input_t* input,
    iree_host_size_t input_origin) {
  IREE_ASSERT(projector->phase == MEDIA_PROJECTOR_IDLE);
  IREE_RETURN_IF_ERROR(loom_serve_text_projected_media_initialize(
      chat, input, input_origin, projector->host_allocator, &projector->media));
  if (!projector->media.result_count) {
    loom_serve_text_projected_media_deinitialize(&projector->media);
    return iree_ok_status();
  }
  projector->chat = chat;
  projector->phase = MEDIA_PROJECTOR_COMPONENT;
  return iree_ok_status();
}

void loom_serve_text_media_projector_cancel(
    loom_serve_text_media_projector_t* projector) {
  if (projector && projector->phase != MEDIA_PROJECTOR_IDLE) {
    projector->cancelled = true;
  }
}

bool loom_serve_text_media_projector_advance(
    loom_serve_text_media_projector_t* projector, bool shared_activity) {
  if (!projector || projector->phase == MEDIA_PROJECTOR_IDLE ||
      projector->phase == MEDIA_PROJECTOR_COMPLETE) {
    return false;
  }
  bool progress = false;
  for (;;) {
    if (projector->phase == MEDIA_PROJECTOR_COMPONENT) {
      if (projector->cancelled) {
        loom_serve_text_media_projector_finish(projector, iree_ok_status());
        return true;
      }
      iree_status_t status = loom_serve_finite_component_prepare(
          projector->component, &projector->model);
      if (!iree_status_is_ok(status)) {
        loom_serve_text_media_projector_finish(projector, status);
        return true;
      }
      if (!projector->model) {
        return progress;
      }
      loom_serve_text_media_projector_schedule(projector);
      return true;
    }
    if (projector->phase == MEDIA_PROJECTOR_PREPARING) {
      iree_status_t status = iree_ok_status();
      if (!loom_serve_text_readiness_take_cpu(&projector->readiness, &status)) {
        return progress;
      }
      progress = true;
      if (!iree_status_is_ok(status)) {
        loom_serve_text_media_projector_release_batch(projector);
        loom_serve_text_media_projector_finish(projector, status);
        return true;
      }
      if (projector->cancelled) {
        loom_serve_text_media_projector_release_batch(projector);
        loom_serve_text_media_projector_finish(projector, iree_ok_status());
        return true;
      }
      projector->phase = MEDIA_PROJECTOR_WAITING;
      continue;
    }
    if (projector->phase == MEDIA_PROJECTOR_WAITING) {
      if (projector->cancelled) {
        loom_serve_text_media_projector_release_batch(projector);
        loom_serve_text_media_projector_finish(projector, iree_ok_status());
        return true;
      }
      loom_serve_residency_admission_t admission;
      iree_status_t status = loom_serve_residency_try_admit(
          loom_serve_finite_model_residency(projector->model), &admission);
      if (!iree_status_is_ok(status)) {
        loom_serve_text_media_projector_fail_model(projector, status);
        return true;
      }
      if (admission == LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY &&
          !shared_activity) {
        loom_serve_text_media_projector_release_batch(projector);
        loom_serve_text_media_projector_finish(
            projector,
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "image component parameters exceed available "
                             "capacity"));
        return true;
      }
      if (admission != LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
        return progress;
      }
      status = loom_serve_finite_model_submit(projector->model,
                                              &projector->frontier);
      if (iree_status_is_ok(status)) {
        status = loom_serve_text_readiness_await(&projector->readiness,
                                                 &projector->frontier);
      }
      if (!iree_status_is_ok(status)) {
        loom_serve_text_media_projector_fail_model(projector, status);
        return true;
      }
      projector->phase = MEDIA_PROJECTOR_RUNNING;
      return true;
    }
    if (projector->phase == MEDIA_PROJECTOR_RUNNING) {
      iree_status_t status = iree_ok_status();
      if (!loom_serve_text_readiness_take_device(&projector->readiness,
                                                 &status)) {
        return progress;
      }
      progress = true;
      if (!iree_status_is_ok(status)) {
        loom_serve_text_media_projector_fail_model(projector, status);
        return true;
      }
      loom_serve_finite_model_complete(projector->model);
      loom_serve_finite_result_t result = {0};
      loom_serve_finite_model_take_result(projector->model, &result);
      projector->batch_prepared = false;
      loom_serve_text_media_projector_release_request(projector);
      loom_serve_text_projected_media_set_result(&projector->media,
                                                 projector->current, &result);
      ++projector->current;
      if (projector->cancelled ||
          projector->current == projector->media.result_count) {
        loom_serve_text_media_projector_finish(projector, iree_ok_status());
        return true;
      }
      loom_serve_text_media_projector_schedule(projector);
      return true;
    }
    if (projector->phase == MEDIA_PROJECTOR_DRAINING) {
      if (loom_serve_finite_component_phase(projector->component) !=
          LOOM_SERVE_FINITE_COMPONENT_FAILED) {
        return progress;
      }
      projector->batch_prepared = false;
      loom_serve_text_media_projector_release_request(projector);
      projector->phase = MEDIA_PROJECTOR_COMPLETE;
      return true;
    }
  }
}

bool loom_serve_text_media_projector_is_active(
    const loom_serve_text_media_projector_t* projector) {
  return projector && projector->phase != MEDIA_PROJECTOR_IDLE;
}

bool loom_serve_text_media_projector_is_pending(
    const loom_serve_text_media_projector_t* projector) {
  return projector &&
         (projector->phase == MEDIA_PROJECTOR_PREPARING ||
          projector->phase == MEDIA_PROJECTOR_RUNNING ||
          projector->phase == MEDIA_PROJECTOR_DRAINING ||
          (projector->phase == MEDIA_PROJECTOR_COMPONENT &&
           loom_serve_finite_component_is_pending(projector->component)));
}

bool loom_serve_text_media_projector_is_complete(
    const loom_serve_text_media_projector_t* projector) {
  return projector && projector->phase == MEDIA_PROJECTOR_COMPLETE;
}

iree_status_t loom_serve_text_media_projector_take(
    loom_serve_text_media_projector_t* projector, bool* out_cancelled,
    loom_serve_text_projected_media_t* out_media) {
  IREE_ASSERT(projector->phase == MEDIA_PROJECTOR_COMPLETE);
  *out_cancelled = projector->cancelled;
  memset(out_media, 0, sizeof(*out_media));
  iree_status_t status = projector->status;
  projector->status = iree_ok_status();
  if (iree_status_is_ok(status) && !projector->cancelled) {
    *out_media = projector->media;
    memset(&projector->media, 0, sizeof(projector->media));
  } else {
    loom_serve_text_projected_media_deinitialize(&projector->media);
  }
  projector->phase = MEDIA_PROJECTOR_IDLE;
  projector->cancelled = false;
  projector->chat = NULL;
  projector->model = NULL;
  projector->current = 0;
  projector->frontier = (loom_serve_execution_frontier_t){0};
  return status;
}

void loom_serve_text_media_projector_free(
    loom_serve_text_media_projector_t* projector) {
  if (!projector) {
    return;
  }
  IREE_ASSERT(projector->phase == MEDIA_PROJECTOR_IDLE);
  loom_serve_text_readiness_deinitialize(&projector->readiness);
  iree_allocator_free(projector->host_allocator, projector);
}
