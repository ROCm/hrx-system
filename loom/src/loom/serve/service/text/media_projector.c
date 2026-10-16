// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_projector.h"

#include <string.h>

#include "loom/serve/media/image_request.h"

typedef struct loom_serve_text_media_image_t {
  // Owned cohort reference, independent of producer parameter residency.
  loom_serve_finite_result_t* result;
  // Original image ordinal in that cohort, including cancelled peers.
  uint32_t result_index;
  // A preparation-window job owns this occurrence until publication/removal.
  bool selected;
} loom_serve_text_media_image_t;

struct loom_serve_text_media_request_t {
  // Allocator for this single request allocation.
  iree_allocator_t host_allocator;
  // Borrowed immutable pending chat, accessed only before language admission.
  const loom_serve_text_chat_t* chat;
  // One owned result reference per original image occurrence.
  loom_serve_text_media_image_t* images;
  // Sorted source input spans rebuilt when retained input coordinates change.
  loom_serve_text_media_input_span_t* spans;
  // Number of allocated image/coordinate records.
  iree_host_size_t count;
  // First image still needed by prefill. Checked source spans name a contiguous
  // suffix of chat occurrences; no parallel image-index array is necessary.
  iree_host_size_t first_image;
  // Current immutable view borrowed by epoch submission.
  loom_serve_text_media_input_t input;
  // Owned request rejection, transferred to the HTTP response owner.
  iree_status_t status;
};

iree_status_t loom_serve_text_media_request_allocate(
    const loom_serve_text_chat_t* chat, iree_allocator_t host_allocator,
    loom_serve_text_media_request_t** out_request) {
  *out_request = NULL;
  iree_host_size_t length = 0, images_offset = 0, spans_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_media_request_t), &length,
      IREE_STRUCT_FIELD(chat->images.count, loom_serve_text_media_image_t,
                        &images_offset),
      IREE_STRUCT_FIELD(chat->images.count, loom_serve_text_media_input_span_t,
                        &spans_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, length, (void**)&storage));
  loom_serve_text_media_request_t* request = (void*)storage;
  *request = (loom_serve_text_media_request_t){
      .host_allocator = host_allocator,
      .chat = chat,
      .images = (void*)(storage + images_offset),
      .spans = (void*)(storage + spans_offset),
      .count = chat->images.count,
  };
  *out_request = request;
  return iree_ok_status();
}

bool loom_serve_text_media_request_bind(
    loom_serve_text_media_request_t* request,
    const loom_serve_text_input_t* input, iree_host_size_t consumed) {
  request->first_image = request->count;
  request->input = (loom_serve_text_media_input_t){.spans = request->spans};
  const iree_string_view_t metadata =
      iree_string_builder_view(&request->chat->images.metadata);
  bool ready = true;
  for (iree_host_size_t i = 0; i < input->media_span_count; ++i) {
    const loom_serve_text_media_span_t* span = &input->media_spans[i];
    if ((iree_host_size_t)span->input_offset + span->input_count <= consumed) {
      continue;
    }
    const uint32_t index = span->image_index;
    const loom_serve_text_chat_image_t* image =
        &request->chat->images.values[index];
    const loom_serve_text_media_image_t* projected = &request->images[index];
    const iree_host_size_t position = request->input.span_count++;
    if (!position) {
      request->first_image = index;
    }
    request->spans[position] = (loom_serve_text_media_input_span_t){
        .input_offset = span->input_offset,
        .input_count = span->input_count,
        .request_index = projected->result_index,
        .result = projected->result,
        .metadata = iree_make_const_byte_span(
            metadata.data + image->metadata_offset, image->metadata_length),
    };
    ready &= projected->result != NULL;
  }
  return ready;
}

void loom_serve_text_media_request_begin(
    loom_serve_text_media_request_t* request) {
  if (!request) {
    return;
  }
  for (iree_host_size_t i = 0; i < request->first_image; ++i) {
    loom_serve_finite_result_release(request->images[i].result);
    request->images[i].result = NULL;
  }
  request->chat = NULL;
}

iree_status_t loom_serve_text_media_request_take_status(
    loom_serve_text_media_request_t* request) {
  if (!request) {
    return iree_ok_status();
  }
  iree_status_t status = request->status;
  request->status = iree_ok_status();
  return status;
}

const loom_serve_text_media_input_t* loom_serve_text_media_request_input(
    const loom_serve_text_media_request_t* request) {
  return request ? &request->input : NULL;
}

void loom_serve_text_media_request_consume(
    loom_serve_text_media_request_t* request, iree_host_size_t consumed) {
  if (!request) {
    return;
  }
  while (request->input.span_count) {
    const loom_serve_text_media_input_span_t* span = request->input.spans;
    if ((iree_host_size_t)span->input_offset + span->input_count > consumed) {
      break;
    }
    loom_serve_text_media_image_t* image =
        &request->images[request->first_image++];
    loom_serve_finite_result_release(image->result);
    image->result = NULL;
    ++request->input.spans;
    --request->input.span_count;
  }
}

void loom_serve_text_media_request_free(
    loom_serve_text_media_request_t* request) {
  if (!request) {
    return;
  }
  for (iree_host_size_t i = 0; i < request->count; ++i) {
    loom_serve_finite_result_release(request->images[i].result);
  }
  iree_allocator_free(request->host_allocator, request);
}

typedef struct loom_serve_text_media_job_t {
  // Borrowed request until publication/cancellation; CPU decode borrows chat.
  loom_serve_text_media_request_t* request;
  // Original image occurrence, independent of input-coordinate changes.
  uint32_t image_index;
  // Expected source feature count, captured before request detachment.
  iree_host_size_t rows;
  // Compressed input retained only during realization.
  loom_serve_media_image_input_t input;
  // Owned decoded packet retained through producer completion.
  iree_vm_buffer_t* packet;
  // Owned source input, possibly aliasing packet.
  loom_serve_finite_prepared_t prepared;
  // CPU-produced per-image status, consumed after the CPU borrow retires.
  iree_status_t status;
} loom_serve_text_media_job_t;

typedef enum loom_serve_text_media_phase_e {
  MEDIA_GATHER = 0,
  MEDIA_PREPARING,
  MEDIA_PREPARED,
  MEDIA_BATCHING,
  MEDIA_SPECIALIZING,
  MEDIA_WAITING,
  MEDIA_RUNNING,
  MEDIA_FAILED,
} loom_serve_text_media_phase_t;

struct loom_serve_text_media_projector_t {
  // Allocator owning this single bounded window allocation.
  iree_allocator_t host_allocator;
  // Borrowed component model; its owner excludes trim during these loans.
  loom_serve_finite_model_t* model;
  // Independent publication channels, never serialized behind text readiness.
  loom_serve_text_readiness_t readiness;
  // Exclusive producer/VM phase.
  loom_serve_text_media_phase_t phase;
  // Maximum image occurrences in one host preparation window.
  iree_host_size_t capacity;
  // Exact decoded packet bytes selected into the gathering window.
  iree_host_size_t packet_bytes;
  // Populated window extent, retaining holes until the window drains.
  iree_host_size_t count;
  // Stable image records across source preparation and cohort execution.
  loom_serve_text_media_job_t* jobs;
  // Selected compatible cohort, or valid codec inputs during CPU preparation.
  loom_serve_text_media_job_t** cohort;
  // Number of selected source requests.
  iree_host_size_t cohort_count;
  // Contiguous codec descriptors borrowing the selected jobs.
  loom_serve_media_image_request_t* requests;
  // Codec result buffers transferred into jobs after realization.
  iree_vm_buffer_t** packets;
  // Per-image realization results transferred into jobs.
  iree_status_t* statuses;
  // Contiguous borrowed source arguments for the selected device cohort.
  loom_serve_finite_prepared_t* prepared;
  // Expected feature counts for the selected device cohort.
  iree_host_size_t* rows;
};

iree_status_t loom_serve_text_media_projector_allocate(
    loom_serve_finite_model_t* model, iree_host_size_t capacity,
    iree_task_executor_t* executor, iree_async_proactor_t* proactor,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_text_media_projector_t** out_projector) {
  *out_projector = NULL;
  capacity = (iree_host_size_t)iree_min(
      capacity, loom_serve_finite_model_batch_limit(model));
  capacity = iree_min(capacity, UINT32_MAX);
  iree_host_size_t length = 0, jobs_offset = 0, cohort_offset = 0,
                   requests_offset = 0, packets_offset = 0, statuses_offset = 0,
                   prepared_offset = 0, rows_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_media_projector_t), &length,
      IREE_STRUCT_FIELD(capacity, loom_serve_text_media_job_t, &jobs_offset),
      IREE_STRUCT_FIELD(capacity, loom_serve_text_media_job_t*, &cohort_offset),
      IREE_STRUCT_FIELD(capacity, loom_serve_media_image_request_t,
                        &requests_offset),
      IREE_STRUCT_FIELD(capacity, iree_vm_buffer_t*, &packets_offset),
      IREE_STRUCT_FIELD(capacity, iree_status_t, &statuses_offset),
      IREE_STRUCT_FIELD(capacity, loom_serve_finite_prepared_t,
                        &prepared_offset),
      IREE_STRUCT_FIELD(capacity, iree_host_size_t, &rows_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, length, iree_alignof(loom_serve_text_media_projector_t),
      0, (void**)&storage));
  loom_serve_text_media_projector_t* projector = (void*)storage;
  *projector = (loom_serve_text_media_projector_t){
      .host_allocator = host_allocator,
      .model = model,
      .capacity = capacity,
      .jobs = (void*)(storage + jobs_offset),
      .cohort = (void*)(storage + cohort_offset),
      .requests = (void*)(storage + requests_offset),
      .packets = (void*)(storage + packets_offset),
      .statuses = (void*)(storage + statuses_offset),
      .prepared = (void*)(storage + prepared_offset),
      .rows = (void*)(storage + rows_offset),
  };
  loom_serve_text_readiness_initialize(executor, proactor,
                                       &projector->readiness);
  loom_serve_text_readiness_set_observer(&projector->readiness, observer);
  loom_serve_finite_model_set_observer(model, observer);
  *out_projector = projector;
  return iree_ok_status();
}

bool loom_serve_text_media_projector_offer(
    loom_serve_text_media_projector_t* projector,
    loom_serve_text_media_request_t* request) {
  if (projector->phase != MEDIA_GATHER) {
    return false;
  }
  if (!iree_status_is_ok(request->status)) {
    return true;
  }
  const iree_host_size_t budget =
      loom_serve_finite_model_input_capacity(projector->model);
  for (iree_host_size_t i = 0; i < request->input.span_count; ++i) {
    const uint32_t index = (uint32_t)(request->first_image + i);
    loom_serve_text_media_image_t* projected = &request->images[index];
    if (projected->result || projected->selected) {
      continue;
    }
    const loom_serve_text_chat_image_t* image =
        &request->chat->images.values[index];
    iree_host_size_t length = 0;
    iree_status_t status = loom_serve_media_image_request_size(
        &image->info, image->metadata_length, &length);
    if (!iree_status_is_ok(status)) {
      request->status = iree_status_join(request->status, status);
      return true;
    }
    if (projector->count == projector->capacity ||
        (projector->count && (projector->packet_bytes >= budget ||
                              length > budget - projector->packet_bytes))) {
      return false;
    }
    projector->jobs[projector->count++] = (loom_serve_text_media_job_t){
        .request = request,
        .image_index = index,
        .rows = image->feature_count,
    };
    projector->packet_bytes += length;
    projected->selected = true;
  }
  return true;
}

static iree_status_t loom_serve_text_media_projector_prepare(void* user_data) {
  loom_serve_text_media_projector_t* projector = user_data;
  iree_host_size_t valid_count = 0;
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    loom_serve_text_media_job_t* job = &projector->jobs[i];
    if (!job->request) {
      continue;
    }
    const loom_serve_text_chat_t* chat = job->request->chat;
    const loom_serve_text_chat_image_t* image =
        &chat->images.values[job->image_index];
    job->status = loom_serve_media_image_input_initialize(
        image->data_url, &chat->policy->images.limits,
        projector->host_allocator, &job->input);
    if (!iree_status_is_ok(job->status)) {
      continue;
    }
    const iree_string_view_t metadata =
        iree_string_builder_view(&chat->images.metadata);
    projector->cohort[valid_count] = job;
    projector->requests[valid_count++] = (loom_serve_media_image_request_t){
        .image = &job->input,
        .metadata = iree_make_const_byte_span(
            metadata.data + image->metadata_offset, image->metadata_length),
        .output_rows = image->feature_count,
    };
  }
  iree_status_t status = loom_serve_media_image_request_create_batch(
      valid_count, projector->requests, projector->host_allocator,
      projector->packets, projector->statuses);
  for (iree_host_size_t i = 0; i < valid_count; ++i) {
    loom_serve_text_media_job_t* job = projector->cohort[i];
    job->packet = projector->packets[i];
    job->status = projector->statuses[i];
    loom_serve_media_image_input_deinitialize(&job->input);
    if (iree_status_is_ok(status) && iree_status_is_ok(job->status)) {
      job->status = loom_serve_finite_model_prepare_request(
          projector->model, job->packet, &job->prepared);
    }
  }
  return status;
}

bool loom_serve_text_media_projector_start(
    loom_serve_text_media_projector_t* projector) {
  if (!projector || projector->phase != MEDIA_GATHER || !projector->count) {
    return false;
  }
  projector->phase = MEDIA_PREPARING;
  loom_serve_text_readiness_schedule(&projector->readiness,
                                     loom_serve_text_media_projector_prepare,
                                     projector);
  return true;
}

bool loom_serve_text_media_projector_is_collecting(
    const loom_serve_text_media_projector_t* projector) {
  return !projector || projector->phase == MEDIA_GATHER;
}

bool loom_serve_text_media_projector_cancel(
    loom_serve_text_media_projector_t* projector,
    loom_serve_text_media_request_t* request) {
  if (!projector || !request) {
    return true;
  }
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    loom_serve_text_media_job_t* job = &projector->jobs[i];
    if (job->request != request) {
      continue;
    }
    if (projector->phase == MEDIA_PREPARING) {
      return false;
    }
    request->images[job->image_index].selected = false;
    request->status = iree_status_join(request->status, job->status);
    job->status = iree_ok_status();
    job->request = NULL;
  }
  return true;
}

bool loom_serve_text_media_projector_borrows(
    const loom_serve_text_media_projector_t* projector,
    const loom_serve_text_media_request_t* request) {
  if (!projector || !request || projector->phase != MEDIA_PREPARING) {
    return false;
  }
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    if (projector->jobs[i].request == request) {
      return true;
    }
  }
  return false;
}

static void loom_serve_text_media_job_release(
    loom_serve_text_media_job_t* job) {
  if (job->request) {
    job->request->images[job->image_index].selected = false;
    job->request->status = iree_status_join(job->request->status, job->status);
  }
  iree_vm_buffer_release(job->prepared.bytes);
  iree_vm_buffer_release(job->packet);
  loom_serve_media_image_input_deinitialize(&job->input);
  memset(job, 0, sizeof(*job));
}

static iree_status_t loom_serve_text_media_projector_batch(void* user_data) {
  loom_serve_text_media_projector_t* projector = user_data;
  return loom_serve_finite_model_prepare_batch(
      projector->model, projector->cohort_count, projector->prepared,
      projector->rows);
}

static void loom_serve_text_media_projector_select(
    loom_serve_text_media_projector_t* projector) {
  uint64_t remaining = loom_serve_finite_model_batch_limit(projector->model);
  uint64_t tag = 0;
  projector->cohort_count = 0;
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    loom_serve_text_media_job_t* job = &projector->jobs[i];
    if (!job->request || !iree_status_is_ok(job->request->status)) {
      loom_serve_text_media_job_release(job);
      continue;
    }
    if (!projector->cohort_count) {
      tag = job->prepared.tag;
    }
    if (job->prepared.tag != tag || job->prepared.units > remaining) {
      continue;
    }
    remaining -= job->prepared.units;
    const iree_host_size_t index = projector->cohort_count++;
    projector->cohort[index] = job;
    projector->prepared[index] = job->prepared;
    projector->rows[index] = job->rows;
  }
  if (projector->cohort_count) {
    projector->phase = MEDIA_BATCHING;
    loom_serve_text_readiness_schedule(&projector->readiness,
                                       loom_serve_text_media_projector_batch,
                                       projector);
  } else {
    projector->count = 0;
    projector->packet_bytes = 0;
    projector->phase = MEDIA_GATHER;
  }
}

static void loom_serve_text_media_projector_complete(
    loom_serve_text_media_projector_t* projector) {
  for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
    loom_serve_text_media_job_release(projector->cohort[i]);
  }
  projector->cohort_count = 0;
  projector->phase = MEDIA_PREPARED;
}

static iree_status_t loom_serve_text_media_projector_submit(
    loom_serve_text_media_projector_t* projector) {
  loom_serve_finite_result_t* result = NULL;
  loom_serve_execution_frontier_t ready;
  IREE_RETURN_IF_ERROR(
      loom_serve_finite_model_submit(projector->model, &result, &ready));
  for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
    loom_serve_text_media_job_t* job = projector->cohort[i];
    if (!job->request) {
      continue;
    }
    loom_serve_text_media_image_t* image =
        &job->request->images[job->image_index];
    loom_serve_finite_result_retain(result);
    image->result = result;
    image->result_index = (uint32_t)i;
    image->selected = false;
    job->request = NULL;
  }
  loom_serve_finite_result_release(result);
  IREE_RETURN_IF_ERROR(
      loom_serve_text_readiness_await(&projector->readiness, &ready));
  projector->phase = MEDIA_RUNNING;
  return iree_ok_status();
}

bool loom_serve_text_media_projector_is_pending(
    const loom_serve_text_media_projector_t* projector) {
  return projector && (projector->phase == MEDIA_PREPARING ||
                       projector->phase == MEDIA_BATCHING ||
                       projector->phase == MEDIA_SPECIALIZING ||
                       projector->phase == MEDIA_RUNNING);
}

void loom_serve_text_media_projector_set_observer(
    loom_serve_text_media_projector_t* projector,
    loom_serve_wake_callback_t observer) {
  if (projector) {
    loom_serve_text_readiness_set_observer(&projector->readiness, observer);
    loom_serve_finite_model_set_observer(projector->model, observer);
  }
}

const char* loom_serve_text_media_projector_phase(
    const loom_serve_text_media_projector_t* projector) {
  if (!projector || !projector->count) {
    return NULL;
  }
  static const char* const phases[] = {
      "image_gather",     "image_prepare", "image_prepared", "image_batch",
      "image_specialize", "image_wait",    "image_execute",  "image_failed",
  };
  return phases[projector->phase];
}

// Request-owned input/source preparation failures do not poison the language
// model. A failure with no remaining requester still propagates to the owner.
static iree_status_t loom_serve_text_media_projector_reject(
    iree_host_size_t count, loom_serve_text_media_job_t* const* jobs,
    iree_status_t status) {
  loom_serve_text_media_request_t* first = NULL;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_text_media_request_t* request = jobs[i]->request;
    if (!request || request == first) {
      continue;
    }
    if (!first) {
      first = request;
    } else if (iree_status_is_ok(request->status)) {
      request->status = iree_status_clone(status);
    }
  }
  if (!first) {
    return status;
  }
  first->status = iree_status_join(first->status, status);
  return iree_ok_status();
}

iree_status_t loom_serve_text_media_projector_advance(
    loom_serve_text_media_projector_t* projector,
    loom_serve_service_flags_t flags, bool* out_progress) {
  *out_progress = false;
  if (!projector || projector->phase == MEDIA_FAILED) {
    return iree_ok_status();
  }
  const bool stopping =
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING);
  iree_status_t status = iree_ok_status();
  if (projector->phase == MEDIA_PREPARING ||
      projector->phase == MEDIA_BATCHING) {
    if (!loom_serve_text_readiness_take_cpu(&projector->readiness, &status)) {
      return status;
    }
    *out_progress = true;
    if (projector->phase == MEDIA_PREPARING) {
      if (!iree_status_is_ok(status)) {
        for (iree_host_size_t i = 0; i < projector->count; ++i) {
          projector->cohort[i] = &projector->jobs[i];
        }
        status = loom_serve_text_media_projector_reject(
            projector->count, projector->cohort, status);
      }
      // Establish each request's terminal input outcome before selecting any
      // of its sibling images into a device cohort.
      for (iree_host_size_t i = 0; i < projector->count; ++i) {
        if (!iree_status_is_ok(projector->jobs[i].status)) {
          loom_serve_text_media_job_release(&projector->jobs[i]);
        }
      }
      projector->phase = MEDIA_PREPARED;
    } else if (!iree_status_is_ok(status)) {
      status = loom_serve_text_media_projector_reject(
          projector->cohort_count, projector->cohort, status);
      loom_serve_text_media_projector_complete(projector);
    } else {
      projector->phase = MEDIA_SPECIALIZING;
    }
  } else if (projector->phase == MEDIA_RUNNING) {
    if (!loom_serve_text_readiness_take_device(&projector->readiness,
                                               &status)) {
      return status;
    }
    *out_progress = true;
    if (iree_status_is_ok(status)) {
      loom_serve_finite_model_complete(projector->model);
      loom_serve_text_media_projector_complete(projector);
    }
  }
  if (!iree_status_is_ok(status)) {
    projector->phase = MEDIA_FAILED;
    return status;
  }
  if (projector->phase == MEDIA_SPECIALIZING) {
    bool live = false;
    for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
      live |= projector->cohort[i]->request != NULL;
    }
    const bool cancel = stopping || !live;
    bool complete = false;
    status = loom_serve_finite_model_advance_batch(projector->model, cancel,
                                                   &complete);
    if (!iree_status_is_ok(status)) {
      loom_serve_finite_model_release_batch(projector->model);
      status = loom_serve_text_media_projector_reject(
          projector->cohort_count, projector->cohort, status);
      loom_serve_text_media_projector_complete(projector);
    } else if (complete) {
      if (cancel) {
        loom_serve_finite_model_release_batch(projector->model);
        loom_serve_text_media_projector_complete(projector);
      } else {
        projector->phase = MEDIA_WAITING;
      }
      *out_progress = true;
    }
  }
  if (iree_status_is_ok(status) && projector->phase == MEDIA_WAITING) {
    bool live = false;
    for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
      live |= projector->cohort[i]->request != NULL;
    }
    if (stopping || !live) {
      loom_serve_residency_cancel_admission(
          loom_serve_finite_model_residency(projector->model));
      loom_serve_finite_model_release_batch(projector->model);
      loom_serve_text_media_projector_complete(projector);
      *out_progress = true;
    } else {
      loom_serve_residency_admission_t admission;
      status = loom_serve_residency_try_admit(
          loom_serve_finite_model_residency(projector->model), &admission);
      if (iree_status_is_ok(status) &&
          admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
        status = loom_serve_text_media_projector_submit(projector);
        *out_progress = true;
      } else if (iree_status_is_ok(status) &&
                 admission == LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY &&
                 !iree_any_bit_set(flags,
                                   LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY)) {
        for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
          loom_serve_text_media_job_t* job = projector->cohort[i];
          if (job->request) {
            job->status =
                iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                 "image parameters exceed available capacity");
          }
        }
        loom_serve_residency_cancel_admission(
            loom_serve_finite_model_residency(projector->model));
        loom_serve_finite_model_release_batch(projector->model);
        loom_serve_text_media_projector_complete(projector);
        *out_progress = true;
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    projector->phase = MEDIA_FAILED;
  } else if (stopping &&
             !loom_serve_text_media_projector_is_pending(projector)) {
    for (iree_host_size_t i = 0; i < projector->count; ++i) {
      loom_serve_text_media_job_release(&projector->jobs[i]);
    }
    projector->count = 0;
    projector->phase = MEDIA_GATHER;
  } else if (projector->phase == MEDIA_PREPARED) {
    loom_serve_text_media_projector_select(projector);
    *out_progress = true;
  }
  return status;
}

void loom_serve_text_media_projector_free(
    loom_serve_text_media_projector_t* projector) {
  if (!projector) {
    return;
  }
  loom_serve_finite_model_set_observer(projector->model,
                                       (loom_serve_wake_callback_t){0});
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    loom_serve_text_media_job_release(&projector->jobs[i]);
  }
  loom_serve_text_readiness_deinitialize(&projector->readiness);
  iree_allocator_free_aligned(projector->host_allocator, projector);
}
