// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_projector.h"

#include <string.h>

#include "loom/runtime/execution/wait.h"
#include "loom/runtime/host/readiness.h"
#include "loom/serve/media/image_request.h"
#include "loom/service/prepared_request.h"

typedef struct loom_serve_text_media_projection_result_t {
  // Owned cohort reference, independent of producer parameter residency.
  loom_service_finite_result_t* result;
  // Original projection ordinal in that cohort, including cancelled peers.
  uint32_t result_index;
  // A preparation-window job owns this projection until publication/removal.
  bool selected;
} loom_serve_text_media_projection_result_t;

struct loom_serve_text_media_request_t {
  // Allocator for this single request allocation.
  iree_allocator_t host_allocator;
  // Borrowed immutable pending chat, accessed only before language admission.
  const loom_serve_text_chat_t* chat;
  // One owned result reference per source projection.
  loom_serve_text_media_projection_result_t* results;
  // Sorted source input spans rebuilt when retained input coordinates change.
  loom_serve_text_media_input_span_t* spans;
  // Number of allocated projection/coordinate records.
  iree_host_size_t count;
  // First projection still needed by prefill. Checked source spans name a
  // contiguous suffix of chat projections; no parallel ordinal array is
  // necessary.
  iree_host_size_t first_projection;
  // Current bound view, immutable once admitted to an epoch.
  loom_serve_text_media_input_t input;
  // Bound projections whose results have not yet been published by the owner.
  iree_host_size_t missing_count;
  // Jobs attached to the sole projector, including accepted producer work.
  iree_host_size_t job_count;
  // Captured language continuation footprint, excluding image features.
  uint64_t consumer_bytes;
  // Complete request forecast evaluated before decoding its first window.
  loom_service_finite_memory_forecast_t forecast;
  // The declared producer and consumer fit only as sequential phases. Keep
  // future results private until producer completion releases its residency.
  bool await_producer;
  // CPU planning has settled this request, including a failed forecast.
  bool forecasted;
  // One selected cohort may share output backing with peers. All subsequent
  // cohorts are private, bounding retained peer storage independently of the
  // number of projections and preparation windows.
  bool shared_cohort;
  // CPU-only forecast result, transferred after the planning loan joins.
  iree_status_t forecast_status;
  // Owned request rejection, transferred to the HTTP response owner.
  iree_status_t status;
};

static iree_const_byte_span_t loom_serve_text_media_projection_metadata(
    const loom_serve_text_chat_t* chat,
    const loom_serve_text_media_projection_t* projection) {
  return iree_make_const_byte_span(
      projection->metadata_length
          ? iree_string_builder_buffer(&chat->projections.metadata) +
                projection->metadata_offset
          : NULL,
      projection->metadata_length);
}

iree_status_t loom_serve_text_media_request_allocate(
    const loom_serve_text_chat_t* chat, iree_allocator_t host_allocator,
    loom_serve_text_media_request_t** out_request) {
  *out_request = NULL;
  iree_host_size_t length = 0, results_offset = 0, spans_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_media_request_t), &length,
      IREE_STRUCT_FIELD(chat->projections.count,
                        loom_serve_text_media_projection_result_t,
                        &results_offset),
      IREE_STRUCT_FIELD(chat->projections.count,
                        loom_serve_text_media_input_span_t, &spans_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, length, (void**)&storage));
  loom_serve_text_media_request_t* request = (void*)storage;
  *request = (loom_serve_text_media_request_t){
      .host_allocator = host_allocator,
      .chat = chat,
      .results = (void*)(storage + results_offset),
      .spans = (void*)(storage + spans_offset),
      .count = chat->projections.count,
  };
  *out_request = request;
  return iree_ok_status();
}

bool loom_serve_text_media_request_bind(
    loom_serve_text_media_request_t* request,
    const loom_serve_text_input_t* input, iree_host_size_t consumed) {
  request->first_projection = request->count;
  request->input = (loom_serve_text_media_input_t){.spans = request->spans};
  request->missing_count = 0;
  for (iree_host_size_t i = 0; i < input->media_span_count; ++i) {
    const loom_serve_text_media_span_t* span = &input->media_spans[i];
    if ((iree_host_size_t)span->input_offset + span->input_count <= consumed) {
      continue;
    }
    const uint32_t index = span->projection_index;
    const loom_serve_text_media_projection_t* projection =
        &request->chat->projections.values[index];
    const loom_serve_text_media_projection_result_t* projected =
        &request->results[index];
    const iree_host_size_t position = request->input.span_count++;
    if (!position) {
      request->first_projection = index;
    }
    request->spans[position] = (loom_serve_text_media_input_span_t){
        .input_offset = span->input_offset,
        .input_count = span->input_count,
        .request_index = projected->result_index,
        .result = projected->result,
        .metadata = loom_serve_text_media_projection_metadata(request->chat,
                                                              projection),
    };
    request->missing_count += projected->result == NULL;
  }
  return request->missing_count == 0;
}

bool loom_serve_text_media_request_is_ready(
    const loom_serve_text_media_request_t* request) {
  return !request || request->missing_count == 0;
}

void loom_serve_text_media_request_begin(
    loom_serve_text_media_request_t* request) {
  if (!request) {
    return;
  }
  for (iree_host_size_t i = 0; i < request->first_projection; ++i) {
    loom_service_finite_result_release(request->results[i].result);
    request->results[i].result = NULL;
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
    loom_serve_text_media_projection_result_t* projected =
        &request->results[request->first_projection++];
    loom_service_finite_result_release(projected->result);
    projected->result = NULL;
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
    loom_service_finite_result_release(request->results[i].result);
  }
  iree_status_ignore(request->forecast_status);
  iree_allocator_free(request->host_allocator, request);
}

typedef struct loom_serve_text_media_job_t {
  // Borrowed request until publication/cancellation; CPU decode borrows chat.
  loom_serve_text_media_request_t* request;
  // Checked projection ordinal, independent of input-coordinate changes.
  uint32_t projection_index;
  // Expected source feature count, captured before request detachment.
  iree_host_size_t rows;
  // Owned decoded packet retained through producer completion.
  iree_vm_buffer_t* packet;
  // Owned source input, possibly aliasing packet.
  loom_service_prepared_request_t prepared;
  // CPU-produced per-projection status, consumed after the CPU borrow retires.
  iree_status_t status;
} loom_serve_text_media_job_t;

typedef enum loom_serve_text_media_phase_e {
  MEDIA_GATHER = 0,
  MEDIA_SEALED,
  MEDIA_PREPARING,
  MEDIA_PREPARED,
  MEDIA_BATCHING,
  MEDIA_SPECIALIZING,
  MEDIA_WAITING,
  MEDIA_RUNNING,
  MEDIA_COLLECTING,
  MEDIA_FAILED,
} loom_serve_text_media_phase_t;

struct loom_serve_text_media_projector_t {
  // Allocator owning this single bounded window allocation.
  iree_allocator_t host_allocator;
  // Borrowed component model; its owner excludes trim during these loans.
  loom_service_finite_model_t* model;
  // Independent publication channels, never serialized behind text readiness.
  loom_runtime_readiness_t readiness;
  // Finite preparation borrower, independent of accepted device readiness.
  loom_runtime_readiness_task_t cpu_task;
  // Captured producer execution frontier through result collection.
  loom_runtime_execution_wait_t device_wait;
  // Exclusive contributor binding retained by the component owner.
  loom_runtime_service_control_binding_t scheduling;
  // Owner-published legal window and its monotonic revision.
  loom_runtime_service_offer_t offer;
  // Last published offer size; multiple projections count once per request.
  uint64_t offered_request_count;
  // Accepted window retained through all its source cohorts and collection.
  loom_runtime_service_operation_t operation;
  // Last published lifetime, independent of legal offer revision.
  uint64_t published_serial;
  // Borrowed monotonic identity source surviving quiescent replacement.
  uint64_t* next_identity;
  // Last composed summary, suppressing unchanged atomic writes.
  loom_runtime_service_observation_t observation;
  // Actual topology ordinal of this component's device.
  uint32_t device_ordinal;
  // Irreversible admission seal after stop or terminal failure.
  bool stopping;
  // Requests with at least one attached job, independent of projection count.
  iree_host_size_t request_count;
  // Exclusive producer/VM phase.
  loom_serve_text_media_phase_t phase;
  // Maximum projections in one host preparation window.
  iree_host_size_t capacity;
  // Exact decoded packet bytes selected into the gathering window.
  iree_host_size_t packet_bytes;
  // Populated window extent, retaining holes until the window drains.
  iree_host_size_t count;
  // Stable projection records across source preparation and cohort execution.
  loom_serve_text_media_job_t* jobs;
  // Selected compatible cohort, or valid codec inputs during CPU preparation.
  loom_serve_text_media_job_t** cohort;
  // Number of selected source requests.
  iree_host_size_t cohort_count;
  // Owned running-cohort result, including unpublished cancelled peer rows.
  // Failed readiness retains it through the owner's normal retirement path.
  loom_service_finite_result_t* result;
  // Contiguous codec descriptors borrowing the selected jobs.
  loom_serve_media_image_request_t* requests;
  // Codec result buffers transferred into jobs after realization.
  iree_vm_buffer_t** packets;
  // Per-projection realization results transferred into jobs.
  iree_status_t* statuses;
  // Contiguous borrowed source arguments for the selected device cohort.
  loom_service_prepared_request_t* prepared;
  // Expected feature counts for the selected device cohort.
  iree_host_size_t* rows;
};

static void loom_serve_text_media_projector_publish_control(
    loom_serve_text_media_projector_t* projector) {
  loom_runtime_service_work_flags_t work = 0;
  if (!projector->stopping && projector->request_count &&
      !projector->operation.key.serial) {
    if (projector->phase == MEDIA_SEALED) {
      work = LOOM_RUNTIME_SERVICE_WORK_PREPARATION;
    } else if (projector->phase == MEDIA_PREPARED) {
      work = LOOM_RUNTIME_SERVICE_WORK_EXECUTION;
    }
  }
  const uint64_t devices = UINT64_C(1) << projector->device_ordinal;
  const loom_runtime_service_use_t use = {.resource_ordinal =
                                              projector->device_ordinal};
  const uint64_t request_count = work ? projector->request_count : 0;
  if (projector->offer.work != work ||
      projector->offered_request_count != request_count) {
    IREE_ASSERT(*projector->next_identity != UINT64_MAX);
    projector->offer.key.revision = ++*projector->next_identity;
    projector->offer.work = work;
    projector->offer.choice_count = work ? 1 : 0;
    projector->offered_request_count = request_count;
    const bool execution = work == LOOM_RUNTIME_SERVICE_WORK_EXECUTION;
    const loom_runtime_service_choice_publication_t choice = {
        .choice = {.instance_ordinal = projector->scheduling.instance_ordinal,
                   .request_count = request_count,
                   .devices = execution ? devices : 0,
                   .use_count = execution ? 1 : 0},
        .uses = &use};
    loom_runtime_service_publish_offer(projector->scheduling.control,
                                       &projector->offer, &choice);
  }
  loom_runtime_service_wait_flags_t waits = 0;
  switch (projector->phase) {
    case MEDIA_PREPARING:
    case MEDIA_BATCHING:
    case MEDIA_SPECIALIZING:
    case MEDIA_COLLECTING:
      waits = LOOM_RUNTIME_SERVICE_WAIT_CPU;
      break;
    case MEDIA_WAITING:
      waits = LOOM_RUNTIME_SERVICE_WAIT_CAPACITY;
      break;
    case MEDIA_RUNNING:
      waits = LOOM_RUNTIME_SERVICE_WAIT_DEVICE;
      break;
    default:
      break;
  }
  loom_runtime_service_operation_t* operation = &projector->operation;
  if (operation->key.serial != projector->published_serial ||
      operation->waits != waits) {
    operation->waits = waits;
    const bool execution =
        operation->key.serial &&
        operation->work == LOOM_RUNTIME_SERVICE_WORK_EXECUTION;
    operation->devices = execution ? devices : 0;
    operation->use_count = execution ? 1 : 0;
    loom_runtime_service_publish_operation(projector->scheduling.control,
                                           operation, &use);
    projector->published_serial = operation->key.serial;
  }
  const loom_runtime_service_observation_t observation = {
      .owned_request_count = projector->request_count,
      .waiting_request_count = projector->stopping || operation->key.serial
                                   ? 0
                                   : projector->request_count,
      .offer_count = projector->offer.choice_count,
      .pending_operation_count = operation->key.serial != 0,
      .offered_devices =
          work == LOOM_RUNTIME_SERVICE_WORK_EXECUTION ? devices : 0,
      .accepted_devices = operation->devices};
  const loom_runtime_service_observation_t* published = &projector->observation;
  if (observation.owned_request_count != published->owned_request_count ||
      observation.waiting_request_count != published->waiting_request_count ||
      observation.offer_count != published->offer_count ||
      observation.pending_operation_count !=
          published->pending_operation_count ||
      observation.offered_devices != published->offered_devices ||
      observation.accepted_devices != published->accepted_devices) {
    projector->scheduling.publish_observation(projector->scheduling.publisher,
                                              &observation);
    projector->observation = observation;
  }
}

// Request records already own all jobs for their sole projection borrower.
// Retain the exact request count at that boundary instead of scanning
// identities.
static void loom_serve_text_media_job_detach(
    loom_serve_text_media_projector_t* projector,
    loom_serve_text_media_job_t* job) {
  if (job->request && !--job->request->job_count) {
    --projector->request_count;
  }
  job->request = NULL;
}

iree_status_t loom_serve_text_media_projector_allocate(
    loom_service_finite_model_t* model, iree_host_size_t capacity,
    iree_task_executor_t* executor, iree_async_proactor_t* proactor,
    loom_runtime_wake_callback_t observer,
    loom_runtime_service_control_binding_t scheduling, uint64_t* next_identity,
    iree_allocator_t host_allocator,
    loom_serve_text_media_projector_t** out_projector) {
  *out_projector = NULL;
  const uint32_t device_ordinal =
      loom_service_finite_model_generator(model).device_ordinal;
  if (!scheduling.control || !scheduling.publish_observation ||
      !next_identity || device_ordinal >= 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "projection requires a retained control, summary "
                            "publisher, and representable device placement");
  }
  const loom_runtime_service_control_layout_t layout =
      loom_runtime_service_control_layout(scheduling.control);
  if (scheduling.offer_ordinal >= layout.offer_capacity ||
      scheduling.operation_ordinal >= layout.operation_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "projection control requires an offer and operation");
  }
  capacity = (iree_host_size_t)iree_min(
      capacity, loom_service_finite_model_batch_limit(model));
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
      IREE_STRUCT_FIELD(capacity, loom_service_prepared_request_t,
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
      .scheduling = scheduling,
      .next_identity = next_identity,
      .offer = {.key = {.ordinal = scheduling.offer_ordinal}},
      .operation = {.key = {.ordinal = scheduling.operation_ordinal}},
      .device_ordinal = device_ordinal,
      .capacity = capacity,
      .jobs = (void*)(storage + jobs_offset),
      .cohort = (void*)(storage + cohort_offset),
      .requests = (void*)(storage + requests_offset),
      .packets = (void*)(storage + packets_offset),
      .statuses = (void*)(storage + statuses_offset),
      .prepared = (void*)(storage + prepared_offset),
      .rows = (void*)(storage + rows_offset),
  };
  loom_runtime_readiness_initialize(executor, proactor, &projector->readiness);
  loom_runtime_readiness_set_observer(&projector->readiness, observer);
  loom_service_finite_model_set_observer(model, observer);
  *out_projector = projector;
  return iree_ok_status();
}

bool loom_serve_text_media_projector_offer(
    loom_serve_text_media_projector_t* projector,
    loom_serve_text_media_request_t* request, uint64_t consumer_bytes) {
  if (projector->stopping || projector->phase != MEDIA_GATHER) {
    return false;
  }
  if (!iree_status_is_ok(request->status)) {
    return true;
  }
  request->consumer_bytes = consumer_bytes;
  const iree_host_size_t budget =
      loom_service_finite_model_input_capacity(projector->model);
  for (iree_host_size_t i = 0; i < request->input.span_count; ++i) {
    const uint32_t index = (uint32_t)(request->first_projection + i);
    loom_serve_text_media_projection_result_t* projected =
        &request->results[index];
    if (projected->result || projected->selected) {
      continue;
    }
    const loom_serve_text_media_projection_t* projection =
        &request->chat->projections.values[index];
    iree_host_size_t length = 0;
    iree_status_t status = loom_serve_media_image_request_size(
        projection->image_count,
        &request->chat->images.values[projection->image_begin].info,
        sizeof(*request->chat->images.values), projection->metadata_length,
        &length);
    if (!iree_status_is_ok(status)) {
      request->status = iree_status_join(request->status, status);
      loom_serve_text_media_projector_publish_control(projector);
      return true;
    }
    if (projector->count == projector->capacity ||
        (projector->count && (projector->packet_bytes >= budget ||
                              length > budget - projector->packet_bytes))) {
      loom_serve_text_media_projector_publish_control(projector);
      return false;
    }
    projector->jobs[projector->count++] = (loom_serve_text_media_job_t){
        .request = request,
        .projection_index = index,
        .rows = projection->feature_count,
    };
    if (!request->job_count++) {
      ++projector->request_count;
    }
    projector->packet_bytes += length;
    projected->selected = true;
  }
  loom_serve_text_media_projector_publish_control(projector);
  return true;
}

// Forecast the complete suffix, not just this bounded decode window. A later
// cohort can specialize different commands while earlier output roots remain
// live. The source consumes image headers and its own resize metadata; no
// pixels, device compilation, or parameter activation are needed here.
static iree_status_t loom_serve_text_media_request_forecast(
    loom_serve_text_media_request_t* request,
    loom_service_finite_model_t* model) {
  request->forecasted = true;
  const uint64_t limit = loom_service_finite_model_memory_limit(model);
  if (!loom_service_finite_model_has_memory_forecast(model)) {
    return limit == UINT64_MAX
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "bounded image composition requires a source "
                                  "memory forecast");
  }
  const iree_host_size_t count = request->input.span_count;
  loom_service_prepared_request_t* descriptions = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(request->host_allocator,
                                                   count, sizeof(*descriptions),
                                                   (void**)&descriptions));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const loom_serve_text_media_projection_t* projection =
        &request->chat->projections.values[request->first_projection + i];
    status = loom_serve_media_image_request_describe(
        projection->image_count,
        &request->chat->images.values[projection->image_begin].info,
        sizeof(*request->chat->images.values),
        loom_serve_text_media_projection_metadata(request->chat, projection),
        projection->feature_count, request->host_allocator,
        &descriptions[i].bytes);
  }
  iree_vm_buffer_t* ranges = NULL;
  iree_vm_buffer_t* packed = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_service_prepared_requests_pack(
        count, descriptions, request->host_allocator, &ranges, &packed);
  }
  if (iree_status_is_ok(status)) {
    status = loom_service_finite_model_forecast_memory(model, ranges, packed,
                                                       &request->forecast);
  }
  iree_vm_buffer_release(ranges);
  iree_vm_buffer_release(packed);
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_service_prepared_request_release(&descriptions[i]);
  }
  iree_allocator_free(request->host_allocator, descriptions);
  if (iree_status_is_ok(status)) {
    // Reject impossible sequential phases before selecting publication order.
    // Their combined footprint permits overlap within this request; neither
    // check reserves completion credit against independent peer requests.
    const loom_service_finite_memory_forecast_t forecast = request->forecast;
    uint64_t available = limit;
    if (forecast.retained_bytes > available) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "retained image features exceed the serving memory limit");
    } else {
      available -= forecast.retained_bytes;
      if (forecast.private_peak_bytes > available ||
          forecast.expanded_bytes > available ||
          request->consumer_bytes > available - forecast.expanded_bytes) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "image producer or language continuation exceeds "
                             "the serving memory limit of %" PRIu64 " bytes",
                             limit);
      } else {
        request->await_producer =
            forecast.private_peak_bytes >
            available - forecast.expanded_bytes - request->consumer_bytes;
      }
    }
  }
  return status;
}

static iree_status_t loom_serve_text_media_projector_prepare(void* user_data) {
  loom_serve_text_media_projector_t* projector = user_data;
  iree_host_size_t input_capacity = 0;
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    loom_serve_text_media_job_t* job = &projector->jobs[i];
    if (!job->request) {
      continue;
    }
    if (!job->request->forecasted) {
      job->request->forecast_status = loom_serve_text_media_request_forecast(
          job->request, projector->model);
    }
    if (!iree_status_is_ok(job->request->forecast_status)) {
      continue;
    }
    const loom_serve_text_chat_t* chat = job->request->chat;
    const loom_serve_text_media_projection_t* projection =
        &chat->projections.values[job->projection_index];
    if (!iree_host_size_checked_add(input_capacity, projection->image_count,
                                    &input_capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "image preparation window exceeds host storage");
    }
  }
  if (!input_capacity) {
    return iree_ok_status();
  }
  loom_serve_media_image_input_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
      projector->host_allocator, input_capacity, sizeof(*inputs),
      (void**)&inputs));
  iree_host_size_t input_count = 0;
  iree_host_size_t valid_count = 0;
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    loom_serve_text_media_job_t* job = &projector->jobs[i];
    if (!job->request || !iree_status_is_ok(job->request->forecast_status)) {
      continue;
    }
    const loom_serve_text_chat_t* chat = job->request->chat;
    const loom_serve_text_media_projection_t* projection =
        &chat->projections.values[job->projection_index];
    const iree_host_size_t input_begin = input_count;
    for (iree_host_size_t j = 0;
         j < projection->image_count && iree_status_is_ok(job->status); ++j) {
      job->status = loom_serve_media_image_input_initialize(
          chat->images.values[projection->image_begin + j].source,
          &chat->policy->images.limits, projector->host_allocator,
          &inputs[input_count++]);
    }
    if (!iree_status_is_ok(job->status)) {
      continue;
    }
    projector->cohort[valid_count] = job;
    projector->requests[valid_count++] = (loom_serve_media_image_request_t){
        .image_count = projection->image_count,
        .images = &inputs[input_begin],
        .metadata = loom_serve_text_media_projection_metadata(chat, projection),
        .output_rows = projection->feature_count,
    };
  }
  iree_status_t status = loom_serve_media_image_request_create_batch(
      valid_count, projector->requests, projector->host_allocator,
      projector->packets, projector->statuses);
  // Failed inputs are zero-initialized by the codec facade. All successful
  // compressed storage retires after decode, including partial failed groups.
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    loom_serve_media_image_input_deinitialize(&inputs[i]);
  }
  iree_allocator_free(projector->host_allocator, inputs);
  for (iree_host_size_t i = 0; i < valid_count; ++i) {
    loom_serve_text_media_job_t* job = projector->cohort[i];
    job->packet = projector->packets[i];
    job->status = projector->statuses[i];
    if (iree_status_is_ok(status) && iree_status_is_ok(job->status)) {
      job->status = loom_service_finite_model_prepare_request(
          projector->model, job->packet, &job->prepared);
    }
  }
  return status;
}

bool loom_serve_text_media_projector_seal(
    loom_serve_text_media_projector_t* projector) {
  if (!projector || projector->stopping || projector->phase != MEDIA_GATHER ||
      !projector->request_count) {
    return false;
  }
  projector->phase = MEDIA_SEALED;
  loom_serve_text_media_projector_publish_control(projector);
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
    request->results[job->projection_index].selected = false;
    request->status = iree_status_join(request->status, job->status);
    job->status = iree_ok_status();
    loom_serve_text_media_job_detach(projector, job);
  }
  loom_serve_text_media_projector_publish_control(projector);
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
    loom_serve_text_media_projector_t* projector,
    loom_serve_text_media_job_t* job) {
  if (job->request) {
    job->request->results[job->projection_index].selected = false;
    job->request->status = iree_status_join(job->request->status, job->status);
  }
  loom_serve_text_media_job_detach(projector, job);
  loom_service_prepared_request_release(&job->prepared);
  iree_vm_buffer_release(job->packet);
  memset(job, 0, sizeof(*job));
}

static iree_status_t loom_serve_text_media_projector_batch(void* user_data) {
  loom_serve_text_media_projector_t* projector = user_data;
  return loom_service_finite_model_prepare_batch(
      projector->model, projector->cohort_count, projector->prepared,
      projector->rows);
}

static iree_status_t loom_serve_text_media_projector_collect(void* user_data) {
  loom_serve_text_media_projector_t* projector = user_data;
  return loom_service_finite_model_collect_batch(
      projector->model, projector->cohort_count, projector->prepared,
      projector->result);
}

static void loom_serve_text_media_projector_select(
    loom_serve_text_media_projector_t* projector) {
  uint64_t remaining = loom_service_finite_model_batch_limit(projector->model);
  uint64_t tag = 0;
  loom_serve_text_media_request_t* first_request = NULL;
  bool shared = false;
  projector->cohort_count = 0;
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    loom_serve_text_media_job_t* job = &projector->jobs[i];
    if (!job->request || !iree_status_is_ok(job->request->status)) {
      loom_serve_text_media_job_release(projector, job);
      continue;
    }
    if (!projector->cohort_count) {
      tag = job->prepared.tag;
      first_request = job->request;
    }
    if (job->prepared.tag != tag || job->prepared.units > remaining) {
      continue;
    }
    if (job->request != first_request &&
        (first_request->shared_cohort || job->request->shared_cohort)) {
      continue;
    }
    shared |= job->request != first_request;
    remaining -= job->prepared.units;
    const iree_host_size_t index = projector->cohort_count++;
    projector->cohort[index] = job;
    projector->prepared[index] = job->prepared;
    projector->rows[index] = job->rows;
  }
  if (projector->cohort_count) {
    // Commit sharing after selection so later projections of the same owners
    // remain eligible in this cohort. Cancellation cannot restore this credit:
    // a surviving peer may already retain the complete output root.
    if (shared) {
      for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
        projector->cohort[i]->request->shared_cohort = true;
      }
    }
    projector->phase = MEDIA_BATCHING;
    loom_runtime_readiness_schedule(&projector->readiness,
                                    loom_serve_text_media_projector_batch,
                                    projector, &projector->cpu_task);
  } else {
    projector->count = 0;
    projector->packet_bytes = 0;
    projector->phase = MEDIA_GATHER;
    projector->operation.key.serial = 0;
  }
}

static void loom_serve_text_media_projector_retire_cohort(
    loom_serve_text_media_projector_t* projector) {
  loom_service_finite_result_release(projector->result);
  projector->result = NULL;
  projector->cohort_count = 0;
  projector->phase = MEDIA_PREPARED;
}

static void loom_serve_text_media_projector_complete(
    loom_serve_text_media_projector_t* projector) {
  for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
    loom_serve_text_media_job_release(projector, projector->cohort[i]);
  }
  loom_serve_text_media_projector_retire_cohort(projector);
}

static void loom_serve_text_media_projector_publish(
    loom_serve_text_media_projector_t* projector, bool completed) {
  for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
    loom_serve_text_media_job_t* job = projector->cohort[i];
    if (!job->request ||
        !loom_service_finite_result_row_count(projector->result, i) ||
        (job->request->await_producer && !completed)) {
      continue;
    }
    loom_serve_text_media_request_t* request = job->request;
    loom_serve_text_media_projection_result_t* projected =
        &request->results[job->projection_index];
    loom_service_finite_result_retain(projector->result);
    projected->result = projector->result;
    projected->result_index = (uint32_t)i;
    projected->selected = false;
    // A retained prefix can elide a projection after its window was gathered.
    // Bound spans are the checked contiguous suffix of projection identities.
    if (job->projection_index >= request->first_projection &&
        job->projection_index - request->first_projection <
            request->input.span_count) {
      loom_serve_text_media_input_span_t* span =
          &request->spans[job->projection_index - request->first_projection];
      span->result = projected->result;
      span->request_index = projected->result_index;
      --request->missing_count;
    }
    loom_serve_text_media_job_detach(projector, job);
  }
}

static iree_status_t loom_serve_text_media_projector_submit(
    loom_serve_text_media_projector_t* projector) {
  loom_runtime_execution_frontier_t ready;
  IREE_RETURN_IF_ERROR(loom_service_finite_model_submit(
      projector->model, &projector->result, &ready));
  loom_serve_text_media_projector_publish(projector, false);
  IREE_RETURN_IF_ERROR(loom_runtime_execution_await(
      &projector->readiness, &ready, &projector->device_wait));
  projector->phase = MEDIA_RUNNING;
  return iree_ok_status();
}

bool loom_serve_text_media_projector_is_pending(
    const loom_serve_text_media_projector_t* projector) {
  return projector && (projector->phase == MEDIA_PREPARING ||
                       projector->phase == MEDIA_BATCHING ||
                       projector->phase == MEDIA_SPECIALIZING ||
                       projector->phase == MEDIA_RUNNING ||
                       projector->phase == MEDIA_COLLECTING);
}

void loom_serve_text_media_projector_set_observer(
    loom_serve_text_media_projector_t* projector,
    loom_runtime_wake_callback_t observer) {
  if (projector) {
    loom_runtime_readiness_set_observer(&projector->readiness, observer);
    loom_service_finite_model_set_observer(projector->model, observer);
  }
}

const char* loom_serve_text_media_projector_phase(
    const loom_serve_text_media_projector_t* projector) {
  if (!projector || !projector->count) {
    return NULL;
  }
  static const char* const phases[] = {
      "image_gather",  "image_sealed",     "image_prepare", "image_prepared",
      "image_batch",   "image_specialize", "image_wait",    "image_execute",
      "image_collect", "image_failed",
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

static iree_status_t loom_serve_text_media_projector_progress_work(
    loom_serve_text_media_projector_t* projector,
    loom_runtime_service_flags_t flags, bool* out_progress) {
  *out_progress = false;
  if (projector->phase == MEDIA_FAILED) {
    return iree_ok_status();
  }
  const bool stopping =
      iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
  iree_status_t status = iree_ok_status();
  if (projector->phase == MEDIA_PREPARING ||
      projector->phase == MEDIA_BATCHING ||
      projector->phase == MEDIA_COLLECTING) {
    if (!loom_runtime_readiness_take(&projector->readiness,
                                     &projector->cpu_task.result, &status)) {
      return status;
    }
    *out_progress = true;
    if (projector->phase == MEDIA_COLLECTING) {
      if (iree_status_is_ok(status)) {
        loom_service_finite_model_complete(projector->model);
        loom_serve_text_media_projector_publish(projector, true);
        for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
          loom_serve_text_media_job_t* job = projector->cohort[i];
          job->prepared = projector->prepared[i];
          if (!job->request || !job->prepared.units) {
            loom_serve_text_media_job_release(projector, job);
          }
        }
        loom_serve_text_media_projector_retire_cohort(projector);
      }
    } else if (projector->phase == MEDIA_PREPARING) {
      for (iree_host_size_t i = 0; i < projector->count; ++i) {
        loom_serve_text_media_request_t* request = projector->jobs[i].request;
        if (request) {
          request->status =
              iree_status_join(request->status, request->forecast_status);
          request->forecast_status = iree_ok_status();
        }
      }
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
        loom_serve_text_media_job_t* job = &projector->jobs[i];
        if (!iree_status_is_ok(job->status) ||
            (job->request && !iree_status_is_ok(job->request->status))) {
          loom_serve_text_media_job_release(projector, job);
        }
      }
      projector->phase = MEDIA_PREPARED;
      if (iree_status_is_ok(status)) {
        projector->operation.key.serial = 0;
      }
    } else if (!iree_status_is_ok(status)) {
      status = loom_serve_text_media_projector_reject(
          projector->cohort_count, projector->cohort, status);
      loom_serve_text_media_projector_complete(projector);
    } else {
      projector->phase = MEDIA_SPECIALIZING;
    }
  } else if (projector->phase == MEDIA_RUNNING) {
    if (!loom_runtime_readiness_take(&projector->readiness,
                                     &projector->device_wait.wait.result,
                                     &status)) {
      return status;
    }
    *out_progress = true;
    if (iree_status_is_ok(status)) {
      projector->phase = MEDIA_COLLECTING;
      loom_runtime_readiness_schedule(&projector->readiness,
                                      loom_serve_text_media_projector_collect,
                                      projector, &projector->cpu_task);
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
    status = loom_service_finite_model_advance_batch(projector->model, cancel,
                                                     &complete);
    if (!iree_status_is_ok(status)) {
      loom_service_finite_model_release_batch(projector->model);
      status = loom_serve_text_media_projector_reject(
          projector->cohort_count, projector->cohort, status);
      loom_serve_text_media_projector_complete(projector);
    } else if (complete) {
      if (cancel) {
        loom_service_finite_model_release_batch(projector->model);
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
      loom_runtime_residency_cancel_admission(
          loom_service_finite_model_residency(projector->model));
      loom_service_finite_model_release_batch(projector->model);
      loom_serve_text_media_projector_complete(projector);
      *out_progress = true;
    } else {
      loom_runtime_residency_admission_t admission;
      status = loom_runtime_residency_try_admit(
          loom_service_finite_model_residency(projector->model), &admission);
      if (iree_status_is_ok(status) &&
          admission == LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED) {
        status = loom_serve_text_media_projector_submit(projector);
        *out_progress = true;
      } else if (iree_status_is_ok(status) &&
                 admission == LOOM_RUNTIME_RESIDENCY_ADMISSION_CAPACITY &&
                 !iree_any_bit_set(flags,
                                   LOOM_RUNTIME_SERVICE_FLAG_SHARED_ACTIVITY)) {
        for (iree_host_size_t i = 0; i < projector->cohort_count; ++i) {
          loom_serve_text_media_job_t* job = projector->cohort[i];
          if (job->request) {
            job->status =
                iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                 "image parameters exceed available capacity");
          }
        }
        loom_runtime_residency_cancel_admission(
            loom_service_finite_model_residency(projector->model));
        loom_service_finite_model_release_batch(projector->model);
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
      loom_serve_text_media_job_release(projector, &projector->jobs[i]);
    }
    projector->count = 0;
    projector->phase = MEDIA_GATHER;
    projector->packet_bytes = 0;
    projector->operation.key.serial = 0;
  } else if ((projector->phase == MEDIA_PREPARED &&
              projector->operation.key.serial) ||
             (!projector->request_count &&
              (projector->phase == MEDIA_PREPARED ||
               projector->phase == MEDIA_SEALED))) {
    loom_serve_text_media_projector_select(projector);
    *out_progress = true;
  }
  return status;
}

iree_status_t loom_serve_text_media_projector_progress(
    loom_serve_text_media_projector_t* projector,
    loom_runtime_service_flags_t flags, bool* out_progress) {
  *out_progress = false;
  if (!projector) {
    return iree_ok_status();
  }
  projector->stopping |=
      iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
  if (projector->stopping) {
    flags |= LOOM_RUNTIME_SERVICE_FLAG_STOPPING;
  }
  const iree_status_t status = loom_serve_text_media_projector_progress_work(
      projector, flags, out_progress);
  projector->stopping |= !iree_status_is_ok(status);
  loom_serve_text_media_projector_publish_control(projector);
  return status;
}

loom_runtime_service_operation_key_t
loom_serve_text_media_projector_try_schedule(
    loom_serve_text_media_projector_t* projector,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags) {
  if (!projector || projector->stopping ||
      iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING) ||
      offer.ordinal != projector->offer.key.ordinal ||
      offer.revision != projector->offer.key.revision ||
      choice_ordinal >= projector->offer.choice_count) {
    return (loom_runtime_service_operation_key_t){0};
  }
  IREE_ASSERT(*projector->next_identity != UINT64_MAX);
  projector->operation = (loom_runtime_service_operation_t){
      .key = {.ordinal = projector->scheduling.operation_ordinal,
              .serial = ++*projector->next_identity},
      .origin = offer,
      .choice_ordinal = choice_ordinal,
      .work = projector->offer.work};
  const loom_runtime_service_operation_key_t operation =
      projector->operation.key;
  if (projector->phase == MEDIA_SEALED) {
    projector->phase = MEDIA_PREPARING;
    loom_runtime_readiness_schedule(&projector->readiness,
                                    loom_serve_text_media_projector_prepare,
                                    projector, &projector->cpu_task);
  } else {
    loom_serve_text_media_projector_select(projector);
  }
  loom_serve_text_media_projector_publish_control(projector);
  return operation;
}

void loom_serve_text_media_projector_free(
    loom_serve_text_media_projector_t* projector) {
  if (!projector) {
    return;
  }
  loom_service_finite_model_set_observer(projector->model,
                                         (loom_runtime_wake_callback_t){0});
  for (iree_host_size_t i = 0; i < projector->count; ++i) {
    loom_serve_text_media_job_release(projector, &projector->jobs[i]);
  }
  projector->stopping = true;
  loom_serve_text_media_projector_publish_control(projector);
  loom_runtime_readiness_deinitialize(&projector->readiness);
  loom_service_finite_result_release(projector->result);
  iree_allocator_free_aligned(projector->host_allocator, projector);
}
