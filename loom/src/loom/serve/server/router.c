// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/router.h"

#include <inttypes.h>

#include "iree/base/internal/json.h"
#include "iree/base/threading/mutex.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/server/http/logs.h"
#include "loom/serve/server/http/metrics.h"
#include "loom/serve/server/http/package.h"
#include "loom/serve/server/http/response.h"
#include "loom/serve/server/http/tokenization.h"
#include "loom/serve/server/http/variants.h"
#include "loom/serve/server/maintenance.h"
#include "loom/serve/service/kind.h"
#include "loom/serve/transport/http/multipart.h"
#include "loom/util/json.h"

typedef struct loom_serve_router_tokenization_t {
  // Owner-only list link, bounded by the transport's connection credit.
  struct loom_serve_router_tokenization_t* next;
  // Independent finite CPU process, never waiting on model or device work.
  iree_task_process_t process;
  // Borrowed tokenizer, immutable and live through final process release.
  const iree_tokenizer_t* tokenizer;
  // Selected service whose tokenizer remains borrowed through task release.
  iree_host_size_t service_index;
  // Claim keeping the request JSON alive while CPU work uses its views.
  loom_serve_http_connection_t* connection;
  // Borrowed immutable request, live until the owner sends or aborts.
  const loom_serve_http_request_t* request;
  // Completed HTTP bytes copied by transport before this job is freed.
  iree_string_builder_t output;
  // Allocator owning this job and independent tokenizer scratch.
  iree_allocator_t host_allocator;
  // Nonblocking router wake, valid until all accepted jobs retire.
  loom_serve_wake_callback_t observer;
  // Joins release publication's final access before owner destruction.
  iree_slim_mutex_t mutex;
  // Owned outcome, transferred by completion and consumed after release.
  iree_status_t status;
  // The executor and its callbacks have relinquished job storage.
  bool released;
} loom_serve_router_tokenization_t;

typedef struct loom_serve_router_held_request_t {
  // Claim retaining completed request bytes until admission or cancellation.
  loom_serve_http_connection_t* connection;
  // Borrowed immutable body and headers protected by the connection claim.
  const loom_serve_http_request_t* request;
  // Stable registration slot, not reusable while this claim remains held.
  iree_host_size_t service_index;
  // Inference-route classification retained until this claim is released.
  bool demands_parameters;
} loom_serve_router_held_request_t;

typedef struct loom_serve_router_log_reader_t {
  // Retained transport claim; NULL marks an available observation slot.
  loom_serve_http_connection_t* connection;
  // Selector borrowed from the retained request, never model storage.
  iree_string_view_t model;
  // Stable registry slot, or router capacity for the server stream.
  iree_host_size_t service_index;
  // Original stream lifetime, compared before borrowing current storage.
  loom_serve_telemetry_identity_t identity;
  // Next absolute record position; send admission never retains a ring slot.
  uint64_t position;
  // The initial HTTP headers and telemetry batch have been copied to transport.
  bool started;
  // Whether the last emitted snapshot had complete counter ownership.
  bool counters_available;
  // Last emitted owner snapshot; unchanged counters do not sustain a send loop.
  loom_serve_telemetry_counters_t counters;
} loom_serve_router_log_reader_t;

typedef struct loom_serve_router_t {
  // Allocator for independently owned utility jobs and routing output.
  iree_allocator_t host_allocator;
  // Shared application workers, borrowed through final process release.
  iree_task_executor_t* executor;
  // Transport owning network readiness and claimed request storage.
  loom_serve_http_server_t* server;
  // Borrowed stable registration table; this process serializes its mutation.
  loom_serve_registry_t* registry;
  // Cold executable management, retired before registry and transport owners.
  loom_serve_http_variants_t* variants;
  // Immutable borrowed resource view, outliving the outer routing call.
  loom_serve_resources_t resources;
  // Shared-proactor expiry, joined before this owner's wake target disappears.
  loom_serve_maintenance_t maintenance;
  // Number of stable slots, including vacant registration credit.
  iree_host_size_t capacity;
  // Maximum claims per turn, independent of model batch sizes.
  iree_host_size_t request_capacity;
  // Bounded parked claims, independent of each modality's admission queue.
  struct {
    // Owned contiguous entries, compacted in arrival order on owner turns.
    loom_serve_router_held_request_t* values;
    // Maximum held requests across every service.
    iree_host_size_t capacity;
    // Claims retained by closed gates.
    iree_host_size_t count;
  } held;
  // Live observation claims, independent of model and stream lifetimes.
  struct {
    // Fixed slots; entries never retain rendered output or producer storage.
    loom_serve_router_log_reader_t* values;
    // Transport capacity minus one, shared with held claims for control credit.
    iree_host_size_t capacity;
    // Occupied observation slots, including backpressured peers.
    iree_host_size_t count;
    // Rotating formatting priority under the ordinary per-turn request quota.
    iree_host_size_t cursor;
  } logs;
  // Rotating service priority, changed only by the exclusive turn owner.
  iree_host_size_t cursor;
  // Cooperative process returning workers between bounded owner turns.
  iree_task_process_t process;
  // Nonblocking ownership guard: compute-slot wake budgets are not exclusion.
  iree_slim_mutex_t mutex;
  // Admission has stopped; accepted service work still drains.
  bool stopping;
  // All service readiness retired and every observer was detached.
  bool completed;
  // Owned terminal outcome retained while accepted callbacks drain.
  iree_status_t status;
  // Routing response scratch, used only by the exclusive turn owner.
  iree_string_builder_t output;
  // Accepted utility jobs, each holding one transport claim until CPU release.
  loom_serve_router_tokenization_t* tokenizations;
  // Final process retirement, independent of drain completion.
  struct {
    // Joins the release callback's last access before stack storage dies.
    iree_slim_mutex_t mutex;
    // Wakes only the outer application caller, never a task worker.
    iree_notification_t notification;
    // The executor has relinquished all process and drain storage.
    bool released;
  } retirement;
} loom_serve_router_t;

static loom_serve_registered_model_t* loom_serve_router_model_at(
    const loom_serve_router_t* router, iree_host_size_t index) {
  return loom_serve_registry_at(router->registry, index);
}

static void loom_serve_router_wake(void* user_data) {
  loom_serve_router_t* router = user_data;
  iree_task_executor_schedule_process(router->executor, &router->process);
}

static iree_status_t loom_serve_router_quote(iree_string_builder_t* output,
                                             iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static iree_status_t loom_serve_router_tokenization_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_router_tokenization_t* job = process->user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, job->request->target.data,
                              job->request->target.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, job->service_index);
  result->did_work = true;
  result->completed = true;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_builder_append_cstring(
              &job->output,
              "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
              "Connection: close\r\n\r\n"));
  if (iree_string_view_equal(job->request->target, IREE_SV("/tokenize"))) {
    const iree_status_t status = loom_serve_http_tokenize(
        job->tokenizer, job->request->body, &job->output, job->host_allocator);
    IREE_RETURN_AND_END_ZONE(z0, status);
  }
  const iree_status_t status = loom_serve_http_detokenize(
      job->tokenizer, job->request->body, &job->output, job->host_allocator);
  IREE_RETURN_AND_END_ZONE(z0, status);
}

static void loom_serve_router_tokenization_completed(
    iree_task_process_t* process, iree_status_t status) {
  loom_serve_router_tokenization_t* job = process->user_data;
  iree_slim_mutex_lock(&job->mutex);
  job->status = status;
  iree_slim_mutex_unlock(&job->mutex);
}

static void loom_serve_router_tokenization_released(
    iree_task_process_t* process) {
  loom_serve_router_tokenization_t* job = process->user_data;
  iree_slim_mutex_lock(&job->mutex);
  job->released = true;
  job->observer.wake(job->observer.user_data);
  iree_slim_mutex_unlock(&job->mutex);
}

static iree_status_t loom_serve_router_tokenization_accept(
    loom_serve_router_t* router, iree_host_size_t service_index,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request) {
  const iree_tokenizer_t* tokenizer =
      loom_serve_router_model_at(router, service_index)->service.tokenizer;
  if (!tokenizer) {
    return loom_serve_http_response_reject(
        router->server, connection, 404, "Not Found",
        IREE_SV("model has no tokenizer capability"), &router->output);
  }
  loom_serve_router_tokenization_t* job = NULL;
  iree_status_t status = iree_allocator_malloc_aligned(
      router->host_allocator, sizeof(*job),
      iree_alignof(loom_serve_router_tokenization_t), 0, (void**)&job);
  if (!iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    return status;
  }
  job->host_allocator = router->host_allocator;
  job->tokenizer = tokenizer;
  job->service_index = service_index;
  job->connection = connection;
  job->request = request;
  job->observer = (loom_serve_wake_callback_t){loom_serve_router_wake, router};
  iree_slim_mutex_initialize(&job->mutex);
  iree_string_builder_initialize(router->host_allocator, &job->output);
  iree_task_process_initialize(loom_serve_router_tokenization_run, 0, 1,
                               &job->process);
  job->process.user_data = job;
  job->process.completion_fn = loom_serve_router_tokenization_completed;
  job->process.release_fn = loom_serve_router_tokenization_released;
  job->next = router->tokenizations;
  router->tokenizations = job;
  ++loom_serve_router_model_at(router, service_index)->tokenizer_count;
  iree_task_executor_schedule_process(router->executor, &job->process);
  return iree_ok_status();
}

static void loom_serve_router_tokenizations_collect(loom_serve_router_t* router,
                                                    bool* out_progress) {
  loom_serve_router_tokenization_t** link = &router->tokenizations;
  while (*link) {
    loom_serve_router_tokenization_t* job = *link;
    iree_slim_mutex_lock(&job->mutex);
    const bool released = job->released;
    iree_slim_mutex_unlock(&job->mutex);
    if (!released) {
      link = &job->next;
      continue;
    }
    *out_progress = true;
    *link = job->next;
    loom_serve_registered_model_t* model =
        loom_serve_router_model_at(router, job->service_index);
    --model->tokenizer_count;
    const bool cancelled =
        router->stopping || model->removal == LOOM_SERVE_REMOVAL_ABORT;
    iree_status_t status = job->status;
    if (!iree_status_is_ok(status)) {
      const bool invalid = iree_status_is_invalid_argument(status);
      const bool too_large = iree_status_is_out_of_range(status);
      iree_string_view_t message = iree_status_message(status);
      if (!message.size) {
        message = IREE_SV("tokenization failed");
      }
      iree_status_t response_status = iree_ok_status();
      if (cancelled) {
        loom_serve_http_connection_abort(job->connection);
      } else {
        response_status =
            loom_serve_http_response_reject(router->server, job->connection,
                                            too_large ? 413
                                            : invalid ? 400
                                                      : 500,
                                            too_large ? "Content Too Large"
                                            : invalid ? "Bad Request"
                                                      : "Internal Server Error",
                                            message, &job->output);
      }
      // A utility owns only independent CPU state. Its failure cannot poison
      // a model timeline or authorize cancellation of another request.
      loom_serve_http_server_record_error(router->server, status);
      iree_status_free(status);
      status = response_status;
    } else if (cancelled) {
      loom_serve_http_connection_abort(job->connection);
    } else {
      loom_serve_http_response_send(router->server, job->connection,
                                    iree_string_builder_view(&job->output));
    }
    router->status = iree_status_join(router->status, status);
    router->stopping |= !iree_status_is_ok(router->status);
    iree_string_builder_deinitialize(&job->output);
    iree_slim_mutex_deinitialize(&job->mutex);
    iree_allocator_free_aligned(router->host_allocator, job);
  }
}

typedef struct loom_serve_router_selector_t {
  // Undecoded selected name borrowed from the request body.
  iree_string_view_t model;
  // Control bodies contain only the selector, unlike modality-specific bodies.
  bool control;
  // Optional unregister mode destination; other controls reject this field.
  loom_serve_removal_t* removal;
  // Whether the optional removal mode has already occurred.
  iree_string_view_t mode;
  // Optional parameter-component destination; other control routes reject it.
  loom_serve_model_component_t* component;
  // Raw component selector used to reject duplicate fields.
  iree_string_view_t component_name;
} loom_serve_router_selector_t;

static iree_status_t loom_serve_router_model_visit(void* user_data,
                                                   iree_string_view_t key,
                                                   iree_json_value_type_t type,
                                                   iree_string_view_t value) {
  loom_serve_router_selector_t* selector = user_data;
  iree_string_view_t* model = &selector->model;
  if (selector->component &&
      iree_string_view_equal(key, IREE_SV("component"))) {
    if (selector->component_name.data || type != IREE_JSON_VALUE_TYPE_STRING) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "component must occur once as a string");
    }
    selector->component_name = value;
    char decoded[7];
    iree_host_size_t length = 0;
    IREE_RETURN_IF_ERROR(
        iree_json_unescape_string(value, sizeof(decoded), decoded, &length));
    const iree_string_view_t name = iree_make_string_view(decoded, length);
    if (iree_string_view_equal(name, IREE_SV("primary"))) {
      *selector->component = LOOM_SERVE_MODEL_COMPONENT_PRIMARY;
    } else if (iree_string_view_equal(name, IREE_SV("image"))) {
      *selector->component = LOOM_SERVE_MODEL_COMPONENT_IMAGE;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "component must be primary or image");
    }
    return iree_ok_status();
  }
  if (selector->removal && iree_string_view_equal(key, IREE_SV("mode"))) {
    if (selector->mode.data || type != IREE_JSON_VALUE_TYPE_STRING) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "mode must occur once as a string");
    }
    selector->mode = value;
    if (iree_string_view_equal(value, IREE_SV("drain"))) {
      *selector->removal = LOOM_SERVE_REMOVAL_DRAIN;
    } else if (iree_string_view_equal(value, IREE_SV("abort"))) {
      *selector->removal = LOOM_SERVE_REMOVAL_ABORT;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "mode must be drain or abort");
    }
    return iree_ok_status();
  }
  if (!iree_string_view_equal(key, IREE_SV("model"))) {
    if (selector->control) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model control accepts only a model selector");
    }
    return iree_ok_status();
  }
  if (model->data || type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model must occur exactly once as a string");
  }
  *model = value;
  return iree_ok_status();
}

static iree_status_t loom_serve_router_model(
    const loom_serve_http_request_t* request, bool control,
    loom_serve_removal_t* removal, loom_serve_model_component_t* component,
    iree_string_builder_t* output) {
  if (request->multipart) {
    if (control) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model control requires a JSON body");
    }
    const loom_serve_http_part_list_t parts =
        loom_serve_http_multipart_parts(request->multipart);
    iree_string_view_t model = iree_string_view_empty();
    for (iree_host_size_t i = 0; i < parts.count; ++i) {
      const loom_serve_http_part_t* part = &parts.values[i];
      if (!iree_string_view_equal(part->name, IREE_SV("model"))) {
        continue;
      }
      if (model.data || part->filename.data || !part->contents.size) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "multipart model must occur once as nonempty text");
      }
      model = part->contents;
    }
    if (!model.data) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "multipart model field is required");
    }
    iree_string_builder_reset(output);
    return iree_string_builder_append_string(output, model);
  }
  iree_string_view_t cursor = request->body;
  iree_string_view_t object;
  loom_serve_router_selector_t selector = {
      .control = control, .removal = removal, .component = component};
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing request JSON");
  }
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_router_model_visit, &selector));
  const iree_string_view_t model = selector.model;
  if (!model.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "request requires a nonempty model string");
  }
  iree_string_builder_reset(output);
  char* data = NULL;
  iree_host_size_t capacity = 0, length = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      output, model.size, &data, &capacity));
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(model, capacity, data, &length));
  iree_string_builder_commit_append(output, length);
  return iree_ok_status();
}

static loom_serve_service_flags_t loom_serve_router_service_flags(
    const loom_serve_router_t* router, iree_host_size_t selected) {
  const void* capacity_domain =
      loom_serve_router_model_at(router, selected)->service.capacity_domain;
  if (!capacity_domain) {
    return 0;
  }
  for (iree_host_size_t i = 0; i < router->capacity; ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_router_model_at(router, i);
    if (i == selected || model->service.capacity_domain != capacity_domain) {
      continue;
    }
    // Retirement still owns backing after service callbacks are gone. Its
    // ownership wake may release capacity; a quiescent failed release cannot.
    if (model->phase == LOOM_SERVE_REGISTRATION_RETIRING ||
        (model->phase == LOOM_SERVE_REGISTRATION_READY &&
         (model->service.is_active(model->service.self) ||
          model->parameter_operation ==
              LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING))) {
      return LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY;
    }
  }
  return 0;
}

static const char* loom_serve_router_admission_phase(
    const loom_serve_registered_model_t* state,
    const loom_serve_service_snapshot_t* snapshot) {
  if (state->removal) {
    return "removing";
  }
  if (!state->closed) {
    return "open";
  }
  return snapshot->queued_requests || snapshot->active_requests ||
                 state->tokenizer_count
             ? "draining"
             : "drained";
}

static iree_status_t loom_serve_router_describe_residency(
    loom_serve_residency_snapshot_t state, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"active\":%s,\"backing\":\"%s\",\"activations\":"
      "%" PRIu64 ",\"deactivations\":%" PRIu64
      ",\"pins\":%zu,"
      "\"admission\":{\"waiting\":%s,\"count\":%" PRIu64
      ",\"duration_seconds\":%.9f,\"age_seconds\":%.9f}",
      state.active ? "true" : "false", state.elastic ? "elastic" : "fixed",
      state.activity.activations, state.activity.deactivations, state.pin_count,
      state.admission.waiting ? "true" : "false", state.admission.count,
      state.admission.duration_ns / 1e9, state.admission.age_ns / 1e9));
  if (state.elastic) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        ",\"reserved_bytes\":%" PRIu64 ",\"committed_bytes\":%" PRIu64
        ",\"pending_bytes\":%" PRIu64 ",\"releasing_bytes\":%" PRIu64,
        state.memory.reserved_bytes, state.memory.committed_bytes,
        state.memory.pending_bytes, state.memory.releasing_bytes));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_router_describe_parameters(
    const loom_serve_registered_model_t* model, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"parameters\":{"));
  iree_status_t status = iree_ok_status();
  for (loom_serve_model_component_t component = 0;
       component < LOOM_SERVE_MODEL_COMPONENT_COUNT &&
       iree_status_is_ok(status);
       ++component) {
    if (!loom_serve_registry_has_component(model, component)) {
      continue;
    }
    loom_serve_registration_parameters_t state = {
        .phase = loom_serve_registry_phase(model)};
    if (model->phase == LOOM_SERVE_REGISTRATION_READY &&
        model->registration.parameters) {
      state =
          model->registration.parameters(model->registration.self, component);
    } else if (component == LOOM_SERVE_MODEL_COMPONENT_IMAGE &&
               (model->phase == LOOM_SERVE_REGISTRATION_DECLARED ||
                model->phase == LOOM_SERVE_REGISTRATION_PREPARING)) {
      state.phase = "declared";
    }
    status = iree_string_builder_append_format(
        output, "%s\"%s\":{\"phase\":\"%s\",\"pinned\":%s",
        component ? "," : "", loom_serve_model_component_name(component),
        state.phase,
        (model->pinned_components & (1u << component)) ? "true" : "false");
    if (iree_status_is_ok(status) && state.prepared) {
      status = loom_serve_router_describe_residency(state.parameters, output);
    }
    if (iree_status_is_ok(status) && !iree_status_is_ok(state.status)) {
      status = iree_string_builder_append_format(
          output, ",\"error\":{\"code\":%d,\"message\":",
          iree_status_code(state.status));
      if (iree_status_is_ok(status)) {
        status =
            loom_serve_router_quote(output, iree_status_message(state.status));
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(output, "}");
      }
    }
    const loom_serve_compilation_report_t* report = state.compilation;
    if (iree_status_is_ok(status) && report &&
        (report->commands || report->severities[0] || report->severities[1] ||
         report->severities[2])) {
      status = iree_string_builder_append_cstring(output, ",\"compilation\":");
      if (iree_status_is_ok(status)) {
        loom_output_stream_t stream;
        loom_output_stream_for_builder(output, &stream);
        status = loom_serve_compilation_write(report, &stream);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, "}");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "}");
  }
  return status;
}

static iree_status_t loom_serve_router_describe_modalities(
    const char* key, iree_string_view_list_t modalities,
    iree_string_builder_t* output) {
  if (!modalities.count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(output, ",\"%s\":[", key));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < modalities.count && iree_status_is_ok(status); ++i) {
    if (i) {
      status = iree_string_builder_append_cstring(output, ",");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_router_quote(output, modalities.values[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "]");
  }
  return status;
}

static iree_status_t loom_serve_router_describe_service(
    const loom_serve_registered_model_t* admission, bool include_status,
    iree_string_builder_t* output) {
  const loom_serve_service_t* service = &admission->service;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, "{\"object\":\"model\",\"id\":"));
  IREE_RETURN_IF_ERROR(loom_serve_router_quote(output, service->name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"created\":%" PRId64 ",\"owned_by\":\"local\"",
      admission->registered_unix_seconds));
  IREE_RETURN_IF_ERROR(loom_serve_router_describe_modalities(
      "input_modalities", service->capabilities.input_modalities, output));
  IREE_RETURN_IF_ERROR(loom_serve_router_describe_modalities(
      "output_modalities", service->capabilities.output_modalities, output));
  if (service->capabilities.context_length) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",\"context_length\":%zu",
        service->capabilities.context_length));
  }
  if (service->capabilities.max_output_tokens) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",\"limits\":{\"max_output_tokens\":%zu}",
        service->capabilities.max_output_tokens));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"loom\":{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_serve_router_quote(
      output,
      iree_make_cstring_view(loom_serve_model_kind_name(service->kind))));
  if (service->endpoint) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"endpoint\":"));
    IREE_RETURN_IF_ERROR(loom_serve_router_quote(
        output, iree_make_cstring_view(service->endpoint)));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"tokenize\":%s", service->tokenizer ? "true" : "false"));
  if (service->capabilities.input_token_count) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"input_tokens\":true"));
  }
  if (admission->phase == LOOM_SERVE_REGISTRATION_READY &&
      service->reconfiguration.pool) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",\"variants\":{\"selected\":%" PRIu64 ",\"count\":%zu}",
        service->reconfiguration.selected(service->reconfiguration.self),
        loom_serve_variant_pool_count(service->reconfiguration.pool)));
  }
  if (admission->incarnation) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        ",\"incarnation\":%" PRIu64 ",\"device\":%zu,\"registration\":\"%s\"",
        admission->incarnation, admission->configuration.device_index,
        loom_serve_registry_phase(admission)));
    if (!iree_status_is_ok(admission->status)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, ",\"error\":{\"code\":%d,\"message\":",
          iree_status_code(admission->status)));
      IREE_RETURN_IF_ERROR(loom_serve_router_quote(
          output, iree_status_message(admission->status)));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
    }
    const loom_serve_compilation_report_t* report = &admission->compilation;
    if (include_status && (report->commands || report->severities[0] ||
                           report->severities[1] || report->severities[2])) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(output, ",\"compilation\":"));
      loom_output_stream_t stream;
      loom_output_stream_for_builder(output, &stream);
      IREE_RETURN_IF_ERROR(loom_serve_compilation_write(report, &stream));
    }
  }
  if (include_status) {
    if (admission->incarnation) {
      static const char* const operations[] = {"idle", "waiting", "activating",
                                               "deactivating"};
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output,
          ",\"parameter_operation\":\"%s\",\"parameter_component\":\"%s\"",
          operations[admission->parameter_operation],
          loom_serve_model_component_name(admission->parameter_component)));
    }
    if (admission->incarnation) {
      IREE_RETURN_IF_ERROR(
          loom_serve_router_describe_parameters(admission, output));
    }
    const loom_serve_service_snapshot_t state =
        loom_serve_registry_snapshot(admission);
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"status\":{\"phase\":"));
    IREE_RETURN_IF_ERROR(
        loom_serve_router_quote(output, iree_make_cstring_view(state.phase)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        ",\"queued_requests\":%zu,\"active_requests\":%zu,"
        "\"oldest_queued_age_seconds\":%.9f,\"oldest_active_age_seconds\":%."
        "9f,\"admission\":\"%s\",\"held_requests\":%zu,"
        "\"tokenizer_requests\":%zu}",
        state.queued_requests, state.active_requests,
        state.oldest_queued_age / 1e9, state.oldest_active_age / 1e9,
        loom_serve_router_admission_phase(admission, &state),
        admission->held_count, admission->tokenizer_count));
  }
  return iree_string_builder_append_cstring(output, "}}");
}

static iree_status_t loom_serve_router_describe_server(
    const loom_serve_router_t* router, iree_string_builder_t* output) {
  const loom_serve_http_request_limits_t limits =
      loom_serve_http_server_request_limits(router->server);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "{\"worker_count\":%zu,\"model_capacity\":%zu,"
      "\"connection_capacity\":%zu,\"request_limits\":{"
      "\"header_byte_capacity\":%zu,\"body_byte_capacity\":%zu,"
      "\"header_capacity\":%zu,\"multipart_header_byte_capacity\":%zu,"
      "\"multipart_part_capacity\":%zu},\"held_request_capacity\":%zu,"
      "\"held_requests\":%zu,\"observer_capacity\":%zu,\"observers\":%zu,"
      "\"idle\":{\"workspace_ms\":%" PRId64 ",\"parameter_ms\":%" PRId64
      "},"
      "\"resources\":",
      iree_task_executor_worker_count(router->executor), router->capacity,
      loom_serve_http_server_connection_capacity(router->server),
      limits.header_byte_capacity, limits.body_byte_capacity,
      limits.header_capacity, limits.multipart_header_byte_capacity,
      limits.multipart_part_capacity, router->held.capacity, router->held.count,
      router->logs.capacity, router->logs.count,
      router->maintenance.policy.workspace_timeout / 1000000,
      router->maintenance.policy.parameter_timeout / 1000000));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_serve_resources_write_json(&router->resources, &stream));
  return iree_string_builder_append_cstring(output, "}");
}

static iree_status_t loom_serve_router_logs(
    loom_serve_router_t* router, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request) {
  const bool streaming =
      iree_string_view_equal(request->target, IREE_SV("/logz/stream"));
  iree_host_size_t service_index = router->capacity;
  const loom_serve_telemetry_stream_t* stream =
      loom_serve_http_server_diagnostics(router->server);
  const loom_serve_telemetry_counters_t* counters = NULL;
  loom_serve_telemetry_counters_t total;
  iree_string_view_t name;
  if (loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Model"),
                                            &name)) {
    const loom_serve_service_t* service = NULL;
    for (iree_host_size_t i = 0; i < router->capacity; ++i) {
      const loom_serve_service_t* candidate =
          &loom_serve_router_model_at(router, i)->service;
      if (candidate->name.size &&
          iree_string_view_equal(name, candidate->name)) {
        service = candidate;
        service_index = i;
        break;
      }
    }
    if (!service) {
      return loom_serve_http_response_reject(
          router->server, connection, 404, "Not Found",
          IREE_SV("model is not registered"), &router->output);
    }
    if (!service->telemetry.stream) {
      return loom_serve_http_response_reject(
          router->server, connection, 501, "Not Implemented",
          IREE_SV("model service has no telemetry stream"), &router->output);
    }
    stream = service->telemetry.stream;
    counters = service->telemetry.counters;
  } else if (loom_serve_registry_counters(router->registry, &total)) {
    counters = &total;
  }
  iree_string_view_t cursor;
  loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Cursor"),
                                        &cursor);
  if (streaming) {
    iree_string_view_t previous;
    loom_serve_http_request_lookup_header(request, IREE_SV("Last-Event-ID"),
                                          &previous);
    if (cursor.size && previous.size &&
        !iree_string_view_equal(cursor, previous)) {
      return loom_serve_http_response_reject(
          router->server, connection, 400, "Bad Request",
          IREE_SV("conflicting telemetry resume cursors"), &router->output);
    }
    if (previous.size) {
      cursor = previous;
    }
  }
  uint64_t position = 0;
  iree_status_t status = loom_serve_http_logs_resume(stream, cursor, &position);
  if (!iree_status_is_ok(status)) {
    const bool stale = iree_status_is_failed_precondition(status);
    iree_status_t response_status = loom_serve_http_response_reject(
        router->server, connection, stale ? 409 : 400,
        stale ? "Conflict" : "Bad Request", iree_status_message(status),
        &router->output);
    // The HTTP error owns the copied diagnostic before this status retires.
    iree_status_free(status);
    return response_status;
  }
  if (streaming) {
    if (router->logs.count + router->held.count +
            loom_serve_http_variants_claims(router->variants) ==
        router->logs.capacity) {
      return loom_serve_http_response_reject(
          router->server, connection, 429, "Too Many Requests",
          IREE_SV("observation and holding capacity is full"), &router->output);
    }
    for (iree_host_size_t i = 0; i < router->logs.capacity; ++i) {
      if (!router->logs.values[i].connection) {
        router->logs.values[i] = (loom_serve_router_log_reader_t){
            .connection = connection,
            .model = name,
            .service_index = service_index,
            .identity = loom_serve_telemetry_stream_identity(stream),
            .position = position};
        ++router->logs.count;
        break;
      }
    }
    return iree_ok_status();
  }
  status = loom_serve_http_logs_format(stream, counters, name,
                                       LOOM_SERVE_HTTP_LOGS_FLAG_HEADERS,
                                       &position, &router->output);
  if (iree_status_is_ok(status)) {
    loom_serve_http_response_send(router->server, connection,
                                  iree_string_builder_view(&router->output));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

// Observation never owns a model or stream across turns. Native removal may
// reclaim and reuse the recorded registry slot while this peer has no credit.
static iree_status_t loom_serve_router_logs_advance(loom_serve_router_t* router,
                                                    bool* out_progress) {
  if (!router->logs.count) {
    return iree_ok_status();
  }
  loom_serve_telemetry_counters_t total;
  const bool total_available =
      loom_serve_registry_counters(router->registry, &total);
  const iree_host_size_t start = router->logs.cursor;
  iree_host_size_t quota = router->request_capacity;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < router->logs.capacity && iree_status_is_ok(status); ++i) {
    const iree_host_size_t index = (start + i) % router->logs.capacity;
    loom_serve_router_log_reader_t* reader = &router->logs.values[index];
    if (!reader->connection) {
      continue;
    }
    if (router->stopping ||
        loom_serve_http_connection_failed(reader->connection)) {
      loom_serve_http_connection_abort(reader->connection);
      reader->connection = NULL;
      --router->logs.count;
      *out_progress = true;
      continue;
    }
    if (!quota || !loom_serve_http_connection_can_send(reader->connection)) {
      continue;
    }
    const loom_serve_telemetry_stream_t* stream =
        loom_serve_http_server_diagnostics(router->server);
    const loom_serve_telemetry_counters_t* counters =
        total_available ? &total : NULL;
    if (reader->service_index < router->capacity) {
      const loom_serve_service_t* service =
          &loom_serve_router_model_at(router, reader->service_index)->service;
      stream = service->telemetry.stream;
      counters = service->telemetry.counters;
    }
    const loom_serve_telemetry_identity_t identity =
        stream ? loom_serve_telemetry_stream_identity(stream)
               : (loom_serve_telemetry_identity_t){0};
    const bool ended = identity.created_ns != reader->identity.created_ns ||
                       identity.ordinal != reader->identity.ordinal;
    const bool counters_changed =
        !ended &&
        (reader->counters_available != (counters != NULL) ||
         (counters && memcmp(&reader->counters, counters, sizeof(*counters))));
    if (!ended && reader->started && !counters_changed &&
        reader->position == loom_serve_telemetry_stream_head(stream)) {
      continue;
    }
    // Charge formatting attempts, including a concurrent busy read that emits
    // no bytes. Other readers retain rotating priority on the next wake.
    --quota;
    router->logs.cursor = (index + 1) % router->logs.capacity;
    if (ended) {
      iree_string_builder_reset(&router->output);
      status = iree_string_builder_append_format(
          &router->output,
          "%sevent: stream_end\ndata: {\"identity\":\"%" PRIu64 ":%" PRIu64
          "\",\"reason\":\"model_unregistered\"}\n\n",
          reader->started
              ? ""
              : "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\nCache-Control: no-store\r\n"
                "Connection: close\r\n\r\n",
          reader->identity.created_ns, reader->identity.ordinal);
    } else {
      const uint64_t previous_position = reader->position;
      status = loom_serve_http_logs_format(
          stream, counters, reader->model,
          LOOM_SERVE_HTTP_LOGS_FLAG_SSE |
              (reader->started ? 0 : LOOM_SERVE_HTTP_LOGS_FLAG_HEADERS),
          &reader->position, &router->output);
      if (iree_status_is_ok(status)) {
        // Concurrent overwrite can make a read attempt busy without advancing
        // its cursor. The producer's publication wake retries it, not an empty
        // send/completion cycle.
        if (reader->started && !counters_changed &&
            reader->position == previous_position) {
          continue;
        }
        reader->counters_available = counters != NULL;
        if (counters) {
          reader->counters = *counters;
        }
        reader->started = true;
      }
    }
    if (iree_status_is_ok(status)) {
      iree_status_t send_status = loom_serve_http_connection_send(
          reader->connection, iree_string_builder_view(&router->output));
      if (!iree_status_is_ok(send_status)) {
        loom_serve_http_server_record_error(router->server, send_status);
        iree_status_free(send_status);
        loom_serve_http_connection_abort(reader->connection);
        reader->connection = NULL;
      } else if (ended) {
        loom_serve_http_connection_finish(reader->connection);
        reader->connection = NULL;
      }
    } else {
      loom_serve_http_connection_abort(reader->connection);
      reader->connection = NULL;
    }
    if (!reader->connection) {
      --router->logs.count;
    }
    *out_progress = true;
  }
  return status;
}

static iree_status_t loom_serve_router_admit(
    loom_serve_router_t* router, iree_host_size_t service_index,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request) {
  if (iree_string_view_equal(request->method, IREE_SV("POST")) &&
      (iree_string_view_equal(request->target, IREE_SV("/tokenize")) ||
       iree_string_view_equal(request->target, IREE_SV("/detokenize")))) {
    return loom_serve_router_tokenization_accept(router, service_index,
                                                 connection, request);
  }
  const loom_serve_service_t* service =
      &loom_serve_router_model_at(router, service_index)->service;
  return service->accept(
      service->self, connection, request,
      loom_serve_router_service_flags(router, service_index));
}

static void loom_serve_router_held_collect(loom_serve_router_t* router,
                                           bool* out_progress) {
  iree_host_size_t remaining = 0;
  iree_host_size_t admitted = 0;
  for (iree_host_size_t i = 0; i < router->held.count; ++i) {
    const loom_serve_router_held_request_t entry = router->held.values[i];
    loom_serve_registered_model_t* state =
        loom_serve_router_model_at(router, entry.service_index);
    if (router->stopping ||
        loom_serve_http_connection_failed(entry.connection)) {
      loom_serve_http_connection_abort(entry.connection);
    } else if (state->removal || !iree_status_is_ok(state->status)) {
      router->status = iree_status_join(
          router->status,
          loom_serve_http_response_reject(
              router->server, entry.connection, state->removal ? 410 : 503,
              state->removal ? "Gone" : "Service Unavailable",
              state->removal
                  ? IREE_SV("model is being unregistered")
                  : IREE_SV("model is unavailable; inspect /v1/status"),
              &router->output));
      router->stopping |= !iree_status_is_ok(router->status);
    } else if (!state->closed &&
               state->phase == LOOM_SERVE_REGISTRATION_READY &&
               admitted < router->request_capacity) {
      router->status = iree_status_join(
          router->status,
          loom_serve_router_admit(router, entry.service_index, entry.connection,
                                  entry.request));
      router->stopping |= !iree_status_is_ok(router->status);
      ++admitted;
    } else {
      router->held.values[remaining++] = entry;
      continue;
    }
    --state->held_count;
    state->parameter_demand_count -= entry.demands_parameters;
    *out_progress = true;
  }
  router->held.count = remaining;
}

static iree_status_t loom_serve_router_control(
    loom_serve_router_t* router, iree_host_size_t service_index,
    loom_serve_http_connection_t* connection, int code) {
  iree_string_builder_t* output = &router->output;
  iree_string_builder_reset(output);
  iree_status_t status = iree_string_builder_append_format(
      output,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n",
      code, code == 202 ? "Accepted" : "OK");
  if (iree_status_is_ok(status)) {
    status = loom_serve_router_describe_service(
        loom_serve_router_model_at(router, service_index), true, output);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_response_send(router->server, connection,
                                  iree_string_builder_view(output));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t loom_serve_router_register(
    loom_serve_router_t* router, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request) {
  loom_serve_model_configuration_t configuration = {0};
  iree_string_view_t json;
  loom_serve_package_t* package = NULL;
  iree_status_t status = loom_serve_http_package_parse(
      request, router->host_allocator, &json, &package);
  if (iree_status_is_ok(status)) {
    status = loom_serve_model_configuration_initialize(
        json, iree_string_view_empty(), package, router->host_allocator,
        &configuration);
  }
  loom_serve_package_release(package);
  iree_host_size_t index = 0;
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_registry_register(router->registry, &configuration, &index);
  }
  loom_serve_model_configuration_deinitialize(&configuration);
  if (!iree_status_is_ok(status)) {
    return loom_serve_http_response_error(router->server, connection, status,
                                          &router->output);
  }
  return loom_serve_router_control(router, index, connection, 202);
}

static iree_status_t loom_serve_router_trim(
    loom_serve_router_t* router, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request) {
  iree_host_size_t device_index = 0;
  uint64_t target_bytes = 0;
  iree_status_t status = loom_serve_resources_parse_trim(
      &router->resources, request->body, &device_index, &target_bytes);
  if (!iree_status_is_ok(status)) {
    return loom_serve_http_response_error(router->server, connection, status,
                                          &router->output);
  }
  iree_string_builder_reset(&router->output);
  status = iree_string_builder_append_cstring(
      &router->output,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Cache-Control: no-store\r\nConnection: close\r\n\r\n");
  if (iree_status_is_ok(status)) {
    loom_output_stream_t output;
    loom_output_stream_for_builder(&router->output, &output);
    status = loom_serve_resources_trim(&router->resources, device_index,
                                       target_bytes, &output);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_response_send(router->server, connection,
                                  iree_string_builder_view(&router->output));
  } else {
    // A partial physical release is terminal for shared backing, not just this
    // request. Propagate to ordinary owner draining before admitting more work.
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t loom_serve_router_inspect(
    loom_serve_router_t* router, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_http_inspection_kind_t kind) {
  iree_string_view_t name;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Model"),
                                             &name) ||
      !name.size || request->body.size) {
    return loom_serve_http_response_reject(
        router->server, connection, 400, "Bad Request",
        IREE_SV("inspection requires X-Loom-Model and no body"),
        &router->output);
  }
  iree_host_size_t offset = 0;
  iree_string_view_t offset_value;
  if (loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Offset"),
                                            &offset_value) &&
      !loom_serve_http_inspection_offset(offset_value, &offset)) {
    return loom_serve_http_response_reject(
        router->server, connection, 400, "Bad Request",
        IREE_SV("X-Loom-Offset must be an unsigned decimal index"),
        &router->output);
  }
  loom_serve_registered_model_t* model = NULL;
  for (iree_host_size_t i = 0; i < router->capacity; ++i) {
    loom_serve_registered_model_t* candidate =
        loom_serve_router_model_at(router, i);
    if (iree_string_view_equal(name, candidate->service.name)) {
      model = candidate;
      break;
    }
  }
  if (!model) {
    return loom_serve_http_response_reject(
        router->server, connection, 404, "Not Found",
        IREE_SV("model is not registered"), &router->output);
  }
  if (model->phase != LOOM_SERVE_REGISTRATION_READY) {
    return loom_serve_http_response_reject(
        router->server, connection, 409, "Conflict",
        IREE_SV("model has no live service; inspect /v1/status"),
        &router->output);
  }
  const loom_serve_http_inspection_t* inspection = &model->service.inspection;
  if (kind == LOOM_SERVE_HTTP_INSPECT_REQUESTS ? !inspection->request_at
                                               : !inspection->session_at) {
    return loom_serve_http_response_reject(
        router->server, connection, 501, "Not Implemented",
        IREE_SV("model service does not expose this inventory"),
        &router->output);
  }
  iree_string_builder_t* output = &router->output;
  iree_string_builder_reset(output);
  const iree_time_t now = iree_time_now();
  iree_status_t status = iree_string_builder_append_cstring(
      output,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
      "{\"object\":\"list\",\"model\":");
  if (iree_status_is_ok(status)) {
    status = loom_serve_router_quote(output, name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        output,
        ",\"incarnation\":%" PRIu64 ",\"time_ns\":%" PRId64
        ",\"offset\":%zu,\"held_requests\":%zu,\"tokenizer_requests\":%zu",
        model->incarnation, now, offset, model->held_count,
        model->tokenizer_count);
  }
  if (iree_status_is_ok(status)) {
    if (model->service.telemetry.stream) {
      const loom_serve_telemetry_identity_t identity =
          loom_serve_telemetry_stream_identity(model->service.telemetry.stream);
      status = iree_string_builder_append_format(
          output, ",\"identity\":\"%" PRIu64 ":%" PRIu64 "\"",
          identity.created_ns, identity.ordinal);
    } else {
      status = iree_string_builder_append_cstring(output, ",\"identity\":null");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_inspection_format(inspection, model->service.self,
                                               kind, offset, now, output);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "}");
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_response_send(router->server, connection,
                                  iree_string_builder_view(output));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t loom_serve_router_accept(
    loom_serve_router_t* router, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request) {
  const iree_host_size_t service_count =
      loom_serve_registry_count(router->registry);
  iree_string_builder_t* output = &router->output;
  const bool get = iree_string_view_equal(request->method, IREE_SV("GET"));
  if (loom_serve_http_variants_handles(request->target)) {
    const bool allow_wait =
        router->held.count + router->logs.count +
            loom_serve_http_variants_claims(router->variants) <
        router->logs.capacity;
    return loom_serve_http_variants_accept(router->variants, connection,
                                           request, allow_wait, output);
  }
  if (get && iree_string_view_equal(request->target, IREE_SV("/v1/requests"))) {
    return loom_serve_router_inspect(router, connection, request,
                                     LOOM_SERVE_HTTP_INSPECT_REQUESTS);
  }
  if (get && iree_string_view_equal(request->target, IREE_SV("/v1/sessions"))) {
    return loom_serve_router_inspect(router, connection, request,
                                     LOOM_SERVE_HTTP_INSPECT_SESSIONS);
  }
  if (get &&
      (iree_string_view_equal(request->target, IREE_SV("/logz")) ||
       iree_string_view_equal(request->target, IREE_SV("/logz/stream")))) {
    return loom_serve_router_logs(router, connection, request);
  }
  if (get && iree_string_view_equal(request->target, IREE_SV("/metrics"))) {
    iree_status_t status =
        loom_serve_http_metrics_format(router->registry, output);
    if (iree_status_is_ok(status)) {
      loom_serve_http_response_send(router->server, connection,
                                    iree_string_builder_view(output));
    } else {
      loom_serve_http_connection_abort(connection);
    }
    return status;
  }
  if (get && (iree_string_view_equal(request->target, IREE_SV("/healthz")) ||
              iree_string_view_equal(request->target, IREE_SV("/readyz")))) {
    loom_serve_http_response_send(
        router->server, connection,
        IREE_SV("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                "Connection: close\r\n\r\n{\"status\":\"ready\"}"));
    return iree_ok_status();
  }
  const bool include_status =
      iree_string_view_equal(request->target, IREE_SV("/v1/status"));
  if (get && (include_status ||
              iree_string_view_equal(request->target, IREE_SV("/v1/models")))) {
    iree_string_builder_reset(output);
    iree_status_t status = iree_string_builder_append_cstring(
        output,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        "Connection: close\r\n\r\n{\"object\":\"list\",\"data\":[");
    bool first = true;
    for (iree_host_size_t i = 0;
         i < router->capacity && iree_status_is_ok(status); ++i) {
      const loom_serve_registered_model_t* model =
          loom_serve_router_model_at(router, i);
      if (!model->service.name.size ||
          (!include_status && (model->phase != LOOM_SERVE_REGISTRATION_READY ||
                               model->removal != LOOM_SERVE_REMOVAL_NONE))) {
        continue;
      }
      if (!first) {
        status = iree_string_builder_append_cstring(output, ",");
      }
      first = false;
      if (iree_status_is_ok(status)) {
        status =
            loom_serve_router_describe_service(model, include_status, output);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, "]");
    }
    if (iree_status_is_ok(status) && include_status) {
      status = iree_string_builder_append_cstring(output, ",\"server\":");
      if (iree_status_is_ok(status)) {
        status = loom_serve_router_describe_server(router, output);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, "}");
    }
    if (iree_status_is_ok(status)) {
      loom_serve_http_response_send(router->server, connection,
                                    iree_string_builder_view(output));
    } else {
      loom_serve_http_connection_abort(connection);
    }
    return status;
  }
  const bool checkpoint = iree_string_view_starts_with(
      request->target, IREE_SV("/v1/checkpoints/"));
  const bool drain =
      iree_string_view_equal(request->target, IREE_SV("/v1/models/drain"));
  const bool undrain =
      iree_string_view_equal(request->target, IREE_SV("/v1/models/undrain"));
  const bool unregister =
      iree_string_view_equal(request->target, IREE_SV("/v1/models/unregister"));
  const bool deactivate =
      iree_string_view_equal(request->target, IREE_SV("/v1/models/deactivate"));
  const bool activate =
      iree_string_view_equal(request->target, IREE_SV("/v1/models/activate"));
  const bool pin =
      iree_string_view_equal(request->target, IREE_SV("/v1/models/pin"));
  const bool unpin =
      iree_string_view_equal(request->target, IREE_SV("/v1/models/unpin"));
  const bool post = iree_string_view_equal(request->method, IREE_SV("POST"));
  if ((!checkpoint && !post) ||
      (checkpoint && !post &&
       !iree_string_view_equal(request->method, IREE_SV("DELETE")))) {
    return loom_serve_http_response_reject(
        router->server, connection, 404, "Not Found",
        IREE_SV("unknown service endpoint"), output);
  }
  if (post &&
      iree_string_view_equal(request->target, IREE_SV("/v1/models/register"))) {
    return loom_serve_router_register(router, connection, request);
  }
  if (post &&
      iree_string_view_equal(request->target, IREE_SV("/v1/devices/trim"))) {
    return loom_serve_router_trim(router, connection, request);
  }
  iree_string_view_t name;
  loom_serve_removal_t removal = LOOM_SERVE_REMOVAL_DRAIN;
  loom_serve_model_component_t component = LOOM_SERVE_MODEL_COMPONENT_PRIMARY;
  if (checkpoint) {
    if (!loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Model"),
                                               &name)) {
      if (service_count != 1) {
        return loom_serve_http_response_reject(
            router->server, connection, 400, "Bad Request",
            IREE_SV("checkpoint operations require X-Loom-Model"), output);
      }
      for (iree_host_size_t i = 0; i < router->capacity; ++i) {
        const iree_string_view_t candidate =
            loom_serve_router_model_at(router, i)->service.name;
        if (candidate.size) {
          name = candidate;
          break;
        }
      }
    }
  } else {
    iree_status_t status = loom_serve_router_model(
        request,
        drain || undrain || unregister || deactivate || activate || pin ||
            unpin,
        unregister ? &removal : NULL,
        activate || deactivate || pin || unpin ? &component : NULL, output);
    if (!iree_status_is_ok(status)) {
      if (!iree_status_is_invalid_argument(status) &&
          !iree_status_is_out_of_range(status)) {
        loom_serve_http_connection_abort(connection);
        return status;
      }
      loom_serve_http_server_record_error(router->server, status);
      iree_status_free(status);
      return loom_serve_http_response_reject(
          router->server, connection, 400, "Bad Request",
          IREE_SV("invalid model selector or request body"), output);
    }
    name = iree_string_builder_view(output);
  }
  for (iree_host_size_t i = 0; i < router->capacity; ++i) {
    loom_serve_registered_model_t* state =
        loom_serve_router_model_at(router, i);
    if (state->service.name.size &&
        iree_string_view_equal(name, state->service.name)) {
      if (unregister) {
        iree_status_t status = loom_serve_registry_remove(state, removal);
        if (!iree_status_is_ok(status)) {
          return loom_serve_http_response_error(router->server, connection,
                                                status, &router->output);
        }
        return loom_serve_router_control(router, i, connection, 202);
      }
      if (pin || unpin) {
        iree_status_t status =
            loom_serve_registry_set_pinned(state, component, pin);
        if (!iree_status_is_ok(status)) {
          return loom_serve_http_response_error(router->server, connection,
                                                status, &router->output);
        }
        return loom_serve_router_control(router, i, connection, 200);
      }
      if (deactivate || activate) {
        iree_status_t status =
            activate ? loom_serve_registry_activate(state, component)
                     : loom_serve_registry_deactivate(state, component);
        if (!iree_status_is_ok(status)) {
          return loom_serve_http_response_error(router->server, connection,
                                                status, &router->output);
        }
        return loom_serve_router_control(router, i, connection, 202);
      }
      if (drain || undrain) {
        if (undrain &&
            state->parameter_operation != LOOM_SERVE_PARAMETER_OPERATION_NONE) {
          return loom_serve_http_response_reject(
              router->server, connection, 409, "Conflict",
              IREE_SV("parameter operation is still pending"), output);
        }
        if (state->removal || (undrain && !iree_status_is_ok(state->status))) {
          return loom_serve_http_response_reject(
              router->server, connection, 409, "Conflict",
              IREE_SV("removing or failed registrations cannot be reopened"),
              output);
        }
        state->closed = drain;
        return loom_serve_router_control(router, i, connection, 200);
      }
      if (state->removal) {
        return loom_serve_http_response_reject(
            router->server, connection, 410, "Gone",
            IREE_SV("model is being unregistered"), output);
      }
      if (!iree_status_is_ok(state->status)) {
        return loom_serve_http_response_reject(
            router->server, connection, 503, "Service Unavailable",
            IREE_SV("model is unavailable; inspect /v1/status"), output);
      }
      if (!state->service.endpoint &&
          !iree_string_view_equal(request->target, IREE_SV("/tokenize")) &&
          !iree_string_view_equal(request->target, IREE_SV("/detokenize"))) {
        return loom_serve_http_response_reject(
            router->server, connection, 404, "Not Found",
            IREE_SV("model has no HTTP inference endpoint"), output);
      }
      const bool voice_removal =
          state->service.kind == LOOM_SERVE_MODEL_KIND_SPEECH &&
          iree_string_view_equal(request->target,
                                 IREE_SV("/v1/audio/voices/remove"));
      if (voice_removal ||
          (checkpoint && !post &&
           (state->service.kind == LOOM_SERVE_MODEL_KIND_TEXT ||
            state->service.kind == LOOM_SERVE_MODEL_KIND_SEQUENCE))) {
        // Deletion releases retained state without opening inference. The
        // owning modality validates and retires its retained references.
        if (state->phase != LOOM_SERVE_REGISTRATION_READY) {
          return loom_serve_http_response_reject(
              router->server, connection, 404, "Not Found",
              IREE_SV("unprepared model has no retained state"), output);
        }
        return loom_serve_router_admit(router, i, connection, request);
      }
      if (state->closed || state->held_count ||
          state->phase != LOOM_SERVE_REGISTRATION_READY) {
        if (router->held.count == router->held.capacity ||
            router->held.count + router->logs.count +
                    loom_serve_http_variants_claims(router->variants) ==
                router->logs.capacity) {
          return loom_serve_http_response_reject(
              router->server, connection, 429, "Too Many Requests",
              IREE_SV("model admission holding capacity is full"), output);
        }
        const bool demands_parameters =
            post && (iree_string_view_equal(
                         request->target,
                         iree_make_cstring_view(state->service.endpoint)) ||
                     (state->service.kind == LOOM_SERVE_MODEL_KIND_SPEECH &&
                      iree_string_view_equal(request->target,
                                             IREE_SV("/v1/audio/voices"))));
        router->held.values[router->held.count++] =
            (loom_serve_router_held_request_t){connection, request, i,
                                               demands_parameters};
        ++state->held_count;
        state->parameter_demand_count += demands_parameters;
        return iree_ok_status();
      }
      return loom_serve_router_admit(router, i, connection, request);
    }
  }
  return loom_serve_http_response_reject(
      router->server, connection, 404, "Not Found",
      IREE_SV("model is not registered"), output);
}

static void loom_serve_router_advance(loom_serve_router_t* router,
                                      bool* out_progress) {
  router->stopping |= loom_serve_http_server_is_stopping(router->server);
  router->status = iree_status_join(
      router->status, loom_serve_resources_advance(&router->resources));
  router->stopping |= !iree_status_is_ok(router->status);
  *out_progress = false;
  loom_serve_router_tokenizations_collect(router, out_progress);
  loom_serve_router_held_collect(router, out_progress);
  for (iree_host_size_t i = 0;
       i < router->request_capacity && !router->stopping; ++i) {
    const loom_serve_http_request_t* request = NULL;
    loom_serve_http_connection_t* connection =
        loom_serve_http_server_take_request(router->server, &request);
    if (!connection) {
      break;
    }
    router->status = loom_serve_router_accept(router, connection, request);
    router->stopping = !iree_status_is_ok(router->status) ||
                       loom_serve_http_server_is_stopping(router->server);
    *out_progress = true;
  }
  for (iree_host_size_t i = 0; i < router->capacity; ++i) {
    const iree_host_size_t index = (router->cursor + i) % router->capacity;
    loom_serve_registered_model_t* model =
        loom_serve_router_model_at(router, index);
    if (!model->service.name.size) {
      continue;
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z_model, "loom_serve_model_advance");
    IREE_TRACE_ZONE_APPEND_TEXT(z_model, model->service.name.data,
                                model->service.name.size);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_model, model->incarnation);
    router->stopping |= loom_serve_http_server_is_stopping(router->server);
    if (router->stopping && model->incarnation &&
        model->removal != LOOM_SERVE_REMOVAL_ABORT) {
      router->status = iree_status_join(
          router->status,
          loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
    }
    if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
      const loom_serve_service_t* service = &model->service;
      bool progress = false;
      const loom_serve_service_flags_t flags =
          loom_serve_router_service_flags(router, index) |
          loom_serve_registry_service_flags(model) |
          (router->stopping ? LOOM_SERVE_SERVICE_FLAG_STOPPING : 0);
      router->status = iree_status_join(
          router->status, service->advance(service->self, flags, &progress));
      router->stopping |= !iree_status_is_ok(router->status);
      *out_progress |= progress;
    }
    bool variant_progress = false;
    router->status = iree_status_join(
        router->status, loom_serve_http_variants_advance(
                            router->variants, index, router->stopping,
                            &router->output, &variant_progress));
    router->stopping |= !iree_status_is_ok(router->status);
    *out_progress |= variant_progress;
    bool progress = false;
    router->status = iree_status_join(
        router->status,
        loom_serve_registry_advance(
            router->registry, index,
            (loom_serve_wake_callback_t){loom_serve_router_wake, router},
            &progress));
    router->stopping |= !iree_status_is_ok(router->status);
    *out_progress |= progress;
    IREE_TRACE_ZONE_END(z_model);
  }
  if (++router->cursor >= router->capacity) {
    router->cursor = 0;
  }
  router->status = iree_status_join(
      router->status, loom_serve_router_logs_advance(router, out_progress));
  router->stopping |= !iree_status_is_ok(router->status);
  bool maintenance_progress = false;
  router->status = iree_status_join(
      router->status,
      loom_serve_maintenance_advance(&router->maintenance, router->stopping,
                                     &maintenance_progress));
  router->stopping |= !iree_status_is_ok(router->status);
  *out_progress |= maintenance_progress;
  if (router->stopping) {
    loom_serve_router_held_collect(router, out_progress);
  }
  bool pending = router->tokenizations != NULL || router->held.count != 0 ||
                 router->logs.count != 0 ||
                 loom_serve_http_variants_pending(router->variants);
  pending |= loom_serve_resources_is_pending(&router->resources);
  pending |= loom_serve_maintenance_is_pending(&router->maintenance);
  for (iree_host_size_t i = 0; i < router->capacity; ++i) {
    loom_serve_registered_model_t* model =
        loom_serve_router_model_at(router, i);
    if (router->stopping && model->incarnation &&
        model->removal != LOOM_SERVE_REMOVAL_ABORT) {
      router->status = iree_status_join(
          router->status,
          loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
      *out_progress = true;
    }
    pending |= loom_serve_registry_pending(model) ||
               (router->stopping && model->incarnation &&
                model->phase != LOOM_SERVE_REGISTRATION_RELEASE_FAILED);
  }
  if (router->stopping && !pending) {
    loom_serve_maintenance_set_observer(&router->maintenance,
                                        (loom_serve_wake_callback_t){0});
    loom_serve_resources_set_observer(&router->resources,
                                      (loom_serve_wake_callback_t){0});
    for (iree_host_size_t i = 0; i < router->capacity; ++i) {
      const loom_serve_registered_model_t* model =
          loom_serve_router_model_at(router, i);
      if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
        model->service.set_observer(model->service.self,
                                    (loom_serve_wake_callback_t){0});
      }
    }
    loom_serve_http_server_set_observer(router->server,
                                        (loom_serve_wake_callback_t){0});
    router->completed = true;
  }
}

static iree_status_t loom_serve_router_drain(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_router_t* router = process->user_data;
  if (!iree_slim_mutex_try_lock(&router->mutex)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!router->completed) {
    loom_serve_router_advance(router, &result->did_work);
  }
  result->completed = router->completed;
  iree_slim_mutex_unlock(&router->mutex);
  IREE_TRACE_ZONE_END(z0);
  // Service failures initiate draining, not premature process completion.
  return iree_ok_status();
}

static void loom_serve_router_released(iree_task_process_t* process) {
  loom_serve_router_t* router = process->user_data;
  iree_slim_mutex_lock(&router->retirement.mutex);
  router->retirement.released = true;
  iree_notification_post(&router->retirement.notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&router->retirement.mutex);
}

static bool loom_serve_router_is_released(void* user_data) {
  loom_serve_router_t* router = user_data;
  iree_slim_mutex_lock(&router->retirement.mutex);
  const bool released = router->retirement.released;
  iree_slim_mutex_unlock(&router->retirement.mutex);
  return released;
}

iree_status_t loom_serve_router_run(iree_task_executor_t* executor,
                                    loom_serve_http_server_t* server,
                                    loom_serve_registry_t* registry,
                                    loom_serve_router_options_t options,
                                    iree_allocator_t host_allocator) {
  if (!options.request_quota ||
      options.holding_capacity >=
          loom_serve_http_server_connection_capacity(server)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "router requires a positive request quota and "
                            "holding capacity below transport capacity");
  }
  loom_serve_router_t router = {
      .host_allocator = host_allocator,
      .executor = executor,
      .server = server,
      .registry = registry,
      .resources = options.resources,
      .capacity = loom_serve_registry_capacity(registry),
      .request_capacity = options.request_quota,
      .held = {.capacity = options.holding_capacity},
      .logs = {.capacity =
                   loom_serve_http_server_connection_capacity(server) - 1}};
  iree_status_t status = loom_serve_maintenance_initialize(
      options.resources.device_group, options.idle, &router.maintenance);
  if (iree_status_is_ok(status) && options.holding_capacity) {
    status = iree_allocator_malloc_array(
        host_allocator, options.holding_capacity, sizeof(*router.held.values),
        (void**)&router.held.values);
  }
  if (iree_status_is_ok(status) && router.logs.capacity) {
    status = iree_allocator_malloc_array(host_allocator, router.logs.capacity,
                                         sizeof(*router.logs.values),
                                         (void**)&router.logs.values);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_variants_allocate(server, registry, host_allocator,
                                               &router.variants);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_maintenance_deinitialize(&router.maintenance);
    iree_allocator_free(host_allocator, router.logs.values);
    iree_allocator_free(host_allocator, router.held.values);
    return status;
  }
  iree_slim_mutex_initialize(&router.mutex);
  iree_slim_mutex_initialize(&router.retirement.mutex);
  iree_notification_initialize(&router.retirement.notification);
  iree_string_builder_initialize(host_allocator, &router.output);
  iree_task_process_initialize(loom_serve_router_drain, 0, 1, &router.process);
  iree_task_process_set_flags(&router.process,
                              IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  router.process.user_data = &router;
  router.process.release_fn = loom_serve_router_released;
  loom_serve_maintenance_set_observer(
      &router.maintenance,
      (loom_serve_wake_callback_t){loom_serve_router_wake, &router});
  loom_serve_resources_set_observer(
      &router.resources,
      (loom_serve_wake_callback_t){loom_serve_router_wake, &router});
  for (iree_host_size_t i = 0; i < router.capacity; ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_router_model_at(&router, i);
    if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
      model->service.set_observer(
          model->service.self,
          (loom_serve_wake_callback_t){loom_serve_router_wake, &router});
    }
  }
  loom_serve_http_server_set_observer(
      server, (loom_serve_wake_callback_t){loom_serve_router_wake, &router});
  iree_notification_await(&router.retirement.notification,
                          loom_serve_router_is_released, &router,
                          iree_infinite_timeout());
  iree_string_builder_deinitialize(&router.output);
  loom_serve_maintenance_deinitialize(&router.maintenance);
  loom_serve_http_variants_free(router.variants);
  iree_allocator_free(host_allocator, router.logs.values);
  iree_allocator_free(host_allocator, router.held.values);
  iree_notification_deinitialize(&router.retirement.notification);
  iree_slim_mutex_deinitialize(&router.retirement.mutex);
  iree_slim_mutex_deinitialize(&router.mutex);
  return router.status;
}
