// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/router.h"

#include <stdio.h>

#include "experimental/loom_serve/http/tokenization.h"
#include "iree/base/internal/json.h"
#include "iree/base/threading/mutex.h"
#include "loom/util/json.h"

typedef struct router_tokenization_t {
  // Owner-only list link, bounded by the transport's connection credit.
  struct router_tokenization_t* next;
  // Independent finite CPU process, never waiting on model or device work.
  iree_task_process_t process;
  // Borrowed tokenizer, immutable and live through final process release.
  const iree_tokenizer_t* tokenizer;
  // Claim keeping the request JSON alive while CPU work uses its views.
  loom_serve_http_connection_t* connection;
  // Borrowed immutable request, live until the owner sends or aborts.
  const loom_serve_http_request_t* request;
  // Completed HTTP bytes copied by transport before this job is freed.
  iree_string_builder_t output;
  // Allocator owning this job and independent tokenizer scratch.
  iree_allocator_t allocator;
  // Nonblocking router wake, valid until all accepted jobs retire.
  loom_serve_http_observer_t observer;
  // Joins release publication's final access before owner destruction.
  iree_slim_mutex_t mutex;
  // Owned outcome, transferred by completion and consumed after release.
  iree_status_t status;
  // The executor and its callbacks have relinquished job storage.
  bool released;
} router_tokenization_t;

typedef struct router_t {
  // Allocator for independently owned utility jobs and routing output.
  iree_allocator_t allocator;
  // Shared application workers, borrowed through final process release.
  iree_task_executor_t* executor;
  // Transport owning network readiness and claimed request storage.
  loom_serve_http_server_t* server;
  // Immutable borrowed service interfaces, valid through run return.
  const loom_serve_http_service_t* services;
  // Number of registered services.
  iree_host_size_t service_count;
  // Maximum claims per turn, independent of model batch sizes.
  iree_host_size_t request_capacity;
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
  // Routing JSON scratch, used only by the exclusive turn owner.
  iree_string_builder_t output;
  // Accepted utility jobs, each holding one transport claim until CPU release.
  router_tokenization_t* tokenizations;
  // Final process retirement, independent of drain completion.
  struct {
    // Joins the release callback's last access before stack storage dies.
    iree_slim_mutex_t mutex;
    // Wakes only the outer application caller, never a task worker.
    iree_notification_t notification;
    // The executor has relinquished all process and drain storage.
    bool released;
  } retirement;
} router_t;

static void router_wake(void* user_data) {
  router_t* router = user_data;
  iree_task_executor_schedule_process(router->executor, &router->process);
}

static iree_status_t router_quote(iree_string_builder_t* output,
                                  iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static void router_send(loom_serve_http_connection_t* connection,
                        iree_string_view_t response) {
  iree_status_t status = loom_serve_http_connection_send(connection, response);
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
  } else {
    fprintf(stderr, "Router peer failed: ");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    loom_serve_http_connection_abort(connection);
  }
}

static iree_status_t router_reject(loom_serve_http_connection_t* connection,
                                   int code, const char* reason,
                                   iree_string_view_t message,
                                   iree_string_builder_t* output) {
  iree_string_builder_reset(output);
  iree_status_t status = iree_string_builder_append_format(
      output,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n{\"error\":{\"message\":",
      code, reason);
  if (iree_status_is_ok(status)) {
    status = router_quote(output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output, ",\"type\":\"loom_route_error\"}}");
  }
  if (iree_status_is_ok(status)) {
    router_send(connection, iree_string_builder_view(output));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t router_tokenization_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  (void)context;
  router_tokenization_t* job = process->user_data;
  result->did_work = true;
  result->completed = true;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      &job->output,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n"));
  if (iree_string_view_equal(job->request->target, IREE_SV("/tokenize"))) {
    return loom_serve_http_tokenize(job->tokenizer, job->request->body,
                                    &job->output, job->allocator);
  }
  return loom_serve_http_detokenize(job->tokenizer, job->request->body,
                                    &job->output, job->allocator);
}

static void router_tokenization_completed(iree_task_process_t* process,
                                          iree_status_t status) {
  router_tokenization_t* job = process->user_data;
  iree_slim_mutex_lock(&job->mutex);
  job->status = status;
  iree_slim_mutex_unlock(&job->mutex);
}

static void router_tokenization_released(iree_task_process_t* process) {
  router_tokenization_t* job = process->user_data;
  iree_slim_mutex_lock(&job->mutex);
  job->released = true;
  job->observer.wake(job->observer.user_data);
  iree_slim_mutex_unlock(&job->mutex);
}

static iree_status_t router_tokenization_accept(
    router_t* router, const iree_tokenizer_t* tokenizer,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request) {
  if (!tokenizer) {
    return router_reject(connection, 404, "Not Found",
                         IREE_SV("model has no tokenizer capability"),
                         &router->output);
  }
  router_tokenization_t* job = NULL;
  iree_status_t status =
      iree_allocator_malloc(router->allocator, sizeof(*job), (void**)&job);
  if (!iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    return status;
  }
  job->allocator = router->allocator;
  job->tokenizer = tokenizer;
  job->connection = connection;
  job->request = request;
  job->observer = (loom_serve_http_observer_t){router_wake, router};
  iree_slim_mutex_initialize(&job->mutex);
  iree_string_builder_initialize(router->allocator, &job->output);
  iree_task_process_initialize(router_tokenization_run, 0, 1, &job->process);
  job->process.user_data = job;
  job->process.completion_fn = router_tokenization_completed;
  job->process.release_fn = router_tokenization_released;
  job->next = router->tokenizations;
  router->tokenizations = job;
  iree_task_executor_schedule_process(router->executor, &job->process);
  return iree_ok_status();
}

static void router_tokenizations_collect(router_t* router, bool* out_progress) {
  router_tokenization_t** link = &router->tokenizations;
  while (*link) {
    router_tokenization_t* job = *link;
    iree_slim_mutex_lock(&job->mutex);
    const bool released = job->released;
    iree_slim_mutex_unlock(&job->mutex);
    if (!released) {
      link = &job->next;
      continue;
    }
    *out_progress = true;
    *link = job->next;
    iree_status_t status = job->status;
    if (!iree_status_is_ok(status)) {
      const bool invalid = iree_status_is_invalid_argument(status);
      const bool too_large = iree_status_is_out_of_range(status);
      iree_string_view_t message = iree_status_message(status);
      if (!message.size) {
        message = IREE_SV("tokenization failed");
      }
      iree_status_t response_status = iree_ok_status();
      if (router->stopping) {
        loom_serve_http_connection_abort(job->connection);
      } else {
        response_status = router_reject(job->connection,
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
      fprintf(stderr, "Tokenization request failed: ");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      status = response_status;
    } else if (router->stopping) {
      loom_serve_http_connection_abort(job->connection);
    } else {
      router_send(job->connection, iree_string_builder_view(&job->output));
    }
    router->status = iree_status_join(router->status, status);
    router->stopping |= !iree_status_is_ok(router->status);
    iree_string_builder_deinitialize(&job->output);
    iree_slim_mutex_deinitialize(&job->mutex);
    iree_allocator_free(router->allocator, job);
  }
}

static iree_status_t router_model_visit(void* user_data, iree_string_view_t key,
                                        iree_json_value_type_t type,
                                        iree_string_view_t value) {
  iree_string_view_t* model = user_data;
  if (!iree_string_view_equal(key, IREE_SV("model"))) {
    return iree_ok_status();
  }
  if (model->data || type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model must occur exactly once as a string");
  }
  *model = value;
  return iree_ok_status();
}

static iree_status_t router_model(const loom_serve_http_request_t* request,
                                  iree_string_builder_t* output) {
  iree_string_view_t cursor = request->body;
  iree_string_view_t object, model = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing request JSON");
  }
  IREE_RETURN_IF_ERROR(
      iree_json_enumerate_object_typed(object, router_model_visit, &model));
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

static loom_serve_http_service_flags_t router_service_flags(
    iree_host_size_t selected, iree_host_size_t service_count,
    const loom_serve_http_service_t* services) {
  for (iree_host_size_t i = 0; i < service_count; ++i) {
    if (i != selected && services[i].is_active(services[i].self)) {
      return LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY;
    }
  }
  return 0;
}

static iree_status_t router_describe_service(
    const loom_serve_http_service_t* service, bool include_status,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, "{\"object\":\"model\",\"id\":"));
  IREE_RETURN_IF_ERROR(router_quote(output, service->name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"loom\":{\"kind\":"));
  IREE_RETURN_IF_ERROR(
      router_quote(output, iree_make_cstring_view(service->kind)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"endpoint\":"));
  IREE_RETURN_IF_ERROR(
      router_quote(output, iree_make_cstring_view(service->endpoint)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"tokenize\":%s", service->tokenizer ? "true" : "false"));
  if (include_status) {
    const loom_serve_http_service_snapshot_t state =
        service->snapshot(service->self);
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"status\":{\"phase\":"));
    IREE_RETURN_IF_ERROR(
        router_quote(output, iree_make_cstring_view(state.phase)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",\"queued_requests\":%zu,\"active_requests\":%zu}",
        state.queued_requests, state.active_requests));
  }
  return iree_string_builder_append_cstring(output, "}}");
}

static iree_status_t router_accept(router_t* router,
                                   loom_serve_http_connection_t* connection,
                                   const loom_serve_http_request_t* request) {
  const iree_host_size_t service_count = router->service_count;
  const loom_serve_http_service_t* services = router->services;
  iree_string_builder_t* output = &router->output;
  const bool get = iree_string_view_equal(request->method, IREE_SV("GET"));
  if (get && (iree_string_view_equal(request->target, IREE_SV("/healthz")) ||
              iree_string_view_equal(request->target, IREE_SV("/readyz")))) {
    router_send(connection,
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
    for (iree_host_size_t i = 0; i < service_count && iree_status_is_ok(status);
         ++i) {
      if (i) {
        status = iree_string_builder_append_cstring(output, ",");
      }
      if (iree_status_is_ok(status)) {
        status = router_describe_service(&services[i], include_status, output);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, "]}");
    }
    if (iree_status_is_ok(status)) {
      router_send(connection, iree_string_builder_view(output));
    } else {
      loom_serve_http_connection_abort(connection);
    }
    return status;
  }
  const bool checkpoint = iree_string_view_starts_with(
      request->target, IREE_SV("/v1/checkpoints/"));
  const bool post = iree_string_view_equal(request->method, IREE_SV("POST"));
  if ((!checkpoint && !post) ||
      (checkpoint && !post &&
       !iree_string_view_equal(request->method, IREE_SV("DELETE")))) {
    return router_reject(connection, 404, "Not Found",
                         IREE_SV("unknown service endpoint"), output);
  }
  iree_string_view_t name;
  if (checkpoint) {
    if (!loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Model"),
                                               &name)) {
      if (service_count != 1) {
        return router_reject(
            connection, 400, "Bad Request",
            IREE_SV("checkpoint operations require X-Loom-Model"), output);
      }
      name = services[0].name;
    }
  } else {
    iree_status_t status = router_model(request, output);
    if (!iree_status_is_ok(status)) {
      if (!iree_status_is_invalid_argument(status) &&
          !iree_status_is_out_of_range(status)) {
        loom_serve_http_connection_abort(connection);
        return status;
      }
      fprintf(stderr, "Invalid route request: ");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      return router_reject(connection, 400, "Bad Request",
                           IREE_SV("invalid model selector or request JSON"),
                           output);
    }
    name = iree_string_builder_view(output);
  }
  for (iree_host_size_t i = 0; i < service_count; ++i) {
    if (iree_string_view_equal(name, services[i].name)) {
      if (post &&
          (iree_string_view_equal(request->target, IREE_SV("/tokenize")) ||
           iree_string_view_equal(request->target, IREE_SV("/detokenize")))) {
        return router_tokenization_accept(router, services[i].tokenizer,
                                          connection, request);
      }
      return services[i].accept(
          services[i].self, connection, request,
          router_service_flags(i, service_count, services));
    }
  }
  return router_reject(connection, 404, "Not Found",
                       IREE_SV("model is not registered"), output);
}

static void router_advance(router_t* router, bool* out_progress) {
  router->stopping |= loom_serve_http_server_is_stopping(router->server);
  *out_progress = false;
  router_tokenizations_collect(router, out_progress);
  for (iree_host_size_t i = 0;
       i < router->request_capacity && !router->stopping; ++i) {
    const loom_serve_http_request_t* request = NULL;
    loom_serve_http_connection_t* connection =
        loom_serve_http_server_take_request(router->server, &request);
    if (!connection) {
      break;
    }
    router->status = router_accept(router, connection, request);
    router->stopping = !iree_status_is_ok(router->status) ||
                       loom_serve_http_server_is_stopping(router->server);
    *out_progress = true;
  }
  for (iree_host_size_t i = 0; i < router->service_count; ++i) {
    const iree_host_size_t index = (router->cursor + i) % router->service_count;
    const loom_serve_http_service_t* service = &router->services[index];
    router->stopping |= loom_serve_http_server_is_stopping(router->server);
    if (router->stopping && !service->is_pending(service->self)) {
      continue;
    }
    bool progress = false;
    const loom_serve_http_service_flags_t flags =
        router_service_flags(index, router->service_count, router->services) |
        (router->stopping ? LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING : 0);
    router->status = iree_status_join(
        router->status, service->advance(service->self, flags, &progress));
    router->stopping |= !iree_status_is_ok(router->status);
    *out_progress |= progress;
  }
  router->cursor = (router->cursor + 1) % router->service_count;
  bool pending = router->tokenizations != NULL;
  for (iree_host_size_t i = 0; i < router->service_count; ++i) {
    pending |= router->services[i].is_pending(router->services[i].self);
  }
  if (router->stopping && !pending) {
    for (iree_host_size_t i = 0; i < router->service_count; ++i) {
      router->services[i].set_observer(router->services[i].self,
                                       (loom_serve_http_observer_t){0});
    }
    loom_serve_http_server_set_observer(router->server,
                                        (loom_serve_http_observer_t){0});
    router->completed = true;
  }
}

static iree_status_t router_drain(iree_task_process_t* process,
                                  const iree_task_worker_context_t* context,
                                  iree_task_process_drain_result_t* result) {
  (void)context;
  router_t* router = process->user_data;
  if (!iree_slim_mutex_try_lock(&router->mutex)) {
    return iree_ok_status();
  }
  if (!router->completed) {
    router_advance(router, &result->did_work);
  }
  result->completed = router->completed;
  iree_slim_mutex_unlock(&router->mutex);
  // Service failures initiate draining, not premature process completion.
  return iree_ok_status();
}

static void router_released(iree_task_process_t* process) {
  router_t* router = process->user_data;
  iree_slim_mutex_lock(&router->retirement.mutex);
  router->retirement.released = true;
  iree_notification_post(&router->retirement.notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&router->retirement.mutex);
}

static bool router_is_released(void* user_data) {
  router_t* router = user_data;
  iree_slim_mutex_lock(&router->retirement.mutex);
  const bool released = router->retirement.released;
  iree_slim_mutex_unlock(&router->retirement.mutex);
  return released;
}

iree_status_t loom_serve_http_router_run(
    iree_task_executor_t* executor, loom_serve_http_server_t* server,
    iree_host_size_t service_count, const loom_serve_http_service_t* services,
    iree_host_size_t request_capacity, iree_allocator_t host_allocator) {
  if (!service_count || !request_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "router requires services and a positive request quota");
  }
  for (iree_host_size_t i = 0; i < service_count; ++i) {
    bool valid_name = services[i].name.size != 0;
    for (iree_host_size_t j = 0; j < services[i].name.size; ++j) {
      const uint8_t c = (uint8_t)services[i].name.data[j];
      valid_name &= c >= 0x21 && c <= 0x7e;
    }
    if (!valid_name) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "service names must be nonempty visible ASCII");
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(services[i].name, services[j].name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate service name");
      }
    }
  }
  router_t router = {.allocator = host_allocator,
                     .executor = executor,
                     .server = server,
                     .services = services,
                     .service_count = service_count,
                     .request_capacity = request_capacity};
  iree_slim_mutex_initialize(&router.mutex);
  iree_slim_mutex_initialize(&router.retirement.mutex);
  iree_notification_initialize(&router.retirement.notification);
  iree_string_builder_initialize(host_allocator, &router.output);
  iree_task_process_initialize(router_drain, 0, 1, &router.process);
  iree_task_process_set_flags(&router.process,
                              IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  router.process.user_data = &router;
  router.process.release_fn = router_released;
  for (iree_host_size_t i = 0; i < service_count; ++i) {
    services[i].set_observer(
        services[i].self, (loom_serve_http_observer_t){router_wake, &router});
  }
  loom_serve_http_server_set_observer(
      server, (loom_serve_http_observer_t){router_wake, &router});
  iree_notification_await(&router.retirement.notification, router_is_released,
                          &router, iree_infinite_timeout());
  iree_string_builder_deinitialize(&router.output);
  iree_notification_deinitialize(&router.retirement.notification);
  iree_slim_mutex_deinitialize(&router.retirement.mutex);
  iree_slim_mutex_deinitialize(&router.mutex);
  return router.status;
}
