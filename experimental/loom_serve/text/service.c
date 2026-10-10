// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/service.h"

#include <string.h>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/host.h"
#include "experimental/loom_serve/scheduling/packing.h"
#include "experimental/loom_serve/text/chat.h"
#include "experimental/loom_serve/text/heartbeat.h"
#include "experimental/loom_serve/text/schedule.h"
#include "experimental/loom_serve/text/session.h"
#include "experimental/loom_serve/text/telemetry.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/mutex.h"
#include "iree/task/executor.h"
#include "loom/util/json.h"

// Exclusive device or CPU work retained until successful readiness/release.
typedef enum text_work_e {
  TEXT_WORK_NONE,
  TEXT_WORK_EPOCH,
  TEXT_WORK_CHECKPOINT,
  TEXT_WORK_TRIM,
  TEXT_WORK_RESERVE,
  TEXT_WORK_PREPARE,
  TEXT_WORK_OUTPUT,
} text_work_t;

typedef enum text_checkpoint_action_e {
  TEXT_CHECKPOINT_PIN,
  TEXT_CHECKPOINT_SUSPEND,
  TEXT_CHECKPOINT_RELEASE,
} text_checkpoint_action_t;

// Explicit local name owns a native endpoint and its source-canonical history.
// Active branches own independent state; only queued admissions borrow this.
typedef struct text_checkpoint_t {
  // Bounded service-local identifier, published only after pin readiness.
  char name[65];
  // Owned native immutable state, released before the borrowed model dies.
  loom_serve_text_checkpoint_t* state;
  // Owned canonical VM transcript, independent of the originating session.
  loom_serve_text_chat_completion_t completion;
  // Exact consumed endpoint, captured with the native pin.
  iree_host_size_t position;
  // Exact selected-but-unconsumed token from that endpoint.
  int32_t token;
  // Source framing classification for the pending token.
  loom_serve_text_chat_boundary_t boundary;
  // Queued readers preventing deletion or reuse of this record.
  iree_host_size_t pending;
} text_checkpoint_t;

// A queued request owns no device reservation. The FIFO head may retain an idle
// slot's host input array; body/name/tool views borrow the immutable HTTP
// claim.
typedef struct text_pending_request_t {
  // Claimed transport identity, relinquished by admission, rejection or abort.
  loom_serve_http_connection_t* connection;
  // Raw JSON retained until source rendering completes.
  iree_string_view_t body;
  // Canonical prompt/tool descriptors; null policy means not yet rendered.
  loom_serve_text_chat_t chat;
  // Validated local session key borrowing the claimed request headers.
  iree_string_view_t name;
  // Borrowed explicit endpoint through admission or queue cancellation.
  text_checkpoint_t* checkpoint;
  // Identity and time retained through rejection, cancellation or admission.
  loom_serve_text_receipt_t receipt;
} text_pending_request_t;

struct loom_serve_text_service_t {
  // Allocator for the service's host-only storage.
  iree_allocator_t allocator;
  // Borrowed model residency and its single host execution owner.
  loom_serve_text_model_t* model;
  // Borrowed deployment name, independent of the source policy identity.
  iree_string_view_t name;
  // Owned fallback shape for isolated execution.
  loom_serve_packing_shape_t isolated_shape;
  // Number of allocated device/session rows.
  iree_host_size_t row_count;
  // Active input tokens issued in one scheduling turn.
  iree_host_size_t chunk_size;
  // Compiled context capacity of each retained row.
  iree_host_size_t context_capacity;
  // Output bound when the client omits one.
  iree_host_size_t default_max_tokens;
  // Same ready partition can execute packed or through isolated controls.
  loom_serve_text_schedule_mode_t schedule_mode;
  // Admission ablation independent of the packed versus isolated math choice.
  loom_serve_text_packing_mode_t packing_mode;
  // Fixed proposal depth: zero for target-only, three for whole verifiers.
  iree_host_size_t mtp_depth;
  // Maximum epochs submitted before checking transport and new arrivals.
  iree_host_size_t continuation_epochs;
  // Borrowed model shapes, or the one isolated control shape.
  const loom_serve_packing_shape_t* shapes;
  // Number of candidate shapes evaluated against each ready cohort.
  iree_host_size_t shape_count;
  // Rotating admission priority, independent of row or request identity.
  iree_host_size_t cursor;
  // Proactor-timed reporting of copied state, independent of model readiness.
  loom_serve_text_heartbeat_t heartbeat;
  // Completion publication only; model mutation remains on the router owner.
  struct {
    // Owner-only operation discriminator; NONE means no callback is accepted.
    text_work_t work;
    // Joins the callback's final storage access before completion consumption.
    iree_slim_mutex_t mutex;
    // Borrowed shared host proactor outliving the service.
    iree_async_proactor_t* proactor;
    // Borrowed router wake target, attached only while the router owns turns.
    loom_serve_http_observer_t observer;
    // One accepted ALL wait, reused after callback retirement.
    iree_async_semaphore_wait_operation_t operation;
    // Exact work and feedback values captured by submission.
    loom_serve_execution_frontier_t frontier;
    // Toll-free handles for asynchronous readiness registration.
    iree_async_semaphore_t* semaphores[2];
    // Callback-published result, consumed together with ready under mutex.
    iree_status_t status;
    // Callback has published its final result and wake.
    bool ready;
  } completion;
  // One finite CPU borrower, exclusive with model invocation and device work.
  struct {
    // Shared host workers; no model-specific worker population.
    iree_task_executor_t* executor;
    // Reused only after its release callback publishes readiness.
    iree_task_process_t process;
  } cpu;
  // Immutable cohort plan through asynchronous submission and publication.
  struct {
    // Number of planned spans; zero outside an epoch.
    iree_host_size_t count;
    // Selected command-program bucket.
    iree_host_size_t shape_index;
    // Current span in isolated/matched execution.
    iree_host_size_t cursor;
    // First issued epoch identity and host-observed start time.
    struct {
      // First issued epoch identity, including device-fed continuations.
      uint64_t serial;
      // Host observation timestamp in nanoseconds.
      iree_time_t time;
    } start;
    // Number of proposed spans in the first epoch.
    iree_host_size_t proposal_count;
    // Number of queued target traversals for a packed cohort.
    iree_host_size_t device_epochs;
    // Known inputs scheduled in a device-fed continuation.
    iree_host_size_t continued_prefill_count;
    // Row association, live token counts and selection flags.
    loom_serve_text_span_t spans[LOOM_SERVE_TEXT_ROW_CAPACITY];
    // Per-span progress filled only after successful readiness.
    loom_serve_text_result_t results[LOOM_SERVE_TEXT_ROW_CAPACITY];
    // Stable pending tokens for later isolated span submissions.
    int32_t decode_tokens[LOOM_SERVE_TEXT_ROW_CAPACITY];
    // Pre-submission consumed positions for metrics.
    iree_host_size_t positions[LOOM_SERVE_TEXT_ROW_CAPACITY];
    // Request output credit, including any continuation.
    uint32_t output_limits[LOOM_SERVE_TEXT_ROW_CAPACITY];
  } epoch;
  // One bounded control claim, retained through any accepted model operation.
  struct {
    // Claimed HTTP ownership; NULL when no control request is queued.
    loom_serve_http_connection_t* connection;
    // Borrowed request storage, valid through response or abort.
    const loom_serve_http_request_t* request;
    // Original receipt retained across control queue re-entry.
    loom_serve_text_receipt_t receipt;
    // Validated operation, meaningful after submission.
    text_checkpoint_action_t action;
    // Public identifier borrowing the retained HTTP request.
    iree_string_view_t name;
    // Owned registry record, unpublished while a new pin is pending.
    text_checkpoint_t* checkpoint;
  } control;
  // Single-owner observation; CPU jobs never publish into this stream.
  loom_serve_text_telemetry_t telemetry;
  // Cold admission queue, independent of device slots and epoch span capacity.
  struct {
    // Fixed host queue storage; entries own their chat and HTTP claim.
    text_pending_request_t* values;
    // Number of queued entries in FIFO order.
    iree_host_size_t count;
    // Maximum queued request count, allocated once during initialization.
    iree_host_size_t capacity;
    // Only the FIFO head prepares input; no device credit is acquired here.
    struct {
      // Stable idle slot owned by preparation through head admission/removal.
      loom_serve_text_session_t* session;
      // Known input tokens in the selected slot's existing host array.
      iree_host_size_t input_count;
      // Consumed prefix shared with the selected continuation or endpoint.
      iree_host_size_t retained_count;
      // Page-rounded completion high-water, including speculative credit.
      iree_host_size_t reservation;
      // Cold image payload before the current reservation attempt.
      iree_host_size_t restored_bytes;
      // Host preparation duration returned by the exclusive CPU borrower.
      iree_duration_t duration;
    } prepared;
  } pending;
  // Request policy over the logical pool, not a second block allocator.
  struct {
    // Immutable logical geometry; availability is queried at observation.
    loom_serve_text_pool_usage_t geometry;
    // One maintenance cut, retained through model readiness.
    struct {
      // Retired rows released storage since the last completed cut.
      bool pending;
      // Host observation time before submitting maintenance.
      iree_time_t start_time;
      // Native relocation plan consumed only after successful readiness.
      loom_serve_text_trim_result_t result;
    } trim;
  } pool;
  // Fixed named endpoint registry, independent of active session row count.
  struct {
    // Owned records; live entries retain their native pin and canonical text.
    text_checkpoint_t* values;
    // Record count, equal to the native model's configured pin capacity.
    iree_host_size_t capacity;
    // Sum of cold native endpoint payloads, updated only at ownership changes.
    iree_host_size_t host_bytes;
  } checkpoints;
  // Shared JSON/SSE serialization scratch, copied into a row packet or carrier.
  iree_string_builder_t scratch;
  // Shared typed tool-call serialization scratch used at generation end.
  iree_string_builder_t tool_calls;
  // Fixed retained rows sharing one model, command set and execution timeline.
  loom_serve_text_session_t sessions[LOOM_SERVE_TEXT_ROW_CAPACITY];
};

static void text_work_ready(void* user_data, iree_async_operation_t* operation,
                            iree_status_t status,
                            iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_text_service_t* service = user_data;
  iree_slim_mutex_lock(&service->completion.mutex);
  service->completion.status = status;
  service->completion.ready = true;
  service->completion.observer.wake(service->completion.observer.user_data);
  iree_slim_mutex_unlock(&service->completion.mutex);
}

static iree_status_t text_await_work(loom_serve_text_service_t* service,
                                     text_work_t work) {
  iree_async_semaphore_wait_operation_t* wait = &service->completion.operation;
  memset(wait, 0, sizeof(*wait));
  iree_async_operation_initialize(
      &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, text_work_ready, service);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    service->completion.semaphores[i] =
        (iree_async_semaphore_t*)service->completion.frontier.semaphores[i];
  }
  wait->semaphores = service->completion.semaphores;
  wait->values = service->completion.frontier.values;
  wait->count = 2;
  wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_submit_one(
      service->completion.proactor, &wait->base));
  service->completion.work = work;
  return iree_ok_status();
}

static void text_cpu_completed(iree_task_process_t* process,
                               iree_status_t status) {
  loom_serve_text_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->completion.mutex);
  service->completion.status = status;
  iree_slim_mutex_unlock(&service->completion.mutex);
}

static void text_cpu_released(iree_task_process_t* process) {
  loom_serve_text_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->completion.mutex);
  service->completion.ready = true;
  service->completion.observer.wake(service->completion.observer.user_data);
  iree_slim_mutex_unlock(&service->completion.mutex);
}

static void text_schedule_cpu(loom_serve_text_service_t* service,
                              text_work_t work,
                              iree_task_process_drain_fn_t drain) {
  service->completion.work = work;
  iree_task_process_initialize(drain, 0, 1, &service->cpu.process);
  service->cpu.process.user_data = service;
  service->cpu.process.completion_fn = text_cpu_completed;
  service->cpu.process.release_fn = text_cpu_released;
  iree_task_executor_schedule_process(service->cpu.executor,
                                      &service->cpu.process);
}

static void text_observe(loom_serve_text_service_t* service,
                         const char* phase) {
  const loom_serve_text_pool_usage_t pool =
      loom_serve_text_model_pool_usage(service->model);
  iree_host_size_t active = 0, prefill = 0, decode = 0, backpressured = 0;
  iree_time_t oldest_active_arrival = IREE_TIME_INFINITE_FUTURE;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    const loom_serve_text_session_t* session = &service->sessions[i];
    if (session->request.connection) {
      ++active;
      const iree_time_t arrival =
          session->request.start_time - session->request.queue_duration;
      oldest_active_arrival = iree_min(oldest_active_arrival, arrival);
      prefill += session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL;
      decode += session->request.phase == LOOM_SERVE_TEXT_REQUEST_DECODE;
      backpressured +=
          iree_string_builder_size(&session->packet) &&
          !loom_serve_http_connection_can_send(session->request.connection);
    }
  }
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  loom_serve_text_heartbeat_snapshot_t* state = &service->heartbeat.snapshot;
  if (strcmp(state->phase, phase)) {
    state->phase = phase;
    state->phase_start = iree_time_now();
  }
  state->active_rows = active;
  state->queued_requests = service->pending.count;
  state->oldest_active_arrival = oldest_active_arrival;
  state->oldest_queued_arrival =
      service->pending.count ? service->pending.values[0].receipt.arrival_time
                             : 0;
  state->pool.capacity = pool.capacity;
  state->pool.reserved = pool.pending;
  state->pool.resident = pool.capacity - pool.available - pool.pending;
  state->checkpoint_host_bytes = service->checkpoints.host_bytes;
  state->state_memory = loom_serve_text_model_memory_statistics(service->model);
  state->weight_memory =
      loom_serve_text_model_weight_statistics(service->model);
  state->weight_activity =
      loom_serve_text_model_weight_activity(service->model);
  state->device_memory = loom_serve_device_memory_statistics(
      loom_serve_text_model_device(service->model));
  state->prefill_rows = prefill;
  state->decode_rows = decode;
  state->backpressured_rows = backpressured;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
}

static iree_status_t text_quote(iree_string_builder_t* output,
                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

// Error translation is an intentional status ownership boundary. Original
// failures have already been copied for owner publication; failure to construct
// the response propagates.
static iree_status_t text_error_json(iree_status_t failure,
                                     iree_string_builder_t* output) {
  iree_status_t status =
      iree_string_builder_append_cstring(output, "{\"error\":{\"message\":");
  if (iree_status_is_ok(status)) {
    iree_string_view_t message = iree_status_message(failure);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(failure)));
    }
    status = text_quote(output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output, ",\"type\":\"loom_request_error\"}}");
  }
  iree_status_free(failure);
  return status;
}

static iree_status_t text_reject(loom_serve_text_service_t* service,
                                 loom_serve_http_connection_t* connection,
                                 loom_serve_text_receipt_t receipt, int code,
                                 const char* reason, iree_status_t failure) {
  // Input rejection may overlap a CPU output job using model scratch.
  loom_serve_text_telemetry_end(&service->telemetry, receipt,
                                LOOM_SERVE_TELEMETRY_REJECTED, 0, failure);
  iree_string_builder_t response;
  iree_string_builder_initialize(service->allocator, &response);
  iree_status_t status = iree_string_builder_append_format(
      &response,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      code, reason);
  if (iree_status_is_ok(status)) {
    status = text_error_json(failure, &response);
  } else {
    status = iree_status_join(status, failure);
  }
  if (iree_status_is_ok(status)) {
    iree_status_t send_status = loom_serve_http_connection_send(
        connection, iree_string_builder_view(&response));
    if (iree_status_is_ok(send_status)) {
      loom_serve_http_connection_finish(connection);
    } else {
      loom_serve_text_telemetry_diagnose(&service->telemetry, receipt.serial,
                                         send_status);
      loom_serve_http_connection_abort(connection);
    }
  } else {
    loom_serve_http_connection_abort(connection);
  }
  iree_string_builder_deinitialize(&response);
  return status;
}

static void text_request_release(loom_serve_text_service_t* service,
                                 loom_serve_text_session_t* session) {
  service->pool.trim.pending = true;
  loom_serve_text_session_release_request(session);
}

static void text_request_cancel(loom_serve_text_service_t* service,
                                loom_serve_text_session_t* session) {
  loom_serve_text_telemetry_request_end(
      &service->telemetry, session,
      service->telemetry.failure_code || session->request.diagnostic.code
          ? LOOM_SERVE_TELEMETRY_FAILED
          : LOOM_SERVE_TELEMETRY_CANCELLED);
  loom_serve_http_connection_abort(session->request.connection);
  loom_serve_text_chat_completion_deinitialize(&session->completion);
  text_request_release(service, session);
}

static void text_request_finish(loom_serve_text_service_t* service,
                                loom_serve_text_session_t* session) {
  loom_serve_text_telemetry_request_end(&service->telemetry, session,
                                        session->request.diagnostic.code
                                            ? LOOM_SERVE_TELEMETRY_FAILED
                                            : LOOM_SERVE_TELEMETRY_COMPLETED);
  loom_serve_http_connection_finish(session->request.connection);
  text_request_release(service, session);
}

static bool text_session_name_valid(iree_string_view_t name) {
  if (name.size > 64) {
    return false;
  }
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    const char c = name.data[i];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) {
      return false;
    }
  }
  return true;
}

static loom_serve_text_session_t* text_session_select(
    loom_serve_text_service_t* service, iree_string_view_t name) {
  loom_serve_text_session_t* oldest = NULL;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    loom_serve_text_session_t* session = &service->sessions[i];
    if (name.size &&
        iree_string_view_equal(name, iree_make_cstring_view(session->name))) {
      return session;
    }
    if (!session->request.connection &&
        (!oldest || session->serial < oldest->serial)) {
      oldest = session;
    }
  }
  return oldest;
}

static text_checkpoint_t* text_checkpoint_find(
    loom_serve_text_service_t* service, iree_string_view_t name) {
  for (iree_host_size_t i = 0; name.size && i < service->checkpoints.capacity;
       ++i) {
    text_checkpoint_t* checkpoint = &service->checkpoints.values[i];
    if (iree_string_view_equal(name,
                               iree_make_cstring_view(checkpoint->name))) {
      return checkpoint;
    }
  }
  return NULL;
}

static iree_status_t text_checkpoint_reply(
    loom_serve_text_service_t* service,
    loom_serve_http_connection_t* connection, loom_serve_text_receipt_t receipt,
    text_checkpoint_action_t action, iree_string_view_t name,
    iree_host_size_t position, iree_host_size_t host_bytes) {
  const bool create = action == TEXT_CHECKPOINT_PIN;
  const bool suspend = action == TEXT_CHECKPOINT_SUSPEND;
  const bool release = action == TEXT_CHECKPOINT_RELEASE;
  loom_serve_text_checkpoint_record_t value = {.position = position,
                                               .host_bytes = host_bytes};
  memcpy(value.name, name.data, name.size);
  loom_serve_text_record(service->telemetry.stream,
                         create    ? &LOOM_SERVE_TEXT_CHECKPOINT_PIN
                         : suspend ? &LOOM_SERVE_TEXT_CHECKPOINT_SUSPEND
                                   : &LOOM_SERVE_TEXT_CHECKPOINT_RELEASE,
                         receipt.serial, &value, sizeof(value));
  iree_string_builder_reset(&service->scratch);
  iree_status_t status = iree_string_builder_append_format(
      &service->scratch,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n"
      "{\"checkpoint\":\"%.*s\",\"position\":%zu,\"pinned\":%s,"
      "\"host_snapshot_bytes\":%zu}",
      create ? "201 Created" : "200 OK", (int)name.size, name.data, position,
      release ? "false" : "true", host_bytes);
  if (iree_status_is_ok(status)) {
    iree_status_t sent = loom_serve_http_connection_send(
        connection, iree_string_builder_view(&service->scratch));
    if (iree_status_is_ok(sent)) {
      loom_serve_text_telemetry_end(&service->telemetry, receipt,
                                    LOOM_SERVE_TELEMETRY_COMPLETED, 0, sent);
      loom_serve_http_connection_finish(connection);
    } else {
      loom_serve_text_telemetry_end(&service->telemetry, receipt,
                                    LOOM_SERVE_TELEMETRY_CANCELLED, 0, sent);
      iree_status_free(sent);
      loom_serve_http_connection_abort(connection);
    }
  } else {
    loom_serve_text_telemetry_end(&service->telemetry, receipt,
                                  LOOM_SERVE_TELEMETRY_FAILED, 0, status);
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t text_complete_checkpoint(
    loom_serve_text_service_t* service, loom_serve_http_service_flags_t flags) {
  text_checkpoint_t* checkpoint = service->control.checkpoint;
  const text_checkpoint_action_t action = service->control.action;
  const iree_string_view_t name = service->control.name;
  if (action == TEXT_CHECKPOINT_PIN) {
    memcpy(checkpoint->name, name.data, name.size);
    checkpoint->name[name.size] = 0;
  } else {
    loom_serve_text_checkpoint_complete_suspend(checkpoint->state);
    service->checkpoints.host_bytes +=
        loom_serve_text_checkpoint_suspended_bytes(checkpoint->state);
    service->pool.trim.pending = true;
  }
  loom_serve_http_connection_t* connection = service->control.connection;
  const loom_serve_text_receipt_t receipt = service->control.receipt;
  memset(&service->control, 0, sizeof(service->control));
  if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
    loom_serve_text_telemetry_end(&service->telemetry, receipt,
                                  LOOM_SERVE_TELEMETRY_CANCELLED, 0,
                                  iree_ok_status());
    loom_serve_http_connection_abort(connection);
    return iree_ok_status();
  }
  return text_checkpoint_reply(
      service, connection, receipt, action, name, checkpoint->position,
      loom_serve_text_checkpoint_suspended_bytes(checkpoint->state));
}

static iree_status_t text_checkpoint_request(
    loom_serve_text_service_t* service,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, loom_serve_text_receipt_t receipt,
    iree_string_view_t name) {
  const bool suspend =
      iree_string_view_consume_suffix(&name, IREE_SV("/suspend"));
  const bool post = iree_string_view_equal(request->method, IREE_SV("POST"));
  const bool create = post && !suspend;
  const bool release =
      !suspend && iree_string_view_equal(request->method, IREE_SV("DELETE"));
  if ((!create && !release && !(suspend && post)) || !name.size ||
      !text_session_name_valid(name) || request->body.size) {
    return text_reject(
        service, connection, receipt, 400, "Bad Request",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "use POST/DELETE /v1/checkpoints/NAME or POST "
                         "/v1/checkpoints/NAME/suspend with an empty body; "
                         "NAME is 1-64 ASCII identifier characters"));
  }
  text_checkpoint_t* checkpoint = text_checkpoint_find(service, name);
  if (create && checkpoint) {
    return text_reject(service, connection, receipt, 409, "Conflict",
                       iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                        "checkpoint name is already pinned"));
  }
  if (!create && !checkpoint) {
    return text_reject(
        service, connection, receipt, 404, "Not Found",
        iree_make_status(IREE_STATUS_NOT_FOUND, "checkpoint is not pinned"));
  }
  if (release && checkpoint->pending) {
    return text_reject(
        service, connection, receipt, 409, "Conflict",
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "checkpoint is borrowed by a queued request"));
  }
  if (create) {
    iree_string_view_t session_name;
    loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Session"),
                                          &session_name);
    if (!session_name.size || !text_session_name_valid(session_name)) {
      return text_reject(
          service, connection, receipt, 400, "Bad Request",
          iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pin requires a nonempty X-Loom-Session identifier"));
    }
    loom_serve_text_session_t* session =
        text_session_select(service, session_name);
    if (!session || !iree_string_view_equal(
                        session_name, iree_make_cstring_view(session->name))) {
      return text_reject(
          service, connection, receipt, 404, "Not Found",
          iree_make_status(IREE_STATUS_NOT_FOUND, "session is not retained"));
    }
    bool busy = session->request.connection != NULL;
    for (iree_host_size_t i = 0; i < service->pending.count; ++i) {
      busy |=
          iree_string_view_equal(session_name, service->pending.values[i].name);
    }
    if (busy || !session->completion.storage) {
      return text_reject(
          service, connection, receipt, 409, "Conflict",
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "pin requires an idle completed session"));
    }
    for (iree_host_size_t i = 0; i < service->checkpoints.capacity; ++i) {
      if (!service->checkpoints.values[i].state) {
        checkpoint = &service->checkpoints.values[i];
        break;
      }
    }
    if (!checkpoint) {
      return text_reject(
          service, connection, receipt, 503, "Service Unavailable",
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "checkpoint capacity exhausted; release a pin or "
                           "configure checkpoint_capacity"));
    }
    iree_status_t status = loom_serve_text_row_submit_pin(
        session->row, &checkpoint->state, &service->completion.frontier);
    if (!iree_status_is_ok(status)) {
      loom_serve_text_telemetry_end(&service->telemetry, receipt,
                                    LOOM_SERVE_TELEMETRY_FAILED, 0, status);
      loom_serve_http_connection_abort(connection);
      return status;
    }
    if (!checkpoint->state) {
      return text_reject(service, connection, receipt, 503,
                         "Service Unavailable",
                         iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                          "native pin capacity exhausted"));
    }
    checkpoint->completion = session->completion;
    iree_vm_buffer_retain(checkpoint->completion.storage);
    checkpoint->position = loom_serve_text_row_position(session->row);
    checkpoint->token = loom_serve_text_row_token(session->row);
    checkpoint->boundary = loom_serve_text_row_is_eos(session->row)
                               ? LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED
                               : LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN;
    service->control.connection = connection;
    service->control.request = request;
    service->control.receipt = receipt;
    service->control.action = TEXT_CHECKPOINT_PIN;
    service->control.name = name;
    service->control.checkpoint = checkpoint;
    return text_await_work(service, TEXT_WORK_CHECKPOINT);
  }
  const iree_host_size_t position = checkpoint->position;
  iree_host_size_t host_bytes =
      loom_serve_text_checkpoint_suspended_bytes(checkpoint->state);
  if (suspend) {
    if (!loom_serve_text_model_memory_statistics(service->model)
             .reserved_bytes) {
      return text_reject(
          service, connection, receipt, 400, "Bad Request",
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "checkpoint suspension requires elastic backing"));
    }
    if (host_bytes) {
      return text_checkpoint_reply(service, connection, receipt,
                                   TEXT_CHECKPOINT_SUSPEND, name, position,
                                   host_bytes);
    }
    text_observe(service, "checkpoint_suspend");
    iree_status_t status = loom_serve_text_checkpoint_submit_suspend(
        checkpoint->state, &service->completion.frontier);
    if (!iree_status_is_ok(status)) {
      loom_serve_text_telemetry_end(&service->telemetry, receipt,
                                    LOOM_SERVE_TELEMETRY_FAILED, 0, status);
      loom_serve_http_connection_abort(connection);
      return status;
    }
    service->control.connection = connection;
    service->control.request = request;
    service->control.receipt = receipt;
    service->control.action = TEXT_CHECKPOINT_SUSPEND;
    service->control.name = name;
    service->control.checkpoint = checkpoint;
    return text_await_work(service, TEXT_WORK_CHECKPOINT);
  }
  service->checkpoints.host_bytes -= host_bytes;
  loom_serve_text_checkpoint_release(checkpoint->state);
  loom_serve_text_chat_completion_deinitialize(&checkpoint->completion);
  memset(checkpoint, 0, sizeof(*checkpoint));
  service->pool.trim.pending = true;
  return text_checkpoint_reply(service, connection, receipt,
                               TEXT_CHECKPOINT_RELEASE, name, position, 0);
}

// Prepares host input without changing retained device state. Rejection leaves
// the previous checkpoint usable. The selected token belongs to the previous
// response but has not yet entered KV/recurrent, so it leads a retained append.
static iree_status_t text_prepare_input(loom_serve_text_service_t* service,
                                        loom_serve_text_session_t* session,
                                        const loom_serve_text_chat_t* chat,
                                        iree_string_view_t name,
                                        const text_checkpoint_t* pinned,
                                        iree_host_size_t* out_count,
                                        iree_host_size_t* out_retained) {
  const iree_string_view_t prompt = iree_string_builder_view(&chat->prompt);
  const iree_string_view_t checkpoint =
      pinned ? pinned->completion.transcript : session->completion.transcript;
  if (pinned && !iree_string_view_starts_with(prompt, checkpoint)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "request history does not extend the named checkpoint");
  }
  const bool retained =
      pinned ||
      (name.size && checkpoint.size &&
       iree_string_view_equal(name, iree_make_cstring_view(session->name)) &&
       iree_string_view_starts_with(prompt, checkpoint));
  *out_retained = pinned     ? pinned->position
                  : retained ? loom_serve_text_row_position(session->row)
                             : 0;
  iree_string_view_t input = prompt;
  loom_serve_text_chat_boundary_t boundary =
      LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH;
  if (retained) {
    session->tokens[0] =
        pinned ? pinned->token : loom_serve_text_row_token(session->row);
    boundary = pinned ? pinned->boundary
               : loom_serve_text_row_is_eos(session->row)
                   ? LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED
                   : LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN;
    input =
        iree_string_view_substr(prompt, checkpoint.size, IREE_HOST_SIZE_MAX);
  }
  const iree_host_size_t prefix_count = retained ? 1 : 0;
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_prepare_input(
      chat->policy, input, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED, boundary,
      service->context_capacity - prefix_count, session->tokens + prefix_count,
      out_count, service->allocator));
  *out_count += prefix_count;
  if (*out_count + chat->max_tokens - 1 >
      service->context_capacity - *out_retained) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "request input and output exceed remaining %zu-token context; compact "
        "history or lower max_tokens",
        service->context_capacity);
  }
  return iree_ok_status();
}

// Removal transfers no resources: the caller first releases the queued chat
// and connection, or transfers both into an admitted request.
static void text_pending_remove(loom_serve_text_service_t* service,
                                iree_host_size_t index) {
  if (!index) {
    loom_serve_residency_cancel_admission(
        loom_serve_text_model_residency(service->model));
    memset(&service->pending.prepared, 0, sizeof(service->pending.prepared));
  }
  if (service->pending.values[index].checkpoint) {
    --service->pending.values[index].checkpoint->pending;
  }
  --service->pending.count;
  memmove(service->pending.values + index, service->pending.values + index + 1,
          (service->pending.count - index) * sizeof(*service->pending.values));
}

static void text_pending_cancel(loom_serve_text_service_t* service,
                                iree_host_size_t index) {
  text_pending_request_t* request = &service->pending.values[index];
  loom_serve_text_telemetry_end(&service->telemetry, request->receipt,
                                service->telemetry.failure_code
                                    ? LOOM_SERVE_TELEMETRY_FAILED
                                    : LOOM_SERVE_TELEMETRY_CANCELLED,
                                0, iree_ok_status());
  loom_serve_text_chat_deinitialize(&request->chat);
  loom_serve_http_connection_abort(request->connection);
  text_pending_remove(service, index);
}

static void text_pending_cancel_failed(loom_serve_text_service_t* service,
                                       bool* out_progress) {
  for (iree_host_size_t i = 0; i < service->pending.count;) {
    if (loom_serve_http_connection_failed(
            service->pending.values[i].connection)) {
      text_pending_cancel(service, i);
      *out_progress = true;
    } else {
      ++i;
    }
  }
}

// Reclaim completed ownership before trying the next physical reservation.
// Compaction itself admits its temporary destinations and may leave a sparse
// map when moving would exceed the shared physical budget.
static iree_status_t text_submit_trim(loom_serve_text_service_t* service) {
  if (!service->pool.trim.pending) {
    return iree_ok_status();
  }
  text_observe(service, "trim");
  service->pool.trim.start_time = iree_time_now();
  IREE_RETURN_IF_ERROR(loom_serve_text_model_submit_trim(
      service->model, &service->pool.trim.result,
      &service->completion.frontier));
  return text_await_work(service, TEXT_WORK_TRIM);
}

static iree_status_t text_complete_trim(loom_serve_text_service_t* service) {
  loom_serve_text_trim_result_t* result = &service->pool.trim.result;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_model_complete_trim(service->model, result));
  service->pool.trim.pending = false;
  const loom_serve_memory_statistics_t memory =
      loom_serve_text_model_memory_statistics(service->model);
  const loom_serve_text_trim_record_t value = {
      .duration_ns = iree_time_now() - service->pool.trim.start_time,
      .moved_blocks = result->moved_blocks,
      .copied_bytes = result->copied_bytes,
      .released_bytes = result->released_bytes,
      .committed_bytes = memory.committed_bytes};
  loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_TRIM, 0,
                         &value, sizeof(value));
  return iree_ok_status();
}

static void text_reset_session(loom_serve_text_service_t* service,
                               loom_serve_text_session_t* session) {
  loom_serve_text_telemetry_session_end(&service->telemetry, session);
  loom_serve_text_row_reset_retired(session->row);
  service->pool.trim.pending = true;
  session->name[0] = 0;
  session->serial = 0;
  loom_serve_text_chat_completion_deinitialize(&session->completion);
}

typedef enum text_admission_result_e {
  // Physical parameters or logical/physical state cannot currently fit.
  TEXT_ADMISSION_CAPACITY = 0,
  // An older weight-pressure waiter must receive its next owner turn.
  TEXT_ADMISSION_QUEUED,
  // Parameter activation and state reservation have both completed.
  TEXT_ADMISSION_READY,
} text_admission_result_t;

// Native ownership knows the source geometry and exact physical slab union;
// the service only decides which idle histories may yield. Selected history,
// named endpoints and active reservations remain protected on every denial.
static iree_status_t text_reclaim_idle(loom_serve_text_service_t* service,
                                       loom_serve_text_session_t* selected,
                                       const text_checkpoint_t* checkpoint,
                                       iree_host_size_t retained,
                                       iree_host_size_t reservation,
                                       text_admission_result_t* out_result) {
  *out_result = TEXT_ADMISSION_CAPACITY;
  const loom_serve_text_reserve_mode_t mode =
      checkpoint ? LOOM_SERVE_TEXT_RESERVE_CHECKPOINT
      : retained ? LOOM_SERVE_TEXT_RESERVE_CONTINUE
                 : LOOM_SERVE_TEXT_RESERVE_FRESH;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    status = text_submit_trim(service);
    if (!iree_status_is_ok(status) ||
        service->completion.work != TEXT_WORK_NONE) {
      break;
    }
    loom_serve_residency_admission_t admission;
    loom_serve_execution_frontier_t frontier;
    status = loom_serve_text_row_submit_reserve(
        selected->row, mode, checkpoint ? checkpoint->state : NULL, reservation,
        &admission, &frontier);
    if (iree_status_is_ok(status) &&
        admission == LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED) {
      *out_result = TEXT_ADMISSION_QUEUED;
      break;
    }
    if (iree_status_is_ok(status) &&
        admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
      // The previous frontier was consumed successfully before this call. An
      // unchanged pair needs no new wait; warm denial must not wake itself.
      const loom_serve_execution_frontier_t previous =
          service->completion.frontier;
      const bool complete = previous.semaphores[0] == frontier.semaphores[0] &&
                            previous.semaphores[1] == frontier.semaphores[1] &&
                            previous.values[0] == frontier.values[0] &&
                            previous.values[1] == frontier.values[1];
      service->completion.frontier = frontier;
      if (complete) {
        if (loom_serve_text_row_complete_reserve(selected->row)) {
          *out_result = TEXT_ADMISSION_READY;
        }
      } else {
        status = text_await_work(service, TEXT_WORK_RESERVE);
        break;
      }
    }
    if (!iree_status_is_ok(status) || *out_result == TEXT_ADMISSION_READY) {
      break;
    }
    loom_serve_text_session_t* oldest = NULL;
    for (iree_host_size_t i = 0; i < service->row_count; ++i) {
      loom_serve_text_session_t* session = &service->sessions[i];
      if (session != selected && !session->request.connection &&
          loom_serve_text_row_position(session->row) &&
          (!oldest || session->serial < oldest->serial)) {
        oldest = session;
      }
    }
    if (!oldest) {
      break;
    }
    text_reset_session(service, oldest);
  }
  return status;
}

// The selected slot stays idle until this head admits or leaves the FIFO.
// Compaction and checkpoint suspension preserve the logical input frontier.
static iree_status_t text_prepare_admission(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  (void)context;
  result->completed = true;
  result->did_work = true;
  loom_serve_text_service_t* service = process->user_data;
  loom_serve_text_session_t* session = service->pending.prepared.session;
  text_pending_request_t* pending = &service->pending.values[0];
  const iree_time_t start = iree_time_now();
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_initialize(
      loom_serve_text_model_chat_policy(service->model), service->name,
      pending->body, service->default_max_tokens, service->allocator,
      &pending->chat));
  iree_host_size_t input_count = 0, retained_count = 0;
  IREE_RETURN_IF_ERROR(text_prepare_input(service, session, &pending->chat,
                                          pending->name, pending->checkpoint,
                                          &input_count, &retained_count));
  const iree_host_size_t reservation = loom_serve_text_request_reservation(
      service->context_capacity, retained_count + input_count,
      pending->chat.max_tokens, service->mtp_depth,
      service->pool.geometry.block_size);
  if (service->pool.geometry.capacity &&
      reservation > service->pool.geometry.capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "request needs %zu page-rounded KV positions including completion "
        "and speculative credit, but the shared pool holds %zu; compact "
        "history, lower max_tokens, or increase pool_capacity",
        reservation, service->pool.geometry.capacity);
  }
  service->pending.prepared.input_count = input_count;
  service->pending.prepared.retained_count = retained_count;
  service->pending.prepared.reservation = reservation;
  service->pending.prepared.duration = iree_time_now() - start;
  return iree_ok_status();
}

static iree_status_t text_complete_preparation(
    loom_serve_text_service_t* service, loom_serve_http_service_flags_t flags,
    iree_status_t status) {
  text_pending_request_t* pending = &service->pending.values[0];
  const bool cancelled =
      iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING) ||
      loom_serve_http_connection_failed(pending->connection);
  if (!iree_status_is_ok(status)) {
    if (!iree_status_is_invalid_argument(status) &&
        !iree_status_is_out_of_range(status)) {
      return status;
    }
    if (cancelled) {
      loom_serve_text_telemetry_diagnose(&service->telemetry,
                                         pending->receipt.serial, status);
      text_pending_cancel(service, 0);
      return iree_ok_status();
    }
    loom_serve_http_connection_t* connection = pending->connection;
    const loom_serve_text_receipt_t receipt = pending->receipt;
    loom_serve_text_chat_deinitialize(&pending->chat);
    text_pending_remove(service, 0);
    return text_reject(service, connection, receipt, 400, "Bad Request",
                       status);
  }
  if (cancelled) {
    text_pending_cancel(service, 0);
  } else {
    const loom_serve_text_admission_record_t value = {
        .row = service->pending.prepared.session - service->sessions,
        .retained_tokens = service->pending.prepared.retained_count,
        .appended_tokens = service->pending.prepared.input_count,
        .reservation_tokens = service->pending.prepared.reservation,
        .max_tokens = pending->chat.max_tokens,
        .prepare_ns = service->pending.prepared.duration};
    loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_PREPARED,
                           pending->receipt.serial, &value, sizeof(value));
  }
  return iree_ok_status();
}

static iree_status_t text_admit_request(loom_serve_text_service_t* service,
                                        loom_serve_http_service_flags_t flags) {
  text_pending_request_t* pending = &service->pending.values[0];
  loom_serve_text_session_t* session = service->pending.prepared.session;
  const iree_host_size_t input_count = service->pending.prepared.input_count;
  const iree_host_size_t retained_count =
      service->pending.prepared.retained_count;
  const iree_host_size_t reservation = service->pending.prepared.reservation;
  const iree_host_size_t restored_bytes =
      service->pending.prepared.restored_bytes;
  service->checkpoints.host_bytes -= restored_bytes;
  if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING) ||
      loom_serve_http_connection_failed(pending->connection)) {
    text_reset_session(service, session);
    text_pending_cancel(service, 0);
    return iree_ok_status();
  }
  const text_pending_request_t request = *pending;
  const iree_duration_t prepare_duration = service->pending.prepared.duration;
  text_pending_remove(service, 0);
  if (!request.name.size ||
      !iree_string_view_equal(request.name,
                              iree_make_cstring_view(session->name))) {
    loom_serve_text_telemetry_session_end(&service->telemetry, session);
  }
  if (request.name.size) {
    memcpy(session->name, request.name.data, request.name.size);
  }
  session->name[request.name.size] = 0;
  session->serial = request.receipt.serial;
  if (!session->telemetry.id) {
    session->telemetry.id = session->serial;
    loom_serve_text_session_record_t value = {
        .session = session->telemetry.id, .row = session - service->sessions};
    memcpy(value.name, session->name, sizeof(value.name));
    loom_serve_text_record(service->telemetry.stream,
                           &LOOM_SERVE_TEXT_SESSION_BEGIN, session->serial,
                           &value, sizeof(value));
  }
  ++session->telemetry.counters.requests;
  session->request.counters.requests = 1;
  session->request.connection = request.connection;
  session->request.chat = request.chat;
  session->request.phase = LOOM_SERVE_TEXT_REQUEST_PREFILL;
  session->request.input_count = input_count;
  session->request.retained_count = retained_count;
  session->request.start_time = iree_time_now();
  session->request.queue_duration =
      session->request.start_time - request.receipt.arrival_time;
  session->request.reserved_tokens = reservation;
  iree_status_t status = loom_serve_text_session_begin(
      session, loom_serve_text_model_tokenizer(service->model), service->name);
  if (iree_status_is_ok(status)) {
    const loom_serve_text_pool_usage_t pool =
        loom_serve_text_model_pool_usage(service->model);
    const loom_serve_text_admission_record_t value = {
        .session = session->telemetry.id,
        .row = session - service->sessions,
        .retained_tokens = retained_count,
        .appended_tokens = input_count,
        .reservation_tokens = reservation,
        .restored_bytes = restored_bytes,
        .max_tokens = request.chat.max_tokens,
        .prepare_ns = prepare_duration,
        .queue_ns = session->request.queue_duration,
        .pool_reserved_tokens = pool.pending,
        .pool_resident_tokens = pool.capacity - pool.available - pool.pending};
    loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_ADMIT,
                           session->serial, &value, sizeof(value));
  }
  return status;
}

static iree_status_t text_complete_reservation(
    loom_serve_text_service_t* service, loom_serve_http_service_flags_t flags) {
  if (loom_serve_text_row_complete_reserve(
          service->pending.prepared.session->row)) {
    return text_admit_request(service, flags);
  }
  if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING) ||
      loom_serve_http_connection_failed(
          service->pending.values[0].connection)) {
    text_pending_cancel(service, 0);
  }
  return iree_ok_status();
}

static iree_status_t text_admit_pending(loom_serve_text_service_t* service,
                                        loom_serve_http_service_flags_t flags) {
  iree_status_t status = iree_ok_status();
  while (service->pending.count && iree_status_is_ok(status)) {
    text_pending_request_t* pending = &service->pending.values[0];
    if (!service->pending.prepared.session) {
      loom_serve_text_session_t* session =
          text_session_select(service, pending->name);
      if (!session) {
        break;
      }
      service->pending.prepared.session = session;
      text_observe(service, "prepare");
      text_schedule_cpu(service, TEXT_WORK_PREPARE, text_prepare_admission);
      break;
    }
    loom_serve_text_session_t* session = service->pending.prepared.session;
    const iree_host_size_t retained_count =
        service->pending.prepared.retained_count;
    const iree_host_size_t reservation = service->pending.prepared.reservation;
    service->pending.prepared.restored_bytes =
        pending->checkpoint ? loom_serve_text_checkpoint_suspended_bytes(
                                  pending->checkpoint->state)
                            : 0;
    text_admission_result_t admission;
    status = text_reclaim_idle(service, session, pending->checkpoint,
                               retained_count, reservation, &admission);
    if (!iree_status_is_ok(status) ||
        service->completion.work != TEXT_WORK_NONE) {
      break;
    }
    if (admission == TEXT_ADMISSION_QUEUED) {
      break;
    }
    if (admission == TEXT_ADMISSION_CAPACITY) {
      bool active =
          iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY);
      for (iree_host_size_t i = 0; i < service->row_count; ++i) {
        active |= service->sessions[i].request.connection != NULL;
      }
      if (active) {
        break;
      }
      loom_serve_http_connection_t* connection = pending->connection;
      const loom_serve_text_receipt_t receipt = pending->receipt;
      loom_serve_text_chat_deinitialize(&pending->chat);
      text_pending_remove(service, 0);
      status = text_reject(
          service, connection, receipt, 503, "Service Unavailable",
          iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "retained state leaves insufficient completion capacity; "
              "release a checkpoint, lower max_tokens, or increase "
              "pool_capacity or memory_bytes"));
      continue;
    }
    status = text_admit_request(service, flags);
  }
  return status;
}

static iree_status_t text_enqueue(loom_serve_text_service_t* service,
                                  loom_serve_http_connection_t* connection,
                                  const loom_serve_http_request_t* request,
                                  loom_serve_text_receipt_t receipt) {
  iree_string_view_t checkpoint_name = request->target;
  if (iree_string_view_consume_prefix(&checkpoint_name,
                                      IREE_SV("/v1/checkpoints/"))) {
    if (service->completion.work != TEXT_WORK_NONE) {
      if (service->control.connection) {
        return text_reject(service, connection, receipt, 503,
                           "Service Unavailable",
                           iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "model control queue is full"));
      }
      service->control.connection = connection;
      service->control.request = request;
      service->control.receipt = receipt;
      return iree_ok_status();
    }
    return text_checkpoint_request(service, connection, request, receipt,
                                   checkpoint_name);
  }
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(request->target,
                              IREE_SV("/v1/chat/completions"))) {
    return text_reject(
        service, connection, receipt, 404, "Not Found",
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "use /v1/chat/completions or /v1/checkpoints/NAME"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("Content-Type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return text_reject(
        service, connection, receipt, 415, "Unsupported Media Type",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Content-Type must be application/json"));
  }
  iree_string_view_t name;
  loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Session"),
                                        &name);
  if (!text_session_name_valid(name)) {
    return text_reject(
        service, connection, receipt, 400, "Bad Request",
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "X-Loom-Session must be at most 64 ASCII identifier characters"));
  }
  text_checkpoint_t* checkpoint = NULL;
  if (loom_serve_http_request_lookup_header(
          request, IREE_SV("X-Loom-Checkpoint"), &checkpoint_name)) {
    if (!checkpoint_name.size || !text_session_name_valid(checkpoint_name)) {
      return text_reject(
          service, connection, receipt, 400, "Bad Request",
          iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "X-Loom-Checkpoint requires a nonempty ASCII identifier"));
    }
    checkpoint = text_checkpoint_find(service, checkpoint_name);
    if (!checkpoint) {
      return text_reject(
          service, connection, receipt, 404, "Not Found",
          iree_make_status(IREE_STATUS_NOT_FOUND, "checkpoint is not pinned"));
    }
  }
  loom_serve_text_session_t* session = text_session_select(service, name);
  bool busy = session && session->request.connection;
  for (iree_host_size_t i = 0; name.size && i < service->pending.count; ++i) {
    busy |= iree_string_view_equal(name, service->pending.values[i].name);
  }
  if (busy) {
    return text_reject(
        service, connection, receipt, 409, "Conflict",
        iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                         "session already has an active or queued request"));
  }
  if (service->pending.count == service->pending.capacity) {
    return text_reject(service, connection, receipt, 503, "Service Unavailable",
                       iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                        "pending request queue is full"));
  }
  if (checkpoint) {
    ++checkpoint->pending;
  }
  service->pending.values[service->pending.count++] = (text_pending_request_t){
      .connection = connection,
      .body = request->body,
      .name = name,
      .checkpoint = checkpoint,
      .receipt = receipt,
  };
  return iree_ok_status();
}

static iree_status_t text_selected_tokens(loom_serve_text_service_t* service,
                                          loom_serve_text_session_t* session,
                                          iree_host_size_t count,
                                          const int32_t* tokens) {
  iree_status_t status = loom_serve_text_session_append_tokens(
      session, service->name, count, tokens, &service->scratch,
      &service->tool_calls);
  if (!iree_status_is_ok(status)) {
    // Device work completed, but translation cannot establish a client history.
    // Preserve prior streamed text and terminate with a real SSE error.
    loom_serve_text_chat_completion_deinitialize(&session->completion);
    loom_serve_telemetry_diagnostic_capture(status,
                                            &session->request.diagnostic);
    iree_string_builder_reset(&session->packet);
    iree_status_t packet_status =
        iree_string_builder_append_cstring(&session->packet, "data: ");
    if (iree_status_is_ok(packet_status)) {
      packet_status = text_error_json(status, &session->packet);
    } else {
      packet_status = iree_status_join(packet_status, status);
    }
    if (iree_status_is_ok(packet_status)) {
      packet_status = iree_string_builder_append_cstring(
          &session->packet, "\n\ndata: [DONE]\n\n");
    }
    session->request.finish_reason = "error";
    session->request.phase = LOOM_SERVE_TEXT_REQUEST_FINISHING;
    status = packet_status;
  }
  return status;
}

// Output credit and cancellation are settled before a row enters an epoch.
// The carrier owns a copied send and the row owns one staging packet. An empty
// staging packet is output credit even while the preceding send is in flight;
// waiting for that send would split an otherwise ready cohort. A full staging
// packet behind a busy carrier pauses only that row.
static iree_status_t text_prepare_ready(loom_serve_text_service_t* service,
                                        loom_serve_text_session_t* session,
                                        bool* out_progress,
                                        loom_serve_ready_span_t* out_ready) {
  *out_ready = (loom_serve_ready_span_t){0};
  if (!session->request.connection) {
    return iree_ok_status();
  }
  if (loom_serve_http_connection_failed(session->request.connection)) {
    text_request_cancel(service, session);
    *out_progress = true;
    service->pool.trim.pending = true;
    loom_serve_text_row_reset_retired(session->row);
    return iree_ok_status();
  }
  if (iree_string_builder_size(&session->packet)) {
    if (!loom_serve_http_connection_can_send(session->request.connection)) {
      return iree_ok_status();
    }
    *out_progress = true;
    iree_status_t status = loom_serve_http_connection_send(
        session->request.connection,
        iree_string_builder_view(&session->packet));
    if (!iree_status_is_ok(status)) {
      loom_serve_text_telemetry_diagnose(&service->telemetry, session->serial,
                                         status);
      text_request_cancel(service, session);
      service->pool.trim.pending = true;
      loom_serve_text_row_reset_retired(session->row);
      return iree_ok_status();
    }
    iree_string_builder_reset(&session->packet);
    if (session->request.phase == LOOM_SERVE_TEXT_REQUEST_FINISHING) {
      text_request_finish(service, session);
      return iree_ok_status();
    }
  }
  if (session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL) {
    *out_ready = (loom_serve_ready_span_t){
        session->request.input_count - session->request.input_offset, 1};
  } else {
    // A final single output cannot amortize drafting. Near the context limit,
    // ordinary decode avoids writing speculative KV beyond allocated capacity.
    const bool verify =
        service->mtp_depth &&
        session->request.chat.max_tokens - session->request.output_count > 1 &&
        service->context_capacity -
                loom_serve_text_row_position(session->row) >=
            4;
    *out_ready = verify ? (loom_serve_ready_span_t){4, 4}
                        : (loom_serve_ready_span_t){1, 1};
  }
  return iree_ok_status();
}

// The rotating first ready row chooses the phase; other rows of that phase
// can still batch together. Explicit request phase matters: a one-token prompt
// tail is not a decode. Credit and cancellation were settled before this
// filter.
static void text_separate_ready(const loom_serve_text_service_t* service,
                                loom_serve_ready_span_t* ready) {
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    const iree_host_size_t first = (service->cursor + i) % service->row_count;
    if (!ready[first].token_count) {
      continue;
    }
    const loom_serve_text_request_phase_t phase =
        service->sessions[first].request.phase;
    for (iree_host_size_t row = 0; row < service->row_count; ++row) {
      if (service->sessions[row].request.phase != phase) {
        ready[row] = (loom_serve_ready_span_t){0};
      }
    }
    return;
  }
}

// Pre-issues ready work from the admitted cohort. Known progress is exact;
// speculative progress is resolved by the device before the second traversal.
// Credit/cancellation remain bounded by the same host publication boundary.
static void text_prepare_continuation(
    loom_serve_text_service_t* service, iree_host_size_t count,
    const loom_serve_text_span_t* spans, const uint32_t* output_limits,
    loom_serve_text_span_t* next_spans, uint32_t* next_limits,
    loom_serve_text_continuation_t* out_continuation) {
  *out_continuation = (loom_serve_text_continuation_t){0};
  if (service->continuation_epochs != 2) {
    return;
  }
  loom_serve_ready_span_t ready[LOOM_SERVE_TEXT_ROW_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    const loom_serve_text_session_t* session =
        &service->sessions[span->row_index];
    const iree_host_size_t remaining_output =
        session->request.chat.max_tokens - session->request.output_count;
    const iree_host_size_t remaining_input =
        session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL
            ? session->request.input_count - session->request.input_offset -
                  span->token_count
            : 0;
    if (remaining_input) {
      ready[i] = (loom_serve_ready_span_t){remaining_input, 1};
    } else {
      const iree_host_size_t minimum_consumed =
          output_limits[i] ? 1 : span->token_count;
      if (remaining_output > 1 &&
          service->context_capacity -
                  loom_serve_text_row_position(session->row) >=
              minimum_consumed + 4) {
        ready[i] = (loom_serve_ready_span_t){4, 4};
      }
    }
  }
  // Rotate within the leased cohort instead of repeatedly giving its first
  // long prompt all of the expansion slots.
  iree_host_size_t cursor = count > 1 ? 1 : 0;
  if (service->packing_mode == LOOM_SERVE_TEXT_PACKING_SEPARATE) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      const iree_host_size_t first = (cursor + i) % count;
      if (!ready[first].token_count) {
        continue;
      }
      const iree_host_size_t minimum = ready[first].minimum_count;
      for (iree_host_size_t j = 0; j < count; ++j) {
        if (ready[j].minimum_count != minimum) {
          ready[j] = (loom_serve_ready_span_t){0};
        }
      }
      break;
    }
  }
  loom_serve_packed_span_t scheduled[LOOM_SERVE_TEXT_ROW_CAPACITY];
  loom_serve_packed_span_t scratch[LOOM_SERVE_TEXT_ROW_CAPACITY];
  iree_host_size_t shape = 0;
  const iree_host_size_t next_count = loom_serve_pack_shapes(
      count, ready, service->shape_count, service->shapes, service->chunk_size,
      &cursor, scheduled, scratch, &shape);
  for (iree_host_size_t i = 0; i < next_count; ++i) {
    const iree_host_size_t first = scheduled[i].row_index;
    const iree_host_size_t row = spans[first].row_index;
    const loom_serve_text_session_t* session = &service->sessions[row];
    const bool speculative = ready[first].minimum_count == 4;
    const bool select =
        speculative || scheduled[i].token_count == ready[first].token_count;
    next_spans[i] = (loom_serve_text_span_t){
        .row_index = row,
        .token_count = scheduled[i].token_count,
        .token_ids = speculative
                         ? NULL
                         : session->tokens + session->request.input_offset +
                               spans[first].token_count,
        .flags = (select ? LOOM_SERVE_TEXT_SPAN_FLAG_SELECT : 0) |
                 (speculative ? LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE : 0),
    };
    next_limits[i] = speculative
                         ? (uint32_t)iree_min(output_limits[first] ? 8 : 5,
                                              session->request.chat.max_tokens -
                                                  session->request.output_count)
                         : 0;
  }
  *out_continuation = (loom_serve_text_continuation_t){
      .shape_index = shape,
      .span_count = next_count,
      .spans = next_spans,
      .output_limits = next_limits,
  };
}

static iree_status_t text_submit_isolated(loom_serve_text_service_t* service) {
  const loom_serve_text_span_t* span =
      &service->epoch.spans[service->epoch.cursor];
  const loom_serve_text_session_t* session =
      &service->sessions[span->row_index];
  if (session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL ||
      service->schedule_mode == LOOM_SERVE_TEXT_SCHEDULE_MATCHED) {
    return loom_serve_text_row_submit_prefill(session->row, span->token_count,
                                              span->token_ids,
                                              &service->completion.frontier);
  }
  return loom_serve_text_row_submit_decode(session->row,
                                           &service->completion.frontier);
}

static iree_status_t text_submit_epoch(
    loom_serve_text_service_t* service, iree_host_size_t shape_index,
    iree_host_size_t count, const loom_serve_packed_span_t* scheduled) {
  memset(&service->epoch, 0, sizeof(service->epoch));
  service->epoch.count = count;
  service->epoch.shape_index = shape_index;
  loom_serve_text_span_t* spans = service->epoch.spans;
  int32_t* decode_tokens = service->epoch.decode_tokens;
  iree_host_size_t* positions = service->epoch.positions;
  uint32_t* output_limits = service->epoch.output_limits;
  iree_host_size_t proposal_count = 0;
  iree_host_size_t input_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t row = scheduled[i].row_index;
    loom_serve_text_session_t* session = &service->sessions[row];
    const bool prefill =
        session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL;
    const bool select = !prefill || scheduled[i].token_count ==
                                        session->request.input_count -
                                            session->request.input_offset;
    positions[i] = loom_serve_text_row_position(session->row);
    if (!prefill) {
      decode_tokens[i] = loom_serve_text_row_token(session->row);
      if (scheduled[i].token_count == 4) {
        ++proposal_count;
        output_limits[i] = (uint32_t)iree_min(
            4 * service->continuation_epochs,
            session->request.chat.max_tokens - session->request.output_count);
      }
    }
    spans[i] = (loom_serve_text_span_t){
        .row_index = row,
        .token_count = scheduled[i].token_count,
        .token_ids = prefill ? session->tokens + session->request.input_offset
                             : &decode_tokens[i],
        .flags = (select ? LOOM_SERVE_TEXT_SPAN_FLAG_SELECT : 0) |
                 (output_limits[i] ? LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE : 0),
    };
    input_count += spans[i].token_count;
  }
  loom_serve_text_span_t next_spans[LOOM_SERVE_TEXT_ROW_CAPACITY];
  uint32_t next_limits[LOOM_SERVE_TEXT_ROW_CAPACITY];
  loom_serve_text_continuation_t continuation;
  text_prepare_continuation(service, count, spans, output_limits, next_spans,
                            next_limits, &continuation);
  if (!continuation.span_count) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      output_limits[i] = iree_min(output_limits[i], 4u);
    }
  }
  iree_host_size_t continued_prefill_count = 0;
  for (iree_host_size_t i = 0; i < continuation.span_count; ++i) {
    continued_prefill_count += next_limits[i] ? 0 : next_spans[i].token_count;
  }
  const bool packed = service->schedule_mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED;
  const iree_host_size_t device_epochs = continuation.span_count ? 2 : 1;
  const bool verifies = proposal_count || continuation.span_count;
  text_observe(service, "execute");
  const iree_time_t start = iree_time_now();
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  loom_serve_text_heartbeat_snapshot_t* state = &service->heartbeat.snapshot;
  service->epoch.start.serial = state->issued_epochs + 1;
  service->epoch.start.time = start;
  service->epoch.proposal_count = proposal_count;
  service->epoch.device_epochs = device_epochs;
  service->epoch.continued_prefill_count = continued_prefill_count;
  state->issued_epochs += device_epochs;
  state->phase_start = start;
  state->epoch_spans = count;
  state->epoch_tokens = input_count;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
  iree_status_t status = iree_ok_status();
  if (packed) {
    status = loom_serve_text_model_submit_epoch(
        service->model, shape_index, count, spans,
        verifies ? output_limits : NULL,
        continuation.span_count ? &continuation : NULL,
        &service->completion.frontier);
  } else {
    status = text_submit_isolated(service);
  }
  IREE_RETURN_IF_ERROR(status);
  return text_await_work(service, TEXT_WORK_EPOCH);
}

static bool text_publish_epoch(loom_serve_text_service_t* service) {
  const iree_host_size_t count = service->epoch.count;
  const iree_host_size_t shape_index = service->epoch.shape_index;
  const loom_serve_text_span_t* spans = service->epoch.spans;
  loom_serve_text_result_t* results = service->epoch.results;
  const iree_host_size_t* positions = service->epoch.positions;
  const uint32_t* output_limits = service->epoch.output_limits;
  const iree_time_t start = service->epoch.start.time;
  const uint64_t epoch = service->epoch.start.serial;
  const iree_host_size_t proposal_count = service->epoch.proposal_count;
  const iree_host_size_t device_epochs = service->epoch.device_epochs;
  const iree_host_size_t continued_prefill_count =
      service->epoch.continued_prefill_count;
  const bool packed = service->schedule_mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED;
  const bool verifies = proposal_count || device_epochs == 2;
  iree_host_size_t prefill_count = 0, decode_count = 0, output_count = 0;
  iree_host_size_t verification_count = 0, accepted_drafts = 0;
  loom_serve_text_heartbeat_snapshot_t* state = &service->heartbeat.snapshot;
  const iree_time_t completed = iree_time_now();
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_text_session_t* session = &service->sessions[spans[i].row_index];
    const bool prefill =
        session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL;
    const bool select =
        iree_any_bit_set(spans[i].flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT);
    if (!verifies) {
      results[i].consumed_count = spans[i].token_count;
      results[i].known_count = spans[i].token_count;
      results[i].output_count = select ? 1 : 0;
      if (select) {
        results[i].tokens[0] = loom_serve_text_row_token(session->row);
      }
    }
    prefill_count += prefill ? results[i].known_count : 0;
    decode_count +=
        results[i].consumed_count - (prefill ? results[i].known_count : 0);
    output_count += results[i].output_count;
    verification_count += results[i].verification_count;
    accepted_drafts += results[i].consumed_count - results[i].known_count -
                       results[i].verification_count;
    session->request.counters.input_tokens += results[i].consumed_count;
    session->request.counters.output_tokens += results[i].output_count;
    ++session->request.counters.batches;
    session->telemetry.counters.input_tokens += results[i].consumed_count;
    session->telemetry.counters.output_tokens += results[i].output_count;
    ++session->telemetry.counters.batches;
    session->telemetry.position = loom_serve_text_row_position(session->row);
    session->telemetry.resident_tokens =
        loom_serve_text_row_pool_usage(session->row);
    const loom_serve_text_span_record_t value = {
        .epoch = epoch,
        .session = session->telemetry.id,
        .row = spans[i].row_index,
        .position = positions[i],
        .tokens = spans[i].token_count,
        .consumed_tokens = results[i].consumed_count,
        .output_tokens = results[i].output_count,
        .output_limit = output_limits[i],
        .known_tokens = results[i].known_count,
        .verification_epochs = results[i].verification_count,
        .prefill = prefill,
        .select = select};
    loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_SPAN,
                           session->serial, &value, sizeof(value));
  }
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  state->completed_epochs += device_epochs;
  state->traversals += packed ? device_epochs : count;
  state->prefill_tokens += prefill_count;
  state->decode_tokens += decode_count;
  state->model_duration += completed - start;
  state->mtp.proposed_tokens += verification_count * 3;
  state->mtp.accepted_inputs += accepted_drafts;
  state->last_completion = completed;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
  service->telemetry.counters.input_tokens += prefill_count + decode_count;
  service->telemetry.counters.output_tokens += output_count;
  ++service->telemetry.counters.batches;
  const loom_serve_text_epoch_record_t value = {
      .epoch = epoch,
      .spans = count,
      .shape = shape_index,
      .token_capacity = service->shapes[shape_index].token_capacity,
      .span_capacity = service->shapes[shape_index].span_capacity,
      .prefill_tokens = prefill_count,
      .decode_tokens = decode_count,
      .output_tokens = output_count,
      .traversals = packed ? device_epochs : count,
      .duration_ns = completed - start,
      .device_epochs = device_epochs,
      .continued_prefill_tokens = continued_prefill_count,
      .verifications = verification_count,
      .accepted_draft_inputs = accepted_drafts,
      .schedule_mode = service->schedule_mode,
      .packing_mode = service->packing_mode};
  loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_EPOCH, 0,
                         &value, sizeof(value));
  text_observe(service, "output");
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_text_session_t* session = &service->sessions[spans[i].row_index];
    const bool prefill =
        session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL;
    if (prefill) {
      session->request.input_offset += results[i].known_count;
      if (session->request.input_offset == session->request.input_count) {
        session->request.phase = LOOM_SERVE_TEXT_REQUEST_DECODE;
      }
    }
  }
  return output_count != 0;
}

static iree_status_t text_process_output(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  (void)context;
  result->completed = true;
  result->did_work = true;
  loom_serve_text_service_t* service = process->user_data;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->epoch.count && iree_status_is_ok(status); ++i) {
    const loom_serve_text_result_t* output = &service->epoch.results[i];
    if (output->output_count) {
      loom_serve_text_session_t* session =
          &service->sessions[service->epoch.spans[i].row_index];
      status = text_selected_tokens(service, session, output->output_count,
                                    output->tokens);
    }
  }
  return status;
}

static void text_complete_output(loom_serve_text_service_t* service) {
  for (iree_host_size_t i = 0; i < service->epoch.count; ++i) {
    loom_serve_text_session_t* session =
        &service->sessions[service->epoch.spans[i].row_index];
    if (session->request.diagnostic.code) {
      loom_serve_telemetry_record_t record = {
          .time_ns = iree_time_now(),
          .request = session->serial,
          .event = &LOOM_SERVE_TELEMETRY_ERROR,
          .data.error = session->request.diagnostic};
      loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
    }
    if (session->request.phase == LOOM_SERVE_TEXT_REQUEST_FINISHING) {
      loom_serve_text_row_release_reservation(session->row);
      service->pool.trim.pending = true;
    }
  }
  service->epoch.count = 0;
}

static iree_status_t text_complete_epoch(
    loom_serve_text_service_t* service, loom_serve_http_service_flags_t flags) {
  if (service->schedule_mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
    IREE_RETURN_IF_ERROR(loom_serve_text_model_complete_epoch(
        service->model, service->epoch.results));
  } else {
    const iree_host_size_t row =
        service->epoch.spans[service->epoch.cursor].row_index;
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_complete_step(service->sessions[row].row));
    ++service->epoch.cursor;
  }
  if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
    if (service->schedule_mode != LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
      // Only these isolated spans were submitted; the remainder never ran.
      service->epoch.count = service->epoch.cursor;
    }
    text_publish_epoch(service);
    service->epoch.count = 0;
    return iree_ok_status();
  }
  if (service->schedule_mode != LOOM_SERVE_TEXT_SCHEDULE_PACKED &&
      service->epoch.cursor < service->epoch.count) {
    IREE_RETURN_IF_ERROR(text_submit_isolated(service));
    return text_await_work(service, TEXT_WORK_EPOCH);
  }
  const bool has_output = text_publish_epoch(service);
  if (has_output) {
    text_schedule_cpu(service, TEXT_WORK_OUTPUT, text_process_output);
  } else {
    service->epoch.count = 0;
  }
  return iree_ok_status();
}

static iree_status_t text_service_initialize(
    loom_serve_text_service_t* service, loom_serve_telemetry_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_serve_text_telemetry_initialize(
      stream, &service->telemetry, service->allocator));
  loom_serve_host_t* host =
      loom_serve_device_host(loom_serve_text_model_device(service->model));
  service->cpu.executor = loom_serve_host_executor(host);
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(host), 0, &service->completion.proactor));
  service->checkpoints.capacity =
      loom_serve_text_model_checkpoint_capacity(service->model);
  if (service->checkpoints.capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        service->allocator, service->checkpoints.capacity,
        sizeof(*service->checkpoints.values),
        (void**)&service->checkpoints.values));
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      service->allocator, service->pending.capacity,
      sizeof(*service->pending.values), (void**)&service->pending.values));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->row_count && iree_status_is_ok(status); ++i) {
    status = loom_serve_text_session_initialize(
        loom_serve_text_model_row(service->model, i),
        loom_serve_text_model_tokenizer(service->model),
        service->context_capacity, &service->sessions[i], service->allocator);
  }
  return status;
}

// Resolves protocol and session lifetimes before the registry snapshots final
// counters. Accepted CPU/device work has relinquished its service payloads;
// failed device readiness leaves model storage pinned for actual retirement.
static void text_stop(loom_serve_text_service_t* service,
                      iree_status_code_t terminal_code) {
  loom_serve_residency_cancel_admission(
      loom_serve_text_model_residency(service->model));
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    loom_serve_text_session_t* session = &service->sessions[i];
    if (session->request.connection) {
      // Failed device readiness keeps pins and reserved views until model
      // destruction actually retires accepted work. Normal shutdown is at a
      // completed model boundary and can release unused completion capacity.
      if (terminal_code == IREE_STATUS_OK) {
        loom_serve_text_row_release_reservation(session->row);
      }
      text_request_cancel(service, session);
    }
    loom_serve_text_telemetry_session_end(&service->telemetry, session);
  }
  while (service->pending.count) {
    text_pending_cancel(service, 0);
  }
  if (service->control.connection) {
    loom_serve_text_telemetry_end(&service->telemetry, service->control.receipt,
                                  service->telemetry.failure_code
                                      ? LOOM_SERVE_TELEMETRY_FAILED
                                      : LOOM_SERVE_TELEMETRY_CANCELLED,
                                  0, iree_ok_status());
    loom_serve_http_connection_abort(service->control.connection);
    memset(&service->control, 0, sizeof(service->control));
  }
  for (iree_host_size_t i = 0;
       service->checkpoints.values && i < service->checkpoints.capacity; ++i) {
    if (terminal_code == IREE_STATUS_OK) {
      loom_serve_text_checkpoint_release(service->checkpoints.values[i].state);
    }
    service->checkpoints.values[i].state = NULL;
  }
  service->pool.trim.pending = false;
  text_observe(service, terminal_code == IREE_STATUS_OK ? "stopped" : "failed");
}

void loom_serve_text_service_destroy(loom_serve_text_service_t* service,
                                     iree_status_code_t terminal_code) {
  if (!service) {
    return;
  }
  if (service->telemetry.failure_code) {
    terminal_code = service->telemetry.failure_code;
  }
  text_stop(service, terminal_code);
  const iree_allocator_t host_allocator = service->allocator;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    loom_serve_text_session_deinitialize(&service->sessions[i]);
  }
  iree_slim_mutex_deinitialize(&service->completion.mutex);
  iree_string_builder_deinitialize(&service->tool_calls);
  iree_string_builder_deinitialize(&service->scratch);
  iree_allocator_free(host_allocator, service->pending.values);
  for (iree_host_size_t i = 0;
       service->checkpoints.values && i < service->checkpoints.capacity; ++i) {
    loom_serve_text_chat_completion_deinitialize(
        &service->checkpoints.values[i].completion);
  }
  iree_allocator_free(host_allocator, service->checkpoints.values);
  service->checkpoints.host_bytes = 0;
  loom_serve_text_heartbeat_deinitialize(&service->heartbeat);
  if (service->telemetry.owns_stream) {
    loom_serve_telemetry_stream_destroy(service->telemetry.stream);
  }
  iree_allocator_free(host_allocator, service);
}

iree_status_t loom_serve_text_service_create(
    loom_serve_text_model_t* model,
    const loom_serve_text_service_options_t* options,
    loom_serve_telemetry_stream_t* stream,
    loom_serve_text_service_t** out_service, iree_allocator_t host_allocator) {
  *out_service = NULL;
  loom_serve_text_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*service),
                                             (void**)&service));
  *service = (loom_serve_text_service_t){
      .allocator = host_allocator,
      .model = model,
      .name = options->name,
      .isolated_shape = {loom_serve_text_model_prefill_capacity(model),
                         options->row_count},
      .row_count = options->row_count,
      .chunk_size = options->chunk_size,
      .context_capacity = loom_serve_text_model_context_capacity(model),
      .default_max_tokens = options->default_max_tokens,
      .pending = {.capacity = options->pending_capacity},
      .pool = {.geometry = loom_serve_text_model_pool_usage(model)},
      .schedule_mode = options->schedule_mode,
      .packing_mode = options->packing_mode,
      .mtp_depth = options->mtp_depth,
      .continuation_epochs = options->continuation_epochs,
      .shapes = loom_serve_text_model_shapes(model),
      .shape_count = loom_serve_text_model_shape_count(model)};
  if (!service->shape_count) {
    service->shapes = &service->isolated_shape;
    service->shape_count = 1;
  }
  loom_serve_text_heartbeat_initialize(&service->heartbeat);
  iree_slim_mutex_initialize(&service->completion.mutex);
  iree_string_builder_initialize(host_allocator, &service->scratch);
  iree_string_builder_initialize(host_allocator, &service->tool_calls);
  iree_status_t status = text_service_initialize(service, stream);
  if (iree_status_is_ok(status)) {
    loom_serve_text_heartbeat_configure(
        &service->heartbeat, service->completion.proactor,
        service->telemetry.stream, &service->telemetry.counters,
        options->heartbeat_interval);
    *out_service = service;
  } else {
    loom_serve_text_service_destroy(service, iree_status_code(status));
  }
  return status;
}

static void text_set_observer(void* self, loom_serve_http_observer_t observer) {
  loom_serve_text_service_t* service = self;
  iree_slim_mutex_lock(&service->completion.mutex);
  service->completion.observer = observer;
  iree_slim_mutex_unlock(&service->completion.mutex);
  loom_serve_http_timer_set_observer(&service->heartbeat.timer, observer);
}

static bool text_is_pending(void* self) {
  loom_serve_text_service_t* service = self;
  return service->completion.work != TEXT_WORK_NONE ||
         loom_serve_text_heartbeat_is_pending(&service->heartbeat);
}

static bool text_is_active(void* self) {
  const loom_serve_text_service_t* service = self;
  if (service->completion.work != TEXT_WORK_NONE ||
      service->pool.trim.pending || service->control.connection) {
    return true;
  }
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    if (service->sessions[i].request.connection) {
      return true;
    }
  }
  return false;
}

static iree_status_t text_advance_work(void* self,
                                       loom_serve_http_service_flags_t flags,
                                       bool* out_progress) {
  loom_serve_text_service_t* service = self;
  *out_progress = false;
  if (service->completion.work != TEXT_WORK_NONE) {
    iree_slim_mutex_lock(&service->completion.mutex);
    if (!service->completion.ready) {
      iree_slim_mutex_unlock(&service->completion.mutex);
      return iree_ok_status();
    }
    iree_status_t status = service->completion.status;
    service->completion.status = iree_ok_status();
    service->completion.ready = false;
    iree_slim_mutex_unlock(&service->completion.mutex);
    const text_work_t work = service->completion.work;
    service->completion.work = TEXT_WORK_NONE;
    *out_progress = true;
    if (work == TEXT_WORK_PREPARE) {
      status = text_complete_preparation(service, flags, status);
    }
    IREE_RETURN_IF_ERROR(status);
    switch (work) {
      case TEXT_WORK_EPOCH:
        status = text_complete_epoch(service, flags);
        break;
      case TEXT_WORK_CHECKPOINT:
        status = text_complete_checkpoint(service, flags);
        break;
      case TEXT_WORK_TRIM:
        status = text_complete_trim(service);
        break;
      case TEXT_WORK_RESERVE:
        status = text_complete_reservation(service, flags);
        break;
      case TEXT_WORK_OUTPUT:
        text_complete_output(service);
        break;
      case TEXT_WORK_PREPARE:
      case TEXT_WORK_NONE:
        break;
    }
    IREE_RETURN_IF_ERROR(status);
    if (service->completion.work != TEXT_WORK_NONE) {
      return iree_ok_status();
    }
  }
  if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
    *out_progress |= service->pending.count || service->control.connection;
    for (iree_host_size_t i = 0; i < service->row_count; ++i) {
      *out_progress |= service->sessions[i].request.connection != NULL;
    }
    text_stop(service, service->telemetry.failure_code);
    return iree_ok_status();
  }
  // Release/suspend controls can free capacity for a stalled generation queue.
  // They run only after the previous epoch relinquishes its retained views.
  if (service->control.connection) {
    loom_serve_http_connection_t* connection = service->control.connection;
    const loom_serve_http_request_t* request = service->control.request;
    const loom_serve_text_receipt_t receipt = service->control.receipt;
    memset(&service->control, 0, sizeof(service->control));
    *out_progress = true;
    if (loom_serve_http_connection_failed(connection)) {
      loom_serve_text_telemetry_end(&service->telemetry, receipt,
                                    LOOM_SERVE_TELEMETRY_CANCELLED, 0,
                                    iree_ok_status());
      loom_serve_http_connection_abort(connection);
    } else {
      IREE_RETURN_IF_ERROR(text_enqueue(service, connection, request, receipt));
    }
    if (service->completion.work != TEXT_WORK_NONE) {
      return iree_ok_status();
    }
  }
  text_pending_cancel_failed(service, out_progress);
  // Another model may have completed requests or maintenance since the last
  // turn. Retry this bounded FIFO even without a service-local admission event.
  const iree_host_size_t pending_count = service->pending.count;
  text_observe(service, "admit");
  iree_status_t status = text_admit_pending(service, flags);
  *out_progress |= pending_count != service->pending.count;
  if (service->completion.work != TEXT_WORK_NONE) {
    *out_progress = true;
    return status;
  }
  loom_serve_ready_span_t ready[LOOM_SERVE_TEXT_ROW_CAPACITY] = {0};
  for (iree_host_size_t i = 0;
       i < service->row_count && iree_status_is_ok(status); ++i) {
    status = text_prepare_ready(service, &service->sessions[i], out_progress,
                                &ready[i]);
  }
  if (iree_status_is_ok(status)) {
    *out_progress |= service->pool.trim.pending;
    status = text_submit_trim(service);
  }
  if (iree_status_is_ok(status) && service->completion.work == TEXT_WORK_NONE) {
    if (service->packing_mode == LOOM_SERVE_TEXT_PACKING_SEPARATE) {
      text_separate_ready(service, ready);
    }
    loom_serve_packed_span_t spans[LOOM_SERVE_TEXT_ROW_CAPACITY];
    loom_serve_packed_span_t scratch[LOOM_SERVE_TEXT_ROW_CAPACITY];
    iree_host_size_t shape_index = 0;
    const iree_host_size_t count = loom_serve_pack_shapes(
        service->row_count, ready, service->shape_count, service->shapes,
        service->chunk_size, &service->cursor, spans, scratch, &shape_index);
    if (count) {
      *out_progress = true;
      status = text_submit_epoch(service, shape_index, count, spans);
    }
  }
  if (service->completion.work == TEXT_WORK_NONE) {
    text_observe(service, iree_status_is_ok(status) ? "idle" : "failed");
  }
  return status;
}

static iree_status_t text_advance(void* self,
                                  loom_serve_http_service_flags_t flags,
                                  bool* out_progress) {
  loom_serve_text_service_t* service = self;
  iree_status_t status = loom_serve_text_heartbeat_advance(
      &service->heartbeat,
      iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING),
      out_progress);
  if (!iree_status_is_ok(status)) {
    flags |= LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING;
  }
  bool work_progress = false;
  iree_status_t work_status = text_advance_work(self, flags, &work_progress);
  // A new model failure needs another owner turn to stop a pending timer.
  *out_progress |= work_progress || !iree_status_is_ok(work_status);
  status = iree_status_join(status, work_status);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_telemetry_fail(&service->telemetry, status);
  }
  return status;
}

static iree_status_t text_accept(void* self,
                                 loom_serve_http_connection_t* connection,
                                 const loom_serve_http_request_t* request,
                                 loom_serve_http_service_flags_t flags) {
  loom_serve_text_service_t* service = self;
  const loom_serve_text_receipt_t receipt =
      loom_serve_text_telemetry_begin(&service->telemetry);
  iree_status_t status = text_enqueue(service, connection, request, receipt);
  if (iree_status_is_ok(status) && service->completion.work == TEXT_WORK_NONE) {
    status = text_admit_pending(service, flags);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_text_telemetry_fail(&service->telemetry, status);
  }
  return status;
}

static loom_serve_http_service_snapshot_t text_snapshot(void* self) {
  loom_serve_text_service_t* service = self;
  const loom_serve_text_heartbeat_snapshot_t state =
      loom_serve_text_heartbeat_read(&service->heartbeat);
  const iree_time_t now = iree_time_now();
  const bool active_control = service->control.connection &&
                              service->completion.work == TEXT_WORK_CHECKPOINT;
  iree_time_t queued_arrival =
      service->pending.count ? service->pending.values[0].receipt.arrival_time
                             : IREE_TIME_INFINITE_FUTURE;
  iree_time_t active_arrival = state.active_rows ? state.oldest_active_arrival
                                                 : IREE_TIME_INFINITE_FUTURE;
  if (service->control.connection) {
    if (active_control) {
      active_arrival =
          iree_min(active_arrival, service->control.receipt.arrival_time);
    } else {
      queued_arrival =
          iree_min(queued_arrival, service->control.receipt.arrival_time);
    }
  }
  return (loom_serve_http_service_snapshot_t){
      .phase = state.phase,
      .queued_requests = service->pending.count +
                         (service->control.connection && !active_control),
      .active_requests = state.active_rows + active_control,
      .oldest_queued_age = queued_arrival == IREE_TIME_INFINITE_FUTURE
                               ? 0
                               : now - queued_arrival,
      .oldest_active_age = active_arrival == IREE_TIME_INFINITE_FUTURE
                               ? 0
                               : now - active_arrival};
}

loom_serve_http_service_t loom_serve_text_service_interface(
    loom_serve_text_service_t* service) {
  return (loom_serve_http_service_t){
      .name = service->name,
      .self = service,
      .set_observer = text_set_observer,
      .is_pending = text_is_pending,
      .is_active = text_is_active,
      .accept = text_accept,
      .advance = text_advance,
      .tokenizer = loom_serve_text_model_tokenizer(service->model),
      .kind = "text",
      .endpoint = "/v1/chat/completions",
      .snapshot = text_snapshot,
      .telemetry = {service->telemetry.stream, &service->telemetry.counters},
      .capacity_domain = loom_serve_residency_cache(
          loom_serve_text_model_residency(service->model))};
}
