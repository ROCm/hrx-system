// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/qwen/service.h"

#include <stdio.h>
#include <string.h>

#include "experimental/loom_serve/models/qwen/chat.h"
#include "experimental/loom_serve/models/qwen/schedule.h"
#include "experimental/loom_serve/scheduling/packing.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "loom/util/json.h"

typedef enum qwen_request_phase_e {
  QWEN_REQUEST_PREFILL,
  QWEN_REQUEST_DECODE,
  QWEN_REQUEST_FINISHING,
} qwen_request_phase_t;

typedef struct qwen_session_t {
  // Borrowed private device row in the shared model residency.
  loom_serve_qwen_row_t* row;
  // Local client cache key; empty means an untagged request.
  char name[65];
  // Last admitted request serial, also the idle-row LRU ordering key.
  uint64_t serial;
  // Canonical completed client history validating the retained raw state.
  iree_string_builder_t checkpoint;
  // Preallocated streaming decoder storage reused between requests.
  iree_byte_span_t decoder_storage;
  // Active decoder within decoder_storage, or null between requests.
  iree_tokenizer_decode_state_t* decoder;
  // Fixed host token storage sized to the compiled context.
  int32_t* tokens;
  // Accumulated decoded response, preserving the original model text.
  iree_string_builder_t response;
  // Copied SSE bytes waiting for one carrier credit; no unbounded send queue.
  iree_string_builder_t packet;
  // Active request and its borrowed HTTP storage. A null connection means idle.
  struct {
    // Application claim, relinquished exactly once by finish or abort.
    loom_serve_http_connection_t* connection;
    // Validated chat borrowing the claimed request body.
    loom_serve_qwen_chat_t chat;
    // Next completed-stage action or terminal output drain.
    qwen_request_phase_t phase;
    // Host input tokens prepared for this turn.
    iree_host_size_t input_count;
    // Input tokens already consumed by prefill.
    iree_host_size_t input_offset;
    // Retained device tokens preceding this request's appended input.
    iree_host_size_t retained_count;
    // Selected output tokens, including EOS.
    iree_host_size_t output_count;
    // Ordinary response bytes already incorporated in copied SSE output.
    iree_host_size_t emitted_length;
    // Terminal wire reason, or "error" for failed output translation.
    const char* finish_reason;
    // Application admission time; network framing is not included.
    iree_time_t start_time;
    // Time spent awaiting row and completion capacity before admission.
    iree_duration_t queue_duration;
    // Page-rounded completion credit, including the row's owned pages.
    iree_host_size_t reserved_tokens;
    // First selected token time, including a possible immediate EOS.
    iree_time_t first_token_time;
    // Completed epochs that consumed prompt input for this request.
    uint64_t prefill_steps;
    // Completed epochs that consumed a pending generated token.
    uint64_t decode_steps;
  } request;
} qwen_session_t;

// A queued request owns no model row. Its rendered chat owns host text while
// name/tool views borrow the connection's bounded, immutable HTTP payload.
typedef struct qwen_pending_request_t {
  // Claimed transport identity, relinquished by admission, rejection or abort.
  loom_serve_http_connection_t* connection;
  // Validated canonical prompt and tool descriptors.
  loom_serve_qwen_chat_t chat;
  // Validated local session key borrowing the claimed request headers.
  iree_string_view_t name;
  // Monotonic receipt identity, also used when this request is admitted.
  uint64_t serial;
  // Application receipt time, before model admission.
  iree_time_t arrival_time;
} qwen_pending_request_t;

typedef struct qwen_heartbeat_snapshot_t {
  // Current application activity; literals outlive the reporting thread.
  const char* phase;
  // Start of the current activity, including a possibly unfinished GPU wait.
  iree_time_t phase_start;
  // Most recent completed epoch, or service startup before any model work.
  iree_time_t last_completion;
  // Requests currently holding retained rows.
  iree_host_size_t active_rows;
  // Validated requests waiting without a device row or completion reservation.
  iree_host_size_t queued_requests;
  // Pool accounting in token positions; reserved and resident overlap.
  struct {
    // Total physical capacity; zero selects the dense comparison layout.
    iree_host_size_t capacity;
    // Active completion credit including already resident active pages.
    iree_host_size_t reserved;
    // Physically owned pages across active and idle rows.
    iree_host_size_t resident;
  } pool;
  // Active requests still consuming prompt input.
  iree_host_size_t prefill_rows;
  // Active requests generating output.
  iree_host_size_t decode_rows;
  // Rows paused because their peer has not returned output credit.
  iree_host_size_t backpressured_rows;
  // Issued epoch identity; a failed epoch never increments completed_epochs.
  uint64_t issued_epochs;
  // Fully completed model epochs.
  uint64_t completed_epochs;
  // Total target weight traversals, including each isolated control span.
  uint64_t traversals;
  // Prompt tokens consumed by completed epochs.
  uint64_t prefill_tokens;
  // Pending generated tokens consumed by completed epochs.
  uint64_t decode_tokens;
  // Selected output tokens published to response processing, including EOS.
  uint64_t output_tokens;
  // Sum of completed model epoch wall times, excluding host text processing.
  iree_duration_t model_duration;
  // Completed speculative work, separate from committed output progress.
  struct {
    // Candidate tokens produced across the three proposal rounds.
    uint64_t proposed_tokens;
    // Draft inputs consumed by the target, excluding each pending anchor.
    uint64_t accepted_inputs;
  } mtp;
  // Span count of the most recently issued epoch.
  iree_host_size_t epoch_spans;
  // Input count of the most recently issued epoch, including proposed inputs.
  iree_host_size_t epoch_tokens;
} qwen_heartbeat_snapshot_t;

typedef struct qwen_heartbeat_t {
  // Protects only the copied snapshot and shutdown flag, never model work.
  iree_slim_mutex_t mutex;
  // Wakes the reporter immediately on shutdown.
  iree_notification_t notification;
  // Owned observer, joined before the service relinquishes its stack storage.
  iree_thread_t* thread;
  // Immutable reporting period in nanoseconds.
  iree_duration_t interval;
  // Shutdown request protected by mutex.
  bool stopping;
  // Application-owned counters copied under mutex before reporting.
  qwen_heartbeat_snapshot_t snapshot;
} qwen_heartbeat_t;

typedef struct qwen_service_t {
  // Allocator for the service's host-only storage.
  iree_allocator_t allocator;
  // Borrowed model residency and its single host execution owner.
  loom_serve_qwen_model_t* model;
  // Borrowed transport with an independent network progress thread.
  loom_serve_http_server_t* server;
  // Number of allocated device/session rows.
  iree_host_size_t row_count;
  // Active input tokens issued in one scheduling turn.
  iree_host_size_t chunk_size;
  // Compiled context capacity of each retained row.
  iree_host_size_t context_capacity;
  // Output bound when the client omits one.
  iree_host_size_t default_max_tokens;
  // Same ready partition can execute packed or through isolated controls.
  loom_serve_qwen_schedule_mode_t schedule_mode;
  // Admission ablation independent of the packed versus isolated math choice.
  loom_serve_qwen_packing_mode_t packing_mode;
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
  // Independent periodic reporting of copied service state.
  qwen_heartbeat_t heartbeat;
  // Monotonic request identity, independent of TCP connection identity.
  uint64_t next_serial;
  // Cold admission queue, independent of device slots and epoch span capacity.
  struct {
    // Fixed host queue storage; entries own their chat and HTTP claim.
    qwen_pending_request_t* values;
    // Number of queued entries in FIFO order.
    iree_host_size_t count;
    // Maximum queued request count, allocated once during initialization.
    iree_host_size_t capacity;
    // A new head or released credit/row makes another admission pass useful.
    bool changed;
  } pending;
  // Request policy over the model's physical pool, not a second page allocator.
  struct {
    // Immutable physical geometry; availability is queried at observation.
    loom_serve_qwen_pool_usage_t geometry;
    // Sum of active requests' completion credit in token positions.
    iree_host_size_t reserved;
  } pool;
  // Shared cold input rendering scratch, never borrowed by device work.
  iree_string_builder_t input_text;
  // Shared JSON/SSE serialization scratch, copied into a row packet or carrier.
  iree_string_builder_t scratch;
  // Shared typed tool-call serialization scratch used at generation end.
  iree_string_builder_t tool_calls;
  // Fixed retained rows sharing one model, command set and execution timeline.
  qwen_session_t sessions[LOOM_SERVE_QWEN_ROW_CAPACITY];
} qwen_service_t;

static int qwen_heartbeat_main(void* argument) {
  qwen_heartbeat_t* heartbeat = argument;
  iree_time_t previous_time = iree_time_now();
  uint64_t previous_prefill = 0;
  uint64_t previous_output = 0;
  for (;;) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(&heartbeat->notification);
    iree_slim_mutex_lock(&heartbeat->mutex);
    const bool stopping = heartbeat->stopping;
    const qwen_heartbeat_snapshot_t state = heartbeat->snapshot;
    iree_slim_mutex_unlock(&heartbeat->mutex);
    const iree_time_t now = iree_time_now();
    const double seconds = (now - previous_time) / 1e9;
    fprintf(
        stderr,
        "{\"event\":\"heartbeat\",\"phase\":\"%s\",\"phase_ms\":%.3f,"
        "\"since_completion_ms\":%.3f,\"active_rows\":%zu,"
        "\"queued_requests\":%zu,\"pool\":{\"capacity_tokens\":%zu,"
        "\"reserved_tokens\":%zu,\"resident_tokens\":%zu},"
        "\"prefill_rows\":%zu,\"decode_rows\":%zu,"
        "\"backpressured_rows\":%zu,\"issued_epochs\":%" PRIu64
        ",\"completed_epochs\":%" PRIu64 ",\"traversals\":%" PRIu64
        ",\"prefill_tokens\":%" PRIu64 ",\"decode_tokens\":%" PRIu64
        ",\"output_tokens_including_eos\":%" PRIu64
        ",\"model_ms\":%.3f,\"epoch_spans\":%zu,\"epoch_tokens\":%zu,"
        "\"mtp\":{\"proposed_tokens\":%" PRIu64
        ",\"accepted_draft_inputs\":%" PRIu64
        "},"
        "\"interval_prefill_tokens_per_second\":%.3f,"
        "\"interval_output_tokens_per_second\":%.3f}\n",
        state.phase, (now - state.phase_start) / 1e6,
        (now - state.last_completion) / 1e6, state.active_rows,
        state.queued_requests, state.pool.capacity, state.pool.reserved,
        state.pool.resident, state.prefill_rows, state.decode_rows,
        state.backpressured_rows, state.issued_epochs, state.completed_epochs,
        state.traversals, state.prefill_tokens, state.decode_tokens,
        state.output_tokens, state.model_duration / 1e6, state.epoch_spans,
        state.epoch_tokens, state.mtp.proposed_tokens,
        state.mtp.accepted_inputs,
        seconds > 0 ? (state.prefill_tokens - previous_prefill) / seconds : 0,
        seconds > 0 ? (state.output_tokens - previous_output) / seconds : 0);
    previous_time = now;
    previous_prefill = state.prefill_tokens;
    previous_output = state.output_tokens;
    if (stopping) {
      iree_notification_cancel_wait(&heartbeat->notification);
      break;
    }
    iree_notification_commit_wait(&heartbeat->notification, token,
                                  IREE_DURATION_ZERO,
                                  now + heartbeat->interval);
  }
  return 0;
}

static void qwen_observe(qwen_service_t* service, const char* phase) {
  const loom_serve_qwen_pool_usage_t pool =
      loom_serve_qwen_model_pool_usage(service->model);
  iree_host_size_t active = 0, prefill = 0, decode = 0, backpressured = 0;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    const qwen_session_t* session = &service->sessions[i];
    if (session->request.connection) {
      ++active;
      prefill += session->request.phase == QWEN_REQUEST_PREFILL;
      decode += session->request.phase == QWEN_REQUEST_DECODE;
      backpressured +=
          iree_string_builder_size(&session->packet) &&
          !loom_serve_http_connection_can_send(session->request.connection);
    }
  }
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  qwen_heartbeat_snapshot_t* state = &service->heartbeat.snapshot;
  if (strcmp(state->phase, phase)) {
    state->phase = phase;
    state->phase_start = iree_time_now();
  }
  state->active_rows = active;
  state->queued_requests = service->pending.count;
  state->pool.capacity = pool.capacity;
  state->pool.reserved = service->pool.reserved;
  state->pool.resident = pool.capacity - pool.available;
  state->prefill_rows = prefill;
  state->decode_rows = decode;
  state->backpressured_rows = backpressured;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
}

static void qwen_diagnose(const char* operation, iree_status_t status) {
  fprintf(stderr, "%s: ", operation);
  iree_status_fprint(stderr, status);
  iree_status_free(status);
}

static iree_status_t qwen_quote(iree_string_builder_t* output,
                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

// Error translation is an intentional status ownership boundary. Original
// failures are diagnosed once; failure to construct the response propagates.
static iree_status_t qwen_error_json(iree_status_t failure,
                                     iree_string_builder_t* output) {
  iree_status_t status =
      iree_string_builder_append_cstring(output, "{\"error\":{\"message\":");
  if (iree_status_is_ok(status)) {
    iree_string_view_t message = iree_status_message(failure);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(failure)));
    }
    status = qwen_quote(output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output, ",\"type\":\"loom_request_error\"}}");
  }
  qwen_diagnose("Chat request failed", failure);
  return status;
}

static iree_status_t qwen_reject(qwen_service_t* service,
                                 loom_serve_http_connection_t* connection,
                                 int code, const char* reason,
                                 iree_status_t failure) {
  iree_string_builder_reset(&service->scratch);
  iree_status_t status = iree_string_builder_append_format(
      &service->scratch,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      code, reason);
  if (iree_status_is_ok(status)) {
    status = qwen_error_json(failure, &service->scratch);
  } else {
    status = iree_status_join(status, failure);
  }
  if (iree_status_is_ok(status)) {
    iree_status_t send_status = loom_serve_http_connection_send(
        connection, iree_string_builder_view(&service->scratch));
    if (iree_status_is_ok(send_status)) {
      loom_serve_http_connection_finish(connection);
    } else {
      qwen_diagnose("Chat rejection peer failed", send_status);
      loom_serve_http_connection_abort(connection);
    }
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static void qwen_request_release(qwen_service_t* service,
                                 qwen_session_t* session) {
  service->pool.reserved -= session->request.reserved_tokens;
  service->pending.changed = true;
  loom_serve_qwen_chat_deinitialize(&session->request.chat);
  iree_tokenizer_decode_state_deinitialize(session->decoder);
  session->decoder = NULL;
  memset(&session->request, 0, sizeof(session->request));
  iree_string_builder_reset(&session->packet);
}

static void qwen_request_cancel(qwen_service_t* service,
                                qwen_session_t* session) {
  fprintf(stderr,
          "{\"event\":\"cancel\",\"request\":%" PRIu64
          ",\"session\":\"%s\",\"position\":%zu}\n",
          session->serial, session->name,
          loom_serve_qwen_row_position(session->row));
  loom_serve_http_connection_abort(session->request.connection);
  iree_string_builder_reset(&session->checkpoint);
  qwen_request_release(service, session);
}

static void qwen_request_finish(qwen_service_t* service,
                                qwen_session_t* session) {
  fprintf(
      stderr,
      "{\"event\":\"complete\",\"request\":%" PRIu64
      ",\"session\":\"%s\","
      "\"finish_reason\":\"%s\",\"retained_tokens\":%zu,\"appended_tokens\":%"
      "zu,"
      "\"output_tokens_including_eos\":%zu,\"position\":%zu,\"prefill_steps\":"
      "%" PRIu64
      ","
      "\"decode_steps\":%" PRIu64
      ","
      "\"queue_ms\":%.3f,\"model_ttft_ms\":%.3f,\"request_ms\":%.3f}\n",
      session->serial, session->name, session->request.finish_reason,
      session->request.retained_count, session->request.input_count,
      session->request.output_count, loom_serve_qwen_row_position(session->row),
      session->request.prefill_steps, session->request.decode_steps,
      session->request.queue_duration / 1e6,
      (session->request.first_token_time - session->request.start_time) / 1e6,
      (iree_time_now() - session->request.start_time) / 1e6);
  loom_serve_http_connection_finish(session->request.connection);
  qwen_request_release(service, session);
}

static bool qwen_session_name_valid(iree_string_view_t name) {
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

static qwen_session_t* qwen_session_select(qwen_service_t* service,
                                           iree_string_view_t name) {
  qwen_session_t* oldest = NULL;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    qwen_session_t* session = &service->sessions[i];
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

// Prepares host input without changing retained device state. Rejection leaves
// the previous checkpoint usable. The selected token belongs to the previous
// response but has not yet entered KV/GDN, so it leads a retained append.
static iree_status_t qwen_prepare_input(qwen_service_t* service,
                                        qwen_session_t* session,
                                        const loom_serve_qwen_chat_t* chat,
                                        iree_string_view_t name,
                                        iree_host_size_t* out_count,
                                        iree_host_size_t* out_retained) {
  const iree_string_view_t prompt = iree_string_builder_view(&chat->prompt);
  const iree_string_view_t checkpoint =
      iree_string_builder_view(&session->checkpoint);
  const bool retained =
      name.size && checkpoint.size &&
      iree_string_view_equal(name, iree_make_cstring_view(session->name)) &&
      iree_string_view_starts_with(prompt, checkpoint);
  *out_retained = retained ? loom_serve_qwen_row_position(session->row) : 0;
  iree_string_builder_reset(&service->input_text);
  iree_string_view_t input = prompt;
  if (retained) {
    session->tokens[0] = loom_serve_qwen_row_token(session->row);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &service->input_text,
        loom_serve_qwen_row_is_eos(session->row) ? "\n" : "<|im_end|>\n"));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        &service->input_text,
        iree_string_view_substr(prompt, checkpoint.size, IREE_HOST_SIZE_MAX)));
    input = iree_string_builder_view(&service->input_text);
  }
  const iree_host_size_t prefix_count = retained ? 1 : 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode(loom_serve_qwen_model_tokenizer(service->model),
                            input, IREE_TOKENIZER_ENCODE_FLAG_NONE,
                            iree_tokenizer_make_token_output(
                                session->tokens + prefix_count, NULL, NULL,
                                service->context_capacity - prefix_count),
                            service->allocator, out_count));
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
static void qwen_pending_remove(qwen_service_t* service,
                                iree_host_size_t index) {
  --service->pending.count;
  memmove(service->pending.values + index, service->pending.values + index + 1,
          (service->pending.count - index) * sizeof(*service->pending.values));
  service->pending.changed = true;
}

static void qwen_pending_cancel_failed(qwen_service_t* service,
                                       bool* out_progress) {
  for (iree_host_size_t i = 0; i < service->pending.count;) {
    qwen_pending_request_t* request = &service->pending.values[i];
    if (!loom_serve_http_connection_failed(request->connection)) {
      ++i;
      continue;
    }
    fprintf(stderr,
            "{\"event\":\"cancel_queued\",\"request\":%" PRIu64
            ",\"session\":\"%.*s\"}\n",
            request->serial, (int)request->name.size, request->name.data);
    loom_serve_qwen_chat_deinitialize(&request->chat);
    loom_serve_http_connection_abort(request->connection);
    qwen_pending_remove(service, i);
    *out_progress = true;
  }
}

static iree_status_t qwen_evict(qwen_session_t* session) {
  fprintf(stderr,
          "{\"event\":\"evict\",\"session\":\"%s\",\"position\":%zu,"
          "\"resident_tokens\":%zu}\n",
          session->name, loom_serve_qwen_row_position(session->row),
          loom_serve_qwen_row_pool_usage(session->row));
  IREE_RETURN_IF_ERROR(loom_serve_qwen_row_reset(session->row));
  session->name[0] = 0;
  session->serial = 0;
  iree_string_builder_reset(&session->checkpoint);
  return iree_ok_status();
}

// Active guarantees take priority over idle cache. No cache is displaced until
// the new request fits alongside every active request's full completion credit.
static iree_status_t qwen_reclaim_idle(qwen_service_t* service,
                                       qwen_session_t* selected,
                                       iree_host_size_t reservation) {
  if (!service->pool.geometry.capacity) {
    return iree_ok_status();
  }
  iree_host_size_t charged = service->pool.reserved + reservation;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    qwen_session_t* session = &service->sessions[i];
    if (session != selected && !session->request.connection) {
      charged += loom_serve_qwen_row_pool_usage(session->row);
    }
  }
  iree_status_t status = iree_ok_status();
  while (charged > service->pool.geometry.capacity &&
         iree_status_is_ok(status)) {
    qwen_session_t* oldest = NULL;
    for (iree_host_size_t i = 0; i < service->row_count; ++i) {
      qwen_session_t* session = &service->sessions[i];
      if (session != selected && !session->request.connection &&
          loom_serve_qwen_row_pool_usage(session->row) &&
          (!oldest || session->serial < oldest->serial)) {
        oldest = session;
      }
    }
    // Admission already proved active credit fits; excess is owned idle cache.
    charged -= loom_serve_qwen_row_pool_usage(oldest->row);
    status = qwen_evict(oldest);
  }
  return status;
}

static iree_status_t qwen_admit_pending(qwen_service_t* service) {
  if (!service->pending.changed) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  while (service->pending.count && iree_status_is_ok(status)) {
    qwen_pending_request_t* pending = &service->pending.values[0];
    qwen_session_t* session = qwen_session_select(service, pending->name);
    if (!session) {
      break;
    }
    iree_host_size_t input_count = 0, retained_count = 0;
    status = qwen_prepare_input(service, session, &pending->chat, pending->name,
                                &input_count, &retained_count);
    iree_host_size_t reservation = 0;
    if (iree_status_is_ok(status) && service->pool.geometry.capacity) {
      reservation = loom_serve_qwen_request_reservation(
          service->context_capacity, retained_count + input_count,
          pending->chat.max_tokens, service->mtp_depth,
          service->pool.geometry.block_size);
      if (reservation > service->pool.geometry.capacity) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "request needs %zu page-rounded KV positions including completion "
            "and speculative credit, but the shared pool holds %zu; compact "
            "history, lower max_tokens, or increase pool_capacity",
            reservation, service->pool.geometry.capacity);
      }
    }
    if (!iree_status_is_ok(status)) {
      loom_serve_http_connection_t* connection = pending->connection;
      loom_serve_qwen_chat_deinitialize(&pending->chat);
      qwen_pending_remove(service, 0);
      status = qwen_reject(service, connection, 400, "Bad Request", status);
      continue;
    }
    if (reservation >
        service->pool.geometry.capacity - service->pool.reserved) {
      break;
    }
    if (!retained_count && loom_serve_qwen_row_position(session->row)) {
      status = qwen_evict(session);
    }
    if (iree_status_is_ok(status)) {
      status = qwen_reclaim_idle(service, session, reservation);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    const qwen_pending_request_t request = *pending;
    qwen_pending_remove(service, 0);
    if (request.name.size) {
      memcpy(session->name, request.name.data, request.name.size);
    }
    session->name[request.name.size] = 0;
    session->serial = request.serial;
    session->request.connection = request.connection;
    session->request.chat = request.chat;
    session->request.phase = QWEN_REQUEST_PREFILL;
    session->request.input_count = input_count;
    session->request.retained_count = retained_count;
    session->request.start_time = iree_time_now();
    session->request.queue_duration =
        session->request.start_time - request.arrival_time;
    session->request.reserved_tokens = reservation;
    service->pool.reserved += reservation;
    iree_string_builder_reset(&session->response);
    iree_string_builder_reset(&session->checkpoint);
    status = iree_tokenizer_decode_state_initialize(
        loom_serve_qwen_model_tokenizer(service->model),
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        session->decoder_storage, &session->decoder);
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(
          &session->packet,
          "HTTP/1.1 200 OK\r\nContent-Type: "
          "text/event-stream\r\nCache-Control: "
          "no-cache\r\nConnection: close\r\n\r\n");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_qwen_chat_event(
          session->serial, IREE_SV("{\"role\":\"assistant\"}"),
          iree_string_view_empty(), &session->packet);
    }
    if (iree_status_is_ok(status)) {
      fprintf(stderr,
              "{\"event\":\"admit\",\"request\":%" PRIu64
              ",\"session\":\"%s\",\"row\":%zu,\"cache\":\"%s\","
              "\"retained_tokens\":%zu,\"appended_tokens\":%zu,"
              "\"max_tokens\":%zu,\"reservation_tokens\":%zu,"
              "\"pool_reserved_tokens\":%zu,\"queue_ms\":%.3f}\n",
              session->serial, session->name,
              (iree_host_size_t)(session - service->sessions),
              retained_count ? "hit" : "replay", retained_count, input_count,
              request.chat.max_tokens, reservation, service->pool.reserved,
              session->request.queue_duration / 1e6);
    }
  }
  service->pending.changed = false;
  return status;
}

static iree_status_t qwen_enqueue(qwen_service_t* service,
                                  loom_serve_http_connection_t* connection,
                                  const loom_serve_http_request_t* request) {
  if (iree_string_view_equal(request->method, IREE_SV("GET")) &&
      iree_string_view_equal(request->target, IREE_SV("/healthz"))) {
    iree_status_t status = loom_serve_http_connection_send(
        connection,
        IREE_SV(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: "
            "close\r\n\r\n{\"status\":\"ready\"}"));
    if (iree_status_is_ok(status)) {
      loom_serve_http_connection_finish(connection);
    } else {
      qwen_diagnose("Health peer failed", status);
      loom_serve_http_connection_abort(connection);
    }
    return iree_ok_status();
  }
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(request->target,
                              IREE_SV("/v1/chat/completions"))) {
    return qwen_reject(
        service, connection, 404, "Not Found",
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "use POST /v1/chat/completions or GET /healthz"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("Content-Type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return qwen_reject(
        service, connection, 415, "Unsupported Media Type",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Content-Type must be application/json"));
  }
  iree_string_view_t name;
  loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Session"),
                                        &name);
  if (!qwen_session_name_valid(name)) {
    return qwen_reject(
        service, connection, 400, "Bad Request",
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "X-Loom-Session must be at most 64 ASCII identifier characters"));
  }
  qwen_session_t* session = qwen_session_select(service, name);
  bool busy = session && session->request.connection;
  for (iree_host_size_t i = 0; name.size && i < service->pending.count; ++i) {
    busy |= iree_string_view_equal(name, service->pending.values[i].name);
  }
  if (busy) {
    return qwen_reject(
        service, connection, 409, "Conflict",
        iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                         "session already has an active or queued request"));
  }
  if (service->pending.count == service->pending.capacity) {
    return qwen_reject(service, connection, 503, "Service Unavailable",
                       iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                        "pending request queue is full"));
  }
  loom_serve_qwen_chat_t chat;
  iree_status_t status = loom_serve_qwen_chat_initialize(
      request->body, service->default_max_tokens, service->allocator, &chat);
  if (!iree_status_is_ok(status)) {
    return qwen_reject(service, connection, 400, "Bad Request", status);
  }
  const uint64_t serial = ++service->next_serial;
  service->pending.changed |= service->pending.count == 0;
  service->pending.values[service->pending.count++] = (qwen_pending_request_t){
      .connection = connection,
      .chat = chat,
      .name = name,
      .serial = serial,
      .arrival_time = iree_time_now(),
  };
  fprintf(stderr,
          "{\"event\":\"enqueue\",\"request\":%" PRIu64
          ",\"session\":\"%.*s\",\"queued_requests\":%zu}\n",
          serial, (int)name.size, name.data, service->pending.count);
  return iree_ok_status();
}

static iree_status_t qwen_emit_text(qwen_service_t* service,
                                    qwen_session_t* session,
                                    iree_host_size_t end) {
  const iree_host_size_t start = session->request.emitted_length;
  if (start == end) {
    return iree_ok_status();
  }
  iree_string_builder_reset(&service->scratch);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&service->scratch, "{\"content\":"));
  IREE_RETURN_IF_ERROR(qwen_quote(
      &service->scratch,
      iree_string_view_substr(iree_string_builder_view(&session->response),
                              start, end - start)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&service->scratch, "}"));
  IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_event(
      session->serial, iree_string_builder_view(&service->scratch),
      iree_string_view_empty(), &session->packet));
  session->request.emitted_length = end;
  return iree_ok_status();
}

static iree_status_t qwen_append_response(qwen_session_t* session,
                                          iree_string_view_t text) {
  if (text.size > 1024 * 1024 - iree_string_builder_size(&session->response)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "generated response exceeds 1 MiB text limit");
  }
  return iree_string_builder_append_string(&session->response, text);
}

static iree_status_t qwen_complete(qwen_service_t* service,
                                   qwen_session_t* session) {
  char text[8192];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_finalize(
      session->decoder, iree_make_mutable_string_view(text, sizeof(text)),
      &length));
  IREE_RETURN_IF_ERROR(
      qwen_append_response(session, iree_make_string_view(text, length)));
  iree_string_builder_reset(&service->tool_calls);
  iree_host_size_t tool_count = 0;
  IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_complete(
      &session->request.chat, iree_string_builder_view(&session->response),
      session->serial, &session->checkpoint, &service->tool_calls,
      &tool_count));
  const iree_string_view_t response =
      iree_string_builder_view(&session->response);
  const iree_host_size_t text_end =
      tool_count ? iree_string_view_find(response, IREE_SV("<tool_call>"),
                                         session->request.emitted_length)
                 : response.size;
  IREE_RETURN_IF_ERROR(qwen_emit_text(service, session, text_end));
  if (tool_count) {
    iree_string_builder_reset(&service->scratch);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &service->scratch, "{\"tool_calls\":%.*s}",
        (int)iree_string_builder_size(&service->tool_calls),
        iree_string_builder_buffer(&service->tool_calls)));
    IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_event(
        session->serial, iree_string_builder_view(&service->scratch),
        iree_string_view_empty(), &session->packet));
  }
  session->request.finish_reason = tool_count ? "tool_calls"
                                   : loom_serve_qwen_row_is_eos(session->row)
                                       ? "stop"
                                       : "length";
  IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_event(
      session->serial, IREE_SV("{}"),
      iree_make_cstring_view(session->request.finish_reason),
      &session->packet));
  if (session->request.chat.include_usage) {
    IREE_RETURN_IF_ERROR(loom_serve_qwen_chat_usage(
        session->serial,
        session->request.retained_count + session->request.input_count,
        session->request.retained_count, session->request.output_count,
        &session->packet));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&session->packet, "data: [DONE]\n\n"));
  session->request.phase = QWEN_REQUEST_FINISHING;
  return iree_ok_status();
}

static iree_status_t qwen_selected_tokens(qwen_service_t* service,
                                          qwen_session_t* session,
                                          iree_host_size_t count,
                                          const int32_t* tokens) {
  if (!session->request.output_count) {
    session->request.first_token_time = iree_time_now();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    ++session->request.output_count;
    iree_host_size_t consumed = 0;
    while (!consumed && iree_status_is_ok(status)) {
      char text[8192];
      iree_host_size_t length = 0;
      status = iree_tokenizer_decode_state_feed(
          session->decoder, iree_tokenizer_make_token_id_list(tokens + i, 1),
          iree_make_mutable_string_view(text, sizeof(text)), &consumed,
          &length);
      if (iree_status_is_ok(status) && !consumed && !length) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "token exceeds streaming text capacity");
      }
      if (iree_status_is_ok(status)) {
        status =
            qwen_append_response(session, iree_make_string_view(text, length));
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const bool complete =
        loom_serve_qwen_row_is_eos(session->row) ||
        session->request.output_count == session->request.chat.max_tokens;
    status = complete ? qwen_complete(service, session)
                      : qwen_emit_text(
                            service, session,
                            loom_serve_qwen_chat_text_end(
                                iree_string_builder_view(&session->response),
                                session->request.emitted_length));
  }
  if (!iree_status_is_ok(status)) {
    // Device work completed, but translation cannot establish a client history.
    // Preserve prior streamed text and terminate with a real SSE error.
    iree_string_builder_reset(&session->checkpoint);
    iree_string_builder_reset(&session->packet);
    iree_status_t packet_status =
        iree_string_builder_append_cstring(&session->packet, "data: ");
    if (iree_status_is_ok(packet_status)) {
      packet_status = qwen_error_json(status, &session->packet);
    } else {
      packet_status = iree_status_join(packet_status, status);
    }
    if (iree_status_is_ok(packet_status)) {
      packet_status = iree_string_builder_append_cstring(
          &session->packet, "\n\ndata: [DONE]\n\n");
    }
    session->request.finish_reason = "error";
    session->request.phase = QWEN_REQUEST_FINISHING;
    status = packet_status;
  }
  if (session->request.phase == QWEN_REQUEST_FINISHING) {
    const iree_host_size_t resident =
        loom_serve_qwen_row_pool_usage(session->row);
    service->pool.reserved -= session->request.reserved_tokens - resident;
    session->request.reserved_tokens = resident;
    service->pending.changed = true;
  }
  return status;
}

// Output credit and cancellation are settled before a row enters an epoch.
// The carrier owns a copied send and the row owns one staging packet. An empty
// staging packet is output credit even while the preceding send is in flight;
// waiting for that send would split an otherwise ready cohort. A full staging
// packet behind a busy carrier pauses only that row.
static void qwen_prepare_ready(qwen_service_t* service, qwen_session_t* session,
                               bool* out_progress,
                               loom_serve_ready_span_t* out_ready) {
  *out_ready = (loom_serve_ready_span_t){0};
  if (!session->request.connection) {
    return;
  }
  if (loom_serve_http_connection_failed(session->request.connection)) {
    qwen_request_cancel(service, session);
    *out_progress = true;
    return;
  }
  if (iree_string_builder_size(&session->packet)) {
    if (!loom_serve_http_connection_can_send(session->request.connection)) {
      return;
    }
    *out_progress = true;
    iree_status_t status = loom_serve_http_connection_send(
        session->request.connection,
        iree_string_builder_view(&session->packet));
    if (!iree_status_is_ok(status)) {
      qwen_diagnose("Chat stream peer failed", status);
      qwen_request_cancel(service, session);
      return;
    }
    iree_string_builder_reset(&session->packet);
    if (session->request.phase == QWEN_REQUEST_FINISHING) {
      qwen_request_finish(service, session);
      return;
    }
  }
  if (session->request.phase == QWEN_REQUEST_PREFILL) {
    *out_ready = (loom_serve_ready_span_t){
        session->request.input_count - session->request.input_offset, 1};
  } else {
    // A final single output cannot amortize drafting. Near the context limit,
    // ordinary decode avoids writing speculative KV beyond allocated capacity.
    const bool verify =
        service->mtp_depth &&
        session->request.chat.max_tokens - session->request.output_count > 1 &&
        service->context_capacity -
                loom_serve_qwen_row_position(session->row) >=
            4;
    *out_ready = verify ? (loom_serve_ready_span_t){4, 4}
                        : (loom_serve_ready_span_t){1, 1};
  }
}

// The rotating first ready row chooses the phase; other rows of that phase
// can still batch together. Explicit request phase matters: a one-token prompt
// tail is not a decode. Credit and cancellation were settled before this
// filter.
static void qwen_separate_ready(const qwen_service_t* service,
                                loom_serve_ready_span_t* ready) {
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    const iree_host_size_t first = (service->cursor + i) % service->row_count;
    if (!ready[first].token_count) {
      continue;
    }
    const qwen_request_phase_t phase = service->sessions[first].request.phase;
    for (iree_host_size_t row = 0; row < service->row_count; ++row) {
      if (service->sessions[row].request.phase != phase) {
        ready[row] = (loom_serve_ready_span_t){0};
      }
    }
    return;
  }
}

static iree_status_t qwen_execute_epoch(
    qwen_service_t* service, iree_host_size_t shape_index,
    iree_host_size_t count, const loom_serve_packed_span_t* scheduled) {
  loom_serve_qwen_span_t spans[LOOM_SERVE_QWEN_ROW_CAPACITY];
  loom_serve_qwen_result_t results[LOOM_SERVE_QWEN_ROW_CAPACITY] = {0};
  int32_t decode_tokens[LOOM_SERVE_QWEN_ROW_CAPACITY];
  iree_host_size_t positions[LOOM_SERVE_QWEN_ROW_CAPACITY];
  uint32_t output_limits[LOOM_SERVE_QWEN_ROW_CAPACITY] = {0};
  iree_host_size_t proposal_count = 0;
  iree_host_size_t prefill_count = 0, decode_count = 0, output_count = 0;
  iree_host_size_t input_count = 0, accepted_drafts = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t row = scheduled[i].row_index;
    qwen_session_t* session = &service->sessions[row];
    const bool prefill = session->request.phase == QWEN_REQUEST_PREFILL;
    const bool select = !prefill || scheduled[i].token_count ==
                                        session->request.input_count -
                                            session->request.input_offset;
    positions[i] = loom_serve_qwen_row_position(session->row);
    if (!prefill) {
      decode_tokens[i] = loom_serve_qwen_row_token(session->row);
      if (scheduled[i].token_count == 4) {
        ++proposal_count;
        output_limits[i] = (uint32_t)iree_min(
            4 * service->continuation_epochs,
            session->request.chat.max_tokens - session->request.output_count);
      }
    }
    spans[i] = (loom_serve_qwen_span_t){
        .row_index = row,
        .token_count = scheduled[i].token_count,
        .token_ids = prefill ? session->tokens + session->request.input_offset
                             : &decode_tokens[i],
        .flags = (select ? LOOM_SERVE_QWEN_SPAN_FLAG_SELECT : 0) |
                 (output_limits[i] ? LOOM_SERVE_QWEN_SPAN_FLAG_PROPOSE : 0),
    };
    prefill_count += prefill ? spans[i].token_count : 0;
    input_count += spans[i].token_count;
  }
  const bool packed = service->schedule_mode == LOOM_SERVE_QWEN_SCHEDULE_PACKED;
  const iree_host_size_t device_epochs =
      proposal_count ? service->continuation_epochs : 1;
  iree_host_size_t verification_count = 0;
  const char* mode =
      packed                                                       ? "packed"
      : service->schedule_mode == LOOM_SERVE_QWEN_SCHEDULE_MATCHED ? "matched"
                                                                   : "isolated";
  qwen_observe(service, "execute");
  const iree_time_t start = iree_time_now();
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  qwen_heartbeat_snapshot_t* state = &service->heartbeat.snapshot;
  const uint64_t epoch = state->issued_epochs + 1;
  state->issued_epochs += device_epochs;
  state->phase_start = start;
  state->epoch_spans = count;
  state->epoch_tokens = input_count;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
  iree_status_t status = iree_ok_status();
  if (packed) {
    status = proposal_count ? loom_serve_qwen_model_verify(
                                  service->model, shape_index, device_epochs,
                                  count, spans, output_limits, results)
                            : loom_serve_qwen_model_epoch(
                                  service->model, shape_index, count, spans);
  } else {
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      qwen_session_t* session = &service->sessions[spans[i].row_index];
      if (session->request.phase == QWEN_REQUEST_PREFILL ||
          service->schedule_mode == LOOM_SERVE_QWEN_SCHEDULE_MATCHED) {
        status = loom_serve_qwen_row_prefill(session->row, spans[i].token_count,
                                             spans[i].token_ids);
      } else {
        status = loom_serve_qwen_row_decode(session->row);
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  const iree_time_t completed = iree_time_now();
  iree_string_builder_reset(&service->scratch);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&service->scratch, "["));
  for (iree_host_size_t i = 0; i < count; ++i) {
    qwen_session_t* session = &service->sessions[spans[i].row_index];
    const bool prefill = session->request.phase == QWEN_REQUEST_PREFILL;
    const bool select =
        iree_any_bit_set(spans[i].flags, LOOM_SERVE_QWEN_SPAN_FLAG_SELECT);
    if (!proposal_count) {
      results[i].consumed_count = spans[i].token_count;
      results[i].output_count = select ? 1 : 0;
      if (select) {
        results[i].tokens[0] = loom_serve_qwen_row_token(session->row);
      }
    }
    decode_count += prefill ? 0 : results[i].consumed_count;
    output_count += results[i].output_count;
    verification_count += results[i].verification_count;
    accepted_drafts += output_limits[i] ? results[i].consumed_count -
                                              results[i].verification_count
                                        : 0;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &service->scratch,
        "%s{\"row\":%zu,\"request\":%" PRIu64
        ",\"kind\":\"%s\",\"position\":%zu,\"tokens\":%zu,\"select\":%s,"
        "\"consumed_tokens\":%zu,\"output_tokens\":%zu,\"output_limit\":%u,"
        "\"verification_epochs\":%zu}",
        i ? "," : "", spans[i].row_index, session->serial,
        prefill            ? "prefill"
        : output_limits[i] ? "verify"
                           : "decode",
        positions[i], spans[i].token_count, select ? "true" : "false",
        results[i].consumed_count, results[i].output_count, output_limits[i],
        results[i].verification_count));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&service->scratch, "]"));
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
  fprintf(stderr,
          "{\"event\":\"epoch\",\"epoch\":%" PRIu64
          ",\"scheduler\":\"%s\",\"spans\":%zu,\"shape\":%zu,"
          "\"token_capacity\":%zu,\"span_capacity\":%zu,"
          "\"packing\":\"%s\","
          "\"prefill_tokens\":%zu,\"decode_tokens\":%zu,"
          "\"selected_tokens_including_eos\":%zu,\"traversals\":%zu,"
          "\"model_ms\":%.3f,\"device_epochs\":%zu,"
          "\"mtp\":{\"rows\":%zu,\"verifications\":%zu,\"proposed_tokens\":%zu,"
          "\"accepted_draft_inputs\":%zu},\"rows\":%.*s}\n",
          epoch, mode, count, shape_index,
          service->shapes[shape_index].token_capacity,
          service->shapes[shape_index].span_capacity,
          service->packing_mode == LOOM_SERVE_QWEN_PACKING_SEPARATE ? "separate"
                                                                    : "mixed",
          prefill_count, decode_count, output_count,
          packed ? device_epochs : count, (completed - start) / 1e6,
          device_epochs, proposal_count, verification_count,
          verification_count * 3, accepted_drafts,
          (int)iree_string_builder_size(&service->scratch),
          iree_string_builder_buffer(&service->scratch));
  qwen_observe(service, "output");
  iree_host_size_t published_count = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    qwen_session_t* session = &service->sessions[spans[i].row_index];
    if (session->request.phase == QWEN_REQUEST_PREFILL) {
      ++session->request.prefill_steps;
      session->request.input_offset += results[i].consumed_count;
      if (session->request.input_offset == session->request.input_count) {
        session->request.phase = QWEN_REQUEST_DECODE;
      }
    } else {
      ++session->request.decode_steps;
    }
    if (results[i].output_count) {
      const iree_host_size_t previous_count = session->request.output_count;
      status = qwen_selected_tokens(service, session, results[i].output_count,
                                    results[i].tokens);
      published_count += session->request.output_count - previous_count;
    }
  }
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  state->output_tokens += published_count;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
  return status;
}

static iree_status_t qwen_service_initialize(qwen_service_t* service) {
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      service->allocator, service->pending.capacity,
      sizeof(*service->pending.values), (void**)&service->pending.values));
  iree_host_size_t decoder_size = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_calculate_size(
      loom_serve_qwen_model_tokenizer(service->model), &decoder_size));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->row_count && iree_status_is_ok(status); ++i) {
    qwen_session_t* session = &service->sessions[i];
    session->row = loom_serve_qwen_model_row(service->model, i);
    status = iree_allocator_malloc_array(
        service->allocator, service->context_capacity, sizeof(*session->tokens),
        (void**)&session->tokens);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(service->allocator, decoder_size,
                                     (void**)&session->decoder_storage.data);
      session->decoder_storage.data_length = decoder_size;
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_reserve(&session->response, 65536);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_reserve(&session->packet, 16384);
    }
  }
  return status;
}

iree_status_t loom_serve_qwen_service_run(
    loom_serve_qwen_model_t* model, loom_serve_http_server_t* server,
    const loom_serve_qwen_service_options_t* options,
    iree_allocator_t host_allocator) {
  const loom_serve_packing_shape_t isolated_shape = {
      loom_serve_qwen_model_prefill_capacity(model), options->row_count};
  qwen_service_t service = {
      .allocator = host_allocator,
      .model = model,
      .server = server,
      .row_count = options->row_count,
      .chunk_size = options->chunk_size,
      .context_capacity = loom_serve_qwen_model_context_capacity(model),
      .default_max_tokens = options->default_max_tokens,
      .pending = {.capacity = options->pending_capacity},
      .pool = {.geometry = loom_serve_qwen_model_pool_usage(model)},
      .schedule_mode = options->schedule_mode,
      .packing_mode = options->packing_mode,
      .mtp_depth = options->mtp_depth,
      .continuation_epochs = options->continuation_epochs,
      .shapes = loom_serve_qwen_model_shapes(model),
      .shape_count = loom_serve_qwen_model_shape_count(model),
      .heartbeat = {.interval = options->heartbeat_interval}};
  if (!service.shape_count) {
    service.shapes = &isolated_shape;
    service.shape_count = 1;
  }
  iree_slim_mutex_initialize(&service.heartbeat.mutex);
  iree_notification_initialize(&service.heartbeat.notification);
  service.heartbeat.snapshot.phase = "starting";
  service.heartbeat.snapshot.phase_start = iree_time_now();
  service.heartbeat.snapshot.last_completion =
      service.heartbeat.snapshot.phase_start;
  iree_string_builder_initialize(host_allocator, &service.input_text);
  iree_string_builder_initialize(host_allocator, &service.scratch);
  iree_string_builder_initialize(host_allocator, &service.tool_calls);
  for (iree_host_size_t i = 0; i < service.row_count; ++i) {
    iree_string_builder_initialize(host_allocator,
                                   &service.sessions[i].checkpoint);
    iree_string_builder_initialize(host_allocator,
                                   &service.sessions[i].response);
    iree_string_builder_initialize(host_allocator, &service.sessions[i].packet);
  }
  iree_status_t status = qwen_service_initialize(&service);
  if (iree_status_is_ok(status) && service.heartbeat.interval) {
    const iree_thread_create_params_t parameters = {
        .name = IREE_SV("qwen-heartbeat"),
    };
    status =
        iree_thread_create(qwen_heartbeat_main, &service.heartbeat, parameters,
                           host_allocator, &service.heartbeat.thread);
  }
  iree_notification_t* notification =
      loom_serve_http_server_notification(server);
  while (iree_status_is_ok(status) &&
         !loom_serve_http_server_is_stopping(server)) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(notification);
    bool progress = false;
    qwen_pending_cancel_failed(&service, &progress);
    status = qwen_admit_pending(&service);
    // Drain a bounded cohort before selecting work, without allowing a stream
    // of new requests or health probes to starve active rows.
    qwen_observe(&service, "admit");
    for (iree_host_size_t i = 0;
         i < service.row_count && iree_status_is_ok(status); ++i) {
      const loom_serve_http_request_t* request = NULL;
      loom_serve_http_connection_t* connection =
          loom_serve_http_server_take_request(server, &request);
      if (!connection) {
        break;
      }
      status = qwen_enqueue(&service, connection, request);
      if (iree_status_is_ok(status)) {
        status = qwen_admit_pending(&service);
      }
      progress = true;
    }
    loom_serve_ready_span_t ready[LOOM_SERVE_QWEN_ROW_CAPACITY] = {0};
    for (iree_host_size_t i = 0;
         i < service.row_count && iree_status_is_ok(status); ++i) {
      qwen_prepare_ready(&service, &service.sessions[i], &progress, &ready[i]);
    }
    if (iree_status_is_ok(status) &&
        !loom_serve_http_server_is_stopping(server)) {
      if (service.packing_mode == LOOM_SERVE_QWEN_PACKING_SEPARATE) {
        qwen_separate_ready(&service, ready);
      }
      loom_serve_packed_span_t spans[LOOM_SERVE_QWEN_ROW_CAPACITY];
      loom_serve_packed_span_t scratch[LOOM_SERVE_QWEN_ROW_CAPACITY];
      iree_host_size_t shape_index = 0;
      const iree_host_size_t count = loom_serve_pack_shapes(
          service.row_count, ready, service.shape_count, service.shapes,
          service.chunk_size, &service.cursor, spans, scratch, &shape_index);
      if (count) {
        progress = true;
        status = qwen_execute_epoch(&service, shape_index, count, spans);
      }
    }
    qwen_observe(&service, iree_status_is_ok(status) ? "idle" : "failed");
    if (progress || !iree_status_is_ok(status) ||
        loom_serve_http_server_is_stopping(server)) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  for (iree_host_size_t i = 0; i < service.row_count; ++i) {
    qwen_session_t* session = &service.sessions[i];
    if (session->request.connection) {
      qwen_request_cancel(&service, session);
    }
    iree_allocator_free(host_allocator, session->decoder_storage.data);
    iree_allocator_free(host_allocator, session->tokens);
    iree_string_builder_deinitialize(&session->packet);
    iree_string_builder_deinitialize(&session->response);
    iree_string_builder_deinitialize(&session->checkpoint);
  }
  qwen_observe(&service, iree_status_is_ok(status) ? "stopped" : "failed");
  iree_slim_mutex_lock(&service.heartbeat.mutex);
  service.heartbeat.stopping = true;
  iree_slim_mutex_unlock(&service.heartbeat.mutex);
  iree_notification_post(&service.heartbeat.notification, IREE_ALL_WAITERS);
  if (service.heartbeat.thread) {
    // Releasing the sole thread reference joins it before reclaiming storage.
    iree_thread_release(service.heartbeat.thread);
  }
  iree_notification_deinitialize(&service.heartbeat.notification);
  iree_slim_mutex_deinitialize(&service.heartbeat.mutex);
  iree_string_builder_deinitialize(&service.tool_calls);
  iree_string_builder_deinitialize(&service.scratch);
  iree_string_builder_deinitialize(&service.input_text);
  for (iree_host_size_t i = 0; i < service.pending.count; ++i) {
    loom_serve_qwen_chat_deinitialize(&service.pending.values[i].chat);
    loom_serve_http_connection_abort(service.pending.values[i].connection);
  }
  iree_allocator_free(host_allocator, service.pending.values);
  return status;
}
