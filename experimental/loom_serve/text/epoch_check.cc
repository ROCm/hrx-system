// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real-weight epoch and maintenance ownership with one JIT/serving worker.

#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

#include "experimental/loom_serve/runtime/device_flags.h"
#include "experimental/loom_serve/runtime/host.h"
#include "experimental/loom_serve/text/chat.h"
#include "experimental/loom_serve/text/flags.h"
#include "iree/async/operations/semaphore.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/task/executor.h"

namespace {

constexpr size_t kSpanCount = 2;
constexpr size_t kTokenCapacity = 512;
constexpr std::array<size_t, kSpanCount> kRows = {5, 2};
constexpr std::array<size_t, kSpanCount> kReferenceRows = {1, 0};
using Spans = std::array<loom_serve_text_span_t, kSpanCount>;
using Results = std::array<loom_serve_text_result_t, kSpanCount>;

struct ModelWork {
  enum class Operation { kEpoch, kTrim, kReserve, kPin, kSuspend, kReset };
  enum class Cancellation { kNone, kBeforeSubmission, kAfterSubmission };
  // Model operation submitted and completed by the same task protocol.
  Operation operation = Operation::kEpoch;
  // Borrowed source model; the outer caller retires all tasks before teardown.
  loom_serve_text_model_t* model = nullptr;
  // Borrowed executor with one worker shared by compilation and serving.
  iree_task_executor_t* executor = nullptr;
  // Borrowed host proactor driven independently of task workers.
  iree_async_proactor_t* proactor = nullptr;
  // Immutable first-plan shape selected by the outer test caller.
  size_t shape = 0;
  // Borrowed caller plan copied into submission-local storage before use.
  Spans spans = {};
  // Optional caller-owned MTP limits copied before submission.
  const uint32_t* limits = nullptr;
  // Optional caller-owned second plan copied before submission.
  const loom_serve_text_continuation_t* continuation = nullptr;
  // Semantic progress produced only by successful completion.
  Results results = {};
  // Compaction accounting remains owned through submission and completion.
  loom_serve_text_trim_result_t trim_result = {};
  // Request admission inputs and results remain owned across readiness.
  struct {
    // Borrowed destination row, retained by model.
    loom_serve_text_row_t* row = nullptr;
    // Starting frontier requested by the caller.
    loom_serve_text_reserve_mode_t mode = LOOM_SERVE_TEXT_RESERVE_FRESH;
    // Borrowed same-model endpoint for CHECKPOINT mode.
    loom_serve_text_checkpoint_t* checkpoint = nullptr;
    // Requested consumed-position high-water, in tokens.
    size_t extent = 0;
    // Submission acquired a pin and requires completion consumption.
    bool submitted = false;
    // Successful completion produced a runnable reservation.
    bool admitted = false;
    // Simulated request cancellation, applied only after retirement.
    bool cancel = false;
  } reservation;
  // Endpoint ownership remains private until successful readiness.
  struct {
    // Borrowed retired source row for pinning.
    loom_serve_text_row_t* row = nullptr;
    // Pending or ready endpoint, retained by the model until released.
    loom_serve_text_checkpoint_t* value = nullptr;
    // Caller abandonment before admission or while work owns storage.
    Cancellation cancellation = Cancellation::kNone;
    // Fails feedback readiness while the capture work branch remains valid.
    bool fail_readiness = false;
    // Successful readiness was consumed by the owner.
    bool completed = false;
    // Free token slots before capture, checked through failed readiness.
    size_t available = 0;
  } checkpoint;
  // Borrowed row whose ownership is cleared at a retired boundary.
  loom_serve_text_row_t* reset_row = nullptr;
  // Row frontiers before submission, checked again before publication.
  std::array<size_t, kSpanCount> positions = {};
  // Task submitting the source-owned device sequence.
  iree_task_process_t submit = {};
  // Task consuming successful readiness on the shared executor.
  iree_task_process_t complete = {};
  // Operation storage retained until the terminal completion callback returns.
  iree_async_semaphore_wait_operation_t wait = {};
  // Borrowed accepted work and feedback timelines returned by the model.
  loom_serve_execution_frontier_t frontier = {};
  // Toll-free async identities for the HAL timelines.
  std::array<iree_async_semaphore_t*, 2> semaphores = {};
  // Serializes terminal status and joins callbacks with outer storage reuse.
  std::mutex mutex;
  // Wakes the outer caller, never a task worker.
  std::condition_variable condition;
  // Number of processes whose final executor access has retired.
  size_t released = 0;
  // Owned terminal status transferred to the outer caller after retirement.
  iree_status_t status = iree_ok_status();

  iree_status_t SubmitCopies() {
    Spans local_spans = spans;
    Spans local_next = {};
    std::array<std::array<int32_t, kTokenCapacity>, 2 * kSpanCount> tokens = {};
    std::array<uint32_t, kSpanCount> local_limits = {};
    std::array<uint32_t, kSpanCount> next_limits = {};
    loom_serve_text_continuation_t local_continuation = {};
    if (limits) {
      memcpy(local_limits.data(), limits, sizeof(local_limits));
    }
    for (size_t i = 0; i < kSpanCount; ++i) {
      auto& span = local_spans[i];
      const size_t count =
          iree_any_bit_set(span.flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE)
              ? 1
              : span.token_count;
      memcpy(tokens[i].data(), span.token_ids, count * sizeof(int32_t));
      span.token_ids = tokens[i].data();
    }
    if (continuation) {
      local_continuation = *continuation;
      for (size_t i = 0; i < continuation->span_count; ++i) {
        auto& span = local_next[i];
        span = continuation->spans[i];
        next_limits[i] = continuation->output_limits[i];
        if (!iree_any_bit_set(span.flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE)) {
          memcpy(tokens[kSpanCount + i].data(), span.token_ids,
                 span.token_count * sizeof(int32_t));
          span.token_ids = tokens[kSpanCount + i].data();
        }
      }
      local_continuation.spans = local_next.data();
      local_continuation.output_limits = next_limits.data();
    }
    auto result = loom_serve_text_model_submit_epoch(
        model, shape, kSpanCount, local_spans.data(),
        limits ? local_limits.data() : nullptr,
        continuation ? &local_continuation : nullptr, &frontier);
    // These are deliberately not invocation-owned. Poison both descriptors and
    // input payloads, then leave their stack scope before registering
    // readiness.
    memset(local_spans.data(), 0xFF, sizeof(local_spans));
    memset(local_next.data(), 0xFF, sizeof(local_next));
    memset(tokens.data(), 0xFF, sizeof(tokens));
    memset(local_limits.data(), 0xFF, sizeof(local_limits));
    memset(next_limits.data(), 0xFF, sizeof(next_limits));
    memset(&local_continuation, 0xFF, sizeof(local_continuation));
    return result;
  }

  static iree_status_t Submit(iree_task_process_t* process,
                              const iree_task_worker_context_t*,
                              iree_task_process_drain_result_t* result) {
    auto* self = static_cast<ModelWork*>(process->user_data);
    auto status = iree_ok_status();
    bool pending = true;
    switch (self->operation) {
      case Operation::kEpoch:
        status = self->SubmitCopies();
        break;
      case Operation::kTrim:
        status = loom_serve_text_model_submit_trim(
            self->model, &self->trim_result, &self->frontier);
        break;
      case Operation::kReserve: {
        auto& reservation = self->reservation;
        status = loom_serve_text_row_submit_reserve(
            reservation.row, reservation.mode, reservation.checkpoint,
            reservation.extent, &reservation.submitted, &self->frontier);
        pending = reservation.submitted;
        if (iree_status_is_ok(status) && pending &&
            loom_serve_device_memory_pool(
                loom_serve_text_model_device(self->model))) {
          // Before completion consumption, even a cancelled or denied request
          // must retain its activation pin. This call cannot unmap weights.
          auto deactivation = loom_serve_text_model_deactivate(self->model);
          if (iree_status_is_failed_precondition(deactivation)) {
            iree_status_free(deactivation);
          } else {
            status = iree_status_join(
                deactivation,
                iree_make_status(IREE_STATUS_DATA_LOSS,
                                 "pending reservation did not pin weights"));
          }
        }
        break;
      }
      case Operation::kPin:
        if (self->checkpoint.cancellation != Cancellation::kBeforeSubmission) {
          status = loom_serve_text_row_submit_pin(
              self->checkpoint.row, &self->checkpoint.value, &self->frontier);
        }
        pending = self->checkpoint.value != nullptr;
        break;
      case Operation::kSuspend:
        self->checkpoint.available =
            loom_serve_text_model_pool_usage(self->model).available;
        status = loom_serve_text_checkpoint_submit_suspend(
            self->checkpoint.value, &self->frontier);
        if (iree_status_is_ok(status) &&
            loom_serve_text_model_pool_usage(self->model).available !=
                self->checkpoint.available) {
          status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                    "capture released IDs before completion");
        }
        break;
      case Operation::kReset:
        loom_serve_text_row_reset_retired(self->reset_row);
        pending = false;
        break;
    }
    if (iree_status_is_ok(status) && pending) {
      if (self->checkpoint.fail_readiness) {
        // Fail the observed feedback branch, not the queued capture. Outer
        // destruction must still join that work before releasing its source.
        iree_hal_semaphore_fail(self->frontier.semaphores[1],
                                iree_status_from_code(IREE_STATUS_CANCELLED));
      }
      for (size_t i = 0; i < self->semaphores.size(); ++i) {
        self->semaphores[i] = reinterpret_cast<iree_async_semaphore_t*>(
            self->frontier.semaphores[i]);
      }
      self->wait.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
      self->wait.base.completion_fn = Ready;
      self->wait.base.user_data = self;
      self->wait.semaphores = self->semaphores.data();
      self->wait.values = self->frontier.values;
      self->wait.count = self->semaphores.size();
      self->wait.mode = IREE_ASYNC_WAIT_MODE_ALL;
      status = iree_async_proactor_submit_one(self->proactor, &self->wait.base);
    }
    if (!iree_status_is_ok(status) || !pending) {
      Ready(self, nullptr, status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
    result->did_work = true;
    result->completed = true;
    return iree_ok_status();
  }

  static void Ready(void* user_data, iree_async_operation_t*,
                    iree_status_t status, iree_async_completion_flags_t) {
    auto* self = static_cast<ModelWork*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = iree_status_join(self->status, status);
    iree_task_executor_schedule_process(self->executor, &self->complete);
  }

  static iree_status_t Complete(iree_task_process_t* process,
                                const iree_task_worker_context_t*,
                                iree_task_process_drain_result_t* result) {
    auto* self = static_cast<ModelWork*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    for (size_t i = 0; (self->operation == Operation::kEpoch ||
                        self->operation == Operation::kTrim) &&
                       i < kSpanCount && iree_status_is_ok(self->status);
         ++i) {
      auto* row =
          loom_serve_text_model_row(self->model, self->spans[i].row_index);
      if (loom_serve_text_row_position(row) != self->positions[i]) {
        self->status =
            iree_make_status(IREE_STATUS_DATA_LOSS,
                             "row advanced before completion consumption");
      }
    }
    if (iree_status_is_ok(self->status)) {
      switch (self->operation) {
        case Operation::kEpoch:
          self->status = loom_serve_text_model_complete_epoch(
              self->model, self->results.data());
          break;
        case Operation::kTrim:
          self->status = loom_serve_text_model_complete_trim(
              self->model, &self->trim_result);
          break;
        case Operation::kReserve: {
          auto& reservation = self->reservation;
          reservation.admitted =
              reservation.submitted &&
              loom_serve_text_row_complete_reserve(reservation.row);
          if (reservation.admitted && reservation.cancel) {
            loom_serve_text_row_release_reservation(reservation.row);
          }
          break;
        }
        case Operation::kPin:
        case Operation::kSuspend:
          if (self->operation == Operation::kSuspend) {
            loom_serve_text_checkpoint_complete_suspend(self->checkpoint.value);
          }
          if (self->checkpoint.cancellation == Cancellation::kAfterSubmission) {
            loom_serve_text_checkpoint_release(
                std::exchange(self->checkpoint.value, nullptr));
          }
          self->checkpoint.completed = true;
          break;
        case Operation::kReset:
          break;
      }
    }
    result->did_work = true;
    result->completed = true;
    return iree_ok_status();
  }

  static void Release(iree_task_process_t* process) {
    auto* self = static_cast<ModelWork*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    ++self->released;
    self->condition.notify_one();
  }

  iree_status_t Run() {
    auto* host = loom_serve_device_host(loom_serve_text_model_device(model));
    executor = loom_serve_host_executor(host);
    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_get(
        loom_serve_host_proactor_pool(host), 0, &proactor));
    for (size_t i = 0; i < kSpanCount; ++i) {
      positions[i] = loom_serve_text_row_position(
          loom_serve_text_model_row(model, spans[i].row_index));
    }
    iree_task_process_initialize(Submit, 0, 1, &submit);
    iree_task_process_initialize(Complete, 0, 1, &complete);
    for (auto* process : {&submit, &complete}) {
      process->user_data = this;
      process->release_fn = Release;
    }
    iree_task_executor_schedule_process(executor, &submit);
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return released == 2; });
    return std::exchange(status, iree_ok_status());
  }
};

static size_t ReferenceRow(size_t row) {
  return row == kRows[0] ? kReferenceRows[0] : kReferenceRows[1];
}

static iree_status_t SelectShape(loom_serve_text_model_t* model,
                                 const Spans& spans, size_t* out_shape) {
  const size_t count = spans[0].token_count + spans[1].token_count;
  const size_t shape_count = loom_serve_text_model_shape_count(model);
  const auto* shapes = loom_serve_text_model_shapes(model);
  for (size_t i = 0; i < shape_count; ++i) {
    if (shapes[i].span_capacity >= kSpanCount &&
        shapes[i].token_capacity >= count) {
      *out_shape = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "provide an epoch admitting %zu tokens and two spans",
                          count);
}

static iree_status_t Compare(const char* label, ModelWork& epoch) {
  IREE_RETURN_IF_ERROR(SelectShape(epoch.model, epoch.spans, &epoch.shape));
  IREE_RETURN_IF_ERROR(epoch.Run());
  Spans reference = epoch.spans;
  for (auto& span : reference) {
    span.row_index = ReferenceRow(span.row_index);
  }
  Spans next = {};
  loom_serve_text_continuation_t continuation = {};
  if (epoch.continuation) {
    continuation = *epoch.continuation;
    for (size_t i = 0; i < continuation.span_count; ++i) {
      next[i] = continuation.spans[i];
      next[i].row_index = ReferenceRow(next[i].row_index);
    }
    continuation.spans = next.data();
  }
  Results expected = {};
  if (epoch.limits) {
    IREE_RETURN_IF_ERROR(loom_serve_text_model_verify(
        epoch.model, epoch.shape, kSpanCount, reference.data(), epoch.limits,
        epoch.continuation ? &continuation : nullptr, expected.data()));
  } else {
    IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(
        epoch.model, epoch.shape, kSpanCount, reference.data()));
    for (size_t i = 0; i < kSpanCount; ++i) {
      expected[i].consumed_count = reference[i].token_count;
      expected[i].known_count = reference[i].token_count;
      if (iree_any_bit_set(reference[i].flags,
                           LOOM_SERVE_TEXT_SPAN_FLAG_SELECT)) {
        expected[i].output_count = 1;
        expected[i].tokens[0] = loom_serve_text_row_token(
            loom_serve_text_model_row(epoch.model, reference[i].row_index));
      }
    }
  }
  size_t outputs = 0;
  for (size_t i = 0; i < kSpanCount; ++i) {
    const auto& actual = epoch.results[i];
    const auto& control = expected[i];
    auto* row =
        loom_serve_text_model_row(epoch.model, epoch.spans[i].row_index);
    auto* reference_row =
        loom_serve_text_model_row(epoch.model, reference[i].row_index);
    if (actual.consumed_count != control.consumed_count ||
        actual.known_count != control.known_count ||
        actual.output_count != control.output_count ||
        actual.verification_count != control.verification_count ||
        memcmp(actual.tokens, control.tokens,
               actual.output_count * sizeof(int32_t)) != 0 ||
        loom_serve_text_row_position(row) !=
            loom_serve_text_row_position(reference_row) ||
        (actual.output_count && loom_serve_text_row_token(row) !=
                                    loom_serve_text_row_token(reference_row))) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "%s changed row %zu output or position", label,
                              i);
    }
    outputs += actual.output_count;
  }
  fprintf(stdout,
          "{\"event\":\"epoch_check\",\"case\":\"%s\",\"spans\":%zu,"
          "\"output_tokens\":%zu,\"work_frontier\":%" PRIu64
          ",\"feedback_frontier\":%" PRIu64 "}\n",
          label, kSpanCount, outputs, epoch.frontier.values[0],
          epoch.frontier.values[1]);
  fflush(stdout);
  return iree_ok_status();
}

static iree_status_t Run(loom_serve_text_model_t* model,
                         iree_allocator_t allocator,
                         iree_status_code_t* out_terminal_code) {
  *out_terminal_code = IREE_STATUS_OK;
  const iree_string_view_t prompts[] = {
      IREE_SV(
          "Repeat these words: amber cedar maple violet birch willow falcon."),
      IREE_SV(
          "Repeat these words: silver pine oak robin copper hazel sparrow."),
  };
  std::array<std::array<int32_t, kTokenCapacity>, kSpanCount> inputs = {};
  std::array<size_t, kSpanCount> counts = {};
  for (size_t i = 0; i < kSpanCount; ++i) {
    IREE_RETURN_IF_ERROR(loom_serve_text_chat_prepare_input(
        loom_serve_text_model_chat_policy(model), prompts[i],
        LOOM_SERVE_TEXT_CHAT_INPUT_USER, LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
        kTokenCapacity, inputs[i].data(), &counts[i], allocator));
  }
  const size_t prefix = counts[0] / 2;
  // Activate cold weights, then cancel before any epoch consumes the row.
  // Completion owns the pin even after the client has abandoned the request.
  ModelWork cancelled;
  cancelled.operation = ModelWork::Operation::kReserve;
  cancelled.model = model;
  cancelled.reservation.row = loom_serve_text_model_row(model, 4);
  cancelled.reservation.extent = 1;
  cancelled.reservation.cancel = true;
  IREE_RETURN_IF_ERROR(cancelled.Run());
  if (!cancelled.reservation.admitted ||
      loom_serve_text_row_pool_usage(cancelled.reservation.row)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "cancelled activation retained reservation credit");
  }
  ModelWork reset;
  reset.operation = ModelWork::Operation::kReset;
  reset.model = model;
  reset.reset_row = cancelled.reservation.row;
  IREE_RETURN_IF_ERROR(reset.Run());
  fprintf(stdout,
          "{\"event\":\"admission_check\","
          "\"case\":\"cancel-after-activation\"}\n");
  fflush(stdout);
  // Low physical IDs belong to independent ballast rows. Releasing them after
  // the target/reference work guarantees a fragmented live pool for trim.
  const Spans ballast = {{{7, counts[0], inputs[0].data(), 0},
                          {6, counts[1], inputs[1].data(), 0}}};
  size_t ballast_shape = 0;
  IREE_RETURN_IF_ERROR(SelectShape(model, ballast, &ballast_shape));
  IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(
      model, ballast_shape, ballast.size(), ballast.data()));
  for (size_t i = 0; i < kSpanCount; ++i) {
    ModelWork fresh;
    fresh.operation = ModelWork::Operation::kReserve;
    fresh.model = model;
    fresh.reservation.row = loom_serve_text_model_row(model, kRows[i]);
    fresh.reservation.extent = counts[i] + 32;
    IREE_RETURN_IF_ERROR(fresh.Run());
    if (!fresh.reservation.admitted) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fixture cannot admit fresh reservation");
    }
  }
  ModelWork prefill;
  prefill.model = model;
  prefill.spans = {{{kRows[0], prefix, inputs[0].data(), 0},
                    {kRows[1], counts[1], inputs[1].data(),
                     LOOM_SERVE_TEXT_SPAN_FLAG_SELECT}}};
  IREE_RETURN_IF_ERROR(Compare("partial-prefill", prefill));
  int32_t token =
      loom_serve_text_row_token(loom_serve_text_model_row(model, kRows[1]));
  ModelWork mixed;
  mixed.model = model;
  mixed.spans = {{{kRows[1], 1, &token, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
                  {kRows[0], counts[0] - prefix, inputs[0].data() + prefix,
                   LOOM_SERVE_TEXT_SPAN_FLAG_SELECT}}};
  IREE_RETURN_IF_ERROR(Compare("prefill-decode", mixed));
  for (size_t i = 0; i < kSpanCount; ++i) {
    auto* row = loom_serve_text_model_row(model, kRows[i]);
    loom_serve_text_row_release_reservation(row);
    ModelWork continuing;
    continuing.operation = ModelWork::Operation::kReserve;
    continuing.model = model;
    continuing.reservation.row = row;
    continuing.reservation.mode = LOOM_SERVE_TEXT_RESERVE_CONTINUE;
    continuing.reservation.extent = counts[i] + 32;
    IREE_RETURN_IF_ERROR(continuing.Run());
    if (!continuing.reservation.admitted) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fixture cannot admit continuing reservation");
    }
  }
  if (loom_serve_text_mtp_from_flags()) {
    for (size_t pass = 0; pass < 2; ++pass) {
      std::array<int32_t, kSpanCount> anchors = {};
      uint32_t limits[kSpanCount] = {4, 4};
      ModelWork speculative;
      speculative.model = model;
      speculative.limits = limits;
      for (size_t i = 0; i < kSpanCount; ++i) {
        auto* row = loom_serve_text_model_row(model, kRows[i]);
        if (loom_serve_text_row_is_eos(row)) {
          return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "fixture ended before speculative work");
        }
        anchors[i] = loom_serve_text_row_token(row);
        speculative.spans[i] = {kRows[i], 4, &anchors[i],
                                LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                                    LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE};
      }
      Spans next = {};
      uint32_t next_limits[kSpanCount] = {8, 4};
      loom_serve_text_continuation_t continuation = {};
      if (pass) {
        limits[0] = 0;
        limits[1] = 8;
        speculative.spans[0].token_count = 1;
        speculative.spans[0].flags = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT;
        for (size_t i = 0; i < kSpanCount; ++i) {
          next[i] = {kRows[kSpanCount - 1 - i], 4, nullptr,
                     LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                         LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE};
        }
        IREE_RETURN_IF_ERROR(
            SelectShape(model, next, &continuation.shape_index));
        continuation.span_count = kSpanCount;
        continuation.spans = next.data();
        continuation.output_limits = next_limits;
        speculative.continuation = &continuation;
      }
      IREE_RETURN_IF_ERROR(
          Compare(pass ? "mtp-continuation" : "mtp", speculative));
    }
  }
  for (size_t index : kRows) {
    loom_serve_text_row_release_reservation(
        loom_serve_text_model_row(model, index));
  }
  for (size_t row : {7u, 6u}) {
    ModelWork release;
    release.operation = ModelWork::Operation::kReset;
    release.model = model;
    release.reset_row = loom_serve_text_model_row(model, row);
    IREE_RETURN_IF_ERROR(release.Run());
  }
  ModelWork trim;
  trim.operation = ModelWork::Operation::kTrim;
  trim.model = model;
  trim.spans = mixed.spans;
  IREE_RETURN_IF_ERROR(trim.Run());
  const bool compactable =
      loom_serve_device_memory_pool(loom_serve_text_model_device(model)) &&
      loom_serve_text_model_pool_usage(model).capacity;
  if ((trim.trim_result.moved_blocks != 0) != compactable) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "compaction did not match the backing policy");
  }
  fprintf(stdout,
          "{\"event\":\"maintenance_check\",\"case\":\"async-trim\","
          "\"moved_blocks\":%u,\"copied_bytes\":%" PRIu64
          ",\"released_bytes\":%" PRIu64 "}\n",
          trim.trim_result.moved_blocks, trim.trim_result.copied_bytes,
          trim.trim_result.released_bytes);
  fflush(stdout);
  std::array<int32_t, kSpanCount> anchors = {};
  ModelWork decode;
  decode.model = model;
  for (size_t i = 0; i < kSpanCount; ++i) {
    anchors[i] =
        loom_serve_text_row_token(loom_serve_text_model_row(model, kRows[i]));
    decode.spans[i] = {kRows[i], 1, &anchors[i],
                       LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
  }
  IREE_RETURN_IF_ERROR(Compare("decode-after-compaction", decode));
  if (compactable && loom_serve_text_model_checkpoint_capacity(model)) {
    auto* source = loom_serve_text_model_row(model, kRows[0]);
    const size_t available = loom_serve_text_model_pool_usage(model).available;
    for (auto cancellation : {ModelWork::Cancellation::kBeforeSubmission,
                              ModelWork::Cancellation::kAfterSubmission}) {
      ModelWork abandoned;
      abandoned.operation = ModelWork::Operation::kPin;
      abandoned.model = model;
      abandoned.checkpoint.row = source;
      abandoned.checkpoint.cancellation = cancellation;
      IREE_RETURN_IF_ERROR(abandoned.Run());
      if (!abandoned.checkpoint.completed || abandoned.checkpoint.value ||
          loom_serve_text_model_pool_usage(model).available != available) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "cancelled pin changed pool ownership");
      }
    }
    ModelWork pin;
    pin.operation = ModelWork::Operation::kPin;
    pin.model = model;
    pin.checkpoint.row = source;
    IREE_RETURN_IF_ERROR(pin.Run());
    auto* checkpoint = pin.checkpoint.value;
    if (!checkpoint) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fixture cannot pin endpoint");
    }
    if (loom_serve_text_model_checkpoint_capacity(model) == 1) {
      ModelWork denied;
      denied.operation = ModelWork::Operation::kPin;
      denied.model = model;
      denied.checkpoint.row = source;
      IREE_RETURN_IF_ERROR(denied.Run());
      if (denied.checkpoint.value) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "pin exceeded configured endpoint capacity");
      }
    }
    // The second submission exercises an already-cold endpoint without
    // returning its physical IDs twice or replacing its retained image.
    for (size_t iteration = 0; iteration < 2; ++iteration) {
      ModelWork capture;
      capture.operation = ModelWork::Operation::kSuspend;
      capture.model = model;
      capture.checkpoint.value = checkpoint;
      IREE_RETURN_IF_ERROR(capture.Run());
      if (!capture.checkpoint.completed ||
          !loom_serve_text_checkpoint_suspended_bytes(checkpoint)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "capture did not publish a cold endpoint");
      }
    }
    ModelWork restoring;
    restoring.operation = ModelWork::Operation::kReserve;
    restoring.model = model;
    restoring.reservation.row = loom_serve_text_model_row(model, 4);
    restoring.reservation.mode = LOOM_SERVE_TEXT_RESERVE_CHECKPOINT;
    restoring.reservation.checkpoint = checkpoint;
    restoring.reservation.extent = loom_serve_text_row_position(source) + 4;
    IREE_RETURN_IF_ERROR(restoring.Run());
    if (!restoring.reservation.admitted ||
        loom_serve_text_checkpoint_suspended_bytes(checkpoint)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "cold checkpoint reservation was not restored");
    }
    const int32_t anchor = loom_serve_text_row_token(source);
    ModelWork fork;
    fork.model = model;
    fork.spans = {{{4, 1, &anchor, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
                   {kRows[0], 1, &anchor, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT}}};
    IREE_RETURN_IF_ERROR(SelectShape(model, fork.spans, &fork.shape));
    IREE_RETURN_IF_ERROR(fork.Run());
    if (fork.results[0].output_count != 1 ||
        fork.results[1].output_count != 1 ||
        fork.results[0].tokens[0] != fork.results[1].tokens[0] ||
        loom_serve_text_row_position(restoring.reservation.row) !=
            loom_serve_text_row_position(source)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "cold admission changed continuation");
    }
    loom_serve_text_row_release_reservation(restoring.reservation.row);
    ModelWork abandoned_capture;
    abandoned_capture.operation = ModelWork::Operation::kSuspend;
    abandoned_capture.model = model;
    abandoned_capture.checkpoint.value = checkpoint;
    abandoned_capture.checkpoint.cancellation =
        ModelWork::Cancellation::kAfterSubmission;
    IREE_RETURN_IF_ERROR(abandoned_capture.Run());
    if (!abandoned_capture.checkpoint.completed ||
        abandoned_capture.checkpoint.value) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "cancelled capture retained its endpoint");
    }
    fprintf(stdout,
            "{\"event\":\"admission_check\","
            "\"case\":\"cold-checkpoint-continuation\","
            "\"output_tokens\":2}\n");
    fflush(stdout);

    // Keep a sole endpoint owner after both branches retire. A failed ALL wait
    // must leave its original IDs retained, even if capture itself succeeds.
    ModelWork terminal_pin;
    terminal_pin.operation = ModelWork::Operation::kPin;
    terminal_pin.model = model;
    terminal_pin.checkpoint.row = source;
    IREE_RETURN_IF_ERROR(terminal_pin.Run());
    if (!terminal_pin.checkpoint.value) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fixture cannot pin terminal endpoint");
    }
    for (auto* row : {source, restoring.reservation.row}) {
      ModelWork release;
      release.operation = ModelWork::Operation::kReset;
      release.model = model;
      release.reset_row = row;
      IREE_RETURN_IF_ERROR(release.Run());
    }
    ModelWork failed_capture;
    failed_capture.operation = ModelWork::Operation::kSuspend;
    failed_capture.model = model;
    failed_capture.checkpoint.value = terminal_pin.checkpoint.value;
    failed_capture.checkpoint.fail_readiness = true;
    auto failure = failed_capture.Run();
    if (!iree_status_is_cancelled(failure)) {
      return iree_status_join(
          failure,
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "capture did not observe failed readiness"));
    }
    iree_status_free(failure);
    if (failed_capture.checkpoint.completed ||
        loom_serve_text_model_pool_usage(model).available !=
            failed_capture.checkpoint.available) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "failed readiness published or released capture");
    }
    *out_terminal_code = IREE_STATUS_CANCELLED;
    fprintf(stdout,
            "{\"event\":\"checkpoint_check\","
            "\"case\":\"failed-readiness-retains-source\"}\n");
    fflush(stdout);
  }
  return iree_ok_status();
}

}  // namespace

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const auto allocator = iree_allocator_system();
  loom_serve_host_t* host = nullptr;
  loom_serve_device_t* device = nullptr;
  loom_serve_text_model_t* model = nullptr;
  iree_status_code_t terminal_code = IREE_STATUS_OK;
  auto status = loom_serve_host_create(1, &host, allocator);
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_create_from_flags(host, &device, allocator);
  }
  const loom_serve_text_flag_defaults_t defaults = {.row_count = 8};
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_model_create_from_flags(device, &defaults, &model,
                                                     allocator);
  }
  if (iree_status_is_ok(status)) {
    status = Run(model, allocator, &terminal_code);
  }
  auto destruction = loom_serve_text_model_destroy(model);
  if (iree_status_code(destruction) == terminal_code) {
    iree_status_free(destruction);
  } else {
    status = iree_status_join(
        status, iree_status_join(
                    destruction,
                    iree_make_status(IREE_STATUS_DATA_LOSS,
                                     "model teardown lost terminal status")));
  }
  status = iree_status_join(status, loom_serve_device_destroy(device));
  loom_serve_host_release(host);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  fprintf(stdout,
          "PASS: one-worker text admission, epochs, compaction and checkpoints "
          "preserve "
          "ownership, output and row publication.\n");
  return EXIT_SUCCESS;
}
