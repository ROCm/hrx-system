// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real-weight epoch ownership witness using one shared JIT/serving worker.

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
#include "iree/task/executor.h"

namespace {

constexpr size_t kSpanCount = 2;
constexpr size_t kTokenCapacity = 512;
constexpr std::array<size_t, kSpanCount> kRows = {5, 2};
constexpr std::array<size_t, kSpanCount> kReferenceRows = {1, 0};
using Spans = std::array<loom_serve_text_span_t, kSpanCount>;
using Results = std::array<loom_serve_text_result_t, kSpanCount>;

struct Epoch {
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
    auto* self = static_cast<Epoch*>(process->user_data);
    auto status = self->SubmitCopies();
    if (iree_status_is_ok(status)) {
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
    if (!iree_status_is_ok(status)) {
      Ready(self, nullptr, status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
    result->did_work = true;
    result->completed = true;
    return iree_ok_status();
  }

  static void Ready(void* user_data, iree_async_operation_t*,
                    iree_status_t status, iree_async_completion_flags_t) {
    auto* self = static_cast<Epoch*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = iree_status_join(self->status, status);
    iree_task_executor_schedule_process(self->executor, &self->complete);
  }

  static iree_status_t Complete(iree_task_process_t* process,
                                const iree_task_worker_context_t*,
                                iree_task_process_drain_result_t* result) {
    auto* self = static_cast<Epoch*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    for (size_t i = 0; i < kSpanCount && iree_status_is_ok(self->status); ++i) {
      auto* row =
          loom_serve_text_model_row(self->model, self->spans[i].row_index);
      if (loom_serve_text_row_position(row) != self->positions[i]) {
        self->status =
            iree_make_status(IREE_STATUS_DATA_LOSS,
                             "row advanced before completion consumption");
      }
    }
    if (iree_status_is_ok(self->status)) {
      self->status = loom_serve_text_model_complete_epoch(self->model,
                                                          self->results.data());
    }
    result->did_work = true;
    result->completed = true;
    return iree_ok_status();
  }

  static void Release(iree_task_process_t* process) {
    auto* self = static_cast<Epoch*>(process->user_data);
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

static iree_status_t Compare(const char* label, Epoch& epoch) {
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
                         iree_allocator_t allocator) {
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
  Epoch prefill;
  prefill.model = model;
  prefill.spans = {{{kRows[0], prefix, inputs[0].data(), 0},
                    {kRows[1], counts[1], inputs[1].data(),
                     LOOM_SERVE_TEXT_SPAN_FLAG_SELECT}}};
  IREE_RETURN_IF_ERROR(Compare("partial-prefill", prefill));
  int32_t token =
      loom_serve_text_row_token(loom_serve_text_model_row(model, kRows[1]));
  Epoch mixed;
  mixed.model = model;
  mixed.spans = {{{kRows[1], 1, &token, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
                  {kRows[0], counts[0] - prefix, inputs[0].data() + prefix,
                   LOOM_SERVE_TEXT_SPAN_FLAG_SELECT}}};
  IREE_RETURN_IF_ERROR(Compare("prefill-decode", mixed));
  if (loom_serve_text_mtp_from_flags()) {
    for (size_t pass = 0; pass < 2; ++pass) {
      std::array<int32_t, kSpanCount> anchors = {};
      uint32_t limits[kSpanCount] = {4, 4};
      Epoch speculative;
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
  std::array<int32_t, kSpanCount> anchors = {};
  Epoch decode;
  decode.model = model;
  for (size_t i = 0; i < kSpanCount; ++i) {
    anchors[i] =
        loom_serve_text_row_token(loom_serve_text_model_row(model, kRows[i]));
    decode.spans[i] = {kRows[i], 1, &anchors[i],
                       LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
  }
  return Compare("decode-after-reuse", decode);
}

}  // namespace

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const auto allocator = iree_allocator_system();
  loom_serve_host_t* host = nullptr;
  loom_serve_device_t* device = nullptr;
  loom_serve_text_model_t* model = nullptr;
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
    status = Run(model, allocator);
  }
  status = iree_status_join(status, loom_serve_text_model_destroy(model));
  status = iree_status_join(status, loom_serve_device_destroy(device));
  loom_serve_host_release(host);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  fprintf(stdout,
          "PASS: one-worker text epochs preserve copied plans, output "
          "and row publication.\n");
  return EXIT_SUCCESS;
}
