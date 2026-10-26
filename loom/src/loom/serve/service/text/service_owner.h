// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SERVICE_OWNER_H_
#define LOOM_SERVE_SERVICE_TEXT_SERVICE_OWNER_H_

#include "iree/base/internal/arena.h"
#include "loom/serve/service/execution_reconfiguration.h"
#include "loom/serve/service/text/checkpoint_request.h"
#include "loom/serve/service/text/epoch.h"
#include "loom/serve/service/text/heartbeat.h"
#include "loom/serve/service/text/input_count.h"
#include "loom/serve/service/text/pending_queue.h"
#include "loom/serve/service/text/readiness.h"
#include "loom/serve/service/text/service.h"
#include "loom/serve/service/text/service_capacity.h"
#include "loom/serve/telemetry/tracing.h"

// Device work retained until readiness is consumed by the model owner.
typedef enum loom_serve_text_work_e {
  TEXT_WORK_NONE,
  TEXT_WORK_EPOCH,
  TEXT_WORK_CHECKPOINT,
  TEXT_WORK_TRIM,
  TEXT_WORK_RESERVE,
  TEXT_WORK_RETAIN,
  TEXT_WORK_PREFIX,
  TEXT_WORK_GROW,
} loom_serve_text_work_t;

// Exclusive host/VM borrow; input preparation may overlap an accepted epoch.
typedef enum loom_serve_text_cpu_work_e {
  TEXT_CPU_WORK_NONE,
  TEXT_CPU_WORK_PREPARE,
  TEXT_CPU_WORK_OUTPUT,
  TEXT_CPU_WORK_INPUT_COUNT,
  TEXT_CPU_WORK_GROW,
} loom_serve_text_cpu_work_t;

// Serialized service ownership shared by request admission, device epochs,
// and retirement. CPU jobs borrow only the fields described by their phase.
struct loom_serve_text_service_t {
  // Allocator for the service's host-only storage.
  iree_allocator_t host_allocator;
  // Borrowed model residency and its single host execution owner.
  loom_serve_text_model_t* model;
  // Borrowed optional image computation with independent preparation/residency.
  loom_serve_finite_component_t* image;
  // Lazy independent projection owner; text-only requests allocate no window.
  loom_serve_text_media_projector_t* projector;
  // Last receipt in the current finite lookahead window; later arrivals do not
  // postpone a gathered cohort. Zero means no lookahead is pending.
  uint64_t media_preparation_end;
  // Native choice pool and text policy, joined before model retirement.
  loom_serve_execution_reconfiguration_t* reconfiguration;
  // Borrowed deployment name, independent of the source policy identity.
  iree_string_view_t name;
  // Owned fallback shape for isolated execution.
  loom_serve_packing_shape_t isolated_shape;
  // Number of allocated device/session rows.
  iree_host_size_t row_count;
  // Resident owner ceiling, independent of epoch shapes; zero is unbounded.
  iree_host_size_t max_sessions;
  // Cold replacement directory, private to its CPU task until readiness joins.
  loom_serve_text_service_capacity_t growth;
  // Output bound when the client omits one.
  iree_host_size_t default_max_tokens;
  // Immutable model epoch geometry and execution choices.
  loom_serve_text_epoch_options_t scheduling;
  // Rotating admission priority, independent of row or request identity.
  iree_host_size_t cursor;
  // Irreversible admission seal after stop or a terminal service failure.
  bool stopping;
  // Proactor-timed reporting of copied state, independent of model readiness.
  loom_serve_text_heartbeat_t heartbeat;
  // Independent CPU/device publication; only the owner mutates model state.
  loom_serve_text_readiness_t readiness;
  // Owner's device operation, retained until readiness is consumed.
  struct {
    // Owner-only operation discriminator; NONE means no callback is accepted.
    loom_serve_text_work_t work;
    // Exact work and feedback values captured by submission.
    loom_serve_execution_frontier_t frontier;
    // Host observation before registering this operation's readiness wait.
    iree_time_t start_time;
    // Accepted epoch through readiness collection; zero for maintenance work.
    IREE_TRACE(iree_section_id_t section;)
  } completion;
  // Exclusive host/VM borrow, cleared only after task process release. Input
  // preparation selects among idle rows and may overlap device epoch work on
  // active rows; the owner leaves all idle completions unchanged until release.
  loom_serve_text_cpu_work_t cpu_work;
  // Retained cohort inputs and feedback through response CPU release.
  loom_serve_text_epoch_t epoch;
  // Independent known-input output bytes retained through response CPU release.
  loom_serve_text_capture_snapshot_t* capture;
  // One delivery receipt retaining validated names through device readiness.
  loom_serve_text_checkpoint_operation_t control;
  // Stateless CPU input queries, independent of generation rows and capacity.
  loom_serve_text_input_count_t input_counts;
  // Single-owner observation; CPU jobs never publish into this stream.
  loom_serve_text_telemetry_t telemetry;
  // Stable queued claims and selected-attempt bookkeeping, not device rows.
  loom_serve_text_pending_queue_t pending;
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
  // Named native endpoints and their immutable host continuation state.
  loom_serve_text_checkpoint_registry_t checkpoints;
  // Evictable idle histories, independent of execution rows and explicit pins.
  loom_serve_text_session_cache_t cache;
  // Shared consumed-token frontiers, independent of completed chat histories.
  loom_serve_text_prefix_cache_t prefixes;
  // Shared CPU encoding scratch, copied into consumer-owned staged output.
  iree_string_builder_t scratch;
  // Shared validated tool records used after complete source parsing.
  loom_serve_text_tool_calls_t tool_calls;
  // Completion parser/formatter backing reused by all serialized output jobs.
  loom_serve_text_chat_scratch_t completion_scratch;
  // Stable retained records backed by the shared application host block pool.
  iree_arena_allocator_t session_records;
  // Owned directory sharing one model, command set and timeline.
  loom_serve_text_session_t** sessions;
  // Reusable readiness entries, indexed by resident identity, not span ordinal.
  loom_serve_ready_span_t* ready;
};

// Whether a CPU or device loan excludes another model-owner operation.
bool loom_serve_text_has_work(const loom_serve_text_service_t* service);

// Registers readiness for the just-submitted frontier without waiting.
iree_status_t loom_serve_text_await_work(loom_serve_text_service_t* service,
                                         loom_serve_text_work_t work);

// Lends the model and phase-specific host payload to one finite CPU job.
void loom_serve_text_schedule_cpu(loom_serve_text_service_t* service,
                                  loom_serve_text_cpu_work_t work,
                                  loom_serve_text_cpu_work_fn_t run);

// Publishes owner-only model/session facts to the heartbeat snapshot.
void loom_serve_text_observe(loom_serve_text_service_t* service,
                             const char* phase);

#endif  // LOOM_SERVE_SERVICE_TEXT_SERVICE_OWNER_H_
