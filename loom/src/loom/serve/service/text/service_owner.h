// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SERVICE_OWNER_H_
#define LOOM_SERVE_SERVICE_TEXT_SERVICE_OWNER_H_

#include "iree/base/internal/arena.h"
#include "loom/runtime/execution/wait.h"
#include "loom/runtime/host/readiness.h"
#include "loom/serve/service/text/checkpoint_request.h"
#include "loom/serve/service/text/epoch.h"
#include "loom/serve/service/text/heartbeat.h"
#include "loom/serve/service/text/input_count.h"
#include "loom/serve/service/text/pending_queue.h"
#include "loom/serve/service/text/service.h"
#include "loom/serve/service/text/service_capacity.h"
#include "loom/service/execution_reconfiguration.h"
#include "loom/service/telemetry/tracing.h"

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

// One source-selected preparation opportunity, captured without lending rows
// or starting a CPU task. The serialized owner accepts it before any mutation
// of the pending queue or resident directory can invalidate the selection.
typedef struct loom_serve_text_preparation_t {
  // Stable queued receipt; NULL denotes no legal preparation opportunity.
  loom_serve_text_pending_request_t* request;
  // Captured FIFO position, independent of the receipt's stable address.
  iree_host_size_t index;
  // Selected idle row, or NULL when resident-directory growth comes first.
  loom_serve_text_session_t* session;
  // Requested resident capacity for growth; zero for an existing row.
  iree_host_size_t growth_count;
} loom_serve_text_preparation_t;

// Independently retained ownership, not source stages or queued requests.
typedef enum loom_serve_text_operation_slot_e {
  TEXT_OPERATION_ADMISSION,
  TEXT_OPERATION_INPUT_COUNT,
  TEXT_OPERATION_EXECUTION,
  TEXT_OPERATION_COUNT,
} loom_serve_text_operation_slot_t;

// The source retains one legal choice using its existing planning storage.
typedef enum loom_serve_text_choice_e {
  TEXT_CHOICE_NONE,
  TEXT_CHOICE_PREPARATION,
  TEXT_CHOICE_INPUT_COUNT,
  TEXT_CHOICE_EPOCH,
  TEXT_CHOICE_CHECKPOINT,
} loom_serve_text_choice_t;

// Serialized service ownership shared by request admission, device epochs,
// and retirement. CPU jobs borrow only the fields described by their phase.
struct loom_serve_text_service_t {
  // Allocator for the service's host-only storage.
  iree_allocator_t host_allocator;
  // Borrowed model residency and its single host execution owner.
  loom_serve_text_model_t* model;
  // Borrowed optional image computation with independent preparation/residency.
  loom_service_finite_component_t* image;
  // Borrowed projection control, surviving each quiescent projector
  // replacement.
  loom_runtime_service_control_binding_t image_scheduling;
  // Monotonic offer/operation identities across those replacements.
  uint64_t image_next_identity;
  // Lazy independent projection owner; text-only requests allocate no window.
  loom_serve_text_media_projector_t* projector;
  // Last receipt in the current finite lookahead window; later arrivals do not
  // postpone a gathered cohort. Zero means no lookahead is pending.
  uint64_t media_preparation_end;
  // Native choice pool and text policy, joined before model retirement.
  loom_service_execution_reconfiguration_t* reconfiguration;
  // Serialized scheduling facts; asynchronous workers never access this state.
  struct {
    // Exclusive source/operation ranges and the retained summary publisher.
    loom_runtime_service_control_binding_t binding;
    // Actual HAL group placement, shared by the text model and its operations.
    uint64_t device_ordinal;
    // Current source-selected opportunity, separate from publication tokens.
    loom_runtime_service_offer_t offer;
    // Captured choice kind; NONE means no new unit can currently start.
    loom_serve_text_choice_t choice;
    // Exact admission receipt/row or growth extent for PREPARATION.
    loom_serve_text_preparation_t preparation;
    // Captured epoch geometry, borrowing epoch.planning.scheduled.
    loom_serve_packing_plan_t plan;
    // Request participation in the retained source choice.
    uint64_t choice_request_count;
    // Device participation in that choice; zero for a stateless host query.
    uint64_t choice_devices;
    // Last published legal revision, independent of operation publication.
    uint64_t published_revision;
    // Monotonically changing identities for accepted units.
    uint64_t next_serial;
    // Local model/endpoint/continuation changes that invalidate capacity
    // denials.
    uint64_t state_revision;
    // Independent owners retained until their explicit completion boundaries.
    loom_runtime_service_operation_t operations[TEXT_OPERATION_COUNT];
    // Last published operation records, suppressing unchanged atomic writes.
    loom_runtime_service_operation_t published[TEXT_OPERATION_COUNT];
    // Unfinished generation receipts, including any accepted epoch cohort.
    uint64_t generation_count;
    // Generation receipts captured by the current epoch through output release.
    uint64_t epoch_request_count;
    // The exclusive execution owner is a checkpoint transaction.
    bool checkpoint;
    // Independent management facts transferred by the compiler owner.
    loom_runtime_service_observation_t management;
    // Independent projector facts; request ownership overlaps the text queue.
    loom_runtime_service_observation_t projection;
    // Last aggregate summary, suppressing unchanged atomic writes.
    loom_runtime_service_observation_t observation;
  } scheduling;
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
  loom_serve_text_epoch_options_t epoch_options;
  // Rotating admission priority, independent of row or request identity.
  iree_host_size_t cursor;
  // Irreversible admission seal after stop or a terminal service failure.
  bool stopping;
  // Proactor-timed reporting of copied state, independent of model readiness.
  loom_serve_text_heartbeat_t heartbeat;
  // Independent CPU/device publication; only the owner mutates model state.
  loom_runtime_readiness_t readiness;
  // Independent finite CPU borrower, reusable after final release.
  loom_runtime_readiness_task_t cpu_task;
  // Captured device dependency, independent of CPU preparation.
  loom_runtime_execution_wait_t device_wait;
  // Owner's device operation, retained until readiness is consumed.
  struct {
    // Owner-only operation discriminator; NONE means no callback is accepted.
    loom_serve_text_work_t work;
    // Exact work and feedback values captured by submission.
    loom_runtime_execution_frontier_t frontier;
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
                                  loom_runtime_cpu_work_fn_t run);

// Publishes owner-only model/session facts to the heartbeat snapshot.
void loom_serve_text_observe(loom_serve_text_service_t* service,
                             const char* phase);

// Records acceptance before any fallible submission can retain ownership.
loom_runtime_service_operation_key_t loom_serve_text_operation_begin(
    loom_serve_text_service_t* service, loom_serve_text_operation_slot_t slot,
    loom_runtime_service_work_flags_t work,
    loom_runtime_service_offer_key_t origin);

// Ends an actually completed unit; failed ownership remains for model
// retirement.
void loom_serve_text_operation_end(loom_serve_text_service_t* service,
                                   loom_serve_text_operation_slot_t slot);

// Transfers copied owner facts only; selection and model queries live
// elsewhere.
void loom_serve_text_publish_control(loom_serve_text_service_t* service);

// Withdraws the retained source choice before its planning storage is mutated.
void loom_serve_text_withdraw_choice(loom_serve_text_service_t* service);

// Invalidates logical-capacity denials after an actual local ownership change.
void loom_serve_text_state_changed(loom_serve_text_service_t* service);

#endif  // LOOM_SERVE_SERVICE_TEXT_SERVICE_OWNER_H_
