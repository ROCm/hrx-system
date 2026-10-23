// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_MODEL_INSTANCE_H_
#define LOOM_SERVE_SERVICE_TEXT_MODEL_INSTANCE_H_

#include "loom/serve/service/text/control.h"
#include "loom/serve/service/text/media_epoch.h"
#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/source.h"
#include "loom/serve/service/text/stages.h"
#include "loom/serve/service/text/state.h"

enum { TEXT_ROW_CAPACITY = LOOM_SERVE_TEXT_ROW_CAPACITY };

enum loom_serve_text_epoch_flag_bits_e {
  TEXT_EPOCH_FLAG_VERIFY = 1u << 0,
  TEXT_EPOCH_FLAG_CONTINUATION = 1u << 1,
};
typedef uint32_t loom_serve_text_epoch_flags_t;

typedef enum loom_serve_text_step_kind_e {
  TEXT_STEP_PREFILL,
  TEXT_STEP_DECODE,
} loom_serve_text_step_kind_t;

// Model-owned execution and row storage shared by bootstrap, invocation, and
// continuation ownership. Services borrow opaque model/row handles, not these
// representations; all mutation is serialized by the owning model service.
struct loom_serve_text_model_t {
  // Host allocation policy used for all owned resources.
  iree_allocator_t host_allocator;
  // Source-owned metadata and bytecode, independent of native model bindings.
  loom_serve_text_source_t source;
  // Borrowed shared owner outliving every model resource and host I/O payload.
  loom_serve_device_t* device_owner;
  // GPU device and allocation domain borrowed from device_owner.
  iree_hal_device_t* device;
  // Semaphore namespace borrowed from device_owner.
  iree_hal_device_group_t* group;
  // Borrowed exact dispatch queue, retained by execution.
  iree_hal_queue_t* dispatch;
  // Borrowed exact transfer queue, retained by execution.
  iree_hal_queue_t* transfer;
  // Owned model stage/input timeline with independently retired feedback.
  loom_serve_execution_t* execution;
  // Cold recording policy for every reusable stage in this residency.
  iree_hal_command_buffer_mode_t command_mode;
  // Shared source index and compiler for this model residency.
  loom_serve_jit_t* jit;
  // Checkpoint domains owning stable parameter reservations across all stages.
  loom_serve_text_parameters_t parameters;
  // Owned physical parameter group, outliving borrowing invocation residencies.
  loom_serve_residency_group_t* parameter_group;
  // Atomic invocation working set and independent admission interest.
  loom_serve_residency_t* residency;
  // Independent demanded parameter I/O, retained across service publication.
  loom_serve_text_warmup_t* warmup;
  // Source-declared command programs, owned in declaration order.
  loom_serve_text_stage_t* stages;
  // Number of allocated stages, including partially initialized cold entries.
  iree_host_size_t stage_count;
  // Original native bindings borrowing stages, retained for explicit restore.
  loom_serve_stage_t* stage_bindings;
  // Original command private peak, retained for explicit variant restoration.
  uint64_t stage_workspace_bytes;
  // Number of target packed shapes, independent of auxiliary draft stages.
  iree_host_size_t shape_count;
  // Immutable prepared shape capacities, owned in epoch option order.
  loom_serve_packing_shape_t* shapes;
  // Context specialization shared by every compiled stage.
  iree_host_size_t context_capacity;
  // Proposed tokens after each known anchor; zero disables drafting.
  iree_host_size_t draft_depth;
  // Number of preallocated rows.
  iree_host_size_t row_count;
  // Fixed row records, owned by this model.
  loom_serve_text_row_t* rows;
  // Fixed semantic endpoint records paired with the state's checkpoint IDs.
  loom_serve_text_checkpoint_t* checkpoints;
  // Retained device state, mappings and final queue ownership.
  loom_serve_text_state_t state;
  // Shared VM invocation, native roots, and retained host transfer payloads.
  loom_serve_text_control_t control;
  // Publication facts retained independently of the submitting caller's plan.
  // One invocation owns the reusable host buffers and a residency pin until
  // successful completion; terminal failure retains them for destruction.
  struct {
    // Number of first-plan spans, zero after successful completion.
    iree_host_size_t span_count;
    // Resident row identity in first-plan caller order.
    iree_host_size_t row_indices[TEXT_ROW_CAPACITY];
    // Source publication mode selected by the submitted plan.
    loom_serve_text_epoch_flags_t flags;
    // First-plan span bits with device-fed speculative continuation.
    uint32_t continued_speculative_mask;
    // Exact mixed-input host plan and packet records; empty for text-only.
    loom_serve_text_media_epoch_t media;
  } invocation;
};

struct loom_serve_text_row_t {
  // Borrowed owning model; all row operations use its single invocation.
  loom_serve_text_model_t* model;
  // Borrowed retained row state, owned by the model-wide state arena.
  loom_serve_text_state_row_t* state;
  // Whether the latest completed work selected a token at the current position.
  bool has_prediction;
  // Completed stage counts and end-to-end host durations.
  loom_serve_text_metrics_t metrics;
  // Single-row publication facts, owned until successful complete_step.
  struct {
    // Source command and completed metric category.
    loom_serve_text_step_kind_t kind;
    // Absolute source-owned position to publish after successful readiness.
    uint32_t source_position;
    // Number of consumed inputs published only after readiness.
    iree_host_size_t token_count;
    // Submission start, including activation and feedback retirement.
    iree_time_t start_time;
  } step;
  // Host payloads remain alive until the model drains accepted transfers.
  struct {
    // Active prefill count, absolute base, and EOS ID.
    int32_t control[3];
    // Pending prediction, retained independently of shared input staging.
    int32_t token;
    // Device-generated count at zero and EOS flag at seven.
    int32_t progress[8];
  } transfer;
};

struct loom_serve_text_checkpoint_t {
  // Borrowed model, outliving this handle.
  loom_serve_text_model_t* model;
  // Serialized handle owners; the last release returns the native endpoint.
  iree_host_size_t references;
  // Prepared continuation borrowers; cold restores require exclusive planning.
  iree_host_size_t reservations;
  // Endpoint storage owned until the last handle release.
  loom_serve_text_state_checkpoint_t* state;
  // Whether the exact endpoint selected an unconsumed prediction.
  bool has_prediction;
  // Pending token; meaningful only when has_prediction is set.
  int32_t token;
};

#endif  // LOOM_SERVE_SERVICE_TEXT_MODEL_INSTANCE_H_
