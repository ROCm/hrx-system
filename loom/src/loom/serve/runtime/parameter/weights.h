// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_WEIGHTS_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_WEIGHTS_H_

#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/storage/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// One compiled parameter root selected from a checkpoint domain.
typedef struct loom_serve_weight_root_t {
  // Borrowed command reflection, valid throughout plan creation.
  const loom_cmd_program_t* program;
  // Selected root's placement requirement from program reflection.
  loom_cmd_program_parameter_root_t root;
  // Initially null output slot. A populated slot owns one reference, including
  // on failure; the model releases it during teardown.
  iree_hal_buffer_t** buffer;
} loom_serve_weight_root_t;

// Retained checkpoint, placement, and in-place preparation plan. A single owner
// serializes activation/deactivation with model submission. Device, queues,
// and optional physical pool outlive this object. The plan owns its resolved
// parameter keys and preparations independently of the compiler.
typedef struct loom_serve_weights_t loom_serve_weights_t;

// Cumulative owner-side submissions, independent of telemetry retention. These
// are accepted queue operations, not measured storage traffic or device time.
typedef struct loom_serve_weight_activity_t {
  // Fully submitted activations, excluding already-active no-ops.
  uint64_t activations;
  // Active-to-inactive releases of physical backing.
  uint64_t deactivations;
  // Payload bytes in successful gather batches, excluding failed partial
  // batches.
  uint64_t read_bytes;
  // Payload bytes whose in-place preparation dispatch was accepted.
  uint64_t prepared_bytes;
  // Gather groups accepted across all activations.
  uint64_t groups;
} loom_serve_weight_activity_t;

// Resolves final weight roots on the serialized pool owner, without waiting,
// indexing files, or compiling source. The first
// shared_root_count roots have identical parameter placement; subsequent roots
// either share existing placement or own new parameters. Zero shared roots is
// valid. Only the selected roots are populated; other roots of their programs
// are untouched. Each output slot is initially null.
//
// A call covers one checkpoint domain. Sharing and key identity are scoped to
// that call. A command using base and adapter checkpoints supplies its selected
// roots in separate calls, then records with both populated slots. The same
// tensor name in distinct checkpoint domains does not imply shared storage.
//
// With pool, roots have stable virtual addresses and start without physical
// backing. NULL selects explicitly fixed storage (including device ASAN use);
// deactivation is unavailable in that mode. Fixed roots use one queued
// allocation batch; activation imports its readiness before file reads.
// Successful source preparation is
// required before activation, and activation before any consumer executes.
// storage_pool supplies fixed roots, is qualified for dispatch and transfer,
// and outlives the plan and every resolved root.
//
// On failure both populated root slots and a non-NULL *out_weights remain
// caller-owned for teardown. Release consumer commands and root slots before
// destroying this plan.
iree_status_t loom_serve_weights_allocate(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, iree_hal_pool_t* storage_pool,
    loom_serve_memory_pool_t* pool, iree_host_size_t shared_root_count,
    iree_host_size_t root_count, const loom_serve_weight_root_t* roots,
    iree_allocator_t host_allocator, loom_serve_weights_t** out_weights);

// Receives the terminal source preparation status on a shared task worker.
// All preparation tasks/compiler accesses have retired before this callback.
// Pool mutation and activation return to the serialized model owner.
typedef void (*loom_serve_weights_completion_fn_t)(void* user_data,
                                                   iree_status_t status);

// Exclusively prepares a newly created plan without mutating its pool or
// submitting device work. CPU source policy/indexing and native compilation
// run on the shared host. Payload reads wait for activation; open checkpoint
// handles and recorded preparers remain cached, not a second weight copy.
//
// The ordered weights_paths populate one parameter index; files may use any
// supported format. Relative checkpoint paths use the source environment's
// server-local base_directory. policy_path uses its package when present.
// Absolute paths name server-local files in both cases. The environment's
// strings/package, path list, and all paths are borrowed until completion.
// policy_path names a text or bytecode source with the VM export:
//   prepare_weight(buffer key) -> (buffer command_root, i64 byte_length)
// The key is a read-only, non-NUL-terminated tensor name. Empty command_root
// and zero byte_length preserve file bytes. Otherwise command_root names a
// fully specified command, and byte_length equals the reflected tensor size.
// A preparer has one mutable binding, no fixed buffers, and no global scratch.
// Each distinct root is JITed once; activation prepares each unique tensor once
// directly in final storage. The temporary VM policy is released before JIT.
//
// Success accepts exclusive access to weights and jit until completion, which
// may run before return. The caller keeps both, paths, host and user_data alive
// through that callback. Rejection owes no callback. Preparation is attempted
// once per plan; failure returns ownership for destruction, not activation.
iree_status_t loom_serve_weights_prepare_async(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources, iree_string_view_list_t weights_paths,
    iree_string_view_t policy_path,
    loom_serve_weights_completion_fn_t completion, void* user_data);

// Cold outer-application join over the same preparation. Never called by a
// shared worker; live model loading consumes prepare_async completion instead.
iree_status_t loom_serve_weights_prepare(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources, iree_string_view_list_t weights_paths,
    iree_string_view_t policy_path);

// Commits storage and streams/prepares directly into final roots through four
// independent queue lanes, importing readiness into the consuming execution
// domain. An already active plan is unchanged. Success means submission, not
// completion; consuming commands use that same domain. Native backing runs on
// the shared cold-memory executor and gates file loading through a semaphore.
// Neither native preparation nor device completion blocks the submitting owner.
// The caller pins backing until consuming work retires. Failure is terminal,
// preserving backing and accepted I/O ownership until final plan destruction.
iree_status_t loom_serve_weights_activate(loom_serve_weights_t* weights,
                                          loom_serve_execution_t* execution);

// Cold outer-thread join for an activation already submitted above. Joins
// accepted read/preparation ownership even on failure. Task workers observe the
// execution frontier asynchronously instead of calling this function.
iree_status_t loom_serve_weights_await(loom_serve_weights_t* weights);

// Advances native backing release, preserving parameter roots and replay plans.
// The caller first retires all consuming work and excludes new submission.
// Retained mutable state is independent. The pool owner receives a completion
// wake; repeated calls join the accepted operation without submitting it again.
// Reactivation requires terminal completion. Retirement joins pending release.
iree_status_t loom_serve_weights_deactivate(loom_serve_weights_t* weights,
                                            bool* out_complete);

// Parameter accounting only, separate from state sharing the physical pool.
// Fixed storage reports zero virtual-pool statistics.
loom_serve_memory_statistics_t loom_serve_weights_statistics(
    const loom_serve_weights_t* weights);

// Copies cumulative submission counters on the serialized activation owner.
loom_serve_weight_activity_t loom_serve_weights_activity(
    const loom_serve_weights_t* weights);

// Observes final ownership of both parameter consumers and activation I/O.
// The serialized owner stops new activation/submission, attaches the wake,
// releases cached commands/root references, then queries is_retired. The wake
// only schedules an owner turn; it may not reenter the plan or tear down state.
// NULL detaches and joins the callback's final payload access. A NULL plan is
// already retired and accepts observer/await operations as no-ops.
void loom_serve_weights_set_retirement_observer(loom_serve_weights_t* weights,
                                                void (*wake)(void* user_data),
                                                void* user_data);
bool loom_serve_weights_is_retired(loom_serve_weights_t* weights);

// Cold outer-thread join of the ownership above, independent of readiness.
// Live owners use the wake and query instead of waiting on a shared worker.
void loom_serve_weights_await_retirement(loom_serve_weights_t* weights);

// Destruction after is_retired or await_retirement proved every consumer and
// activation view retired. Native unmapping/release is synchronous; no queue or
// callback completion is waited on here. A platform cleanup failure preserves
// the plan and outstanding reservations. NULL is accepted.
iree_status_t loom_serve_weights_free(loom_serve_weights_t* weights);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_WEIGHTS_H_
