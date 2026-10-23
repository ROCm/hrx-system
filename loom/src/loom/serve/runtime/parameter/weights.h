// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_WEIGHTS_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_WEIGHTS_H_

#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/execution/workspace.h"
#include "loom/serve/runtime/parameter/sources.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/preparation.h"
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

// Retained checkpoint, placement, and source preparation plan. A single owner
// serializes activation/deactivation with all consuming models. Device, queues,
// and optional physical pool outlive this object. The plan owns its resolved
// parameter keys and preparations independently of the compiler.
typedef struct loom_serve_weights_t loom_serve_weights_t;

// Cumulative owner-side submissions, independent of telemetry retention. These
// are accepted queue operations, not measured storage traffic or device time.
typedef struct loom_serve_weight_activity_t {
  // Fully submitted loads, excluding readiness imports by later consumers.
  uint64_t activations;
  // Active-to-inactive releases of physical backing.
  uint64_t deactivations;
  // Payload bytes in successful gather batches, excluding failed partial
  // batches.
  uint64_t read_bytes;
  // Final tensor bytes whose preparation dispatch was accepted.
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
// backing. geometry is the device's qualified weight replacement geometry.
// A NULL pool selects explicitly fixed storage (including device ASAN use);
// deactivation is unavailable in that mode. Fixed roots use one queued
// allocation batch. Source preparation joins its readiness before consumer
// commands can be recorded, and activation imports it before file reads.
// Successful source preparation is required before activation, and activation
// before any consumer executes.
// storage_pool supplies fixed roots, is qualified for dispatch and transfer,
// and outlives the plan and every resolved root. workspace supplies bounded
// transient staging for out-of-place preparation and outlives the plan.
//
// On failure both populated root slots and a non-NULL *out_weights remain
// caller-owned for teardown. Release consumer commands and root slots before
// destroying this plan.
iree_status_t loom_serve_weights_allocate(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, iree_hal_pool_t* storage_pool,
    loom_serve_memory_pool_t* pool, loom_serve_memory_geometry_t geometry,
    loom_serve_workspace_t* workspace, iree_host_size_t shared_root_count,
    iree_host_size_t root_count, const loom_serve_weight_root_t* roots,
    iree_allocator_t host_allocator, loom_serve_weights_t** out_weights);

// Folds the missing backing of unique roots, excluding accepted commitments.
// Shared root bindings contribute once; fixed backing has no elastic demand.
loom_serve_memory_demand_t loom_serve_weights_missing_demand(
    const loom_serve_weights_t* weights);

// Receives terminal source preparation and fixed-root allocation status.
// All preparation tasks/compiler accesses have retired before this callback,
// which runs on the completing task or allocation publication path.
// Successful completion permits reusable command recording with the roots.
// Pool mutation and activation return to the serialized model owner.
typedef void (*loom_serve_weights_completion_fn_t)(void* user_data,
                                                   iree_status_t status);

// Exclusively prepares a newly created plan without mutating its pool or
// submitting device work. CPU source policy and native compilation
// run on the shared host. Payload reads wait for activation; open checkpoint
// handles and recorded preparers remain cached, not a second weight copy.
//
// indexed_sources retains the already opened immutable source catalog. The
// declaration maps policy-local inputs to its source ordinals. Source indexes
// and imported-file caches are shared, while each plan owns its input recipes.
// The declaration's policy uses the source environment's package when present;
// absolute paths name server-local files. The environment, declaration, and
// indexed_sources are borrowed until completion; the plan retains its selected
// indexes/providers independently after preparation, not unused catalog peers.
// The policy names a text or bytecode source with the VM export:
//   prepare_weight(buffer key) -> i64 byte_length
// The key is a read-only, non-NUL-terminated tensor name. The policy declares
// at most one command through prepare.stage and its config/config_i64 imports.
// No declaration and zero byte_length preserve file bytes. Otherwise
// byte_length equals the reflected final tensor size. Checkpoint declarations
// are unavailable inside weight policies.
// The "weights" capability exposes source metadata and captures explicit input
// ranges (see inputs.h). The standard "json" and "input" capabilities support
// source-owned metadata interpretation and requirements. A preparer has no
// fixed buffers or global scratch. It binds final storage alone for in-place
// preparation, or final storage followed by the captured input blob for
// out-of-place preparation. Each distinct declaration is JITed once; activation
// prepares each tensor once using bounded staging shared across tensors. The
// temporary VM policy is released before JIT.
//
// Every explicit input range and implicit source-zero read must fit its indexed
// checkpoint entry before preparation succeeds. This does not impose an
// encoding or exact-size identity check.
//
// Success accepts exclusive access to weights until completion, which may run
// before return. Independent compilations may share jit; policy evidence is
// accumulated privately and merged through its report lock. The caller keeps
// both, declaration, host and user_data
// alive through that callback. Rejection owes no callback. Preparation is
// attempted once per plan; failure returns ownership for destruction, not
// activation.
iree_status_t loom_serve_weights_prepare_async(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources,
    const loom_serve_preparation_parameter_t* parameter,
    loom_serve_parameter_sources_t* indexed_sources,
    loom_serve_weights_completion_fn_t completion, void* user_data);

// Cold outer-application join over the same preparation. Never called by a
// shared worker; live model loading consumes prepare_async completion instead.
iree_status_t loom_serve_weights_prepare(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources,
    const loom_serve_preparation_parameter_t* parameter,
    loom_serve_parameter_sources_t* indexed_sources);

// Borrowed placement facts retained independently of the original compiler
// product. Rows remain in construction order; index is below parameter_count.
// No source decoding, search, or allocation occurs in these queries.
iree_host_size_t loom_serve_weights_parameter_count(
    const loom_serve_weights_t* weights);
loom_cmd_program_parameter_t loom_serve_weights_parameter_at(
    const loom_serve_weights_t* weights, iree_host_size_t index);

// Complete captured-input staging allocation for one cold activation. Source
// preparation has finalized the lane geometry before this query. All tensors
// reuse these lanes; the extent is independent of tensor count. Canonical and
// in-place preparation require zero bytes. Consuming execution imports a
// readiness frontier that includes this allocation's queued deallocation.
iree_hal_pool_reservation_request_t loom_serve_weights_activation_workspace(
    const loom_serve_weights_t* weights);

// Commits storage and streams directly into final roots through four read
// lanes. Four preparation lanes join each tensor's reads and captured inputs.
// An already active plan does not reload or prepare again. Success means
// producer submission, not completion or consumer ordering. Native backing
// runs on the shared cold-memory executor and gates file loading with each
// slab's readiness. Neither native preparation nor device
// completion blocks the submitting owner. The caller pins backing until every
// consuming domain retires. A failed load is terminal: later consumers receive
// its error without replaying accepted I/O, whose ownership remains protected
// through plan destruction.
iree_status_t loom_serve_weights_activate(loom_serve_weights_t* weights);

// Imports successfully submitted activation into a consuming execution.
// Each execution must acquire this dependency before consuming a new
// activation; already-dependent warm residencies need no call. Does not reload
// or reprepare. Rejected readiness import leaves a successfully submitted load
// active; it does not cancel or replay the producer or other consumers.
iree_status_t loom_serve_weights_import_readiness(
    loom_serve_weights_t* weights, loom_serve_execution_t* execution);

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

// Collects completed native activation records on the serialized owner and
// reports parameter accounting, separate from state sharing the physical pool.
// Fixed storage reports zero virtual-pool statistics.
loom_serve_memory_statistics_t loom_serve_weights_statistics(
    loom_serve_weights_t* weights);

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
