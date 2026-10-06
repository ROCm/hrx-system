// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_WEIGHTS_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_WEIGHTS_H_

#include "experimental/loom_serve/runtime/jit.h"
#include "experimental/loom_serve/storage/memory.h"

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

// Resolves final weight roots without submitting device work. Checkpoint
// metadata is indexed here; payload reads wait for activation. The first
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
// policy_path names source with the VM export:
//   prepare_weight(buffer key) -> (buffer command_root, i64 byte_length)
// The key is a read-only, non-NUL-terminated tensor name. Empty command_root
// and zero byte_length preserve file bytes. Otherwise command_root names a
// fully specified source command, and byte_length must equal the reflected
// tensor size. A preparer has one mutable binding, no fixed buffers, and no
// global scratch. Each distinct root is JITed once; each activation prepares
// every unique tensor once directly in final storage. Invalid policy fails
// plan creation.
//
// With pool, roots have stable virtual addresses and start without physical
// backing. NULL selects explicitly fixed storage (including device ASAN use);
// deactivation is unavailable in that mode. Activation is required in either
// mode before any consumer executes. The cold VM policy is released here;
// resolved preparation commands and the checkpoint provider remain cached.
//
// On failure both populated root slots and a non-NULL *out_weights remain
// caller-owned for teardown. Release consumer commands and root slots before
// destroying this plan. The normal file I/O mode retains open checkpoint
// handles, not a second in-memory copy of the parameter payload.
iree_status_t loom_serve_weights_create(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, loom_serve_memory_pool_t* pool,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    iree_host_size_t shared_root_count, iree_host_size_t root_count,
    const loom_serve_weight_root_t* roots, iree_string_view_t weights_path,
    iree_string_view_t policy_path, loom_serve_weights_t** out_weights,
    iree_allocator_t host_allocator);

// Commits storage and streams/prepares directly into final roots through four
// independent queue lanes. An already active plan is unchanged. Success joins
// all readiness frontiers; every accepted read/preparation retires before
// return, including failures. Failure ends this model run, not a retry signal.
iree_status_t loom_serve_weights_activate(loom_serve_weights_t* weights);

// Releases physical backing, preserving parameter roots and replay plans.
// The caller first retires all consuming work and excludes new submission.
// Retained mutable state is independent. An inactive plan is unchanged.
iree_status_t loom_serve_weights_deactivate(loom_serve_weights_t* weights);

// Parameter accounting only, separate from state sharing the physical pool.
// Fixed storage reports zero virtual-pool statistics.
loom_serve_memory_statistics_t loom_serve_weights_statistics(
    const loom_serve_weights_t* weights);

// Consumer commands and root references must be released and work retired.
// A cleanup failure preserves outstanding ownership; NULL is accepted.
iree_status_t loom_serve_weights_destroy(loom_serve_weights_t* weights);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_WEIGHTS_H_
