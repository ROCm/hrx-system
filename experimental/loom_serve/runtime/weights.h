// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_WEIGHTS_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_WEIGHTS_H_

#include "experimental/loom_serve/runtime/jit.h"

#ifdef __cplusplus
extern "C" {
#endif

// One compiled parameter root selected from a checkpoint domain.
typedef struct loom_serve_weight_root_t {
  // Borrowed command reflection, valid throughout loading.
  const loom_cmd_program_t* program;
  // Selected root's placement requirement from program reflection.
  loom_cmd_program_parameter_root_t root;
  // Initially null output slot. A populated slot owns one reference, including
  // on failure; the model releases it during teardown.
  iree_hal_buffer_t** buffer;
} loom_serve_weight_root_t;

// Populates final weight roots with one copy of each parameter. The first
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
// global scratch. Each distinct root is JITed once; preparation happens once
// per unique tensor directly in final storage. Invalid policy fails loading.
//
// Four independent read/preparation lanes overlap without per-tensor host
// waits; success joins all final readiness frontiers. The cold policy process
// and its returned references are released before this call returns.
//
// On failure, dependent lanes are failed and accepted queue operations retain
// their resources through terminal completion. The caller abandons the model
// and destroys its device after releasing the populated roots; failed readiness
// must not be treated as permission to reuse their contents.
iree_status_t loom_serve_weights_load(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, loom_serve_jit_t* jit,
    iree_hal_command_buffer_mode_t command_mode,
    iree_host_size_t shared_root_count, iree_host_size_t root_count,
    const loom_serve_weight_root_t* roots, iree_string_view_t weights_path,
    iree_string_view_t policy_path, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_WEIGHTS_H_
