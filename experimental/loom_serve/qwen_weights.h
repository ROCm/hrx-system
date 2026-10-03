// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_QWEN_WEIGHTS_H_
#define EXPERIMENTAL_LOOM_SERVE_QWEN_WEIGHTS_H_

#include "experimental/loom_serve/jit.h"

#ifdef __cplusplus
extern "C" {
#endif

// One already compiled stage's immutable parameter placement and output slots.
typedef struct loom_serve_qwen_weight_stage_t {
  // Borrowed command reflection, valid throughout loading.
  const loom_cmd_program_t* program;
  // Initially null slots in fixed-root order. Each populated slot owns a
  // reference, including on failure; the model releases them during teardown.
  iree_hal_buffer_t** buffers;
} loom_serve_qwen_weight_stage_t;

// Populates final target and auxiliary weight roots with one copy of each
// parameter. The first target_stage_count stages have identical placement;
// auxiliary roots either share that placement or own new model parameters.
// FFN gate/up ranges are prepared in place by the source-JIT command
// qwen38_prepare_ffn. Four independent read/preparation lanes overlap without
// per-tensor host waits; success joins all final readiness frontiers.
//
// On failure, dependent lanes are failed and accepted queue operations retain
// their resources through terminal completion. The caller abandons the model
// and destroys its device after releasing the populated roots; failed readiness
// must not be treated as permission to reuse their contents.
iree_status_t loom_serve_qwen_weights_load(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, loom_serve_jit_t* jit,
    iree_hal_command_buffer_mode_t command_mode,
    iree_host_size_t target_stage_count, iree_host_size_t stage_count,
    const loom_serve_qwen_weight_stage_t* stages,
    iree_string_view_t weights_path, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_QWEN_WEIGHTS_H_
