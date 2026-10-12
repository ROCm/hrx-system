// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_STAGES_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_STAGES_H_

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/preparation.h"
#include "experimental/loom_serve/runtime/weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_stage_t {
  // Owned JIT command image and loaded executable entries.
  loom_serve_jit_stage_t* compiled;
  // Parsed view borrowing compiled.
  loom_cmd_program_t program;
  // Reusable command retaining its executable and fixed weight resources.
  iree_hal_command_buffer_t* command;
  // Number of owned fixed-root slots, including partial initialization.
  uint32_t fixed_buffer_count;
  // Owned views of immutable parameter groups, in reflected fixed-root order.
  iree_hal_buffer_t** fixed_buffers;
} loom_serve_text_stage_t;

// Checkpoint domains shared by every stage and retained by one model residency.
typedef struct loom_serve_text_parameters_t {
  // Number of domains, including partially initialized plans.
  iree_host_size_t count;
  // Owned array of retained weight plans; each may be null after a failed bind.
  loom_serve_weights_t** plans;
  // Owned array borrowing immutable checkpoint declarations from preparation.
  const loom_serve_preparation_parameter_t** sources;
} loom_serve_text_parameters_t;

// Binds compiled stages without loading checkpoint payloads. The shared prefix
// has identical target placement; auxiliary roots are grouped by checkpoint
// path and preparation policy. Equal keys in different domains never alias.
// Stages and parameters start zeroed except for compiled stage ownership.
// On failure the caller still owns every populated plan and fixed-buffer slot.
// Consumers retire before plans; declarations outlive the returned sources.
iree_status_t loom_serve_text_stages_bind(
    loom_serve_device_t* device, const loom_serve_preparation_t* preparation,
    iree_host_size_t shared_count, iree_host_size_t stage_count,
    loom_serve_text_stage_t* stages, loom_serve_text_parameters_t* parameters,
    iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_STAGES_H_
