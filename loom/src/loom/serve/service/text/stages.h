// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_STAGES_H_
#define LOOM_SERVE_SERVICE_TEXT_STAGES_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/runtime/program/module.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loomc/task.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_stage_t {
  // Owned JIT command image and loaded executable entries.
  loom_serve_jit_stage_t* compiled;
  // Immutable layout borrowing compiled (or its still-running producer).
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

// Binds immutable command layouts without loading checkpoint payloads. Native
// compilation may still be running. The shared prefix has identical target
// placement; auxiliary roots are grouped by checkpoint path and preparation
// policy. Equal keys in different domains never alias.
// Stages and parameters start zeroed except for their published program views.
// On failure the caller still owns every populated plan and fixed-buffer slot.
// Consumers retire before plans; declarations outlive the returned sources.
iree_status_t loom_serve_text_stages_bind(
    loom_serve_device_t* device, loom_serve_pool_t* pool,
    const loom_serve_preparation_t* preparation, iree_host_size_t shared_count,
    iree_host_size_t stage_count, loom_serve_text_stage_t* stages,
    loom_serve_text_parameters_t* parameters, iree_allocator_t host_allocator);

// Binds replacement code to the original stages' parameter roots. Reflection
// must preserve parameter keys, placement, alignment and model-buffer arity;
// private workspace may change. No weight plan, storage or I/O is created.
// Stages start zeroed except for compiled ownership and program views. The
// caller releases all populated fixed-buffer slots even on failure. Preparation
// and baseline are immutable and outlive this call; successful slots own
// independent references.
iree_status_t loom_serve_text_stages_rebind(
    const loom_serve_preparation_t* preparation, iree_host_size_t shared_count,
    iree_host_size_t stage_count, const loom_serve_text_stage_t* baseline,
    loom_serve_text_stage_t* stages, iree_allocator_t host_allocator);

// Transfers the joined recording outcome after every task has retired. All
// successfully recorded commands remain owned by stages, including on failure.
typedef void (*loom_serve_text_stages_completion_fn_t)(void* user_data,
                                                       iree_status_t status);

// Records bound stages concurrently on an existing scheduling domain. The
// positive concurrency bounds tasks and temporary job storage, independently
// of stage_count. The queue family, compiled stages, fixed buffers, and
// user_data remain borrowed through completion; command slots start NULL and
// have disjoint writers. No commands are submitted for device execution.
// Success owes one completion, possibly inline, after every accepted task
// retires, even if a later submission or recording fails. Allocation rejection
// owes no callback. The caller may discard the result only after completion.
iree_status_t loom_serve_text_stages_record_async(
    loomc_task_sink_t sink, iree_host_size_t concurrency,
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode, iree_host_size_t stage_count,
    loom_serve_text_stage_t* stages,
    loom_serve_text_stages_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator);

// Borrows the command and reflects its model-facing native submission ABI.
loom_serve_stage_t loom_serve_text_stage_binding(
    const loom_serve_text_stage_t* stage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_STAGES_H_
