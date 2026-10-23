// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_EXECUTABLES_H_
#define LOOM_SERVE_SERVICE_EXECUTABLES_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/parameter/bank.h"
#include "loom/serve/runtime/parameter/sources.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/module.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/runtime/program/stage_table.h"
#include "loom/serve/service/execution_reconfiguration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Code ownership for one serialized model. Management selects a source
// generation for future invocations; preparation retains its own generation.
// Device results have no dependency on this owner after producer retirement.
typedef struct loom_serve_executables_t loom_serve_executables_t;

typedef struct loom_serve_executable_options_t {
  // Shared device and compiler host, borrowed through destruction.
  loom_serve_device_t* device;
  // Original indexed source generation; its report outlives this owner.
  loom_serve_jit_t* jit;
  // Original policy/checkpoint namespace, copied during allocation. Files stay
  // immutable while registered; replacement native providers do not replace it.
  const loom_serve_sources_t* sources;
  // Borrowed execution and cold working set, outliving every request choice.
  loom_serve_execution_t* execution;
  // Default admission, including every root reachable through cold ordinals.
  loom_serve_residency_t* residency;
  // Immutable cold declarations, borrowed through every replacement job.
  const loom_serve_preparation_t* preparation;
  // Immutable cold source catalog, borrowed through every request choice.
  loom_serve_parameter_sources_t* parameter_sources;
  // Borrowed checkpoint owner and placement, live through every recording.
  // NULL for a parameter-free model.
  const loom_serve_parameter_bank_t* parameters;
  // Original recorded table, borrowed until terminal retirement.
  const loom_serve_stage_table_t* stages;
  // Recording and workspace ownership policy shared by every generation.
  loom_serve_stage_record_options_t recording;
  // Borrowed runner, initially bound to stages and alive through retirement.
  iree_vm_module_t* runner;
  // Borrowed demand-read capability, bound alongside each invocation's code.
  iree_vm_module_t* parameter_module;
  // Bounds reusable request products without restricting one active request.
  loom_serve_variant_cache_options_t cache;
  // Independent bounds on retained checkpoint entries and final packed bytes.
  // One active request may exceed these idle retention limits.
  loom_serve_variant_cache_options_t parameter_cache;
} loom_serve_executable_options_t;

iree_status_t loom_serve_executables_allocate(
    const loom_serve_executable_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_executables_t** out_executables);

// All accepted compilation and invocation execution has retired. Management
// payloads have been disposed; a captured invocation may remain after a
// terminal submission failure. NULL is accepted.
void loom_serve_executables_retire(loom_serve_executables_t* executables);
bool loom_serve_executables_is_retired(loom_serve_executables_t* executables);
void loom_serve_executables_await_retirement(
    loom_serve_executables_t* executables);
// Native release failure preserves the owner and outstanding banks for retry.
iree_status_t loom_serve_executables_free(
    loom_serve_executables_t* executables);

// Collects idle bank retirement on the serialized owner. Pending release owns
// its wake; a new checkpoint miss waits for it before allocating another bank.
iree_status_t loom_serve_executables_query_status(
    loom_serve_executables_t* executables);
bool loom_serve_executables_is_pending(
    const loom_serve_executables_t* executables);

// Captured invocation working set, including cold and selected checkpoints.
// Called after successful advance and before releasing the invocation.
loom_serve_residency_t* loom_serve_executables_residency(
    const loom_serve_executables_t* executables);

// Model-wide management counts each bank once, including idle entries; request
// residencies overlap and must never be summed for physical accounting.
loom_serve_residency_snapshot_t loom_serve_executables_parameters(
    const loom_serve_executables_t* executables);
void loom_serve_executables_set_pinned(loom_serve_executables_t* executables,
                                       bool pinned);
iree_status_t loom_serve_executables_deactivate(
    loom_serve_executables_t* executables, bool* out_complete);

loom_serve_execution_update_t loom_serve_executables_update(
    loom_serve_executables_t* executables);

// Begins the sole invocation's code borrow before source preparation. The
// previous invocation has been released. Serialized with management selection.
void loom_serve_executables_capture(loom_serve_executables_t* executables);

// Transfers source declarations after CPU preparation, including NULL for a
// cold-table invocation. Command stages select checkpoint domains by source
// identity; new banks are prepared on demand. Omitted declarations preserve the
// whole original layout. All request stages share one binding map. Serialized
// with owner advance and selection.
void loom_serve_executables_declare(loom_serve_executables_t* executables,
                                    loom_serve_preparation_t* preparation);

// Installs a nonblocking owner wake before accepting compilations. Detachment
// follows terminal advance of every accepted request compilation.
void loom_serve_executables_set_observer(loom_serve_executables_t* executables,
                                         loom_serve_wake_callback_t observer);

// Resolves request declarations and binds the captured table before parameter
// admission. A miss submits compilation without waiting; completion schedules
// another owner turn. Cancellation skips unstarted work but joins accepted
// producers. Terminal failure retains the invocation until release. New banks
// reserve their final storage and prepare source plans; checkpoint payload I/O
// and execution admission remain separate.
iree_status_t loom_serve_executables_advance(
    loom_serve_executables_t* executables, bool cancel, bool* out_complete);

// Releases an unsubmitted or fully retired invocation; no compiler/device wait.
void loom_serve_executables_release(loom_serve_executables_t* executables);

// Borrows an invocation-workspace stage's launch facts. Resolution succeeded,
// and the invocation retains its table through producer retirement.
// Runner-owned workspace stages use source ordinals directly without an
// additional mapping.
const loom_serve_stage_launch_t* loom_serve_executables_stage(
    const loom_serve_executables_t* executables, uint32_t ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_EXECUTABLES_H_
