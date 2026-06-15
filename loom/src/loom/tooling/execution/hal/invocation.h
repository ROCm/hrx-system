// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic HAL executable invocation helpers for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_
#define LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "loom/target/types.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Maximum number of HAL dispatch constants accepted by the generic runner.
  LOOM_RUN_HAL_MAX_CONSTANT_COUNT = 64,
  // Maximum number of HAL dispatch bindings accepted by the generic runner.
  LOOM_RUN_HAL_MAX_BINDING_COUNT = 64,
};

typedef struct loom_run_hal_invocation_options_t {
  // HAL executable function symbol to dispatch. Empty selects the only named
  // function in the prepared executable.
  iree_string_view_t function_name;
  // Dispatch workgroup count in x, y, z order.
  uint32_t workgroup_count[3];
  // Dispatch constants in HAL ABI order.
  uint32_t constants[LOOM_RUN_HAL_MAX_CONSTANT_COUNT];
  // Number of entries in |constants|.
  iree_host_size_t constant_count;
} loom_run_hal_invocation_options_t;

typedef struct loom_run_hal_binding_specs_t {
  // Textual binding value specs in HAL binding ordinal order.
  const iree_string_view_t* values;
  // Calling-convention character for each binding value spec.
  const char* conventions;
  // Number of entries in |values| and |conventions|.
  iree_host_size_t count;
} loom_run_hal_binding_specs_t;

typedef struct loom_run_hal_invocation_plan_t {
  // HAL executable function symbol and dispatch geometry.
  loom_run_hal_invocation_options_t options;
  // Plan-owned materialized binding values in HAL binding ordinal order.
  iree_vm_list_t* bindings;
  // Plan-owned optional expected binding values compared after dispatch.
  iree_vm_list_t* expected_bindings;
  // Plan-owned heap HAL allocator backing expected host binding buffers.
  iree_hal_allocator_t* expected_binding_allocator;
  // Maximum number of output elements to format.
  iree_host_size_t max_output_element_count;
} loom_run_hal_invocation_plan_t;

typedef struct loom_run_hal_prepared_candidate_t {
  // Target-neutral bundle resolved for |executable|, when available.
  const loom_target_bundle_t* target_bundle;
  // Prepared HAL executable retained for repeated dispatches.
  iree_hal_executable_t* executable;
} loom_run_hal_prepared_candidate_t;

typedef struct loom_run_hal_iteration_t {
  // Iteration-owned binding list cloned from the invocation plan.
  iree_vm_list_t* bindings;
} loom_run_hal_iteration_t;

typedef struct loom_run_hal_dispatch_batch_options_t {
  // Number of identical dispatches recorded into the reusable command buffer.
  iree_host_size_t dispatch_count;
  // Command-buffer mode used while recording the batch.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Queue execute flags used for each batch submission.
  iree_hal_execute_flags_t execute_flags;
} loom_run_hal_dispatch_batch_options_t;

typedef struct loom_run_hal_dispatch_batch_t {
  // Host allocator used for batch-owned arrays.
  iree_allocator_t host_allocator;
  // Batch-owned binding lists kept alive for command-buffer executions.
  iree_vm_list_t** binding_lists;
  // Number of entries in |binding_lists|.
  iree_host_size_t binding_list_count;
  // Reusable command buffer containing |dispatch_count| dispatch commands.
  iree_hal_command_buffer_t* command_buffer;
  // Timeline semaphore signaled after each batch submission.
  iree_hal_semaphore_t* semaphore;
  // Payload value to signal on the next batch submission.
  uint64_t next_signal_value;
  // Number of dispatches recorded in |command_buffer|.
  iree_host_size_t dispatch_count;
  // Queue execute flags used for each batch submission.
  iree_hal_execute_flags_t execute_flags;
} loom_run_hal_dispatch_batch_t;

typedef struct loom_run_hal_invocation_request_t {
  // Initialized HAL runtime that owns the device used for dispatch.
  const loom_run_hal_runtime_t* runtime;
  // Provider-produced artifact bytes to prepare and dispatch.
  const loom_run_hal_artifact_t* artifact;
  // HAL dispatch function symbol and workgroup count.
  loom_run_hal_invocation_options_t options;
  // Textual input/output binding specs parsed before dispatch.
  loom_run_hal_binding_specs_t bindings;
  // Optional textual expected binding specs compared after dispatch.
  loom_run_hal_binding_specs_t expected_bindings;
  // Maximum number of output elements to format for human-readable output.
  iree_host_size_t max_output_element_count;
} loom_run_hal_invocation_request_t;

typedef struct loom_run_hal_invocation_result_t {
  // Human-readable execution output or comparison diagnostics.
  iree_string_builder_t output;
  // Process-style exit code: zero for success, non-zero for mismatches.
  int exit_code;
} loom_run_hal_invocation_result_t;

// Initializes invocation options to the single executable function and a single
// workgroup.
void loom_run_hal_invocation_options_initialize(
    loom_run_hal_invocation_options_t* out_options);

// Initializes a request to dispatch the single executable function over one
// workgroup.
void loom_run_hal_invocation_request_initialize(
    loom_run_hal_invocation_request_t* out_request);

// Initializes an empty invocation plan.
void loom_run_hal_invocation_plan_initialize(
    loom_run_hal_invocation_plan_t* out_plan);

// Releases storage owned by |plan|.
void loom_run_hal_invocation_plan_deinitialize(
    loom_run_hal_invocation_plan_t* plan);

// Initializes an empty prepared HAL candidate.
void loom_run_hal_prepared_candidate_initialize(
    loom_run_hal_prepared_candidate_t* out_candidate);

// Releases storage owned by |candidate|.
void loom_run_hal_prepared_candidate_deinitialize(
    loom_run_hal_prepared_candidate_t* candidate);

// Initializes an empty HAL dispatch iteration.
void loom_run_hal_iteration_initialize(loom_run_hal_iteration_t* out_iteration);

// Releases storage owned by |iteration|.
void loom_run_hal_iteration_deinitialize(loom_run_hal_iteration_t* iteration);

// Initializes batch options for one reusable unvalidated/unretained dispatch.
void loom_run_hal_dispatch_batch_options_initialize(
    loom_run_hal_dispatch_batch_options_t* out_options);

// Initializes an empty HAL dispatch batch.
void loom_run_hal_dispatch_batch_initialize(
    loom_run_hal_dispatch_batch_t* out_batch);

// Releases storage owned by |batch|.
void loom_run_hal_dispatch_batch_deinitialize(
    loom_run_hal_dispatch_batch_t* batch);

// Initializes an invocation result. Must be paired with
// loom_run_hal_invocation_result_deinitialize().
void loom_run_hal_invocation_result_initialize(
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* out_result);

// Releases storage owned by |result|.
void loom_run_hal_invocation_result_deinitialize(
    loom_run_hal_invocation_result_t* result);

// Returns the total byte length of HAL buffer or buffer-view refs in
// |binding_list|.
iree_status_t loom_run_hal_binding_list_total_byte_length(
    iree_vm_list_t* binding_list, uint64_t* out_byte_length);

// Prepares a HAL executable object from provider-produced artifact bytes.
iree_status_t loom_run_hal_artifact_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_artifact_t* artifact,
    iree_hal_executable_t** out_hal_executable);

// Prepares |artifact| once for repeated dispatches.
iree_status_t loom_run_hal_prepared_candidate_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_artifact_t* artifact,
    loom_run_hal_prepared_candidate_t* out_candidate);

// Dispatches a prepared HAL executable with |binding_list|.
iree_status_t loom_run_hal_dispatch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options);

// Records a reusable command buffer containing |batch_options->dispatch_count|
// copies of |plan|'s dispatch. The batch clones |plan| bindings once and may be
// executed repeatedly by loom_run_hal_dispatch_batch_execute.
iree_status_t loom_run_hal_dispatch_batch_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_allocator_t allocator, loom_run_hal_dispatch_batch_t* out_batch);

// Records a reusable command buffer containing |batch_options->dispatch_count|
// copies of |plan|'s dispatch. Dispatch slot i uses binding list
// |(binding_list_offset + i) % binding_list_count| from |binding_lists|. The
// batch clones each binding list once and may be executed repeatedly by
// loom_run_hal_dispatch_batch_execute.
iree_status_t loom_run_hal_dispatch_batch_prepare_from_binding_ring(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    iree_host_size_t binding_list_count, iree_vm_list_t* const* binding_lists,
    iree_host_size_t binding_list_offset,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_allocator_t allocator, loom_run_hal_dispatch_batch_t* out_batch);

// Records a reusable command buffer containing a sequence of HAL dispatches for
// each logical batch slot. |candidates| has |sequence_count| entries. |plans|
// is a flattened row-major array with |plan_ring_count * sequence_count|
// entries, indexed as ring slot first and sequence step second. Each logical
// batch slot i records every sequence step using ring slot
// |(plan_ring_offset + i) % plan_ring_count|.
iree_status_t loom_run_hal_dispatch_sequence_batch_prepare_from_plan_ring(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    iree_host_size_t plan_ring_count,
    const loom_run_hal_invocation_plan_t* const* plans,
    iree_host_size_t plan_ring_offset,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_allocator_t allocator, loom_run_hal_dispatch_batch_t* out_batch);

// Submits |batch| once and waits for completion.
iree_status_t loom_run_hal_dispatch_batch_execute(
    const loom_run_hal_runtime_t* runtime,
    loom_run_hal_dispatch_batch_t* batch);

// Transfers |batch| bindings back to host-visible storage and records formatted
// outputs or expected comparison diagnostics in |result|.
iree_status_t loom_run_hal_dispatch_batch_collect_results(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_dispatch_batch_t* batch, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result);

// Prepares and dispatches |artifact| through |runtime|.
iree_status_t loom_run_hal_invocation_execute(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_artifact_t* artifact, iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options);

// Parses textual binding specs into a reusable typed invocation plan.
iree_status_t loom_run_hal_invocation_plan_prepare_from_specs(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_options_t* options,
    const loom_run_hal_binding_specs_t* bindings,
    const loom_run_hal_binding_specs_t* expected_bindings,
    iree_host_size_t max_output_element_count, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan);

// Prepares a reusable invocation plan from caller-materialized binding lists.
// |out_plan| retains |bindings| and optional |expected_bindings|.
iree_status_t loom_run_hal_invocation_plan_prepare_from_lists(
    const loom_run_hal_invocation_options_t* options, iree_vm_list_t* bindings,
    iree_vm_list_t* expected_bindings,
    iree_host_size_t max_output_element_count,
    loom_run_hal_invocation_plan_t* out_plan);

// Clones plan bindings and dispatches one iteration through |candidate| without
// transferring or comparing results. Callers that benchmark dispatch-only paths
// can deinitialize the returned iteration without collecting.
iree_status_t loom_run_hal_invocation_dispatch_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_iteration_t* out_iteration);

// Transfers |iteration| bindings back to host-visible storage and records
// formatted outputs or expected comparison diagnostics in |result|.
iree_status_t loom_run_hal_invocation_collect_results(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_iteration_t* iteration, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result);

// Dispatches one iteration of a prepared HAL candidate and collects results.
iree_status_t loom_run_hal_invocation_run_prepared(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result);

// Dispatches |artifact| using |plan| and records either formatted outputs or
// expected comparison diagnostics in |result|. The plan bindings are cloned
// before dispatch so host-transfer helpers cannot mutate the plan's list
// container.
iree_status_t loom_run_hal_invocation_run_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_artifact_t* artifact,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result);

// Transfers dispatch bindings back to host-visible storage for inspection.
iree_status_t loom_run_hal_transfer_bindings_to_host(
    const loom_run_hal_runtime_t* runtime, iree_vm_list_t* binding_list);

// Parses bindings, dispatches |request->artifact|, transfers bindings back
// to host-visible storage, and records either formatted outputs or expected
// comparison diagnostics in |result|.
iree_status_t loom_run_hal_invocation_run(
    const loom_run_hal_invocation_request_t* request,
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_
