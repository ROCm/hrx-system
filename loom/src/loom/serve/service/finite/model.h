// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_MODEL_H_
#define LOOM_SERVE_SERVICE_FINITE_MODEL_H_

#include "iree/tokenizer/tokenizer.h"
#include "iree/vm/buffer.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"

#ifdef __cplusplus
extern "C" {
#endif

// One cached source program for finite vector-producing invocations. Calls are
// serialized by its service; independent requests are data, not VM processes.
typedef struct loom_serve_finite_model_t loom_serve_finite_model_t;

// Base alignment guaranteed for the source program's device input/output
// packets. Interior records retain their source-defined alignment.
#define LOOM_SERVE_FINITE_BUFFER_ALIGNMENT 16

typedef struct loom_serve_finite_model_options_t {
  // Explicit program inputs and local base, borrowed through load completion.
  loom_serve_sources_t sources;
  // Asset directory interpreted by the cold source program.
  iree_string_view_t asset_directory;
  // Source-owned JSON object configuring capacity and model policy.
  iree_string_view_t options;
} loom_serve_finite_model_options_t;

// Owner-driven preparation. Private CPU/JIT work runs on the shared host;
// advance alone binds shared memory, residency and runtime capabilities.
typedef struct loom_serve_finite_load_t loom_serve_finite_load_t;

// Called after accepted preparation work retires. May only notify/schedule
// the serialized owner; it must not call back into the load or block. Payload
// and borrowed device/options remain live through the load's terminal advance.
typedef void (*loom_serve_finite_load_wake_fn_t)(void* user_data);

// Allocates an unstarted operation without model storage reservation, source
// I/O or work submission. Options' views are borrowed, not copied. A registered
// but drained deployment can defer creating this operation altogether.
iree_status_t loom_serve_finite_load_create(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_finite_load_wake_fn_t wake, void* user_data,
    loom_serve_finite_load_t** out_load, iree_allocator_t host_allocator);

enum loom_serve_finite_load_flag_bits_e {
  // Stops subsequent phases, retiring any accepted CPU/JIT work before the
  // terminal CANCELLED result. No callback tears down shared resources.
  LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL = 1u << 0,
};
typedef uint32_t loom_serve_finite_load_flags_t;

// Serialized pool-owner turn. Never waits for tasks or device work. After a
// nonterminal result, wake schedules another turn. A terminal result transfers
// exclusive access to the owner, including on failure; no callbacks remain.
// Call until *out_complete, then inspect evidence and take or destroy the load.
iree_status_t loom_serve_finite_load_advance(
    loom_serve_finite_load_t* load, loom_serve_finite_load_flags_t flags,
    bool* out_complete);

// Terminal-only queries. Evidence is borrowed until take_model or destruction;
// take_model transfers a successfully prepared, still-unactivated model.
const loom_serve_compilation_report_t* loom_serve_finite_load_report(
    const loom_serve_finite_load_t* load);
loom_serve_finite_model_t* loom_serve_finite_load_take_model(
    loom_serve_finite_load_t* load);

// Destroys a terminal operation and any untaken model. Preparation submits no
// queue work, so partial-load cleanup requires no asynchronous retirement.
// A platform release failure preserves the operation and remaining ownership.
iree_status_t loom_serve_finite_load_destroy(loom_serve_finite_load_t* load);

// Source ABI (all metadata integers and F32 results are little-endian):
// prepare.loom: prepare(asset_root, options: buffer) ->
//   (batch_units, upload_capacity, output_capacity, feedback_capacity: i64,
//    tokenizer_asset, name, state: buffer).
// control.loom:
//   prepare_request(state, input: buffer) -> (bytes: buffer, units, tag: i64)
//   prepare_batch(state, stage_tags, ranges, bytes: buffer) ->
//     (stage: i32, payload, result_plan: buffer)
//   run(stage: i32, feedback_bytes: i64, input, output: hal.buffer)
// Cold prepare imports declare command or direct-dispatch stages and their
// configuration. Command stages can also declare checkpoint parameter domains.
// Tokenizer asset is relative to asset_root; empty omits tokenization support.
// State is immutable across requests. Positive units bound cohort work/storage;
// equal tags permit batching. Ranges contain [offset, length] i64 pairs into
// concatenated prepared bytes, in request order. Both stage kinds have two
// dynamic roots (input/output) aligned to LOOM_SERVE_FINITE_BUFFER_ALIGNMENT.
// Commands additionally have compiler-reflected private workspace; direct
// stages use their compiled launch configuration without a command buffer or
// runner workspace.
// Weighted command stages share exactly the same fixed parameter placement.
// Direct stages have no checkpoint parameters. Source run uses the same runner
// execute_2/feedback imports for both kinds; only compact final vectors cross
// to the host.
// The cold outer-application constructor drives the same asynchronous load and
// joins it outside the shared worker pool. Creation indexes weights without
// loading them. The shared device outlives this model. Failure releases partial
// ownership and leaves *out_model NULL. Live owners use load_advance instead.
iree_status_t loom_serve_finite_model_create(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_finite_model_t** out_model, iree_allocator_t host_allocator);

// Begins irreversible removal after service CPU tasks, waits and external
// tokenizer borrowers retire. No further invocation/admission is allowed.
// Attaches nonblocking ownership wakes before releasing cached VM/command/root
// consumers. Payloads, feedback and residency protection remain until actual
// queue ownership retires, independently of success/failure readiness. The wake
// only schedules the serialized owner; it must not reenter or destroy
// resources.
void loom_serve_finite_model_retire(loom_serve_finite_model_t* model,
                                    void (*wake)(void* user_data),
                                    void* user_data);

// Query after retire. A true result authorizes destroy_retired; no device wait,
// new submission or retry is needed. NULL is already retired.
bool loom_serve_finite_model_is_retired(loom_serve_finite_model_t* model);

// Final owner-side destruction after is_retired. Never waits for worker/device
// work. On platform cleanup failure the model and remaining reservations stay
// owned by the caller; otherwise the model is consumed. NULL is accepted.
iree_status_t loom_serve_finite_model_destroy_retired(
    loom_serve_finite_model_t* model);

// Cold outer-owner teardown after service CPU tasks and waits retire. Begins
// retirement and joins actual queue ownership before destroy_retired. Live
// model removal uses the wake/query path instead of blocking a shared worker.
iree_status_t loom_serve_finite_model_destroy(loom_serve_finite_model_t* model);

uint64_t loom_serve_finite_model_batch_limit(
    const loom_serve_finite_model_t* model);
const iree_tokenizer_t* loom_serve_finite_model_tokenizer(
    const loom_serve_finite_model_t* model);
loom_serve_residency_t* loom_serve_finite_model_residency(
    const loom_serve_finite_model_t* model);

typedef struct loom_serve_finite_prepared_t {
  // Owned source bytes. The input buffer remains retained by the request owner
  // through batch completion, including when bytes alias it.
  iree_vm_buffer_t* bytes;
  // Positive additive source cost, at most the model's declared batch limit.
  uint64_t units;
  // Source-defined compatibility class; no native geometry interpretation.
  uint64_t tag;
} loom_serve_finite_prepared_t;

// CPU-only calls, serialized with other model calls; suitable for finite task
// processes. Input errors originate in source input.require. Preparation does
// not acquire residency or submit device work. Successful requests are retained
// by their owner while admission waits, without repeated preparation.
iree_status_t loom_serve_finite_model_prepare_request(
    loom_serve_finite_model_t* model, iree_vm_buffer_t* input,
    loom_serve_finite_prepared_t* out_prepared);

// Builds one retained cohort from prepared requests in stable client order.
// expected_rows is protocol-derived, one count per request. The source result
// plan is checked once at this boundary: header [device_output_bytes,
// feedback_bytes, request_count, row_count], request records
// [first_row, row_count, input_tokens], then rows [byte_offset, f32_count].
// Device extent may exceed compact feedback; it is never inferred from rows.
// A prior cohort must have been released. Failure leaves no prepared cohort.
iree_status_t loom_serve_finite_model_prepare_batch(
    loom_serve_finite_model_t* model, iree_host_size_t count,
    const loom_serve_finite_prepared_t* requests,
    const iree_host_size_t* expected_rows);

// Scheduler-only submission after residency_try_acquire succeeds. Acquires
// input/output roots from the shared queue pool, uploads and invokes source,
// then releases input after work and output after feedback via a semaphore
// edge. Returns accepted work/feedback frontiers without a host wait. Failure
// is terminal; payload and parameter protection remain until destruction.
iree_status_t loom_serve_finite_model_submit(
    loom_serve_finite_model_t* model,
    loom_serve_execution_frontier_t* out_frontier);

// Scheduler consumes successful ALL readiness, releasing the residency pin.
// Borrowed result rows remain live until release_batch. If a CPU encoder has
// accepted those views, release_batch follows its process release callback, not
// merely its completion callback. When no consumers remain, the owner may
// release the batch immediately.
void loom_serve_finite_model_complete(loom_serve_finite_model_t* model);

// Trusted queries into the checked plan. Request ordinals name the original
// prepared cohort, not a filtered set of surviving clients. Row ordinals are
// local to each original HTTP request.
uint64_t loom_serve_finite_model_input_tokens(
    const loom_serve_finite_model_t* model, iree_host_size_t request);
iree_const_byte_span_t loom_serve_finite_model_row(
    const loom_serve_finite_model_t* model, iree_host_size_t request,
    iree_host_size_t row);

// Releases an unsubmitted or successfully completed cohort. Never used after
// partial submission failure, where outer teardown must join real ownership.
void loom_serve_finite_model_release_batch(loom_serve_finite_model_t* model);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_FINITE_MODEL_H_
