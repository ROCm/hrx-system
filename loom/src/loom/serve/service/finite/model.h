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
#include "loom/serve/service/execution_reconfiguration.h"
#include "loom/serve/service/finite/result.h"
#include "loom/serve/service/prepared_request.h"

#ifdef __cplusplus
extern "C" {
#endif

// One cached source program for finite vector-producing invocations. Calls are
// serialized by its service; independent requests are data, not VM processes.
typedef struct loom_serve_finite_model_t loom_serve_finite_model_t;

// Replaces command/direct execution over the same prepared parameters and VM
// packet contract. A prepared batch retains the generation it captured;
// selection at an owner boundary affects subsequent batches only.
loom_serve_execution_update_t loom_serve_finite_model_execution_update(
    loom_serve_finite_model_t* model);

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
  // Bounded idle reuse of source-declared request executables.
  loom_serve_variant_cache_options_t specialization_cache;
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
iree_status_t loom_serve_finite_load_allocate(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_finite_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_finite_load_t** out_load);

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
iree_status_t loom_serve_finite_load_free(loom_serve_finite_load_t* load);

// Source ABI (all metadata integers and F32 results are little-endian):
// prepare.loom: prepare(asset_root, options: buffer) ->
//   (batch_units, upload_capacity, output_capacity, result_capacity: i64,
//    tokenizer_asset, name, state: buffer).
// control.loom:
//   prepare_request(state, input: buffer) -> (bytes: buffer, units, tag: i64)
//   prepare_batch(state, stage_tags, ranges, bytes: buffer) ->
//     (stage: i32, payload, result_plan: buffer)
//   run(stage: i32, input, output: hal.buffer)
// Cold prepare imports declare command or direct-dispatch stages and their
// configuration. Command stages can also declare checkpoint parameter domains.
// prepare_batch may use the same stage/config imports to declare request
// executables, with ordinals following the cold table. Explicit checkpoint
// declarations reuse retained placement or prepare independent banks on demand;
// omitted declarations preserve the cold checkpoint layout.
// Tokenizer asset is relative to asset_root; empty omits tokenization support.
// State is immutable across requests. Positive units bound cohort work/storage;
// equal tags permit batching. Ranges contain [offset, length] i64 pairs into
// concatenated prepared bytes, in request order. Both stage kinds have two
// dynamic roots (input/output) aligned to LOOM_SERVE_FINITE_BUFFER_ALIGNMENT.
// Commands may additionally have compiler-reflected private workspace; direct
// stages use their compiled launch configuration without a command buffer or
// runner workspace.
// Command stages declared by one request share a parameter binding map;
// retained request choices can bind different checkpoint banks.
// Direct stages have no checkpoint parameters. Source run uses runner.execute_2
// for both kinds and publishes device results. Native consumers choose whether
// to compose device work or explicitly download a host snapshot.
// Cold prepare may call prepare.memory_forecast(root) to select a synchronous
// control entry with signature (state, ranges, descriptors: buffer) ->
// (private_peak, retained, expanded: i64). It forecasts the complete descriptor
// set across all producer cohorts, before pixel decoding or device work. The
// consumer defines descriptor contents; ranges use the same offset/length
// pairs as prepare_batch. Values are trusted byte estimates, not limits checked
// against later compilation or allocation. Negative values are invalid.
// The cold outer-application constructor drives the same asynchronous load and
// joins it outside the shared worker pool. Creation indexes weights without
// loading them. The shared device outlives this model. Failure releases partial
// ownership and leaves *out_model NULL. Compilation evidence is available
// through load_report; this wrapper performs no process-global output. Live
// owners use load_advance instead.
iree_status_t loom_serve_finite_model_allocate(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    iree_allocator_t host_allocator, loom_serve_finite_model_t** out_model);

// Begins irreversible removal after service CPU tasks, waits and external
// tokenizer borrowers retire. No further invocation/admission is allowed.
// Attaches nonblocking ownership wakes before releasing cached VM/command/root
// consumers. Payloads and residency protection remain until actual
// queue ownership retires, independently of success/failure readiness. The wake
// only schedules the serialized owner; it must not reenter or destroy
// resources.
void loom_serve_finite_model_retire(loom_serve_finite_model_t* model,
                                    void (*wake)(void* user_data),
                                    void* user_data);

// Query after retire. A true result authorizes free_retired; no device wait,
// new submission or retry is needed. NULL is already retired.
bool loom_serve_finite_model_is_retired(loom_serve_finite_model_t* model);

// Final owner-side destruction after is_retired. Never waits for worker/device
// work. On platform cleanup failure the model and remaining reservations stay
// owned by the caller; otherwise the model is consumed. NULL is accepted.
iree_status_t loom_serve_finite_model_free_retired(
    loom_serve_finite_model_t* model);

// Cold outer-owner teardown after service CPU tasks and waits retire. Begins
// retirement and joins actual queue ownership before free_retired. Live
// model removal uses the wake/query path instead of blocking a shared worker.
iree_status_t loom_serve_finite_model_free(loom_serve_finite_model_t* model);

uint64_t loom_serve_finite_model_batch_limit(
    const loom_serve_finite_model_t* model);
// Source-declared maximum upload extent, independent of prepared host packets.
iree_host_size_t loom_serve_finite_model_input_capacity(
    const loom_serve_finite_model_t* model);
const iree_tokenizer_t* loom_serve_finite_model_tokenizer(
    const loom_serve_finite_model_t* model);
loom_serve_residency_t* loom_serve_finite_model_residency(
    const loom_serve_finite_model_t* model);

// Source-owned forecast for producing one complete logical request, possibly
// through several cohorts. Estimates exclude unrelated peers and host storage.
typedef struct loom_serve_finite_memory_forecast_t {
  // Peak producer parameters, activation staging, input, and private workspace;
  // excludes outputs accounted by retained_bytes.
  uint64_t private_peak_bytes;
  // Aggregate output backing retained until downstream consumption. Counts
  // complete cohort roots, including padding and shared peer rows when needed.
  uint64_t retained_bytes;
  // Total expanded feature storage if every output row is gathered separately,
  // including 16-byte row alignment. Aliases need not occupy distinct backing
  // in retained_bytes; source-private metadata stays on the host.
  uint64_t expanded_bytes;
} loom_serve_finite_memory_forecast_t;

// Whether cold source preparation declared the optional forecast entry.
bool loom_serve_finite_model_has_memory_forecast(
    const loom_serve_finite_model_t* model);

// Configured physical ceiling, not current native availability. UINT64_MAX
// denotes an unbounded domain. Immutable and safe for a CPU planning loan.
uint64_t loom_serve_finite_model_memory_limit(
    const loom_serve_finite_model_t* model);

// CPU-only source invocation, serialized with this model's other calls. Borrows
// descriptor buffers through return and submits no device work. A missing
// declaration reports UNIMPLEMENTED rather than inventing a bound. Source
// failures and invalid estimates leave the output zeroed.
iree_status_t loom_serve_finite_model_forecast_memory(
    loom_serve_finite_model_t* model, iree_vm_buffer_t* ranges,
    iree_vm_buffer_t* descriptors,
    loom_serve_finite_memory_forecast_t* out_forecast);

// Model-wide checkpoint accounting and policy, including idle cached banks.
loom_serve_residency_snapshot_t loom_serve_finite_model_parameters(
    const loom_serve_finite_model_t* model);
void loom_serve_finite_model_set_pinned(loom_serve_finite_model_t* model,
                                        bool pinned);
iree_status_t loom_serve_finite_model_deactivate(
    loom_serve_finite_model_t* model, bool* out_complete);
// Owner-side idle retirement; native failures retain ownership for teardown.
iree_status_t loom_serve_finite_model_query_status(
    loom_serve_finite_model_t* model);
bool loom_serve_finite_model_is_pending(const loom_serve_finite_model_t* model);

// CPU-only calls, serialized with other model calls; suitable for finite task
// processes. Input errors originate in source input.require. Preparation does
// not acquire residency or submit device work. Successful requests are retained
// by their owner while admission waits, without repeated preparation.
iree_status_t loom_serve_finite_model_prepare_request(
    loom_serve_finite_model_t* model, iree_vm_buffer_t* input,
    loom_serve_prepared_request_t* out_prepared);

// Builds one retained cohort from prepared requests in stable client order.
// expected_rows is protocol-derived, one count per request. The source result
// plan is checked once at this boundary: header [device_output_bytes,
// result_bytes, request_count, row_count], request records
// [first_row, row_count, input_tokens], then rows [byte_offset, f32_count].
// Device extent may exceed the live result prefix; it is never inferred from
// rows. A prior cohort must have been released. Failure leaves no prepared
// cohort.
iree_status_t loom_serve_finite_model_prepare_batch(
    loom_serve_finite_model_t* model, iree_host_size_t count,
    const loom_serve_prepared_request_t* requests,
    const iree_host_size_t* expected_rows);

// Observer used by request compilation only, installed before advance_batch.
// Detachment follows terminal advance of every accepted compilation.
void loom_serve_finite_model_set_observer(loom_serve_finite_model_t* model,
                                          loom_serve_wake_callback_t observer);

// Resolves and binds the prepared batch's captured executable choice before
// parameter admission. A cache miss compiles asynchronously and wakes the
// owner. Call again until complete, including after cancellation. A terminal
// result permits release_batch; success without cancellation permits
// admission/submit. No worker/device wait or checkpoint admission occurs here.
iree_status_t loom_serve_finite_model_advance_batch(
    loom_serve_finite_model_t* model, bool cancel, bool* out_complete);

// Scheduler-only submission after residency_try_admit/acquire succeeds. Ends
// this cohort's admission interest without releasing its parameter pin.
// Acquires input/output roots from the shared queue pool, uploads and invokes
// source, then releases input after work. Success transfers a device result
// that can be composed before host completion by importing its ready frontier.
// Results retain neither model parameters nor command programs. Failure is
// terminal; payload and parameter protection remain until producer retirement.
iree_status_t loom_serve_finite_model_submit(
    loom_serve_finite_model_t* model, loom_serve_finite_result_t** out_result,
    loom_serve_execution_frontier_t* out_frontier);

// Scheduler consumes successful producer readiness, releasing the residency pin
// and uploaded payload. Independent results may remain live across subsequent
// cohorts, parameter eviction, reconfiguration, and model destruction.
void loom_serve_finite_model_complete(loom_serve_finite_model_t* model);

// Releases an unsubmitted cohort. Never used after
// partial submission failure, where outer teardown must join real ownership.
void loom_serve_finite_model_release_batch(loom_serve_finite_model_t* model);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_FINITE_MODEL_H_
