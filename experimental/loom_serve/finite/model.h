// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_FINITE_MODEL_H_
#define EXPERIMENTAL_LOOM_SERVE_FINITE_MODEL_H_

#include "experimental/loom_serve/runtime/device.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/vm/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

// One cached source program for finite vector-producing invocations. Calls are
// serialized by its service; independent requests are data, not VM processes.
typedef struct loom_serve_finite_model_t loom_serve_finite_model_t;

typedef struct loom_serve_finite_model_options_t {
  // Source package directory, borrowed during creation.
  iree_string_view_t source_directory;
  // Asset directory interpreted by the cold source program.
  iree_string_view_t asset_directory;
  // Source-owned JSON object configuring capacity and model policy.
  iree_string_view_t options;
} loom_serve_finite_model_options_t;

// Source ABI (all metadata integers and F32 results are little-endian):
// prepare.loom: prepare(asset_root, options: buffer) ->
//   (batch_units, upload_capacity, output_capacity, feedback_capacity: i64,
//    tokenizer_asset, name, state: buffer).
// control.loom:
//   prepare_request(state, input: buffer) -> (bytes: buffer, units, tag: i64)
//   prepare_batch(state, stage_tags, ranges, bytes: buffer) ->
//     (stage: i32, payload, result_plan: buffer)
//   run(stage: i32, feedback_bytes: i64, input, output: hal.buffer)
// Cold prepare imports declare commands, configuration and parameter domains.
// Tokenizer asset is relative to asset_root; empty omits tokenization support.
// State is immutable across requests. Positive units bound cohort work/storage;
// equal tags permit batching. Ranges contain [offset, length] i64 pairs into
// concatenated prepared bytes, in request order. Commands have two dynamic
// roots (input/output) plus compiler-reflected private workspace. Every stage
// shares exactly the same fixed parameter placement. Source run uses runner
// execute_2/feedback; only compact final vectors cross to the host.
// Creation indexes weights without loading them. The shared device outlives
// this model. Failure releases partial ownership and leaves *out_model NULL.
iree_status_t loom_serve_finite_model_create(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_finite_model_t** out_model, iree_allocator_t host_allocator);

// Outer-owner teardown after service CPU tasks and waits retire. Joins actual
// queue ownership even on failure before releasing payloads and parameter pins.
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
// Borrowed result rows remain live until release_batch, which follows the CPU
// encoder process release callback, not merely its completion callback.
void loom_serve_finite_model_complete(loom_serve_finite_model_t* model);

// Trusted queries into the checked plan. Ordinals are supplied by the cohort
// owner; row ordinals are local to each original HTTP request.
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
#endif  // EXPERIMENTAL_LOOM_SERVE_FINITE_MODEL_H_
