// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_PREFILL_GROUP_H_
#define LOOM_SERVE_SERVICE_TEXT_PREFILL_GROUP_H_
#include "loom/runtime/program/module.h"
#include "loom/serve/service/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_resident_binding_t {
  uint32_t producer;
  uint32_t consumer;
  uint64_t request_offset;
  uint64_t response_offset;
  uint64_t generations;
} loom_serve_text_resident_binding_t;

// Publishes source-owned state mapping after complete A+B growth.
iree_status_t loom_serve_text_model_publish_state_bindings(
    loom_serve_text_model_t* model);

// Source ABI text_prefill_group_v1:
//   text_prefill_group_plan_v1(state) -> descriptor
//     LE u32 header {version, shape_count, layer_count, resource_abi}.
//     Version 1 has stage IDs [shape][layer][prefix,ffn,residual] and preserves
//     total leaf ordering. Version 2 has eight phases from plan.h and explicit
//     FFN/down fork/join edges. UINT32_MAX omits optional phases. Version 2
//     may omit every phase of a shape to keep it on ordinary execution;
//     selection requires both shapes to have a complete grouped implementation.
//     requires residual and a merge for each secondary producer. Prefix
//     publishes all shared state required by B and packs the FFN input.
//     Concurrent producers write disjoint persistent output ranges. Reflected
//     temporary workspace is private to each bank/lane and cannot carry values
//     between commands. Merge owns activation/down-input packing, and residual
//     owns source-defined captures after both down outputs are available.
//     Resource ABI 1 supplies the six ordinary packed roots; ABI 2 additionally
//     supplies the six draft roots in ordinary epoch order. Absent optional
//     carry, draft cache and draft page table roots are bound as null and
//     must not be consumed. Draft models must declare ABI 2 and include
//     bank-local warm/capture work in their leaves.
//   text_encode_prefill_group_v1(state, plan_a, input_a, plan_b, input_b,
//                                metadata_a, tokens_a, metadata_b, tokens_b)
//     Receives owned nine-u32 semantic plans and independent packet storage.
//     Reserves B's position/source_position without committing row state. The
//     complete packet extents are uploaded before either bank executes.
//   text_publish_prefill_group_v1(state, plan_a, plan_b, outputs_b, progress)
//     Runs once after successful joined readiness with the model-owned control
//     state, as ordinary epoch publication does. Writes one ordinary progress
//     record for the total known input and optional B selection. No partial
//     group result is published. Source must not require mid-group host reads.
// Storage extents come from the source's existing packed storage declaration.
// The model owner freezes the complete A+B page map before group submission.
// Version 3 uses the twelve resident phases in plan.h. After its stage table
// comes LE u32 transport_count, then transport_count 24-byte records:
// {u32 producer_stage, u32 consumer_stage, u64 request_offset,
// u64 response_offset}. Offsets address each bank's features root. Each
// producer is a one-generation resident command using the six native bindings
// A/B/C/request/response/scales; command parameter reflection owns B/scales.
// Ranges follow group_exchange.h, are distinct per live producer and may not
// alias semantic tensors. Consumers run on another lane and consume the whole
// payload. Serving appends ACK after consumption and a producer terminal join
// before returning the consumer's ordinal to the plan. Descriptor replacement
// cannot change transport geometry. The deployment pool requires uncached
// device access and persistent host mapping, qualified before model load.
// Version 4 retains the twelve phases but moves resident commands out of the
// stage table. Its transport records are {u32 lifetime_stage, u32 generations,
// u64 request_offset, u64 response_offset}, relative to bank A's features.
// Each lifetime is submitted once after packet upload, independently of GPU
// leaves, and joined after all consumers. Seven-resource leaves append bank
// A's features to the ordinary six roots. Source owns GPU generation publish,
// response polling, consumption and final ACK; the host owns RUN/ABORT and
// terminal drain. No per-generation HAL readiness or ACK is submitted.
//
// Called during linking after source and native stages are ready. Resolves the
// explicit text_prefill_group_v1 exports and reserves both banks before
// serving.
iree_status_t loom_serve_text_prefill_group_initialize(
    loom_serve_text_model_t* model);
iree_status_t loom_serve_text_prefill_group_complete(
    loom_serve_text_model_t* model, loom_serve_text_result_t* result);

// Replacement group configuration built at a drained boundary. Preparation
// re-runs the source plan export and provider preparation outside the
// execution lock; commit swaps it into the model and cannot fail. A zeroed
// candidate (single-mode models) commits and discards as a no-op.
typedef struct loom_serve_text_group_candidate_t {
  // Allocator and count of the owned descriptor array.
  iree_allocator_t host_allocator;
  iree_host_size_t stage_count;
  // Owned replacement plan; NULL keeps the current plan.
  loom_runtime_execution_plan_t* plan;
  // Whether stages replaces the model's leaf descriptors.
  bool rebind;
  // Owned retained descriptors; NULL with rebind selects stage_bindings.
  loom_runtime_stage_t* stages;
  // Owned tracked bank workspaces; NULL keeps the current workspaces.
  iree_hal_buffer_t* workspaces[2];
  // Request backing the replacement workspaces.
  iree_hal_pool_reservation_request_t workspace;
} loom_serve_text_group_candidate_t;

// Exports and prepares a plan for |control_state| against the active leaf
// descriptors without touching device memory. Used by resident growth.
iree_status_t loom_serve_text_prefill_group_prepare_plan(
    loom_serve_text_model_t* model, iree_vm_buffer_t* control_state,
    loom_runtime_execution_plan_t** out_plan);

// Prepares the plan, leaf descriptors and bank workspaces for selecting
// |variant| (NULL restores the original stages). Requires a drained group.
// Workspace replacement is one allocation transaction after the bank
// working-set check. Failure leaves the model's configuration unchanged.
iree_status_t loom_serve_text_prefill_group_prepare_variant(
    loom_serve_text_model_t* model, const loom_serve_text_variant_t* variant,
    loom_serve_text_group_candidate_t* out_candidate);
void loom_serve_text_prefill_group_commit(
    loom_serve_text_model_t* model,
    loom_serve_text_group_candidate_t* candidate);
void loom_serve_text_prefill_group_discard(
    loom_serve_text_group_candidate_t* candidate);

// Drops retained variant descriptors so model retirement can observe its
// parameter and command references.
void loom_serve_text_prefill_group_retire(loom_serve_text_model_t* model);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_PREFILL_GROUP_H_
