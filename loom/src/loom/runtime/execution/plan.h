// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_EXECUTION_PLAN_H_
#define LOOM_RUNTIME_EXECUTION_PLAN_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_runtime_group_mode_e {
  LOOM_RUNTIME_GROUP_MODE_DEFAULT = 0,
  LOOM_RUNTIME_GROUP_MODE_SINGLE,
  LOOM_RUNTIME_GROUP_MODE_TWO_BANK,
} loom_runtime_group_mode_t;

typedef enum loom_runtime_plan_phase_e {
  // Publishes every layer-local state dependency needed by bank B.
  LOOM_RUNTIME_PLAN_PREFIX = 0,
  LOOM_RUNTIME_PLAN_FFN = 1,
  LOOM_RUNTIME_PLAN_RESIDUAL = 2,
} loom_runtime_plan_phase_t;

// Version 2 separates disjoint output-column producers from their consumers.
// Prefix includes activation packing. FFN merge concatenates gate/up columns,
// applies activation and packs the down input. Down merge concatenates output
// columns before residual/norm/capture. Each merge waits for both producers.
typedef enum loom_runtime_plan_ffn_phase_e {
  LOOM_RUNTIME_PLAN_FFN_PREFIX = 0,
  LOOM_RUNTIME_PLAN_FFN_PRIMARY,
  LOOM_RUNTIME_PLAN_FFN_SECONDARY,
  LOOM_RUNTIME_PLAN_FFN_MERGE,
  LOOM_RUNTIME_PLAN_DOWN_PRIMARY,
  LOOM_RUNTIME_PLAN_DOWN_SECONDARY,
  LOOM_RUNTIME_PLAN_DOWN_MERGE,
  LOOM_RUNTIME_PLAN_FFN_RESIDUAL,
  LOOM_RUNTIME_PLAN_FFN_PHASE_COUNT,
} loom_runtime_plan_ffn_phase_t;

// Version 3 consumes each resident projection before starting the next one.
// Each projection forks into primary/remainder and secondary/resident work,
// then joins at its consumer. Activation prepares the down input.
enum {
  LOOM_RUNTIME_PLAN_RESIDENT_PREFIX = 0,
  LOOM_RUNTIME_PLAN_GATE_PRIMARY,
  LOOM_RUNTIME_PLAN_GATE_SECONDARY,
  LOOM_RUNTIME_PLAN_GATE_CONSUME,
  LOOM_RUNTIME_PLAN_UP_PRIMARY,
  LOOM_RUNTIME_PLAN_UP_SECONDARY,
  LOOM_RUNTIME_PLAN_UP_CONSUME,
  LOOM_RUNTIME_PLAN_ACTIVATION,
  LOOM_RUNTIME_PLAN_RESIDENT_DOWN_PRIMARY,
  LOOM_RUNTIME_PLAN_RESIDENT_DOWN_SECONDARY,
  LOOM_RUNTIME_PLAN_RESIDENT_DOWN_CONSUME,
  LOOM_RUNTIME_PLAN_RESIDENT_RESIDUAL,
  LOOM_RUNTIME_PLAN_RESIDENT_PHASE_COUNT,
};

// Source-owned semantic table, indexed [shape][layer][phase]. UINT32_MAX
// omits an optional phase; a prefix is required in every supported layer.
// An eight-phase shape with every stage omitted has no grouped implementation
// and is rejected by selection before encoding. Partial shapes are invalid.
// Leaves use
// the source extension's declared bank/state resource ABI, never kernel names.
typedef struct loom_runtime_plan_description_t {
  iree_host_size_t shape_count;
  iree_host_size_t layer_count;
  iree_host_size_t stage_count;
  const uint32_t* stages;
  // Zero or 3 selects ordered phases, 8 selects FFN fork/join, and 12 selects
  // resident projections. UINT32_MAX omits an optional phase. A projection
  // producer requires its consumer; versions 2/3 require a residual leaf.
  iree_host_size_t phase_count;
} loom_runtime_plan_description_t;

typedef struct loom_runtime_execution_plan_t loom_runtime_execution_plan_t;

// Produces an immutable semantic plan. Validation and cloning happen here,
// before numerical enqueue. Refcounted plans retain source tables by value.
iree_status_t loom_runtime_execution_plan_create(
    const loom_runtime_plan_description_t* description, bool interleave,
    iree_allocator_t allocator, loom_runtime_execution_plan_t** out_plan);
void loom_runtime_execution_plan_retain(loom_runtime_execution_plan_t* plan);
void loom_runtime_execution_plan_release(loom_runtime_execution_plan_t* plan);
bool loom_runtime_execution_plan_equal(
    const loom_runtime_execution_plan_t* lhs,
    const loom_runtime_execution_plan_t* rhs);
iree_host_size_t loom_runtime_execution_plan_node_capacity(
    const loom_runtime_execution_plan_t* plan);

// Visits the causal order outside any execution lock. Each emitted stage is
// appended to an explicit group; emission never accepts numerical work.
typedef iree_status_t (*loom_runtime_plan_emit_fn_t)(void* self, uint32_t bank,
                                                     uint32_t stage);
iree_status_t loom_runtime_execution_plan_encode(
    const loom_runtime_execution_plan_t* plan, iree_host_size_t first_shape,
    iree_host_size_t second_shape, loom_runtime_plan_emit_fn_t emit,
    void* self);

// Emits explicit dependencies using the ordinals returned by the caller.
// Version 1 retains its total ordering. Version 2 permits both FFN branches
// to overlap and orders bank B's prefix after bank A's same-layer prefix.
// Every callback runs before numerical submission. Out-node is set only on
// success, and a callback failure stops encoding immediately.
typedef iree_status_t (*loom_runtime_plan_emit_dag_fn_t)(
    void* self, uint32_t bank, uint32_t stage,
    iree_host_size_t predecessor_count, const uint32_t* predecessors,
    uint32_t* out_node);
iree_status_t loom_runtime_execution_plan_encode_dag(
    const loom_runtime_execution_plan_t* plan, iree_host_size_t first_shape,
    iree_host_size_t second_shape, loom_runtime_plan_emit_dag_fn_t emit,
    void* self);

// Immutable chunk-boundary facts. Selection is pure and cannot enqueue work,
// mutate source state, or call back into the model owner.
typedef struct loom_runtime_plan_group_query_t {
  iree_host_size_t position;
  iree_host_size_t shapes[2];
  iree_host_size_t tokens[2];
} loom_runtime_plan_group_query_t;

// Registered during host initialization, frozen by first model preparation.
// prepare cannot enqueue work. Context is retained by the registry and each
// selected model; no callback runs under an execution submission lock.
// Non-NULL contexts require paired retain/release callbacks.
typedef struct loom_runtime_plan_provider_v1_t {
  uint32_t struct_size;
  uint32_t abi_version;
  void* self;
  iree_status_t (*prepare)(void* self,
                           const loom_runtime_plan_description_t* description,
                           iree_allocator_t allocator,
                           loom_runtime_execution_plan_t** out_plan);
  void (*retain)(void* self);
  void (*release)(void* self);
  bool (*select)(void* self, const loom_runtime_execution_plan_t* plan,
                 const loom_runtime_plan_group_query_t* query);
} loom_runtime_plan_provider_v1_t;

const loom_runtime_plan_provider_v1_t* loom_runtime_two_bank_plan_provider(
    void);
const loom_runtime_plan_provider_v1_t* loom_runtime_sequential_plan_provider(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_EXECUTION_PLAN_H_
