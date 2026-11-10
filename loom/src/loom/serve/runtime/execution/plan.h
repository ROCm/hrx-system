// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_PLAN_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_PLAN_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_group_mode_e {
  LOOM_SERVE_GROUP_MODE_DEFAULT = 0,
  LOOM_SERVE_GROUP_MODE_SINGLE,
  LOOM_SERVE_GROUP_MODE_TWO_BANK,
} loom_serve_group_mode_t;

typedef enum loom_serve_plan_phase_e {
  // Publishes every layer-local state dependency needed by bank B.
  LOOM_SERVE_PLAN_PREFIX = 0,
  LOOM_SERVE_PLAN_FFN = 1,
  LOOM_SERVE_PLAN_RESIDUAL = 2,
} loom_serve_plan_phase_t;

// Source-owned semantic table, indexed [shape][layer][phase]. UINT32_MAX
// omits an FFN/residual phase; a prefix is required in every layer. Leaves use
// the source extension's declared bank/state resource ABI, never kernel names.
typedef struct loom_serve_plan_description_t {
  iree_host_size_t shape_count;
  iree_host_size_t layer_count;
  iree_host_size_t stage_count;
  const uint32_t* stages;
} loom_serve_plan_description_t;

typedef struct loom_serve_execution_plan_t loom_serve_execution_plan_t;

// Produces an immutable primary-only plan. Validation and cloning happen here,
// before numerical enqueue. Refcounted plans retain source tables by value.
iree_status_t loom_serve_execution_plan_create(
    const loom_serve_plan_description_t* description, bool interleave,
    iree_allocator_t allocator, loom_serve_execution_plan_t** out_plan);
void loom_serve_execution_plan_retain(loom_serve_execution_plan_t* plan);
void loom_serve_execution_plan_release(loom_serve_execution_plan_t* plan);
iree_host_size_t loom_serve_execution_plan_node_capacity(
    const loom_serve_execution_plan_t* plan);

// Visits the causal order outside any execution lock. Each emitted stage is
// appended to an explicit group; emission never accepts numerical work.
typedef iree_status_t (*loom_serve_plan_emit_fn_t)(void* self, uint32_t bank,
                                                   uint32_t stage);
iree_status_t loom_serve_execution_plan_encode(
    const loom_serve_execution_plan_t* plan, iree_host_size_t first_shape,
    iree_host_size_t second_shape, loom_serve_plan_emit_fn_t emit, void* self);

// Immutable chunk-boundary facts. Selection is pure and cannot enqueue work,
// mutate source state, or call back into the model owner.
typedef struct loom_serve_plan_group_query_t {
  iree_host_size_t position;
  iree_host_size_t shapes[2];
  iree_host_size_t tokens[2];
} loom_serve_plan_group_query_t;

// Registered during host initialization, frozen by first model preparation.
// prepare cannot enqueue work. Context is retained by the registry and each
// selected model; no callback runs under an execution submission lock.
// Non-NULL contexts require paired retain/release callbacks.
typedef struct loom_serve_plan_provider_v1_t {
  uint32_t struct_size;
  uint32_t abi_version;
  void* self;
  iree_status_t (*prepare)(void* self,
                           const loom_serve_plan_description_t* description,
                           iree_allocator_t allocator,
                           loom_serve_execution_plan_t** out_plan);
  void (*retain)(void* self);
  void (*release)(void* self);
  bool (*select)(void* self, const loom_serve_execution_plan_t* plan,
                 const loom_serve_plan_group_query_t* query);
} loom_serve_plan_provider_v1_t;

const loom_serve_plan_provider_v1_t* loom_serve_two_bank_plan_provider(void);
const loom_serve_plan_provider_v1_t* loom_serve_sequential_plan_provider(void);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_RUNTIME_EXECUTION_PLAN_H_
