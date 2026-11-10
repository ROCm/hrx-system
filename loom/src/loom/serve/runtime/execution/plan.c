// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/plan.h"

struct loom_serve_execution_plan_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;
  iree_host_size_t shape_count;
  iree_host_size_t layer_count;
  bool interleave;
  uint32_t stages[];
};

iree_status_t loom_serve_execution_plan_create(
    const loom_serve_plan_description_t* desc, bool interleave,
    iree_allocator_t allocator, loom_serve_execution_plan_t** out_plan) {
  *out_plan = NULL;
  if (!desc->shape_count || desc->shape_count > 1024 || !desc->layer_count ||
      desc->layer_count > 1024 || !desc->stage_count || !desc->stages) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid plan geometry (maximum 1024 shapes/layers)");
  }
  const iree_host_size_t count = desc->shape_count * desc->layer_count * 3;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (desc->stages[i] >= desc->stage_count &&
        (i % 3 == LOOM_SERVE_PLAN_PREFIX || desc->stages[i] != UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "plan stage %zu is not loaded", i);
    }
  }
  loom_serve_execution_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      allocator, sizeof(*plan), count, sizeof(*plan->stages), (void**)&plan));
  iree_atomic_ref_count_init(&plan->ref_count);
  plan->allocator = allocator;
  plan->shape_count = desc->shape_count;
  plan->layer_count = desc->layer_count;
  plan->interleave = interleave;
  memcpy(plan->stages, desc->stages, count * sizeof(*plan->stages));
  *out_plan = plan;
  return iree_ok_status();
}

void loom_serve_execution_plan_retain(loom_serve_execution_plan_t* plan) {
  if (plan) {
    iree_atomic_ref_count_inc(&plan->ref_count);
  }
}
void loom_serve_execution_plan_release(loom_serve_execution_plan_t* plan) {
  if (plan && iree_atomic_ref_count_dec(&plan->ref_count) == 1) {
    iree_allocator_free(plan->allocator, plan);
  }
}
iree_host_size_t loom_serve_execution_plan_node_capacity(
    const loom_serve_execution_plan_t* plan) {
  return 6 * plan->layer_count;
}

static iree_status_t loom_serve_plan_emit(
    const loom_serve_execution_plan_t* plan, iree_host_size_t shape,
    uint32_t bank, iree_host_size_t layer, uint32_t phase,
    loom_serve_plan_emit_fn_t emit, void* self) {
  uint32_t stage =
      plan->stages[(shape * plan->layer_count + layer) * 3 + phase];
  return stage == UINT32_MAX ? iree_ok_status() : emit(self, bank, stage);
}

iree_status_t loom_serve_execution_plan_encode(
    const loom_serve_execution_plan_t* plan, iree_host_size_t a,
    iree_host_size_t b, loom_serve_plan_emit_fn_t emit, void* self) {
  if (a >= plan->shape_count || b >= plan->shape_count || !emit) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "plan shape is not prepared");
  }
  if (!plan->interleave) {
    for (uint32_t bank = 0; bank < 2; ++bank) {
      for (iree_host_size_t layer = 0; layer < plan->layer_count; ++layer) {
        for (uint32_t phase = 0; phase < 3; ++phase) {
          IREE_RETURN_IF_ERROR(loom_serve_plan_emit(plan, bank ? b : a, bank,
                                                    layer, phase, emit, self));
        }
      }
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_serve_plan_emit(plan, a, 0, 0, 0, emit, self));
    IREE_RETURN_IF_ERROR(loom_serve_plan_emit(plan, a, 0, 0, 1, emit, self));
    for (iree_host_size_t layer = 0; layer < plan->layer_count; ++layer) {
      IREE_RETURN_IF_ERROR(
          loom_serve_plan_emit(plan, b, 1, layer, 0, emit, self));
      IREE_RETURN_IF_ERROR(
          loom_serve_plan_emit(plan, b, 1, layer, 1, emit, self));
      IREE_RETURN_IF_ERROR(
          loom_serve_plan_emit(plan, a, 0, layer, 2, emit, self));
      if (layer + 1 < plan->layer_count) {
        IREE_RETURN_IF_ERROR(
            loom_serve_plan_emit(plan, a, 0, layer + 1, 0, emit, self));
        IREE_RETURN_IF_ERROR(
            loom_serve_plan_emit(plan, a, 0, layer + 1, 1, emit, self));
      }
      IREE_RETURN_IF_ERROR(
          loom_serve_plan_emit(plan, b, 1, layer, 2, emit, self));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_plan_prepare_two_bank(
    void* self, const loom_serve_plan_description_t* desc,
    iree_allocator_t allocator, loom_serve_execution_plan_t** out_plan) {
  return loom_serve_execution_plan_create(desc, true, allocator, out_plan);
}
static iree_status_t loom_serve_plan_prepare_sequential(
    void* self, const loom_serve_plan_description_t* desc,
    iree_allocator_t allocator, loom_serve_execution_plan_t** out_plan) {
  return loom_serve_execution_plan_create(desc, false, allocator, out_plan);
}
static bool loom_serve_plan_select(void* self,
                                   const loom_serve_execution_plan_t* plan,
                                   const loom_serve_plan_group_query_t* query) {
  return query->tokens[0] && query->tokens[1] &&
         query->shapes[0] < plan->shape_count &&
         query->shapes[1] < plan->shape_count;
}

static const loom_serve_plan_provider_v1_t two_bank = {
    .struct_size = sizeof(loom_serve_plan_provider_v1_t),
    .abi_version = 1,
    .prepare = loom_serve_plan_prepare_two_bank,
    .select = loom_serve_plan_select};
static const loom_serve_plan_provider_v1_t sequential = {
    .struct_size = sizeof(loom_serve_plan_provider_v1_t),
    .abi_version = 1,
    .prepare = loom_serve_plan_prepare_sequential,
    .select = loom_serve_plan_select};
const loom_serve_plan_provider_v1_t* loom_serve_two_bank_plan_provider(void) {
  return &two_bank;
}
const loom_serve_plan_provider_v1_t* loom_serve_sequential_plan_provider(void) {
  return &sequential;
}
