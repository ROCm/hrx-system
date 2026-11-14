// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/execution/plan.h"

struct loom_runtime_execution_plan_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;
  iree_host_size_t shape_count;
  iree_host_size_t layer_count;
  iree_host_size_t phase_count;
  bool interleave;
  uint32_t stages[];
};

bool loom_runtime_execution_plan_equal(
    const loom_runtime_execution_plan_t* lhs,
    const loom_runtime_execution_plan_t* rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs || lhs->shape_count != rhs->shape_count ||
      lhs->layer_count != rhs->layer_count ||
      lhs->phase_count != rhs->phase_count ||
      lhs->interleave != rhs->interleave) {
    return false;
  }
  return memcmp(lhs->stages, rhs->stages,
                lhs->shape_count * lhs->layer_count * lhs->phase_count *
                    sizeof(*lhs->stages)) == 0;
}

iree_status_t loom_runtime_execution_plan_create(
    const loom_runtime_plan_description_t* desc, bool interleave,
    iree_allocator_t allocator, loom_runtime_execution_plan_t** out_plan) {
  *out_plan = NULL;
  if (!desc->shape_count || desc->shape_count > 1024 || !desc->layer_count ||
      desc->layer_count > 1024 || !desc->stage_count || !desc->stages) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid plan geometry (maximum 1024 shapes/layers)");
  }
  const iree_host_size_t phases = desc->phase_count ? desc->phase_count : 3;
  if (phases != 3 && phases != LOOM_RUNTIME_PLAN_FFN_PHASE_COUNT &&
      phases != LOOM_RUNTIME_PLAN_RESIDENT_PHASE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "plan requires 3, 8 or 12 phases");
  }
  const iree_host_size_t count = desc->shape_count * desc->layer_count * phases;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t shape_begin =
        (i / (desc->layer_count * phases)) * desc->layer_count * phases;
    const bool unsupported =
        phases != 3 && desc->stages[shape_begin] == UINT32_MAX;
    if (unsupported) {
      if (desc->stages[i] != UINT32_MAX) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported plan shape must omit every stage");
      }
      continue;
    }
    if (desc->stages[i] >= desc->stage_count &&
        (i % phases == 0 || (phases != 3 && i % phases == phases - 1) ||
         desc->stages[i] != UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "plan stage %zu is not loaded", i);
    }
  }
  if (phases == LOOM_RUNTIME_PLAN_RESIDENT_PHASE_COUNT) {
    for (iree_host_size_t i = 0; i < count; i += phases) {
      if (desc->stages[i] == UINT32_MAX) {
        continue;
      }
      const uint32_t starts[] = {1, 4, 8};
      for (iree_host_size_t p = 0; p < IREE_ARRAYSIZE(starts); ++p) {
        if ((desc->stages[i + starts[p]] != UINT32_MAX ||
             desc->stages[i + starts[p] + 1] != UINT32_MAX) &&
            desc->stages[i + starts[p] + 2] == UINT32_MAX) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "resident projection requires its consumer");
        }
      }
    }
  }
  if (phases == LOOM_RUNTIME_PLAN_FFN_PHASE_COUNT) {
    for (iree_host_size_t i = 0; i < count; i += phases) {
      const uint32_t* row = desc->stages + i;
      if ((row[LOOM_RUNTIME_PLAN_FFN_SECONDARY] != UINT32_MAX &&
           row[LOOM_RUNTIME_PLAN_FFN_MERGE] == UINT32_MAX) ||
          (row[LOOM_RUNTIME_PLAN_DOWN_SECONDARY] != UINT32_MAX &&
           row[LOOM_RUNTIME_PLAN_DOWN_MERGE] == UINT32_MAX)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "secondary FFN output requires a merge");
      }
    }
  }
  loom_runtime_execution_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      allocator, sizeof(*plan), count, sizeof(*plan->stages), (void**)&plan));
  iree_atomic_ref_count_init(&plan->ref_count);
  plan->allocator = allocator;
  plan->shape_count = desc->shape_count;
  plan->layer_count = desc->layer_count;
  plan->phase_count = phases;
  plan->interleave = interleave;
  memcpy(plan->stages, desc->stages, count * sizeof(*plan->stages));
  *out_plan = plan;
  return iree_ok_status();
}

void loom_runtime_execution_plan_retain(loom_runtime_execution_plan_t* plan) {
  if (plan) {
    iree_atomic_ref_count_inc(&plan->ref_count);
  }
}
void loom_runtime_execution_plan_release(loom_runtime_execution_plan_t* plan) {
  if (plan && iree_atomic_ref_count_dec(&plan->ref_count) == 1) {
    iree_allocator_free(plan->allocator, plan);
  }
}
iree_host_size_t loom_runtime_execution_plan_node_capacity(
    const loom_runtime_execution_plan_t* plan) {
  return 2 * plan->phase_count * plan->layer_count;
}

typedef struct loom_runtime_plan_encoder_t {
  const loom_runtime_execution_plan_t* plan;
  iree_host_size_t shapes[2];
  loom_runtime_plan_emit_dag_fn_t emit;
  void* self;
  uint32_t prefix[2];
  uint32_t branch[2][2];
  uint32_t last[2];
  uint32_t ordered;
} loom_runtime_plan_encoder_t;

static iree_status_t loom_runtime_plan_emit(
    loom_runtime_plan_encoder_t* encoder, uint32_t bank, iree_host_size_t layer,
    uint32_t phase, uint32_t first, uint32_t second, uint32_t* out_node) {
  const loom_runtime_execution_plan_t* plan = encoder->plan;
  const uint32_t stage =
      plan->stages[(encoder->shapes[bank] * plan->layer_count + layer) *
                       plan->phase_count +
                   phase];
  *out_node = first;
  if (stage == UINT32_MAX) {
    return iree_ok_status();
  }
  uint32_t predecessors[2];
  iree_host_size_t count = 0;
  if (first != UINT32_MAX) {
    predecessors[count++] = first;
  }
  if (second != UINT32_MAX && second != first) {
    predecessors[count++] = second;
  }
  return encoder->emit(encoder->self, bank, stage, count, predecessors,
                       out_node);
}

static iree_status_t loom_runtime_plan_prefix(
    loom_runtime_plan_encoder_t* encoder, uint32_t bank,
    iree_host_size_t layer) {
  if (encoder->plan->phase_count == 3) {
    for (uint32_t phase = 0; phase < 2; ++phase) {
      IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(encoder, bank, layer, phase,
                                                  encoder->ordered, UINT32_MAX,
                                                  &encoder->ordered));
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(
        encoder, bank, layer, LOOM_RUNTIME_PLAN_FFN_PREFIX, encoder->last[bank],
        bank ? encoder->prefix[0] : UINT32_MAX, &encoder->prefix[bank]));
    if (encoder->plan->phase_count == LOOM_RUNTIME_PLAN_RESIDENT_PHASE_COUNT) {
      return iree_ok_status();
    }
    for (uint32_t branch = 0; branch < 2; ++branch) {
      IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(
          encoder, bank, layer, LOOM_RUNTIME_PLAN_FFN_PRIMARY + branch,
          encoder->prefix[bank], UINT32_MAX, &encoder->branch[bank][branch]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_runtime_plan_residual(
    loom_runtime_plan_encoder_t* encoder, uint32_t bank,
    iree_host_size_t layer) {
  if (encoder->plan->phase_count == 3) {
    return loom_runtime_plan_emit(encoder, bank, layer,
                                  LOOM_RUNTIME_PLAN_RESIDUAL, encoder->ordered,
                                  UINT32_MAX, &encoder->ordered);
  }
  uint32_t joined = UINT32_MAX;
  if (encoder->plan->phase_count == LOOM_RUNTIME_PLAN_RESIDENT_PHASE_COUNT) {
    joined = encoder->prefix[bank];
    const uint32_t starts[] = {LOOM_RUNTIME_PLAN_GATE_PRIMARY,
                               LOOM_RUNTIME_PLAN_UP_PRIMARY,
                               LOOM_RUNTIME_PLAN_RESIDENT_DOWN_PRIMARY};
    for (iree_host_size_t p = 0; p < IREE_ARRAYSIZE(starts); ++p) {
      if (p == 2) {
        IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(
            encoder, bank, layer, LOOM_RUNTIME_PLAN_ACTIVATION, joined,
            UINT32_MAX, &joined));
      }
      for (uint32_t branch = 0; branch < 2; ++branch) {
        IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(
            encoder, bank, layer, starts[p] + branch, joined, UINT32_MAX,
            &encoder->branch[bank][branch]));
      }
      IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(
          encoder, bank, layer, starts[p] + 2, encoder->branch[bank][0],
          encoder->branch[bank][1], &joined));
    }
    return loom_runtime_plan_emit(encoder, bank, layer,
                                  LOOM_RUNTIME_PLAN_RESIDENT_RESIDUAL, joined,
                                  UINT32_MAX, &encoder->last[bank]);
  }
  IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(
      encoder, bank, layer, LOOM_RUNTIME_PLAN_FFN_MERGE,
      encoder->branch[bank][0], encoder->branch[bank][1], &joined));
  for (uint32_t branch = 0; branch < 2; ++branch) {
    IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(
        encoder, bank, layer, LOOM_RUNTIME_PLAN_DOWN_PRIMARY + branch, joined,
        UINT32_MAX, &encoder->branch[bank][branch]));
  }
  IREE_RETURN_IF_ERROR(loom_runtime_plan_emit(
      encoder, bank, layer, LOOM_RUNTIME_PLAN_DOWN_MERGE,
      encoder->branch[bank][0], encoder->branch[bank][1], &joined));
  return loom_runtime_plan_emit(encoder, bank, layer,
                                LOOM_RUNTIME_PLAN_FFN_RESIDUAL, joined,
                                UINT32_MAX, &encoder->last[bank]);
}

iree_status_t loom_runtime_execution_plan_encode_dag(
    const loom_runtime_execution_plan_t* plan, iree_host_size_t a,
    iree_host_size_t b, loom_runtime_plan_emit_dag_fn_t emit, void* self) {
  if (a >= plan->shape_count || b >= plan->shape_count || !emit) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "plan shape is not prepared");
  }
  if (plan->stages[a * plan->layer_count * plan->phase_count] == UINT32_MAX ||
      plan->stages[b * plan->layer_count * plan->phase_count] == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "plan shape has no grouped implementation");
  }
  loom_runtime_plan_encoder_t encoder = {.plan = plan,
                                         .shapes = {a, b},
                                         .emit = emit,
                                         .self = self,
                                         .prefix = {UINT32_MAX, UINT32_MAX},
                                         .last = {UINT32_MAX, UINT32_MAX},
                                         .ordered = UINT32_MAX};
  if (!plan->interleave) {
    for (uint32_t bank = 0; bank < 2; ++bank) {
      if (bank) {
        encoder.last[bank] = encoder.last[0];
      }
      for (iree_host_size_t layer = 0; layer < plan->layer_count; ++layer) {
        IREE_RETURN_IF_ERROR(loom_runtime_plan_prefix(&encoder, bank, layer));
        IREE_RETURN_IF_ERROR(loom_runtime_plan_residual(&encoder, bank, layer));
      }
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_runtime_plan_prefix(&encoder, 0, 0));
    for (iree_host_size_t layer = 0; layer < plan->layer_count; ++layer) {
      IREE_RETURN_IF_ERROR(loom_runtime_plan_prefix(&encoder, 1, layer));
      IREE_RETURN_IF_ERROR(loom_runtime_plan_residual(&encoder, 0, layer));
      if (layer + 1 < plan->layer_count) {
        IREE_RETURN_IF_ERROR(loom_runtime_plan_prefix(&encoder, 0, layer + 1));
      }
      IREE_RETURN_IF_ERROR(loom_runtime_plan_residual(&encoder, 1, layer));
    }
  }
  return iree_ok_status();
}

typedef struct loom_runtime_plan_ordered_emitter_t {
  loom_runtime_plan_emit_fn_t emit;
  void* self;
  uint32_t count;
} loom_runtime_plan_ordered_emitter_t;

static iree_status_t loom_runtime_plan_emit_ordered(
    void* self, uint32_t bank, uint32_t stage, iree_host_size_t count,
    const uint32_t* predecessors, uint32_t* out_node) {
  loom_runtime_plan_ordered_emitter_t* emitter = self;
  IREE_RETURN_IF_ERROR(emitter->emit(emitter->self, bank, stage));
  *out_node = emitter->count++;
  return iree_ok_status();
}

iree_status_t loom_runtime_execution_plan_encode(
    const loom_runtime_execution_plan_t* plan, iree_host_size_t a,
    iree_host_size_t b, loom_runtime_plan_emit_fn_t emit, void* self) {
  if (!emit) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "emitter required");
  }
  loom_runtime_plan_ordered_emitter_t emitter = {.emit = emit, .self = self};
  return loom_runtime_execution_plan_encode_dag(
      plan, a, b, loom_runtime_plan_emit_ordered, &emitter);
}

static iree_status_t loom_runtime_plan_prepare_two_bank(
    void* self, const loom_runtime_plan_description_t* desc,
    iree_allocator_t allocator, loom_runtime_execution_plan_t** out_plan) {
  return loom_runtime_execution_plan_create(desc, true, allocator, out_plan);
}
static iree_status_t loom_runtime_plan_prepare_sequential(
    void* self, const loom_runtime_plan_description_t* desc,
    iree_allocator_t allocator, loom_runtime_execution_plan_t** out_plan) {
  return loom_runtime_execution_plan_create(desc, false, allocator, out_plan);
}
static bool loom_runtime_plan_select(
    void* self, const loom_runtime_execution_plan_t* plan,
    const loom_runtime_plan_group_query_t* query) {
  return query->tokens[0] && query->tokens[1] &&
         query->shapes[0] < plan->shape_count &&
         query->shapes[1] < plan->shape_count &&
         plan->stages[query->shapes[0] * plan->layer_count *
                      plan->phase_count] != UINT32_MAX &&
         plan->stages[query->shapes[1] * plan->layer_count *
                      plan->phase_count] != UINT32_MAX;
}

static const loom_runtime_plan_provider_v1_t two_bank = {
    .struct_size = sizeof(loom_runtime_plan_provider_v1_t),
    .abi_version = 1,
    .prepare = loom_runtime_plan_prepare_two_bank,
    .select = loom_runtime_plan_select};
static const loom_runtime_plan_provider_v1_t sequential = {
    .struct_size = sizeof(loom_runtime_plan_provider_v1_t),
    .abi_version = 1,
    .prepare = loom_runtime_plan_prepare_sequential,
    .select = loom_runtime_plan_select};
const loom_runtime_plan_provider_v1_t* loom_runtime_two_bank_plan_provider(
    void) {
  return &two_bank;
}
const loom_runtime_plan_provider_v1_t* loom_runtime_sequential_plan_provider(
    void) {
  return &sequential;
}
