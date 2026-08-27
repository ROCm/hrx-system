// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/bank_sroa.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_VECTOR_BANK_SROA_STATISTICS(V, statistics_type)     \
  V(statistics_type, loops_checked, "loops-checked",             \
    "Number of scf.for loops checked for vector banks.")         \
  V(statistics_type, loops_scalarized, "loops-scalarized",       \
    "Number of scf.for loops rebuilt with scalarized banks.")    \
  V(statistics_type, banks_scalarized, "banks-scalarized",       \
    "Number of loop-carried vector banks scalarized.")           \
  V(statistics_type, slots_materialized, "slots-materialized",   \
    "Number of scalar or tail-vector bank slots materialized.")  \
  V(statistics_type, extracts_eliminated, "extracts-eliminated", \
    "Number of vector.extract operations eliminated.")           \
  V(statistics_type, inserts_eliminated, "inserts-eliminated",   \
    "Number of vector.insert operations eliminated.")

LOOM_PASS_STATISTICS_DEFINE(loom_vector_bank_sroa_statistics,
                            loom_vector_bank_sroa_statistics_t,
                            LOOM_VECTOR_BANK_SROA_STATISTICS)

static const loom_pass_info_t loom_vector_bank_sroa_pass_info_storage = {
    .name = IREE_SVL("sroa-vector-banks"),
    .description = IREE_SVL(
        "Split statically addressed loop-carried vector banks into slots."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_vector_bank_sroa_statistics_layout,
};

const loom_pass_info_t* loom_vector_bank_sroa_pass_info(void) {
  return &loom_vector_bank_sroa_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Planning
//===----------------------------------------------------------------------===//

typedef struct loom_vector_bank_sroa_bank_t {
  // Original loop-carried ordinal.
  uint16_t ordinal;
  // Number of leading dimensions used to address one payload slot.
  uint8_t prefix_rank;
  // Number of statically addressed payload slots.
  uint16_t slot_count;
  // Original aggregate vector type.
  loom_type_t bank_type;
  // Scalar or trailing vector type carried for each slot.
  loom_type_t payload_type;
  // First expanded loop-carried ordinal assigned to this bank.
  uint16_t expanded_base;
  // True when an exact-prefix extract or insert selected this bank.
  bool active;
} loom_vector_bank_sroa_bank_t;

typedef struct loom_vector_bank_sroa_plan_t {
  // Loop body block being rebuilt.
  loom_block_t* body_block;
  // Loop body scf.yield terminator.
  loom_op_t* yield;
  // One potential bank descriptor per loop-carried ordinal.
  loom_vector_bank_sroa_bank_t* banks;
  // Number of loop-carried ordinals.
  uint16_t carried_count;
  // Expanded loop-carried result count after scalar replacement.
  uint16_t expanded_count;
  // Original value ID -> bank ordinal for aggregate recurrence states.
  uint16_t* state_bank_by_value;
  // Number of value IDs represented by state_bank_by_value.
  iree_host_size_t value_snapshot_count;
  // Number of active banks in banks.
  uint16_t active_bank_count;
} loom_vector_bank_sroa_plan_t;

typedef struct loom_vector_bank_sroa_loop_list_t {
  // Collected scf.for operations in post-order.
  loom_op_t** ops;
  // Number of collected loops.
  iree_host_size_t count;
  // Allocated loop pointer capacity.
  iree_host_size_t capacity;
} loom_vector_bank_sroa_loop_list_t;

typedef struct loom_vector_bank_sroa_collect_context_t {
  // Arena owning the collected pointer list.
  iree_arena_allocator_t* arena;
  // Post-order loop list being populated.
  loom_vector_bank_sroa_loop_list_t* loops;
} loom_vector_bank_sroa_collect_context_t;

typedef struct loom_vector_bank_sroa_context_t {
  // Current pass invocation.
  loom_pass_t* pass;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for cloning, replacement, and erasure.
  loom_rewriter_t* rewriter;
  // Candidate-local scratch arena restored after every loop.
  iree_arena_allocator_t* scratch_arena;
  // Typed pass statistics.
  loom_vector_bank_sroa_statistics_t* statistics;
} loom_vector_bank_sroa_context_t;

static iree_status_t loom_vector_bank_sroa_emit_error(
    loom_vector_bank_sroa_context_t* context, loom_op_t* op, uint16_t slot,
    iree_string_view_t reason) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(context->module, op)),
      loom_param_string(context->pass->info->name),
      loom_param_u32(slot),
      loom_param_string(reason),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_LOWERING_047,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(context->pass->diagnostic_emitter, &emission);
}

static bool loom_vector_bank_sroa_state_bank(
    const loom_vector_bank_sroa_plan_t* plan, loom_value_id_t value,
    uint16_t* out_bank_ordinal) {
  *out_bank_ordinal = UINT16_MAX;
  if (value == LOOM_VALUE_ID_INVALID || value >= plan->value_snapshot_count) {
    return false;
  }
  uint16_t bank_ordinal = plan->state_bank_by_value[value];
  if (bank_ordinal == UINT16_MAX) return false;
  *out_bank_ordinal = bank_ordinal;
  return true;
}

static iree_status_t loom_vector_bank_sroa_payload_type(
    loom_module_t* module, loom_type_t bank_type, uint8_t prefix_rank,
    loom_type_t* out_payload_type) {
  uint8_t bank_rank = loom_type_rank(bank_type);
  uint8_t payload_rank = (uint8_t)(bank_rank - prefix_rank);
  if (payload_rank == 0) {
    *out_payload_type = loom_type_scalar(loom_type_element_type(bank_type));
    return iree_ok_status();
  }

  loom_overflow_dim_t payload_dims[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t i = 0; i < payload_rank; ++i) {
    payload_dims[i] = loom_type_dim(bank_type, (uint8_t)(prefix_rank + i));
  }

  loom_type_t payload_type = {0};
  uint8_t flags = LOOM_TYPE_FLAG_ALL_STATIC;
  if (payload_rank <= 2) flags |= LOOM_TYPE_FLAG_INLINE_DIMS;
  payload_type.header = loom_type_make_header(
      LOOM_TYPE_VECTOR, loom_type_element_type(bank_type), payload_rank, flags);
  if (payload_rank <= 2) {
    for (uint8_t i = 0; i < payload_rank; ++i) {
      payload_type.dims[i] = payload_dims[i];
    }
  } else {
    payload_type.dims[0] = (uint64_t)(uintptr_t)payload_dims;
  }
  return loom_module_intern_type(module, payload_type, out_payload_type);
}

static bool loom_vector_bank_sroa_static_access_ordinal(
    const loom_vector_bank_sroa_bank_t* bank, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices, loom_type_t payload_type,
    uint16_t* out_slot) {
  *out_slot = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != bank->prefix_rank ||
      (static_indices.count != 0 && !static_indices.i64_array) ||
      dynamic_indices.count != 0 ||
      !loom_type_equal(payload_type, bank->payload_type)) {
    return false;
  }

  uint32_t slot = 0;
  for (uint8_t axis = 0; axis < bank->prefix_rank; ++axis) {
    int64_t index = static_indices.i64_array[axis];
    int64_t extent = loom_type_dim_static_size_at(bank->bank_type, axis);
    if (index < 0 || index == INT64_MIN || index >= extent) return false;
    slot = slot * (uint32_t)extent + (uint32_t)index;
  }
  if (slot >= bank->slot_count) return false;
  *out_slot = (uint16_t)slot;
  return true;
}

static iree_status_t loom_vector_bank_sroa_prepare_access(
    loom_vector_bank_sroa_context_t* context, loom_op_t* access_op,
    loom_vector_bank_sroa_bank_t* bank, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices, loom_type_t payload_type,
    uint16_t* out_slot) {
  *out_slot = 0;
  uint8_t bank_rank = loom_type_rank(bank->bank_type);
  if (!loom_type_is_all_static(bank->bank_type)) {
    return loom_vector_bank_sroa_emit_error(
        context, access_op, bank->ordinal,
        IREE_SV("the bank has a dynamic extent"));
  }
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count == 0 ||
      static_indices.count > bank_rank || !static_indices.i64_array) {
    return loom_vector_bank_sroa_emit_error(
        context, access_op, bank->ordinal,
        IREE_SV("the access does not select a non-empty leading prefix"));
  }
  if (dynamic_indices.count != 0) {
    return loom_vector_bank_sroa_emit_error(
        context, access_op, bank->ordinal,
        IREE_SV("the access has a dynamic prefix index"));
  }
  for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
    int64_t index = static_indices.i64_array[axis];
    int64_t extent = loom_type_dim_static_size_at(bank->bank_type, axis);
    if (index < 0 || index == INT64_MIN || index >= extent) {
      return loom_vector_bank_sroa_emit_error(
          context, access_op, bank->ordinal,
          IREE_SV("the static prefix index is outside the bank extent"));
    }
  }

  if (!bank->active) {
    bank->prefix_rank = (uint8_t)static_indices.count;
    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_payload_type(
        context->module, bank->bank_type, bank->prefix_rank,
        &bank->payload_type));

    uint32_t slot_count = 1;
    for (uint8_t axis = 0; axis < bank->prefix_rank; ++axis) {
      int64_t extent = loom_type_dim_static_size_at(bank->bank_type, axis);
      if (extent <= 0 || (uint64_t)slot_count * (uint64_t)extent > UINT16_MAX) {
        return loom_vector_bank_sroa_emit_error(
            context, access_op, bank->ordinal,
            IREE_SV("the static prefix has no representable payload slots"));
      }
      slot_count *= (uint32_t)extent;
    }
    bank->slot_count = (uint16_t)slot_count;
    bank->active = true;
  }

  if (!loom_vector_bank_sroa_static_access_ordinal(
          bank, static_indices, dynamic_indices, payload_type, out_slot)) {
    return loom_vector_bank_sroa_emit_error(
        context, access_op, bank->ordinal,
        IREE_SV("the access prefix or payload type differs from the bank"));
  }
  return iree_ok_status();
}

static bool loom_vector_bank_sroa_loop_shape(loom_op_t* op,
                                             loom_block_t** out_body_block,
                                             loom_op_t** out_yield) {
  *out_body_block = NULL;
  *out_yield = NULL;
  loom_region_t* body = loom_scf_for_body(op);
  if (!body || body->block_count != 1) return false;
  loom_block_t* block = loom_region_entry_block(body);
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  if (iter_args.count != op->result_count ||
      block->arg_count != (iree_host_size_t)op->result_count + 1) {
    return false;
  }
  loom_op_t* yield = block->last_op;
  if (!yield || !loom_scf_yield_isa(yield) ||
      loom_scf_yield_values(yield).count != op->result_count) {
    return false;
  }
  *out_body_block = block;
  *out_yield = yield;
  return true;
}

static iree_status_t loom_vector_bank_sroa_scan_states(
    loom_vector_bank_sroa_context_t* context, loom_op_t* loop,
    loom_vector_bank_sroa_plan_t* plan) {
  const loom_value_id_t* results = loom_op_const_results(loop);
  loom_value_slice_t iter_args = loom_scf_for_iter_args(loop);
  for (uint16_t i = 0; i < plan->carried_count; ++i) {
    loom_type_t bank_type = loom_module_value_type(context->module, results[i]);
    plan->banks[i] = (loom_vector_bank_sroa_bank_t){
        .ordinal = i,
        .bank_type = bank_type,
    };
    if (!loom_type_is_vector(bank_type) || loom_type_rank(bank_type) < 2) {
      continue;
    }
    loom_value_id_t body_arg =
        loom_block_arg_id(plan->body_block, (uint16_t)(1 + i));
    loom_type_t body_type = loom_module_value_type(context->module, body_arg);
    loom_type_t initial_type =
        loom_module_value_type(context->module, iter_args.values[i]);
    if (!loom_type_equal(bank_type, body_type) ||
        !loom_type_equal(bank_type, initial_type)) {
      continue;
    }
    plan->state_bank_by_value[body_arg] = i;
  }

  loom_op_t* child_op = NULL;
  loom_block_for_each_op(plan->body_block, child_op) {
    if (child_op == plan->yield) break;
    uint16_t bank_ordinal = UINT16_MAX;
    if (loom_vector_extract_isa(child_op) &&
        loom_vector_bank_sroa_state_bank(
            plan, loom_vector_extract_source(child_op), &bank_ordinal)) {
      loom_vector_bank_sroa_bank_t* bank = &plan->banks[bank_ordinal];
      uint16_t slot = 0;
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_prepare_access(
          context, child_op, bank, loom_vector_extract_static_indices(child_op),
          loom_vector_extract_indices(child_op),
          loom_module_value_type(context->module,
                                 loom_vector_extract_result(child_op)),
          &slot));
      (void)slot;
      continue;
    }
    if (loom_vector_insert_isa(child_op) &&
        loom_vector_bank_sroa_state_bank(
            plan, loom_vector_insert_dest(child_op), &bank_ordinal)) {
      loom_vector_bank_sroa_bank_t* bank = &plan->banks[bank_ordinal];
      uint16_t slot = 0;
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_prepare_access(
          context, child_op, bank, loom_vector_insert_static_indices(child_op),
          loom_vector_insert_indices(child_op),
          loom_module_value_type(context->module,
                                 loom_vector_insert_value(child_op)),
          &slot));
      (void)slot;
      loom_value_id_t result = loom_vector_insert_result(child_op);
      if (result >= plan->value_snapshot_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "vector.insert result is outside pass snapshot");
      }
      plan->state_bank_by_value[result] = bank_ordinal;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_validate_state_uses(
    loom_vector_bank_sroa_context_t* context,
    const loom_vector_bank_sroa_plan_t* plan, loom_value_id_t state,
    uint16_t bank_ordinal) {
  const loom_vector_bank_sroa_bank_t* bank = &plan->banks[bank_ordinal];
  const loom_value_t* value = loom_module_value(context->module, state);
  if (loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(context->module, state)) {
    return loom_vector_bank_sroa_emit_error(
        context, plan->yield, bank_ordinal,
        IREE_SV("an aggregate state escapes through an attribute or type"));
  }

  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    loom_op_t* user = loom_use_user_op(*use);
    uint16_t operand_index = loom_use_operand_index(*use);
    if (user == plan->yield && operand_index == bank_ordinal) continue;

    loom_attribute_t static_indices = {0};
    loom_value_slice_t dynamic_indices = {0};
    loom_type_t payload_type = {0};
    bool recognized = false;
    if (loom_vector_extract_isa(user) && operand_index == 0 &&
        user->parent_block == plan->body_block) {
      recognized = true;
      static_indices = loom_vector_extract_static_indices(user);
      dynamic_indices = loom_vector_extract_indices(user);
      payload_type = loom_module_value_type(context->module,
                                            loom_vector_extract_result(user));
    } else if (loom_vector_insert_isa(user) && operand_index == 1 &&
               user->parent_block == plan->body_block) {
      recognized = true;
      static_indices = loom_vector_insert_static_indices(user);
      dynamic_indices = loom_vector_insert_indices(user);
      payload_type = loom_module_value_type(context->module,
                                            loom_vector_insert_value(user));
    }
    uint16_t slot = 0;
    if (!recognized ||
        !loom_vector_bank_sroa_static_access_ordinal(
            bank, static_indices, dynamic_indices, payload_type, &slot)) {
      return loom_vector_bank_sroa_emit_error(
          context, user, bank_ordinal,
          IREE_SV("an aggregate state has a non-prefix or nested-region use"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_validate_result_uses(
    loom_vector_bank_sroa_context_t* context, loom_op_t* loop,
    const loom_vector_bank_sroa_plan_t* plan, uint16_t bank_ordinal) {
  const loom_vector_bank_sroa_bank_t* bank = &plan->banks[bank_ordinal];
  loom_value_id_t result = loom_op_const_results(loop)[bank_ordinal];
  const loom_value_t* value = loom_module_value(context->module, result);
  if (loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(context->module, result)) {
    return loom_vector_bank_sroa_emit_error(
        context, loop, bank_ordinal,
        IREE_SV("the aggregate result escapes through an attribute or type"));
  }

  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    loom_op_t* user = loom_use_user_op(*use);
    if (!loom_vector_extract_isa(user) || loom_use_operand_index(*use) != 0) {
      return loom_vector_bank_sroa_emit_error(
          context, user, bank_ordinal,
          IREE_SV("the aggregate result has a non-extract use"));
    }
    uint16_t slot = 0;
    if (!loom_vector_bank_sroa_static_access_ordinal(
            bank, loom_vector_extract_static_indices(user),
            loom_vector_extract_indices(user),
            loom_module_value_type(context->module,
                                   loom_vector_extract_result(user)),
            &slot)) {
      return loom_vector_bank_sroa_emit_error(
          context, user, bank_ordinal,
          IREE_SV("the aggregate result has an incompatible extract"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_plan_loop(
    loom_vector_bank_sroa_context_t* context, loom_op_t* loop,
    loom_vector_bank_sroa_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  ++context->statistics->loops_checked;

  loom_block_t* body_block = NULL;
  loom_op_t* yield = NULL;
  if (!loom_vector_bank_sroa_loop_shape(loop, &body_block, &yield) ||
      loop->result_count == 0) {
    return iree_ok_status();
  }

  out_plan->body_block = body_block;
  out_plan->yield = yield;
  out_plan->carried_count = loop->result_count;
  out_plan->value_snapshot_count = context->module->values.count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->scratch_arena, out_plan->carried_count, sizeof(*out_plan->banks),
      (void**)&out_plan->banks));
  memset(out_plan->banks, 0,
         (iree_host_size_t)out_plan->carried_count * sizeof(*out_plan->banks));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->scratch_arena, out_plan->value_snapshot_count,
      sizeof(*out_plan->state_bank_by_value),
      (void**)&out_plan->state_bank_by_value));
  memset(
      out_plan->state_bank_by_value, 0xFF,
      out_plan->value_snapshot_count * sizeof(*out_plan->state_bank_by_value));

  IREE_RETURN_IF_ERROR(
      loom_vector_bank_sroa_scan_states(context, loop, out_plan));
  if (loom_pass_has_error_diagnostics(context->pass)) return iree_ok_status();

  loom_value_slice_t yielded = loom_scf_yield_values(yield);
  uint32_t expanded_count = 0;
  for (uint16_t i = 0; i < out_plan->carried_count; ++i) {
    loom_vector_bank_sroa_bank_t* bank = &out_plan->banks[i];
    if (!bank->active) {
      if (expanded_count == UINT16_MAX) {
        return loom_vector_bank_sroa_emit_error(
            context, loop, i,
            IREE_SV("the expanded loop result count exceeds uint16"));
      }
      ++expanded_count;
      continue;
    }
    ++out_plan->active_bank_count;
    uint16_t yielded_bank = UINT16_MAX;
    if (!loom_vector_bank_sroa_state_bank(out_plan, yielded.values[i],
                                          &yielded_bank) ||
        yielded_bank != i) {
      return loom_vector_bank_sroa_emit_error(
          context, yield, i,
          IREE_SV("the yielded aggregate is not a known bank state"));
    }
    if (loop->tied_result_count != 0) {
      return loom_vector_bank_sroa_emit_error(
          context, loop, i, IREE_SV("the loop carries tied-result metadata"));
    }
    if (expanded_count > UINT16_MAX - bank->slot_count) {
      return loom_vector_bank_sroa_emit_error(
          context, loop, i,
          IREE_SV("the expanded loop result count exceeds uint16"));
    }
    expanded_count += bank->slot_count;
  }
  if (out_plan->active_bank_count == 0) return iree_ok_status();
  out_plan->expanded_count = (uint16_t)expanded_count;

  for (loom_value_id_t state = 0; state < out_plan->value_snapshot_count;
       ++state) {
    uint16_t bank_ordinal = out_plan->state_bank_by_value[state];
    if (bank_ordinal == UINT16_MAX || !out_plan->banks[bank_ordinal].active) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_validate_state_uses(
        context, out_plan, state, bank_ordinal));
    if (loom_pass_has_error_diagnostics(context->pass)) return iree_ok_status();
  }
  for (uint16_t i = 0; i < out_plan->carried_count; ++i) {
    if (!out_plan->banks[i].active) continue;
    IREE_RETURN_IF_ERROR(
        loom_vector_bank_sroa_validate_result_uses(context, loop, out_plan, i));
    if (loom_pass_has_error_diagnostics(context->pass)) return iree_ok_status();
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Rewriting
//===----------------------------------------------------------------------===//

static void loom_vector_bank_sroa_slot_indices(
    const loom_vector_bank_sroa_bank_t* bank, uint16_t slot,
    int64_t* out_indices) {
  uint32_t remaining = slot;
  for (uint8_t i = bank->prefix_rank; i > 0; --i) {
    uint8_t axis = (uint8_t)(i - 1);
    uint32_t extent =
        (uint32_t)loom_type_dim_static_size_at(bank->bank_type, axis);
    out_indices[axis] = remaining % extent;
    remaining /= extent;
  }
}

static iree_status_t loom_vector_bank_sroa_name_slot(loom_rewriter_t* rewriter,
                                                     loom_value_id_t source,
                                                     loom_value_id_t target,
                                                     uint16_t slot) {
  char suffix[32] = {0};
  int length = iree_snprintf(suffix, sizeof(suffix), "slot_%" PRIu16, slot);
  if (length <= 0 || (iree_host_size_t)length >= sizeof(suffix)) {
    return iree_ok_status();
  }
  return loom_rewriter_try_set_derived_value_name(
      rewriter, source, target,
      iree_make_string_view(suffix, (iree_host_size_t)length));
}

static iree_status_t loom_vector_bank_sroa_build_initial_values(
    loom_vector_bank_sroa_context_t* context, loom_op_t* loop,
    loom_vector_bank_sroa_plan_t* plan, loom_value_id_t* new_iter_args) {
  loom_value_slice_t old_iter_args = loom_scf_for_iter_args(loop);
  uint16_t expanded_ordinal = 0;
  for (uint16_t i = 0; i < plan->carried_count; ++i) {
    loom_vector_bank_sroa_bank_t* bank = &plan->banks[i];
    bank->expanded_base = expanded_ordinal;
    if (!bank->active) {
      new_iter_args[expanded_ordinal] = old_iter_args.values[i];
      ++expanded_ordinal;
      continue;
    }

    for (uint16_t slot = 0; slot < bank->slot_count; ++slot) {
      int64_t indices[LOOM_TYPE_MAX_RANK] = {0};
      loom_vector_bank_sroa_slot_indices(bank, slot, indices);
      loom_op_t* extract_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_extract_build(
          &context->rewriter->builder, old_iter_args.values[i],
          /*indices=*/NULL, /*indices_count=*/0, indices, bank->prefix_rank,
          bank->payload_type, loop->location, &extract_op));
      loom_value_id_t payload = loom_vector_extract_result(extract_op);
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_name_slot(
          context->rewriter, old_iter_args.values[i], payload, slot));
      new_iter_args[expanded_ordinal] = payload;
      ++expanded_ordinal;
    }
  }
  return iree_ok_status();
}

static loom_scf_for_build_flags_t loom_vector_bank_sroa_build_flags(
    loom_op_t* loop, loom_value_id_t* out_unroll_factor,
    loom_scf_for_unroll_policy_t* out_unroll_policy,
    loom_scf_for_unroll_schedule_t* out_unroll_schedule) {
  loom_scf_for_build_flags_t flags = 0;
  *out_unroll_factor = LOOM_VALUE_ID_INVALID;
  *out_unroll_policy = 0;
  *out_unroll_schedule = 0;
  if (loom_scf_for_unroll_factor_is_present(loop)) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    *out_unroll_factor = loom_scf_for_unroll_factor(loop);
  }
  if (!loom_attr_is_absent(
          loom_op_attrs(loop)[loom_scf_for_unroll_policy_ATTR_INDEX])) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    *out_unroll_policy = loom_scf_for_unroll_policy(loop);
  }
  if (!loom_attr_is_absent(
          loom_op_attrs(loop)[loom_scf_for_unroll_schedule_ATTR_INDEX])) {
    flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_SCHEDULE;
    *out_unroll_schedule = loom_scf_for_unroll_schedule(loop);
  }
  return flags;
}

static iree_status_t loom_vector_bank_sroa_clone_body(
    loom_vector_bank_sroa_context_t* context, loom_op_t* new_loop,
    const loom_vector_bank_sroa_plan_t* plan) {
  loom_block_t* new_block =
      loom_region_entry_block(loom_scf_for_body(new_loop));
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, context->scratch_arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      &remap));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_value(&remap, loom_block_arg_id(plan->body_block, 0),
                              loom_block_arg_id(new_block, 0)));
  IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
      context->rewriter, loom_block_arg_id(plan->body_block, 0),
      loom_block_arg_id(new_block, 0)));

  loom_value_id_t** state_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->scratch_arena, plan->value_snapshot_count, sizeof(*state_slots),
      (void**)&state_slots));
  memset(state_slots, 0, plan->value_snapshot_count * sizeof(*state_slots));

  for (uint16_t i = 0; i < plan->carried_count; ++i) {
    const loom_vector_bank_sroa_bank_t* bank = &plan->banks[i];
    loom_value_id_t old_arg =
        loom_block_arg_id(plan->body_block, (uint16_t)(1 + i));
    if (!bank->active) {
      loom_value_id_t new_arg =
          loom_block_arg_id(new_block, (uint16_t)(1 + bank->expanded_base));
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(&remap, old_arg, new_arg));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_copy_value_name(context->rewriter, old_arg, new_arg));
      continue;
    }

    loom_value_id_t* slots = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->scratch_arena, bank->slot_count,
                                  sizeof(*slots), (void**)&slots));
    for (uint16_t slot = 0; slot < bank->slot_count; ++slot) {
      slots[slot] = loom_block_arg_id(
          new_block, (uint16_t)(1 + bank->expanded_base + slot));
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_name_slot(
          context->rewriter, old_arg, slots[slot], slot));
    }
    state_slots[old_arg] = slots;
  }

  loom_op_t* child_op = NULL;
  loom_block_for_each_op(plan->body_block, child_op) {
    if (child_op == plan->yield) break;
    uint16_t bank_ordinal = UINT16_MAX;
    if (loom_vector_extract_isa(child_op) &&
        loom_vector_bank_sroa_state_bank(
            plan, loom_vector_extract_source(child_op), &bank_ordinal) &&
        plan->banks[bank_ordinal].active) {
      const loom_vector_bank_sroa_bank_t* bank = &plan->banks[bank_ordinal];
      uint16_t slot = 0;
      if (!loom_vector_bank_sroa_static_access_ordinal(
              bank, loom_vector_extract_static_indices(child_op),
              loom_vector_extract_indices(child_op),
              loom_module_value_type(context->module,
                                     loom_vector_extract_result(child_op)),
              &slot)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "validated vector bank extract changed");
      }
      loom_value_id_t* slots =
          state_slots[loom_vector_extract_source(child_op)];
      if (!slots) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "vector bank extract state is unavailable");
      }
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
          &remap, loom_vector_extract_result(child_op), slots[slot]));
      ++context->statistics->extracts_eliminated;
      continue;
    }
    if (loom_vector_insert_isa(child_op) &&
        loom_vector_bank_sroa_state_bank(
            plan, loom_vector_insert_dest(child_op), &bank_ordinal) &&
        plan->banks[bank_ordinal].active) {
      const loom_vector_bank_sroa_bank_t* bank = &plan->banks[bank_ordinal];
      uint16_t slot = 0;
      if (!loom_vector_bank_sroa_static_access_ordinal(
              bank, loom_vector_insert_static_indices(child_op),
              loom_vector_insert_indices(child_op),
              loom_module_value_type(context->module,
                                     loom_vector_insert_value(child_op)),
              &slot)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "validated vector bank insert changed");
      }
      loom_value_id_t inserted_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
          &remap, loom_vector_insert_value(child_op), &inserted_value));
      loom_value_id_t* old_slots =
          state_slots[loom_vector_insert_dest(child_op)];
      if (!old_slots) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "vector bank insert state is unavailable");
      }
      loom_value_id_t* new_slots = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(context->scratch_arena, bank->slot_count,
                                    sizeof(*new_slots), (void**)&new_slots));
      memcpy(new_slots, old_slots,
             (iree_host_size_t)bank->slot_count * sizeof(*new_slots));
      new_slots[slot] = inserted_value;
      state_slots[loom_vector_insert_result(child_op)] = new_slots;
      ++context->statistics->inserts_eliminated;
      continue;
    }

    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(loom_ir_clone_op(&context->rewriter->builder, child_op,
                                          &remap, &cloned_op));
    (void)cloned_op;
  }

  loom_value_slice_t old_yielded = loom_scf_yield_values(plan->yield);
  loom_value_id_t* new_yielded = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->scratch_arena, plan->expanded_count,
                                sizeof(*new_yielded), (void**)&new_yielded));
  for (uint16_t i = 0; i < plan->carried_count; ++i) {
    const loom_vector_bank_sroa_bank_t* bank = &plan->banks[i];
    if (!bank->active) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
          &remap, old_yielded.values[i], &new_yielded[bank->expanded_base]));
      continue;
    }
    loom_value_id_t yielded_state = old_yielded.values[i];
    loom_value_id_t* slots = state_slots[yielded_state];
    if (!slots) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "vector bank yielded state is unavailable");
    }
    memcpy(&new_yielded[bank->expanded_base], slots,
           (iree_host_size_t)bank->slot_count * sizeof(*slots));
  }
  loom_op_t* new_yield = NULL;
  return loom_scf_yield_build(&context->rewriter->builder, new_yielded,
                              plan->expanded_count, plan->yield->location,
                              &new_yield);
}

static iree_status_t loom_vector_bank_sroa_replace_results(
    loom_vector_bank_sroa_context_t* context, loom_op_t* old_loop,
    loom_op_t* new_loop, const loom_vector_bank_sroa_plan_t* plan) {
  const loom_value_id_t* old_results = loom_op_const_results(old_loop);
  loom_value_slice_t new_results = loom_scf_for_results(new_loop);
  for (uint16_t i = 0; i < plan->carried_count; ++i) {
    const loom_vector_bank_sroa_bank_t* bank = &plan->banks[i];
    if (!bank->active) {
      loom_value_id_t replacement = new_results.values[bank->expanded_base];
      IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
          context->rewriter, old_results[i], replacement));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
          context->rewriter, old_results[i], replacement));
      continue;
    }

    for (uint16_t slot = 0; slot < bank->slot_count; ++slot) {
      IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_name_slot(
          context->rewriter, old_results[i],
          new_results.values[bank->expanded_base + slot], slot));
    }
    loom_value_t* old_result =
        loom_module_value(context->module, old_results[i]);
    while (old_result->use_count != 0) {
      loom_use_t use = loom_value_uses(old_result)[0];
      loom_op_t* extract_op = loom_use_user_op(use);
      uint16_t slot = 0;
      if (!loom_vector_bank_sroa_static_access_ordinal(
              bank, loom_vector_extract_static_indices(extract_op),
              loom_vector_extract_indices(extract_op),
              loom_module_value_type(context->module,
                                     loom_vector_extract_result(extract_op)),
              &slot)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "validated loop result extract changed");
      }
      loom_value_id_t replacement =
          new_results.values[bank->expanded_base + slot];
      IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
          context->rewriter, loom_vector_extract_result(extract_op),
          replacement));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
          context->rewriter, extract_op, &replacement, 1));
      ++context->statistics->extracts_eliminated;
    }
  }
  return loom_rewriter_erase(context->rewriter, old_loop);
}

static iree_status_t loom_vector_bank_sroa_rewrite_loop(
    loom_vector_bank_sroa_context_t* context, loom_op_t* loop,
    loom_vector_bank_sroa_plan_t* plan) {
  if (plan->active_bank_count == 0) return iree_ok_status();

  loom_builder_set_before(&context->rewriter->builder, loop);
  loom_value_id_t* new_iter_args = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->scratch_arena, plan->expanded_count, sizeof(*new_iter_args),
      (void**)&new_iter_args));
  IREE_RETURN_IF_ERROR(loom_vector_bank_sroa_build_initial_values(
      context, loop, plan, new_iter_args));

  loom_value_id_t unroll_factor = LOOM_VALUE_ID_INVALID;
  loom_scf_for_unroll_policy_t unroll_policy = 0;
  loom_scf_for_unroll_schedule_t unroll_schedule = 0;
  loom_scf_for_build_flags_t build_flags = loom_vector_bank_sroa_build_flags(
      loop, &unroll_factor, &unroll_policy, &unroll_schedule);
  loom_op_t* new_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &context->rewriter->builder, build_flags, loom_scf_for_lower_bound(loop),
      loom_scf_for_upper_bound(loop), loom_scf_for_step(loop), new_iter_args,
      plan->expanded_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
      unroll_factor, unroll_policy, unroll_schedule, loop->location,
      &new_loop));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->rewriter->builder, new_loop, loom_scf_for_body(new_loop));
  iree_status_t status =
      loom_vector_bank_sroa_clone_body(context, new_loop, plan);
  loom_builder_restore(&context->rewriter->builder, saved_ip);
  if (!iree_status_is_ok(status)) return status;

  IREE_RETURN_IF_ERROR(
      loom_vector_bank_sroa_replace_results(context, loop, new_loop, plan));
  ++context->statistics->loops_scalarized;
  context->statistics->banks_scalarized += plan->active_bank_count;
  context->statistics->slots_materialized +=
      plan->expanded_count - (plan->carried_count - plan->active_bank_count);
  loom_pass_mark_changed(context->pass);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

#define LOOM_VECTOR_BANK_SROA_INITIAL_LOOP_CAPACITY 16

static iree_status_t loom_vector_bank_sroa_loop_list_initialize(
    iree_arena_allocator_t* arena, loom_vector_bank_sroa_loop_list_t* list) {
  *list = (loom_vector_bank_sroa_loop_list_t){
      .capacity = LOOM_VECTOR_BANK_SROA_INITIAL_LOOP_CAPACITY,
  };
  return iree_arena_allocate_array(arena, list->capacity, sizeof(*list->ops),
                                   (void**)&list->ops);
}

static iree_status_t loom_vector_bank_sroa_loop_list_push(
    iree_arena_allocator_t* arena, loom_vector_bank_sroa_loop_list_t* list,
    loom_op_t* op) {
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, list->count, list->count + 1, sizeof(*list->ops),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_vector_bank_sroa_collect_loop(
    void* user_data, loom_op_t* op, const loom_walk_context_t* walk_context,
    loom_walk_result_t* out_result) {
  (void)walk_context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_scf_for_isa(op)) return iree_ok_status();
  loom_vector_bank_sroa_collect_context_t* collect_context =
      (loom_vector_bank_sroa_collect_context_t*)user_data;
  return loom_vector_bank_sroa_loop_list_push(collect_context->arena,
                                              collect_context->loops, op);
}

static iree_status_t loom_vector_bank_sroa_collect_loops(
    loom_pass_t* pass, const loom_module_t* module, loom_func_like_t function,
    loom_vector_bank_sroa_loop_list_t* out_loops) {
  IREE_RETURN_IF_ERROR(
      loom_vector_bank_sroa_loop_list_initialize(pass->arena, out_loops));
  loom_vector_bank_sroa_collect_context_t collect_context = {
      .arena = pass->arena,
      .loops = out_loops,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_function(module, function, LOOM_WALK_POST_ORDER,
                            (loom_walk_callback_t){
                                .fn = loom_vector_bank_sroa_collect_loop,
                                .user_data = &collect_context,
                            },
                            pass->arena, &walk_result);
}

iree_status_t loom_vector_bank_sroa_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_vector_bank_sroa_loop_list_t loops = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_bank_sroa_collect_loops(pass, module, function, &loops));
  if (loops.count == 0) return iree_ok_status();

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  iree_arena_allocator_t scratch_arena = {0};
  iree_arena_initialize(pass->arena->block_pool, &scratch_arena);
  loom_vector_bank_sroa_context_t context = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
      .scratch_arena = &scratch_arena,
      .statistics = loom_vector_bank_sroa_statistics(pass),
  };

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < loops.count && iree_status_is_ok(status) &&
                               !loom_pass_has_error_diagnostics(pass);
       ++i) {
    loom_op_t* loop = loops.ops[i];
    if (loop->flags & LOOM_OP_FLAG_DEAD) continue;
    const iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(&scratch_arena);
    loom_vector_bank_sroa_plan_t plan = {0};
    status = loom_vector_bank_sroa_plan_loop(&context, loop, &plan);
    if (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass)) {
      status = loom_vector_bank_sroa_rewrite_loop(&context, loop, &plan);
    }
    iree_arena_checkpoint_restore(&checkpoint);
  }

  iree_arena_deinitialize(&scratch_arena);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
