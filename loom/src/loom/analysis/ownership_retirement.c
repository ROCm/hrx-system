// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/ownership_retirement.h"

#include <string.h>

#include "loom/analysis/ownership.h"
#include "loom/ops/op_defs.h"

static bool loom_ownership_retirement_selected(const uint64_t* words,
                                               loom_value_ordinal_t ordinal) {
  return (words[ordinal / 64] & (UINT64_C(1) << (ordinal % 64))) != 0;
}

static loom_value_ordinal_t loom_ownership_retirement_owner(
    const loom_local_value_domain_t* domain, const loom_value_ordinal_t* owners,
    loom_value_id_t value) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(domain, value);
  return owners ? owners[ordinal] : ordinal;
}

static iree_status_t loom_ownership_retirement_append(
    iree_arena_allocator_t* arena, loom_value_ordinal_t ordinal,
    bool transferred, uint32_t* head, iree_host_size_t* capacity,
    loom_ownership_retirement_plan_t* plan) {
  if (plan->entry_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "ownership retirement index exceeds 32 bits");
  }
  if (plan->entry_count == *capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, plan->entry_count, plan->entry_count + 1, sizeof(*plan->entries),
        capacity, (void**)&plan->entries));
  }
  plan->entries[plan->entry_count] =
      (loom_ownership_retirement_entry_t){ordinal, transferred, *head};
  *head = (uint32_t)plan->entry_count++;
  return iree_ok_status();
}

iree_status_t loom_ownership_retirement_build(
    const loom_local_value_domain_t* domain, const loom_value_ordinal_t* owners,
    const uint64_t* selected_words, const loom_liveness_analysis_t* liveness,
    iree_arena_allocator_t* arena, loom_ownership_retirement_plan_t* out_plan) {
  *out_plan = (loom_ownership_retirement_plan_t){0};
  bool any_selected = false;
  for (iree_host_size_t i = 0; i < (domain->value_count + 63u) / 64u; ++i) {
    any_selected |= selected_words[i] != 0;
  }
  if (!any_selected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->operation_count, sizeof(*out_plan->operations),
      (void**)&out_plan->operations));
  memset(out_plan->operations, 0xFF,
         liveness->operation_count * sizeof(*out_plan->operations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, liveness->block_count,
                                                 sizeof(*out_plan->blocks),
                                                 (void**)&out_plan->blocks));
  memset(out_plan->blocks, 0xFF,
         liveness->block_count * sizeof(*out_plan->blocks));
  uint32_t* live_epochs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain->value_count, sizeof(*live_epochs), (void**)&live_epochs));
  memset(live_epochs, 0, domain->value_count * sizeof(*live_epochs));
  iree_host_size_t capacity = 0;
  for (uint32_t b = 0; b < liveness->block_count; ++b) {
    const loom_liveness_block_info_t* block = &liveness->blocks[b];
    const uint32_t epoch = b + 1;
    for (iree_host_size_t i = 0; i < block->live_out_count; ++i) {
      live_epochs[loom_ownership_retirement_owner(
          domain, owners, block->live_out_values[i])] = epoch;
    }
    for (uint32_t end = block->operation_start + block->operation_count;
         end > block->operation_start; --end) {
      const uint32_t index = end - 1;
      const loom_liveness_operation_point_t* point =
          loom_liveness_operation_at(liveness, index);
      const loom_op_t* op = point->op;
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        const loom_value_ordinal_t ordinal =
            loom_local_value_domain_ordinal(domain, results[i]);
        if ((owners && owners[ordinal] != ordinal) ||
            !loom_ownership_retirement_selected(selected_words, ordinal)) {
          continue;
        }
        if (live_epochs[ordinal] != epoch) {
          IREE_RETURN_IF_ERROR(loom_ownership_retirement_append(
              arena, ordinal, false, &out_plan->operations[index], &capacity,
              out_plan));
        }
        live_epochs[ordinal] = 0;
        loom_ownership_result_effect_t effect = {0};
        if (loom_ownership_result_effect_at(domain->module, NULL, op, i,
                                            &effect) &&
            effect.effect == LOOM_RESULT_OWNERSHIP_COPIED) {
          const loom_value_ordinal_t source = loom_ownership_retirement_owner(
              domain, owners,
              loom_op_const_operands(op)[effect.source_operand_index]);
          // Structural copy transport already moves a source's final-use
          // protection. A synthetic release would extend its liveness and
          // force an unnecessary retain followed by that release.
          if (loom_ownership_retirement_selected(selected_words, source) &&
              live_epochs[source] != epoch) {
            IREE_RETURN_IF_ERROR(loom_ownership_retirement_append(
                arena, source, true, &out_plan->operations[index], &capacity,
                out_plan));
          }
          live_epochs[source] = epoch;
        }
      }
      const loom_call_like_t call =
          loom_call_like_const_cast(domain->module, op);
      const bool transferred =
          loom_call_like_isa(call) &&
          loom_call_like_kind_is_runtime(loom_call_like_kind(call));
      for (uint32_t i = 0; i < point->direct_use_count; ++i) {
        loom_value_ordinal_t ordinal =
            loom_liveness_operation_use_ordinal(liveness, point->use_start + i);
        if (owners) {
          ordinal = owners[ordinal];
        }
        if (!loom_ownership_retirement_selected(selected_words, ordinal)) {
          continue;
        }
        if (live_epochs[ordinal] != epoch &&
            !iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR)) {
          IREE_RETURN_IF_ERROR(loom_ownership_retirement_append(
              arena, ordinal, transferred, &out_plan->operations[index],
              &capacity, out_plan));
        }
        live_epochs[ordinal] = epoch;
      }
    }
    for (uint16_t i = 0; i < block->block->arg_count; ++i) {
      const loom_value_ordinal_t ordinal = loom_ownership_retirement_owner(
          domain, owners, loom_block_arg_id(block->block, i));
      if (loom_ownership_retirement_selected(selected_words, ordinal) &&
          live_epochs[ordinal] != epoch) {
        IREE_RETURN_IF_ERROR(loom_ownership_retirement_append(
            arena, ordinal, false, &out_plan->blocks[b], &capacity, out_plan));
      }
    }
  }
  return iree_ok_status();
}
