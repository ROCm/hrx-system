// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/cse.h"

#include <string.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/function_version.h"
#include "loom/transforms/cleanup/expression_scope.h"

#define LOOM_LOW_CSE_STATISTICS(V, statistics_type)                    \
  V(statistics_type, expressions_eliminated, "expressions-eliminated", \
    "Number of redundant machine expressions removed.")

LOOM_PASS_STATISTICS_DEFINE(loom_low_cse_statistics, loom_low_cse_statistics_t,
                            LOOM_LOW_CSE_STATISTICS)

static const loom_pass_info_t loom_low_cse_pass_info_storage = {
    .name = IREE_SVL("low-cse"),
    .description = IREE_SVL("Eliminate redundant machine expressions."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_low_cse_statistics_layout,
};

const loom_pass_info_t* loom_low_cse_pass_info(void) {
  return &loom_low_cse_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Architectural state identity
//===----------------------------------------------------------------------===//

enum loom_low_cse_packet_flag_bits_e {
  // Whether the packet depends on architectural state not explicit in SSA.
  LOOM_LOW_CSE_PACKET_FLAG_IMPLICIT_STATE = 1u << 0,
  // Whether the packet writes architectural state.
  LOOM_LOW_CSE_PACKET_FLAG_WRITES_STATE = 1u << 1,
};
typedef uint8_t loom_low_cse_packet_flags_t;

typedef struct loom_low_cse_packet_identity_t {
  // Most recent implicit-state write that prevents reuse of an older producer.
  uint32_t minimum_epoch;
  // Architectural state read and write classification.
  loom_low_cse_packet_flags_t flags;
  // Retained identical assignment, or NULL when this packet must remain.
  loom_op_t* assignment;
} loom_low_cse_packet_identity_t;

typedef struct loom_low_cse_state_class_t {
  // Most recent retained write's value-definition frontier.
  uint32_t write_epoch;
  // Most recent reusable assignment to this class, or NULL after another write.
  loom_op_t* assignment;
  // Execution-boundary identity at the retained assignment.
  const loom_op_t* boundary;
} loom_low_cse_state_class_t;

// Each class retains its latest write and assignment in the existing walk.
// Writes to unrelated classes neither scan nor invalidate other assignments.
static loom_low_cse_packet_identity_t loom_low_cse_observe_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_expression_cursor_t* cursor,
    const loom_low_descriptor_packet_t* packet, bool is_assignment,
    const loom_op_t* boundary, loom_low_cse_state_class_t* state_classes) {
  loom_low_cse_packet_identity_t identity = {0};
  if (packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return identity;
  }

  // Descriptor semantics permit removing only the repeated state effect. SSA
  // results still obey the shared identity and ownership eligibility rules.
  const bool can_reuse_assignment =
      is_assignment &&
      !iree_any_bit_set(cursor->traits, LOOM_TRAIT_OBSERVABLE_EFFECT |
                                            LOOM_TRAIT_UNIQUE_IDENTITY |
                                            LOOM_TRAIT_CONVERGENT) &&
      loom_expression_can_share_result_ownership(cursor->module, cursor->op);
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (iree_any_bit_set(operand->flags,
                         LOOM_LOW_OPERAND_FLAG_SCHEDULE_ONLY_STATE)) {
      continue;
    }
    const loom_low_operand_flags_t state_flags =
        operand->flags &
        (LOOM_LOW_OPERAND_FLAG_STATE_READ | LOOM_LOW_OPERAND_FLAG_STATE_WRITE);
    if (!state_flags) {
      continue;
    }
    // Python validates exactly one register class for architectural state.
    const uint16_t class_id =
        descriptor_set->reg_class_alts[operand->reg_class_alt_start]
            .reg_class_id;
    loom_low_cse_state_class_t* state_class = &state_classes[class_id];
    if (iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      if (can_reuse_assignment && state_class->assignment &&
          state_class->boundary == boundary &&
          loom_expression_equal(cursor->module, state_class->assignment,
                                cursor->op)) {
        identity.assignment = state_class->assignment;
      } else {
        state_class->write_epoch = cursor->epoch;
        state_class->assignment = can_reuse_assignment ? cursor->op : NULL;
        state_class->boundary = boundary;
      }
      identity.flags |= LOOM_LOW_CSE_PACKET_FLAG_WRITES_STATE;
    }
    const bool has_explicit_packet_value =
        i >= descriptor->result_count &&
        loom_low_operand_role_is_packet_operand(operand->role);
    if (iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_READ) &&
        !has_explicit_packet_value) {
      identity.flags |= LOOM_LOW_CSE_PACKET_FLAG_IMPLICIT_STATE;
      identity.minimum_epoch =
          iree_max(identity.minimum_epoch, state_class->write_epoch);
    }
  }

  return identity;
}

static iree_status_t loom_low_cse_region(
    loom_pass_t* pass, loom_module_t* module, loom_region_t* region,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena) {
  loom_expression_walk_t* walk = NULL;
  IREE_RETURN_IF_ERROR(
      loom_expression_walk_create(module, region, arena, &walk));
  loom_low_cse_state_class_t* state_classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptor_set->reg_class_count, sizeof(*state_classes),
      (void**)&state_classes));
  memset(state_classes, 0,
         descriptor_set->reg_class_count * sizeof(*state_classes));
  loom_low_cse_statistics_t* statistics = loom_low_cse_statistics(pass);
  loom_expression_barriers_t barriers = {0};
  loom_block_t* assignment_block = NULL;
  const loom_op_t* assignment_boundary = NULL;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    loom_expression_cursor_t cursor;
    status = loom_expression_walk_next(walk, &cursor);
    if (!iree_status_is_ok(status) || !cursor.op) {
      break;
    }
    loom_low_descriptor_packet_t packet = {0};
    loom_low_descriptor_packet_initialize(descriptor_set, cursor.op, &packet);
    const bool is_assignment =
        packet.descriptor &&
        iree_any_bit_set(packet.descriptor->flags,
                         LOOM_LOW_DESCRIPTOR_FLAG_STATE_ASSIGNMENT);
    // Assignment descriptors refine their conservative UNKNOWN_EFFECTS trait.
    // A boundary identity, unlike the value epoch, distinguishes zero-result
    // operations and invalidates every class without clearing the class table.
    if (cursor.op->parent_block != assignment_block ||
        cursor.op->region_count != 0 ||
        iree_any_bit_set(cursor.traits,
                         LOOM_TRAIT_CALLABLE_BOUNDARY | LOOM_TRAIT_TERMINATOR |
                             LOOM_TRAIT_CONVERGENT | LOOM_TRAIT_MEMORY_FENCE |
                             LOOM_TRAIT_OBSERVABLE_EFFECT) ||
        (!is_assignment &&
         iree_any_bit_set(cursor.traits, LOOM_TRAIT_UNKNOWN_EFFECTS)) ||
        (packet.descriptor &&
         iree_any_bit_set(packet.descriptor->flags,
                          LOOM_LOW_DESCRIPTOR_FLAG_BARRIER)) ||
        cursor.op->kind == LOOM_OP_LOW_SCHEDULE_FENCE) {
      assignment_boundary = cursor.op;
    }
    assignment_block = cursor.op->parent_block;
    const loom_low_cse_packet_identity_t identity = loom_low_cse_observe_packet(
        descriptor_set, &cursor, &packet, is_assignment, assignment_boundary,
        state_classes);
    if (identity.assignment) {
      status = loom_expression_replace(module, cursor.op, identity.assignment);
      if (iree_status_is_ok(status)) {
        loom_pass_mark_changed(pass);
        ++statistics->expressions_eliminated;
      }
      continue;
    }
    const uint32_t minimum_epoch =
        iree_max(identity.minimum_epoch,
                 loom_expression_observe_barriers(&cursor, &barriers));
    if (iree_any_bit_set(identity.flags,
                         LOOM_LOW_CSE_PACKET_FLAG_WRITES_STATE) ||
        !loom_expression_is_reusable(&cursor)) {
      continue;
    }

    loom_expression_entry_t candidate = {
        .op = cursor.op,
        .hash = loom_expression_hash(module, cursor.op),
        .epoch = cursor.epoch,
    };
    loom_expression_lookup_flags_t flags = 0;
    if (iree_any_bit_set(identity.flags,
                         LOOM_LOW_CSE_PACKET_FLAG_IMPLICIT_STATE) ||
        !iree_any_bit_set(cursor.traits, LOOM_TRAIT_PURE)) {
      flags = LOOM_EXPRESSION_LOOKUP_FLAG_STATEFUL;
    }
    loom_expression_entry_t* existing =
        loom_expression_scope_find(&cursor, candidate.hash, flags);
    if (existing && existing->epoch >= minimum_epoch) {
      status = loom_expression_replace(module, cursor.op, existing->op);
      if (iree_status_is_ok(status)) {
        loom_pass_mark_changed(pass);
        ++statistics->expressions_eliminated;
      }
    } else {
      loom_expression_scope_insert(&cursor, candidate);
    }
  }
  return status;
}

iree_status_t loom_low_cse_run(loom_pass_t* pass, loom_module_t* module,
                               loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op) ||
      !loom_func_like_body(function)) {
    return iree_ok_status();
  }
  const loom_low_pass_capability_t* capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* registry =
      loom_low_pass_capability_descriptor_registry(capability);
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, pass->arena);
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &symbol_facts, function.op,
      loom_target_function_version_target_facts(pass->function_version),
      registry, pass->diagnostic_emitter, &target));
  if (!target.descriptor_set) {
    return iree_ok_status();
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(pass->arena->block_pool, &arena);
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0;
       i < loom_func_like_region_count(function) && iree_status_is_ok(status);
       ++i) {
    loom_region_t* region = loom_func_like_region(function, i);
    if (!region) {
      continue;
    }
    iree_arena_reset(&arena);
    status = loom_low_cse_region(pass, module, region, target.descriptor_set,
                                 &arena);
  }
  iree_arena_deinitialize(&arena);
  return status;
}
