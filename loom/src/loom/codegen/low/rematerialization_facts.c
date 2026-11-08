// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/rematerialization_facts.h"

#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

static bool loom_low_descriptor_packet_kind_may_rematerialize(
    loom_low_descriptor_packet_kind_t kind) {
  return kind == LOOM_LOW_DESCRIPTOR_PACKET_OP ||
         kind == LOOM_LOW_DESCRIPTOR_PACKET_CONST;
}

static bool loom_low_rematerialization_packet_is_eligible(
    loom_module_t* module, const loom_low_resolved_target_t* target,
    loom_value_id_t value_id, loom_op_t** out_defining_op) {
  *out_defining_op = NULL;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value) || loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(module, value_id)) {
    return false;
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  const loom_low_register_type_resolver_t register_type_resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          target->descriptor_set);
  if (loom_low_register_type_resolver_has_class_flags(
          &register_type_resolver, value_type,
          LOOM_LOW_REG_CLASS_FLAG_REFERENCE)) {
    return false;
  }

  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
      defining_op->region_count != 0 || defining_op->successor_count != 0 ||
      iree_any_bit_set(loom_op_effective_traits(module, defining_op),
                       LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return false;
  }

  const uint16_t result_index = loom_value_def_index(value);
  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(target->descriptor_set, defining_op,
                                        &packet);
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    // Structural composition is repeatable when its inputs can be recreated
    // privately. Recipe planning establishes that ownership boundary.
    if (!iree_any_bit_set(loom_op_vtable(module, defining_op)->vtable_flags,
                          LOOM_OP_VTABLE_STRUCTURAL_MATERIALIZATION)) {
      return false;
    }
  } else if (!loom_low_descriptor_packet_kind_may_rematerialize(packet.kind) ||
             !loom_low_descriptor_result_can_rematerialize(
                 target->descriptor_set, packet.descriptor, result_index)) {
    return false;
  }
  *out_defining_op = defining_op;
  return true;
}

typedef struct loom_low_rematerialization_closure_cursor_t {
  // Local value whose structural inputs are being visited.
  loom_value_ordinal_t ordinal;
  // Next operand to visit before completing this producer.
  uint16_t next_operand;
  // Whether every input visited so far belongs to a closed materialization.
  bool is_closed;
} loom_low_rematerialization_closure_cursor_t;

static loom_low_rematerialization_closure_cursor_t
loom_low_rematerialization_closure_cursor_initialize(
    loom_value_ordinal_t ordinal) {
  return (loom_low_rematerialization_closure_cursor_t){
      .ordinal = ordinal,
      .is_closed = true,
  };
}

static iree_status_t loom_low_rematerialization_classify(
    const loom_low_rematerialization_scope_t* scope, bool reconstruct_constants,
    uint8_t* value_flags, iree_arena_allocator_t* arena) {
  loom_module_t* module = scope->module;
  for (loom_value_ordinal_t ordinal = 0; ordinal < scope->value_count;
       ++ordinal) {
    const loom_value_id_t value_id = scope->value_ids[ordinal];
    uint8_t flags =
        loom_value_consuming_use_count(loom_module_value(module, value_id)) != 0
            ? LOOM_LOW_REMATERIALIZATION_VALUE_CONSUMED
            : 0;
    loom_op_t* op = NULL;
    bool eligible = loom_low_rematerialization_packet_is_eligible(
        module, scope->target, value_id, &op);
    if (eligible && scope->liveness != NULL) {
      const uint32_t interval_index =
          scope->liveness->value_interval_indices[ordinal];
      eligible = interval_index != UINT32_MAX &&
                 scope->liveness->intervals[interval_index].definition_point !=
                     UINT32_MAX;
    } else if (eligible) {
      eligible = op->parent_block->parent_region == scope->region;
    }
    if (!eligible) {
      flags |= LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED;
    } else {
      flags |= LOOM_LOW_REMATERIALIZATION_VALUE_ELIGIBLE;
      if (op->operand_count == 0) {
        flags |= LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED |
                 LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED;
      } else if (!iree_any_bit_set(loom_op_vtable(module, op)->vtable_flags,
                                   LOOM_OP_VTABLE_STRUCTURAL_MATERIALIZATION)) {
        // Late reconstruction initializes constants and composes their storage.
        // Replaying arithmetic retains the ordinary execution-domain contract.
        flags |= LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED;
      }
    }
    value_flags[ordinal] = flags;
  }

  if (!reconstruct_constants) {
    return iree_ok_status();
  }

  // Memoized postorder handles inlined CFG layouts whose definition list is
  // not dominance ordered. Each eligible producer and edge is visited once.
  loom_low_rematerialization_closure_cursor_t* stack = NULL;
  iree_host_size_t capacity = 0;
  for (loom_value_ordinal_t root = 0; root < scope->value_count; ++root) {
    if (iree_any_bit_set(value_flags[root],
                         LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, 0, 1, sizeof(*stack),
                                               &capacity, (void**)&stack));
    iree_host_size_t depth = 1;
    stack[0] = loom_low_rematerialization_closure_cursor_initialize(root);
    while (depth != 0) {
      loom_low_rematerialization_closure_cursor_t* cursor = &stack[depth - 1];
      const loom_op_t* op = loom_value_def_op(
          loom_module_value(module, scope->value_ids[cursor->ordinal]));
      if (!cursor->is_closed || cursor->next_operand == op->operand_count) {
        value_flags[cursor->ordinal] |=
            LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED |
            (cursor->is_closed ? LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED : 0);
        --depth;
        continue;
      }
      const loom_value_id_t input =
          loom_op_const_operands(op)[cursor->next_operand];
      const loom_value_ordinal_t input_ordinal =
          loom_module_value_ordinal_scratch_lookup(module, input);
      const uint8_t input_flags = value_flags[input_ordinal];
      if (loom_type_kind(loom_module_value_type(module, input)) !=
          LOOM_TYPE_REGISTER) {
        cursor->is_closed = !iree_any_bit_set(
            input_flags, LOOM_LOW_REMATERIALIZATION_VALUE_CONSUMED);
      } else if (loom_low_register_type_class_id(
                     loom_module_value_type(module, input)) !=
                 loom_low_register_type_class_id(loom_module_value_type(
                     module, scope->value_ids[cursor->ordinal]))) {
        cursor->is_closed = false;
      } else if (loom_low_register_type_unit_count(
                     loom_module_value_type(module, input)) >
                 loom_low_register_type_unit_count(loom_module_value_type(
                     module, scope->value_ids[cursor->ordinal]))) {
        // A narrow projection must not reconstruct its wider input. Retain
        // that input as a capture instead of multiplying its materialization.
        cursor->is_closed = false;
      } else if (!iree_any_bit_set(
                     input_flags,
                     LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED)) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, depth, depth + 1,
                                                   sizeof(*stack), &capacity,
                                                   (void**)&stack));
        stack[depth++] =
            loom_low_rematerialization_closure_cursor_initialize(input_ordinal);
        continue;
      } else {
        cursor->is_closed = iree_any_bit_set(
            input_flags, LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED);
      }
      ++cursor->next_operand;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_rematerialization_facts_build(
    const loom_low_rematerialization_scope_t* scope, bool reconstruct_constants,
    iree_arena_allocator_t* arena, uint8_t** out_value_flags) {
  *out_value_flags = NULL;
  uint8_t* value_flags = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, scope->value_count, sizeof(*value_flags), (void**)&value_flags));
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  iree_status_t status = loom_low_rematerialization_classify(
      scope, reconstruct_constants, value_flags, arena);
  iree_arena_checkpoint_restore(&checkpoint);
  if (iree_status_is_ok(status)) {
    *out_value_flags = value_flags;
  }
  return status;
}
