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
// Semantic value identity
//===----------------------------------------------------------------------===//

// Low structural operations create distinct virtual-register identities even
// when they carry the same bits. The operations themselves must remain, but a
// descriptor with no storage-sensitive ownership can use the indexed content
// identity when proving that its computation is redundant.
typedef struct loom_low_cse_value_identity_slot_t {
  // SSA value whose content has a distinct canonical representative.
  loom_value_id_t value_id;
  // Canonical content representative.
  loom_value_id_t identity;
} loom_low_cse_value_identity_slot_t;

typedef struct loom_low_cse_constructor_slot_t {
  // Retained structural constructor naming this content class.
  const loom_op_t* op;
  // Content-aware structural hash of |op|.
  uint32_t hash;
} loom_low_cse_constructor_slot_t;

typedef struct loom_low_cse_value_identities_t {
  // Descriptor set used to identify the narrowly licensed expressions.
  const loom_low_descriptor_set_t* descriptor_set;
  // Scratch arena owning every table generation.
  iree_arena_allocator_t* arena;
  // Open-addressed aliases; values absent from the table identify themselves.
  loom_low_cse_value_identity_slot_t* value_slots;
  // Number of populated aliases.
  iree_host_size_t value_count;
  // Power-of-two slot capacity, kept at most half full.
  iree_host_size_t value_capacity;
  // Open-addressed exact slice/concat constructors keyed by carried content.
  loom_low_cse_constructor_slot_t* constructor_slots;
  // Number of populated constructor slots.
  iree_host_size_t constructor_count;
  // Power-of-two constructor capacity, kept at most half full.
  iree_host_size_t constructor_capacity;
} loom_low_cse_value_identities_t;

static iree_host_size_t loom_low_cse_value_identity_slot_index(
    const loom_low_cse_value_identities_t* identities,
    loom_value_id_t value_id) {
  iree_host_size_t slot = ((iree_host_size_t)value_id * 2654435761u) &
                          (identities->value_capacity - 1);
  while (identities->value_slots[slot].value_id != LOOM_VALUE_ID_INVALID &&
         identities->value_slots[slot].value_id != value_id) {
    slot = (slot + 1) & (identities->value_capacity - 1);
  }
  return slot;
}

static loom_value_id_t loom_low_cse_value_identity(void* user_data,
                                                   loom_value_id_t value_id) {
  const loom_low_cse_value_identities_t* identities =
      (const loom_low_cse_value_identities_t*)user_data;
  if (identities->value_capacity == 0) {
    return value_id;
  }
  const loom_low_cse_value_identity_slot_t* slot =
      &identities->value_slots[loom_low_cse_value_identity_slot_index(
          identities, value_id)];
  return slot->value_id == value_id ? slot->identity : value_id;
}

static iree_status_t loom_low_cse_value_identities_reserve(
    loom_low_cse_value_identities_t* identities,
    iree_host_size_t additional_count) {
  iree_host_size_t required_count = 0;
  if (!iree_host_size_checked_add(identities->value_count, additional_count,
                                  &required_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Low CSE value identity capacity overflow");
  }
  if (required_count <= identities->value_capacity / 2) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      identities->value_capacity ? identities->value_capacity : 16;
  while (required_count > new_capacity / 2) {
    if (!iree_host_size_checked_mul(new_capacity, 2, &new_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Low CSE value identity capacity overflow");
    }
  }
  loom_low_cse_value_identity_slot_t* new_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      identities->arena, new_capacity, sizeof(*new_slots), (void**)&new_slots));
  memset(new_slots, 0xFF, new_capacity * sizeof(*new_slots));

  loom_low_cse_value_identity_slot_t* old_slots = identities->value_slots;
  const iree_host_size_t old_capacity = identities->value_capacity;
  identities->value_slots = new_slots;
  identities->value_capacity = new_capacity;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (old_slots[i].value_id == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const iree_host_size_t slot = loom_low_cse_value_identity_slot_index(
        identities, old_slots[i].value_id);
    new_slots[slot] = old_slots[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_low_cse_record_value_identity(
    loom_low_cse_value_identities_t* identities, loom_value_id_t value_id,
    loom_value_id_t identity) {
  identity = loom_low_cse_value_identity(identities, identity);
  if (value_id == identity) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_cse_value_identities_reserve(identities, 1));
  loom_low_cse_value_identity_slot_t* slot =
      &identities->value_slots[loom_low_cse_value_identity_slot_index(
          identities, value_id)];
  if (slot->value_id == LOOM_VALUE_ID_INVALID) {
    *slot = (loom_low_cse_value_identity_slot_t){
        .value_id = value_id,
        .identity = identity,
    };
    ++identities->value_count;
  } else {
    IREE_ASSERT_EQ(slot->identity, identity,
                   "Low value content identity must be assigned once");
  }
  return iree_ok_status();
}

static bool loom_low_cse_op_is_idempotent_state_update(
    const loom_low_cse_value_identities_t* identities, const loom_op_t* op) {
  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(identities->descriptor_set, op,
                                        &packet);
  return packet.descriptor &&
         iree_any_bit_set(packet.descriptor->flags,
                          LOOM_LOW_DESCRIPTOR_FLAG_IDEMPOTENT_STATE_UPDATE);
}

static bool loom_low_cse_expression_equal(void* user_data,
                                          const loom_module_t* module,
                                          const loom_op_t* lhs,
                                          const loom_op_t* rhs) {
  if (loom_expression_equal(module, lhs, rhs)) {
    return true;
  }
  const loom_low_cse_value_identities_t* identities =
      (const loom_low_cse_value_identities_t*)user_data;
  if (!loom_low_cse_op_is_idempotent_state_update(identities, lhs) ||
      !loom_low_cse_op_is_idempotent_state_update(identities, rhs)) {
    return false;
  }
  return loom_expression_equal_with_value_identity(
      module, lhs, rhs, loom_low_cse_value_identity, user_data);
}

static uint32_t loom_low_cse_expression_hash(
    const loom_module_t* module, const loom_op_t* op,
    loom_low_cse_value_identities_t* identities,
    bool is_idempotent_state_update) {
  if (!is_idempotent_state_update) {
    return loom_expression_hash(module, op);
  }
  return loom_expression_hash_with_value_identity(
      module, op, loom_low_cse_value_identity, identities);
}

static iree_status_t loom_low_cse_constructor_slots_reserve(
    loom_low_cse_value_identities_t* identities,
    iree_host_size_t additional_count) {
  iree_host_size_t required_count = 0;
  if (!iree_host_size_checked_add(identities->constructor_count,
                                  additional_count, &required_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Low CSE constructor capacity overflow");
  }
  if (required_count <= identities->constructor_capacity / 2) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      identities->constructor_capacity ? identities->constructor_capacity : 16;
  while (required_count > new_capacity / 2) {
    if (!iree_host_size_checked_mul(new_capacity, 2, &new_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Low CSE constructor capacity overflow");
    }
  }
  loom_low_cse_constructor_slot_t* new_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      identities->arena, new_capacity, sizeof(*new_slots), (void**)&new_slots));
  memset(new_slots, 0, new_capacity * sizeof(*new_slots));

  loom_low_cse_constructor_slot_t* old_slots = identities->constructor_slots;
  const iree_host_size_t old_capacity = identities->constructor_capacity;
  identities->constructor_slots = new_slots;
  identities->constructor_capacity = new_capacity;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (!old_slots[i].op) {
      continue;
    }
    iree_host_size_t slot =
        old_slots[i].hash & (identities->constructor_capacity - 1);
    while (new_slots[slot].op) {
      slot = (slot + 1) & (identities->constructor_capacity - 1);
    }
    new_slots[slot] = old_slots[i];
  }
  return iree_ok_status();
}

static const loom_op_t* loom_low_cse_find_constructor(
    const loom_module_t* module,
    const loom_low_cse_value_identities_t* identities, const loom_op_t* op,
    uint32_t hash) {
  if (identities->constructor_capacity == 0) {
    return NULL;
  }
  iree_host_size_t slot = hash & (identities->constructor_capacity - 1);
  while (identities->constructor_slots[slot].op) {
    const loom_low_cse_constructor_slot_t* constructor =
        &identities->constructor_slots[slot];
    if (constructor->hash == hash &&
        loom_expression_equal_with_value_identity(module, constructor->op, op,
                                                  loom_low_cse_value_identity,
                                                  (void*)identities)) {
      return constructor->op;
    }
    slot = (slot + 1) & (identities->constructor_capacity - 1);
  }
  return NULL;
}

// Records content carried by Low's structural virtual-register constructors.
// Copies, moves, and assumptions are direct aliases. A separate pass-local
// interner recognizes identical slice and concat constructors without placing
// those storage identities in the ordinary CSE availability index.
static iree_status_t loom_low_cse_observe_structural_identity(
    const loom_module_t* module, loom_op_t* op,
    loom_low_cse_value_identities_t* identities) {
  if (loom_low_copy_isa(op)) {
    return loom_low_cse_record_value_identity(
        identities, loom_low_copy_result(op), loom_low_copy_source(op));
  }
  if (loom_low_move_isa(op)) {
    return loom_low_cse_record_value_identity(
        identities, loom_low_move_result(op), loom_low_move_source(op));
  }
  if (loom_low_assume_isa(op)) {
    const loom_value_slice_t values = loom_low_assume_values(op);
    const loom_value_slice_t results = loom_low_assume_results(op);
    IREE_ASSERT_EQ(values.count, results.count,
                   "verified low.assume must preserve value arity");
    for (uint16_t i = 0; i < results.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_cse_record_value_identity(
          identities, results.values[i], values.values[i]));
    }
    return iree_ok_status();
  }
  if (!loom_low_slice_isa(op) && !loom_low_concat_isa(op)) {
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(op->result_count, 1, "Low slice/concat must have one result");
  const uint32_t hash = loom_expression_hash_with_value_identity(
      module, op, loom_low_cse_value_identity, identities);
  const loom_op_t* existing =
      loom_low_cse_find_constructor(module, identities, op, hash);
  if (existing) {
    return loom_low_cse_record_value_identity(
        identities, loom_op_results(op)[0], loom_op_const_results(existing)[0]);
  }
  IREE_RETURN_IF_ERROR(loom_low_cse_constructor_slots_reserve(identities, 1));
  iree_host_size_t slot = hash & (identities->constructor_capacity - 1);
  while (identities->constructor_slots[slot].op) {
    slot = (slot + 1) & (identities->constructor_capacity - 1);
  }
  identities->constructor_slots[slot] = (loom_low_cse_constructor_slot_t){
      .op = op,
      .hash = hash,
  };
  ++identities->constructor_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Architectural state identity
//===----------------------------------------------------------------------===//

enum loom_low_cse_packet_flag_bits_e {
  // Whether the packet depends on architectural state not explicit in SSA.
  LOOM_LOW_CSE_PACKET_FLAG_IMPLICIT_STATE = 1u << 0,
  // Whether the packet writes architectural state.
  LOOM_LOW_CSE_PACKET_FLAG_WRITES_STATE = 1u << 1,
  // Whether every write is a result-coupled idempotent state update.
  LOOM_LOW_CSE_PACKET_FLAG_IDEMPOTENT_UPDATE = 1u << 2,
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
  // Most recent retained write that invalidates idempotent-update reuse.
  uint32_t invalidating_write_epoch;
  // Most recent reusable assignment to this class, or NULL after another write.
  loom_op_t* assignment;
  // Execution-boundary identity at the retained assignment.
  const loom_op_t* boundary;
} loom_low_cse_state_class_t;

static bool loom_low_cse_can_reuse_assignment(
    const loom_expression_cursor_t* cursor, bool is_assignment) {
  return is_assignment &&
         !iree_any_bit_set(cursor->traits, LOOM_TRAIT_UNIQUE_IDENTITY |
                                               LOOM_TRAIT_CONVERGENT) &&
         loom_expression_can_share_result_ownership(cursor->module, cursor->op);
}

// Analyzes state dependencies without committing this packet's writes. A
// packet proven redundant must leave the state frontier unchanged.
static loom_low_cse_packet_identity_t loom_low_cse_analyze_packet(
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
      loom_low_cse_can_reuse_assignment(cursor, is_assignment);
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const bool is_idempotent_update = iree_any_bit_set(
      descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_IDEMPOTENT_STATE_UPDATE);
  if (is_idempotent_update) {
    identity.flags |= LOOM_LOW_CSE_PACKET_FLAG_IDEMPOTENT_UPDATE;
  }
  bool assignment_matches = can_reuse_assignment;
  loom_op_t* assignment = NULL;
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
      if (is_idempotent_update) {
        identity.minimum_epoch = iree_max(
            identity.minimum_epoch, state_class->invalidating_write_epoch);
      }
      if (can_reuse_assignment) {
        if (!state_class->assignment || state_class->boundary != boundary ||
            (assignment && assignment != state_class->assignment)) {
          assignment_matches = false;
        } else {
          assignment = state_class->assignment;
        }
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

  if (assignment_matches && assignment &&
      loom_expression_equal(cursor->module, assignment, cursor->op)) {
    identity.assignment = assignment;
  }

  return identity;
}

// Commits the state effects of one retained packet. All writes advance hidden
// read epochs. Only replacement and non-idempotent writes invalidate a prior
// idempotent update; every non-assignment write invalidates exact assignments.
static void loom_low_cse_commit_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_expression_cursor_t* cursor,
    const loom_low_descriptor_packet_t* packet, bool is_assignment,
    const loom_op_t* boundary, loom_low_cse_state_class_t* state_classes) {
  if (packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return;
  }
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const bool is_idempotent_update = iree_any_bit_set(
      descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_IDEMPOTENT_STATE_UPDATE);
  const bool can_reuse_assignment =
      loom_low_cse_can_reuse_assignment(cursor, is_assignment);
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (!iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE) ||
        iree_any_bit_set(operand->flags,
                         LOOM_LOW_OPERAND_FLAG_SCHEDULE_ONLY_STATE)) {
      continue;
    }
    const uint16_t class_id =
        descriptor_set->reg_class_alts[operand->reg_class_alt_start]
            .reg_class_id;
    loom_low_cse_state_class_t* state_class = &state_classes[class_id];
    state_class->write_epoch = cursor->epoch;
    if (!is_idempotent_update) {
      state_class->invalidating_write_epoch = cursor->epoch;
    }
    state_class->assignment = can_reuse_assignment ? cursor->op : NULL;
    state_class->boundary = can_reuse_assignment ? boundary : NULL;
  }
}

static iree_status_t loom_low_cse_region(
    loom_pass_t* pass, loom_module_t* module, loom_region_t* region,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena) {
  loom_low_cse_value_identities_t value_identities = {
      .descriptor_set = descriptor_set,
      .arena = arena,
  };
  const loom_expression_identity_policy_t identity_policy = {
      .user_data = &value_identities,
      .equal = loom_low_cse_expression_equal,
  };
  const bool has_idempotent_state_updates =
      iree_any_bit_set(descriptor_set->flags,
                       LOOM_LOW_DESCRIPTOR_SET_FLAG_IDEMPOTENT_STATE_UPDATES);
  loom_expression_walk_t* walk = NULL;
  IREE_RETURN_IF_ERROR(loom_expression_walk_allocate(
      module, region, has_idempotent_state_updates ? &identity_policy : NULL,
      /*maximum_entry_count=*/0, arena, &walk));
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
    // The assignment's observable effect belongs to its exact state class.
    // A boundary identity, unlike the value epoch, distinguishes zero-result
    // operations and invalidates every class without clearing the class table.
    if (cursor.op->parent_block != assignment_block ||
        cursor.op->region_count != 0 ||
        iree_any_bit_set(cursor.traits,
                         LOOM_TRAIT_CALLABLE_BOUNDARY | LOOM_TRAIT_TERMINATOR |
                             LOOM_TRAIT_CONVERGENT | LOOM_TRAIT_MEMORY_FENCE) ||
        (!is_assignment &&
         iree_any_bit_set(cursor.traits, LOOM_TRAIT_UNKNOWN_EFFECTS |
                                             LOOM_TRAIT_OBSERVABLE_EFFECT)) ||
        (packet.descriptor &&
         iree_any_bit_set(packet.descriptor->flags,
                          LOOM_LOW_DESCRIPTOR_FLAG_BARRIER)) ||
        cursor.op->kind == LOOM_OP_LOW_SCHEDULE_FENCE) {
      assignment_boundary = cursor.op;
    }
    assignment_block = cursor.op->parent_block;
    const loom_low_cse_packet_identity_t identity = loom_low_cse_analyze_packet(
        descriptor_set, &cursor, &packet, is_assignment, assignment_boundary,
        state_classes);
    if (identity.assignment) {
      status = loom_expression_replace(&cursor, cursor.op, identity.assignment);
      if (iree_status_is_ok(status)) {
        loom_pass_mark_changed(pass);
        ++statistics->expressions_eliminated;
      }
      continue;
    }
    const uint32_t minimum_epoch =
        iree_max(identity.minimum_epoch,
                 loom_expression_observe_barriers(&cursor, &barriers));
    const bool is_idempotent_update = iree_any_bit_set(
        identity.flags, LOOM_LOW_CSE_PACKET_FLAG_IDEMPOTENT_UPDATE);
    const bool writes_state =
        iree_any_bit_set(identity.flags, LOOM_LOW_CSE_PACKET_FLAG_WRITES_STATE);
    const bool is_reusable =
        (!writes_state || is_idempotent_update) &&
        loom_expression_op_is_reusable(module, cursor.op, cursor.traits);
    bool eliminated = false;
    if (is_reusable) {
      loom_expression_entry_t candidate = {
          .op = cursor.op,
          .hash = loom_low_cse_expression_hash(
              module, cursor.op, &value_identities, is_idempotent_update),
          .metadata = cursor.epoch,
      };
      loom_expression_lookup_flags_t flags = 0;
      if (iree_any_bit_set(identity.flags,
                           LOOM_LOW_CSE_PACKET_FLAG_IMPLICIT_STATE) ||
          is_idempotent_update ||
          !iree_any_bit_set(cursor.traits, LOOM_TRAIT_PURE)) {
        flags = LOOM_EXPRESSION_LOOKUP_FLAG_STATEFUL;
      }
      loom_expression_entry_t* existing =
          loom_expression_scope_find(&cursor, candidate.hash, flags);
      if (existing && existing->metadata >= minimum_epoch) {
        status = loom_expression_replace(&cursor, cursor.op, existing->op);
        if (iree_status_is_ok(status)) {
          eliminated = true;
          loom_pass_mark_changed(pass);
          ++statistics->expressions_eliminated;
        }
      } else {
        loom_expression_scope_insert(&cursor, candidate);
      }
    }
    if (eliminated || !iree_status_is_ok(status)) {
      continue;
    }
    loom_low_cse_commit_packet(descriptor_set, &cursor, &packet, is_assignment,
                               assignment_boundary, state_classes);
    if (has_idempotent_state_updates) {
      status = loom_low_cse_observe_structural_identity(module, cursor.op,
                                                        &value_identities);
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
