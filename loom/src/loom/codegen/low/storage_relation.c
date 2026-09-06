// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_relation.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

static uint32_t loom_low_storage_relation_value_unit_count(
    const loom_module_t* module, loom_value_id_t value_id) {
  return loom_low_register_type_unit_count(
      loom_module_value_type(module, value_id));
}

static uint32_t loom_low_storage_relation_checked_add_units(uint32_t lhs,
                                                            uint32_t rhs) {
  IREE_ASSERT(rhs <= UINT32_MAX - lhs,
              "verified low storage relation unit offsets must fit u32");
  return lhs + rhs;
}

static uint16_t loom_low_storage_relation_count_checked(uint32_t count) {
  IREE_ASSERT(count <= UINT16_MAX,
              "verified low storage relation count must fit u16");
  return (uint16_t)count;
}

static loom_value_relation_mask_t loom_low_storage_value_relation_mask(
    const loom_op_t* op) {
  loom_value_relation_mask_t mask =
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_TIED_RESULT);
  if (!loom_traits_have_storage_relation(op->traits)) return mask;
  if (loom_traits_are_fact_identity(op->traits)) {
    IREE_ASSERT_EQ(op->tied_result_count, 0,
                   "low fact identity must use ordinal storage aliases");
  }
  return mask | LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
         LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS) |
         LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_CFG_ARGUMENT) |
         LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_CARRIED) |
         LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REGION_RESULT);
}

static uint32_t loom_low_storage_low_relation_count(const loom_op_t* op) {
  if (!loom_traits_have_storage_relation(op->traits)) return 0;
  switch (op->kind) {
    case LOOM_OP_LOW_COPY:
    case LOOM_OP_LOW_MOVE:
    case LOOM_OP_LOW_SLICE:
      return 1;
    case LOOM_OP_LOW_CONCAT:
      return loom_low_concat_sources(op).count;
    case LOOM_OP_LOW_ASSUME:
    case LOOM_OP_LOW_BR:
    case LOOM_OP_LOW_SCF_YIELD:
    case LOOM_OP_LOW_SCF_CONDITION:
    case LOOM_OP_LOW_SCF_FOR:
    case LOOM_OP_LOW_SCF_WHILE:
      return 0;
    default:
      IREE_ASSERT_UNREACHABLE(
          "op with storage-relation trait must have a low storage relation "
          "implementation");
      IREE_BUILTIN_UNREACHABLE();
  }
}

uint16_t loom_low_storage_relation_count(const loom_module_t* module,
                                         const loom_op_t* op) {
  if (op == NULL || (op->tied_result_count == 0 &&
                     !loom_traits_have_storage_relation(op->traits))) {
    return 0;
  }
  uint32_t count = 0;
  loom_value_relation_iterator_t iterator;
  loom_value_relation_iterator_initialize(
      module, op, loom_low_storage_value_relation_mask(op), &iterator);
  loom_value_relation_t relation;
  while (loom_value_relation_iterator_next(&iterator, &relation)) {
    ++count;
  }
  count += loom_low_storage_low_relation_count(op);
  return loom_low_storage_relation_count_checked(count);
}

static void loom_low_storage_relation_from_value_relation(
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_relation_t* value_relation,
    loom_low_storage_relation_t* out_relation) {
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(
          module, value_relation->destination_value_id);
  const uint32_t source_unit_count = loom_low_storage_relation_value_unit_count(
      module, value_relation->source_value_id);
  IREE_ASSERT_EQ(destination_unit_count, source_unit_count,
                 "verified whole-value storage relation must use matching "
                 "unit counts");

  loom_low_storage_relation_cause_t cause =
      LOOM_LOW_STORAGE_RELATION_CAUSE_UNKNOWN;
  loom_low_storage_relation_flags_t flags = 0;
  switch (value_relation->kind) {
    case LOOM_VALUE_RELATION_TIED_RESULT:
      IREE_ASSERT(!iree_any_bit_set(value_relation->flags,
                                    LOOM_VALUE_RELATION_FLAG_TYPE_CHANGE),
                  "verified low tied result must not change storage type");
      cause = LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT;
      flags = LOOM_LOW_STORAGE_RELATION_FLAG_HARD;
      break;
    case LOOM_VALUE_RELATION_FACT_IDENTITY:
    case LOOM_VALUE_RELATION_VALUE_ALIAS:
      cause = LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT;
      flags = LOOM_LOW_STORAGE_RELATION_FLAG_HARD;
      break;
    case LOOM_VALUE_RELATION_CFG_ARGUMENT:
      cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH;
      flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED;
      break;
    case LOOM_VALUE_RELATION_LOOP_CARRIED:
      if (loom_low_scf_for_isa(op) || loom_low_scf_while_isa(op)) {
        cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_LOOP_ENTRY;
      } else if (loom_low_scf_yield_isa(op)) {
        cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD;
      } else {
        IREE_ASSERT(loom_low_scf_condition_isa(op),
                    "verified low loop relation must use a structural op");
        cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_CONDITION;
      }
      flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED;
      break;
    case LOOM_VALUE_RELATION_REGION_RESULT:
      IREE_ASSERT(loom_low_scf_yield_isa(op),
                  "verified low region result must use low.scf.yield");
      cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD;
      flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED;
      break;
    case LOOM_VALUE_RELATION_UNKNOWN:
    case LOOM_VALUE_RELATION_SELECT_PAYLOAD:
    case LOOM_VALUE_RELATION_ELEMENTWISE:
    case LOOM_VALUE_RELATION_LOOP_BYPASS:
    case LOOM_VALUE_RELATION_COUNT_:
      IREE_ASSERT_UNREACHABLE("unsupported low storage value relation");
      IREE_BUILTIN_UNREACHABLE();
  }

  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = value_relation->destination_value_id,
      .source_value_id = value_relation->source_value_id,
      .source_operand_index = value_relation->source_operand_index,
      .destination_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = destination_unit_count,
      .kind = LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
      .cause = cause,
      .flags = flags,
  };
}

static void loom_low_storage_relation_get_copy_or_move(
    const loom_module_t* module, const loom_op_t* op,
    loom_low_storage_relation_t* out_relation) {
  const bool is_copy = loom_low_copy_isa(op);
  IREE_ASSERT(is_copy || loom_low_move_isa(op),
              "copy or move storage relation requires matching low op");
  const loom_value_id_t destination_value_id =
      is_copy ? loom_low_copy_result(op) : loom_low_move_result(op);
  const loom_value_id_t source_value_id =
      is_copy ? loom_low_copy_source(op) : loom_low_move_source(op);
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  IREE_ASSERT_EQ(destination_unit_count, source_unit_count,
                 "verified low copy/move relation must use matching units");
  const bool detached =
      is_copy ? loom_low_copy_detached(op) : loom_low_move_detached(op);
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = 0,
      .destination_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = destination_unit_count,
      .kind = detached ? LOOM_LOW_STORAGE_RELATION_DISJOINT_STORAGE
                       : LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
      .cause = is_copy ? LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_COPY
                       : LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_MOVE,
      .flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED,
  };
}

static void loom_low_storage_relation_get_slice(
    const loom_module_t* module, const loom_op_t* op,
    loom_low_storage_relation_t* out_relation) {
  const int64_t offset = loom_low_slice_offset(op);
  IREE_ASSERT(offset >= 0 && offset <= UINT32_MAX,
              "verified low.slice offset must fit in uint32_t");
  const loom_value_id_t destination_value_id = loom_low_slice_result(op);
  const loom_value_id_t source_value_id = loom_low_slice_source(op);
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  const uint32_t source_unit_offset = (uint32_t)offset;
  IREE_ASSERT(
      source_unit_offset <= source_unit_count &&
          destination_unit_count <= source_unit_count - source_unit_offset,
      "verified low.slice range must fit source unit count");
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = 0,
      .destination_unit_offset = 0,
      .source_unit_offset = source_unit_offset,
      .unit_count = destination_unit_count,
      .kind = LOOM_LOW_STORAGE_RELATION_SUBRANGE,
      .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SLICE,
      .flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED,
  };
}

static void loom_low_storage_relation_get_concat(
    const loom_module_t* module, const loom_op_t* op, uint16_t relation_index,
    uint32_t destination_unit_offset,
    loom_low_storage_relation_t* out_relation) {
  const loom_value_id_t destination_value_id = loom_low_concat_result(op);
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const loom_value_slice_t sources = loom_low_concat_sources(op);
  IREE_ASSERT(relation_index < sources.count,
              "low.concat storage relation index must be in range");
  const loom_value_id_t source_value_id = sources.values[relation_index];
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  IREE_ASSERT(
      destination_unit_offset <= destination_unit_count &&
          source_unit_count <= destination_unit_count - destination_unit_offset,
      "verified low.concat source units must fit result unit count");
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = relation_index,
      .destination_unit_offset = destination_unit_offset,
      .source_unit_offset = 0,
      .unit_count = source_unit_count,
      .kind = LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART,
      .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_CONCAT,
      .flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED,
  };
}

void loom_low_storage_relation_iterator_initialize(
    const loom_module_t* module, const loom_op_t* op,
    loom_low_storage_relation_iterator_t* out_iterator) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(out_iterator);
  *out_iterator = (loom_low_storage_relation_iterator_t){
      .module = module,
      .op = op,
  };
  if (op->tied_result_count == 0 &&
      !loom_traits_have_storage_relation(op->traits)) {
    return;
  }
  loom_value_relation_iterator_initialize(
      module, op, loom_low_storage_value_relation_mask(op),
      &out_iterator->value_relations);
}

bool loom_low_storage_relation_iterator_next(
    loom_low_storage_relation_iterator_t* iterator,
    loom_low_storage_relation_t* out_relation) {
  IREE_ASSERT_ARGUMENT(iterator);
  IREE_ASSERT_ARGUMENT(iterator->module);
  IREE_ASSERT_ARGUMENT(iterator->op);
  IREE_ASSERT_ARGUMENT(out_relation);
  if (iterator->op->tied_result_count == 0 &&
      !loom_traits_have_storage_relation(iterator->op->traits)) {
    return false;
  }

  loom_value_relation_t value_relation;
  if (loom_value_relation_iterator_next(&iterator->value_relations,
                                        &value_relation)) {
    loom_low_storage_relation_from_value_relation(
        iterator->module, iterator->op, &value_relation, out_relation);
    return true;
  }

  const uint32_t low_relation_count =
      loom_low_storage_low_relation_count(iterator->op);
  if (iterator->low_relation_index >= low_relation_count) return false;
  const uint16_t relation_index = iterator->low_relation_index++;
  switch (iterator->op->kind) {
    case LOOM_OP_LOW_COPY:
    case LOOM_OP_LOW_MOVE:
      IREE_ASSERT_EQ(relation_index, 0);
      loom_low_storage_relation_get_copy_or_move(iterator->module, iterator->op,
                                                 out_relation);
      return true;
    case LOOM_OP_LOW_SLICE:
      IREE_ASSERT_EQ(relation_index, 0);
      loom_low_storage_relation_get_slice(iterator->module, iterator->op,
                                          out_relation);
      return true;
    case LOOM_OP_LOW_CONCAT:
      loom_low_storage_relation_get_concat(
          iterator->module, iterator->op, relation_index,
          iterator->concat_destination_unit_offset, out_relation);
      iterator->concat_destination_unit_offset =
          loom_low_storage_relation_checked_add_units(
              iterator->concat_destination_unit_offset,
              out_relation->unit_count);
      return true;
    default:
      IREE_ASSERT_UNREACHABLE("unknown low-specific storage relation");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_low_storage_unit_ranges_overlap(uint32_t lhs_offset,
                                                 uint32_t lhs_count,
                                                 uint32_t rhs_offset,
                                                 uint32_t rhs_count) {
  const uint64_t lhs_end = (uint64_t)lhs_offset + lhs_count;
  const uint64_t rhs_end = (uint64_t)rhs_offset + rhs_count;
  return lhs_offset < rhs_end && rhs_offset < lhs_end;
}

bool loom_low_storage_operand_may_read_unit_range(const loom_module_t* module,
                                                  const loom_op_t* op,
                                                  uint16_t operand_index,
                                                  uint32_t unit_offset,
                                                  uint32_t unit_count) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_LT(operand_index, op->operand_count);
  const loom_value_id_t operand_value_id =
      loom_op_const_operands(op)[operand_index];
  const uint32_t operand_unit_count =
      loom_low_storage_relation_value_unit_count(module, operand_value_id);
  IREE_ASSERT(unit_offset <= operand_unit_count &&
                  unit_count <= operand_unit_count - unit_offset,
              "verified low storage query must fit operand units");
  if (unit_count == 0) return false;

  bool has_source_relation = false;
  loom_low_storage_relation_iterator_t iterator;
  loom_low_storage_relation_iterator_initialize(module, op, &iterator);
  loom_low_storage_relation_t relation;
  while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
    if (relation.source_operand_index != operand_index) continue;
    IREE_ASSERT_EQ(relation.source_value_id, operand_value_id);
    has_source_relation = true;
    if (loom_low_storage_unit_ranges_overlap(unit_offset, unit_count,
                                             relation.source_unit_offset,
                                             relation.unit_count)) {
      return true;
    }
  }
  return !has_source_relation;
}
