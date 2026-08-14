// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_relation.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
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

uint16_t loom_low_storage_relation_count(const loom_module_t* module,
                                         const loom_op_t* op) {
  (void)module;
  if (op == NULL) return 0;

  uint32_t count = op->tied_result_count;
  if (!loom_traits_have_storage_relation(op->traits)) {
    return loom_low_storage_relation_count_checked(count);
  }

  if (loom_traits_are_fact_identity(op->traits)) {
    IREE_ASSERT_EQ(op->tied_result_count, 0,
                   "fact identity storage relations must be ordinal aliases");
    IREE_ASSERT_EQ(op->operand_count, op->result_count,
                   "verified low fact identity field counts must match");
    count += op->result_count;
    return loom_low_storage_relation_count_checked(count);
  }

  switch (op->kind) {
    case LOOM_OP_LOW_COPY:
      ++count;
      break;
    case LOOM_OP_LOW_SLICE:
      ++count;
      break;
    case LOOM_OP_LOW_CONCAT:
      count += loom_low_concat_sources(op).count;
      break;
    case LOOM_OP_LOW_BR:
      count += loom_low_br_args(op).count;
      break;
    case LOOM_OP_LOW_SCF_YIELD: {
      IREE_ASSERT(op->parent_op != NULL,
                  "verified low.scf.yield must have a structured parent");
      loom_value_slice_t values = loom_low_scf_yield_values(op);
      if (loom_low_scf_for_isa(op->parent_op)) {
        count += values.count * 2;
      } else {
        count += values.count;
      }
      break;
    }
    case LOOM_OP_LOW_SCF_FOR:
      count += loom_low_scf_for_iter_args(op).count;
      break;
    default:
      IREE_ASSERT_UNREACHABLE(
          "op with storage-relation trait must have a low "
          "storage relation implementation");
      IREE_BUILTIN_UNREACHABLE();
  }
  return loom_low_storage_relation_count_checked(count);
}

static void loom_low_storage_relation_get_tied_result(
    const loom_module_t* module, const loom_op_t* op, uint16_t relation_index,
    loom_low_storage_relation_t* out_relation) {
  const loom_tied_result_t tied = loom_op_tied_results(op)[relation_index];
  IREE_ASSERT(tied.result_index < op->result_count &&
                  tied.operand_index < op->operand_count,
              "verified tied result metadata must reference existing fields");
  IREE_ASSERT(!tied.has_type_change,
              "verified low tied result metadata must not require storage "
              "type conversion");
  const loom_value_id_t destination_value_id =
      loom_op_const_results(op)[tied.result_index];
  const loom_value_id_t source_value_id =
      loom_op_const_operands(op)[tied.operand_index];
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  IREE_ASSERT_EQ(destination_unit_count, source_unit_count,
                 "verified tied storage relation must use matching unit "
                 "counts");
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = tied.operand_index,
      .destination_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = destination_unit_count,
      .kind = LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
      .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT,
      .flags = LOOM_LOW_STORAGE_RELATION_FLAG_HARD,
  };
}

static void loom_low_storage_relation_get_fact_identity(
    const loom_module_t* module, const loom_op_t* op, uint16_t relation_index,
    loom_low_storage_relation_t* out_relation) {
  IREE_ASSERT(
      relation_index < op->operand_count && relation_index < op->result_count,
      "verified low fact identity relation must reference matching "
      "fields");
  const loom_value_id_t destination_value_id =
      loom_op_const_results(op)[relation_index];
  const loom_value_id_t source_value_id =
      loom_op_const_operands(op)[relation_index];
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  IREE_ASSERT_EQ(destination_unit_count, source_unit_count,
                 "verified low fact identity must use matching unit counts");
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = relation_index,
      .destination_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = destination_unit_count,
      .kind = LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
      .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT,
      .flags = LOOM_LOW_STORAGE_RELATION_FLAG_HARD,
  };
}

static void loom_low_storage_relation_get_copy(
    const loom_module_t* module, const loom_op_t* op,
    loom_low_storage_relation_t* out_relation) {
  const loom_value_id_t destination_value_id = loom_low_copy_result(op);
  const loom_value_id_t source_value_id = loom_low_copy_source(op);
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  IREE_ASSERT_EQ(destination_unit_count, source_unit_count,
                 "verified low.copy storage relation must use matching unit "
                 "counts");
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = 0,
      .destination_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = destination_unit_count,
      .kind = loom_low_copy_detached(op)
                  ? LOOM_LOW_STORAGE_RELATION_DISJOINT_STORAGE
                  : LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
      .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_COPY,
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

static void loom_low_storage_relation_get_concat_at_offset(
    const loom_module_t* module, const loom_op_t* op, uint16_t relation_index,
    uint32_t destination_unit_offset,
    loom_low_storage_relation_t* out_relation) {
  const loom_value_id_t destination_value_id = loom_low_concat_result(op);
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  loom_value_slice_t sources = loom_low_concat_sources(op);
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

static void loom_low_storage_relation_get_concat(
    const loom_module_t* module, const loom_op_t* op, uint16_t relation_index,
    loom_low_storage_relation_t* out_relation) {
  uint32_t destination_unit_offset = 0;
  const loom_value_slice_t sources = loom_low_concat_sources(op);
  IREE_ASSERT(relation_index < sources.count,
              "low.concat storage relation index must be in range");
  for (uint16_t i = 0; i < relation_index; ++i) {
    destination_unit_offset = loom_low_storage_relation_checked_add_units(
        destination_unit_offset,
        loom_low_storage_relation_value_unit_count(module, sources.values[i]));
  }
  loom_low_storage_relation_get_concat_at_offset(
      module, op, relation_index, destination_unit_offset, out_relation);
}

static void loom_low_storage_relation_get_branch(
    const loom_module_t* module, const loom_op_t* op, uint16_t relation_index,
    loom_low_storage_relation_t* out_relation) {
  const loom_block_t* dest = loom_low_br_dest(op);
  loom_value_slice_t args = loom_low_br_args(op);
  IREE_ASSERT_EQ(args.count, dest->arg_count,
                 "verified low.br payload must match destination block");
  IREE_ASSERT(relation_index < args.count,
              "low.br storage relation index must be in range");
  const loom_value_id_t destination_value_id = dest->arg_ids[relation_index];
  const loom_value_id_t source_value_id = args.values[relation_index];
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  IREE_ASSERT_EQ(destination_unit_count, source_unit_count,
                 "verified low.br storage relation must use matching unit "
                 "counts");
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = relation_index,
      .destination_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = destination_unit_count,
      .kind = LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
      .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH,
      .flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED,
  };
}

static void loom_low_storage_relation_get_scf_for(
    const loom_module_t* module, const loom_op_t* op, uint16_t relation_index,
    loom_low_storage_relation_t* out_relation) {
  loom_value_slice_t iter_args = loom_low_scf_for_iter_args(op);
  IREE_ASSERT(relation_index < iter_args.count,
              "low.scf.for storage relation index must be in range");
  const loom_block_t* body_block =
      loom_region_const_entry_block(loom_low_scf_for_body(op));
  IREE_ASSERT(body_block->arg_count == iter_args.count + 1,
              "verified low.scf.for body args must match iter_args");
  const loom_value_id_t destination_value_id =
      loom_block_arg_id(body_block, (uint16_t)(relation_index + 1));
  const loom_value_id_t source_value_id = iter_args.values[relation_index];
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  IREE_ASSERT_EQ(destination_unit_count, source_unit_count,
                 "verified low.scf.for storage relation must use matching "
                 "unit counts");
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = (uint16_t)(relation_index + 3),
      .destination_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = destination_unit_count,
      .kind = LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
      .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_FOR,
      .flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED,
  };
}

static void loom_low_storage_relation_get_scf_yield(
    const loom_module_t* module, const loom_op_t* op, uint16_t relation_index,
    loom_low_storage_relation_t* out_relation) {
  const loom_op_t* parent_op = op->parent_op;
  IREE_ASSERT(parent_op != NULL,
              "verified low.scf.yield must have a structured parent");
  loom_value_slice_t values = loom_low_scf_yield_values(op);
  IREE_ASSERT(relation_index < values.count * 2,
              "low.scf.yield storage relation index must be in range");
  const uint16_t value_index =
      (uint16_t)(relation_index >= values.count ? relation_index - values.count
                                                : relation_index);
  const loom_value_id_t source_value_id = values.values[value_index];
  loom_value_id_t destination_value_id = LOOM_VALUE_ID_INVALID;
  if (loom_low_scf_for_isa(parent_op)) {
    if (relation_index < values.count) {
      const loom_block_t* body_block =
          loom_region_const_entry_block(loom_low_scf_for_body(parent_op));
      IREE_ASSERT(body_block->arg_count == values.count + 1,
                  "verified low.scf.for body args must match yield values");
      destination_value_id =
          loom_block_arg_id(body_block, (uint16_t)(value_index + 1));
    } else {
      IREE_ASSERT(value_index < parent_op->result_count,
                  "verified low.scf.for results must match yield values");
      destination_value_id = loom_op_const_results(parent_op)[value_index];
    }
  } else {
    IREE_ASSERT(relation_index < values.count,
                "non-loop low.scf.yield relation index must match values");
    IREE_ASSERT(loom_low_scf_if_isa(parent_op),
                "verified low.scf.yield parent must be low.scf.if or "
                "low.scf.for");
    IREE_ASSERT(value_index < parent_op->result_count,
                "verified low.scf.if results must match yield values");
    destination_value_id = loom_op_const_results(parent_op)[value_index];
  }
  const uint32_t destination_unit_count =
      loom_low_storage_relation_value_unit_count(module, destination_value_id);
  const uint32_t source_unit_count =
      loom_low_storage_relation_value_unit_count(module, source_value_id);
  IREE_ASSERT_EQ(destination_unit_count, source_unit_count,
                 "verified low.scf.yield storage relation must use matching "
                 "unit counts");
  *out_relation = (loom_low_storage_relation_t){
      .op = op,
      .destination_value_id = destination_value_id,
      .source_value_id = source_value_id,
      .source_operand_index = value_index,
      .destination_unit_offset = 0,
      .source_unit_offset = 0,
      .unit_count = destination_unit_count,
      .kind = LOOM_LOW_STORAGE_RELATION_SAME_STORAGE,
      .cause = LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD,
      .flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED,
  };
}

void loom_low_storage_relation_get(const loom_module_t* module,
                                   const loom_op_t* op, uint16_t relation_index,
                                   loom_low_storage_relation_t* out_relation) {
  IREE_ASSERT(module != NULL && op != NULL && out_relation != NULL);
  IREE_ASSERT(relation_index < loom_low_storage_relation_count(module, op),
              "low storage relation index must be in range");

  if (relation_index < op->tied_result_count) {
    loom_low_storage_relation_get_tied_result(module, op, relation_index,
                                              out_relation);
    return;
  }
  relation_index -= op->tied_result_count;
  IREE_ASSERT(loom_traits_have_storage_relation(op->traits),
              "non-tied low storage relation requires storage-relation trait");
  if (loom_traits_are_fact_identity(op->traits)) {
    loom_low_storage_relation_get_fact_identity(module, op, relation_index,
                                                out_relation);
    return;
  }

  switch (op->kind) {
    case LOOM_OP_LOW_COPY:
      IREE_ASSERT_EQ(relation_index, 0);
      loom_low_storage_relation_get_copy(module, op, out_relation);
      return;
    case LOOM_OP_LOW_SLICE:
      IREE_ASSERT_EQ(relation_index, 0);
      loom_low_storage_relation_get_slice(module, op, out_relation);
      return;
    case LOOM_OP_LOW_CONCAT:
      loom_low_storage_relation_get_concat(module, op, relation_index,
                                           out_relation);
      return;
    case LOOM_OP_LOW_BR:
      loom_low_storage_relation_get_branch(module, op, relation_index,
                                           out_relation);
      return;
    case LOOM_OP_LOW_SCF_YIELD:
      loom_low_storage_relation_get_scf_yield(module, op, relation_index,
                                              out_relation);
      return;
    case LOOM_OP_LOW_SCF_FOR:
      loom_low_storage_relation_get_scf_for(module, op, relation_index,
                                            out_relation);
      return;
    default:
      IREE_ASSERT_UNREACHABLE(
          "op with storage-relation trait must have a low "
          "storage relation implementation");
      IREE_BUILTIN_UNREACHABLE();
  }
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
      .relation_count = loom_low_storage_relation_count(module, op),
  };
}

bool loom_low_storage_relation_iterator_next(
    loom_low_storage_relation_iterator_t* iterator,
    loom_low_storage_relation_t* out_relation) {
  IREE_ASSERT_ARGUMENT(iterator);
  IREE_ASSERT_ARGUMENT(iterator->module);
  IREE_ASSERT_ARGUMENT(iterator->op);
  IREE_ASSERT_ARGUMENT(out_relation);
  if (iterator->relation_index >= iterator->relation_count) {
    return false;
  }

  const uint16_t relation_index = iterator->relation_index++;
  if (relation_index < iterator->op->tied_result_count) {
    loom_low_storage_relation_get_tied_result(iterator->module, iterator->op,
                                              relation_index, out_relation);
    return true;
  }

  const uint16_t structural_relation_index =
      (uint16_t)(relation_index - iterator->op->tied_result_count);
  if (iterator->op->kind == LOOM_OP_LOW_CONCAT) {
    loom_low_storage_relation_get_concat_at_offset(
        iterator->module, iterator->op, structural_relation_index,
        iterator->concat_destination_unit_offset, out_relation);
    iterator->concat_destination_unit_offset =
        loom_low_storage_relation_checked_add_units(
            iterator->concat_destination_unit_offset, out_relation->unit_count);
    return true;
  }

  loom_low_storage_relation_get(iterator->module, iterator->op, relation_index,
                                out_relation);
  return true;
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
  if (unit_count == 0) {
    return false;
  }

  bool has_source_relation = false;
  loom_low_storage_relation_iterator_t iterator;
  loom_low_storage_relation_iterator_initialize(module, op, &iterator);
  loom_low_storage_relation_t relation;
  while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
    if (relation.source_operand_index != operand_index) {
      continue;
    }
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
