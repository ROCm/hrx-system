// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/memory_access_ir.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

// Version zero is a fixed projection of one source memory access summary. IDs
// are function-local comparison labels, intervals are relative to the alias
// root, and expression IDs are deliberately excluded because their arena dies
// with source analysis.
enum loom_low_memory_access_ir_field_e {
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_VERSION = 0,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_MEMORY_SPACE = 1,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_ALIAS_ROOT_ID = 2,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_ALIAS_GROUP_ID = 3,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_PRECISION_FLAGS = 4,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDE_BYTES = 5,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDED_BEGIN_BYTES = 6,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDED_END_BYTES = 7,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_PRECISION_FLAGS = 8,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_BEGIN_LO = 9,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_BEGIN_HI = 10,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_END_LO = 11,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_END_HI = 12,
  LOOM_LOW_MEMORY_ACCESS_IR_FIELD_COUNT = 13,
};

#define LOOM_LOW_MEMORY_ACCESS_IR_VERSION 0

static const loom_low_memory_access_precision_flags_t
    kLoomLowMemoryAccessIrPrecisionMask =
        LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE |
        LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
        LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP |
        LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL |
        LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL;

static const loom_low_byte_interval_precision_flags_t
    kLoomLowMemoryAccessIrIntervalPrecisionMask =
        LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
        LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE;

static bool loom_low_memory_access_ir_has_refinement(
    loom_low_memory_access_precision_flags_t precision_flags) {
  return iree_any_bit_set(
      precision_flags, LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
                           LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP |
                           LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL |
                           LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL);
}

static iree_status_t loom_low_memory_access_ir_validate_summary(
    const loom_low_memory_access_summary_t* summary,
    loom_low_memory_access_precision_flags_t precision_flags) {
  if (iree_any_bit_set(summary->precision_flags,
                       ~kLoomLowMemoryAccessIrPrecisionMask)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "low memory access summary precision 0x%08" PRIX32
        " cannot be represented in durable low IR",
        summary->precision_flags & ~kLoomLowMemoryAccessIrPrecisionMask);
  }
  const loom_low_memory_space_t memory_space =
      loom_low_memory_access_normalize_space(summary->memory_space);
  const bool has_space_precision =
      iree_any_bit_set(precision_flags, LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE);
  if (has_space_precision != (memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low memory access space disagrees with its precision flags");
  }
  const bool has_root_precision =
      iree_any_bit_set(precision_flags, LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT);
  if (has_root_precision !=
      (summary->alias_root_id != LOOM_LOW_MEMORY_ALIAS_ID_NONE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low memory access root disagrees with its precision flags");
  }
  const bool has_group_precision =
      iree_any_bit_set(precision_flags, LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP);
  if (has_group_precision !=
      (summary->alias_group_id != LOOM_LOW_MEMORY_ALIAS_ID_NONE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low memory access group disagrees with its precision flags");
  }
  const bool has_interval_precision = iree_any_bit_set(
      precision_flags, LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL);
  if (has_interval_precision) {
    if (summary->byte_interval == NULL ||
        !iree_all_bits_set(summary->byte_interval->precision_flags,
                           kLoomLowMemoryAccessIrIntervalPrecisionMask) ||
        summary->byte_interval->begin_facts.range_lo >
            summary->byte_interval->begin_facts.range_hi ||
        summary->byte_interval->end_facts.range_lo >
            summary->byte_interval->end_facts.range_hi) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low memory access interval precision requires bounded ranges");
    }
  } else if (summary->byte_interval != NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low memory access interval disagrees with its precision flags");
  }
  const loom_low_strided_byte_interval_t* interval = &summary->strided_interval;
  const bool has_strided_precision = iree_any_bit_set(
      precision_flags, LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL);
  if (has_strided_precision) {
    if (!has_root_precision || interval->stride_bytes == 0 ||
        interval->stride_bytes > INT64_MAX ||
        interval->begin_bytes >= interval->end_bytes ||
        interval->end_bytes > interval->stride_bytes) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low strided memory interval requires a root and a non-wrapping "
          "positive i64 stride");
    }
  } else if (interval->stride_bytes != 0 || interval->begin_bytes != 0 ||
             interval->end_bytes != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low strided memory interval disagrees with its precision flags");
  }
  return iree_ok_status();
}

iree_status_t loom_low_memory_access_ir_attach(
    loom_module_t* module, loom_op_t* low_op,
    const loom_low_memory_access_summary_t* summary) {
  if (!loom_low_op_isa(low_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory access summaries require low.op packets");
  }
  loom_low_memory_access_precision_flags_t precision_flags =
      summary->precision_flags & kLoomLowMemoryAccessIrPrecisionMask;
  IREE_RETURN_IF_ERROR(
      loom_low_memory_access_ir_validate_summary(summary, precision_flags));
  if (!loom_low_memory_access_ir_has_refinement(precision_flags)) {
    return iree_ok_status();
  }
  if (!loom_attr_is_absent(loom_low_op_memory_access(low_op))) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "low.op already carries a source memory access summary");
  }

  int64_t* fields = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, LOOM_LOW_MEMORY_ACCESS_IR_FIELD_COUNT, sizeof(*fields),
      (void**)&fields));
  memset(fields, 0, LOOM_LOW_MEMORY_ACCESS_IR_FIELD_COUNT * sizeof(*fields));
  fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_VERSION] =
      LOOM_LOW_MEMORY_ACCESS_IR_VERSION;
  fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_MEMORY_SPACE] =
      loom_low_memory_access_normalize_space(summary->memory_space);
  fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_ALIAS_ROOT_ID] =
      iree_any_bit_set(precision_flags, LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT)
          ? (int64_t)summary->alias_root_id
          : -1;
  fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_ALIAS_GROUP_ID] =
      iree_any_bit_set(precision_flags, LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP)
          ? (int64_t)summary->alias_group_id
          : -1;
  fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_PRECISION_FLAGS] = precision_flags;
  if (iree_any_bit_set(precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL)) {
    fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDE_BYTES] =
        (int64_t)summary->strided_interval.stride_bytes;
    fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDED_BEGIN_BYTES] =
        (int64_t)summary->strided_interval.begin_bytes;
    fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDED_END_BYTES] =
        (int64_t)summary->strided_interval.end_bytes;
  }
  if (iree_any_bit_set(precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL)) {
    const loom_low_byte_interval_t* interval = summary->byte_interval;
    fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_PRECISION_FLAGS] =
        interval->precision_flags & kLoomLowMemoryAccessIrIntervalPrecisionMask;
    fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_BEGIN_LO] =
        interval->begin_facts.range_lo;
    fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_BEGIN_HI] =
        interval->begin_facts.range_hi;
    fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_END_LO] =
        interval->end_facts.range_lo;
    fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_END_HI] =
        interval->end_facts.range_hi;
  }
  loom_op_attrs(low_op)[loom_low_op_memory_access_ATTR_INDEX] =
      loom_attr_i64_array(fields, LOOM_LOW_MEMORY_ACCESS_IR_FIELD_COUNT);
  return iree_ok_status();
}

static iree_status_t loom_low_memory_access_ir_decode(
    const loom_op_t* low_op, loom_low_memory_access_record_t* out_record) {
  const loom_attribute_t attr = loom_low_op_memory_access(low_op);
  if (attr.kind != LOOM_ATTR_I64_ARRAY ||
      attr.count != LOOM_LOW_MEMORY_ACCESS_IR_FIELD_COUNT ||
      attr.i64_array == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low.op memory_access must be a %u-element i64 array",
        LOOM_LOW_MEMORY_ACCESS_IR_FIELD_COUNT);
  }
  const int64_t* fields = attr.i64_array;
  if (fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_VERSION] !=
      LOOM_LOW_MEMORY_ACCESS_IR_VERSION) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "low.op memory_access version %" PRId64
                            " is unsupported",
                            fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_VERSION]);
  }
  const int64_t encoded_space =
      fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_MEMORY_SPACE];
  const int64_t encoded_root =
      fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_ALIAS_ROOT_ID];
  const int64_t encoded_group =
      fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_ALIAS_GROUP_ID];
  const int64_t encoded_precision =
      fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_PRECISION_FLAGS];
  if (encoded_space < LOOM_LOW_MEMORY_SPACE_NONE ||
      encoded_space > LOOM_LOW_MEMORY_SPACE_WASM_MEMORY || encoded_root < -1 ||
      encoded_root >= (int64_t)UINT32_MAX || encoded_group < -1 ||
      encoded_group >= (int64_t)UINT32_MAX || encoded_precision < 0 ||
      (uint64_t)encoded_precision > UINT32_MAX ||
      iree_any_bit_set((uint32_t)encoded_precision,
                       ~kLoomLowMemoryAccessIrPrecisionMask)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low.op memory_access header is malformed");
  }
  const loom_low_memory_access_precision_flags_t precision_flags =
      (loom_low_memory_access_precision_flags_t)encoded_precision;
  const loom_low_memory_space_t memory_space =
      loom_low_memory_access_normalize_space(
          (loom_low_memory_space_t)encoded_space);
  if ((iree_any_bit_set(precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) !=
       (encoded_root >= 0)) ||
      (iree_any_bit_set(precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP) !=
       (encoded_group >= 0)) ||
      (iree_any_bit_set(precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE) !=
       (memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low.op memory_access fields disagree with precision flags");
  }

  loom_low_memory_access_summary_t summary = {
      .memory_space = memory_space,
      .alias_root_id = encoded_root >= 0 ? (uint32_t)encoded_root
                                         : LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = encoded_group >= 0 ? (uint32_t)encoded_group
                                           : LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = precision_flags,
  };
  if (iree_any_bit_set(precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL)) {
    const int64_t stride = fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDE_BYTES];
    const int64_t begin =
        fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDED_BEGIN_BYTES];
    const int64_t end =
        fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDED_END_BYTES];
    if (!iree_any_bit_set(precision_flags,
                          LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) ||
        stride <= 0 || begin < 0 || begin >= end || end > stride) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low.op memory_access strided interval is malformed");
    }
    summary.strided_interval = (loom_low_strided_byte_interval_t){
        .stride_bytes = (uint64_t)stride,
        .begin_bytes = (uint64_t)begin,
        .end_bytes = (uint64_t)end,
    };
  } else if (fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDE_BYTES] != 0 ||
             fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDED_BEGIN_BYTES] != 0 ||
             fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_STRIDED_END_BYTES] != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low.op memory_access has strided fields without precision");
  }

  if (iree_any_bit_set(precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL)) {
    const int64_t interval_precision =
        fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_PRECISION_FLAGS];
    const int64_t begin_lo =
        fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_BEGIN_LO];
    const int64_t begin_hi =
        fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_BEGIN_HI];
    const int64_t end_lo =
        fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_END_LO];
    const int64_t end_hi =
        fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_END_HI];
    if (interval_precision !=
            (int64_t)kLoomLowMemoryAccessIrIntervalPrecisionMask ||
        begin_lo > begin_hi || end_lo > end_hi) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low.op memory_access bounded interval is malformed");
    }
    out_record->byte_interval = (loom_low_byte_interval_t){
        .begin_facts = loom_value_facts_make(begin_lo, begin_hi, 1),
        .end_facts = loom_value_facts_make(end_lo, end_hi, 1),
        .begin_expr_id = LOOM_LOW_MEMORY_EXPR_ID_NONE,
        .end_expr_id = LOOM_LOW_MEMORY_EXPR_ID_NONE,
        .precision_flags =
            (loom_low_byte_interval_precision_flags_t)interval_precision,
    };
    summary.byte_interval = &out_record->byte_interval;
  } else if (fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_PRECISION_FLAGS] !=
                 0 ||
             fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_BEGIN_LO] != 0 ||
             fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_BEGIN_HI] != 0 ||
             fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_END_LO] != 0 ||
             fields[LOOM_LOW_MEMORY_ACCESS_IR_FIELD_INTERVAL_END_HI] != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low.op memory_access has bounded interval fields without precision");
  }
  out_record->summary = summary;
  return iree_ok_status();
}

iree_status_t loom_low_memory_access_table_build_from_ir(
    const loom_op_t* low_func_op, iree_arena_allocator_t* arena,
    loom_low_memory_access_table_t* out_table) {
  *out_table = loom_low_memory_access_table_empty();
  loom_region_t* body = loom_low_function_body((loom_op_t*)low_func_op);
  if (body == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory access table requires a low function");
  }

  iree_host_size_t record_count = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_op_isa(op) &&
          !loom_attr_is_absent(loom_low_op_memory_access(op))) {
        ++record_count;
      }
    }
  }
  if (record_count == 0) {
    return iree_ok_status();
  }

  loom_low_memory_access_record_t* records = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, record_count, sizeof(*records), (void**)&records));
  memset(records, 0, record_count * sizeof(*records));
  iree_host_size_t record_index = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_op_isa(op) ||
          loom_attr_is_absent(loom_low_op_memory_access(op))) {
        continue;
      }
      loom_low_memory_access_record_t* record = &records[record_index++];
      record->position = (loom_low_memory_access_position_t){
          .block_index = block_index,
          .block_ordinal = op->block_ordinal,
      };
      record->op = op;
      IREE_RETURN_IF_ERROR(loom_low_memory_access_ir_decode(op, record));
    }
  }
  IREE_ASSERT_EQ(record_index, record_count);
  *out_table = (loom_low_memory_access_table_t){
      .function_op = low_func_op,
      .values = records,
      .count = record_count,
  };
  return iree_ok_status();
}
