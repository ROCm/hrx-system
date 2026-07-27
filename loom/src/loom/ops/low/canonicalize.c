// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/registers.h"

static loom_op_t* loom_low_defining_op(loom_rewriter_t* rewriter,
                                       loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      (iree_host_size_t)value_id >= rewriter->module->values.count) {
    return NULL;
  }
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) return NULL;
  return loom_value_def_op(value);
}

static iree_status_t loom_low_replace_single_result_with_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static bool loom_low_slice_matches_concat_source(loom_rewriter_t* rewriter,
                                                 loom_op_t* slice_op,
                                                 loom_op_t* concat_op,
                                                 loom_value_id_t* out_source) {
  *out_source = LOOM_VALUE_ID_INVALID;
  const int64_t slice_offset = loom_low_slice_offset(slice_op);
  if (slice_offset < 0) return false;

  const loom_type_t slice_type =
      loom_module_value_type(rewriter->module, loom_low_slice_result(slice_op));
  if (!loom_type_is_register(slice_type)) return false;
  const uint32_t slice_unit_count =
      loom_low_register_type_unit_count(slice_type);

  uint32_t source_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(concat_op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_value_id_t source = sources.values[i];
    const loom_type_t source_type =
        loom_module_value_type(rewriter->module, source);
    if (!loom_type_is_register(source_type)) return false;

    const uint32_t source_unit_count =
        loom_low_register_type_unit_count(source_type);
    if ((uint64_t)slice_offset == source_offset &&
        slice_unit_count == source_unit_count &&
        loom_type_equal(slice_type, source_type)) {
      *out_source = source;
      return true;
    }
    if (source_unit_count > UINT32_MAX - source_offset) return false;
    source_offset += source_unit_count;
  }
  return false;
}

static iree_status_t loom_low_slice_canonicalize_concat_slice(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* concat_op,
    bool* out_changed) {
  *out_changed = false;
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (!loom_low_slice_matches_concat_source(rewriter, op, concat_op,
                                            &replacement)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_low_replace_single_result_with_value(op, rewriter, replacement));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_low_slice_canonicalize_nested_slice(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* inner_slice_op,
    bool* out_changed) {
  *out_changed = false;
  const int64_t outer_offset = loom_low_slice_offset(op);
  const int64_t inner_offset = loom_low_slice_offset(inner_slice_op);
  if (outer_offset < 0 || inner_offset < 0 ||
      outer_offset > INT64_MAX - inner_offset) {
    return iree_ok_status();
  }

  const loom_value_id_t inner_source = loom_low_slice_source(inner_slice_op);
  const loom_type_t inner_source_type =
      loom_module_value_type(rewriter->module, inner_source);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_low_slice_result(op));
  const int64_t combined_offset = inner_offset + outer_offset;
  if (combined_offset == 0 && loom_type_equal(result_type, inner_source_type)) {
    IREE_RETURN_IF_ERROR(
        loom_low_replace_single_result_with_value(op, rewriter, inner_source));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(&rewriter->builder, inner_source,
                                            combined_offset, result_type,
                                            op->location, &replacement_op));
  loom_value_id_t replacement = loom_low_slice_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_low_replace_single_result_with_value(op, rewriter, replacement));
  *out_changed = true;
  return iree_ok_status();
}

iree_status_t loom_low_slice_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  const loom_value_id_t source = loom_low_slice_source(op);
  const loom_type_t source_type =
      loom_module_value_type(rewriter->module, source);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_low_slice_result(op));
  if (loom_low_slice_offset(op) == 0 &&
      loom_type_equal(source_type, result_type)) {
    return loom_low_replace_single_result_with_value(op, rewriter, source);
  }

  loom_op_t* source_op = loom_low_defining_op(rewriter, source);
  if (loom_low_concat_isa(source_op)) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_low_slice_canonicalize_concat_slice(
        op, rewriter, source_op, &changed));
    if (changed) return iree_ok_status();
  }
  if (loom_low_slice_isa(source_op)) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_low_slice_canonicalize_nested_slice(
        op, rewriter, source_op, &changed));
    if (changed) return iree_ok_status();
  }
  return iree_ok_status();
}
