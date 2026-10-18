// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/target_legalization.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/transpose.h"
#include "loom/transforms/vector/bitstream_packet_legalization.h"
#include "loom/transforms/vector/cast_packet_legalization.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/packet_plan.h"
#include "loom/transforms/vector/reduction_legalization.h"
#include "loom/transforms/vector/to_scalar.h"
#include "loom/util/fact_table.h"
#include "loom/util/numeric_format.h"

iree_status_t loom_vector_variadic_concat_to_binary_rewrite_op(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_rewritten) {
  *out_rewritten = false;
  const loom_value_slice_t inputs = loom_vector_concat_inputs(op);
  if (inputs.count <= 2) {
    return iree_ok_status();
  }

  const uint8_t axis = (uint8_t)loom_vector_concat_axis(op);
  bool has_nonempty_input = false;
  for (iree_host_size_t i = 0; i < inputs.count; ++i) {
    const loom_type_t input_type =
        loom_module_value_type(rewriter->module, inputs.values[i]);
    if (loom_type_dim_is_dynamic_at(input_type, axis)) {
      return iree_ok_status();
    }
    has_nonempty_input |= loom_type_dim_static_size_at(input_type, axis) != 0;
  }
  if (!has_nonempty_input) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_concat_canonicalize(op, rewriter));
  *out_rewritten = true;
  return iree_ok_status();
}

static loom_vector_mma_to_scalar_options_t loom_vector_mma_options(
    const loom_target_legalization_context_t* context) {
  const loom_target_contract_query_result_t* query_result =
      context->contract_query_result;
  loom_vector_to_scalar_flags_t flags = LOOM_VECTOR_TO_SCALAR_FLAG_NONE;
  if (loom_kernel_def_isa(context->function.op)) {
    flags |= LOOM_VECTOR_TO_SCALAR_FLAG_ALLOW_SUBGROUP_COMMUNICATION;
  }
  return (loom_vector_mma_to_scalar_options_t){
      .matrix_fragment_layout =
          query_result ? query_result->selected_matrix_fragment_layout : NULL,
      .flags = flags,
  };
}

static bool loom_vector_mma_has_fragment_store_user(const loom_module_t* module,
                                                    const loom_op_t* op) {
  const loom_value_id_t result_id = loom_vector_mma_result(op);
  const loom_value_t* result = loom_module_value(module, result_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result, use) {
    const loom_op_t* user = loom_use_user_op(*use);
    if (loom_vector_memory_op_footprint_kind(module, user) !=
        LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT) {
      continue;
    }
    loom_memory_access_t access = loom_memory_access_cast(module, user);
    if (loom_memory_access_operation_kind(access) ==
            LOOM_MEMORY_ACCESS_OPERATION_STORE &&
        loom_memory_access_value(access) == result_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_vector_legalize_reduce_axes(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_to_vector_rewrite_op(
      context->rewriter, op, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_descriptor(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  // Keep lane programs compact while target rewrites and source cleanup can
  // share their operands. Native packet selection retains first claim before
  // complete legalization expands the remaining unsupported operations.
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
      context->policy != LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    return iree_ok_status();
  }
  // Lane programs may eliminate transient aggregates with no target carrier.
  // The callable boundary still needs a complete representation.
  if (context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_build_fixed_shift(
    loom_builder_t* builder, loom_op_kind_t kind, loom_value_id_t value,
    loom_type_t type, int64_t amount, loom_location_id_t location,
    loom_value_id_t* out_result) {
  loom_op_t* amount_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      builder, loom_attr_i64(amount), type, location, &amount_op));
  const loom_value_id_t amount_value = loom_vector_constant_result(amount_op);

  loom_op_t* shift_op = NULL;
  switch (kind) {
    case LOOM_OP_VECTOR_SHLI: {
      IREE_RETURN_IF_ERROR(loom_vector_shli_build(builder, /*instance_flags=*/0,
                                                  value, amount_value, type,
                                                  location, &shift_op));
      break;
    }
    case LOOM_OP_VECTOR_SHRSI: {
      IREE_RETURN_IF_ERROR(loom_vector_shrsi_build(builder, value, amount_value,
                                                   type, location, &shift_op));
      break;
    }
    case LOOM_OP_VECTOR_SHRUI: {
      IREE_RETURN_IF_ERROR(loom_vector_shrui_build(builder, value, amount_value,
                                                   type, location, &shift_op));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("invalid fixed vector shift kind");
      IREE_BUILTIN_UNREACHABLE();
  }
  *out_result = loom_op_results(shift_op)[0];
  return iree_ok_status();
}

// Preserves rejected bitfield extraction in the vector domain. Targets can
// select the resulting fixed shifts and conversion independently instead of
// exposing every authored lane to scalar IR.
static iree_status_t loom_vector_legalize_bitfield_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }

  const bool signed_extract = loom_vector_bitfield_extracts_isa(op);
  const loom_value_id_t source = loom_op_operands(op)[0];
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  const int32_t source_width =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  const int64_t offset = signed_extract
                             ? loom_vector_bitfield_extracts_offset(op)
                             : loom_vector_bitfield_extractu_offset(op);
  const int64_t width = signed_extract
                            ? loom_vector_bitfield_extracts_width(op)
                            : loom_vector_bitfield_extractu_width(op);

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);

  const bool full_width = offset == 0 && width == source_width;
  const bool top_widen =
      result_width > source_width && offset + width == source_width;
  loom_value_id_t replacement = source;
  loom_type_t working_type = source_type;
  int32_t working_width = source_width;
  if (result_width > source_width) {
    loom_op_t* extension_op = NULL;
    if (signed_extract && (full_width || top_widen)) {
      IREE_RETURN_IF_ERROR(
          loom_vector_extsi_build(builder, replacement, source_type,
                                  result_type, op->location, &extension_op));
      replacement = loom_vector_extsi_result(extension_op);
    } else {
      // Extraction interprets the source as raw bits. Zero extension retains
      // those bits before the field is aligned in the wider result carrier.
      IREE_RETURN_IF_ERROR(
          loom_vector_extui_build(builder, replacement, source_type,
                                  result_type, op->location, &extension_op));
      replacement = loom_vector_extui_result(extension_op);
    }
    working_type = result_type;
    working_width = result_width;
  }

  if (!full_width) {
    if (top_widen) {
      IREE_RETURN_IF_ERROR(loom_vector_build_fixed_shift(
          builder, signed_extract ? LOOM_OP_VECTOR_SHRSI : LOOM_OP_VECTOR_SHRUI,
          replacement, working_type, offset, op->location, &replacement));
    } else {
      const int64_t left_shift = working_width - offset - width;
      if (left_shift != 0) {
        IREE_RETURN_IF_ERROR(loom_vector_build_fixed_shift(
            builder, LOOM_OP_VECTOR_SHLI, replacement, working_type, left_shift,
            op->location, &replacement));
      }
      const int64_t right_shift = working_width - width;
      if (right_shift != 0) {
        IREE_RETURN_IF_ERROR(loom_vector_build_fixed_shift(
            builder,
            signed_extract ? LOOM_OP_VECTOR_SHRSI : LOOM_OP_VECTOR_SHRUI,
            replacement, working_type, right_shift, op->location,
            &replacement));
      }
    }
  }

  if (result_width < source_width) {
    loom_op_t* truncation_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_trunci_build(builder, replacement, source_type, result_type,
                                 op->location, &truncation_op));
    replacement = loom_vector_trunci_result(truncation_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_bitstream(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  if (context->bitstream_policy == NULL) {
    return loom_vector_legalize_descriptor(entry, context, op, out_result);
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  if (!context->source_function_has_unsupported_vector_carrier &&
      context->policy != LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_legalize_bitstream_words(context, op, &rewritten));
  }
  if (!rewritten) {
    const loom_value_id_t source =
        loom_vector_bitpack_isa(op) ? loom_vector_bitpack_source(op)
                                    : (loom_vector_bitunpacku_isa(op)
                                           ? loom_vector_bitunpacku_source(op)
                                           : loom_vector_bitunpacks_source(op));
    const loom_type_t source_type =
        loom_module_value_type(context->module, source);
    if (!loom_target_legalization_has_bitstream_packet_carrier(context,
                                                               source_type)) {
      const loom_value_t* source_value =
          loom_module_value(context->module, source);
      loom_op_t* source_op = loom_value_is_block_arg(source_value)
                                 ? NULL
                                 : loom_value_def_op(source_value);
      if (source_op != NULL && loom_vector_load_isa(source_op)) {
        if (context->vector_packet_policy != NULL) {
          IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_load(
              context, source_op, context->vector_packet_policy, &rewritten));
        } else {
          IREE_RETURN_IF_ERROR(loom_vector_packet_prepare_bitstream_source_load(
              context, source_op, &rewritten));
        }
        if (!rewritten &&
            (context->vector_packet_policy == NULL ||
             context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL)) {
          IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
              context->pass, context->rewriter, source_op, &rewritten));
        }
      }
    }
  }
  if (!rewritten && !context->source_function_has_unsupported_vector_carrier &&
      loom_target_legalization_op_has_bitstream_packet_carriers(context, op)) {
    IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_legalize_index_cast(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_vector_index_cast_isa(op)) {
    return iree_ok_status();
  }
  if (context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_cast(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  return loom_vector_index_cast_isa(op)
             ? loom_vector_legalize_index_cast(entry, context, op, out_result)
             : loom_vector_legalize_descriptor(entry, context, op, out_result);
}

// Returns true when every logical shift-count element is the same exact
// integer. Fixed shifts must fall back to normal descriptor legalization: the
// logarithmic barrel below is itself built from fixed shifts, so expanding one
// again would never reach the target contract.
static bool loom_vector_shift_count_is_fixed(const loom_rewriter_t* rewriter,
                                             loom_value_id_t count) {
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  int64_t exact_count = 0;
  return loom_value_facts_query_all_equal_element(
             &rewriter->fact_table->context,
             loom_rewriter_value_facts(rewriter, count), &element_facts) &&
         loom_value_facts_as_exact_i64(element_facts, &exact_count);
}

// Bounds packet-local widening independently of the authored lane count.
#define LOOM_VECTOR_SHIFT_PACKET_OP_LIMIT 64u

static loom_type_t loom_vector_shift_packet_type(loom_type_t source_type,
                                                 uint32_t lane_count) {
  return loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(source_type),
      loom_dim_pack_static(lane_count),
      loom_type_rank(source_type) == 1 ? source_type.encoding_id : 0);
}

static bool loom_vector_shift_packet_carriers_supported(
    const loom_target_legalization_context_t* context, loom_type_t source_type,
    loom_type_t working_type, uint32_t lane_count) {
  return loom_target_legalization_has_source_vector_carrier(
             context, loom_vector_shift_packet_type(source_type, lane_count),
             LOOM_VALUE_ID_INVALID) &&
         loom_target_legalization_has_source_vector_carrier(
             context, loom_vector_shift_packet_type(working_type, lane_count),
             LOOM_VALUE_ID_INVALID);
}

// Selects the widest declared packet width that admits both the authored
// narrow carrier and the widened working carrier, including the final tail.
// A zero result leaves the ordinary descriptor fallback responsible.
static uint32_t loom_vector_shift_select_packet_lane_count(
    const loom_target_legalization_context_t* context, loom_type_t source_type,
    loom_type_t working_type) {
  const loom_target_vector_packet_policy_t* policy =
      context->vector_packet_policy;
  if (policy == NULL || loom_type_rank(source_type) != 1 ||
      !loom_type_is_all_static(source_type)) {
    return 0;
  }
  const int64_t source_lane_count_i64 =
      loom_type_dim_static_size_at(source_type, 0);
  if (source_lane_count_i64 < 2 || source_lane_count_i64 > UINT32_MAX) {
    return 0;
  }
  const uint32_t source_lane_count = (uint32_t)source_lane_count_i64;

  uint32_t selected_lane_count = 0;
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    const uint32_t candidate_lane_count = policy->native_lane_counts[i];
    if (candidate_lane_count == 0 ||
        candidate_lane_count >= source_lane_count ||
        candidate_lane_count <= selected_lane_count ||
        !loom_vector_shift_packet_carriers_supported(
            context, source_type, working_type, candidate_lane_count)) {
      continue;
    }
    const uint32_t tail_lane_count = source_lane_count % candidate_lane_count;
    if (tail_lane_count != 0 &&
        !loom_vector_shift_packet_carriers_supported(
            context, source_type, working_type, tail_lane_count)) {
      continue;
    }
    const uint32_t packet_count =
        (source_lane_count - 1u) / candidate_lane_count + 1u;
    // Each packet has two slices, two extensions, one shift, and one
    // truncation. The final concat is shared by all packets.
    if ((uint64_t)packet_count * 6u + 1u > LOOM_VECTOR_SHIFT_PACKET_OP_LIMIT) {
      continue;
    }
    selected_lane_count = candidate_lane_count;
  }
  return selected_lane_count;
}

static iree_status_t loom_vector_build_widened_shift(
    loom_builder_t* builder, const loom_op_t* source_op,
    const loom_value_id_t operands[2], loom_type_t source_type,
    loom_type_t working_type, loom_value_id_t* out_result) {
  const bool sign_extend = source_op->kind == LOOM_OP_VECTOR_SHLI ||
                           source_op->kind == LOOM_OP_VECTOR_SHRSI;
  loom_value_id_t widened_operands[2];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(widened_operands); ++i) {
    loom_op_t* extension_op = NULL;
    if (sign_extend) {
      IREE_RETURN_IF_ERROR(loom_vector_extsi_build(
          builder, operands[i], source_type, working_type, source_op->location,
          &extension_op));
      widened_operands[i] = loom_vector_extsi_result(extension_op);
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_extui_build(
          builder, operands[i], source_type, working_type, source_op->location,
          &extension_op));
      widened_operands[i] = loom_vector_extui_result(extension_op);
    }
  }

  // Narrow no-wrap flags do not describe the widened intermediate.
  loom_op_t* widened_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, source_op->kind, 2, 1,
                                                0, 0, 0, source_op->location,
                                                &widened_op));
  loom_op_operands(widened_op)[0] = widened_operands[0];
  loom_op_operands(widened_op)[1] = widened_operands[1];
  IREE_RETURN_IF_ERROR(loom_builder_define_result(
      builder, working_type, &loom_op_results(widened_op)[0]));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, widened_op));
  loom_op_t* truncation_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_trunci_build(
      builder, loom_op_results(widened_op)[0], working_type, source_type,
      source_op->location, &truncation_op));
  *out_result = loom_vector_trunci_result(truncation_op);
  return iree_ok_status();
}

// Keeps rejected shifts in vector IR when another vector representation can
// satisfy a target contract. A valid 64-bit shift count is in [0, 63], so a
// truncate/zero-extend round trip preserves the defined domain while exposing
// an exact 32-bit lane origin. Sixteen-bit shifts use a logarithmic barrel made
// from fixed shifts; sub-halfword shifts first widen to that domain and
// truncate at the source operation boundary. When widening the complete
// authored vector would exceed the target carrier budget, the rewrite widens
// bounded native packets instead of creating an oversized transient carrier.
static iree_status_t loom_vector_legalize_shift(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  const int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (bit_width == 64) {
    const loom_value_id_t count = loom_op_operands(op)[1];
    loom_value_fact_exact_lane_origin_t count_origin = {0};
    const bool has_i32_count_origin =
        loom_value_fact_table_query_exact_lane_origin(
            context->fact_table, context->module, count, &count_origin) &&
        loom_type_element_type(loom_module_value_type(
            context->module, count_origin.source_value_id)) ==
            LOOM_SCALAR_TYPE_I32;
    if (!has_i32_count_origin) {
      loom_rewriter_t* rewriter = context->rewriter;
      loom_builder_t* builder = &rewriter->builder;
      loom_builder_set_before(builder, op);
      const loom_value_id_t checkpoint =
          loom_rewriter_value_checkpoint(rewriter);
      loom_type_t narrow_type = result_type;
      narrow_type.header = loom_type_make_header(
          loom_type_kind(result_type), LOOM_SCALAR_TYPE_I32,
          loom_type_rank(result_type), loom_type_flags(result_type));
      loom_op_t* truncate_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_trunci_build(builder, count, result_type,
                                                    narrow_type, op->location,
                                                    &truncate_op));
      loom_op_t* extend_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_extui_build(
          builder, loom_vector_trunci_result(truncate_op), narrow_type,
          result_type, op->location, &extend_op));

      loom_op_t* replacement_op = NULL;
      IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
          builder, op->kind, 2, 1, 0, 0, op->instance_flags, op->location,
          &replacement_op));
      loom_op_operands(replacement_op)[0] = loom_op_operands(op)[0];
      loom_op_operands(replacement_op)[1] = loom_vector_extui_result(extend_op);
      IREE_RETURN_IF_ERROR(loom_builder_define_result(
          builder, result_type, &loom_op_results(replacement_op)[0]));
      IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, replacement_op));
      loom_value_id_t replacement = loom_op_results(replacement_op)[0];
      IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
          rewriter, op, &replacement, 1, checkpoint));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
          rewriter, op, &replacement, 1));
      *out_result = (loom_target_legalizer_result_t){
          .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
      };
      return iree_ok_status();
    }
  }
  if (bit_width >= 32) {
    return loom_vector_legalize_descriptor(entry, context, op, out_result);
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  if (bit_width == 16) {
    if (loom_vector_shift_count_is_fixed(rewriter, loom_op_operands(op)[1])) {
      return loom_vector_legalize_descriptor(entry, context, op, out_result);
    }

    loom_type_t predicate_type = result_type;
    predicate_type.header = loom_type_make_header(
        loom_type_kind(result_type), LOOM_SCALAR_TYPE_I1,
        loom_type_rank(result_type), loom_type_flags(result_type));
    loom_op_t* zero_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(0), result_type, op->location, &zero_op));
    loom_value_id_t current = loom_op_operands(op)[0];
    for (int64_t amount = 1; amount < bit_width; amount <<= 1) {
      loom_op_t* amount_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_vector_constant_build(builder, loom_attr_i64(amount),
                                     result_type, op->location, &amount_op));
      loom_op_t* selected_bit_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_vector_andi_build(builder, loom_op_operands(op)[1],
                                 loom_vector_constant_result(amount_op),
                                 result_type, op->location, &selected_bit_op));
      loom_op_t* selected_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
          builder, LOOM_VECTOR_CMPI_PREDICATE_NE,
          loom_vector_andi_result(selected_bit_op),
          loom_vector_constant_result(zero_op), result_type, predicate_type,
          op->location, &selected_op));

      loom_op_t* shifted_op = NULL;
      IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
          builder, op->kind, 2, 1, 0, 0, 0, op->location, &shifted_op));
      loom_op_operands(shifted_op)[0] = current;
      loom_op_operands(shifted_op)[1] = loom_vector_constant_result(amount_op);
      IREE_RETURN_IF_ERROR(loom_builder_define_result(
          builder, result_type, &loom_op_results(shifted_op)[0]));
      IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, shifted_op));
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_select_build(
          builder, loom_vector_cmpi_result(selected_op),
          loom_op_results(shifted_op)[0], current, result_type, op->location,
          &select_op));
      current = loom_vector_select_result(select_op);
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, &current, 1, checkpoint));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_and_erase(rewriter, op, &current, 1));
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
    return iree_ok_status();
  }

  loom_type_t working_type = result_type;
  working_type.header = loom_type_make_header(
      loom_type_kind(result_type), LOOM_SCALAR_TYPE_I16,
      loom_type_rank(result_type), loom_type_flags(result_type));
  const bool working_carrier_supported =
      loom_target_legalization_has_source_vector_carrier(context, working_type,
                                                         LOOM_VALUE_ID_INVALID);
  const uint32_t packet_lane_count =
      working_carrier_supported ? 0
                                : loom_vector_shift_select_packet_lane_count(
                                      context, result_type, working_type);
  if (!working_carrier_supported && packet_lane_count == 0) {
    return loom_vector_legalize_descriptor(entry, context, op, out_result);
  }

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (packet_lane_count == 0) {
    const loom_value_id_t operands[2] = {loom_op_operands(op)[0],
                                         loom_op_operands(op)[1]};
    IREE_RETURN_IF_ERROR(loom_vector_build_widened_shift(
        builder, op, operands, result_type, working_type, &replacement));
  } else {
    const uint32_t source_lane_count =
        (uint32_t)loom_type_dim_static_size_at(result_type, 0);
    const uint32_t packet_count =
        (source_lane_count - 1u) / packet_lane_count + 1u;
    loom_value_id_t* packet_results = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, packet_count,
                                                   sizeof(*packet_results),
                                                   (void**)&packet_results));
    uint32_t lane_offset = 0;
    for (uint32_t i = 0; i < packet_count; ++i) {
      const uint32_t lane_count =
          iree_min(source_lane_count - lane_offset, packet_lane_count);
      const int64_t static_lane_offset = lane_offset;
      const loom_type_t packet_source_type =
          loom_vector_shift_packet_type(result_type, lane_count);
      loom_value_id_t packet_operands[2];
      for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(packet_operands); ++j) {
        loom_op_t* slice_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_slice_build(
            builder, loom_op_operands(op)[j], /*offsets=*/NULL,
            /*offsets_count=*/0, &static_lane_offset,
            /*static_offsets_count=*/1, packet_source_type, op->location,
            &slice_op));
        packet_operands[j] = loom_vector_slice_result(slice_op);
      }
      const loom_type_t packet_working_type =
          loom_vector_shift_packet_type(working_type, lane_count);
      IREE_RETURN_IF_ERROR(loom_vector_build_widened_shift(
          builder, op, packet_operands, packet_source_type, packet_working_type,
          &packet_results[i]));
      lane_offset += lane_count;
    }
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        builder, /*axis=*/0, packet_results, packet_count, result_type,
        op->location, &concat_op));
    replacement = loom_vector_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

// Decomposes a rotation while preserving its vector and logical element
// domains. Target packet contracts lower the resulting shifts and bitwise
// operations without exposing authored lanes as scalar IR.
static iree_status_t loom_vector_legalize_rotation(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t value = loom_op_operands(op)[0];
  const loom_value_id_t count = loom_op_operands(op)[1];
  const loom_type_t type = loom_module_value_type(context->module, value);
  const int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(type));

  loom_value_id_t replacement = value;
  if (bit_width != 1) {
    loom_op_t* mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(bit_width - 1), type, op->location, &mask_op));
    loom_op_t* amount_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_andi_build(
        builder, count, loom_vector_constant_result(mask_op), type,
        op->location, &amount_op));

    loom_op_t* complement_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_xori_build(builder, loom_vector_andi_result(amount_op),
                               loom_vector_constant_result(mask_op), type,
                               op->location, &complement_op));
    loom_op_t* one_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(1), type, op->location, &one_op));

    const loom_value_id_t amount = loom_vector_andi_result(amount_op);
    const loom_value_id_t complement = loom_vector_xori_result(complement_op);
    const loom_value_id_t one = loom_vector_constant_result(one_op);
    const bool rotate_left = loom_vector_rotli_isa(op);
    loom_op_t* complementary_op = NULL;
    if (rotate_left) {
      IREE_RETURN_IF_ERROR(loom_vector_shrui_build(
          builder, value, complement, type, op->location, &complementary_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_shli_build(
          builder, /*instance_flags=*/0, value, complement, type, op->location,
          &complementary_op));
    }
    loom_op_t* left_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_shli_build(
        builder, /*instance_flags=*/0,
        rotate_left ? value : loom_vector_shli_result(complementary_op),
        rotate_left ? amount : one, type, op->location, &left_op));
    loom_op_t* right_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_shrui_build(
        builder,
        rotate_left ? loom_vector_shrui_result(complementary_op) : value,
        rotate_left ? one : amount, type, op->location, &right_op));
    loom_op_t* combine_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_ori_build(
        builder, loom_vector_shli_result(left_op),
        loom_vector_shrui_result(right_op), type, op->location, &combine_op));
    replacement = loom_vector_ori_result(combine_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_non_dense_memory(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_atomic(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_atomic_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_transpose_to_shuffle_rewrite_op(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_transpose_isa(op)) {
    return iree_ok_status();
  }

  const loom_value_id_t source = loom_vector_transpose_source(op);
  const loom_type_t source_type =
      loom_module_value_type(rewriter->module, source);
  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_transpose_result(op));
  uint64_t element_count = 0;
  if (!loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type) ||
      !loom_type_static_element_count(source_type, &element_count) ||
      element_count > UINT16_MAX) {
    return iree_ok_status();
  }

  int64_t* source_lanes = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, (iree_host_size_t)element_count, sizeof(*source_lanes),
        (void**)&source_lanes));
  }

  const loom_attribute_t permutation = loom_vector_transpose_permutation(op);
  bool is_identity = true;
  for (uint64_t result_lane = 0; result_lane < element_count; ++result_lane) {
    const uint64_t source_lane = loom_vector_transpose_source_lane(
        source_type, result_type, permutation.i64_array, result_lane);
    source_lanes[result_lane] = (int64_t)source_lane;
    is_identity &= source_lane == result_lane;
  }

  const loom_type_t flat_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(source_type),
      loom_dim_pack_static((int64_t)element_count), /*encoding_id=*/0);
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t flat_source = source;
  if (!loom_type_equal(source_type, flat_type)) {
    loom_op_t* flatten_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(&rewriter->builder, source,
                                                   source_type, flat_type,
                                                   op->location, &flatten_op));
    flat_source = loom_vector_bitcast_result(flatten_op);
  }

  loom_value_id_t replacement = flat_source;
  if (!is_identity) {
    loom_op_t* shuffle_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_shuffle_build(
        &rewriter->builder, source_lanes, (iree_host_size_t)element_count,
        flat_source, flat_type, op->location, &shuffle_op));
    replacement = loom_vector_shuffle_result(shuffle_op);
  }
  if (!loom_type_equal(flat_type, result_type)) {
    loom_op_t* restore_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_bitcast_build(&rewriter->builder, replacement, flat_type,
                                  result_type, op->location, &restore_op));
    replacement = loom_vector_bitcast_result(restore_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_transpose(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_transpose_to_shuffle_rewrite_op(
      context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static int loom_vector_compare_value_ids(const void* lhs, const void* rhs) {
  const loom_value_id_t lhs_id = *(const loom_value_id_t*)lhs;
  const loom_value_id_t rhs_id = *(const loom_value_id_t*)rhs;
  if (lhs_id < rhs_id) {
    return -1;
  }
  if (lhs_id > rhs_id) {
    return 1;
  }
  return 0;
}

// Selects the most frequent SSA value so the fallback needs the fewest scalar
// insertions. Ties retain lane zero when it is one of the modes; other ties use
// value ID order for deterministic output.
static iree_status_t loom_vector_from_elements_select_seed(
    loom_rewriter_t* rewriter, loom_value_slice_t elements,
    loom_value_id_t* out_seed) {
  loom_value_id_t* sorted_elements = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_value_id_scratch(
      rewriter->fact_table, elements.count, &sorted_elements));
  memcpy(sorted_elements, elements.values,
         elements.count * sizeof(*sorted_elements));
  qsort(sorted_elements, elements.count, sizeof(*sorted_elements),
        loom_vector_compare_value_ids);

  loom_value_id_t seed = elements.values[0];
  iree_host_size_t best_count = 0;
  loom_value_id_t current = sorted_elements[0];
  iree_host_size_t current_count = 1;
  for (iree_host_size_t i = 1; i <= elements.count; ++i) {
    if (i < elements.count && sorted_elements[i] == current) {
      ++current_count;
      continue;
    }
    if (current_count > best_count ||
        (current_count == best_count && current == elements.values[0])) {
      seed = current;
      best_count = current_count;
    }
    if (i < elements.count) {
      current = sorted_elements[i];
      current_count = 1;
    }
  }

  *out_seed = seed;
  return iree_ok_status();
}

iree_status_t loom_vector_from_elements_linearize_rewrite_op(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_from_elements_isa(op)) {
    return iree_ok_status();
  }
  const loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == 0) {
    return iree_ok_status();
  }
  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_from_elements_result(op));
  if (!loom_type_is_vector(result_type) ||
      !loom_type_is_all_static(result_type)) {
    return iree_ok_status();
  }

  const bool restore_shape = loom_type_rank(result_type) != 1;
  const loom_type_t construction_type =
      restore_shape ? loom_type_shaped_1d(
                          LOOM_TYPE_VECTOR, loom_type_element_type(result_type),
                          loom_dim_pack_static((int64_t)elements.count),
                          /*encoding_id=*/0)
                    : result_type;

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t seed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_from_elements_select_seed(rewriter, elements, &seed));
  loom_op_t* splat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      &rewriter->builder, seed, construction_type, op->location, &splat_op));
  loom_value_id_t replacement = loom_vector_splat_result(splat_op);

  for (iree_host_size_t element_index = 0; element_index < elements.count;
       ++element_index) {
    if (elements.values[element_index] == seed) {
      continue;
    }
    const int64_t static_index = (int64_t)element_index;
    loom_op_t* insert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        &rewriter->builder, elements.values[element_index], replacement, NULL,
        0, &static_index, 1, construction_type, op->location, &insert_op));
    replacement = loom_vector_insert_result(insert_op);
  }

  if (restore_shape) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        &rewriter->builder, replacement, construction_type, result_type,
        op->location, &bitcast_op));
    replacement = loom_vector_bitcast_result(bitcast_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

// Expands a variadic rank-one vector constructor into the fixed-arity
// structural ops that targets commonly select for linear register vectors.
// Splatting the first lane provides a defined seed without requiring a
// target-level poison or zero materialization.
static iree_status_t loom_vector_legalize_from_elements(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  // Source cleanup consumes constructor lanes directly. Expanding to inserts
  // earlier hides that sharing and creates transient aggregates to clean up.
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
      context->policy != LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    return iree_ok_status();
  }
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_from_elements_result(op));
  if (loom_type_rank(result_type) != 1) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_linearize_rewrite_op(
      context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_dotf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_dotf_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_transform(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_transform_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_decode(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_decode_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
    return iree_ok_status();
  }
  if (context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
        .source_rejection_bits =
            loom_vector_decode_to_scalar_reference_rejection_bits(
                context->pass, context->rewriter, op),
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_mma(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }
  if (loom_vector_mma_has_fragment_store_user(context->module, op)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  const loom_vector_mma_to_scalar_options_t options =
      loom_vector_mma_options(context);
  IREE_RETURN_IF_ERROR(loom_vector_mma_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, options, &rewritten));
  *out_result = (loom_target_legalizer_result_t){
      .action = rewritten
                    ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                    : LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
      .source_rejection_bits =
          rewritten ? 0
                    : loom_vector_mma_to_scalar_reference_rejection_bits(
                          context->pass, context->rewriter, op, options),
      .source_rejection_detail =
          rewritten ? 0
                    : loom_vector_mma_to_scalar_reference_rejection_detail(
                          context->pass, context->rewriter, op, options),
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_store_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_fragment_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_fragment_store_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  *out_result = (loom_target_legalizer_result_t){
      .action = rewritten
                    ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                    : LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
      .source_rejection_bits =
          rewritten
              ? 0
              : loom_vector_fragment_store_to_scalar_reference_rejection_bits(
                    context->pass, context->rewriter, op),
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_extract_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_predicate_extension(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    bool rewritten = false;
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_predicate_extension(
        context, op, context->vector_packet_policy, &rewritten));
    if (rewritten) {
      out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
      return iree_ok_status();
    }
  }
  loom_value_id_t input = loom_op_operands(op)[0];
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  loom_op_t* true_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &rewriter->builder,
      loom_attr_i64(op->kind == LOOM_OP_VECTOR_EXTSI ? -1 : 1), result_type,
      op->location, &true_op));
  loom_op_t* false_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(&rewriter->builder,
                                                  loom_attr_i64(0), result_type,
                                                  op->location, &false_op));
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(
      &rewriter->builder, input, loom_vector_constant_result(true_op),
      loom_vector_constant_result(false_op), result_type, op->location,
      &select_op));
  loom_value_id_t replacement = loom_vector_select_result(select_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

typedef uint8_t loom_vector_float_extremum_kind_t;
enum loom_vector_float_extremum_kind_e {
  LOOM_VECTOR_FLOAT_EXTREMUM_MINIMUM = 0,
  LOOM_VECTOR_FLOAT_EXTREMUM_MAXIMUM = 1,
};

static iree_status_t loom_vector_legalize_build_select(
    loom_builder_t* builder, loom_location_id_t location, loom_type_t type,
    loom_value_id_t condition, loom_value_id_t true_value,
    loom_value_id_t false_value, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(
      builder, condition, true_value, false_value, type, location, &op));
  *out_value = loom_vector_select_result(op);
  return iree_ok_status();
}

// Builds an exact number-preferring extremum without expanding lanes.
// Ordered comparisons select unequal numeric operands. Equal finite operands
// have identical bits, except for opposing signed zeros: OR selects -0 for
// minimum and AND selects +0 for maximum. The final selects choose the numeric
// peer when exactly one operand is NaN.
static iree_status_t loom_vector_legalize_build_number_extremum(
    loom_builder_t* builder, loom_location_id_t location,
    loom_vector_float_extremum_kind_t kind, uint8_t fastmath,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t type,
    loom_value_id_t* out_result) {
  loom_numeric_float_encoding_t encoding = {0};
  const bool has_encoding =
      loom_numeric_float_encoding(loom_type_element_type(type), &encoding);
  IREE_ASSERT(has_encoding && "floating extremum type must have an encoding");
  (void)has_encoding;
  loom_type_t integer_type = type;
  integer_type.header =
      loom_type_make_header(loom_type_kind(type), encoding.integer_type,
                            loom_type_rank(type), loom_type_flags(type));
  loom_type_t predicate_type = type;
  predicate_type.header =
      loom_type_make_header(loom_type_kind(type), LOOM_SCALAR_TYPE_I1,
                            loom_type_rank(type), loom_type_flags(type));

  loom_op_t* lhs_bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
      builder, lhs, type, integer_type, location, &lhs_bits_op));
  loom_op_t* rhs_bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
      builder, rhs, type, integer_type, location, &rhs_bits_op));
  loom_op_t* tie_bits_op = NULL;
  if (kind == LOOM_VECTOR_FLOAT_EXTREMUM_MINIMUM) {
    IREE_RETURN_IF_ERROR(
        loom_vector_ori_build(builder, loom_vector_bitcast_result(lhs_bits_op),
                              loom_vector_bitcast_result(rhs_bits_op),
                              integer_type, location, &tie_bits_op));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_vector_andi_build(builder, loom_vector_bitcast_result(lhs_bits_op),
                               loom_vector_bitcast_result(rhs_bits_op),
                               integer_type, location, &tie_bits_op));
  }
  loom_op_t* tie_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_bitcast_build(builder, loom_op_results(tie_bits_op)[0],
                                integer_type, type, location, &tie_op));
  const loom_value_id_t tie = loom_vector_bitcast_result(tie_op);

  const loom_vector_cmpf_predicate_t predicate =
      kind == LOOM_VECTOR_FLOAT_EXTREMUM_MINIMUM
          ? LOOM_VECTOR_CMPF_PREDICATE_OLT
          : LOOM_VECTOR_CMPF_PREDICATE_OGT;
  loom_op_t* lhs_ordered_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(builder, fastmath, predicate, lhs,
                                              rhs, type, predicate_type,
                                              location, &lhs_ordered_op));
  loom_op_t* rhs_ordered_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(builder, fastmath, predicate, rhs,
                                              lhs, type, predicate_type,
                                              location, &rhs_ordered_op));
  loom_value_id_t rhs_or_tie = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_legalize_build_select(
      builder, location, type, loom_vector_cmpf_result(rhs_ordered_op), rhs,
      tie, &rhs_or_tie));
  loom_value_id_t numeric = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_legalize_build_select(
      builder, location, type, loom_vector_cmpf_result(lhs_ordered_op), lhs,
      rhs_or_tie, &numeric));

  loom_op_t* lhs_nan_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
      builder, fastmath, LOOM_VECTOR_CMPF_PREDICATE_UNE, lhs, lhs, type,
      predicate_type, location, &lhs_nan_op));
  loom_op_t* rhs_nan_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
      builder, fastmath, LOOM_VECTOR_CMPF_PREDICATE_UNE, rhs, rhs, type,
      predicate_type, location, &rhs_nan_op));
  loom_value_id_t rhs_checked = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_legalize_build_select(
      builder, location, type, loom_vector_cmpf_result(rhs_nan_op), lhs,
      numeric, &rhs_checked));
  return loom_vector_legalize_build_select(builder, location, type,
                                           loom_vector_cmpf_result(lhs_nan_op),
                                           rhs, rhs_checked, out_result);
}

static iree_status_t loom_vector_legalize_build_clamp(
    loom_builder_t* builder, loom_op_t* op, loom_type_t type,
    loom_value_id_t* out_result) {
  loom_type_t predicate_type = type;
  predicate_type.header =
      loom_type_make_header(loom_type_kind(type), LOOM_SCALAR_TYPE_I1,
                            loom_type_rank(type), loom_type_flags(type));
  const loom_value_id_t value = loom_vector_clampf_value(op);
  const loom_value_id_t lower = loom_vector_clampf_lower(op);
  const loom_value_id_t upper = loom_vector_clampf_upper(op);
  const uint8_t fastmath = op->instance_flags;
  loom_value_id_t lower_bounded = LOOM_VALUE_ID_INVALID;
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  switch (loom_vector_clampf_mode(op)) {
    case LOOM_VECTOR_CLAMPF_MODE_ORDERED: {
      loom_op_t* below_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
          builder, fastmath, LOOM_VECTOR_CMPF_PREDICATE_OLT, value, lower, type,
          predicate_type, op->location, &below_op));
      IREE_RETURN_IF_ERROR(loom_vector_legalize_build_select(
          builder, op->location, type, loom_vector_cmpf_result(below_op), lower,
          value, &lower_bounded));
      loom_op_t* above_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
          builder, fastmath, LOOM_VECTOR_CMPF_PREDICATE_OGT, lower_bounded,
          upper, type, predicate_type, op->location, &above_op));
      IREE_RETURN_IF_ERROR(loom_vector_legalize_build_select(
          builder, op->location, type, loom_vector_cmpf_result(above_op), upper,
          lower_bounded, &replacement));
      break;
    }
    case LOOM_VECTOR_CLAMPF_MODE_NUMBER: {
      IREE_RETURN_IF_ERROR(loom_vector_legalize_build_number_extremum(
          builder, op->location, LOOM_VECTOR_FLOAT_EXTREMUM_MAXIMUM, fastmath,
          value, lower, type, &lower_bounded));
      IREE_RETURN_IF_ERROR(loom_vector_legalize_build_number_extremum(
          builder, op->location, LOOM_VECTOR_FLOAT_EXTREMUM_MINIMUM, fastmath,
          lower_bounded, upper, type, &replacement));
      break;
    }
    case LOOM_VECTOR_CLAMPF_MODE_IEEE: {
      loom_op_t* maximum_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_maximumf_build(
          builder, fastmath, value, lower, type, op->location, &maximum_op));
      loom_op_t* minimum_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_minimumf_build(
          builder, fastmath, loom_vector_maximumf_result(maximum_op), upper,
          type, op->location, &minimum_op));
      replacement = loom_vector_minimumf_result(minimum_op);
      break;
    }
    case LOOM_VECTOR_CLAMPF_MODE_COUNT_:
      IREE_ASSERT_UNREACHABLE("verified vector clamp has an invalid mode");
      IREE_BUILTIN_UNREACHABLE();
  }
  *out_result = replacement;
  return iree_ok_status();
}

// Number-preferring extrema already implement signed-zero ordering. An
// unordered comparison adds the IEEE NaN policy without changing numeric
// operands or exposing a NaN payload guarantee.
static iree_status_t loom_vector_legalize_build_ieee_extremum(
    loom_builder_t* builder, loom_op_t* op, loom_type_t type,
    loom_value_id_t* out_result) {
  const loom_value_id_t lhs = loom_op_operands(op)[0];
  const loom_value_id_t rhs = loom_op_operands(op)[1];
  loom_type_t predicate_type = type;
  predicate_type.header =
      loom_type_make_header(loom_type_kind(type), LOOM_SCALAR_TYPE_I1,
                            loom_type_rank(type), loom_type_flags(type));

  loom_op_t* number = NULL;
  if (loom_vector_minimumf_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_minnumf_build(
        builder, op->instance_flags, lhs, rhs, type, op->location, &number));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_maxnumf_build(
        builder, op->instance_flags, lhs, rhs, type, op->location, &number));
  }
  loom_op_t* unordered = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
      builder, op->instance_flags, LOOM_VECTOR_CMPF_PREDICATE_UNO, lhs, rhs,
      type, predicate_type, op->location, &unordered));
  loom_op_t* nan = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(builder, loom_attr_f64(NAN),
                                                  type, op->location, &nan));
  loom_op_t* select = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(
      builder, loom_vector_cmpf_result(unordered),
      loom_vector_constant_result(nan), loom_op_results(number)[0], type,
      op->location, &select));
  *out_result = loom_vector_select_result(select);
  return iree_ok_status();
}

iree_status_t loom_vector_float_extrema_rewrite_op(loom_rewriter_t* rewriter,
                                                   loom_op_t* op,
                                                   bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_minnumf_isa(op) && !loom_vector_maxnumf_isa(op) &&
      !loom_vector_minimumf_isa(op) && !loom_vector_maximumf_isa(op) &&
      !loom_vector_clampf_isa(op)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t type =
      loom_module_value_type(rewriter->module, loom_op_results(op)[0]);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (loom_vector_minnumf_isa(op) || loom_vector_maxnumf_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_legalize_build_number_extremum(
        builder, op->location,
        loom_vector_minnumf_isa(op) ? LOOM_VECTOR_FLOAT_EXTREMUM_MINIMUM
                                    : LOOM_VECTOR_FLOAT_EXTREMUM_MAXIMUM,
        op->instance_flags, loom_op_operands(op)[0], loom_op_operands(op)[1],
        type, &replacement));
  } else if (loom_vector_clampf_isa(op)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_legalize_build_clamp(builder, op, type, &replacement));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_legalize_build_ieee_extremum(
        builder, op, type, &replacement));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

static bool loom_vector_legalize_f32_aggregate_is_supported(
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  return !context->source_function_has_unsupported_vector_carrier &&
         loom_target_legalization_op_has_source_vector_carriers(context, op);
}

iree_status_t loom_vector_legalize_f32_extrema(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_vector_legalize_f32_aggregate_is_supported(context, op)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_float_extrema_rewrite_op(context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_float_classification(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_value_id_t input = loom_op_operands(op)[0];
  const loom_type_t float_type = loom_module_value_type(context->module, input);
  loom_numeric_float_encoding_t encoding = {0};
  if (!loom_numeric_float_encoding(loom_type_element_type(float_type),
                                   &encoding)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (op->kind == LOOM_OP_VECTOR_ISINFF &&
      encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_FINITE_NAN) {
    loom_op_t* false_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(0), result_type, op->location, &false_op));
    replacement = loom_vector_constant_result(false_op);
  } else {
    loom_type_t integer_type = float_type;
    integer_type.header = loom_type_make_header(
        loom_type_kind(float_type), encoding.integer_type,
        loom_type_rank(float_type), loom_type_flags(float_type));
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder, input, float_type, integer_type, op->location, &bitcast_op));
    loom_op_t* magnitude_mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64((int64_t)encoding.magnitude_mask), integer_type,
        op->location, &magnitude_mask_op));
    loom_op_t* magnitude_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_andi_build(builder, loom_vector_bitcast_result(bitcast_op),
                               loom_vector_constant_result(magnitude_mask_op),
                               integer_type, op->location, &magnitude_op));
    loom_op_t* special_magnitude_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64((int64_t)encoding.special_magnitude),
        integer_type, op->location, &special_magnitude_op));

    loom_vector_cmpi_predicate_t predicate = LOOM_VECTOR_CMPI_PREDICATE_EQ;
    if (op->kind == LOOM_OP_VECTOR_ISNANF) {
      predicate =
          encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_IEEE
              ? LOOM_VECTOR_CMPI_PREDICATE_UGT
              : LOOM_VECTOR_CMPI_PREDICATE_EQ;
    } else if (op->kind == LOOM_OP_VECTOR_ISFINITEF) {
      predicate =
          encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_IEEE
              ? LOOM_VECTOR_CMPI_PREDICATE_ULT
              : LOOM_VECTOR_CMPI_PREDICATE_NE;
    }
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
        builder, predicate, loom_vector_andi_result(magnitude_op),
        loom_vector_constant_result(special_magnitude_op), integer_type,
        result_type, op->location, &comparison_op));
    replacement = loom_vector_cmpi_result(comparison_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

// Projects independent payload components without splitting their convergence
// point. Unlike pure operation packetization, this keeps one collective and
// one synchronization protocol for every authored result.
static iree_status_t loom_vector_legalize_workgroup_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->vector_packet_policy == NULL ||
      context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  const loom_value_slice_t sources = loom_kernel_workgroup_reduce_values(op);
  loom_vector_packet_shape_t* shapes = NULL;
  iree_host_size_t packet_count = sources.count;
  bool projection_supported = true;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < sources.count && iree_status_is_ok(status);
       ++i) {
    const loom_type_t source_type =
        loom_module_value_type(context->module, sources.values[i]);
    if (!loom_type_is_vector(source_type) ||
        loom_target_legalization_has_source_vector_carrier(context, source_type,
                                                           sources.values[i])) {
      continue;
    }
    loom_vector_packet_shape_t shape;
    if (!loom_vector_packet_shape_from_type(
            context, context->vector_packet_policy, source_type, &shape)) {
      projection_supported = false;
      break;
    }
    packet_count += shape.chunk_count - 1;
    // Source operations have 16-bit operand/result counts. Leave an
    // unrepresentable projection for the ordinary target support diagnostic.
    if (packet_count > UINT16_MAX) {
      projection_supported = false;
      break;
    }
    if (!shapes) {
      status = iree_arena_allocate_array(context->arena, sources.count,
                                         sizeof(*shapes), (void**)&shapes);
      if (iree_status_is_ok(status)) {
        memset(shapes, 0, sources.count * sizeof(*shapes));
      }
    }
    if (iree_status_is_ok(status)) {
      shapes[i] = shape;
    }
  }
  if (!iree_status_is_ok(status) || !projection_supported || !shapes) {
    return status;
  }

  loom_value_id_t* values = NULL;
  loom_type_t* result_types = NULL;
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, packet_count, sizeof(*values), (void**)&values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, packet_count,
                                                 sizeof(*result_types),
                                                 (void**)&result_types));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, sources.count,
                                                 sizeof(*replacements),
                                                 (void**)&replacements));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  iree_host_size_t packet_index = 0;
  for (iree_host_size_t i = 0; i < sources.count && iree_status_is_ok(status);
       ++i) {
    const loom_type_t source_type =
        loom_module_value_type(context->module, sources.values[i]);
    const loom_vector_packet_shape_t shape = shapes[i];
    if (shape.chunk_count == 0) {
      values[packet_index] = sources.values[i];
      result_types[packet_index++] = source_type;
      continue;
    }
    for (uint32_t j = 0; j < shape.chunk_count && iree_status_is_ok(status);
         ++j) {
      const int64_t offset = j * shape.chunk_lane_count;
      const uint32_t lane_count =
          iree_min(shape.chunk_lane_count, shape.lane_count - offset);
      result_types[packet_index] =
          loom_vector_packet_type(source_type, lane_count);
      loom_op_t* slice = NULL;
      status = loom_vector_slice_build(builder, sources.values[i], NULL, 0,
                                       &offset, 1, result_types[packet_index],
                                       op->location, &slice);
      if (iree_status_is_ok(status)) {
        values[packet_index++] = loom_vector_slice_result(slice);
      }
    }
  }
  loom_op_t* reduction = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_kernel_workgroup_reduce_build(
        builder, loom_kernel_workgroup_reduce_kind(op), op->instance_flags,
        values, packet_count, result_types, packet_count, op->location,
        &reduction);
  }
  packet_index = 0;
  for (iree_host_size_t i = 0; i < sources.count && iree_status_is_ok(status);
       ++i) {
    const loom_value_id_t* results =
        loom_op_const_results(reduction) + packet_index;
    if (shapes[i].chunk_count == 0) {
      replacements[i] = *results;
      ++packet_index;
      continue;
    }
    loom_op_t* assembled = NULL;
    status = loom_vector_concat_build(
        builder, 0, results, shapes[i].chunk_count,
        loom_module_value_type(context->module, sources.values[i]),
        op->location, &assembled);
    if (iree_status_is_ok(status)) {
      replacements[i] = loom_vector_concat_result(assembled);
      packet_index += shapes[i].chunk_count;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, replacements, sources.count, checkpoint);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_replace_all_uses_and_erase(
        rewriter, op, replacements, sources.count);
  }
  if (iree_status_is_ok(status)) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return status;
}

static const loom_target_legalizer_rule_t kVectorLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_KERNEL_WORKGROUP_REDUCE,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .legalize = loom_vector_legalize_workgroup_reduce,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD_MASK,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE_MASK,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD_EXPAND,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE_COMPRESS,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CLAMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED,
        .legalize = loom_vector_legalize_f32_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED,
        .legalize = loom_vector_legalize_f32_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXNUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED,
        .legalize = loom_vector_legalize_f32_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_f32_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_f32_extrema,
    },
    // Combining operations share one complete reference family. Native target
    // contracts retain precedence; explicit rejection selects the ordinary
    // scalar lane recipe. Keep this list aligned with loom_combining_kind_t.
    {
        .root_kind = LOOM_OP_VECTOR_ADDI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ADDF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MULI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MULF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINSI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXSI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINUI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXUI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ANDI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ORI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_XORI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINIMUMF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXIMUMF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINNUMF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXNUMF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ISNANF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_vector_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ISINFF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_vector_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ISFINITEF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_vector_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_VECTOR_GATHER,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_GATHER_MASK,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SCATTER,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SCATTER_MASK,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_REDUCE,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_RMW,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_RMW_MASK,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_CMPXCHG,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_vector_legalize_predicate_extension,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_vector_legalize_predicate_extension,
    },
// Every cast has a generated or explicit lane program. Keep this family tied
// to the dialect definition so adding a cast cannot silently omit its target
// reference fallback. Target-native contracts and the predicate-extension
// specializations above retain precedence over these rejected-contract rows.
#define LOOM_VECTOR_CAST_LEGALIZER_ROW(op_kind)                             \
  {                                                                         \
      .root_kind = (op_kind),                                               \
      .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION, \
      .legalize = loom_vector_legalize_cast,                                \
  },
#include "loom/ops/vector/cast_legalizer_rows.inl"
#undef LOOM_VECTOR_CAST_LEGALIZER_ROW
    {
        .root_kind = LOOM_OP_SCF_SELECT,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FROM_ELEMENTS,
        .legalize = loom_vector_legalize_from_elements,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MASK_RANGE,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSPOSE,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_transpose,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHUFFLE,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CMPI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CMPF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_vector_legalize_reduce,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE_AXES,
        .legalize = loom_vector_legalize_reduce_axes,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
        .legalize = loom_vector_legalize_bitfield_extract,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
        .legalize = loom_vector_legalize_bitfield_extract,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_INSERT,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITPACK,
        .legalize = loom_vector_legalize_bitstream,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKU,
        .legalize = loom_vector_legalize_bitstream,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKS,
        .legalize = loom_vector_legalize_bitstream,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_QUANTIZE,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_LOOKUP,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DIVSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DIVUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DIVREMSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DIVREMUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REMSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REMUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CEILDIVSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CEILDIVUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FLOORDIVSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHLI,
        .legalize = loom_vector_legalize_shift,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHRSI,
        .legalize = loom_vector_legalize_shift,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHRUI,
        .legalize = loom_vector_legalize_shift,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ROTLI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .legalize = loom_vector_legalize_rotation,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ROTRI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .legalize = loom_vector_legalize_rotation,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CTLZI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CTTZI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CTPOPI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOTF,
        .legalize = loom_vector_legalize_dotf,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSFORM,
        .legalize = loom_vector_legalize_transform,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT2F,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4I,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT8I4,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4F8,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DECODE,
        .legalize = loom_vector_legalize_decode,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MMA,
        .legalize = loom_vector_legalize_mma,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_vector_legalize_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FRAGMENT_STORE,
        .legalize = loom_vector_legalize_fragment_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTRACT,
        .legalize = loom_vector_legalize_extract,
    },
};

static const loom_target_legalizer_provider_t kVectorLegalizerProvider = {
    .name = IREE_SVL("vector"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kVectorLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kVectorLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_vector_target_legalizer_provider(
    void) {
  return &kVectorLegalizerProvider;
}
