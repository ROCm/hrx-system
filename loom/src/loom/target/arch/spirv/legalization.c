// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/legalization.h"

#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/transforms/kernel/target_legalization.h"
#include "loom/transforms/scalar/target_legalization.h"
#include "loom/transforms/vector/packet_insert_legalization.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/shape_legalization.h"
#include "loom/transforms/vector/target_legalization.h"
#include "loom/transforms/vector/to_scalar.h"

static bool loom_spirv_legalizer_descriptor_set_is_spirv(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id ==
             loom_spirv_logical_core_descriptor_set()->target_stable_id;
}

static loom_type_t loom_spirv_subgroup_packet_type(
    loom_scalar_type_t element_type, uint32_t lane_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                             loom_dim_pack_static(lane_count), 0);
}

static iree_status_t loom_spirv_build_subgroup_exchange(
    loom_builder_t* builder, const loom_op_t* source_op, loom_value_id_t value,
    loom_type_t type, loom_op_t** out_op) {
  switch (source_op->kind) {
    case LOOM_OP_KERNEL_SUBGROUP_SHUFFLE:
      return loom_kernel_subgroup_shuffle_build(
          builder, loom_kernel_subgroup_shuffle_mode(source_op), value,
          loom_kernel_subgroup_shuffle_offset(source_op),
          loom_kernel_subgroup_shuffle_width(source_op), type,
          source_op->location, out_op);
    case LOOM_OP_KERNEL_SUBGROUP_BROADCAST:
      return loom_kernel_subgroup_broadcast_build(
          builder, value, loom_kernel_subgroup_broadcast_lane(source_op), type,
          source_op->location, out_op);
    case LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST:
      return loom_kernel_subgroup_broadcast_first_build(
          builder, value, type, source_op->location, out_op);
    default:
      IREE_ASSERT_UNREACHABLE("expected a subgroup transport operation");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_spirv_legalize_subgroup_payload(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }
  const loom_value_id_t source = loom_op_operands(op)[0];
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const bool is_vector = loom_type_is_vector(source_type);
  if (is_vector && loom_type_dim_is_dynamic_at(source_type, 0)) {
    return iree_ok_status();
  }
  const int64_t source_lane_count =
      is_vector ? loom_type_dim_static_size_at(source_type, 0) : 1;
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  const bool is_address = element_type == LOOM_SCALAR_TYPE_INDEX ||
                          element_type == LOOM_SCALAR_TYPE_OFFSET;
  const bool native_bfloat =
      element_type == LOOM_SCALAR_TYPE_BF16 &&
      iree_all_bits_set(loom_target_legalization_context_bundle(context)
                            ->config->contract_feature_bits,
                        LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR);
  // Address packets preserve their target-selected width. Native BF16 packets
  // use the descriptor's pair packing, which needs no Int16 capability.
  const bool preserve_element_type = is_address || native_bfloat;
  const uint32_t element_bits = loom_scalar_type_bitwidth(element_type);
  if (loom_kernel_subgroup_shuffle_isa(op)) {
    // Word-sized and wider native payloads need no integer repacking. Narrow
    // software payloads retain their packed-word transport to amortize lane
    // exchanges. Both paths normalize addressing once before splitting.
    if (preserve_element_type || element_bits >= 32) {
      IREE_RETURN_IF_ERROR(loom_kernel_target_legalize_subgroup_packets(
          entry, context, op, out_result));
      if (out_result->action == LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN) {
        return iree_ok_status();
      }
    }
    if (loom_kernel_match_subgroup_shuffle(entry, context, op)) {
      return loom_kernel_target_legalize_subgroup_shuffle(entry, context, op,
                                                          out_result);
    }
  }
  // Rejected native packets have no further payload representation.
  if ((element_type == LOOM_SCALAR_TYPE_I32 || preserve_element_type) &&
      source_lane_count <= 4) {
    return iree_ok_status();
  }
  if (source_lane_count <= 0) {
    return iree_ok_status();
  }
  const uint32_t packet_lanes =
      preserve_element_type ? 4u : 128u / element_bits;
  const uint64_t packet_count =
      ((uint64_t)source_lane_count - 1u) / packet_lanes + 1u;
  // The resulting concatenation has one operand per packet.
  if (packet_count > UINT16_MAX) {
    return iree_ok_status();
  }
  loom_value_id_t single_packet = LOOM_VALUE_ID_INVALID;
  loom_value_id_t* packets = &single_packet;
  if (packet_count > 1) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, packet_count, sizeof(*packets), (void**)&packets));
  }
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t valid = LOOM_VALUE_ID_INVALID;
  const int64_t zero = 0;
  for (uint32_t i = 0; i < packet_count; ++i) {
    const int64_t offset = (int64_t)i * packet_lanes;
    const uint32_t lanes = iree_min(source_lane_count - offset, packet_lanes);
    const uint32_t word_count = (lanes * element_bits + 31u) / 32u;
    const uint32_t padded_lanes =
        preserve_element_type ? lanes : word_count * 32u / element_bits;
    const loom_type_t packet_type =
        loom_spirv_subgroup_packet_type(element_type, lanes);
    const loom_type_t padded_type =
        loom_spirv_subgroup_packet_type(element_type, padded_lanes);
    const loom_type_t carrier_type =
        preserve_element_type
            ? packet_type
            : loom_spirv_subgroup_packet_type(LOOM_SCALAR_TYPE_I32, word_count);
    loom_value_id_t packet = source;
    loom_op_t* current = NULL;
    if (!is_vector) {
      IREE_RETURN_IF_ERROR(loom_vector_splat_build(builder, source, packet_type,
                                                   op->location, &current));
      packet = loom_vector_splat_result(current);
    } else if (packet_count > 1) {
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(builder, source, NULL, 0,
                                                   &offset, 1, packet_type,
                                                   op->location, &current));
      packet = loom_vector_slice_result(current);
    }
    if (lanes != padded_lanes) {
      // Padding repeats captured bits; it neither reads past a source memory
      // footprint nor contributes to the observable result.
      IREE_RETURN_IF_ERROR(loom_vector_extract_build(
          builder, packet, NULL, 0, &zero, 1, loom_type_scalar(element_type),
          op->location, &current));
      const loom_value_id_t fill = loom_vector_extract_result(current);
      IREE_RETURN_IF_ERROR(loom_vector_splat_build(
          builder, fill,
          loom_spirv_subgroup_packet_type(element_type, padded_lanes - lanes),
          op->location, &current));
      const loom_value_id_t pieces[] = {packet,
                                        loom_vector_splat_result(current)};
      IREE_RETURN_IF_ERROR(
          loom_vector_concat_build(builder, 0, pieces, IREE_ARRAYSIZE(pieces),
                                   padded_type, op->location, &current));
      packet = loom_vector_concat_result(current);
    }
    if (!preserve_element_type) {
      IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
          builder, packet, padded_type, carrier_type, op->location, &current));
      packet = loom_vector_bitcast_result(current);
    }
    IREE_RETURN_IF_ERROR(loom_spirv_build_subgroup_exchange(
        builder, op, packet, carrier_type, &current));
    if (i == 0 && loom_kernel_subgroup_shuffle_isa(op)) {
      valid = loom_kernel_subgroup_shuffle_valid(current);
    }
    packet = loom_op_results(current)[0];
    if (!preserve_element_type) {
      IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
          builder, packet, carrier_type, padded_type, op->location, &current));
      packet = loom_vector_bitcast_result(current);
    }
    if (!is_vector) {
      IREE_RETURN_IF_ERROR(loom_vector_extract_build(builder, packet, NULL, 0,
                                                     &zero, 1, source_type,
                                                     op->location, &current));
      packet = loom_vector_extract_result(current);
    } else if (lanes != padded_lanes) {
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(builder, packet, NULL, 0,
                                                   &zero, 1, packet_type,
                                                   op->location, &current));
      packet = loom_vector_slice_result(current);
    }
    packets[i] = packet;
  }
  loom_value_id_t replacements[2] = {packets[0], valid};
  if (packet_count > 1) {
    loom_op_t* concat = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        builder, 0, packets, packet_count, source_type, op->location, &concat));
    replacements[0] = loom_vector_concat_result(concat);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, op->result_count));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_spirv_legalize_vector_to_scalar(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  // Capture every lane at the source operation. For reads, delaying this
  // rewrite until an arithmetic consumer would change the snapshot across an
  // aliasing write. FP8 vector extension must also become scalar before SPIR-V
  // value mapping because the target has no ordinary FP8 vector representation.
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static bool loom_spirv_type_has_ordinary_vector_carrier(loom_type_t type) {
  loom_spirv_value_type_t value_type = {0};
  if (!loom_spirv_value_type_from_loom_type(type, &value_type)) {
    return false;
  }
  return value_type.value_class == LOOM_SPIRV_VALUE_CLASS_VECTOR ||
         value_type.value_class == LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR;
}

static iree_status_t loom_spirv_legalize_vector_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_store(
      context, op, context->vector_packet_policy, &rewritten));
  if (!rewritten && context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    const loom_type_t value_type =
        loom_module_value_type(context->module, loom_vector_store_value(op));
    if (loom_spirv_type_has_ordinary_vector_carrier(value_type)) {
      IREE_RETURN_IF_ERROR(loom_vector_store_captured_to_scalar_rewrite_op(
          context->pass, context->rewriter, op, &rewritten));
    }
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_legalize_vector_concat(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }

  // Binary concat descriptors cover native ordinary vectors. Leave wider and
  // otherwise unsupported source vectors intact for the fallback legalizer.
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_concat_result(op));
  loom_spirv_value_type_t result_value_type;
  if (!loom_spirv_value_type_from_loom_type(result_type, &result_value_type)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_variadic_concat_to_binary_rewrite_op(
      context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_legalize_vector_slice(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  if (!rewritten && loom_vector_slice_offsets(op).count != 0) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_dynamic_slice(
        context, op, context->vector_packet_policy,
        /*consumer_materializer=*/NULL, &rewritten));
  }
  if (!rewritten && context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  } else if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_legalize_float8_to_bfloat_extension(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }
  (void)entry;
  IREE_RETURN_IF_ERROR(loom_scalar_rewrite_float8_extension(context, op));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_spirv_legalize_vector_insert(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_insert(
      context, op, context->vector_packet_policy, NULL, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

// SPV_KHR_bfloat16 supplies transport, not collective arithmetic. Converting
// at the boundary reuses the F32 native tree and its integer-key extrema,
// including their signed-zero and NaN policies. Exact BF16 conversions own
// native versus software carriers; the collective never needs that distinction.
static iree_status_t loom_spirv_legalize_bfloat_collective(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }
  loom_combining_kind_t kind;
  if (loom_kernel_workgroup_reduce_isa(op)) {
    kind = loom_kernel_workgroup_reduce_kind(op);
  } else if (loom_kernel_subgroup_scan_isa(op)) {
    kind = loom_kernel_subgroup_scan_kind(op);
  } else {
    kind = loom_kernel_subgroup_reduce_kind(op);
  }
  // Native intermediate precision permits F32 accumulation. Preservation
  // instead requires BF16 rounding at every combine, not just at the result.
  if (iree_any_bit_set(op->instance_flags,
                       LOOM_KERNEL_COLLECTIVEMATHFLAGS_PRESERVE) &&
      (kind == LOOM_COMBINING_KIND_ADDF || kind == LOOM_COMBINING_KIND_MULF)) {
    return iree_ok_status();
  }
  uint16_t first_bfloat = 0;
  for (; first_bfloat < op->operand_count; ++first_bfloat) {
    if (loom_type_element_type(loom_module_value_type(
            context->module, loom_op_const_operands(op)[first_bfloat])) ==
        LOOM_SCALAR_TYPE_BF16) {
      break;
    }
  }
  if (first_bfloat == op->operand_count) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* collective = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, op->kind, op->operand_count, op->result_count, 0, 0,
      op->attribute_count, op->location, &collective));
  collective->instance_flags = op->instance_flags;
  memcpy(loom_op_attrs(collective), loom_op_const_attrs(op),
         op->attribute_count * sizeof(loom_attribute_t));
  // All three source operations have positionally corresponding payloads and
  // results. Retain one operation, including mixed workgroup payloads, rather
  // than introducing a separate synchronization protocol for each BF16 input.
  loom_builder_set_before(builder, collective);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < op->operand_count && iree_status_is_ok(status);
       ++i) {
    loom_value_id_t value = loom_op_const_operands(op)[i];
    const loom_type_t type = loom_module_value_type(context->module, value);
    loom_type_t working_type = type;
    if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_BF16) {
      working_type =
          loom_type_is_scalar(type)
              ? loom_type_scalar(LOOM_SCALAR_TYPE_F32)
              : loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                                    loom_type_dim(type, 0), 0);
      loom_op_t* extension = NULL;
      status =
          loom_type_is_scalar(type)
              ? loom_scalar_extf_build(builder, 0, value, type, working_type,
                                       op->location, &extension)
              : loom_vector_extf_build(builder, 0, value, type, working_type,
                                       op->location, &extension);
      if (iree_status_is_ok(status)) {
        value = loom_op_results(extension)[0];
      }
    }
    if (iree_status_is_ok(status)) {
      loom_op_operands(collective)[i] = value;
      status = loom_builder_define_result(builder, working_type,
                                          &loom_op_results(collective)[i]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, collective));
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  loom_builder_set_before(builder, op);
  for (uint16_t i = 0; i < op->result_count && iree_status_is_ok(status); ++i) {
    replacements[i] = loom_op_results(collective)[i];
    const loom_type_t type =
        loom_module_value_type(context->module, loom_op_const_results(op)[i]);
    if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_BF16) {
      continue;
    }
    const loom_type_t working_type =
        loom_module_value_type(context->module, replacements[i]);
    loom_op_t* truncation = NULL;
    status = loom_type_is_scalar(type)
                 ? loom_scalar_fptrunc_build(builder, 0, replacements[i],
                                             working_type, type, op->location,
                                             &truncation)
                 : loom_vector_fptrunc_build(builder, 0, replacements[i],
                                             working_type, type, op->location,
                                             &truncation);
    if (iree_status_is_ok(status)) {
      replacements[i] = loom_op_results(truncation)[0];
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, op->result_count));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kSpirvLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_REDUCE,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_BF16,
        .legalize = loom_spirv_legalize_bfloat_collective,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SCAN,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_BF16,
        .legalize = loom_spirv_legalize_bfloat_collective,
    },
    {
        .root_kind = LOOM_OP_KERNEL_WORKGROUP_REDUCE,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .legalize = loom_spirv_legalize_bfloat_collective,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SHUFFLE,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_spirv_legalize_subgroup_payload,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_BROADCAST,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_spirv_legalize_subgroup_payload,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_spirv_legalize_subgroup_payload,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INSERT,
        .legalize = loom_spirv_legalize_vector_insert,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTF,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2,
        .match = loom_scalar_match_float8_to_bfloat_extension,
        .legalize = loom_spirv_legalize_float8_to_bfloat_extension,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD,
        .legalize = loom_spirv_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_spirv_legalize_vector_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CONCAT,
        .legalize = loom_spirv_legalize_vector_concat,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SLICE,
        .legalize = loom_spirv_legalize_vector_slice,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTF,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2,
        .legalize = loom_spirv_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITCAST,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_spirv_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FPTRUNC,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_spirv_legalize_vector_to_scalar,
    },
};

const loom_target_legalizer_provider_t
    loom_spirv_target_legalizer_provider_storage = {
        .name = IREE_SVL("spirv"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .rules = kSpirvLegalizerRules,
        .rule_count = IREE_ARRAYSIZE(kSpirvLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_spirv_target_legalizer_provider(
    void) {
  return &loom_spirv_target_legalizer_provider_storage;
}
