// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/legalization.h"

#include <stdint.h>
#include <string.h>

#include "loom/analysis/control_uniformity.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/atomic.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/atomic_ordering.h"
#include "loom/target/arch/amdgpu/lower/collective_combine.h"
#include "loom/target/arch/amdgpu/lower/collective_payload.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/encoding/vector_conversion.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/structural.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/table.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/vector_transform.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/transforms/kernel/target_legalization.h"
#include "loom/transforms/scalar/target_legalization.h"
#include "loom/transforms/vector/cast_packet_legalization.h"
#include "loom/transforms/vector/packet_insert_legalization.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/shape_legalization.h"
#include "loom/transforms/vector/table_legalization.h"
#include "loom/transforms/vector/target_legalization.h"
#include "loom/transforms/vector/to_scalar.h"
#include "loom/transforms/view/atomic.h"

static bool loom_amdgpu_match_f32_clamp_decomposition(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  (void)context;
  if (loom_scalar_clampf_isa(op)) {
    const loom_scalar_clampf_mode_t mode = loom_scalar_clampf_mode(op);
    return mode == LOOM_SCALAR_CLAMPF_MODE_NUMBER ||
           mode == LOOM_SCALAR_CLAMPF_MODE_IEEE;
  }
  if (!loom_vector_clampf_isa(op)) {
    return false;
  }
  const loom_vector_clampf_mode_t mode = loom_vector_clampf_mode(op);
  return mode == LOOM_VECTOR_CLAMPF_MODE_NUMBER ||
         mode == LOOM_VECTOR_CLAMPF_MODE_IEEE;
}

static bool loom_amdgpu_match_subgroup_shuffle_reference_lowering(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  const uint32_t wavefront_size = loom_amdgpu_target_wavefront_size(
      loom_target_legalization_context_bundle(context));
  const uint32_t native_width = loom_amdgpu_target_native_subgroup_width(
      loom_amdgpu_target_facts_cast(context->target_facts), wavefront_size);
  // The reference normal form uses a full physical subgroup exchange. A wider
  // source wave on a partitioned target still requires cross-partition support.
  return loom_amdgpu_wavefront_size_is_valid(wavefront_size) &&
         native_width == wavefront_size &&
         !loom_amdgpu_subgroup_shuffle_addressing_is_native(
             context->module, context->fact_table, op, wavefront_size,
             native_width);
}

static bool loom_amdgpu_arithmetic_collective_is_native(
    const loom_target_legalization_context_t* context,
    loom_value_slice_t values, loom_combining_kind_t kind) {
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    loom_amdgpu_subgroup_payload_kind_t payload_kind =
        LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
    uint32_t register_count = 0;
    loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_collective_arithmetic_payload_is_supported(
            context->module, values.values[i], &payload_kind,
            &register_count) ||
        !loom_amdgpu_collective_combine_descriptor_ref(
            kind, payload_kind, &combine_descriptor_ref)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_subgroup_collective_is_native(
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  loom_combining_kind_t kind = LOOM_COMBINING_KIND_COUNT_;
  if (loom_kernel_subgroup_reduce_isa(op)) {
    value = loom_kernel_subgroup_reduce_value(op);
    kind = loom_kernel_subgroup_reduce_kind(op);
  } else if (loom_kernel_subgroup_scan_isa(op)) {
    value = loom_kernel_subgroup_scan_value(op);
    kind = loom_kernel_subgroup_scan_kind(op);
  } else {
    return false;
  }
  return loom_amdgpu_arithmetic_collective_is_native(
      context, (loom_value_slice_t){.values = &value, .count = 1}, kind);
}

static bool loom_amdgpu_subgroup_execution_is_uniform(
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  if (context->fact_table == NULL) {
    return false;
  }
  loom_control_uniformity_info_t control_uniformity;
  loom_control_uniformity_info_initialize(context->module, context->fact_table,
                                          context->arena, &control_uniformity);
  return loom_control_uniformity_prove_execution(
      &control_uniformity, op, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP, NULL);
}

static iree_status_t loom_amdgpu_legalize_subgroup_reduce_reference_lowering(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (loom_amdgpu_subgroup_collective_is_native(context, op) &&
      loom_amdgpu_subgroup_reduce_shape_is_native(
          context->module, context->function,
          loom_target_legalization_context_bundle(context), op) &&
      loom_amdgpu_subgroup_execution_is_uniform(context, op)) {
    return iree_ok_status();
  }
  return loom_kernel_target_legalize_subgroup_reduce(entry, context, op,
                                                     out_result);
}

static iree_status_t loom_amdgpu_legalize_subgroup_scan_reference_lowering(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (loom_amdgpu_subgroup_collective_is_native(context, op) &&
      loom_amdgpu_subgroup_scan_shape_is_native(
          context->module, context->function,
          loom_target_legalization_context_bundle(context), op) &&
      loom_amdgpu_subgroup_execution_is_uniform(context, op)) {
    return iree_ok_status();
  }
  return loom_kernel_target_legalize_subgroup_scan(entry, context, op,
                                                   out_result);
}

static bool loom_amdgpu_match_native_workgroup_collective(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;

  loom_value_id_t scan_value = LOOM_VALUE_ID_INVALID;
  loom_value_slice_t values = {0};
  loom_combining_kind_t kind = LOOM_COMBINING_KIND_COUNT_;
  if (loom_kernel_workgroup_reduce_isa(op)) {
    values = loom_kernel_workgroup_reduce_values(op);
    kind = loom_kernel_workgroup_reduce_kind(op);
  } else if (loom_kernel_workgroup_scan_isa(op)) {
    scan_value = loom_kernel_workgroup_scan_value(op);
    values = (loom_value_slice_t){.values = &scan_value, .count = 1};
    kind = loom_kernel_workgroup_scan_kind(op);
  } else {
    return false;
  }

  return loom_amdgpu_arithmetic_collective_is_native(context, values, kind);
}

static iree_status_t loom_amdgpu_retain_native_workgroup_collective(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  (void)context;
  (void)op;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
  };
  return iree_ok_status();
}

static bool loom_amdgpu_facts_are_positive_u32_power_of_two(
    loom_value_facts_t facts) {
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &value) || value <= 0 ||
      (uint64_t)value > UINT32_MAX) {
    return false;
  }
  const uint64_t unsigned_value = (uint64_t)value;
  return (unsigned_value & (unsigned_value - 1)) == 0;
}

static bool loom_amdgpu_facts_are_inline_u24(loom_value_facts_t facts) {
  int64_t value = 0;
  return loom_value_facts_as_exact_i64(facts, &value) && value >= 0 &&
         value <= 64;
}

static bool loom_amdgpu_selected_result_is_vgpr(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_target_contract_query_result_t* query_result) {
  if (query_result == NULL ||
      query_result->outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL ||
      query_result->selected_descriptor == NULL) {
    return false;
  }

  const loom_low_descriptor_t* descriptor = query_result->selected_descriptor;
  IREE_ASSERT_EQ(descriptor->result_count, 1);
  const loom_low_operand_t* result_operand =
      loom_low_descriptor_operand(descriptor_set, descriptor, 0);
  IREE_ASSERT_EQ(result_operand->reg_class_alt_count, 1);
  IREE_ASSERT_LT(result_operand->reg_class_alt_start,
                 descriptor_set->reg_class_alt_count);
  return descriptor_set->reg_class_alts[result_operand->reg_class_alt_start]
             .reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR;
}

static bool loom_amdgpu_match_scalar_multiply_add(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL ||
      context->fact_table == NULL) {
    return false;
  }

  loom_scalar_multiply_add_match_t match = {0};
  if (!loom_scalar_match_multiply_add(context->module, op, &match)) {
    return false;
  }

  const loom_value_facts_t lhs_facts = loom_value_fact_table_lookup(
      context->fact_table, loom_scalar_muli_lhs(match.multiply_op));
  const loom_value_facts_t rhs_facts = loom_value_fact_table_lookup(
      context->fact_table, loom_scalar_muli_rhs(match.multiply_op));
  if ((loom_amdgpu_facts_are_positive_u32_power_of_two(lhs_facts) ||
       loom_amdgpu_facts_are_positive_u32_power_of_two(rhs_facts)) &&
      loom_amdgpu_descriptor_set_has_ref(
          context->descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHL_ADD_U32_SHIFT_IMM)) {
    return true;
  }

  if (!loom_value_facts_fit_unsigned_bit_count(lhs_facts, 24) ||
      !loom_value_facts_fit_unsigned_bit_count(rhs_facts, 24)) {
    return false;
  }
  return (loom_amdgpu_facts_are_inline_u24(lhs_facts) &&
          loom_amdgpu_descriptor_set_has_ref(
              context->descriptor_set,
              LOOM_AMDGPU_DESCRIPTOR_REF_V_MAD_U32_U24_SRC0_INLINE)) ||
         (loom_amdgpu_facts_are_inline_u24(rhs_facts) &&
          loom_amdgpu_descriptor_set_has_ref(
              context->descriptor_set,
              LOOM_AMDGPU_DESCRIPTOR_REF_V_MAD_U32_U24_SRC1_INLINE));
}

static iree_status_t loom_amdgpu_legalize_scalar_multiply_add(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  loom_scalar_multiply_add_match_t match = {0};
  const bool matched =
      loom_scalar_match_multiply_add(context->module, op, &match);
  IREE_ASSERT(matched);
  (void)matched;

  bool result_is_vgpr = loom_amdgpu_selected_result_is_vgpr(
      context->descriptor_set, context->contract_query_result);
  if (!result_is_vgpr) {
    loom_target_contract_query_result_t add_query_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
        context, match.add_op, &add_query_result));
    result_is_vgpr = loom_amdgpu_selected_result_is_vgpr(
        context->descriptor_set, &add_query_result);
  }
  if (!result_is_vgpr) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_scalar_fuse_multiply_add_match(context->rewriter, &match));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static loom_target_legalizer_action_t loom_amdgpu_defer_or_reject_final(
    const loom_target_legalization_context_t* context) {
  return context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL
             ? LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL
             : LOOM_TARGET_LEGALIZER_ACTION_DEFER;
}

static bool loom_amdgpu_static_shape_carrier_type(
    const loom_module_t* module, const loom_op_t* op,
    loom_type_t* out_carrier_type) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_BROADCAST:
      *out_carrier_type =
          loom_module_value_type(module, loom_vector_broadcast_result(op));
      return true;
    case LOOM_OP_VECTOR_CONCAT:
      *out_carrier_type =
          loom_module_value_type(module, loom_vector_concat_result(op));
      return true;
    case LOOM_OP_VECTOR_EXTRACT:
      *out_carrier_type =
          loom_module_value_type(module, loom_vector_extract_source(op));
      return true;
    case LOOM_OP_VECTOR_INSERT:
      *out_carrier_type =
          loom_module_value_type(module, loom_vector_insert_dest(op));
      return true;
    case LOOM_OP_VECTOR_SLICE:
      *out_carrier_type =
          loom_module_value_type(module, loom_vector_slice_source(op));
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_legalize_static_vector_shape(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  bool rewritten = false;
  loom_type_t carrier_type = loom_type_none();
  loom_amdgpu_vector_storage_t carrier_storage = {0};
  // Tail extracts, rectangular slices, and row insertions can preserve native
  // packets within a larger aggregate. The shared rewrite bounds expansion;
  // packet legalization retains ownership of the resulting rank-one windows.
  const bool is_tail_extract =
      loom_vector_extract_isa(op) &&
      loom_type_is_vector(loom_module_value_type(
          context->module, loom_vector_extract_result(op)));
  if (is_tail_extract || loom_vector_slice_isa(op) ||
      loom_vector_insert_isa(op) ||
      (loom_amdgpu_static_shape_carrier_type(context->module, op,
                                             &carrier_type) &&
       loom_amdgpu_type_vector_storage(carrier_type, &carrier_storage) &&
       carrier_storage.register_count != 0 &&
       carrier_storage.register_count <=
           LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  }
  if (!rewritten && loom_vector_insert_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_insert(
        context, op, context->vector_packet_policy,
        /*insert_materializer=*/NULL, &rewritten));
  }
  if (!rewritten && loom_vector_concat_isa(op)) {
    loom_amdgpu_vector_concat_plan_t plan = {0};
    if (!loom_amdgpu_vector_concat_plan_from_op(context->module, op, &plan)) {
      IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
          context->pass, context->rewriter, op, &rewritten));
    }
  }
  if (!rewritten && loom_vector_slice_isa(op) &&
      loom_vector_slice_offsets(op).count != 0) {
    const loom_vector_packet_consumer_materializer_t materializer =
        loom_vector_predicate_extension_packet_materializer();
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_dynamic_slice(
        context, op, context->vector_packet_policy, &materializer, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_permutation(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL ||
      loom_amdgpu_vector_permutation_can_lower(context->module,
                                               context->descriptor_set, op)) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    return iree_ok_status();
  }

  bool rewritten = false;
  if (loom_vector_transpose_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_transpose_to_shuffle_rewrite_op(
        context->rewriter, op, &rewritten));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_masked_load(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_const_results(op)[0]);
  if (loom_type_rank(result_type) != 1 ||
      !loom_target_legalization_has_source_vector_carrier(
          context, result_type, loom_op_const_results(op)[0])) {
    return iree_ok_status();
  }
  const loom_memory_access_t access =
      loom_memory_access_cast(context->module, op);
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  if (loom_value_fact_table_query_uniform_element_origin(
          context->fact_table, context->module, loom_memory_access_mask(access),
          &condition)) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_load(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_load_result(op));
  if (loom_type_rank(result_type) > 1) {
    IREE_RETURN_IF_ERROR(
        loom_vector_static_shape_memory_rewrite_op(context, op, &rewritten));
  }
  if (!rewritten && (context->boundary_representation ==
                         LOOM_TARGET_BOUNDARY_REPRESENTATION_PROJECTED ||
                     context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL)) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_load(
        context, op, context->vector_packet_policy, &rewritten));
  }
  // Eager legalization lets store-root plans retain shared producers or stage
  // fanout. Final legalization must also complete loads feeding structural
  // consumers whose stores cannot own the entire producer graph. Preserve
  // those reads as packet snapshots at their original observation points.
  out_result->action = rewritten ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                                 : LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  const loom_type_t value_type =
      loom_module_value_type(context->module, loom_vector_store_value(op));
  if (loom_type_rank(value_type) > 1) {
    IREE_RETURN_IF_ERROR(
        loom_vector_static_shape_memory_rewrite_op(context, op, &rewritten));
  }
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_store(
        context, op, context->vector_packet_policy, &rewritten));
  }
  // AMDGPU memory lowering owns stores that do not need packetization here.
  // Retain them for that lowering instead of invoking scalar reference stores.
  out_result->action = rewritten ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                                 : LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_oversized_vector_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  loom_vector_packet_reduce_result_t packet_result =
      LOOM_VECTOR_PACKET_REDUCE_RESULT_NONE;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_reduce(
      context, op, context->vector_packet_policy, &packet_result));
  bool rewritten = packet_result == LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
  if (packet_result == LOOM_VECTOR_PACKET_REDUCE_RESULT_CAPTURE_INPUT) {
    IREE_RETURN_IF_ERROR(loom_vector_reduce_captured_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_retain_native_vector_op(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  (void)context;
  (void)op;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_bitcast(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  if (!loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_bitcast(
        context, op, context->vector_packet_policy, &rewritten));
  }
  out_result->action = rewritten ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                                 : LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_bitstream(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  if (context->contract_query_result != NULL &&
      context->contract_query_result->outcome ==
          LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED &&
      context->boundary_representation ==
          LOOM_TARGET_BOUNDARY_REPRESENTATION_PROJECTED) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_bitstream(
        context, op, context->vector_packet_policy, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  } else if (context->contract_query_result == NULL ||
             context->contract_query_result->outcome !=
                 LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_table_lookup(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL ||
      loom_amdgpu_vector_table_lookup_is_supported(
          context->module, context->fact_table, context->descriptor_set, op)) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_table_quantize(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_scalar_type_t input_element_type =
      loom_type_element_type(loom_module_value_type(
          context->module, loom_vector_table_quantize_input(op)));
  if (input_element_type != LOOM_SCALAR_TYPE_F8E4M3 &&
      input_element_type != LOOM_SCALAR_TYPE_F8E5M2 &&
      input_element_type != LOOM_SCALAR_TYPE_F16 &&
      input_element_type != LOOM_SCALAR_TYPE_BF16 &&
      input_element_type != LOOM_SCALAR_TYPE_F32) {
    return iree_ok_status();
  }
  const loom_vector_table_quantize_policy_t policy = {
      .packet_bit_count = LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES * 32u,
      .comparison_element_type = LOOM_SCALAR_TYPE_F32,
      .ordinal_element_type = LOOM_SCALAR_TYPE_I32,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_table_quantize_rewrite(context, op, &policy, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_decode(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
          context->module, context->fact_table, context->descriptor_set, op)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_vector_transform(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (loom_amdgpu_vector_transform_can_lower(context->module,
                                             context->descriptor_set, op)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_atomic(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_memory_access_t access =
      loom_memory_access_cast(context->module, op);
  const uint8_t atomic_kind =
      loom_attr_as_enum(loom_memory_access_atomic_kind(access));
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table,
                                       loom_memory_access_view(access)),
          &view_reference)) {
    return iree_ok_status();
  }
  if (view_reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    // Kernel metadata enables preserving scalar arithmetic for all float modes.
    // Invocation-private updates need no memory atomicity or inter-thread
    // order.
    IREE_RETURN_IF_ERROR(
        loom_view_atomic_rewrite_private(context->rewriter, op));
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
    return iree_ok_status();
  }
  const loom_type_t value_type =
      loom_module_value_type(context->module, loom_memory_access_value(access));
  if (!loom_scalar_type_set_contains(
          LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
          loom_type_element_type(value_type))) {
    return iree_ok_status();
  }

  const loom_amdgpu_atomic_operation_kind_t operation_kind =
      loom_view_atomic_reduce_isa(op) ? LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE
                                      : LOOM_AMDGPU_ATOMIC_OPERATION_RMW;
  const uint8_t scope =
      loom_attr_as_enum(loom_memory_access_atomic_scope(access));
  if (loom_amdgpu_atomic_has_native_candidate(
          context->descriptor_set, view_reference.memory_space, operation_kind,
          atomic_kind, scope, loom_memory_access_flags(access), value_type)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }
  // Narrow logical CAS uses a masked word packet and a full-width pointer for
  // device-visible storage. Check the same physical carrier here while the
  // shared rewrite retains the source payload and memory space.
  const bool subword =
      loom_scalar_type_bitwidth(loom_type_element_type(value_type)) < 32;
  const loom_type_t cmpxchg_type =
      subword ? loom_type_scalar(LOOM_SCALAR_TYPE_I32) : value_type;
  if (!loom_amdgpu_atomic_has_native_candidate(
          context->descriptor_set, view_reference.memory_space,
          LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG, atomic_kind, scope,
          /*access_flags=*/0, cmpxchg_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_view_atomic_rewrite_cmpxchg(context->rewriter, op));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_packed_atomic(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_memory_access_t access =
      loom_memory_access_cast(context->module, op);
  const loom_type_t value_type =
      loom_module_value_type(context->module, loom_memory_access_value(access));
  IREE_RETURN_IF_ERROR(
      loom_target_legalization_resolve_function_analysis(context));
  loom_low_source_memory_access_plan_t source = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(context->view_regions, op,
                                                /*axis_byte_strides=*/NULL,
                                                &source, &diagnostic) ||
      !loom_amdgpu_atomic_packed_half_source_shape(&source, value_type) ||
      !loom_amdgpu_atomic_scope_supported(context->descriptor_set, &source,
                                          value_type) ||
      !loom_amdgpu_atomic_orderings_supported(context->descriptor_set,
                                              &source)) {
    return iree_ok_status();
  }
  const loom_amdgpu_atomic_operation_kind_t operation_kind =
      loom_vector_atomic_reduce_isa(op) ? LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE
                                        : LOOM_AMDGPU_ATOMIC_OPERATION_RMW;
  if (loom_amdgpu_atomic_has_native_candidate(
          context->descriptor_set, source.memory_space, operation_kind,
          loom_attr_as_enum(loom_memory_access_atomic_kind(access)),
          source.atomic.scope, source.access_flags, value_type)) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  }
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kAmdgpuLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SHUFFLE,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_kernel_target_legalize_subgroup_predicate,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_BROADCAST,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_kernel_target_legalize_subgroup_predicate,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_kernel_target_legalize_subgroup_predicate,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SHUFFLE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .match = loom_amdgpu_match_subgroup_shuffle_reference_lowering,
        .legalize = loom_kernel_target_legalize_subgroup_shuffle,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_REDUCE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_amdgpu_legalize_subgroup_reduce_reference_lowering,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SCAN,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_amdgpu_legalize_subgroup_scan_reference_lowering,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED,
        .root_kind = LOOM_OP_KERNEL_WORKGROUP_REDUCE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .match = loom_amdgpu_match_native_workgroup_collective,
        .legalize = loom_amdgpu_retain_native_workgroup_collective,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED,
        .root_kind = LOOM_OP_KERNEL_WORKGROUP_SCAN,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .match = loom_amdgpu_match_native_workgroup_collective,
        .legalize = loom_amdgpu_retain_native_workgroup_collective,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_SCALAR_MULI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I32,
        .match = loom_amdgpu_match_scalar_multiply_add,
        .legalize = loom_amdgpu_legalize_scalar_multiply_add,
    },
    {
        .root_kind = LOOM_OP_SCALAR_CLAMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .match = loom_amdgpu_match_f32_clamp_decomposition,
        .legalize = loom_scalar_legalize_clamp,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CLAMPF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .match = loom_amdgpu_match_f32_clamp_decomposition,
        .legalize = loom_vector_legalize_float_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BROADCAST,
        .legalize = loom_amdgpu_legalize_static_vector_shape,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CONCAT,
        .legalize = loom_amdgpu_legalize_static_vector_shape,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTRACT,
        .legalize = loom_amdgpu_legalize_static_vector_shape,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INSERT,
        .legalize = loom_amdgpu_legalize_static_vector_shape,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INDEX_CAST,
        .legalize = loom_vector_legalize_index_cast,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SLICE,
        .legalize = loom_amdgpu_legalize_static_vector_shape,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHUFFLE,
        .legalize = loom_amdgpu_legalize_vector_permutation,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSPOSE,
        .legalize = loom_amdgpu_legalize_vector_permutation,
    },
    // Retain native packed pairs. Other shapes and profiles use the shared
    // per-element reference, with policy enforced by that provider.
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_REDUCE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_BF16,
        .legalize = loom_amdgpu_legalize_packed_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_RMW,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_BF16,
        .legalize = loom_amdgpu_legalize_packed_atomic,
    },
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_REDUCE,
        .legalize = loom_amdgpu_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_RMW,
        .legalize = loom_amdgpu_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD,
        .legalize = loom_amdgpu_legalize_vector_load,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD_MASK,
        .legalize = loom_amdgpu_legalize_masked_load,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_amdgpu_legalize_vector_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FRAGMENT_STORE,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_amdgpu_legalize_oversized_vector_reduce,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_INSERT,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITCAST,
        .legalize = loom_amdgpu_legalize_vector_bitcast,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITPACK,
        .legalize = loom_amdgpu_legalize_vector_bitstream,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKU,
        .legalize = loom_amdgpu_legalize_vector_bitstream,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKS,
        .legalize = loom_amdgpu_legalize_vector_bitstream,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_LOOKUP,
        .legalize = loom_amdgpu_legalize_vector_table_lookup,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_QUANTIZE,
        .legalize = loom_amdgpu_legalize_vector_table_quantize,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOTF,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSFORM,
        .legalize = loom_amdgpu_legalize_vector_transform,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DECODE,
        .legalize = loom_amdgpu_legalize_vector_decode,
    },
};

const loom_target_legalizer_provider_t
    loom_amdgpu_target_legalizer_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .scope = {.target_stable_id = LOOM_AMDGPU_TARGET_STABLE_ID},
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .rules = kAmdgpuLegalizerRules,
        .rule_count = IREE_ARRAYSIZE(kAmdgpuLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_amdgpu_target_legalizer_provider(
    void) {
  return &loom_amdgpu_target_legalizer_provider_storage;
}
