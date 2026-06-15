// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>

#include "loom/ops/cache.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

iree_status_t loom_amdgpu_emit_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_base_addr,
    loom_value_id_t* out_low_vaddr) {
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  loom_value_id_t low_accumulator = low_base_addr;
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    switch (access->dynamic_term_kinds[i]) {
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
        break;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
        continue;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
        IREE_ASSERT_UNREACHABLE("unknown AMDGPU memory dynamic index kind");
        IREE_BUILTIN_UNREACHABLE();
    }
    const loom_low_source_memory_dynamic_term_t* term =
        &access->source.dynamic_terms[i];
    loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
        context, source_op, term->index, &low_index));
    loom_value_id_t low_offset = low_index;
    for (uint8_t stride_ordinal = 0; stride_ordinal < term->stride_value_count;
         ++stride_ordinal) {
      loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
          context, source_op, term->stride_values[stride_ordinal],
          &low_stride));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32,
          low_offset, low_stride, vgpr_type, &low_offset));
    }
    if (term->byte_stride != 1) {
      IREE_ASSERT(term->byte_stride >= 0 && term->byte_stride <= UINT32_MAX);
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
          context, source_op, low_offset, (uint32_t)term->byte_stride,
          LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &low_offset));
    }
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      low_accumulator = low_offset;
      continue;
    }
    loom_value_id_t add_operands[] = {low_accumulator, low_offset};
    loom_op_t* low_add_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, add_operands,
        IREE_ARRAYSIZE(add_operands), loom_make_named_attr_slice(NULL, 0),
        &vgpr_type, 1, &low_add_op));
    low_accumulator = loom_value_slice_get(loom_low_op_results(low_add_op), 0);
  }

  if (access->vaddr_static_byte_offset != 0) {
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          access->vaddr_static_byte_offset, vgpr_type, &low_static_offset));
      *out_low_vaddr = low_static_offset;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
        low_accumulator, access->vaddr_static_byte_offset, vgpr_type,
        &low_accumulator));
  }

  if (low_accumulator == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, out_low_vaddr);
  }
  *out_low_vaddr = low_accumulator;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_soffset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access,
    loom_value_id_t* out_low_soffset) {
  return loom_amdgpu_emit_sgpr_byte_offset_terms(
      context, source_op, &access->source, access->dynamic_term_kinds,
      access->scalar_byte_offset, out_low_soffset);
}

static bool loom_amdgpu_memory_saddr_has_offset(
    const loom_amdgpu_memory_access_t* access, uint64_t static_byte_offset) {
  if (static_byte_offset != 0) {
    return true;
  }
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] ==
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_memory_saddr_offset_facts(
    const loom_amdgpu_memory_access_t* access, uint64_t static_byte_offset,
    loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (static_byte_offset > INT64_MAX) {
    return false;
  }
  loom_value_facts_t offset_facts =
      loom_value_facts_exact_i64((int64_t)static_byte_offset);
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      continue;
    }
    loom_value_facts_addi(&offset_facts,
                          &access->source.dynamic_terms[i].byte_facts,
                          &offset_facts);
  }
  *out_facts = offset_facts;
  return true;
}

static iree_status_t loom_amdgpu_emit_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_op_t** out_op) {
  IREE_ASSERT(packet->access.descriptor != NULL);
  IREE_ASSERT(packet->opcode_id != LOOM_STRING_ID_INVALID);
  *out_op = NULL;
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = packet->access.descriptor,
      .opcode_id = packet->opcode_id,
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count, attrs, result_types,
      result_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, out_op));
  return loom_low_lower_record_source_memory_access(context, *out_op,
                                                    &packet->access.source);
}

static bool loom_amdgpu_memory_descriptor_has_implicit_resource_operand(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_packet_plan_t* packet) {
  return loom_amdgpu_descriptor_has_implicit_resource_operand(
      loom_low_lower_context_descriptor_set(context),
      packet->access.descriptor);
}

static iree_status_t loom_amdgpu_emit_memory_implicit_m0(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    loom_value_id_t* out_low_m0) {
  *out_low_m0 = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_resolved_descriptor_t packet_descriptor = {
      .descriptor = packet->access.descriptor,
      .opcode_id = packet->opcode_id,
  };
  return loom_amdgpu_emit_m0_u32(context, source_op, &packet_descriptor, 0,
                                 out_low_m0);
}

static iree_status_t loom_amdgpu_memory_payload_low_type(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_type_t* out_type) {
  if (access->payload_register_class ==
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_SGPR) {
    if (access->payload_register_count == 1) {
      return loom_amdgpu_make_sgpr_type(context, out_type);
    }
    return loom_amdgpu_make_sgpr_range_type(
        context, access->payload_register_count, out_type);
  }
  if (access->payload_register_count == 1) {
    return loom_amdgpu_make_vgpr_type(context, out_type);
  }
  return loom_amdgpu_make_vgpr_range_type(
      context, access->payload_register_count, out_type);
}

static iree_status_t loom_amdgpu_try_emit_exact_vgpr_store_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t source_register_offset,
    uint32_t payload_register_count, loom_value_id_t* out_low_value,
    bool* out_emitted) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL || payload_register_count == 0 ||
      payload_register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_ok_status();
  }

  uint32_t bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < payload_register_count; ++i) {
    if (source_register_offset > UINT32_MAX - i ||
        !loom_amdgpu_source_lane_as_u32_bits(fact_table, module, source_value,
                                             source_register_offset + i,
                                             &bit_patterns[i])) {
      return iree_ok_status();
    }
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < payload_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        bit_patterns[i], vgpr_type, &low_lanes[i]));
  }
  if (payload_register_count == 1) {
    *out_low_value = low_lanes[0];
    *out_emitted = true;
    return iree_ok_status();
  }

  loom_type_t vgpr_range_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, payload_register_count, &vgpr_range_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context), low_lanes,
                            payload_register_count, vgpr_range_type,
                            source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_ensure_memory_store_payload_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t source_value,
    uint32_t source_register_offset, loom_value_id_t low_value,
    loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  if (access->payload_register_class !=
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    return iree_ok_status();
  }
  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (is_sgpr && loom_low_register_type_unit_count(low_type) ==
                     access->payload_register_count) {
    bool emitted_exact = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_exact_vgpr_store_payload(
        context, source_op, source_value, source_register_offset,
        access->payload_register_count, out_low_value, &emitted_exact));
    if (emitted_exact) {
      return iree_ok_status();
    }
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_value, out_low_value);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_sgpr_mul_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, loom_type_t sgpr_type,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {
      low_lhs,
      low_rhs,
  };
  loom_op_t* low_mul_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &sgpr_type, 1, &low_mul_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_mul_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_sgpr64_mul_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_wide_lhs, loom_value_id_t low_rhs,
    loom_value_id_t* out_low_product) {
  *out_low_product = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_wide_lhs, /*offset=*/0, sgpr_type, &low_lhs_lo));
  loom_value_id_t low_lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_wide_lhs, /*offset=*/1, sgpr_type, &low_lhs_hi));

  loom_value_id_t low_product_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_mul_u32(
      context, source_op, low_lhs_lo, low_rhs, sgpr_type,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32, &low_product_lo));
  loom_value_id_t low_product_lo_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_mul_u32(
      context, source_op, low_lhs_lo, low_rhs, sgpr_type,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_HI_U32, &low_product_lo_hi));
  loom_value_id_t low_product_hi_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_mul_u32(
      context, source_op, low_lhs_hi, low_rhs, sgpr_type,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32, &low_product_hi_low));
  loom_value_id_t add_operands[] = {
      low_product_lo_hi,
      low_product_hi_low,
  };
  loom_op_t* low_add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, add_operands,
      IREE_ARRAYSIZE(add_operands), loom_make_named_attr_slice(NULL, 0),
      &sgpr_type, 1, &low_add_op));
  const loom_value_id_t low_product_hi =
      loom_value_slice_get(loom_low_op_results(low_add_op), 0);

  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  loom_value_id_t sources[] = {
      low_product_lo,
      low_product_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      sgpr_x2_type, source_op->location, &concat_op));
  *out_low_product = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_sgpr64_scale_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_unscaled_offset, int64_t byte_stride,
    uint32_t byte_shift, loom_value_id_t* out_low_offset) {
  *out_low_offset = low_unscaled_offset;
  if (byte_stride == 1) {
    return iree_ok_status();
  }
  IREE_ASSERT(byte_stride >= 0 && byte_stride <= UINT32_MAX);
  if (byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
    loom_type_t sgpr_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
    loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, byte_shift,
        sgpr_type, &low_shift));

    loom_type_t sgpr_x2_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
    loom_value_id_t shift_operands[] = {
        low_unscaled_offset,
        low_shift,
    };
    loom_op_t* low_shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL_B64,
        shift_operands, IREE_ARRAYSIZE(shift_operands),
        loom_make_named_attr_slice(NULL, 0), &sgpr_x2_type, 1, &low_shift_op));
    *out_low_offset =
        loom_value_slice_get(loom_low_op_results(low_shift_op), 0);
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      (uint32_t)byte_stride, sgpr_type, &low_scale));
  return loom_amdgpu_emit_sgpr64_mul_u32(
      context, source_op, low_unscaled_offset, low_scale, out_low_offset);
}

static iree_status_t loom_amdgpu_emit_memory_saddr_dynamic_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t* out_low_term) {
  *out_low_term = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, term->index, &low_index));
  loom_value_id_t low_wide_index = LOOM_VALUE_ID_INVALID;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_index_type = loom_module_value_type(module, low_index);
  bool index_is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_index_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &index_is_sgpr));
  if (!index_is_sgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar memory address term selected non-SGPR value");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t index_unit_count =
      loom_low_register_type_unit_count(low_index_type);
  if (index_unit_count == 2) {
    low_wide_index = low_index;
  } else {
    if (index_unit_count != 1) {
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU scalar memory address term selected unsupported SGPR width");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_from_u32(
        context, source_op, low_index, &low_wide_index));
  }
  loom_value_id_t low_wide_offset = low_wide_index;
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->stride_values[i], &low_stride));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_mul_u32(
        context, source_op, low_wide_offset, low_stride, &low_wide_offset));
  }
  return loom_amdgpu_emit_sgpr64_scale_byte_offset(
      context, source_op, low_wide_offset, term->byte_stride, term->byte_shift,
      out_low_term);
}

iree_status_t loom_amdgpu_emit_memory_saddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_binding,
    loom_value_id_t* out_low_saddr) {
  *out_low_saddr = low_binding;
  const uint64_t static_byte_offset =
      access->scalar_offset_placement ==
              LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE
          ? access->scalar_base_byte_offset
          : access->scalar_byte_offset;
  if (!loom_amdgpu_memory_saddr_has_offset(access, static_byte_offset)) {
    return iree_ok_status();
  }
  loom_value_facts_t offset_facts = loom_value_facts_unknown();
  const bool has_offset_facts = loom_amdgpu_memory_saddr_offset_facts(
      access, static_byte_offset, &offset_facts);
  if (has_offset_facts && loom_value_facts_is_zero(offset_facts)) {
    return iree_ok_status();
  }
  if (static_byte_offset <= UINT32_MAX && has_offset_facts &&
      loom_value_facts_fit_unsigned_bit_count(offset_facts, 32)) {
    loom_value_id_t low_u32_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_terms(
        context, source_op, &access->source, access->dynamic_term_kinds,
        (uint32_t)static_byte_offset, &low_u32_offset));
    return loom_amdgpu_emit_sgpr64_add_u32_offset(
        context, source_op, low_binding, low_u32_offset, out_low_saddr);
  }

  loom_value_id_t low_offset = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      continue;
    }
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr_dynamic_term(
        context, source_op, &access->source.dynamic_terms[i], &low_term));
    if (low_offset == LOOM_VALUE_ID_INVALID) {
      low_offset = low_term;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add(
        context, source_op, low_offset, low_term, &low_offset));
  }
  if (static_byte_offset != 0) {
    loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
        context, source_op, static_byte_offset, &low_static_offset));
    if (low_offset == LOOM_VALUE_ID_INVALID) {
      low_offset = low_static_offset;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add(
          context, source_op, low_offset, low_static_offset, &low_offset));
    }
  }
  if (low_offset == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_sgpr64_add(context, source_op, low_binding,
                                     low_offset, out_low_saddr);
}

typedef struct loom_amdgpu_hal_buffer_descriptor_extent_t {
  // Static descriptor range word used when dynamic_extent is absent.
  int64_t static_extent;
  // Optional SGPR descriptor range word for dynamically sized views.
  loom_value_id_t dynamic_extent;
} loom_amdgpu_hal_buffer_descriptor_extent_t;

static bool loom_amdgpu_source_access_view_range_facts(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_access,
    int64_t* out_static_base, loom_value_facts_t* out_range_facts) {
  *out_static_base = 0;
  *out_range_facts = loom_value_facts_unknown();
  if (source_access == NULL) {
    return false;
  }

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (source_access->view_value_id >= module->values.count) {
    return false;
  }

  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table,
                                       source_access->view_value_id),
          &view_reference)) {
    return false;
  }

  if (!loom_value_facts_fit_unsigned_bit_count(view_reference.base_byte_offset,
                                               32)) {
    return false;
  }
  const bool has_dynamic_view_base =
      source_access->dynamic_view_base_term_count != 0;
  int64_t static_base = 0;
  if (!has_dynamic_view_base &&
      !loom_value_facts_as_exact_i64(view_reference.base_byte_offset,
                                     &static_base)) {
    return false;
  }
  loom_value_facts_t range_facts = view_reference.base_byte_offset;
  loom_value_facts_addi(&range_facts, &view_reference.footprint_byte_length,
                        &range_facts);
  if (!loom_value_facts_fit_unsigned_bit_count(range_facts, 32)) {
    return false;
  }

  *out_static_base = has_dynamic_view_base
                         ? source_access->static_view_base_byte_offset
                         : static_base;
  *out_range_facts = range_facts;
  return true;
}

static bool loom_amdgpu_source_access_view_has_dense_layout(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_access) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL || source_access == NULL) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (source_access->view_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t view_type =
      loom_module_value_type(module, source_access->view_value_id);
  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  return loom_encoding_query_type_address_layout(
             &fact_table->context, module, view_type, stride_storage,
             IREE_ARRAYSIZE(stride_storage), &layout) &&
         layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE;
}

static iree_status_t
loom_amdgpu_source_access_dynamic_view_base_term_can_emit_u32(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_dynamic_term_t* term, bool* out_can_emit) {
  *out_can_emit = false;
  if (term->byte_stride < 0 || term->byte_stride > UINT32_MAX ||
      !loom_value_facts_fit_unsigned_bit_count(term->byte_facts, 32)) {
    return iree_ok_status();
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, term->index, &low_index));
  bool is_sgpr_b32 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_value_is_register_class_count(
      context, low_index, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &is_sgpr_b32));
  if (!is_sgpr_b32) {
    return iree_ok_status();
  }

  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->stride_values[i], &low_stride));
    bool stride_is_sgpr_b32 = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_value_is_register_class_count(
        context, low_stride, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
        &stride_is_sgpr_b32));
    if (!stride_is_sgpr_b32) {
      return iree_ok_status();
    }
  }

  *out_can_emit = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_access_dynamic_view_base_can_emit_u32(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_access,
    bool* out_can_emit) {
  *out_can_emit = false;
  if (source_access->dynamic_view_base_term_count == 0) {
    *out_can_emit = true;
    return iree_ok_status();
  }
  IREE_ASSERT_LE(source_access->dynamic_view_base_term_count,
                 source_access->dynamic_term_count);

  for (uint8_t i = 0; i < source_access->dynamic_view_base_term_count; ++i) {
    bool can_emit = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_source_access_dynamic_view_base_term_can_emit_u32(
            context, &source_access->dynamic_terms[i], &can_emit));
    if (!can_emit) {
      return iree_ok_status();
    }
  }
  *out_can_emit = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_source_access_dynamic_view_base_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source_access,
    loom_type_t sgpr_type, loom_value_id_t* out_low_base, bool* out_emitted) {
  *out_low_base = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;
  if (source_access->dynamic_view_base_term_count == 0) {
    return iree_ok_status();
  }
  bool can_emit = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_access_dynamic_view_base_can_emit_u32(
      context, source_access, &can_emit));
  if (!can_emit) {
    return iree_ok_status();
  }

  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < source_access->dynamic_view_base_term_count; ++i) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_term(
        context, source_op, &source_access->dynamic_terms[i], &low_term));
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      low_accumulator = low_term;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        low_accumulator, low_term, sgpr_type, &low_accumulator));
  }

  *out_low_base = low_accumulator;
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_source_access_dense_dynamic_range(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source_access,
    int64_t static_view_base, loom_value_id_t* out_dynamic_extent,
    bool* out_emitted) {
  *out_dynamic_extent = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;
  if (!loom_amdgpu_source_access_view_has_dense_layout(context,
                                                       source_access) ||
      static_view_base < 0 || static_view_base > UINT32_MAX) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t view_type =
      loom_module_value_type(module, source_access->view_value_id);
  if (!loom_type_is_view(view_type)) {
    return iree_ok_status();
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (element_bit_count <= 0 || (element_bit_count % 8) != 0) {
    return iree_ok_status();
  }

  int64_t static_scale = element_bit_count / 8;
  loom_value_id_t low_extent = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  bool can_emit_dynamic_base = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_access_dynamic_view_base_can_emit_u32(
      context, source_access, &can_emit_dynamic_base));
  if (!can_emit_dynamic_base) {
    return iree_ok_status();
  }

  const uint8_t rank = loom_type_rank(view_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (!loom_type_dim_is_dynamic_at(view_type, axis)) {
      const int64_t dim_size = loom_type_dim_static_size_at(view_type, axis);
      if (dim_size < 0 ||
          !iree_checked_mul_i64(static_scale, dim_size, &static_scale) ||
          static_scale > UINT32_MAX) {
        return iree_ok_status();
      }
      continue;
    }

    loom_value_id_t low_dim = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_type_dim_value_id_at(view_type, axis), &low_dim));
    bool is_sgpr_b32 = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_value_is_register_class_count(
        context, low_dim, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &is_sgpr_b32));
    if (!is_sgpr_b32) {
      return iree_ok_status();
    }
    if (low_extent == LOOM_VALUE_ID_INVALID) {
      low_extent = low_dim;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32, low_extent,
        low_dim, sgpr_type, &low_extent));
  }

  if (low_extent == LOOM_VALUE_ID_INVALID &&
      source_access->dynamic_view_base_term_count == 0) {
    return iree_ok_status();
  }
  if (static_scale < 0 || static_scale > UINT32_MAX) {
    return iree_ok_status();
  }
  if (low_extent == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        (uint32_t)static_scale, sgpr_type, &low_extent));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_scale_u32(
        context, source_op, low_extent, (uint32_t)static_scale, sgpr_type,
        &low_extent));
  }

  loom_value_id_t low_dynamic_base = LOOM_VALUE_ID_INVALID;
  bool emitted_dynamic_base = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_source_access_dynamic_view_base_u32(
      context, source_op, source_access, sgpr_type, &low_dynamic_base,
      &emitted_dynamic_base));
  if (emitted_dynamic_base) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        low_dynamic_base, low_extent, sgpr_type, &low_extent));
  }
  if (static_view_base != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, low_extent,
        (uint32_t)static_view_base, sgpr_type, &low_extent));
  }

  *out_dynamic_extent = low_extent;
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_buffer_descriptor_extent_from_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source_access,
    loom_amdgpu_hal_buffer_descriptor_extent_t* inout_extent) {
  int64_t static_base = 0;
  loom_value_facts_t range_facts = loom_value_facts_unknown();
  if (!loom_amdgpu_source_access_view_range_facts(context, source_access,
                                                  &static_base, &range_facts)) {
    return iree_ok_status();
  }

  int64_t exact_range = 0;
  if (loom_value_facts_as_exact_i64(range_facts, &exact_range)) {
    IREE_ASSERT_GE(exact_range, 0);
    IREE_ASSERT_LE((uint64_t)exact_range, (uint64_t)UINT32_MAX);
    inout_extent->static_extent = exact_range;
    return iree_ok_status();
  }

  bool emitted = false;
  loom_value_id_t dynamic_extent = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_source_access_dense_dynamic_range(
      context, source_op, source_access, static_base, &dynamic_extent,
      &emitted));
  if (emitted) {
    inout_extent->dynamic_extent = dynamic_extent;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_hal_buffer_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_binding,
    const loom_low_source_memory_access_plan_t* source_access,
    loom_value_id_t* out_low_descriptor) {
  *out_low_descriptor = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_hal_buffer_descriptor_extent_t extent = {
      .static_extent = UINT32_MAX,
      .dynamic_extent = LOOM_VALUE_ID_INVALID,
  };
  bool resource_has_extent = false;
  int64_t cache_swizzle_stride = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* binding = loom_module_value(module, low_binding);
  const loom_op_t* binding_op = loom_value_def_op(binding);
  if (binding_op != NULL && loom_low_resource_isa(binding_op)) {
    if (loom_low_resource_extent_value_is_present(binding_op)) {
      extent.dynamic_extent = loom_low_resource_extent_value(binding_op);
      resource_has_extent = true;
    } else {
      const loom_attribute_t extent_attr =
          loom_op_attrs(binding_op)[loom_low_resource_extent_ATTR_INDEX];
      if (!loom_attr_is_absent(extent_attr)) {
        const int64_t resource_extent = loom_low_resource_extent(binding_op);
        if (resource_extent <= UINT32_MAX) {
          extent.static_extent = resource_extent;
        }
        resource_has_extent = true;
      }
    }

    const loom_attribute_t cache_swizzle_stride_attr = loom_op_attrs(
        binding_op)[loom_low_resource_cache_swizzle_stride_ATTR_INDEX];
    if (!loom_attr_is_absent(cache_swizzle_stride_attr)) {
      cache_swizzle_stride = loom_low_resource_cache_swizzle_stride(binding_op);
    }
  }
  if (!resource_has_extent) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_buffer_descriptor_extent_from_source(
        context, source_op, source_access, &extent));
  }

  loom_type_t descriptor_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 4, &descriptor_type));
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("cache_swizzle_stride"), cache_swizzle_stride, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  loom_value_id_t operands[2] = {low_binding, extent.dynamic_extent};
  uint16_t operand_count = 1;
  uint16_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR;
  if (extent.dynamic_extent != LOOM_VALUE_ID_INVALID) {
    descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR_EXTENT;
    operand_count = 2;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("extent"), extent.static_extent, attrs,
        IREE_ARRAYSIZE(attrs), &attr_count));
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &descriptor_type, 1,
      &low_op));
  *out_low_descriptor = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_flat_wide_dynamic_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t low_index, loom_type_t vgpr_type,
    loom_value_id_t* out_low_lo, loom_value_id_t* out_low_hi,
    bool* out_emitted) {
  *out_emitted = false;
  if (term->byte_stride != 1 || term->stride_value_count != 0) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_index_type = loom_module_value_type(module, low_index);
  if (!loom_low_type_is_register(low_index_type) ||
      loom_low_register_type_unit_count(low_index_type) != 2) {
    return iree_ok_status();
  }

  loom_type_t source_lane_type =
      loom_low_register_type_with_unit_count(low_index_type, 1);
  loom_value_id_t low_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_index, /*offset=*/0, source_lane_type, &low_lo));
  loom_value_id_t low_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_index, /*offset=*/1, source_lane_type, &low_hi));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_lo, out_low_lo));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_hi, out_low_hi));
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_flat_dynamic_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term, loom_type_t vgpr_type,
    loom_value_id_t* out_low_lo, loom_value_id_t* out_low_hi) {
  *out_low_lo = LOOM_VALUE_ID_INVALID;
  *out_low_hi = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, term->index, &low_index));
  bool emitted_wide = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_wide_dynamic_term(
      context, source_op, term, low_index, vgpr_type, out_low_lo, out_low_hi,
      &emitted_wide));
  if (emitted_wide) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
      context, source_op, term->index, &low_index));
  if (term->byte_stride == 1) {
    *out_low_lo = low_index;
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, out_low_hi);
  }

  IREE_ASSERT(term->byte_shift !=
              LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      low_index, term->byte_shift, vgpr_type, out_low_lo));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      low_index, 32u - term->byte_shift, vgpr_type, out_low_hi);
}

static iree_status_t loom_amdgpu_emit_memory_flat_add_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_term_lo, loom_value_id_t low_term_hi,
    loom_type_t vgpr_type, loom_type_t sgpr_x2_type,
    loom_value_id_t* inout_low_vaddr_lo, loom_value_id_t* inout_low_vaddr_hi) {
  loom_value_id_t add_lo_operands[] = {
      *inout_low_vaddr_lo,
      low_term_lo,
  };
  loom_type_t add_lo_result_types[] = {
      vgpr_type,
      sgpr_x2_type,
  };
  loom_op_t* low_add_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
      add_lo_operands, IREE_ARRAYSIZE(add_lo_operands),
      loom_make_named_attr_slice(NULL, 0), add_lo_result_types,
      IREE_ARRAYSIZE(add_lo_result_types), &low_add_lo_op));
  *inout_low_vaddr_lo =
      loom_value_slice_get(loom_low_op_results(low_add_lo_op), 0);
  const loom_value_id_t low_carry =
      loom_value_slice_get(loom_low_op_results(low_add_lo_op), 1);

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, *inout_low_vaddr_hi, inout_low_vaddr_hi));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_term_hi, &low_term_hi));
  loom_value_id_t add_hi_operands[] = {
      *inout_low_vaddr_hi,
      low_term_hi,
      low_carry,
  };
  loom_type_t add_hi_result_types[] = {
      vgpr_type,
      sgpr_x2_type,
  };
  loom_op_t* low_add_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
      add_hi_operands, IREE_ARRAYSIZE(add_hi_operands),
      loom_make_named_attr_slice(NULL, 0), add_hi_result_types,
      IREE_ARRAYSIZE(add_hi_result_types), &low_add_hi_op));
  *inout_low_vaddr_hi =
      loom_value_slice_get(loom_low_op_results(low_add_hi_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_flat_scalar_dynamic_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t* out_low_term, bool* out_emitted) {
  *out_low_term = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, term->index, &low_index));
  bool index_is_sgpr_b64 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_value_is_register_class_count(
      context, low_index, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &index_is_sgpr_b64));
  if (index_is_sgpr_b64 && term->byte_stride == 1 &&
      term->stride_value_count == 0) {
    *out_low_term = low_index;
    *out_emitted = true;
    return iree_ok_status();
  }

  bool index_is_sgpr_b32 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_value_is_register_class_count(
      context, low_index, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
      &index_is_sgpr_b32));
  if (!index_is_sgpr_b32) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr_dynamic_term(
      context, source_op, term, out_low_term));
  *out_emitted = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_memory_flat_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_resource,
    loom_value_id_t* out_low_vaddr) {
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

  loom_value_id_t low_scalar_base = low_resource;
  if (access->vaddr_static_byte_offset != 0) {
    loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
        context, source_op, access->vaddr_static_byte_offset,
        &low_static_offset));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_sgpr64_add(context, source_op, low_scalar_base,
                                    low_static_offset, &low_scalar_base));
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &vgpr_x2_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));

  loom_value_id_t low_vaddr_lo = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_vaddr_hi = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      continue;
    }
    const loom_low_source_memory_dynamic_term_t* term =
        &access->source.dynamic_terms[i];
    loom_value_id_t low_scalar_term = LOOM_VALUE_ID_INVALID;
    bool scalar_term_emitted = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_scalar_dynamic_term(
        context, source_op, term, &low_scalar_term, &scalar_term_emitted));
    if (scalar_term_emitted && low_vaddr_lo == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_sgpr64_add(context, source_op, low_scalar_base,
                                      low_scalar_term, &low_scalar_base));
      continue;
    }

    loom_value_id_t low_term_lo = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_term_hi = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_dynamic_term(
        context, source_op, term, vgpr_type, &low_term_lo, &low_term_hi));
    if (low_vaddr_lo == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_low_slice(context, source_op, low_scalar_base,
                                     /*offset=*/0, sgpr_type, &low_vaddr_lo));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_low_slice(context, source_op, low_scalar_base,
                                     /*offset=*/1, sgpr_type, &low_vaddr_hi));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_add_term(
        context, source_op, low_term_lo, low_term_hi, vgpr_type, sgpr_x2_type,
        &low_vaddr_lo, &low_vaddr_hi));
  }

  if (low_vaddr_lo == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_low_slice(context, source_op, low_scalar_base,
                                   /*offset=*/0, sgpr_type, &low_vaddr_lo));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_low_slice(context, source_op, low_scalar_base,
                                   /*offset=*/1, sgpr_type, &low_vaddr_hi));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_vaddr_lo, &low_vaddr_lo));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_vaddr_hi, &low_vaddr_hi));
  loom_value_id_t sources[] = {
      low_vaddr_lo,
      low_vaddr_hi,
  };
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), sources, IREE_ARRAYSIZE(sources),
      vgpr_x2_type, source_op->location, &concat_op));
  *out_low_vaddr = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

typedef struct loom_amdgpu_memory_cache_attr_field_t {
  // Presence bit required for this descriptor attribute.
  loom_amdgpu_memory_cache_policy_attr_flags_t flag;
  // Descriptor attribute name.
  iree_string_view_t name;
  // Byte offset to the encoded attribute value field.
  iree_host_size_t value_offset;
} loom_amdgpu_memory_cache_attr_field_t;

static const loom_amdgpu_memory_cache_attr_field_t
    kLoomAmdgpuMemoryCacheAttrFields[] = {
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE,
            .name = {.data = "scope", .size = 5},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, scope),
        },
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH,
            .name = {.data = "th", .size = 2},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, th),
        },
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT,
            .name = {.data = "nt", .size = 2},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, nt),
        },
};

static int64_t loom_amdgpu_memory_cache_attr_field_value(
    const loom_amdgpu_memory_cache_policy_attrs_t* cache_attrs,
    const loom_amdgpu_memory_cache_attr_field_t* field) {
  const uint8_t* attrs_bytes = (const uint8_t*)cache_attrs;
  const void* value_bytes = attrs_bytes + field->value_offset;
  return *(const int64_t*)value_bytes;
}

static iree_status_t loom_amdgpu_append_memory_cache_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  if (!loom_amdgpu_memory_cache_policy_is_present(policy)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_memory_cache_policy_attrs_t cache_attrs = {0};
  const bool encoded = loom_amdgpu_memory_cache_policy_encode(
      descriptor_set, access, &cache_attrs);
  IREE_ASSERT(encoded);

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuMemoryCacheAttrFields); ++i) {
    const loom_amdgpu_memory_cache_attr_field_t* field =
        &kLoomAmdgpuMemoryCacheAttrFields[i];
    if (!iree_any_bit_set(cache_attrs.flags, field->flag)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, field->name,
        loom_amdgpu_memory_cache_attr_field_value(&cache_attrs, field), attrs,
        attr_capacity, inout_attr_count));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_make_memory_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset0"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset1"), access->secondary_immediate_offset, attrs,
        attr_capacity, out_attr_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
  }
  return loom_amdgpu_append_memory_cache_attrs(context, access, attrs,
                                               attr_capacity, out_attr_count);
}

static loom_value_id_t loom_amdgpu_memory_load_view(
    const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      return loom_vector_load_view(source_op);
    case LOOM_OP_VIEW_LOAD:
      return loom_view_load_view(source_op);
    default:
      IREE_ASSERT_UNREACHABLE("expected AMDGPU memory load op");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static loom_value_id_t loom_amdgpu_memory_load_result(
    const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      return loom_vector_load_result(source_op);
    case LOOM_OP_VIEW_LOAD:
      return loom_view_load_result(source_op);
    default:
      IREE_ASSERT_UNREACHABLE("expected AMDGPU memory load op");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_memory_load_result_is_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_is_vgpr) {
  *out_is_vgpr = false;
  const loom_value_id_t source_result =
      loom_amdgpu_memory_load_result(source_op);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  return loom_amdgpu_low_type_register_class_is(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, out_is_vgpr);
}

static iree_status_t loom_amdgpu_materialize_memory_load_packet_for_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool result_is_vgpr, loom_value_id_t low_packet,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = low_packet;
  if (!result_is_vgpr) {
    return iree_ok_status();
  }
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_packet, out_low_packet);
}

static bool loom_amdgpu_memory_load_packet_needs_signed_i16_repair(
    const loom_amdgpu_memory_access_t* access) {
  return access->source.memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         access->payload_format ==
             LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_SIGNED_16BIT_INTEGER;
}

static iree_status_t loom_amdgpu_emit_signed_i16_memory_load_repair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_packet, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), 0, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("width"), 16, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));

  const loom_value_id_t operands[] = {low_packet};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_I32_OFFSET_WIDTH_INLINE, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &lane_type, 1, &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_repair_memory_load_packet_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_packet,
    loom_value_id_t* out_low_result) {
  if (!loom_amdgpu_memory_load_packet_needs_signed_i16_repair(access)) {
    *out_low_result = low_packet;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_signed_i16_memory_load_repair(
      context, source_op, low_packet, out_low_result);
}

static loom_value_id_t loom_amdgpu_memory_store_value(
    const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_STORE:
      return loom_vector_store_value(source_op);
    case LOOM_OP_VIEW_STORE:
      return loom_view_store_value(source_op);
    default:
      IREE_ASSERT_UNREACHABLE("expected AMDGPU memory store op");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static loom_value_id_t loom_amdgpu_memory_store_view(
    const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_STORE:
      return loom_vector_store_view(source_op);
    case LOOM_OP_VIEW_STORE:
      return loom_view_store_view(source_op);
    default:
      IREE_ASSERT_UNREACHABLE("expected AMDGPU memory store op");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_bind_memory_load_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_result) {
  const loom_value_id_t source_result =
      loom_amdgpu_memory_load_result(source_op);
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_load_result_is_vgpr(
      context, source_op, &result_is_vgpr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_memory_load_packet_for_result(
      context, source_op, result_is_vgpr, low_result, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_memory_packet_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_memory_access_t* access = &packet->access;
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  if (access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_amdgpu_memory_load_view(source_op), &low_resource));
  }

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
          context, source_op, access, low_resource, &low_vaddr));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
          context, source_op, access, LOOM_VALUE_ID_INVALID, &low_vaddr));
    }
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_memory_payload_low_type(context, access, &result_type));

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      const loom_low_lower_resolved_descriptor_t packet_descriptor = {
          .descriptor = access->descriptor,
          .opcode_id = packet->opcode_id,
      };
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_m0_u32(
          context, source_op, &packet_descriptor, 0, &low_m0));
      loom_value_id_t operands[] = {low_m0};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
          &low_op));
      const loom_value_id_t raw_result =
          loom_value_slice_get(loom_low_op_results(low_op), 0);
      return loom_amdgpu_repair_memory_load_packet_result(
          context, source_op, access, raw_result, out_low_result);
    }
    loom_value_id_t operands[] = {low_vaddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, access, low_resource, &low_saddr));
    loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_memory_descriptor_has_implicit_resource_operand(context,
                                                                    packet)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_implicit_m0(
          context, source_op, packet, &low_m0));
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_saddr,
        low_m0,
    };
    const iree_host_size_t operand_count =
        low_m0 == LOOM_VALUE_ID_INVALID ? 2 : 3;
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, operand_count,
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    loom_value_id_t low_sbase = low_resource;
    loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
    if (access->scalar_offset_placement ==
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
          context, source_op, access, low_resource, &low_sbase));
      loom_type_t sgpr_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
          sgpr_type, &low_soffset));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_soffset(
          context, source_op, access, &low_soffset));
    }
    loom_value_id_t operands[] = {
        low_sbase,
        low_soffset,
    };
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
        context, source_op, low_resource, &access->source, &low_descriptor));
    loom_value_id_t operands[] = {low_descriptor};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
    loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_memory_descriptor_has_implicit_resource_operand(context,
                                                                    packet)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_implicit_m0(
          context, source_op, packet, &low_m0));
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_m0,
    };
    const iree_host_size_t operand_count =
        low_m0 == LOOM_VALUE_ID_INVALID ? 1 : 2;
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, operand_count,
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_soffset(context, source_op,
                                                       access, &low_soffset));
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, low_resource, &access->source, &low_descriptor));
  loom_value_id_t operands[] = {
      low_descriptor,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
      context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, &low_op));
  const loom_value_id_t raw_result =
      loom_value_slice_get(loom_low_op_results(low_op), 0);
  return loom_amdgpu_repair_memory_load_packet_result(
      context, source_op, access, raw_result, out_low_result);
}

static iree_status_t loom_amdgpu_lower_memory_packet_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    loom_value_id_t source_value, loom_value_id_t low_value) {
  const loom_amdgpu_memory_access_t* access = &packet->access;
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_memory_store_payload_vgpr(
      context, source_op, access, source_value, packet->source_register_offset,
      low_value, &low_value));
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  if (access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_amdgpu_memory_store_view(source_op), &low_resource));
  }

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
          context, source_op, access, low_resource, &low_vaddr));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
          context, source_op, access, LOOM_VALUE_ID_INVALID, &low_vaddr));
    }
  }

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      const loom_low_lower_resolved_descriptor_t packet_descriptor = {
          .descriptor = access->descriptor,
          .opcode_id = packet->opcode_id,
      };
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_m0_u32(
          context, source_op, &packet_descriptor, 0, &low_m0));
      loom_value_id_t operands[] = {
          low_value,
          low_m0,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
          /*result_count=*/0, &low_op);
    }
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_value0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, 0, lane_type, &low_value0));
      loom_value_id_t low_value1 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, 1, lane_type, &low_value1));
      loom_value_id_t operands[] = {
          low_vaddr,
          low_value0,
          low_value1,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
          /*result_count=*/0, &low_op);
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
        context, source_op, low_resource, &access->source, &low_descriptor));
    loom_value_id_t operands[] = {
        low_value,
        low_descriptor,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, access, low_resource, &low_saddr));
    loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_memory_descriptor_has_implicit_resource_operand(context,
                                                                    packet)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_implicit_m0(
          context, source_op, packet, &low_m0));
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
        low_saddr,
        low_m0,
    };
    const iree_host_size_t operand_count =
        low_m0 == LOOM_VALUE_ID_INVALID ? 3 : 4;
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, operand_count,
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
    loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_memory_descriptor_has_implicit_resource_operand(context,
                                                                    packet)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_implicit_m0(
          context, source_op, packet, &low_m0));
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
        low_m0,
    };
    const iree_host_size_t operand_count =
        low_m0 == LOOM_VALUE_ID_INVALID ? 2 : 3;
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, operand_count,
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_soffset(context, source_op,
                                                       access, &low_soffset));
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, low_resource, &access->source, &low_descriptor));
  loom_value_id_t operands[] = {
      low_value,
      low_descriptor,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_memory_packet(
      context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, &low_op);
}

iree_status_t loom_amdgpu_lower_memory_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan) {
  IREE_ASSERT_GT(plan->packet_count, 0);
  if (plan->packet_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_load(
        context, source_op, &plan->packets[0], &low_result));
    return loom_amdgpu_bind_memory_load_result(context, source_op, low_result);
  }

  loom_value_id_t low_results[LOOM_AMDGPU_MAX_MEMORY_PACKET_COUNT];
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_load_result_is_vgpr(
      context, source_op, &result_is_vgpr));
  for (uint32_t i = 0; i < plan->packet_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_load(
        context, source_op, &plan->packets[i], &low_results[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_memory_load_packet_for_result(
        context, source_op, result_is_vgpr, low_results[i], &low_results[i]));
  }

  const loom_value_id_t source_result =
      loom_amdgpu_memory_load_result(source_op);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_results, plan->packet_count,
      result_type, source_op->location, &concat_op));
  return loom_amdgpu_bind_memory_load_result(context, source_op,
                                             loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_memory_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan) {
  IREE_ASSERT_GT(plan->packet_count, 0);
  const loom_value_id_t source_value =
      loom_amdgpu_memory_store_value(source_op);
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  if (plan->packet_count == 1) {
    return loom_amdgpu_lower_memory_packet_store(
        context, source_op, &plan->packets[0], source_value, low_value);
  }

  for (uint32_t i = 0; i < plan->packet_count; ++i) {
    const loom_amdgpu_memory_packet_plan_t* packet = &plan->packets[i];
    loom_type_t packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_memory_payload_low_type(
        context, &packet->access, &packet_type));
    loom_value_id_t packet_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_value, packet->source_register_offset,
        packet_type, &packet_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_store(
        context, source_op, packet, source_value, packet_value));
  }
  return iree_ok_status();
}
