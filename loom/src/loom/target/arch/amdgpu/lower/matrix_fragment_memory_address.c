// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_address.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/types.h"

static loom_amdgpu_fragment_memory_address_register_kind_t
loom_amdgpu_fragment_memory_low_register_kind(loom_low_lower_context_t* context,
                                              loom_value_id_t low_value) {
  if (low_value == LOOM_VALUE_ID_INVALID) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 1) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR;
  }

  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr && loom_low_register_type_unit_count(low_type) == 1) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU fragment memory address selected a non-scalar register");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_add_address_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_term,
    loom_amdgpu_fragment_memory_address_register_kind_t term_register_kind,
    loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  if (term_register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    return iree_ok_status();
  }
  if (inout_accumulator->register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    *inout_accumulator = (loom_amdgpu_fragment_memory_address_accumulator_t){
        .value = low_term,
        .register_kind = term_register_kind,
    };
    return iree_ok_status();
  }
  if (inout_accumulator->register_kind ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR &&
      term_register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        inout_accumulator->value, low_term, sgpr_type,
        &inout_accumulator->value));
    return iree_ok_status();
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  if (inout_accumulator->register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR) {
    if (term_register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      low_lhs = low_term;
      low_rhs = inout_accumulator->value;
    } else {
      low_lhs = inout_accumulator->value;
      low_rhs = low_term;
    }
  } else {
    low_lhs = inout_accumulator->value;
    low_rhs = low_term;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, low_lhs,
      low_rhs, vgpr_type, &inout_accumulator->value));
  inout_accumulator->register_kind =
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_memory_address_base_key_is_equal(
    const loom_amdgpu_fragment_memory_address_base_key_t* lhs,
    const loom_amdgpu_fragment_memory_address_base_key_t* rhs) {
  if (lhs->block != rhs->block ||
      !loom_type_equal(lhs->vgpr_type, rhs->vgpr_type) ||
      lhs->lane_term_count != rhs->lane_term_count ||
      lhs->dynamic_term_count != rhs->dynamic_term_count) {
    return false;
  }
  if (memcmp(lhs->lane_terms, rhs->lane_terms,
             lhs->lane_term_count * sizeof(lhs->lane_terms[0])) != 0) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->dynamic_term_count; ++i) {
    if (lhs->dynamic_values[i] != rhs->dynamic_values[i] ||
        lhs->dynamic_byte_strides[i] != rhs->dynamic_byte_strides[i]) {
      return false;
    }
  }
  return true;
}

uint64_t loom_amdgpu_fragment_memory_relative_lane_byte_offset(
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    uint8_t lane) {
  if (address_layout->linear_lane_byte_stride != 0) {
    return (uint64_t)lane * address_layout->linear_lane_byte_stride;
  }
  uint64_t byte_offset = 0;
  for (uint8_t i = 0; i < address_layout->lane_term_count; ++i) {
    const loom_amdgpu_fragment_memory_lane_term_t* term =
        &address_layout->lane_terms[i];
    uint64_t digit = lane / term->divisor;
    if (term->modulus > 1) digit %= term->modulus;
    byte_offset += digit * term->byte_stride;
  }
  return byte_offset;
}

bool loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint8_t term_index) {
  // The source plan keeps normalized dynamic terms for legality, but the
  // original byte-domain view-base value may already be materialized and shared
  // by nearby fragment ops. When using that value, emission subtracts the
  // extracted static view-base delta from the immediate side of the address.
  return term_index == 0 && plan->source.dynamic_view_base_term_count == 1 &&
         plan->source.dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID;
}

static bool loom_amdgpu_fragment_memory_address_base_key_for_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_base_key_t* out_key) {
  memset(out_key, 0, sizeof(*out_key));
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  if (builder->ip.block == NULL || builder->ip.before_op != NULL) {
    return false;
  }
  out_key->block = builder->ip.block;
  out_key->vgpr_type = vgpr_type;
  out_key->lane_term_count = plan->address_layout.lane_term_count;
  memcpy(out_key->lane_terms, plan->address_layout.lane_terms,
         out_key->lane_term_count * sizeof(out_key->lane_terms[0]));
  out_key->dynamic_term_count = plan->source.dynamic_term_count;
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    if (term->stride_value_count != 0) {
      return false;
    }
    if (loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(plan, i)) {
      out_key->dynamic_values[i] = plan->source.dynamic_view_base_value_id;
      out_key->dynamic_byte_strides[i] = 1;
    } else {
      out_key->dynamic_values[i] = term->index;
      out_key->dynamic_byte_strides[i] = term->byte_stride;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_emit_fragment_memory_dynamic_source_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_type_t sgpr_type,
    loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    IREE_ASSERT_EQ(term->stride_value_count, 0u);
    IREE_ASSERT_GE(term->byte_stride, 0);
    IREE_ASSERT_LE(term->byte_stride, UINT32_MAX);

    const bool use_view_base_value =
        loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(plan, i);
    const loom_value_id_t source_value =
        use_view_base_value ? plan->source.dynamic_view_base_value_id
                            : term->index;
    const uint32_t byte_stride =
        use_view_base_value ? 1u : (uint32_t)term->byte_stride;
    loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_value, &low_index));
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    loom_amdgpu_fragment_memory_address_register_kind_t register_kind =
        loom_amdgpu_fragment_memory_low_register_kind(context, low_index);
    if (register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_scale_u32(
          context, source_op, low_index, byte_stride, sgpr_type, &low_term));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
          context, source_op, low_index, byte_stride,
          LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &low_term));
      register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term, register_kind, sgpr_type, vgpr_type,
        inout_accumulator));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_lane_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids, loom_type_t sgpr_type,
    loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  if (address_layout->linear_lane_byte_stride != 0) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, lane_ids->lane,
        address_layout->linear_lane_byte_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    return loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        inout_accumulator);
  }

  for (uint8_t i = 0; i < address_layout->lane_term_count; ++i) {
    const loom_amdgpu_fragment_memory_lane_term_t* term =
        &address_layout->lane_terms[i];
    loom_value_id_t low_digit = lane_ids->lane;
    if (term->divisor > 1) {
      if (term->divisor == address_layout->primary_lane_divisor) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_div(
            context, source_op, term->divisor, vgpr_type, lane_ids));
        low_digit = lane_ids->lane_div;
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
            (uint16_t)iree_math_count_trailing_zeros_u32(term->divisor),
            lane_ids->lane, vgpr_type, &low_digit));
      }
    }
    if (term->modulus > 1) {
      if (term->divisor == 1 &&
          term->modulus == address_layout->primary_lane_divisor) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_mod(
            context, source_op, term->modulus, vgpr_type, lane_ids));
        low_digit = lane_ids->lane_mod;
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
            low_digit, term->modulus - 1u, vgpr_type, &low_digit));
      }
    }

    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, low_digit, term->byte_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        inout_accumulator));
  }
  return iree_ok_status();
}

bool loom_amdgpu_fragment_memory_register_terms(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint64_t* out_static_byte_offset) {
  if (register_index >= plan->register_count) {
    return false;
  }
  *out_static_byte_offset =
      plan->address_layout.register_byte_offsets[register_index];
  return true;
}

static bool loom_amdgpu_fragment_memory_descriptor_offset_info(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_descriptor_offset_immediate_info_t* out_info) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const uint32_t descriptor_ordinal =
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return false;
  }
  if (loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          out_info)) {
    return true;
  }
  return loom_amdgpu_descriptor_offset_immediate_info(
      descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED,
      out_info);
}

bool loom_amdgpu_fragment_memory_register_group_is_contiguous(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t register_count, uint32_t register_byte_count) {
  if (register_count == 0 ||
      register_index + register_count > plan->register_count) {
    return false;
  }

  uint64_t base_static_byte_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(plan, register_index,
                                                  &base_static_byte_offset)) {
    return false;
  }
  for (uint16_t i = 1; i < register_count; ++i) {
    uint64_t static_byte_offset = 0;
    if (!loom_amdgpu_fragment_memory_register_terms(plan, register_index + i,
                                                    &static_byte_offset)) {
      return false;
    }
    if (static_byte_offset !=
        base_static_byte_offset + (uint64_t)i * register_byte_count) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_register_static_offset(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, uint64_t* out_static_byte_offset) {
  *out_static_byte_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(plan, register_index,
                                                  out_static_byte_offset)) {
    return false;
  }
  if (element_index == 0) {
    return true;
  }

  if (element_index >= plan->address_layout.payload_elements_per_register ||
      plan->address_layout.packed_element_byte_stride == 0) {
    return false;
  }
  const uint64_t element_static_offset =
      (uint64_t)element_index * plan->address_layout.packed_element_byte_stride;
  if (*out_static_byte_offset > UINT64_MAX - element_static_offset) {
    return false;
  }
  *out_static_byte_offset += element_static_offset;
  return true;
}

bool loom_amdgpu_fragment_memory_static_offset_i64(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, int64_t* out_static_byte_offset) {
  *out_static_byte_offset = plan->source.static_byte_offset;
  if (plan->source.static_byte_offset < 0) {
    return false;
  }
  uint64_t register_static_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_static_offset(
          plan, register_index, element_index, &register_static_offset) ||
      register_static_offset > INT64_MAX) {
    return false;
  }
  return iree_checked_add_i64(plan->source.static_byte_offset,
                              (int64_t)register_static_offset,
                              out_static_byte_offset);
}

bool loom_amdgpu_fragment_memory_vaddr_static_offset_u32(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, uint64_t* out_static_byte_offset) {
  *out_static_byte_offset = 0;
  if (plan->source.static_byte_offset < 0) {
    return false;
  }
  int64_t static_byte_offset = plan->source.static_byte_offset;
  if (loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(
          plan, /*term_index=*/0) &&
      !iree_checked_sub_i64(static_byte_offset,
                            plan->source.static_view_base_byte_offset,
                            &static_byte_offset)) {
    return false;
  }

  uint64_t register_static_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_static_offset(
          plan, register_index, element_index, &register_static_offset) ||
      register_static_offset > INT64_MAX ||
      !iree_checked_add_i64(static_byte_offset, (int64_t)register_static_offset,
                            &static_byte_offset) ||
      static_byte_offset < 0) {
    return false;
  }
  *out_static_byte_offset = (uint64_t)static_byte_offset;
  return *out_static_byte_offset <= UINT32_MAX;
}

static void loom_amdgpu_fragment_memory_split_static_offset(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_fact_memory_space_t memory_space, uint64_t static_byte_offset,
    uint64_t* out_vaddr_static_byte_offset, int64_t* out_immediate_offset) {
  *out_vaddr_static_byte_offset = static_byte_offset;
  *out_immediate_offset = 0;
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info = {0};
  if (!loom_amdgpu_fragment_memory_descriptor_offset_info(
          context, descriptor_ref, &offset_info) ||
      offset_info.unit_byte_count == 0) {
    return;
  }

  if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if ((static_byte_offset % offset_info.unit_byte_count) != 0) {
      return;
    }
    const uint64_t encoded_offset =
        static_byte_offset / offset_info.unit_byte_count;
    if (encoded_offset > offset_info.unsigned_max ||
        encoded_offset > INT64_MAX) {
      return;
    }
    *out_vaddr_static_byte_offset = 0;
    *out_immediate_offset = (int64_t)encoded_offset;
    return;
  }

  if (offset_info.unsigned_max == UINT64_MAX) {
    return;
  }
  const uint64_t window_unit_count = offset_info.unsigned_max + 1;
  if (offset_info.unit_byte_count > UINT64_MAX / window_unit_count) {
    return;
  }
  const uint64_t window_byte_count =
      window_unit_count * offset_info.unit_byte_count;
  if (window_byte_count == 0) {
    return;
  }
  const uint64_t window_base =
      (static_byte_offset / window_byte_count) * window_byte_count;
  const uint64_t window_offset = static_byte_offset - window_base;
  const uint64_t encoded_offset = window_offset / offset_info.unit_byte_count;
  if (encoded_offset > offset_info.unsigned_max || encoded_offset > INT64_MAX) {
    return;
  }
  const uint64_t immediate_byte_offset =
      encoded_offset * offset_info.unit_byte_count;
  *out_vaddr_static_byte_offset =
      window_base + (window_offset - immediate_byte_offset);
  *out_immediate_offset = (int64_t)encoded_offset;
}

iree_status_t loom_amdgpu_emit_fragment_memory_base_address_accumulator(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* out_accumulator) {
  *out_accumulator = (loom_amdgpu_fragment_memory_address_accumulator_t){
      .value = LOOM_VALUE_ID_INVALID,
      .register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE,
  };
  uint64_t unused_static_byte_offset = 0;
  // The compiled layout keeps register coordinates as static byte offsets. The
  // lane terms are shared by every packet in this fragment memory operation.
  if (!loom_amdgpu_fragment_memory_register_terms(plan, /*register_index=*/0,
                                                  &unused_static_byte_offset)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment address layout");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_amdgpu_fragment_memory_address_base_key_t key;
  const bool cacheable = loom_amdgpu_fragment_memory_address_base_key_for_plan(
      context, plan, vgpr_type, &key);
  loom_amdgpu_matrix_fragment_state_t* cache = NULL;
  if (cacheable) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_fragment_state(context, &cache));
    if (cache->address_base.accumulator.register_kind !=
            LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE &&
        loom_amdgpu_fragment_memory_address_base_key_is_equal(
            &cache->address_base.key, &key)) {
      *out_accumulator = cache->address_base.accumulator;
      return iree_ok_status();
    }
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_dynamic_source_terms(
      context, source_op, plan, sgpr_type, vgpr_type, out_accumulator));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_terms(
      context, source_op, &plan->address_layout, lane_ids, sgpr_type, vgpr_type,
      out_accumulator));
  if (cache != NULL && out_accumulator->register_kind !=
                           LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    cache->address_base.key = key;
    cache->address_base.accumulator = *out_accumulator;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fragment_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_fragment_memory_address_accumulator_t* base_accumulator,
    loom_value_id_t low_resource, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_t* out_address) {
  *out_address = (loom_amdgpu_fragment_memory_address_t){
      .low_vaddr = LOOM_VALUE_ID_INVALID,
      .immediate_offset = 0,
  };
  uint64_t static_byte_offset = 0;
  if (!loom_amdgpu_fragment_memory_vaddr_static_offset_u32(
          plan, register_index, element_index, &static_byte_offset)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment memory address range");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_amdgpu_fragment_memory_address_accumulator_t accumulator =
      *base_accumulator;

  loom_amdgpu_fragment_memory_split_static_offset(
      context, descriptor_ref, plan->source.memory_space, static_byte_offset,
      &static_byte_offset, &out_address->immediate_offset);

  if (static_byte_offset != 0) {
    if (accumulator.register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
      return loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          (uint32_t)static_byte_offset, vgpr_type, &out_address->low_vaddr);
    }
    if (accumulator.register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          (uint32_t)static_byte_offset, sgpr_type, &low_static_offset));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
          accumulator.value, low_static_offset, sgpr_type, &accumulator.value));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
          accumulator.value, (uint32_t)static_byte_offset, vgpr_type,
          &accumulator.value));
    }
  }

  if (accumulator.register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, &out_address->low_vaddr);
  }
  return loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, accumulator.value, &out_address->low_vaddr);
}
