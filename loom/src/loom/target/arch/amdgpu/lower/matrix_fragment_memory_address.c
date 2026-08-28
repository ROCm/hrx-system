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

typedef enum loom_amdgpu_fragment_memory_binary_operand_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_BINARY_OPERAND_LHS = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_BINARY_OPERAND_RHS = 1,
} loom_amdgpu_fragment_memory_binary_operand_t;

static iree_status_t loom_amdgpu_emit_fragment_memory_destructive_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs,
    loom_amdgpu_fragment_memory_binary_operand_t accumulator_operand,
    loom_type_t result_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = (uint16_t)accumulator_operand,
      .has_type_change = false,
  };
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, &tied_result, 1,
      source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

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
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_sgpr || is_vgpr) {
    const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
    if (unit_count == 1 || unit_count == 2) {
      return is_sgpr ? LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR
                     : LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
    }
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU fragment memory address selected an unsupported register");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_add_address_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_term,
    loom_amdgpu_fragment_memory_address_register_kind_t term_register_kind,
    loom_type_t sgpr_type, loom_type_t vgpr_type, bool may_reuse_accumulator,
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
  const bool reuse_vgpr_accumulator =
      may_reuse_accumulator &&
      inout_accumulator->register_kind ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  if (reuse_vgpr_accumulator) {
    low_lhs = low_term;
    low_rhs = inout_accumulator->value;
  } else if (inout_accumulator->register_kind ==
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
  if (reuse_vgpr_accumulator) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_destructive_vgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_RHS_TIED,
            low_lhs, low_rhs, LOOM_AMDGPU_FRAGMENT_MEMORY_BINARY_OPERAND_RHS,
            vgpr_type, &inout_accumulator->value));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, low_lhs,
        low_rhs, vgpr_type, &inout_accumulator->value));
  }
  inout_accumulator->register_kind =
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_memory_address_product_key_is_equal(
    const loom_amdgpu_fragment_memory_address_product_key_t* lhs,
    const loom_amdgpu_fragment_memory_address_product_key_t* rhs) {
  return lhs->value_count == rhs->value_count &&
         lhs->static_byte_coefficient == rhs->static_byte_coefficient &&
         memcmp(lhs->values, rhs->values,
                lhs->value_count * sizeof(lhs->values[0])) == 0;
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
    if (!loom_amdgpu_fragment_memory_address_product_key_is_equal(
            &lhs->dynamic_terms[i], &rhs->dynamic_terms[i])) {
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lhs->runtime_axes); ++i) {
    const loom_amdgpu_fragment_memory_runtime_axis_base_key_t* lhs_axis =
        &lhs->runtime_axes[i];
    const loom_amdgpu_fragment_memory_runtime_axis_base_key_t* rhs_axis =
        &rhs->runtime_axes[i];
    if (lhs_axis->lane_divisor != rhs_axis->lane_divisor ||
        lhs_axis->lane_modulus != rhs_axis->lane_modulus ||
        lhs_axis->lane_coordinate_scale != rhs_axis->lane_coordinate_scale ||
        !loom_amdgpu_fragment_memory_address_product_key_is_equal(
            &lhs_axis->byte_stride, &rhs_axis->byte_stride)) {
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
    loom_amdgpu_fragment_memory_address_product_key_t* product =
        &out_key->dynamic_terms[i];
    if (loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(plan, i)) {
      product->values[0] = plan->source.dynamic_view_base_value_id;
      product->value_count = 1;
      product->static_byte_coefficient = 1;
    } else {
      product->values[0] = term->index;
      for (uint8_t j = 0; j < term->stride_value_count; ++j) {
        product->values[j + 1] = term->stride_values[j];
      }
      product->value_count = (uint8_t)(term->stride_value_count + 1u);
      product->static_byte_coefficient = term->byte_stride;
    }
  }
  for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* plan_axis =
        &plan->runtime_axes[view_axis];
    loom_amdgpu_fragment_memory_runtime_axis_base_key_t* key_axis =
        &out_key->runtime_axes[view_axis];
    key_axis->lane_divisor = plan_axis->lane_divisor;
    key_axis->lane_modulus = plan_axis->lane_modulus;
    key_axis->lane_coordinate_scale = plan_axis->lane_coordinate_scale;
    key_axis->byte_stride.value_count =
        plan_axis->byte_stride.dynamic_factor_count;
    key_axis->byte_stride.static_byte_coefficient =
        plan_axis->byte_stride.static_byte_coefficient;
    for (uint8_t j = 0; j < plan_axis->byte_stride.dynamic_factor_count; ++j) {
      key_axis->byte_stride.values[j] =
          plan_axis->byte_stride.dynamic_factors[j];
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_fragment_memory_lookup_product_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_facts_t product_facts,
    loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* out_value) {
  *out_value = (loom_amdgpu_fragment_memory_address_accumulator_t){
      .value = LOOM_VALUE_ID_INVALID,
      .register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE,
  };
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  loom_amdgpu_fragment_memory_address_register_kind_t register_kind =
      loom_amdgpu_fragment_memory_low_register_kind(context, low_value);
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count == 2) {
    // Address products are computed modulo 2^32. Planning proves the complete
    // product fits u32, so the low half of a wide factor is sufficient.
    IREE_ASSERT(loom_value_facts_fit_unsigned_bit_count(product_facts, 32));
    const loom_type_t narrow_type =
        register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR
            ? sgpr_type
            : vgpr_type;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_value, /*offset=*/0, narrow_type, &low_value));
  } else {
    IREE_ASSERT_EQ(unit_count, 1u);
  }
  *out_value = (loom_amdgpu_fragment_memory_address_accumulator_t){
      .value = low_value,
      .register_kind = register_kind,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_memory_multiply_address_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_address_accumulator_t lhs,
    loom_amdgpu_fragment_memory_address_accumulator_t rhs,
    loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* out_product) {
  if (lhs.register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR &&
      rhs.register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    out_product->register_kind =
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR;
    return loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32, lhs.value,
        rhs.value, sgpr_type, &out_product->value);
  }

  loom_value_id_t low_lhs = lhs.value;
  loom_value_id_t low_rhs = rhs.value;
  if (rhs.register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    low_lhs = rhs.value;
    low_rhs = lhs.value;
  }
  out_product->register_kind =
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32, low_lhs,
      low_rhs, vgpr_type, &out_product->value);
}

static iree_status_t loom_amdgpu_fragment_memory_materialize_product(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* source_values, uint8_t source_value_count,
    uint32_t static_coefficient, loom_value_facts_t product_facts,
    loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* out_product) {
  IREE_ASSERT_GT(source_value_count, 0u);
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_lookup_product_value(
      context, source_op, source_values[0], product_facts, sgpr_type, vgpr_type,
      out_product));
  for (uint8_t i = 1; i < source_value_count; ++i) {
    loom_amdgpu_fragment_memory_address_accumulator_t factor;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_lookup_product_value(
        context, source_op, source_values[i], product_facts, sgpr_type,
        vgpr_type, &factor));
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_multiply_address_values(
        context, source_op, *out_product, factor, sgpr_type, vgpr_type,
        out_product));
  }
  if (static_coefficient == 1) {
    return iree_ok_status();
  }
  if (out_product->register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    return loom_amdgpu_emit_sgpr_scale_u32(
        context, source_op, out_product->value, static_coefficient, sgpr_type,
        &out_product->value);
  }
  return loom_amdgpu_emit_vgpr_scale_u32(
      context, source_op, out_product->value, static_coefficient,
      LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &out_product->value);
}

static bool loom_amdgpu_fragment_memory_axis_stride_product_is_equal(
    const loom_low_source_memory_axis_byte_stride_t* lhs,
    const loom_low_source_memory_axis_byte_stride_t* rhs) {
  return lhs->static_byte_coefficient == rhs->static_byte_coefficient &&
         lhs->dynamic_factor_count == rhs->dynamic_factor_count &&
         memcmp(lhs->dynamic_factors, rhs->dynamic_factors,
                lhs->dynamic_factor_count * sizeof(lhs->dynamic_factors[0])) ==
             0;
}

static bool loom_amdgpu_fragment_memory_term_uses_axis_stride(
    const loom_low_source_memory_dynamic_term_t* term,
    const loom_low_source_memory_axis_byte_stride_t* axis_stride) {
  return term->byte_stride == axis_stride->static_byte_coefficient &&
         term->stride_value_count == axis_stride->dynamic_factor_count &&
         memcmp(term->stride_values, axis_stride->dynamic_factors,
                term->stride_value_count * sizeof(term->stride_values[0])) == 0;
}

static iree_status_t loom_amdgpu_emit_fragment_memory_dynamic_source_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_type_t sgpr_type,
    loom_type_t vgpr_type,
    const loom_amdgpu_fragment_memory_address_state_t* address_state,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    IREE_ASSERT_GE(term->byte_stride, 0);
    IREE_ASSERT_LE(term->byte_stride, UINT32_MAX);

    const bool use_view_base_value =
        loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(plan, i);
    loom_amdgpu_fragment_memory_address_accumulator_t low_term;
    if (use_view_base_value) {
      const loom_value_id_t source_value =
          plan->source.dynamic_view_base_value_id;
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_materialize_product(
          context, source_op, &source_value, /*source_value_count=*/1,
          /*static_coefficient=*/1, term->byte_facts, sgpr_type, vgpr_type,
          &low_term));
    } else {
      const loom_amdgpu_fragment_memory_address_accumulator_t*
          shared_axis_stride = NULL;
      for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
        const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
            &plan->runtime_axes[view_axis];
        if (runtime_axis->byte_stride.kind !=
            LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC) {
          continue;
        }
        if (loom_amdgpu_fragment_memory_term_uses_axis_stride(
                term, &runtime_axis->byte_stride)) {
          shared_axis_stride =
              &address_state->runtime_axes[view_axis].byte_stride;
          break;
        }
      }
      if (shared_axis_stride != NULL) {
        loom_amdgpu_fragment_memory_address_accumulator_t low_index;
        IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_lookup_product_value(
            context, source_op, term->index, term->byte_facts, sgpr_type,
            vgpr_type, &low_index));
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_fragment_memory_multiply_address_values(
                context, source_op, low_index, *shared_axis_stride, sgpr_type,
                vgpr_type, &low_term));
      } else {
        loom_value_id_t source_values
            [LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_PRODUCT_VALUE_CAPACITY] = {
                term->index};
        for (uint8_t j = 0; j < term->stride_value_count; ++j) {
          source_values[j + 1] = term->stride_values[j];
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_materialize_product(
            context, source_op, source_values,
            (uint8_t)(term->stride_value_count + 1u),
            (uint32_t)term->byte_stride, term->byte_facts, sgpr_type, vgpr_type,
            &low_term));
      }
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term.value, low_term.register_kind, sgpr_type,
        vgpr_type, /*may_reuse_accumulator=*/false, inout_accumulator));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_lane_digit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t divisor, uint16_t modulus, uint16_t primary_lane_divisor,
    loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_value_id_t* out_low_digit) {
  *out_low_digit = lane_ids->lane;
  if (divisor > 1) {
    if (divisor == primary_lane_divisor) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_div(
          context, source_op, divisor, vgpr_type, lane_ids));
      *out_low_digit = lane_ids->lane_div;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          (uint16_t)iree_math_count_trailing_zeros_u32(divisor), lane_ids->lane,
          vgpr_type, out_low_digit));
    }
  }
  if (modulus > 1) {
    if (divisor == 1 && modulus == primary_lane_divisor) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_mod(
          context, source_op, modulus, vgpr_type, lane_ids));
      *out_low_digit = lane_ids->lane_mod;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          *out_low_digit, modulus - 1u, vgpr_type, out_low_digit));
    }
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
        /*may_reuse_accumulator=*/false, inout_accumulator);
  }

  for (uint8_t i = 0; i < address_layout->lane_term_count; ++i) {
    const loom_amdgpu_fragment_memory_lane_term_t* term =
        &address_layout->lane_terms[i];
    loom_value_id_t low_digit = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_digit(
        context, source_op, term->divisor, term->modulus,
        address_layout->primary_lane_divisor, lane_ids, vgpr_type, &low_digit));

    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, low_digit, term->byte_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        /*may_reuse_accumulator=*/false, inout_accumulator));
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
    for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
      const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
          &plan->runtime_axes[view_axis];
      if (runtime_axis->register_coordinates[register_index + i] !=
          runtime_axis->register_coordinates[register_index]) {
        return false;
      }
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

  if (element_index >= plan->address_layout.payload_elements_per_register) {
    return false;
  }
  if (plan->address_layout.packed_element_byte_stride == 0) {
    for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
      if (plan->runtime_axes[view_axis].packed_element_coordinate_stride != 0) {
        return true;
      }
    }
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

bool loom_amdgpu_fragment_memory_runtime_packet_offset_is_subgroup_uniform(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index) {
  IREE_ASSERT_LT(register_index, plan->register_count);
  for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
        &plan->runtime_axes[view_axis];
    if (runtime_axis->lane_coordinate_scale != 0) {
      return false;
    }
    const uint64_t packet_coordinate =
        (uint64_t)runtime_axis->register_coordinates[register_index] +
        (uint64_t)element_index *
            runtime_axis->packed_element_coordinate_stride;
    if (packet_coordinate != 0 && !loom_value_facts_is_subgroup_uniform(
                                      runtime_axis->byte_stride.byte_facts)) {
      return false;
    }
  }
  return true;
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

iree_status_t loom_amdgpu_initialize_fragment_memory_address_state(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_state_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  out_state->cursor.value = LOOM_VALUE_ID_INVALID;
  uint64_t unused_static_byte_offset = 0;
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
    if (cache->address_base.valid &&
        loom_amdgpu_fragment_memory_address_base_key_is_equal(
            &cache->address_base.key, &key)) {
      out_state->cursor = cache->address_base.accumulator;
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(out_state->runtime_axes);
           ++i) {
        out_state->runtime_axes[i].byte_stride =
            cache->address_base.runtime_axis_byte_strides[i];
      }
      return iree_ok_status();
    }
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
        &plan->runtime_axes[view_axis];
    if (runtime_axis->byte_stride.kind !=
        LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC) {
      continue;
    }
    loom_amdgpu_fragment_memory_runtime_axis_address_state_t* axis_state =
        &out_state->runtime_axes[view_axis];
    bool reused_product = false;
    for (uint8_t previous_axis = 0; previous_axis < view_axis;
         ++previous_axis) {
      if (plan->runtime_axes[previous_axis].byte_stride.kind !=
          LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC) {
        continue;
      }
      if (loom_amdgpu_fragment_memory_axis_stride_product_is_equal(
              &runtime_axis->byte_stride,
              &plan->runtime_axes[previous_axis].byte_stride)) {
        axis_state->byte_stride =
            out_state->runtime_axes[previous_axis].byte_stride;
        reused_product = true;
        break;
      }
    }
    if (!reused_product) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_materialize_product(
          context, source_op, runtime_axis->byte_stride.dynamic_factors,
          runtime_axis->byte_stride.dynamic_factor_count,
          (uint32_t)runtime_axis->byte_stride.static_byte_coefficient,
          runtime_axis->byte_stride.byte_facts, sgpr_type, vgpr_type,
          &axis_state->byte_stride));
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_dynamic_source_terms(
      context, source_op, plan, sgpr_type, vgpr_type, out_state,
      &out_state->cursor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_terms(
      context, source_op, &plan->address_layout, lane_ids, sgpr_type, vgpr_type,
      &out_state->cursor));
  for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
        &plan->runtime_axes[view_axis];
    loom_amdgpu_fragment_memory_runtime_axis_address_state_t* axis_state =
        &out_state->runtime_axes[view_axis];
    if (runtime_axis->lane_coordinate_scale == 0) {
      continue;
    }
    loom_value_id_t low_lane_coordinate = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_digit(
        context, source_op, runtime_axis->lane_divisor,
        runtime_axis->lane_modulus, plan->address_layout.primary_lane_divisor,
        lane_ids, vgpr_type, &low_lane_coordinate));
    if (runtime_axis->lane_coordinate_scale != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
          context, source_op, low_lane_coordinate,
          runtime_axis->lane_coordinate_scale,
          LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
          &low_lane_coordinate));
    }
    const loom_amdgpu_fragment_memory_address_accumulator_t lane_coordinate = {
        .value = low_lane_coordinate,
        .register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR,
    };
    loom_amdgpu_fragment_memory_address_accumulator_t lane_byte_offset;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_multiply_address_values(
        context, source_op, axis_state->byte_stride, lane_coordinate, sgpr_type,
        vgpr_type, &lane_byte_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, lane_byte_offset.value,
        lane_byte_offset.register_kind, sgpr_type, vgpr_type,
        /*may_reuse_accumulator=*/false, &out_state->cursor));
  }
  if (cache != NULL) {
    cache->address_base.valid = true;
    cache->address_base.key = key;
    cache->address_base.accumulator = out_state->cursor;
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(cache->address_base.runtime_axis_byte_strides);
         ++i) {
      cache->address_base.runtime_axis_byte_strides[i] =
          out_state->runtime_axes[i].byte_stride;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_memory_scale_address_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_address_accumulator_t value, uint32_t scale,
    loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* out_scaled_value) {
  *out_scaled_value = value;
  if (scale == 1) {
    return iree_ok_status();
  }
  if (value.register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    return loom_amdgpu_emit_sgpr_scale_u32(context, source_op, value.value,
                                           scale, sgpr_type,
                                           &out_scaled_value->value);
  }
  return loom_amdgpu_emit_vgpr_scale_u32(context, source_op, value.value, scale,
                                         LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
                                         vgpr_type, &out_scaled_value->value);
}

static iree_status_t loom_amdgpu_fragment_memory_subtract_address_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_address_accumulator_t term,
    loom_type_t sgpr_type, loom_type_t vgpr_type, bool may_reuse_accumulator,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  IREE_ASSERT_NE(inout_accumulator->register_kind,
                 LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE);
  if (inout_accumulator->register_kind ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR &&
      term.register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    return loom_amdgpu_emit_sgpr_binary(context, source_op,
                                        LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_U32,
                                        inout_accumulator->value, term.value,
                                        sgpr_type, &inout_accumulator->value);
  }
  if (term.register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, term.value, &term.value));
  }
  const bool reuse_vgpr_accumulator =
      may_reuse_accumulator &&
      inout_accumulator->register_kind ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  if (reuse_vgpr_accumulator) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_destructive_vgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32_LHS_TIED,
            inout_accumulator->value, term.value,
            LOOM_AMDGPU_FRAGMENT_MEMORY_BINARY_OPERAND_LHS, vgpr_type,
            &inout_accumulator->value));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32,
        inout_accumulator->value, term.value, vgpr_type,
        &inout_accumulator->value));
  }
  inout_accumulator->register_kind =
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_memory_advance_runtime_coordinates(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_state_t* address_state) {
  for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
        &plan->runtime_axes[view_axis];
    const uint64_t target_coordinate_u64 =
        (uint64_t)runtime_axis->register_coordinates[register_index] +
        (uint64_t)element_index *
            runtime_axis->packed_element_coordinate_stride;
    IREE_ASSERT_LE(target_coordinate_u64, UINT32_MAX);
    const uint32_t target_coordinate = (uint32_t)target_coordinate_u64;
    const uint32_t current_coordinate =
        address_state->current_coordinates[view_axis];
    if (target_coordinate == current_coordinate) {
      continue;
    }
    const uint32_t coordinate_magnitude =
        target_coordinate > current_coordinate
            ? target_coordinate - current_coordinate
            : current_coordinate - target_coordinate;
    loom_amdgpu_fragment_memory_runtime_axis_address_state_t* axis_state =
        &address_state->runtime_axes[view_axis];
    loom_amdgpu_fragment_memory_address_accumulator_t scaled_byte_stride =
        axis_state->byte_stride;
    if (coordinate_magnitude > 1) {
      if (axis_state->scaled_coordinate != coordinate_magnitude) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_scale_address_value(
            context, source_op, axis_state->byte_stride, coordinate_magnitude,
            sgpr_type, vgpr_type, &axis_state->scaled_byte_stride));
        axis_state->scaled_coordinate = coordinate_magnitude;
      }
      scaled_byte_stride = axis_state->scaled_byte_stride;
    }
    if (target_coordinate > current_coordinate) {
      const bool cursor_was_present =
          address_state->cursor.register_kind !=
          LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
          context, source_op, scaled_byte_stride.value,
          scaled_byte_stride.register_kind, sgpr_type, vgpr_type,
          plan->packet_address_sources_consumed_at_issue &&
              address_state->cursor_storage_is_distinct,
          &address_state->cursor));
      address_state->cursor_storage_is_distinct =
          address_state->cursor_storage_is_distinct || cursor_was_present;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_subtract_address_term(
          context, source_op, scaled_byte_stride, sgpr_type, vgpr_type,
          plan->packet_address_sources_consumed_at_issue &&
              address_state->cursor_storage_is_distinct,
          &address_state->cursor));
      address_state->cursor_storage_is_distinct = true;
    }
    address_state->current_coordinates[view_axis] = target_coordinate;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fragment_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_fragment_memory_address_state_t* address_state,
    loom_type_t vgpr_type, loom_amdgpu_fragment_memory_address_t* out_address) {
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_advance_runtime_coordinates(
      context, source_op, plan, register_index, element_index, sgpr_type,
      vgpr_type, address_state));
  loom_amdgpu_fragment_memory_address_accumulator_t accumulator =
      address_state->cursor;

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
