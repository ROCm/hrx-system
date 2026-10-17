// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_address.h"

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/realization.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/source_integer_representation.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static_assert(LOOM_LOW_SOURCE_MEMORY_DYNAMIC_REALIZATION_CAPACITY <= 8,
              "VADDR realization proofs must fit the retained mask");

static_assert(2 * (LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY + 1) < 64,
              "memory operand forms must leave the high bit for literals");

static loom_amdgpu_memory_dynamic_term_plan_t
loom_amdgpu_plan_memory_dynamic_term(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_source_memory_dynamic_term_t* term) {
  const loom_value_facts_t index_facts =
      loom_value_fact_table_lookup(fact_table, term->index);
  const bool has_product =
      term->stride_value_count != 0 ||
      (term->byte_stride != 1 &&
       term->byte_shift == LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE);
  loom_amdgpu_memory_dynamic_term_plan_t plan = {
      .index_minimum = index_facts.range_lo,
      .index_maximum = index_facts.range_hi,
      .operand_forms = loom_amdgpu_select_integer_operand_form(
          index_facts, has_product ? 1 : 2),
  };
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    const loom_amdgpu_integer_operand_form_t form =
        loom_amdgpu_select_integer_operand_form(
            loom_value_fact_table_lookup(fact_table, term->stride_values[i]),
            1);
    plan.operand_forms |= (uint64_t)form << (2 * (i + 1));
  }
  int64_t literal = 0;
  if (loom_amdgpu_type_is_address_scalar(
          loom_module_value_type(module, term->index)) &&
      loom_value_facts_as_exact_i64(index_facts, &literal) && literal >= 0 &&
      literal <= UINT32_MAX) {
    plan.operand_forms |= LOOM_AMDGPU_MEMORY_DYNAMIC_TERM_INDEX_LITERAL;
  }
  return plan;
}

iree_status_t loom_amdgpu_plan_memory_dynamic_terms(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_dynamic_term_plan_t** out_plans) {
  *out_plans = NULL;
  const uint32_t count = source->dynamic_term_count +
                         source->dynamic_realization_count +
                         (source->retained_component.term != NULL);
  if (count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_memory_dynamic_term_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, count * sizeof(*plans), (void**)&plans));
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint32_t index = 0;
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    plans[index++] = loom_amdgpu_plan_memory_dynamic_term(
        module, fact_table, &source->dynamic_terms[i]);
  }
  for (uint8_t i = 0; i < source->dynamic_realization_count; ++i) {
    plans[index++] = loom_amdgpu_plan_memory_dynamic_term(
        module, fact_table, &source->dynamic_realizations[i].term);
  }
  if (source->retained_component.term) {
    plans[index] = loom_amdgpu_plan_memory_dynamic_term(
        module, fact_table, source->retained_component.term);
  }
  *out_plans = plans;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_plan_memory_term_sequence(
    loom_low_lower_context_t* context,
    loom_amdgpu_memory_dynamic_term_sequence_t* sequence) {
  if (sequence->count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_memory_dynamic_term_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sequence->count * sizeof(*plans), (void**)&plans));
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  for (uint8_t i = 0; i < sequence->count; ++i) {
    plans[i] = loom_amdgpu_plan_memory_dynamic_term(module, fact_table,
                                                    sequence->terms[i]);
    sequence->plans[i] = &plans[i];
  }
  return iree_ok_status();
}

static loom_amdgpu_integer_operand_form_t loom_amdgpu_memory_term_operand_form(
    const loom_amdgpu_memory_dynamic_term_plan_t* plan, uint8_t operand_index) {
  return (loom_amdgpu_integer_operand_form_t)((plan->operand_forms >>
                                               (2 * operand_index)) &
                                              UINT64_C(3));
}

static iree_status_t loom_amdgpu_normalize_signed_memory_index_word(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t low_value,
    loom_value_id_t* out_low_value) {
  const loom_type_t lane_type =
      loom_module_value_type(loom_low_lower_context_module(context), low_value);
  return loom_amdgpu_normalize_narrow_source_integer(
      context, source_op, source_value, low_value,
      LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED, lane_type,
      out_low_value);
}

static void loom_amdgpu_memory_access_select_retained_component(
    loom_amdgpu_memory_access_t* access) {
  access->retained_component_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE;
  const loom_low_source_memory_dynamic_component_t* component =
      &access->source->retained_component;
  if (component->term == NULL ||
      access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
    return;
  }
  bool has_vaddr = false;
  bool has_soffset = false;
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    if (!(component->term_mask & (1u << i))) {
      continue;
    }
    has_vaddr |=
        access->dynamic_term_kinds[i] == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
    has_soffset |= access->dynamic_term_kinds[i] ==
                   LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  }
  if (!has_vaddr) {
    if (!has_soffset) {
      return;
    }
    bool has_other_vaddr = false;
    for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
      if (!(component->term_mask & (1u << i))) {
        has_other_vaddr |= access->dynamic_term_kinds[i] ==
                           LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
      }
    }
    if (!has_other_vaddr ||
        access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
      access->retained_component_kind =
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
      return;
    }
  }
  if (has_soffset &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    return;
  }
  if (component->term->byte_stride <= 0 ||
      component->term->byte_stride > UINT32_MAX ||
      !loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(
          component->term, 32)) {
    return;
  }
  // Unlike opportunistic whole-address promotion, this sum is retained across
  // accesses. A remaining scalar term need not erase that reuse benefit. Its
  // full-width contribution stays separate; the complete new VADDR must fit.
  loom_value_facts_t facts =
      loom_value_facts_exact_i64((int64_t)access->vaddr_static_byte_offset);
  loom_value_facts_addi(&facts, &component->term->byte_facts, &facts);
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    if (!(component->term_mask & (1u << i)) &&
        access->dynamic_term_kinds[i] ==
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      loom_value_facts_addi(
          &facts, &access->source->dynamic_terms[i].byte_facts, &facts);
    }
  }
  if (loom_value_facts_fit_unsigned_bit_count(facts, 32)) {
    access->retained_component_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  }
}

void loom_amdgpu_memory_access_select_vaddr_realizations(
    loom_amdgpu_memory_access_t* access) {
  access->vaddr_realization_mask = 0;
  loom_amdgpu_memory_access_select_retained_component(access);
  if (access->retained_component_kind !=
      LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE) {
    return;
  }
  const loom_low_source_memory_access_plan_t* source = access->source;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR ||
      source->dynamic_realization_count == 0) {
    return;
  }

  // Promotion is all-or-none, so each retained realization contributes its
  // correlated facts to the bound for the complete vector offset.
  loom_value_facts_t offset_facts =
      loom_value_facts_exact_i64((int64_t)access->vaddr_static_byte_offset);
  uint8_t realization_mask = 0;
  bool has_any_vaddr = false;
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    has_any_vaddr |=
        access->dynamic_term_kinds[i] == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  }
  uint16_t covered_term_mask = 0;
  for (uint8_t realization_index = 0;
       realization_index < source->dynamic_realization_count;
       ++realization_index) {
    const loom_low_source_memory_dynamic_realization_t* realization =
        &source->dynamic_realizations[realization_index];
    bool has_vaddr = false;
    bool has_soffset = false;
    for (uint8_t term_index = 0; term_index < source->dynamic_term_count;
         ++term_index) {
      if (!(realization->term_mask & (1u << term_index))) {
        continue;
      }
      if (access->dynamic_term_kinds[term_index] ==
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
        has_vaddr = true;
      } else {
        has_soffset = true;
      }
    }
    const bool can_promote =
        has_soffset && (has_vaddr || has_any_vaddr) &&
        realization->term.byte_stride > 0 &&
        realization->term.byte_stride <= UINT32_MAX &&
        loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(
            &realization->term, 32);
    if (can_promote) {
      realization_mask |= (uint8_t)(1u << realization_index);
    }
    if (can_promote || !has_soffset) {
      IREE_ASSERT_EQ(covered_term_mask & realization->term_mask, 0u);
      covered_term_mask |= realization->term_mask;
      loom_value_facts_addi(&offset_facts, &realization->term.byte_facts,
                            &offset_facts);
    }
  }
  bool has_unpromoted_scalar_term = false;
  for (uint8_t term_index = 0; term_index < source->dynamic_term_count;
       ++term_index) {
    if (covered_term_mask & (1u << term_index)) {
      continue;
    }
    if (access->dynamic_term_kinds[term_index] ==
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      loom_value_facts_addi(&offset_facts,
                            &source->dynamic_terms[term_index].byte_facts,
                            &offset_facts);
    } else {
      has_unpromoted_scalar_term = true;
    }
  }
  // A mapped expression can have died before this access. Extending its live
  // range only earns promotion when it removes the dynamic scalar address
  // calculation, not merely some terms from a still-required scalar sum.
  if (!has_unpromoted_scalar_term &&
      loom_value_facts_fit_unsigned_bit_count(offset_facts, 32)) {
    access->vaddr_realization_mask = realization_mask;
  }
}

void loom_amdgpu_memory_access_resolve_dynamic_terms(
    const loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_dynamic_term_sequence_t* out_sequence) {
  *out_sequence = (loom_amdgpu_memory_dynamic_term_sequence_t){0};
  if (access->realization.vaddr) {
    return;
  }
  const loom_low_source_memory_access_plan_t* source = access->source;
  const loom_amdgpu_memory_dynamic_index_kind_t* dynamic_term_kinds =
      access->dynamic_term_kinds;
  const uint16_t retained_mask =
      access->retained_component_kind != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE
          ? source->retained_component.term_mask
          : 0;
  if (retained_mask != 0) {
    out_sequence->terms[0] = source->retained_component.term;
    out_sequence->plans[0] =
        &access->dynamic_term_plans[source->dynamic_term_count +
                                    source->dynamic_realization_count];
    out_sequence->kinds[0] = access->retained_component_kind;
    out_sequence->count = 1;
  }
  uint8_t available_realization_mask = 0;
  for (uint8_t i = 0; i < source->dynamic_realization_count; ++i) {
    if (loom_low_lower_source_value_has_low_mapping(
            context, source->dynamic_realizations[i].term.index)) {
      available_realization_mask |= (uint8_t)(1u << i);
    }
  }
  // Partial availability would leave dynamic scalar computation in place
  // while extending vector lifetimes. Keep the original partition instead.
  const uint8_t promotion_mask =
      (available_realization_mask & access->vaddr_realization_mask) ==
              access->vaddr_realization_mask
          ? access->vaddr_realization_mask
          : 0;
  uint16_t consumed_term_mask = retained_mask;
  uint8_t term_index = 0;
  while (term_index < source->dynamic_term_count) {
    if (consumed_term_mask & (1u << term_index)) {
      ++term_index;
      continue;
    }
    const loom_low_source_memory_dynamic_realization_t* realization = NULL;
    uint8_t realization_index = 0;
    const uint16_t preceding_mask =
        term_index == 0 ? 0 : (uint16_t)((1u << term_index) - 1u);
    for (; realization_index < source->dynamic_realization_count;
         ++realization_index) {
      const loom_low_source_memory_dynamic_realization_t* candidate =
          &source->dynamic_realizations[realization_index];
      if ((candidate->term_mask & (1u << term_index)) != 0 &&
          (candidate->term_mask & preceding_mask) == 0) {
        realization = candidate;
        break;
      }
    }
    const bool promote_realization =
        realization != NULL &&
        (promotion_mask & (1u << realization_index)) != 0;
    bool use_realization =
        realization != NULL &&
        (available_realization_mask & (1u << realization_index)) != 0 &&
        !(retained_mask & realization->term_mask);
    if (use_realization && !promote_realization) {
      const loom_amdgpu_memory_dynamic_index_kind_t realization_kind =
          dynamic_term_kinds[term_index];
      for (uint8_t i = term_index + 1; i < source->dynamic_term_count; ++i) {
        if (realization->term_mask & (1u << i)) {
          use_realization &= dynamic_term_kinds[i] == realization_kind;
        }
      }
    }
    const uint8_t sequence_index = out_sequence->count++;
    if (use_realization) {
      out_sequence->terms[sequence_index] = &realization->term;
      out_sequence->plans[sequence_index] =
          &access->dynamic_term_plans[source->dynamic_term_count +
                                      realization_index];
      out_sequence->kinds[sequence_index] =
          promote_realization ? LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR
                              : dynamic_term_kinds[term_index];
      consumed_term_mask |= realization->term_mask;
    } else {
      out_sequence->terms[sequence_index] = &source->dynamic_terms[term_index];
      out_sequence->plans[sequence_index] =
          &access->dynamic_term_plans[term_index];
      out_sequence->kinds[sequence_index] = dynamic_term_kinds[term_index];
      consumed_term_mask |= (uint16_t)(1u << term_index);
    }
    ++term_index;
  }
}

static iree_status_t loom_amdgpu_lookup_memory_u32_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (loom_amdgpu_type_is_i1(loom_module_value_type(module, source_value))) {
    // EXEC-width masks describe individual lanes, not a scalar integer.
    const uint32_t register_class_id = unit_count == 2
                                           ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
                                           : LOOM_AMDGPU_REG_CLASS_ID_SGPR;
    return loom_amdgpu_lookup_or_materialize_i1_integer(
        context, source_op, source_value, register_class_id, out_low_value);
  }
  loom_value_id_t low_word = low_value;
  if (unit_count == 1) {
    return loom_amdgpu_normalize_signed_memory_index_word(
        context, source_op, source_value, low_word, out_low_value);
  }
  // The selected address form consumes the low word of this term. A product's
  // low word depends only on the low words of its operands, in either bank.
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_value,
      /*offset=*/0, loom_low_register_carrier_type_with_unit_count(low_type, 1),
      &low_word));
  return loom_amdgpu_normalize_signed_memory_index_word(
      context, source_op, source_value, low_word, out_low_value);
}

// Reuse an existing VGPR carrier. Otherwise the retained literal avoids an
// SGPR-to-VGPR copy, without querying the source's numeric facts again.
static iree_status_t loom_amdgpu_emit_memory_vgpr_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t index, const loom_amdgpu_memory_dynamic_term_plan_t* plan,
    loom_value_id_t* out_low_value) {
  const uint32_t literal = (uint32_t)plan->index_minimum;
  const void* materialization = NULL;
  if (iree_any_bit_set(plan->operand_forms,
                       LOOM_AMDGPU_MEMORY_DYNAMIC_TERM_INDEX_LITERAL) &&
      !loom_amdgpu_low_type_is_register_class(
          context, loom_low_lower_value_binding_type(context, index),
          LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    materialization = &literal;
  }
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_vgpr_address(
      context, source_op, index, materialization, &low_value));
  return loom_amdgpu_normalize_signed_memory_index_word(
      context, source_op, index, low_value, out_low_value);
}

static iree_status_t loom_amdgpu_emit_memory_u32_product(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    const loom_amdgpu_memory_dynamic_term_plan_t* plan,
    loom_value_id_t* out_low_product) {
  if (term->stride_value_count == 0) {
    return loom_amdgpu_emit_memory_vgpr_index(context, source_op, term->index,
                                              plan, out_low_product);
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_value_id_t low_product = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_memory_u32_operand(
      context, source_op, term->index, &low_product));
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_factor = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_memory_u32_operand(
        context, source_op, term->stride_values[i], &low_factor));
    loom_type_t product_type = loom_module_value_type(module, low_product);
    const loom_type_t factor_type = loom_module_value_type(module, low_factor);
    const bool product_is_scalar =
        loom_low_register_type_class_id(product_type) ==
        LOOM_AMDGPU_REG_CLASS_ID_SGPR;
    const bool factor_is_scalar =
        loom_low_register_type_class_id(factor_type) ==
        LOOM_AMDGPU_REG_CLASS_ID_SGPR;
    const loom_amdgpu_descriptor_ref_t multiply =
        product_is_scalar && factor_is_scalar
            ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32
            : LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32;
    // Vector multiplication accepts a scalar in SRC0, but SRC1 must be a VGPR.
    // Commuting the operands preserves both banks without an intermediate copy.
    loom_value_id_t low_lhs = low_product;
    loom_value_id_t low_rhs = low_factor;
    if (!product_is_scalar && factor_is_scalar) {
      low_lhs = low_factor;
      low_rhs = low_product;
    } else if (product_is_scalar && !factor_is_scalar) {
      product_type = factor_type;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(context, source_op, multiply,
                                                 low_lhs, low_rhs, product_type,
                                                 &low_product));
  }
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_product, out_low_product);
}

typedef struct loom_amdgpu_memory_vaddr_affine_group_t {
  // Low VGPR containing the sum of all indices with this coefficient.
  loom_value_id_t low_value;
  // Static byte coefficient shared by the grouped indices.
  uint32_t coefficient;
  // Combined source facts for the grouped indices.
  loom_value_facts_t facts;
} loom_amdgpu_memory_vaddr_affine_group_t;

static iree_status_t loom_amdgpu_try_emit_memory_vaddr_mad_u24(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_facts_t value_facts,
    uint32_t coefficient, loom_value_id_t low_addend, loom_type_t vgpr_type,
    loom_value_id_t* out_low_value, bool* out_selected) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  *out_selected = false;
  if (coefficient > 0xFFFFFFu ||
      !loom_value_facts_fit_unsigned_bit_count(value_facts, 24)) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MAD_U32_U24_SRC1_LIT, &descriptor,
      &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), coefficient, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {low_value, low_addend};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &vgpr_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_emit_memory_vaddr_affine_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_dynamic_term_sequence_t* sequence,
    loom_value_id_t low_base_addr, loom_type_t vgpr_type,
    loom_value_id_t* out_low_vaddr, bool* out_selected) {
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  *out_selected = false;

  uint32_t common_byte_shift = UINT32_MAX;
  uint8_t vaddr_term_count = 0;
  for (uint8_t i = 0; i < sequence->count; ++i) {
    if (sequence->kinds[i] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      continue;
    }
    const loom_low_source_memory_dynamic_term_t* term = sequence->terms[i];
    if (term->stride_value_count != 0 || term->byte_stride <= 0 ||
        term->byte_stride > UINT32_MAX) {
      return iree_ok_status();
    }
    common_byte_shift = iree_min(common_byte_shift,
                                 (uint32_t)iree_math_count_trailing_zeros_u32(
                                     (uint32_t)term->byte_stride));
    ++vaddr_term_count;
  }
  // The ordinary term emitter already selects the cheapest scale operation for
  // one term. Affine grouping only earns its extra scheduling structure when
  // it can combine multiple terms.
  if (vaddr_term_count < 2) {
    return iree_ok_status();
  }
  if (low_base_addr != LOOM_VALUE_ID_INVALID) {
    common_byte_shift = 0;
  }

  loom_amdgpu_memory_vaddr_affine_group_t
      groups[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {0};
  uint8_t group_count = 0;
  for (uint8_t i = 0; i < sequence->count; ++i) {
    if (sequence->kinds[i] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      continue;
    }
    const loom_low_source_memory_dynamic_term_t* term = sequence->terms[i];
    const uint32_t coefficient =
        (uint32_t)term->byte_stride >> common_byte_shift;
    uint8_t group_ordinal = 0;
    while (group_ordinal < group_count &&
           groups[group_ordinal].coefficient != coefficient) {
      ++group_ordinal;
    }

    loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
    const loom_amdgpu_memory_dynamic_term_plan_t* term_plan =
        sequence->plans[i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vgpr_index(
        context, source_op, term->index, term_plan, &low_index));
    const loom_value_facts_t index_facts = loom_value_facts_make(
        term_plan->index_minimum, term_plan->index_maximum, 1);
    if (group_ordinal == group_count) {
      groups[group_count++] = (loom_amdgpu_memory_vaddr_affine_group_t){
          .low_value = low_index,
          .coefficient = coefficient,
          .facts = index_facts,
      };
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
        groups[group_ordinal].low_value, low_index, vgpr_type,
        &groups[group_ordinal].low_value));
    loom_value_facts_addi(&groups[group_ordinal].facts, &index_facts,
                          &groups[group_ordinal].facts);
  }

  bool group_emitted[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {false};
  loom_value_id_t low_accumulator = low_base_addr;
  if (low_accumulator == LOOM_VALUE_ID_INVALID) {
    for (uint8_t i = 0; i < group_count; ++i) {
      if (groups[i].coefficient == 1) {
        low_accumulator = groups[i].low_value;
        group_emitted[i] = true;
        break;
      }
    }
  }
  if (low_accumulator == LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_GT(group_count, 0u);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, groups[0].low_value, groups[0].coefficient,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &low_accumulator));
    group_emitted[0] = true;
  }

  for (uint8_t i = 0; i < group_count; ++i) {
    if (group_emitted[i]) {
      continue;
    }
    loom_value_id_t low_sum = LOOM_VALUE_ID_INVALID;
    bool fused = false;
    uint32_t coefficient_shift = 0;
    if (iree_math_is_power_of_two_i64((int64_t)groups[i].coefficient)) {
      coefficient_shift =
          (uint32_t)iree_math_count_trailing_zeros_u32(groups[i].coefficient);
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_lshl_add_u32(
          context, source_op, groups[i].low_value, low_accumulator,
          coefficient_shift, vgpr_type, &low_sum, &fused));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_memory_vaddr_mad_u24(
          context, source_op, groups[i].low_value, groups[i].facts,
          groups[i].coefficient, low_accumulator, vgpr_type, &low_sum, &fused));
    }
    if (!fused) {
      loom_value_id_t low_scaled = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
          context, source_op, groups[i].low_value, groups[i].coefficient,
          LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &low_scaled));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
          low_accumulator, low_scaled, vgpr_type, &low_sum));
    }
    low_accumulator = low_sum;
  }

  if (common_byte_shift != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        common_byte_shift, low_accumulator, vgpr_type, &low_accumulator));
  }
  *out_low_vaddr = low_accumulator;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_memory_dynamic_term_sequence_t* sequence,
    loom_value_id_t low_base_addr, loom_value_id_t* out_low_vaddr) {
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  if (access->realization.vaddr) {
    *out_low_vaddr =
        loom_low_lower_realization_value(access->realization.vaddr);
    return iree_ok_status();
  }
  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  bool affine_terms_selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_memory_vaddr_affine_terms(
      context, source_op, sequence, low_base_addr, vgpr_type, &low_accumulator,
      &affine_terms_selected));
  if (!affine_terms_selected) {
    low_accumulator = low_base_addr;
    for (uint8_t i = 0; i < sequence->count; ++i) {
      switch (sequence->kinds[i]) {
        case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
          break;
        case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
          continue;
        case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
          IREE_ASSERT_UNREACHABLE("unknown AMDGPU memory dynamic index kind");
          IREE_BUILTIN_UNREACHABLE();
      }
      const loom_low_source_memory_dynamic_term_t* term = sequence->terms[i];
      loom_value_id_t low_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_u32_product(
          context, source_op, term, sequence->plans[i], &low_offset));
      if (term->byte_stride != 1) {
        // The selected address form proves the complete offset fits u32.
        // Scaling by the coefficient's low word therefore preserves signed
        // affine terms exactly modulo 2^32.
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
            context, source_op, low_offset, (uint32_t)term->byte_stride,
            LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &low_offset));
      }
      if (low_accumulator == LOOM_VALUE_ID_INVALID) {
        low_accumulator = low_offset;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
          low_accumulator, low_offset, vgpr_type, &low_accumulator));
    }
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

iree_status_t loom_amdgpu_emit_sgpr_byte_offset_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_dynamic_term_sequence_t* sequence,
    uint32_t static_byte_offset, loom_value_id_t* out_low_offset) {
  *out_low_offset = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < sequence->count; ++i) {
    switch (sequence->kinds[i]) {
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
        break;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
        continue;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
        IREE_ASSERT_UNREACHABLE("unknown AMDGPU memory dynamic index kind");
        IREE_BUILTIN_UNREACHABLE();
    }
    const loom_low_source_memory_dynamic_term_t* term = sequence->terms[i];
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_term(
        context, source_op, term, &low_term));
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      low_accumulator = low_term;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        low_accumulator, low_term, sgpr_type, &low_accumulator));
  }

  if (low_accumulator == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_sgpr_byte_offset(
        context, source_op, LOOM_VALUE_ID_INVALID,
        /*dynamic_index_byte_stride=*/1,
        LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE, static_byte_offset,
        out_low_offset);
  }
  if (static_byte_offset == 0) {
    *out_low_offset = low_accumulator;
    return iree_ok_status();
  }

  loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      static_byte_offset, sgpr_type, &low_static_offset));
  return loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, low_accumulator,
      low_static_offset, sgpr_type, out_low_offset);
}

static loom_value_facts_t loom_amdgpu_memory_saddr_offset_facts(
    const loom_amdgpu_memory_dynamic_term_sequence_t* sequence,
    uint64_t static_byte_offset) {
  if (static_byte_offset > INT64_MAX) {
    return loom_value_facts_unknown();
  }
  loom_value_facts_t offset_facts =
      loom_value_facts_exact_i64((int64_t)static_byte_offset);
  for (uint8_t i = 0; i < sequence->count; ++i) {
    if (sequence->kinds[i] == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      loom_value_facts_addi(&offset_facts, &sequence->terms[i]->byte_facts,
                            &offset_facts);
    }
  }
  return offset_facts;
}

static iree_status_t loom_amdgpu_emit_sgpr64_scale_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_unscaled_offset, int64_t byte_stride,
    uint32_t byte_shift, loom_value_id_t* out_low_offset) {
  *out_low_offset = low_unscaled_offset;
  if (byte_stride == 1) {
    return iree_ok_status();
  }
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

  if (byte_stride < 0 || byte_stride > UINT32_MAX) {
    loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
        context, source_op, (uint64_t)byte_stride, &low_scale));
    return loom_amdgpu_emit_i64_mul_lo(context, source_op, low_unscaled_offset,
                                       low_scale, out_low_offset);
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      (uint32_t)byte_stride, sgpr_type, &low_scale));
  return loom_amdgpu_emit_i64_mul_lo(context, source_op, low_unscaled_offset,
                                     low_scale, out_low_offset);
}

static iree_status_t loom_amdgpu_emit_memory_saddr_dynamic_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    const loom_amdgpu_memory_dynamic_term_plan_t* plan,
    loom_value_id_t* out_low_term) {
  *out_low_term = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_wide_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_integer_operand(
      context, source_op, term->index, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      loom_amdgpu_memory_term_operand_form(plan, 0), &low_wide_offset));
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_integer_operand(
        context, source_op, term->stride_values[i],
        LOOM_AMDGPU_REG_CLASS_ID_SGPR,
        loom_amdgpu_memory_term_operand_form(plan, i + 1), &low_stride));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_mul_lo(
        context, source_op, low_wide_offset, low_stride, &low_wide_offset));
  }
  return loom_amdgpu_emit_sgpr64_scale_byte_offset(
      context, source_op, low_wide_offset, term->byte_stride, term->byte_shift,
      out_low_term);
}

iree_status_t loom_amdgpu_emit_memory_saddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_memory_dynamic_term_sequence_t* sequence,
    loom_value_id_t low_binding, loom_value_id_t* out_low_saddr) {
  const uint64_t static_byte_offset =
      access->scalar_offset_placement ==
              LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE
          ? access->scalar_base_byte_offset
          : access->scalar_byte_offset;
  return loom_amdgpu_emit_sgpr_base_byte_offset_terms(
      context, source_op, sequence, static_byte_offset, low_binding,
      out_low_saddr);
}

iree_status_t loom_amdgpu_emit_sgpr_base_byte_offset_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_dynamic_term_sequence_t* sequence,
    uint64_t static_byte_offset, loom_value_id_t low_binding,
    loom_value_id_t* out_low_saddr) {
  *out_low_saddr = low_binding;
  const loom_value_facts_t offset_facts =
      loom_amdgpu_memory_saddr_offset_facts(sequence, static_byte_offset);
  if (loom_value_facts_is_zero(offset_facts)) {
    return iree_ok_status();
  }
  if (static_byte_offset <= UINT32_MAX &&
      loom_value_facts_fit_unsigned_bit_count(offset_facts, 32)) {
    loom_value_id_t low_u32_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_terms(
        context, source_op, sequence, (uint32_t)static_byte_offset,
        &low_u32_offset));
    loom_value_id_t low_words[2];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add_u32_offset(
        context, source_op, low_binding, low_u32_offset, low_words));
    loom_type_t pointer_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &pointer_type));
    return loom_amdgpu_build_low_register_range(context, source_op, low_words,
                                                IREE_ARRAYSIZE(low_words),
                                                pointer_type, out_low_saddr);
  }

  loom_value_id_t low_offset = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < sequence->count; ++i) {
    if (sequence->kinds[i] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      continue;
    }
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr_dynamic_term(
        context, source_op, sequence->terms[i], sequence->plans[i], &low_term));
    if (low_offset == LOOM_VALUE_ID_INVALID) {
      low_offset = low_term;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_binary_carry(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
        LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, low_offset, low_term,
        &low_offset));
  }
  if (static_byte_offset != 0) {
    loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
        context, source_op, static_byte_offset, &low_static_offset));
    if (low_offset == LOOM_VALUE_ID_INVALID) {
      low_offset = low_static_offset;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_binary_carry(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, low_offset, low_static_offset,
          &low_offset));
    }
  }
  if (low_offset == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_sgpr64_binary_carry(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, low_binding, low_offset,
      out_low_saddr);
}

// Preserves full-width operands and products. Narrow complete products and
// single 32-by-32 products use their specialized emitters below.
static iree_status_t loom_amdgpu_emit_memory_flat_wide_dynamic_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    const loom_amdgpu_memory_dynamic_term_plan_t* plan,
    loom_value_id_t low_index, loom_type_t vgpr_type,
    loom_value_id_t* out_low_lo, loom_value_id_t* out_low_hi,
    bool* out_emitted) {
  *out_emitted = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_index_type = loom_module_value_type(module, low_index);
  const bool predicate_index =
      loom_amdgpu_type_is_i1(loom_module_value_type(module, term->index));
  const bool wide_product =
      term->stride_value_count != 0 &&
      !loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(term, 32);
  if ((loom_low_register_type_unit_count(low_index_type) == 1 ||
       predicate_index) &&
      term->byte_stride >= 0 && term->byte_stride <= UINT32_MAX &&
      !wide_product) {
    return iree_ok_status();
  }

  loom_value_id_t low_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_integer_operand(
      context, source_op, term->index, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      loom_amdgpu_memory_term_operand_form(plan, 0), &low_offset));
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_integer_operand(
        context, source_op, term->stride_values[i],
        LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        loom_amdgpu_memory_term_operand_form(plan, i + 1), &low_stride));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_mul_lo(
        context, source_op, low_offset, low_stride, &low_offset));
  }
  if (term->byte_stride != 1) {
    if (term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
      loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          term->byte_shift, vgpr_type, &low_shift));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_shl(
          context, source_op, low_offset, low_shift, &low_offset));
    } else {
      loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
      if (term->byte_stride >= 0 && term->byte_stride <= UINT32_MAX) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            (uint32_t)term->byte_stride, vgpr_type, &low_stride));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
            context, source_op, (uint64_t)term->byte_stride, &low_stride));
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
            context, source_op, low_stride, &low_stride));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_mul_lo(
          context, source_op, low_offset, low_stride, &low_offset));
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_offset, /*offset=*/0, vgpr_type, out_low_lo));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_offset, /*offset=*/1, vgpr_type, out_low_hi));
  *out_emitted = true;
  return iree_ok_status();
}

// Materializes a dynamic-stride product whose complete byte value is proven
// unsigned 32-bit. The low-word products are exact under that proof and the
// high word of the resulting flat-address term is zero.
static iree_status_t loom_amdgpu_emit_memory_flat_bounded_u32_dynamic_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    const loom_amdgpu_memory_dynamic_term_plan_t* plan, loom_type_t vgpr_type,
    loom_value_id_t* out_low_lo, loom_value_id_t* out_low_hi,
    bool* out_emitted) {
  *out_emitted = false;
  if (term->stride_value_count == 0 ||
      !loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(term, 32)) {
    return iree_ok_status();
  }

  loom_value_id_t low_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_u32_product(
      context, source_op, term, plan, &low_offset));
  if (term->byte_stride != 1) {
    IREE_ASSERT(term->byte_stride > 0 && term->byte_stride <= UINT32_MAX);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, low_offset, (uint32_t)term->byte_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &low_offset));
  }

  *out_low_lo = low_offset;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
      out_low_hi));
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_flat_dynamic_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    const loom_amdgpu_memory_dynamic_term_plan_t* plan, loom_type_t vgpr_type,
    loom_value_id_t* out_low_lo, loom_value_id_t* out_low_hi) {
  *out_low_lo = LOOM_VALUE_ID_INVALID;
  *out_low_hi = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_index = loom_low_lower_lookup_value(context, term->index);
  bool emitted_wide = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_wide_dynamic_term(
      context, source_op, term, plan, low_index, vgpr_type, out_low_lo,
      out_low_hi, &emitted_wide));
  if (emitted_wide) {
    return iree_ok_status();
  }

  bool emitted_bounded_u32 = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_bounded_u32_dynamic_term(
      context, source_op, term, plan, vgpr_type, out_low_lo, out_low_hi,
      &emitted_bounded_u32));
  if (emitted_bounded_u32) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vgpr_index(
      context, source_op, term->index, plan, &low_index));
  const bool signed_index = plan->index_minimum < 0;
  if (term->byte_stride == 1) {
    *out_low_lo = low_index;
    if (signed_index) {
      return loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
          low_index, 31, vgpr_type, out_low_hi);
    }
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, out_low_hi);
  }

  if (term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        low_index, term->byte_shift, vgpr_type, out_low_lo));
    return loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op,
        signed_index ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT
                     : LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        low_index, 32u - term->byte_shift, vgpr_type, out_low_hi);
  }

  IREE_ASSERT(term->byte_stride > 0 && term->byte_stride <= UINT32_MAX);
  loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      (uint32_t)term->byte_stride, vgpr_type, &low_stride));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32, low_index,
      low_stride, vgpr_type, out_low_lo));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
      context, source_op,
      signed_index ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_I32
                   : LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32,
      low_index, low_stride, vgpr_type, out_low_hi));
  if (signed_index && term->byte_stride > INT32_MAX) {
    // The positive coefficient is represented as a negative signed i32.
    // Adding index * 2^32 restores its unsigned coefficient interpretation.
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, *out_low_hi,
        low_index, vgpr_type, out_low_hi));
  }
  return iree_ok_status();
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
    const loom_amdgpu_memory_dynamic_term_plan_t* plan,
    loom_value_id_t* out_low_term, bool* out_emitted) {
  *out_low_term = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;

  loom_value_id_t low_index = loom_low_lower_lookup_value(context, term->index);
  const bool index_is_sgpr_b64 = loom_amdgpu_low_value_is_register_class_count(
      context, low_index, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  const bool index_is_sgpr_b32 = loom_amdgpu_low_value_is_register_class_count(
      context, low_index, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  if (!index_is_sgpr_b32 && !index_is_sgpr_b64) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (index_is_sgpr_b64 &&
      loom_amdgpu_type_is_i1(loom_module_value_type(module, term->index))) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_stride =
        loom_low_lower_lookup_value(context, term->stride_values[i]);
    if (!loom_amdgpu_low_type_is_register_class(
            context, loom_module_value_type(module, low_stride),
            LOOM_AMDGPU_REG_CLASS_ID_SGPR)) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr_dynamic_term(
      context, source_op, term, plan, out_low_term));
  *out_emitted = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_memory_flat_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_memory_dynamic_term_sequence_t* sequence,
    loom_value_id_t low_resource, loom_value_id_t* out_low_vaddr) {
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

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
  loom_value_id_t low_scalar_base = low_resource;
  if (loom_amdgpu_low_value_is_register_class_count(
          context, low_resource, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                    low_resource, /*offset=*/0,
                                                    vgpr_type, &low_vaddr_lo));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                    low_resource, /*offset=*/1,
                                                    vgpr_type, &low_vaddr_hi));
  }
  if (access->vaddr_static_byte_offset != 0) {
    loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_constant_u64(
        context, source_op, access->vaddr_static_byte_offset,
        &low_static_offset));
    if (low_vaddr_lo == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_binary_carry(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, low_scalar_base,
          low_static_offset, &low_scalar_base));
    } else {
      loom_value_id_t low_offset_lo = LOOM_VALUE_ID_INVALID;
      loom_value_id_t low_offset_hi = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_low_slice(context, source_op, low_static_offset,
                                     /*offset=*/0, sgpr_type, &low_offset_lo));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_low_slice(context, source_op, low_static_offset,
                                     /*offset=*/1, sgpr_type, &low_offset_hi));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_add_term(
          context, source_op, low_offset_lo, low_offset_hi, vgpr_type,
          sgpr_x2_type, &low_vaddr_lo, &low_vaddr_hi));
    }
  }
  for (uint8_t i = 0; i < sequence->count; ++i) {
    if (sequence->kinds[i] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      continue;
    }
    const loom_low_source_memory_dynamic_term_t* term = sequence->terms[i];
    loom_value_id_t low_scalar_term = LOOM_VALUE_ID_INVALID;
    bool scalar_term_emitted = false;
    if (low_vaddr_lo == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_scalar_dynamic_term(
          context, source_op, term, sequence->plans[i], &low_scalar_term,
          &scalar_term_emitted));
    }
    if (scalar_term_emitted) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_binary_carry(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32, low_scalar_base,
          low_scalar_term, &low_scalar_base));
      continue;
    }

    loom_value_id_t low_term_lo = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_term_hi = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_dynamic_term(
        context, source_op, term, sequence->plans[i], vgpr_type, &low_term_lo,
        &low_term_hi));
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
