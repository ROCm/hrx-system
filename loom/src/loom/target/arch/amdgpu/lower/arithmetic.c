// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/arithmetic.h"

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic_lower_rules.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/candidates/arithmetic_candidates.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

iree_status_t loom_amdgpu_select_arithmetic_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  loom_low_lower_rule_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select_contract(
      context, &loom_amdgpu_arithmetic_lower_rule_set, source_op, &selection));
  *out_selected = selection.rule != NULL;
  return iree_ok_status();
}

static bool loom_amdgpu_type_is_f16(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F16;
}

static bool loom_amdgpu_type_is_even_packed_f16_vector(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_F16 ||
      !loom_amdgpu_type_packed_16bit_float_storage(type, &payload_bit_count,
                                                   &register_count)) {
    return false;
  }
  if (payload_bit_count == 0 || (payload_bit_count % 32u) != 0 ||
      register_count == 0 ||
      register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_register_count = register_count;
  return true;
}

static bool loom_amdgpu_type_is_even_packed_i16_vector(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_I16 ||
      !loom_amdgpu_type_packed_integer_storage(type, &payload_bit_count,
                                               &register_count)) {
    return false;
  }
  if (payload_bit_count == 0 || (payload_bit_count % 32u) != 0 ||
      register_count == 0 ||
      register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_register_count = register_count;
  return true;
}

static bool loom_amdgpu_type_is_even_packed_f32_vector(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  const uint32_t register_count = loom_amdgpu_vector_f32_register_count(type);
  if (register_count == 0 || (register_count % 2u) != 0 ||
      register_count / 2u > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_register_count = register_count;
  return true;
}

static bool loom_amdgpu_fma_mix_source_is_f16(
    loom_amdgpu_fma_mix_source_kind_t source_kind) {
  return source_kind == LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO ||
         source_kind == LOOM_AMDGPU_FMA_MIX_SOURCE_F16HI;
}

static bool loom_amdgpu_fma_mix_source_ref_is_valid(
    loom_amdgpu_fma_mix_source_kind_t source_kind) {
  return (uint32_t)source_kind < LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT_;
}

static const loom_op_t* loom_amdgpu_source_defining_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

static bool loom_amdgpu_select_fma_mix_packed_extract_source(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t* out_source,
    loom_amdgpu_fma_mix_source_kind_t* out_source_kind,
    uint32_t* out_source_register_offset) {
  const loom_op_t* extract_op =
      loom_amdgpu_source_defining_op(module, value_id);
  if (extract_op == NULL || !loom_vector_extract_isa(extract_op) ||
      loom_vector_extract_result(extract_op) != value_id) {
    return false;
  }
  const loom_attribute_t static_indices =
      loom_vector_extract_static_indices(extract_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      loom_vector_extract_indices(extract_op).count != 0) {
    return false;
  }
  const int64_t lane_index = static_indices.i64_array[0];
  if (lane_index < 0 || lane_index == INT64_MIN || lane_index > UINT32_MAX) {
    return false;
  }

  const loom_value_id_t source = loom_vector_extract_source(extract_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  if (!loom_type_is_vector(source_type) ||
      loom_type_element_type(source_type) != LOOM_SCALAR_TYPE_F16) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          source_type, &payload_bit_count, &register_count)) {
    return false;
  }
  const uint32_t lane_count = payload_bit_count / 16u;
  const uint32_t source_register_offset = (uint32_t)lane_index / 2u;
  if ((uint32_t)lane_index >= lane_count ||
      source_register_offset >= register_count) {
    return false;
  }

  *out_source = source;
  *out_source_kind = ((uint32_t)lane_index & 1u)
                         ? LOOM_AMDGPU_FMA_MIX_SOURCE_F16HI
                         : LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO;
  *out_source_register_offset = source_register_offset;
  return true;
}

bool loom_amdgpu_select_fma_mix_source(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t* out_source,
    loom_amdgpu_fma_mix_source_kind_t* out_source_kind,
    uint32_t* out_source_register_offset) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_source_kind = LOOM_AMDGPU_FMA_MIX_SOURCE_F32;
  *out_source_register_offset = 0;

  const loom_op_t* defining_op =
      loom_amdgpu_source_defining_op(module, value_id);
  if (defining_op != NULL && defining_op->operand_count == 1 &&
      defining_op->result_count == 1 &&
      loom_op_defines_value(defining_op, value_id) &&
      loom_op_operand_has_role(module, defining_op, 0,
                               LOOM_OPERAND_ROLE_FLOAT_EXTENSION_SOURCE)) {
    const loom_value_id_t input = loom_op_const_operands(defining_op)[0];
    const loom_type_t input_type = loom_module_value_type(module, input);
    const loom_type_t result_type = loom_module_value_type(module, value_id);
    if (loom_amdgpu_type_is_f16(input_type) &&
        loom_amdgpu_type_is_f32(result_type)) {
      if (loom_amdgpu_select_fma_mix_packed_extract_source(
              module, input, out_source, out_source_kind,
              out_source_register_offset)) {
        return true;
      }
      *out_source = input;
      *out_source_kind = LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO;
      return true;
    }
    return false;
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  if (loom_amdgpu_type_is_f32(value_type)) {
    *out_source = value_id;
    *out_source_kind = LOOM_AMDGPU_FMA_MIX_SOURCE_F32;
    *out_source_register_offset = 0;
    return true;
  }

  return false;
}

bool loom_amdgpu_scalar_mulf_fastmath_allows_zero_add(
    const loom_op_t* source_op) {
  const uint8_t required_flags = LOOM_SCALAR_FASTMATHFLAGS_NNAN |
                                 LOOM_SCALAR_FASTMATHFLAGS_NSZ |
                                 LOOM_SCALAR_FASTMATHFLAGS_CONTRACT;
  return iree_all_bits_set(loom_scalar_mulf_fastmath(source_op),
                           required_flags);
}

static bool loom_amdgpu_vector_mulf_fastmath_allows_zero_add(
    const loom_op_t* source_op) {
  const uint8_t required_flags = LOOM_VECTOR_FASTMATHFLAGS_NNAN |
                                 LOOM_VECTOR_FASTMATHFLAGS_NSZ |
                                 LOOM_VECTOR_FASTMATHFLAGS_CONTRACT;
  return iree_all_bits_set(loom_vector_mulf_fastmath(source_op),
                           required_flags);
}

static bool loom_amdgpu_descriptor_ref_is_present(
    loom_low_lower_context_t* context, loom_amdgpu_descriptor_ref_t ref) {
  if (ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, ref);
}

static loom_amdgpu_descriptor_ref_t loom_amdgpu_fma_mix_descriptor_ref_at(
    const loom_amdgpu_fma_mix_descriptor_ref_cube_t* refs,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds) {
  return (*refs)[source_kinds[0]][source_kinds[1]][source_kinds[2]];
}

static bool loom_amdgpu_select_fma_mix_descriptor_from_cubes(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds,
    const loom_amdgpu_fma_mix_descriptor_ref_cube_t* preferred_refs,
    const loom_amdgpu_fma_mix_descriptor_ref_cube_t* fallback_refs,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;

  for (uint32_t i = 0; i < LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT; ++i) {
    IREE_ASSERT(loom_amdgpu_fma_mix_source_ref_is_valid(source_kinds[i]));
  }

  const loom_amdgpu_descriptor_ref_t preferred_ref =
      loom_amdgpu_fma_mix_descriptor_ref_at(preferred_refs, source_kinds);
  if (loom_amdgpu_descriptor_ref_is_present(context, preferred_ref)) {
    *out_descriptor_ref = preferred_ref;
    return true;
  }

  const loom_amdgpu_descriptor_ref_t fallback_ref =
      loom_amdgpu_fma_mix_descriptor_ref_at(fallback_refs, source_kinds);
  if (loom_amdgpu_descriptor_ref_is_present(context, fallback_ref)) {
    *out_descriptor_ref = fallback_ref;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_fma_mix_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  return loom_amdgpu_select_fma_mix_descriptor_from_cubes(
      context, source_kinds, &kLoomAmdgpuFmaMixF32DescriptorRefs,
      &kLoomAmdgpuMadMixF32DescriptorRefs, out_descriptor_ref);
}

bool loom_amdgpu_select_fma_mix_half_result_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds, bool high_result,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  return loom_amdgpu_select_fma_mix_descriptor_from_cubes(
      context, source_kinds,
      high_result ? &kLoomAmdgpuFmaMixhiF16DescriptorRefs
                  : &kLoomAmdgpuFmaMixloF16DescriptorRefs,
      high_result ? &kLoomAmdgpuMadMixhiF16DescriptorRefs
                  : &kLoomAmdgpuMadMixloF16DescriptorRefs,
      out_descriptor_ref);
}

bool loom_amdgpu_select_fma_mix_half_result_zero_addend_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds, bool high_result,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref,
    loom_amdgpu_fma_mix_plan_flags_t* out_flags) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_flags = 0;

  for (uint32_t i = 0; i < LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT; ++i) {
    IREE_ASSERT(loom_amdgpu_fma_mix_source_ref_is_valid(source_kinds[i]));
  }

  if (source_kinds[2] == LOOM_AMDGPU_FMA_MIX_SOURCE_F32) {
    const loom_amdgpu_fma_mix_src2_literal_descriptor_ref_table_t*
        literal_refs =
            high_result ? &kLoomAmdgpuFmaMixhiF16Src2LiteralDescriptorRefs
                        : &kLoomAmdgpuFmaMixloF16Src2LiteralDescriptorRefs;
    const loom_amdgpu_descriptor_ref_t literal_ref =
        (*literal_refs)[source_kinds[0]][source_kinds[1]];
    if (loom_amdgpu_descriptor_ref_is_present(context, literal_ref)) {
      *out_descriptor_ref = literal_ref;
      *out_flags = LOOM_AMDGPU_FMA_MIX_PLAN_SRC2_LITERAL_ZERO;
      return true;
    }
  }

  if (!loom_amdgpu_select_fma_mix_half_result_descriptor(
          context, source_kinds, high_result, out_descriptor_ref)) {
    return false;
  }
  *out_flags = LOOM_AMDGPU_FMA_MIX_PLAN_SRC2_MATERIALIZED_ZERO;
  return true;
}

static bool loom_amdgpu_select_fma_mix_zero_addend_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref,
    bool* out_addend_literal_zero) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_addend_literal_zero = false;

  for (uint32_t i = 0; i < LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT; ++i) {
    IREE_ASSERT(loom_amdgpu_fma_mix_source_ref_is_valid(source_kinds[i]));
  }

  if (source_kinds[2] == LOOM_AMDGPU_FMA_MIX_SOURCE_F32) {
    const loom_amdgpu_descriptor_ref_t literal_ref =
        kLoomAmdgpuFmaMixF32Src2LiteralDescriptorRefs[source_kinds[0]]
                                                     [source_kinds[1]];
    if (loom_amdgpu_descriptor_ref_is_present(context, literal_ref)) {
      *out_descriptor_ref = literal_ref;
      *out_addend_literal_zero = true;
      return true;
    }
  }

  return loom_amdgpu_select_fma_mix_descriptor(context, source_kinds,
                                               out_descriptor_ref);
}

static void loom_amdgpu_reset_mulf_mix_plan(
    loom_amdgpu_mulf_mix_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_mulf_mix_plan_t){
      .sources = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID},
      .source_register_offsets = {0, 0},
      .result = LOOM_VALUE_ID_INVALID,
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .addend_literal_zero = false,
      .source_kinds = {LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
                       LOOM_AMDGPU_FMA_MIX_SOURCE_F32},
      .lane_count = 0,
  };
}

void loom_amdgpu_canonicalize_mulf_mix_sources(
    loom_value_id_t* sources, uint32_t* source_register_offsets,
    loom_amdgpu_fma_mix_source_kind_t* source_kinds) {
  if (source_kinds[0] != LOOM_AMDGPU_FMA_MIX_SOURCE_F32 ||
      !loom_amdgpu_fma_mix_source_is_f16(source_kinds[1])) {
    return;
  }
  const loom_value_id_t source = sources[0];
  sources[0] = sources[1];
  sources[1] = source;
  const uint32_t source_register_offset = source_register_offsets[0];
  source_register_offsets[0] = source_register_offsets[1];
  source_register_offsets[1] = source_register_offset;
  const loom_amdgpu_fma_mix_source_kind_t source_kind = source_kinds[0];
  source_kinds[0] = source_kinds[1];
  source_kinds[1] = source_kind;
}

static bool loom_amdgpu_source_value_has_exact_f32_immediate(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id) {
  if (fact_table == NULL) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_vector(type)) {
    if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_F32) {
      return false;
    }
    loom_value_fact_uniform_element_t uniform = {0};
    if (!loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                                &uniform)) {
      return false;
    }
    facts = uniform.element;
  } else if (loom_type_is_scalar(type)) {
    if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_F32) {
      return false;
    }
  } else {
    return false;
  }
  return loom_value_facts_is_exact(facts) && loom_value_facts_is_float(facts);
}

typedef struct loom_amdgpu_fmaf_literal_operand_form_t {
  // Selected f32 FMA literal operand form.
  iree_string_view_t operand_form;
  // Source operand index carrying the exact f32 literal.
  uint32_t source_operand_index;
  // Diagnostic role for the literal operand.
  iree_string_view_t literal_role;
  // Descriptor ref implementing the literal operand form.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_fmaf_literal_operand_form_t;

static const loom_amdgpu_fmaf_literal_operand_form_t
    kAmdgpuFmafLiteralOperandForms[] = {
        {
            .operand_form = IREE_SVL("fmaak"),
            .source_operand_index = 2,
            .literal_role = IREE_SVL("addend"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAAK_F32,
        },
        {
            .operand_form = IREE_SVL("fmamk"),
            .source_operand_index = 0,
            .literal_role = IREE_SVL("multiply_lhs"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAMK_F32,
        },
        {
            .operand_form = IREE_SVL("fmamk"),
            .source_operand_index = 1,
            .literal_role = IREE_SVL("multiply_rhs"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAMK_F32,
        },
};

static bool loom_amdgpu_fmaf_operand_layout(
    const loom_op_t* source_op, const loom_value_id_t** out_sources) {
  *out_sources = NULL;
  if ((source_op->kind != LOOM_OP_SCALAR_FMAF &&
       source_op->kind != LOOM_OP_VECTOR_FMAF) ||
      source_op->operand_count != LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT ||
      source_op->result_count != 1) {
    return false;
  }
  *out_sources = loom_op_const_operands(source_op);
  return true;
}

static bool loom_amdgpu_fmaf_literal_operand_form(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fmaf_literal_operand_form_t* out_form) {
  *out_form = (loom_amdgpu_fmaf_literal_operand_form_t){0};

  const loom_value_id_t* sources = NULL;
  if (!loom_amdgpu_fmaf_operand_layout(source_op, &sources)) {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(kAmdgpuFmafLiteralOperandForms);
       ++i) {
    const loom_amdgpu_fmaf_literal_operand_form_t* form =
        &kAmdgpuFmafLiteralOperandForms[i];
    IREE_ASSERT_LT(form->source_operand_index,
                   LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT);
    if (loom_amdgpu_source_value_has_exact_f32_immediate(
            module, fact_table, sources[form->source_operand_index])) {
      *out_form = *form;
      return true;
    }
  }
  return false;
}

iree_status_t loom_amdgpu_emit_fmaf_literal_operand_form_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!iree_any_bit_set(loom_low_lower_context_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM)) {
    return iree_ok_status();
  }

  loom_amdgpu_fmaf_literal_operand_form_t literal_form = {0};
  if (!loom_amdgpu_fmaf_literal_operand_form(context, source_op,
                                             &literal_form)) {
    return iree_ok_status();
  }

  loom_low_lower_rule_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select(
      context, &loom_amdgpu_arithmetic_lower_rule_set, source_op, &selection));
  if (selection.rule == NULL) {
    return iree_ok_status();
  }

  const loom_low_lower_descriptor_ref_t descriptor_ref =
      loom_low_lower_rule_first_descriptor_ref(
          &loom_amdgpu_arithmetic_lower_rule_set, selection.rule);
  if (descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(descriptor_ref,
                 loom_amdgpu_arithmetic_lower_rule_set.descriptor_ref_count);
  const iree_string_view_t descriptor_name =
      loom_amdgpu_arithmetic_lower_rule_set.descriptor_refs[descriptor_ref].key;
  const bool selected_literal =
      iree_string_view_equal(descriptor_name, IREE_SV("amdgpu.v_fmaak_f32")) ||
      iree_string_view_equal(descriptor_name, IREE_SV("amdgpu.v_fmamk_f32"));
  const bool selected_plain_fma =
      iree_string_view_equal(descriptor_name, IREE_SV("amdgpu.v_fma_f32"));
  if (!selected_literal && !selected_plain_fma) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const iree_string_view_t descriptor_set_name =
      loom_amdgpu_descriptor_set_key(descriptor_set);
  const iree_string_view_t decision_key =
      selected_literal ? IREE_SV("selected") : IREE_SV("rejected");
  const iree_string_view_t reason_key =
      selected_literal ? IREE_SV("literal_descriptor_selected")
                       : (loom_amdgpu_descriptor_set_has_ref(
                              descriptor_set, literal_form.descriptor_ref)
                              ? IREE_SV("literal_operand_contract_unmatched")
                              : IREE_SV("literal_descriptor_unavailable"));

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(descriptor_name),
      loom_param_string(literal_form.operand_form),
      loom_param_u32(literal_form.source_operand_index),
      loom_param_string(literal_form.literal_role),
      loom_param_string(descriptor_set_name),
      loom_param_string(IREE_SV("exact_f32_literal")),
      loom_param_string(decision_key),
      loom_param_string(reason_key),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_030_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_emit_mulf_mix_operand_form_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan) {
  if (!iree_any_bit_set(loom_low_lower_context_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM)) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, plan->descriptor_ref, &descriptor));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const iree_string_view_t descriptor_name = loom_low_descriptor_set_string(
      descriptor_set, descriptor.descriptor->key_string_offset);
  const iree_string_view_t descriptor_set_name =
      loom_amdgpu_descriptor_set_key(descriptor_set);
  const iree_string_view_t decision_key =
      plan->addend_literal_zero ? IREE_SV("selected") : IREE_SV("rejected");
  const iree_string_view_t reason_key =
      plan->addend_literal_zero ? IREE_SV("literal_descriptor_available")
                                : IREE_SV("literal_descriptor_unavailable");
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(descriptor_name),
      loom_param_u32(2),
      loom_param_string(descriptor_set_name),
      loom_param_string(IREE_SV("exact_f32_positive_zero")),
      loom_param_string(decision_key),
      loom_param_string(reason_key),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_027_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static void loom_amdgpu_reset_packed_ternary_plan(
    loom_amdgpu_packed_ternary_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_packed_ternary_plan_t){
      .sources = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                  LOOM_VALUE_ID_INVALID},
      .result = LOOM_VALUE_ID_INVALID,
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .flags = 0,
      .register_count = 0,
      .packet_unit_count = 0,
      .packet_count = 0,
  };
}

typedef enum loom_amdgpu_packed_ternary_type_kind_e {
  LOOM_AMDGPU_PACKED_TERNARY_TYPE_F16 = 0,
  LOOM_AMDGPU_PACKED_TERNARY_TYPE_F32 = 1,
  LOOM_AMDGPU_PACKED_TERNARY_TYPE_I16 = 2,
} loom_amdgpu_packed_ternary_type_kind_t;

typedef struct loom_amdgpu_packed_ternary_selection_row_t {
  // Source op kind using the packed ternary descriptor candidates.
  loom_op_kind_t op_kind;
  // Result/source type family accepted by this row.
  loom_amdgpu_packed_ternary_type_kind_t type_kind;
  // Required vector integer-overflow flags for this row.
  uint8_t required_overflow_flags;
  // Overflow flags that make this row inapplicable.
  uint8_t rejected_overflow_flags;
  // Descriptor candidates in preference order.
  const loom_amdgpu_packed_ternary_descriptor_candidate_t* candidates;
  // Number of descriptor candidates.
  uint32_t candidate_count;
} loom_amdgpu_packed_ternary_selection_row_t;

static const loom_amdgpu_packed_ternary_selection_row_t
    kAmdgpuPackedTernarySelectionRows[] = {
        {
            .op_kind = LOOM_OP_VECTOR_FMAF,
            .type_kind = LOOM_AMDGPU_PACKED_TERNARY_TYPE_F16,
            .candidates = kLoomAmdgpuPackedFmafF16DescriptorCandidates,
            .candidate_count = kLoomAmdgpuPackedFmafF16DescriptorCandidateCount,
        },
        {
            .op_kind = LOOM_OP_VECTOR_FMAF,
            .type_kind = LOOM_AMDGPU_PACKED_TERNARY_TYPE_F32,
            .candidates = kLoomAmdgpuPackedFmafF32DescriptorCandidates,
            .candidate_count = kLoomAmdgpuPackedFmafF32DescriptorCandidateCount,
        },
        {
            .op_kind = LOOM_OP_VECTOR_FMAI,
            .type_kind = LOOM_AMDGPU_PACKED_TERNARY_TYPE_I16,
            .required_overflow_flags = LOOM_VECTOR_INTOVERFLOWFLAGS_NUW,
            .candidates =
                kLoomAmdgpuPackedFmaiUnsignedPreferenceDescriptorCandidates,
            .candidate_count =
                kLoomAmdgpuPackedFmaiUnsignedPreferenceDescriptorCandidateCount,
        },
        {
            .op_kind = LOOM_OP_VECTOR_FMAI,
            .type_kind = LOOM_AMDGPU_PACKED_TERNARY_TYPE_I16,
            .rejected_overflow_flags = LOOM_VECTOR_INTOVERFLOWFLAGS_NUW,
            .candidates =
                kLoomAmdgpuPackedFmaiSignedPreferenceDescriptorCandidates,
            .candidate_count =
                kLoomAmdgpuPackedFmaiSignedPreferenceDescriptorCandidateCount,
        },
};

enum {
  LOOM_AMDGPU_PACKED_TERNARY_FMAF_ROW_OFFSET = 0u,
  LOOM_AMDGPU_PACKED_TERNARY_FMAF_ROW_COUNT = 2u,
  LOOM_AMDGPU_PACKED_TERNARY_FMAI_ROW_OFFSET =
      LOOM_AMDGPU_PACKED_TERNARY_FMAF_ROW_OFFSET +
      LOOM_AMDGPU_PACKED_TERNARY_FMAF_ROW_COUNT,
  LOOM_AMDGPU_PACKED_TERNARY_FMAI_ROW_COUNT = 2u,
};

static_assert(IREE_ARRAYSIZE(kAmdgpuPackedTernarySelectionRows) ==
                  LOOM_AMDGPU_PACKED_TERNARY_FMAI_ROW_OFFSET +
                      LOOM_AMDGPU_PACKED_TERNARY_FMAI_ROW_COUNT,
              "AMDGPU packed ternary row spans must cover all rows");

static bool loom_amdgpu_packed_ternary_type_register_count(
    loom_amdgpu_packed_ternary_type_kind_t type_kind, loom_type_t type,
    uint32_t* out_register_count) {
  switch (type_kind) {
    case LOOM_AMDGPU_PACKED_TERNARY_TYPE_F16:
      return loom_amdgpu_type_is_even_packed_f16_vector(type,
                                                        out_register_count);
    case LOOM_AMDGPU_PACKED_TERNARY_TYPE_F32:
      return loom_amdgpu_type_is_even_packed_f32_vector(type,
                                                        out_register_count);
    case LOOM_AMDGPU_PACKED_TERNARY_TYPE_I16:
      return loom_amdgpu_type_is_even_packed_i16_vector(type,
                                                        out_register_count);
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU packed ternary type kind");
  IREE_BUILTIN_UNREACHABLE();
}

static uint8_t loom_amdgpu_packed_ternary_overflow_flags(
    const loom_op_t* source_op) {
  return source_op->kind == LOOM_OP_VECTOR_FMAI
             ? loom_vector_fmai_overflow(source_op)
             : 0;
}

static bool loom_amdgpu_packed_ternary_row_matches(
    const loom_amdgpu_packed_ternary_selection_row_t* row,
    const loom_op_t* source_op, loom_type_t result_type,
    uint32_t* out_register_count) {
  if (source_op->kind != row->op_kind) {
    return false;
  }
  const uint8_t overflow_flags =
      loom_amdgpu_packed_ternary_overflow_flags(source_op);
  if (!iree_all_bits_set(overflow_flags, row->required_overflow_flags) ||
      iree_any_bit_set(overflow_flags, row->rejected_overflow_flags)) {
    return false;
  }
  return loom_amdgpu_packed_ternary_type_register_count(
      row->type_kind, result_type, out_register_count);
}

static bool loom_amdgpu_select_packed_ternary_candidate_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_packed_ternary_descriptor_candidate_t* candidates,
    uint32_t candidate_count, const loom_value_id_t* sources,
    loom_value_id_t result, uint32_t register_count,
    loom_amdgpu_packed_ternary_plan_t* out_plan) {
  const loom_amdgpu_packed_ternary_descriptor_candidate_t* candidate = NULL;
  for (uint32_t i = 0; i < candidate_count; ++i) {
    if (loom_amdgpu_descriptor_ref_is_present(context,
                                              candidates[i].descriptor_ref)) {
      candidate = &candidates[i];
      break;
    }
  }
  if (candidate == NULL) {
    return false;
  }

  IREE_ASSERT(candidate->packet_unit_count != 0);
  if ((register_count % candidate->packet_unit_count) != 0) {
    return false;
  }

  loom_value_id_t descriptor_sources[LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT] =
      {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(descriptor_sources); ++i) {
    IREE_ASSERT_LT(candidate->source_permutation[i],
                   LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT);
    descriptor_sources[i] = sources[candidate->source_permutation[i]];
  }
  *out_plan = (loom_amdgpu_packed_ternary_plan_t){
      .sources = {descriptor_sources[0], descriptor_sources[1],
                  descriptor_sources[2]},
      .result = result,
      .descriptor_ref = candidate->descriptor_ref,
      .flags = candidate->flags,
      .register_count = register_count,
      .packet_unit_count = candidate->packet_unit_count,
      .packet_count = register_count / candidate->packet_unit_count,
  };
  return true;
}

static bool loom_amdgpu_packed_ternary_op_layout(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_value_id_t** out_sources, loom_value_id_t* out_result,
    loom_type_t* out_result_type) {
  *out_sources = NULL;
  *out_result = LOOM_VALUE_ID_INVALID;
  *out_result_type = loom_type_none();
  if (source_op->operand_count != LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT ||
      source_op->result_count != 1) {
    return false;
  }
  const loom_value_id_t* sources = loom_op_const_operands(source_op);
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_type_t result_type = loom_module_value_type(module, result);
  for (uint32_t i = 0; i < LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT; ++i) {
    const loom_type_t source_type = loom_module_value_type(module, sources[i]);
    if (!loom_type_equal(source_type, result_type)) {
      return false;
    }
  }
  *out_sources = sources;
  *out_result = result;
  *out_result_type = result_type;
  return true;
}

static bool loom_amdgpu_select_vector_packed_ternary_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_packed_ternary_selection_row_t* rows, uint32_t row_count,
    loom_amdgpu_packed_ternary_plan_t* out_plan) {
  loom_amdgpu_reset_packed_ternary_plan(out_plan);
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t* sources = NULL;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  loom_type_t result_type = loom_type_none();
  if (!loom_amdgpu_packed_ternary_op_layout(module, source_op, &sources,
                                            &result, &result_type)) {
    return false;
  }

  for (uint32_t i = 0; i < row_count; ++i) {
    const loom_amdgpu_packed_ternary_selection_row_t* row = &rows[i];
    uint32_t register_count = 0;
    if (!loom_amdgpu_packed_ternary_row_matches(row, source_op, result_type,
                                                &register_count)) {
      continue;
    }
    if (loom_amdgpu_select_packed_ternary_candidate_plan(
            context, row->candidates, row->candidate_count, sources, result,
            register_count, out_plan)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_amdgpu_select_vector_packed_fmaf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_packed_ternary_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_select_vector_packed_ternary_plan(
      context, source_op,
      &kAmdgpuPackedTernarySelectionRows
          [LOOM_AMDGPU_PACKED_TERNARY_FMAF_ROW_OFFSET],
      LOOM_AMDGPU_PACKED_TERNARY_FMAF_ROW_COUNT, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_packed_fmai_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_packed_ternary_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_select_vector_packed_ternary_plan(
      context, source_op,
      &kAmdgpuPackedTernarySelectionRows
          [LOOM_AMDGPU_PACKED_TERNARY_FMAI_ROW_OFFSET],
      LOOM_AMDGPU_PACKED_TERNARY_FMAI_ROW_COUNT, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_fmaf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fma_mix_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_fma_mix_plan_t){
      .sources = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                  LOOM_VALUE_ID_INVALID},
      .source_register_offsets = {0, 0, 0},
      .result = LOOM_VALUE_ID_INVALID,
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .source_kinds = {LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
                       LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
                       LOOM_AMDGPU_FMA_MIX_SOURCE_F32},
  };
  *out_selected = false;
  if (!loom_scalar_fmaf_isa(source_op)) {
    return iree_ok_status();
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t operands[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      loom_scalar_fmaf_a(source_op),
      loom_scalar_fmaf_b(source_op),
      loom_scalar_fmaf_c(source_op),
  };
  const loom_value_id_t result = loom_scalar_fmaf_result(source_op);
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }

  loom_value_id_t sources[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t source_register_offsets[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {0, 0,
                                                                        0};
  loom_amdgpu_fma_mix_source_kind_t
      source_kinds[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      };
  bool has_f16_source = false;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
    if (!loom_amdgpu_select_fma_mix_source(module, operands[i], &sources[i],
                                           &source_kinds[i],
                                           &source_register_offsets[i])) {
      return iree_ok_status();
    }
    has_f16_source =
        has_f16_source || loom_amdgpu_fma_mix_source_is_f16(source_kinds[i]);
  }
  if (!has_f16_source) {
    return iree_ok_status();
  }

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_select_fma_mix_descriptor(context, source_kinds,
                                             &descriptor_ref)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_fma_mix_plan_t){
      .sources = {sources[0], sources[1], sources[2]},
      .source_register_offsets = {source_register_offsets[0],
                                  source_register_offsets[1],
                                  source_register_offsets[2]},
      .result = result,
      .descriptor_ref = descriptor_ref,
      .source_kinds = {source_kinds[0], source_kinds[1], source_kinds[2]},
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_mulf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_mulf_mix_plan_t* out_plan, bool* out_selected) {
  loom_amdgpu_reset_mulf_mix_plan(out_plan);
  *out_selected = false;
  if (!loom_scalar_mulf_isa(source_op) ||
      !loom_amdgpu_scalar_mulf_fastmath_allows_zero_add(source_op)) {
    return iree_ok_status();
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t operands[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT] = {
      loom_scalar_mulf_lhs(source_op),
      loom_scalar_mulf_rhs(source_op),
  };
  const loom_value_id_t result = loom_scalar_mulf_result(source_op);
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }

  loom_value_id_t sources[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  uint32_t source_register_offsets[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT] = {0, 0};
  loom_amdgpu_fma_mix_source_kind_t
      source_kinds[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      };
  bool has_f16_source = false;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
    if (!loom_amdgpu_select_fma_mix_source(module, operands[i], &sources[i],
                                           &source_kinds[i],
                                           &source_register_offsets[i])) {
      return iree_ok_status();
    }
    has_f16_source =
        has_f16_source || loom_amdgpu_fma_mix_source_is_f16(source_kinds[i]);
  }
  if (!has_f16_source) {
    return iree_ok_status();
  }
  loom_amdgpu_canonicalize_mulf_mix_sources(sources, source_register_offsets,
                                            source_kinds);

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  bool addend_literal_zero = false;
  if (!loom_amdgpu_select_fma_mix_zero_addend_descriptor(
          context, source_kinds, &descriptor_ref, &addend_literal_zero)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_mulf_mix_plan_t){
      .sources = {sources[0], sources[1]},
      .source_register_offsets = {source_register_offsets[0],
                                  source_register_offsets[1]},
      .result = result,
      .descriptor_ref = descriptor_ref,
      .addend_literal_zero = addend_literal_zero,
      .source_kinds = {source_kinds[0], source_kinds[1]},
      .lane_count = 1,
  };
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_select_vector_splatted_f16_mix_source(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, uint32_t expected_lane_count,
    loom_value_id_t* out_source,
    loom_amdgpu_fma_mix_source_kind_t* out_source_kind,
    uint32_t* out_source_register_offset) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_source_kind = LOOM_AMDGPU_FMA_MIX_SOURCE_F32;
  *out_source_register_offset = 0;

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  if (loom_amdgpu_vector_f32_lane_count(value_type) != expected_lane_count) {
    return false;
  }

  loom_value_id_t scalar_source = LOOM_VALUE_ID_INVALID;
  if (!loom_value_fact_table_query_uniform_element_origin(
          fact_table, module, value_id, &scalar_source)) {
    return false;
  }
  if (!loom_amdgpu_select_fma_mix_source(module, scalar_source, out_source,
                                         out_source_kind,
                                         out_source_register_offset)) {
    return false;
  }
  return loom_amdgpu_fma_mix_source_is_f16(*out_source_kind);
}

iree_status_t loom_amdgpu_select_vector_mulf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_mulf_mix_plan_t* out_plan, bool* out_selected) {
  loom_amdgpu_reset_mulf_mix_plan(out_plan);
  *out_selected = false;
  if (!loom_vector_mulf_isa(source_op) ||
      !loom_amdgpu_vector_mulf_fastmath_allows_zero_add(source_op)) {
    return iree_ok_status();
  }

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t lhs = loom_vector_mulf_lhs(source_op);
  const loom_value_id_t rhs = loom_vector_mulf_rhs(source_op);
  const loom_value_id_t result = loom_vector_mulf_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const uint32_t lane_count = loom_amdgpu_vector_f32_lane_count(result_type);
  if (lane_count == 0) {
    return iree_ok_status();
  }

  loom_value_id_t vector_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t splat_source = LOOM_VALUE_ID_INVALID;
  uint32_t splat_source_register_offset = 0;
  loom_amdgpu_fma_mix_source_kind_t splat_source_kind =
      LOOM_AMDGPU_FMA_MIX_SOURCE_F32;
  const bool lhs_is_vector =
      loom_amdgpu_vector_f32_lane_count(loom_module_value_type(module, lhs)) ==
      lane_count;
  const bool rhs_is_vector =
      loom_amdgpu_vector_f32_lane_count(loom_module_value_type(module, rhs)) ==
      lane_count;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (lhs_is_vector && loom_amdgpu_select_vector_splatted_f16_mix_source(
                           module, fact_table, rhs, lane_count, &splat_source,
                           &splat_source_kind, &splat_source_register_offset)) {
    vector_source = lhs;
  } else if (rhs_is_vector &&
             loom_amdgpu_select_vector_splatted_f16_mix_source(
                 module, fact_table, lhs, lane_count, &splat_source,
                 &splat_source_kind, &splat_source_register_offset)) {
    vector_source = rhs;
  } else {
    return iree_ok_status();
  }

  loom_value_id_t sources[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT] = {
      splat_source,
      vector_source,
  };
  uint32_t source_register_offsets[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT] = {
      splat_source_register_offset, 0};
  loom_amdgpu_fma_mix_source_kind_t
      source_kinds[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
          splat_source_kind,
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
          LOOM_AMDGPU_FMA_MIX_SOURCE_F32,
      };
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  bool addend_literal_zero = false;
  if (!loom_amdgpu_select_fma_mix_zero_addend_descriptor(
          context, source_kinds, &descriptor_ref, &addend_literal_zero)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_mulf_mix_plan_t){
      .sources = {sources[0], sources[1]},
      .source_register_offsets = {source_register_offsets[0],
                                  source_register_offsets[1]},
      .result = result,
      .descriptor_ref = descriptor_ref,
      .addend_literal_zero = addend_literal_zero,
      .source_kinds = {source_kinds[0], source_kinds[1]},
      .lane_count = lane_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_extract_fma_mix_register_unit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t register_offset,
    loom_value_id_t* out_source) {
  *out_source = low_source;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_source_type =
      loom_module_value_type(module, low_source);
  const uint32_t unit_count =
      loom_low_register_type_unit_count(low_source_type);
  if (unit_count == 1 && register_offset == 0) {
    return iree_ok_status();
  }
  if (unit_count == 0 || register_offset >= unit_count) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU mixed-FMA register unit");
    IREE_BUILTIN_UNREACHABLE();
  }
  const loom_type_t unit_type =
      loom_low_register_carrier_type_with_unit_count(low_source_type, 1);
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    register_offset, unit_type, out_source);
}

static bool loom_amdgpu_fma_mix_plan_src2_literal_zero(
    const loom_amdgpu_fma_mix_plan_t* plan) {
  return iree_any_bit_set(plan->flags,
                          LOOM_AMDGPU_FMA_MIX_PLAN_SRC2_LITERAL_ZERO);
}

static bool loom_amdgpu_fma_mix_plan_src2_materialized_zero(
    const loom_amdgpu_fma_mix_plan_t* plan) {
  return iree_any_bit_set(plan->flags,
                          LOOM_AMDGPU_FMA_MIX_PLAN_SRC2_MATERIALIZED_ZERO);
}

static bool loom_amdgpu_fma_mix_plan_has_implicit_src2_zero(
    const loom_amdgpu_fma_mix_plan_t* plan) {
  return loom_amdgpu_fma_mix_plan_src2_literal_zero(plan) ||
         loom_amdgpu_fma_mix_plan_src2_materialized_zero(plan);
}

static iree_status_t loom_amdgpu_lookup_fma_mix_packet_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan,
    loom_value_id_t out_operands[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT]) {
  const uint32_t source_count =
      loom_amdgpu_fma_mix_plan_has_implicit_src2_zero(plan)
          ? LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT - 1
          : LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT;
  for (uint32_t i = 0; i < source_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &out_operands[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_fma_mix_register_unit(
        context, source_op, out_operands[i], plan->source_register_offsets[i],
        &out_operands[i]));
    if (loom_amdgpu_fma_mix_source_is_f16(plan->source_kinds[i])) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, out_operands[i], &out_operands[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fma_mix_src2_literal_zero_attrs(
    loom_low_lower_context_t* context, loom_named_attr_t* out_attrs,
    iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("imm32"), &out_attrs[0].name_id));
  out_attrs[0].value = loom_attr_i64(0);
  *out_attr_count = 1;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_fma_mix_packet_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan, loom_type_t result_type,
    loom_value_id_t sources[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT],
    iree_host_size_t* out_source_count, loom_named_attr_t* attrs,
    iree_host_size_t* out_attr_count) {
  *out_source_count = loom_amdgpu_fma_mix_plan_has_implicit_src2_zero(plan)
                          ? LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT - 1
                          : LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT;
  *out_attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_fma_mix_packet_sources(
      context, source_op, plan, sources));
  if (loom_amdgpu_fma_mix_plan_src2_literal_zero(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_fma_mix_src2_literal_zero_attrs(
        context, attrs, out_attr_count));
  } else if (loom_amdgpu_fma_mix_plan_src2_materialized_zero(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
        result_type, &sources[2]));
    *out_source_count = LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fma_mix_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan, loom_type_t result_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  iree_host_size_t operand_count = 0;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_fma_mix_packet_sources(
      context, source_op, plan, result_type, operands, &operand_count, attrs,
      &attr_count));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, plan->descriptor_ref, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_tied_fma_mix_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan, loom_value_id_t accumulator,
    loom_type_t result_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t sources[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  iree_host_size_t source_count = 0;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_fma_mix_packet_sources(
      context, source_op, plan, result_type, sources, &source_count, attrs,
      &attr_count));
  loom_value_id_t operands[1 + LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
      accumulator,
      sources[0],
      sources[1],
      sources[2],
  };
  const iree_host_size_t operand_count = 1 + source_count;
  const loom_tied_result_t tied_results[] = {
      {
          .result_index = 0,
          .operand_index = 0,
          .has_type_change = false,
      },
  };

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, plan->descriptor_ref, &descriptor));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
      tied_results, IREE_ARRAYSIZE(tied_results), source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_scalar_fmaf_mix(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fma_mix_packet(context, source_op, plan,
                                                       result_type, &result));
  return loom_low_lower_bind_value(context, plan->result, result);
}

static iree_status_t loom_amdgpu_packed_ternary_packet_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t register_offset,
    loom_type_t result_packet_type, loom_value_id_t* out_source) {
  *out_source = low_source;
  const loom_type_t low_source_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_source);
  const uint32_t unit_count =
      loom_low_register_type_unit_count(low_source_type);
  const uint32_t packet_unit_count =
      loom_low_register_type_unit_count(result_packet_type);
  if (unit_count == 0 || packet_unit_count == 0 ||
      register_offset > unit_count ||
      packet_unit_count > unit_count - register_offset) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU packed ternary register range");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (unit_count == packet_unit_count && register_offset == 0) {
    return iree_ok_status();
  }
  const loom_type_t source_packet_type =
      loom_low_register_carrier_type_with_unit_count(low_source_type,
                                                     packet_unit_count);
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    register_offset, source_packet_type,
                                    out_source);
}

static iree_status_t loom_amdgpu_emit_packed_ternary_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_packed_ternary_flags_t flags, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_type_t packet_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_tied_result_t tied_accumulator[] = {
      {
          .result_index = 0,
          .operand_index = 0,
          .has_type_change = false,
      },
  };
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  if (iree_any_bit_set(flags,
                       LOOM_AMDGPU_PACKED_TERNARY_FLAG_TIED_ACCUMULATOR)) {
    tied_results = tied_accumulator;
    tied_result_count = IREE_ARRAYSIZE(tied_accumulator);
  }

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), &packet_type, 1, tied_results,
      tied_result_count, source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_packed_ternary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_packed_ternary_plan_t* plan) {
  if (plan->packet_count == 0 || plan->packet_unit_count == 0 ||
      plan->packet_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      plan->packet_count > UINT32_MAX / plan->packet_unit_count ||
      plan->register_count != plan->packet_count * plan->packet_unit_count) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU packed ternary packet shape");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t low_sources[LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(low_sources); ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &low_sources[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_legalize_vop3_scalar_sources(
      context, source_op, low_sources));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  const loom_type_t packet_type =
      loom_low_register_carrier_type_with_unit_count(result_type,
                                                     plan->packet_unit_count);

  loom_value_id_t packet_results[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    loom_value_id_t operands[LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    const uint32_t register_offset = packet_index * plan->packet_unit_count;
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_packed_ternary_packet_source(
          context, source_op, low_sources[i], register_offset, packet_type,
          &operands[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_ternary_packet(
        context, source_op, plan->descriptor_ref, plan->flags, operands,
        IREE_ARRAYSIZE(operands), packet_type, &packet_results[packet_index]));
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, packet_results, plan->packet_count, result_type,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

static bool loom_amdgpu_mulf_mix_source_is_vector(
    loom_low_lower_context_t* context, const loom_amdgpu_mulf_mix_plan_t* plan,
    uint32_t source_index) {
  if (plan->lane_count == 1) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(
      loom_low_lower_context_module(context), plan->sources[source_index]);
  return loom_amdgpu_vector_f32_lane_count(type) == plan->lane_count;
}

static iree_status_t loom_amdgpu_mulf_mix_lane_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan, const loom_value_id_t* low_sources,
    uint32_t source_index, uint32_t lane, loom_type_t lane_type,
    loom_value_id_t* out_source) {
  *out_source = low_sources[source_index];
  if (loom_amdgpu_mulf_mix_source_is_vector(context, plan, source_index)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_sources[source_index], lane, lane_type,
        out_source));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_fma_mix_register_unit(
      context, source_op, *out_source,
      plan->source_register_offsets[source_index], out_source));
  if (loom_amdgpu_fma_mix_source_is_f16(plan->source_kinds[source_index])) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, *out_source, out_source));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_mulf_mix(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_mulf_mix_operand_form_diagnostic(
      context, source_op, plan));

  loom_value_id_t low_sources[LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(low_sources); ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &low_sources[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_type_t lane_type = result_type;
  if (plan->lane_count > 1) {
    lane_type = loom_low_register_carrier_type_with_unit_count(result_type, 1);
  }

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  if (plan->addend_literal_zero) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_intern(context, IREE_SV("imm32"), &attrs[0].name_id));
    attrs[0].value = loom_attr_i64(0);
    attr_count = 1;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &zero));
  }

  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t lane = 0; lane < plan->lane_count; ++lane) {
    loom_value_id_t operands[LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        zero,
    };
    for (uint32_t i = 0; i < LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_mulf_mix_lane_source(
          context, source_op, plan, low_sources, i, lane, lane_type,
          &operands[i]));
    }
    const iree_host_size_t operand_count =
        plan->addend_literal_zero ? LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT
                                  : IREE_ARRAYSIZE(operands);
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, plan->descriptor_ref, operands, operand_count,
        loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1, &low_op));
    lane_results[lane] = loom_value_slice_get(loom_low_op_results(low_op), 0);
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, lane_results, plan->lane_count, result_type,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}
