// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/vector_construct.h"

#include <string.h>

#include "loom/codegen/low/lower/representation_observer.h"
#include "loom/codegen/low/lower/rule_value.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/descriptors/vector_construct.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/arch/x86/lower/predicate_representation.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_X86_VECTOR_CONSTRUCT_PLAN = 0x7838636Fu,
  LOOM_X86_VECTOR_CONSTRUCT_MAX_LANES = 64,
  LOOM_X86_VECTOR_CONSTRUCT_MAX_CHUNKS = 4,
};

typedef enum loom_x86_vector_construct_element_kind_e {
  LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_INTEGER = 0,
  LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F32 = 1,
  LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F64 = 2,
  LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F16_XMM = 3,
  LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_PREDICATE = 4,
} loom_x86_vector_construct_element_kind_t;

typedef struct loom_x86_vector_construct_shape_t {
  // Logical element type of the source vector.
  loom_scalar_type_t element_type;
  // Number of logical source lanes.
  uint16_t lane_count;
  // Declared bit width of one numeric lane, or one for predicates.
  uint8_t element_bit_width;
  // Logical numeric vector width; predicates select a physical width later.
  uint16_t vector_bit_width;
  // True when the source vector is a predicate.
  bool is_predicate;
} loom_x86_vector_construct_shape_t;

typedef struct loom_x86_vector_construct_plan_t {
  // Number of source scalar operands in logical lane order.
  uint16_t lane_count;
  // Physical bits in each constructed SIMD lane.
  uint8_t physical_lane_bit_width;
  // Physical result register class.
  loom_x86_register_class_t result_register_class;
  // Selected predicate representation, or NONE for numeric vectors.
  loom_low_representation_id_t predicate_representation;
  // Scalar-lane construction mechanism.
  loom_x86_vector_construct_element_kind_t element_kind;
  // True when every predicate lane is exact.
  bool predicate_is_exact;
  // Predicate lanes with exact compile-time values.
  uint64_t predicate_exact_lane_mask;
  // Packed exact predicate bits in logical lane order.
  uint64_t predicate_bits;
  // Retained template slots used by the exact or mixed AVX2 construction.
  uint8_t predicate_template_chunk_mask;
  // Retained full carrier or per-XMM-chunk predicate templates.
  loom_low_lower_read_only_data_id_t
      predicate_template_ids[LOOM_X86_VECTOR_CONSTRUCT_MAX_CHUNKS];
} loom_x86_vector_construct_plan_t;

typedef struct loom_x86_vector_construct_emitter_t {
  // Active lowering context.
  loom_low_lower_context_t* context;
  // Source constructor supplying locations and operands.
  const loom_op_t* source_op;
  // Descriptor vocabulary for the selected x86 view.
  const loom_x86_vector_construct_descriptors_t* descriptors;
  // GPR32 intermediate type.
  loom_type_t gpr32_type;
  // GPR64 intermediate type.
  loom_type_t gpr64_type;
  // XMM intermediate type.
  loom_type_t xmm_type;
  // YMM intermediate type when available.
  loom_type_t ymm_type;
  // ZMM intermediate type when available.
  loom_type_t zmm_type;
  // K intermediate type when available.
  loom_type_t mask_type;
  // Interned immediate field name "lane".
  loom_string_id_t lane_name;
  // Interned immediate field name "control".
  loom_string_id_t control_name;
  // Interned immediate field name "shift".
  loom_string_id_t shift_name;
  // Interned immediate field name "imm64".
  loom_string_id_t immediate_i64_name;
  // Interned immediate field name "imm32".
  loom_string_id_t immediate_i32_name;
  // Interned immediate field name "data".
  loom_string_id_t data_name;
} loom_x86_vector_construct_emitter_t;

static bool loom_x86_vector_construct_numeric_element_bit_width(
    loom_scalar_type_t element_type, uint8_t* out_bit_width) {
  switch (element_type) {
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_bit_width = 8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      *out_bit_width = 16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32:
      *out_bit_width = 32;
      return true;
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F64:
      *out_bit_width = 64;
      return true;
    default:
      *out_bit_width = 0;
      return false;
  }
}

static bool loom_x86_vector_construct_match_shape(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_x86_vector_construct_descriptors_t* descriptors,
    loom_x86_vector_construct_shape_t* out_shape) {
  *out_shape = (loom_x86_vector_construct_shape_t){0};
  if (descriptors == NULL || !loom_vector_from_elements_isa(source_op)) {
    return false;
  }
  const loom_type_t result_type = loom_module_value_type(
      module, loom_vector_from_elements_result(source_op));
  if (!loom_type_is_vector(result_type) || loom_type_rank(result_type) != 1 ||
      !loom_type_is_all_static(result_type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(result_type, 0);
  if (lane_count <= 0 || lane_count > LOOM_X86_VECTOR_CONSTRUCT_MAX_LANES) {
    return false;
  }

  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  if (element_type == LOOM_SCALAR_TYPE_I1) {
    const bool supported =
        descriptors->supports_mask
            ? loom_x86_avx512_predicate_type(result_type, NULL)
            : loom_x86_avx2_predicate_type(result_type, NULL);
    if (!supported) {
      return false;
    }
    *out_shape = (loom_x86_vector_construct_shape_t){
        .element_type = element_type,
        .lane_count = (uint16_t)lane_count,
        .element_bit_width = 1,
        .vector_bit_width = (uint16_t)lane_count,
        .is_predicate = true,
    };
    return true;
  }

  uint8_t element_bit_width = 0;
  if (!loom_x86_vector_construct_numeric_element_bit_width(
          element_type, &element_bit_width)) {
    return false;
  }
  const uint32_t vector_bit_width = (uint32_t)lane_count * element_bit_width;
  loom_x86_register_class_t register_class = 0;
  if (vector_bit_width > descriptors->maximum_vector_bit_width ||
      !loom_x86_register_class_for_vector_bit_width(vector_bit_width,
                                                    &register_class)) {
    return false;
  }
  *out_shape = (loom_x86_vector_construct_shape_t){
      .element_type = element_type,
      .lane_count = (uint16_t)lane_count,
      .element_bit_width = element_bit_width,
      .vector_bit_width = (uint16_t)vector_bit_width,
      .is_predicate = false,
  };
  return true;
}

static bool loom_x86_vector_construct_numeric_is_exact_native_vector(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_vector_construct_shape_t* shape) {
  if (shape->is_predicate || shape->vector_bit_width < 128) {
    return false;
  }
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  uint64_t bits = 0;
  for (iree_host_size_t i = 0; i < elements.count; ++i) {
    if (!loom_low_lower_rule_value_exact_bits(
            module, fact_table, elements.values[i], shape->element_bit_width,
            &bits)) {
      return false;
    }
  }
  return true;
}

static void loom_x86_vector_construct_predicate_facts(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t* out_exact_lane_mask, uint64_t* out_bits) {
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  uint64_t exact_lane_mask = 0;
  uint64_t bits = 0;
  for (iree_host_size_t i = 0; i < elements.count; ++i) {
    bool value = false;
    if (!loom_value_facts_as_exact_bool(
            loom_value_fact_table_lookup(fact_table, elements.values[i]),
            &value)) {
      continue;
    }
    exact_lane_mask |= UINT64_C(1) << i;
    bits |= (uint64_t)value << i;
  }
  *out_exact_lane_mask = exact_lane_mask;
  *out_bits = bits;
}

static iree_status_t loom_x86_vector_construct_retain_avx2_templates(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_x86_vector_construct_plan_t* construction) {
  uint8_t payload[32] = {0};
  const uint8_t lane_byte_count = construction->physical_lane_bit_width / 8u;
  for (uint16_t lane = 0; lane < construction->lane_count; ++lane) {
    const uint64_t lane_bit = UINT64_C(1) << lane;
    if (iree_all_bits_set(construction->predicate_exact_lane_mask, lane_bit) &&
        iree_all_bits_set(construction->predicate_bits, lane_bit)) {
      memset(&payload[lane * lane_byte_count], 0xFF, lane_byte_count);
    }
  }

  loom_low_lower_module_state_t* module_state =
      loom_low_lower_context_module_state(context);
  if (construction->predicate_is_exact) {
    const iree_host_size_t byte_length =
        construction->result_register_class == LOOM_X86_REGISTER_CLASS_YMM ? 32
                                                                           : 16;
    construction->predicate_template_chunk_mask = 1u;
    return loom_low_lower_module_state_intern_read_only_data(
        module_state, iree_make_const_byte_span(payload, byte_length),
        byte_length, source_op->location,
        &construction->predicate_template_ids[0]);
  }

  const uint16_t chunk_lane_count =
      128u / construction->physical_lane_bit_width;
  const uint8_t chunk_count =
      (uint8_t)((construction->lane_count + chunk_lane_count - 1) /
                chunk_lane_count);
  for (uint8_t chunk = 0; chunk < chunk_count; ++chunk) {
    const uint16_t begin = chunk * chunk_lane_count;
    const uint16_t remaining = construction->lane_count - begin;
    const uint16_t count =
        remaining < chunk_lane_count ? remaining : chunk_lane_count;
    const uint64_t chunk_lane_mask = ((UINT64_C(1) << count) - UINT64_C(1))
                                     << begin;
    if (!iree_any_bit_set(construction->predicate_exact_lane_mask,
                          chunk_lane_mask)) {
      continue;
    }
    construction->predicate_template_chunk_mask |= (uint8_t)(1u << chunk);
    IREE_RETURN_IF_ERROR(loom_low_lower_module_state_intern_read_only_data(
        module_state,
        iree_make_const_byte_span(&payload[chunk * 16u], /*data_length=*/16),
        /*minimum_alignment=*/16, source_op->location,
        &construction->predicate_template_ids[chunk]));
  }
  return iree_ok_status();
}

iree_status_t loom_x86_query_vector_construct_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();
  if (environment->vector_lane_projection.source_lane_count != 0) {
    return iree_ok_status();
  }
  loom_x86_vector_construct_shape_t shape;
  if (loom_x86_vector_construct_match_shape(
          environment->module, source_op,
          loom_x86_vector_construct_descriptors(environment->descriptor_set),
          &shape)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  }
  return iree_ok_status();
}

bool loom_x86_vector_construct_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_X86_VECTOR_CONSTRUCT_PLAN;
}

iree_status_t loom_x86_select_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const loom_x86_vector_construct_descriptors_t* descriptors =
      loom_x86_vector_construct_descriptors(
          loom_low_lower_context_descriptor_set(context));
  loom_x86_vector_construct_shape_t shape;
  if (!loom_x86_vector_construct_match_shape(
          loom_low_lower_context_module(context), source_op, descriptors,
          &shape) ||
      loom_x86_vector_construct_numeric_is_exact_native_vector(
          context, source_op, &shape)) {
    return iree_ok_status();
  }

  loom_x86_vector_construct_plan_t selected = {
      .lane_count = shape.lane_count,
      .physical_lane_bit_width = shape.element_bit_width,
      .predicate_representation = LOOM_LOW_REPRESENTATION_ID_NONE,
      .element_kind = LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_INTEGER,
  };
  if (shape.is_predicate) {
    loom_low_lower_representation_lookup_if_ready(
        context, loom_vector_from_elements_result(source_op),
        &selected.predicate_representation);
    if (selected.predicate_representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
      selected.predicate_representation =
          loom_x86_predicate_callable_representation(shape.lane_count);
    }
    const bool mapped =
        descriptors->supports_mask
            ? loom_x86_avx512_predicate_register_class(
                  loom_module_value_type(
                      loom_low_lower_context_module(context),
                      loom_vector_from_elements_result(source_op)),
                  selected.predicate_representation,
                  &selected.result_register_class)
            : loom_x86_avx2_predicate_register_class(
                  loom_module_value_type(
                      loom_low_lower_context_module(context),
                      loom_vector_from_elements_result(source_op)),
                  selected.predicate_representation,
                  &selected.result_register_class);
    IREE_ASSERT(mapped);
    if (selected.predicate_representation ==
        LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
      selected.physical_lane_bit_width = 1;
    } else {
      selected.physical_lane_bit_width =
          (uint8_t)selected.predicate_representation;
    }
    selected.element_kind = LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_PREDICATE;
    loom_x86_vector_construct_predicate_facts(
        context, source_op, &selected.predicate_exact_lane_mask,
        &selected.predicate_bits);
    const uint64_t complete_lane_mask =
        shape.lane_count == 64
            ? UINT64_MAX
            : (UINT64_C(1) << shape.lane_count) - UINT64_C(1);
    selected.predicate_is_exact =
        selected.predicate_exact_lane_mask == complete_lane_mask;
    if (!descriptors->supports_mask) {
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_retain_avx2_templates(
          context, source_op, &selected));
    }
  } else {
    const bool mapped = loom_x86_register_class_for_vector_bit_width(
        shape.vector_bit_width, &selected.result_register_class);
    IREE_ASSERT(mapped);
    if (shape.element_type == LOOM_SCALAR_TYPE_F32) {
      selected.element_kind = LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F32;
    } else if (shape.element_type == LOOM_SCALAR_TYPE_F64) {
      selected.element_kind = LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F64;
    } else if (shape.element_type == LOOM_SCALAR_TYPE_F16 &&
               (((loom_x86_feature_bits_t)loom_low_lower_context_bundle(context)
                     ->config->contract_feature_bits &
                 LOOM_X86_FEATURE_AVX512_FP16) != 0)) {
      selected.element_kind = LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F16_XMM;
    }
  }

  loom_x86_vector_construct_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained_plan), (void**)&retained_plan));
  *retained_plan = selected;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, selected.result_register_class, 1, &result_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_vector_from_elements_result(source_op), result_type));
  *out_plan =
      loom_low_lower_plan_make(LOOM_X86_VECTOR_CONSTRUCT_PLAN, retained_plan);
  return iree_ok_status();
}

void loom_x86_mark_vector_construct_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  const loom_x86_vector_construct_plan_t* construction = plan.target_data;
  if (construction->element_kind ==
      LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_PREDICATE) {
    const loom_value_slice_t elements =
        loom_vector_from_elements_elements(source_op);
    for (iree_host_size_t i = 0; i < elements.count; ++i) {
      if (!iree_any_bit_set(construction->predicate_exact_lane_mask,
                            UINT64_C(1) << i)) {
        loom_low_lower_require_source_value_storage(context,
                                                    elements.values[i]);
      }
    }
    return;
  }
  loom_low_lower_require_source_operands_storage(context, source_op);
}

void loom_x86_describe_vector_construct_plan(
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report) {
  const loom_x86_vector_construct_plan_t* construction = plan.target_data;
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = construction->element_kind ==
                          LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_PREDICATE
                      ? IREE_SV("vector-construction.predicate-pack")
                      : IREE_SV("vector-construction.scalar-chunks"),
  };
}

static iree_status_t loom_x86_vector_construct_emitter_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_x86_vector_construct_emitter_t* out_emitter) {
  *out_emitter = (loom_x86_vector_construct_emitter_t){
      .context = context,
      .source_op = source_op,
      .descriptors = loom_x86_vector_construct_descriptors(
          loom_low_lower_context_descriptor_set(context)),
      .ymm_type = loom_type_none(),
      .zmm_type = loom_type_none(),
      .mask_type = loom_type_none(),
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_GPR32, 1, &out_emitter->gpr32_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_GPR64, 1, &out_emitter->gpr64_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_XMM, 1, &out_emitter->xmm_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_YMM, 1, &out_emitter->ymm_type));
  if (out_emitter->descriptors->supports_mask) {
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_ZMM, 1, &out_emitter->zmm_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_K, 1, &out_emitter->mask_type));
  }
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("lane"),
                                                  &out_emitter->lane_name));
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("control"),
                                                  &out_emitter->control_name));
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("shift"),
                                                  &out_emitter->shift_name));
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      builder, IREE_SV("imm64"), &out_emitter->immediate_i64_name));
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      builder, IREE_SV("imm32"), &out_emitter->immediate_i32_name));
  return loom_builder_intern_string(builder, IREE_SV("data"),
                                    &out_emitter->data_name);
}

static const loom_low_descriptor_t* loom_x86_vector_construct_descriptor(
    const loom_x86_vector_construct_emitter_t* emitter, uint32_t ordinal) {
  IREE_ASSERT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  return &loom_low_lower_context_descriptor_set(emitter->context)
              ->descriptors[ordinal];
}

static iree_status_t loom_x86_vector_construct_emit_op(
    loom_x86_vector_construct_emitter_t* emitter, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, loom_type_t result_type, bool tie_first,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          loom_x86_vector_construct_descriptor(emitter, descriptor_ordinal),
  };
  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      emitter->context, &descriptor, operands, operand_count, attrs,
      &result_type, 1, tie_first ? &tied_result : NULL, tie_first ? 1 : 0,
      emitter->source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_construct_emit_unary(
    loom_x86_vector_construct_emitter_t* emitter, uint32_t descriptor_ordinal,
    loom_value_id_t operand, loom_type_t result_type,
    loom_value_id_t* out_result) {
  return loom_x86_vector_construct_emit_op(
      emitter, descriptor_ordinal, &operand, 1, loom_named_attr_slice_empty(),
      result_type, /*tie_first=*/false, out_result);
}

static iree_status_t loom_x86_vector_construct_emit_immediate(
    loom_x86_vector_construct_emitter_t* emitter, uint32_t descriptor_ordinal,
    loom_value_id_t operand, loom_string_id_t name, int64_t value,
    loom_type_t result_type, bool tie_first, loom_value_id_t* out_result) {
  const loom_named_attr_t attr = {
      .name_id = name,
      .value = loom_attr_i64(value),
  };
  return loom_x86_vector_construct_emit_op(
      emitter, descriptor_ordinal, &operand, 1,
      loom_make_named_attr_slice(&attr, 1), result_type, tie_first, out_result);
}

static iree_status_t loom_x86_vector_construct_emit_binary(
    loom_x86_vector_construct_emitter_t* emitter, uint32_t descriptor_ordinal,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    bool tie_lhs, loom_value_id_t* out_result) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_x86_vector_construct_emit_op(
      emitter, descriptor_ordinal, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), result_type, tie_lhs, out_result);
}

static iree_status_t loom_x86_vector_construct_emit_constant_i64(
    loom_x86_vector_construct_emitter_t* emitter, uint64_t value,
    loom_value_id_t* out_result) {
  const loom_named_attr_t attr = {
      .name_id = emitter->immediate_i64_name,
      .value = loom_attr_i64((int64_t)value),
  };
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_x86_vector_construct_descriptor(
          emitter, emitter->descriptors->scalar.constant_i64),
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      emitter->context, &descriptor, loom_make_named_attr_slice(&attr, 1),
      emitter->gpr64_type, emitter->source_op->location, &low_op));
  *out_result = loom_low_const_result(low_op);
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_construct_emit_constant_i32(
    loom_x86_vector_construct_emitter_t* emitter, uint32_t value,
    loom_value_id_t* out_result) {
  const loom_named_attr_t attr = {
      .name_id = emitter->immediate_i32_name,
      .value = loom_attr_i64((int32_t)value),
  };
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_x86_vector_construct_descriptor(
          emitter, emitter->descriptors->scalar.constant_i32),
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      emitter->context, &descriptor, loom_make_named_attr_slice(&attr, 1),
      emitter->gpr32_type, emitter->source_op->location, &low_op));
  *out_result = loom_low_const_result(low_op);
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_construct_emit_read_only(
    loom_x86_vector_construct_emitter_t* emitter, uint32_t descriptor_ordinal,
    loom_low_lower_read_only_data_id_t data_id, loom_type_t result_type,
    loom_value_id_t* out_result) {
  loom_symbol_ref_t symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_low_lower_module_state_reference_read_only_data(
      loom_low_lower_context_module_state(emitter->context),
      loom_low_lower_context_module(emitter->context), data_id, &symbol));
  const loom_named_attr_t attr = {
      .name_id = emitter->data_name,
      .value = loom_attr_symbol(symbol),
  };
  return loom_x86_vector_construct_emit_op(
      emitter, descriptor_ordinal, /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(&attr, 1), result_type, /*tie_first=*/false,
      out_result);
}

static bool loom_x86_vector_construct_source_is_exclusive(
    const loom_x86_vector_construct_emitter_t* emitter,
    loom_value_id_t source_value_id) {
  const loom_module_t* module = loom_low_lower_context_module(emitter->context);
  const loom_value_t* source_value = loom_module_value(module, source_value_id);
  return source_value != NULL && loom_value_has_single_use(source_value) &&
         !loom_value_has_attribute_uses(source_value) &&
         !loom_module_value_has_type_uses(module, source_value_id) &&
         loom_use_user_op(*loom_value_single_use(source_value)) ==
             emitter->source_op;
}

static iree_status_t loom_x86_vector_construct_copy_shared_source(
    loom_x86_vector_construct_emitter_t* emitter,
    loom_value_id_t source_value_id, loom_value_id_t* inout_low_value_id) {
  if (loom_x86_vector_construct_source_is_exclusive(emitter, source_value_id)) {
    return iree_ok_status();
  }
  const loom_type_t type = loom_module_value_type(
      loom_low_lower_context_module(emitter->context), *inout_low_value_id);
  loom_op_t* copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(
      loom_low_lower_context_builder(emitter->context), *inout_low_value_id,
      /*detached=*/false, type, emitter->source_op->location, &copy_op));
  *inout_low_value_id = loom_low_copy_result(copy_op);
  return iree_ok_status();
}

static uint8_t loom_x86_vector_construct_width_index(
    uint8_t element_bit_width) {
  switch (element_bit_width) {
    case 8:
      return LOOM_X86_VECTOR_CONSTRUCT_WIDTH_I8;
    case 16:
      return LOOM_X86_VECTOR_CONSTRUCT_WIDTH_I16;
    case 32:
      return LOOM_X86_VECTOR_CONSTRUCT_WIDTH_I32;
    case 64:
      return LOOM_X86_VECTOR_CONSTRUCT_WIDTH_I64;
    default:
      IREE_ASSERT_UNREACHABLE("selected x86 vector construction lane width");
      return LOOM_X86_VECTOR_CONSTRUCT_WIDTH_I8;
  }
}

static iree_status_t loom_x86_vector_construct_emit_integer_chunk(
    loom_x86_vector_construct_emitter_t* emitter, const loom_value_id_t* values,
    uint16_t count, uint8_t element_bit_width, loom_value_id_t* out_chunk) {
  const uint8_t width_index =
      loom_x86_vector_construct_width_index(element_bit_width);
  const bool wide_lane = element_bit_width == 64;
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_unary(
      emitter,
      wide_lane ? emitter->descriptors->chunk.move_i64_to_xmm
                : emitter->descriptors->chunk.move_i32_to_xmm,
      values[0], emitter->xmm_type, out_chunk));
  for (uint16_t lane = 1; lane < count; ++lane) {
    const loom_named_attr_t attr = {
        .name_id = emitter->lane_name,
        .value = loom_attr_i64(lane),
    };
    const loom_value_id_t operands[] = {*out_chunk, values[lane]};
    IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_op(
        emitter, emitter->descriptors->chunk.insert_integer[width_index],
        operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(&attr, 1), emitter->xmm_type,
        /*tie_first=*/false, out_chunk));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_construct_emit_f32_chunk(
    loom_x86_vector_construct_emitter_t* emitter, const loom_value_id_t* values,
    uint16_t count, loom_value_id_t* out_chunk) {
  *out_chunk = values[0];
  for (uint16_t lane = 1; lane < count; ++lane) {
    const loom_named_attr_t attr = {
        .name_id = emitter->control_name,
        .value = loom_attr_i64((int64_t)lane << 4),
    };
    const loom_value_id_t operands[] = {*out_chunk, values[lane]};
    IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_op(
        emitter, emitter->descriptors->chunk.insert_f32, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
        emitter->xmm_type, /*tie_first=*/false, out_chunk));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_construct_emit_f64_chunk(
    loom_x86_vector_construct_emitter_t* emitter, const loom_value_id_t* values,
    uint16_t count, loom_value_id_t* out_chunk) {
  *out_chunk = values[0];
  if (count == 1) {
    return iree_ok_status();
  }
  const loom_named_attr_t attr = {
      .name_id = emitter->control_name,
      .value = loom_attr_i64(0),
  };
  const loom_value_id_t operands[] = {values[0], values[1]};
  return loom_x86_vector_construct_emit_op(
      emitter, emitter->descriptors->chunk.join_f64, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
      emitter->xmm_type, /*tie_first=*/false, out_chunk);
}

static iree_status_t loom_x86_vector_construct_emit_f16_xmm_chunk(
    loom_x86_vector_construct_emitter_t* emitter, const loom_value_id_t* values,
    uint16_t count, loom_value_id_t* out_chunk) {
  *out_chunk = values[0];
  for (uint16_t lane = 1; lane < count; ++lane) {
    loom_value_id_t scalar_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_unary(
        emitter, emitter->descriptors->chunk.move_xmm_to_i32, values[lane],
        emitter->gpr32_type, &scalar_bits));
    const loom_named_attr_t attr = {
        .name_id = emitter->lane_name,
        .value = loom_attr_i64(lane),
    };
    const loom_value_id_t operands[] = {*out_chunk, scalar_bits};
    IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_op(
        emitter,
        emitter->descriptors->chunk
            .insert_integer[LOOM_X86_VECTOR_CONSTRUCT_WIDTH_I16],
        operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(&attr, 1), emitter->xmm_type,
        /*tie_first=*/false, out_chunk));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_construct_assemble_chunks(
    loom_x86_vector_construct_emitter_t* emitter,
    loom_x86_register_class_t result_register_class,
    const loom_value_id_t* chunks, uint8_t chunk_count,
    loom_value_id_t* out_result) {
  if (result_register_class == LOOM_X86_REGISTER_CLASS_XMM) {
    *out_result = chunks[0];
    return iree_ok_status();
  }
  const bool is_ymm = result_register_class == LOOM_X86_REGISTER_CLASS_YMM;
  const uint32_t zero_descriptor = is_ymm ? emitter->descriptors->wide.zero_ymm
                                          : emitter->descriptors->wide.zero_zmm;
  const uint32_t insert_descriptor =
      is_ymm ? emitter->descriptors->wide.insert_ymm
             : emitter->descriptors->wide.insert_zmm;
  const loom_type_t result_type =
      is_ymm ? emitter->ymm_type : emitter->zmm_type;
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_op(
      emitter, zero_descriptor, NULL, 0, loom_named_attr_slice_empty(),
      result_type, /*tie_first=*/false, out_result));
  for (uint8_t lane = 0; lane < chunk_count; ++lane) {
    const loom_named_attr_t attr = {
        .name_id = emitter->lane_name,
        .value = loom_attr_i64(lane),
    };
    const loom_value_id_t operands[] = {*out_result, chunks[lane]};
    IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_op(
        emitter, insert_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(&attr, 1), result_type,
        /*tie_first=*/false, out_result));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_construct_emit_simd_values(
    loom_x86_vector_construct_emitter_t* emitter,
    const loom_x86_vector_construct_plan_t* construction,
    const loom_value_id_t* values, loom_value_id_t* out_result) {
  const uint16_t chunk_lane_count =
      128u / construction->physical_lane_bit_width;
  const uint8_t chunk_count =
      (uint8_t)((construction->lane_count + chunk_lane_count - 1) /
                chunk_lane_count);
  IREE_ASSERT_LE(chunk_count, LOOM_X86_VECTOR_CONSTRUCT_MAX_CHUNKS);
  loom_value_id_t chunks[LOOM_X86_VECTOR_CONSTRUCT_MAX_CHUNKS];
  for (uint8_t chunk = 0; chunk < chunk_count; ++chunk) {
    const uint16_t begin = chunk * chunk_lane_count;
    const uint16_t remaining = construction->lane_count - begin;
    const uint16_t count =
        remaining < chunk_lane_count ? remaining : chunk_lane_count;
    switch (construction->element_kind) {
      case LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_INTEGER:
      case LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_PREDICATE: {
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_integer_chunk(
            emitter, &values[begin], count,
            construction->physical_lane_bit_width, &chunks[chunk]));
        break;
      }
      case LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F32: {
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_f32_chunk(
            emitter, &values[begin], count, &chunks[chunk]));
        break;
      }
      case LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F64: {
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_f64_chunk(
            emitter, &values[begin], count, &chunks[chunk]));
        break;
      }
      case LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_F16_XMM: {
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_f16_xmm_chunk(
            emitter, &values[begin], count, &chunks[chunk]));
        break;
      }
    }
  }
  return loom_x86_vector_construct_assemble_chunks(
      emitter, construction->result_register_class, chunks, chunk_count,
      out_result);
}

static iree_status_t loom_x86_vector_construct_pack_predicate_half(
    loom_x86_vector_construct_emitter_t* emitter,
    const loom_x86_vector_construct_plan_t* construction,
    const loom_value_id_t* values, uint16_t begin, uint16_t count,
    loom_value_id_t* out_bits) {
  *out_bits = LOOM_VALUE_ID_INVALID;
  const uint32_t exact_bits = (uint32_t)(construction->predicate_bits >> begin);
  if (exact_bits != 0) {
    IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_constant_i32(
        emitter, exact_bits, out_bits));
  }
  uint16_t accumulator_source_lane = UINT16_MAX;
  const loom_value_slice_t source_elements =
      loom_vector_from_elements_elements(emitter->source_op);
  for (uint16_t lane = 0; lane < count; ++lane) {
    const uint16_t source_lane = begin + lane;
    if (iree_any_bit_set(construction->predicate_exact_lane_mask,
                         UINT64_C(1) << source_lane)) {
      continue;
    }
    loom_value_id_t lane_bits = values[source_lane];
    if (lane != 0) {
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_copy_shared_source(
          emitter, source_elements.values[source_lane], &lane_bits));
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_immediate(
          emitter, emitter->descriptors->scalar.shift_left_i32, lane_bits,
          emitter->shift_name, lane, emitter->gpr32_type,
          /*tie_first=*/true, &lane_bits));
    }
    if (*out_bits == LOOM_VALUE_ID_INVALID) {
      *out_bits = lane_bits;
      accumulator_source_lane = lane == 0 ? source_lane : UINT16_MAX;
    } else {
      if (accumulator_source_lane != UINT16_MAX) {
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_copy_shared_source(
            emitter, source_elements.values[accumulator_source_lane],
            out_bits));
        accumulator_source_lane = UINT16_MAX;
      }
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_binary(
          emitter, emitter->descriptors->scalar.or_i32, *out_bits, lane_bits,
          emitter->gpr32_type, /*tie_lhs=*/true, out_bits));
    }
  }
  if (*out_bits == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_x86_vector_construct_emit_constant_i32(emitter, 0, out_bits));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_construct_pack_predicate(
    loom_x86_vector_construct_emitter_t* emitter,
    const loom_x86_vector_construct_plan_t* construction,
    const loom_value_id_t* values, loom_value_id_t* out_bits) {
  if (construction->predicate_is_exact) {
    return loom_x86_vector_construct_emit_constant_i64(
        emitter, construction->predicate_bits, out_bits);
  }
  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  const uint16_t low_count =
      construction->lane_count < 32 ? construction->lane_count : 32;
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_pack_predicate_half(
      emitter, construction, values, 0, low_count, &low));
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_unary(
      emitter, emitter->descriptors->scalar.zero_extend_i32, low,
      emitter->gpr64_type, out_bits));
  if (construction->lane_count <= 32) {
    return iree_ok_status();
  }

  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_pack_predicate_half(
      emitter, construction, values, 32, construction->lane_count - 32, &high));
  loom_value_id_t high_wide = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_unary(
      emitter, emitter->descriptors->scalar.zero_extend_i32, high,
      emitter->gpr64_type, &high_wide));
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_immediate(
      emitter, emitter->descriptors->scalar.shift_left_i64, high_wide,
      emitter->shift_name, 32, emitter->gpr64_type,
      /*tie_first=*/true, &high_wide));
  return loom_x86_vector_construct_emit_binary(
      emitter, emitter->descriptors->scalar.or_i64, *out_bits, high_wide,
      emitter->gpr64_type, /*tie_lhs=*/true, out_bits);
}

static uint8_t loom_x86_vector_construct_register_class_index(
    loom_x86_register_class_t register_class) {
  switch (register_class) {
    case LOOM_X86_REGISTER_CLASS_XMM:
      return LOOM_X86_VECTOR_CONSTRUCT_CLASS_XMM;
    case LOOM_X86_REGISTER_CLASS_YMM:
      return LOOM_X86_VECTOR_CONSTRUCT_CLASS_YMM;
    case LOOM_X86_REGISTER_CLASS_ZMM:
      return LOOM_X86_VECTOR_CONSTRUCT_CLASS_ZMM;
    default:
      IREE_ASSERT_UNREACHABLE("selected x86 SIMD predicate carrier");
      return LOOM_X86_VECTOR_CONSTRUCT_CLASS_XMM;
  }
}

static iree_status_t loom_x86_vector_construct_emit_avx512_predicate(
    loom_x86_vector_construct_emitter_t* emitter,
    const loom_x86_vector_construct_plan_t* construction,
    const loom_value_id_t* values, loom_value_id_t* out_result) {
  loom_value_id_t packed_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_pack_predicate(
      emitter, construction, values, &packed_bits));
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_unary(
      emitter, emitter->descriptors->predicate.move_to_mask, packed_bits,
      emitter->mask_type, &mask));
  if (construction->predicate_representation ==
      LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
    *out_result = mask;
    return iree_ok_status();
  }

  const uint8_t width_index = loom_x86_vector_construct_width_index(
      construction->physical_lane_bit_width);
  const uint8_t class_index = loom_x86_vector_construct_register_class_index(
      construction->result_register_class);
  loom_type_t result_type = emitter->xmm_type;
  if (construction->result_register_class == LOOM_X86_REGISTER_CLASS_YMM) {
    result_type = emitter->ymm_type;
  } else if (construction->result_register_class ==
             LOOM_X86_REGISTER_CLASS_ZMM) {
    result_type = emitter->zmm_type;
  }
  return loom_x86_vector_construct_emit_unary(
      emitter, emitter->descriptors->predicate.expand[width_index][class_index],
      mask, result_type, out_result);
}

static iree_status_t loom_x86_vector_construct_emit_avx2_predicate(
    loom_x86_vector_construct_emitter_t* emitter,
    const loom_x86_vector_construct_plan_t* construction,
    const loom_value_id_t* values, loom_value_id_t* out_result) {
  if (construction->predicate_is_exact) {
    const bool is_ymm =
        construction->result_register_class == LOOM_X86_REGISTER_CLASS_YMM;
    IREE_ASSERT(is_ymm || construction->result_register_class ==
                              LOOM_X86_REGISTER_CLASS_XMM);
    return loom_x86_vector_construct_emit_read_only(
        emitter,
        is_ymm ? emitter->descriptors->predicate.read_only_ymm
               : emitter->descriptors->predicate.read_only_xmm,
        construction->predicate_template_ids[0],
        is_ymm ? emitter->ymm_type : emitter->xmm_type, out_result);
  }

  const uint16_t chunk_lane_count =
      128u / construction->physical_lane_bit_width;
  const uint8_t chunk_count =
      (uint8_t)((construction->lane_count + chunk_lane_count - 1) /
                chunk_lane_count);
  IREE_ASSERT_LE(chunk_count, LOOM_X86_VECTOR_CONSTRUCT_MAX_CHUNKS);
  loom_value_id_t chunks[LOOM_X86_VECTOR_CONSTRUCT_MAX_CHUNKS];
  const bool wide_lane = construction->physical_lane_bit_width == 64;
  const uint8_t width_index = loom_x86_vector_construct_width_index(
      construction->physical_lane_bit_width);
  for (uint8_t chunk = 0; chunk < chunk_count; ++chunk) {
    const uint16_t begin = chunk * chunk_lane_count;
    const uint16_t remaining = construction->lane_count - begin;
    const uint16_t count =
        remaining < chunk_lane_count ? remaining : chunk_lane_count;
    const bool has_template = iree_any_bit_set(
        construction->predicate_template_chunk_mask, (uint8_t)(1u << chunk));
    if (has_template) {
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_read_only(
          emitter, emitter->descriptors->predicate.read_only_xmm,
          construction->predicate_template_ids[chunk], emitter->xmm_type,
          &chunks[chunk]));
      for (uint16_t lane = 0; lane < count; ++lane) {
        const uint16_t source_lane = begin + lane;
        if (iree_any_bit_set(construction->predicate_exact_lane_mask,
                             UINT64_C(1) << source_lane)) {
          continue;
        }
        loom_value_id_t normalized = LOOM_VALUE_ID_INVALID;
        loom_value_id_t source = values[source_lane];
        const loom_value_slice_t source_elements =
            loom_vector_from_elements_elements(emitter->source_op);
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_copy_shared_source(
            emitter, source_elements.values[source_lane], &source));
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_op(
            emitter, emitter->descriptors->scalar.negate_i32, &source, 1,
            loom_named_attr_slice_empty(), emitter->gpr32_type,
            /*tie_first=*/true, &normalized));
        if (wide_lane) {
          IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_unary(
              emitter, emitter->descriptors->scalar.sign_extend_i32, normalized,
              emitter->gpr64_type, &normalized));
        }
        const loom_named_attr_t attr = {
            .name_id = emitter->lane_name,
            .value = loom_attr_i64(lane),
        };
        const loom_value_id_t operands[] = {chunks[chunk], normalized};
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_op(
            emitter, emitter->descriptors->chunk.insert_integer[width_index],
            operands, IREE_ARRAYSIZE(operands),
            loom_make_named_attr_slice(&attr, 1), emitter->xmm_type,
            /*tie_first=*/false, &chunks[chunk]));
      }
      continue;
    }

    loom_value_id_t normalized[16];
    IREE_ASSERT_LE(count, IREE_ARRAYSIZE(normalized));
    const loom_value_slice_t source_elements =
        loom_vector_from_elements_elements(emitter->source_op);
    for (uint16_t lane = 0; lane < count; ++lane) {
      const uint16_t source_lane = begin + lane;
      loom_value_id_t source = values[source_lane];
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_copy_shared_source(
          emitter, source_elements.values[source_lane], &source));
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_op(
          emitter, emitter->descriptors->scalar.negate_i32, &source, 1,
          loom_named_attr_slice_empty(), emitter->gpr32_type,
          /*tie_first=*/true, &normalized[lane]));
      if (wide_lane) {
        IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_unary(
            emitter, emitter->descriptors->scalar.sign_extend_i32,
            normalized[lane], emitter->gpr64_type, &normalized[lane]));
      }
    }
    IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_integer_chunk(
        emitter, normalized, count, construction->physical_lane_bit_width,
        &chunks[chunk]));
  }
  return loom_x86_vector_construct_assemble_chunks(
      emitter, construction->result_register_class, chunks, chunk_count,
      out_result);
}

iree_status_t loom_x86_emit_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  const loom_x86_vector_construct_plan_t* construction = plan.target_data;
  loom_x86_vector_construct_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emitter_initialize(
      context, source_op, &emitter));

  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  loom_value_id_t values[LOOM_X86_VECTOR_CONSTRUCT_MAX_LANES] = {
      LOOM_VALUE_ID_INVALID};
  const bool skips_exact_predicate_lanes =
      construction->element_kind == LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_PREDICATE;
  for (iree_host_size_t i = 0; i < elements.count; ++i) {
    if (!skips_exact_predicate_lanes ||
        !iree_any_bit_set(construction->predicate_exact_lane_mask, UINT64_C(1)
                                                                       << i)) {
      values[i] = loom_low_lower_lookup_value(context, elements.values[i]);
    }
  }

  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (construction->element_kind ==
      LOOM_X86_VECTOR_CONSTRUCT_ELEMENT_PREDICATE) {
    if (emitter.descriptors->supports_mask) {
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_avx512_predicate(
          &emitter, construction, values, &result));
    } else {
      IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_avx2_predicate(
          &emitter, construction, values, &result));
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_x86_vector_construct_emit_simd_values(
        &emitter, construction, values, &result));
  }
  return loom_low_lower_bind_value(
      context, loom_vector_from_elements_result(source_op), result);
}
