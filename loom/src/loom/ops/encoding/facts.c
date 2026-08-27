// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the encoding dialect.

#include "loom/ops/encoding/facts.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/encoding/operand.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/encoding/summary.h"
#include "loom/util/fact_table.h"

static loom_value_fact_address_layout_t loom_encoding_facts_dense_layout(void) {
  return (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
      .rank = 0,
      .strides = NULL,
  };
}

static iree_status_t loom_encoding_facts_make_summary(
    loom_fact_context_t* context, loom_encoding_role_t role,
    uint16_t static_spec_encoding_id,
    loom_value_fact_address_layout_t address_layout,
    loom_value_fact_storage_schema_t storage_schema, loom_value_facts_t* out) {
  loom_value_fact_encoding_summary_t summary = {
      .role = role,
      .static_spec_encoding_id = static_spec_encoding_id,
      .address_layout = address_layout,
      .storage_schema = storage_schema,
  };
  return loom_value_facts_make_encoding_summary(context, summary, out);
}

static bool loom_encoding_define_has_dynamic_params(
    const loom_encoding_define_param_view_t* params) {
  return params->dynamic_values.count != 0 || params->dynamic_names.count != 0;
}

iree_status_t loom_encoding_static_value_facts(loom_fact_context_t* context,
                                               const loom_module_t* module,
                                               uint16_t encoding_id,
                                               loom_type_t result_type,
                                               loom_value_facts_t* out_facts) {
  const loom_encoding_t* spec = loom_module_encoding(module, encoding_id);
  loom_encoding_role_t role = loom_type_is_encoding(result_type)
                                  ? loom_type_encoding_role(result_type)
                                  : LOOM_ENCODING_ROLE_UNKNOWN;
  if (role == LOOM_ENCODING_ROLE_UNKNOWN) {
    role = loom_encoding_static_role(module, spec);
  }

  loom_value_facts_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_encoding_family_summary_t family_summary;
  loom_encoding_summarize_verified_static(module, encoding_id, static_strides,
                                          IREE_ARRAYSIZE(static_strides),
                                          &family_summary);
  family_summary.encoding.role = role;
  family_summary.encoding.static_spec_encoding_id = encoding_id;
  return loom_value_facts_make_encoding_summary(
      context, family_summary.encoding, out_facts);
}

iree_status_t loom_encoding_layout_dense_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_encoding_facts_make_summary(
      context, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, /*static_spec_encoding_id=*/0,
      loom_encoding_facts_dense_layout(), (loom_value_fact_storage_schema_t){0},
      &result_facts[0]);
}

iree_status_t loom_encoding_layout_strided_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_attribute_t static_strides =
      loom_encoding_layout_strided_static_strides(op);
  loom_value_facts_t strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  uint16_t dynamic_ordinal = 0;
  for (uint16_t i = 0; i < static_strides.count; ++i) {
    int64_t static_stride = static_strides.i64_array[i];
    if (static_stride == INT64_MIN) {
      strides[i] = operand_facts[dynamic_ordinal++];
    } else {
      strides[i] = loom_value_facts_exact_i64(static_stride);
    }
  }

  loom_value_fact_address_layout_t address_layout = {
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .rank = (uint8_t)static_strides.count,
      .strides = strides,
  };
  return loom_encoding_facts_make_summary(
      context, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, /*static_spec_encoding_id=*/0,
      address_layout, (loom_value_fact_storage_schema_t){0}, &result_facts[0]);
}

iree_status_t loom_encoding_layout_assume_dense_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_encoding_facts_make_summary(
      context, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, /*static_spec_encoding_id=*/0,
      loom_encoding_facts_dense_layout(), (loom_value_fact_storage_schema_t){0},
      &result_facts[0]);
}

iree_status_t loom_encoding_layout_assume_strided_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  int64_t rank = loom_encoding_layout_assume_strided_rank(op);
  if (rank < 0 || rank > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    return loom_encoding_facts_make_summary(
        context, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        /*static_spec_encoding_id=*/0, (loom_value_fact_address_layout_t){0},
        (loom_value_fact_storage_schema_t){0}, &result_facts[0]);
  }

  loom_value_facts_t strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (int64_t i = 0; i < rank; ++i) {
    strides[i] = loom_value_facts_make(0, INT64_MAX, 1);
  }

  loom_value_fact_address_layout_t address_layout = {
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .rank = (uint8_t)rank,
      .strides = strides,
  };
  return loom_encoding_facts_make_summary(
      context, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, /*static_spec_encoding_id=*/0,
      address_layout, (loom_value_fact_storage_schema_t){0}, &result_facts[0]);
}

iree_status_t loom_encoding_define_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, op);
  if (!params.spec) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  const loom_encoding_vtable_t* vtable =
      loom_module_encoding_vtable(module, params.spec);

  loom_type_t result_type =
      loom_module_value_type(module, loom_encoding_define_result(op));
  loom_encoding_role_t role = loom_type_is_encoding(result_type)
                                  ? loom_type_encoding_role(result_type)
                                  : LOOM_ENCODING_ROLE_UNKNOWN;
  if (role == LOOM_ENCODING_ROLE_UNKNOWN) {
    role = vtable->descriptor->role;
  }

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_encoding_family_summary_t family_summary;
  loom_encoding_summarize_verified_define(
      vtable, context, module, op, &params, operand_facts, stride_storage,
      IREE_ARRAYSIZE(stride_storage), &family_summary);

  const bool has_dynamic_params =
      loom_encoding_define_has_dynamic_params(&params);
  if (has_dynamic_params) {
    family_summary.encoding.storage_schema.static_spec_encoding_id = 0;
  }
  family_summary.encoding.role = role;
  family_summary.encoding.static_spec_encoding_id =
      has_dynamic_params ? 0 : loom_encoding_define_spec(op);
  return loom_value_facts_make_encoding_summary(
      context, family_summary.encoding, &result_facts[0]);
}

iree_status_t loom_encoding_assume_spec_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_encoding_static_value_facts(
      context, module, loom_encoding_assume_spec_spec(op),
      loom_module_value_type(module, loom_encoding_assume_spec_result(op)),
      &result_facts[0]);
}

iree_status_t loom_encoding_isa_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary;
  if (!loom_value_facts_query_encoding_summary(context, operand_facts[0],
                                               &summary) ||
      summary.static_spec_encoding_id == 0) {
    result_facts[0] = loom_value_facts_make(0, 1, 1);
    return iree_ok_status();
  }

  result_facts[0] = loom_value_facts_exact_i64(
      summary.static_spec_encoding_id == loom_encoding_isa_spec(op));
  return iree_ok_status();
}

typedef enum loom_encoding_match_result_e {
  LOOM_ENCODING_MATCH_RESULT_UNKNOWN = 0,
  LOOM_ENCODING_MATCH_RESULT_FALSE = 1,
  LOOM_ENCODING_MATCH_RESULT_TRUE = 2,
} loom_encoding_match_result_t;

// Matches one exclusive enum fact. Multiple possible actual values remain
// unknown unless they are disjoint from the requested value. Explicit absence
// facts distinguish semantic none from an unknown zero-valued summary field.
static loom_encoding_match_result_t loom_encoding_match_exclusive_fact(
    uint64_t actual, uint64_t required, bool known_absent,
    bool schema_complete) {
  if (required == 0) {
    if (actual != 0) return LOOM_ENCODING_MATCH_RESULT_FALSE;
    return known_absent || schema_complete ? LOOM_ENCODING_MATCH_RESULT_TRUE
                                           : LOOM_ENCODING_MATCH_RESULT_UNKNOWN;
  }
  if (actual == 0) {
    return known_absent ? LOOM_ENCODING_MATCH_RESULT_FALSE
                        : LOOM_ENCODING_MATCH_RESULT_UNKNOWN;
  }
  if ((actual & required) == 0) return LOOM_ENCODING_MATCH_RESULT_FALSE;
  return actual == required ? LOOM_ENCODING_MATCH_RESULT_TRUE
                            : LOOM_ENCODING_MATCH_RESULT_UNKNOWN;
}

static loom_encoding_match_result_t loom_encoding_match_and(
    loom_encoding_match_result_t lhs, loom_encoding_match_result_t rhs) {
  if (lhs == LOOM_ENCODING_MATCH_RESULT_FALSE ||
      rhs == LOOM_ENCODING_MATCH_RESULT_FALSE) {
    return LOOM_ENCODING_MATCH_RESULT_FALSE;
  }
  if (lhs == LOOM_ENCODING_MATCH_RESULT_UNKNOWN ||
      rhs == LOOM_ENCODING_MATCH_RESULT_UNKNOWN) {
    return LOOM_ENCODING_MATCH_RESULT_UNKNOWN;
  }
  return LOOM_ENCODING_MATCH_RESULT_TRUE;
}

static loom_encoding_match_result_t loom_encoding_match_requirements(
    loom_value_fact_encoding_summary_t summary, loom_attribute_t requirements) {
  const loom_value_fact_encoded_operand_schema_t schema =
      summary.storage_schema.encoded_operand;
  const bool schema_complete =
      summary.storage_schema.static_spec_encoding_id != 0 &&
      !loom_value_fact_encoded_operand_schema_is_unknown(schema);
  loom_encoding_match_result_t match = LOOM_ENCODING_MATCH_RESULT_TRUE;

  if (loom_encoding_match_attr_has_element_format(requirements)) {
    match = loom_encoding_match_and(
        match,
        loom_encoding_match_exclusive_fact(
            schema.element_format,
            loom_encoding_numeric_format_fact(
                loom_encoding_match_attr_element_format(requirements)),
            iree_any_bit_set(
                schema.flags,
                LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ELEMENT_FORMAT_NONE),
            schema_complete));
  }
  if (loom_encoding_match_attr_has_payload_packing(requirements)) {
    match = loom_encoding_match_and(
        match, loom_encoding_match_exclusive_fact(
                   schema.payload_packing,
                   loom_encoding_payload_packing_fact(
                       loom_encoding_match_attr_payload_packing(requirements)),
                   /*known_absent=*/false, schema_complete));
  }
  if (loom_encoding_match_attr_has_affine(requirements)) {
    match = loom_encoding_match_and(
        match,
        loom_encoding_match_exclusive_fact(
            schema.affine_policy,
            loom_encoding_affine_policy_fact(
                loom_encoding_match_attr_affine(requirements)),
            iree_any_bit_set(schema.flags,
                             LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_AFFINE_NONE),
            schema_complete));
  }
  return match;
}

iree_status_t loom_encoding_matches_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary;
  if (!loom_value_facts_query_encoding_summary(context, operand_facts[0],
                                               &summary)) {
    result_facts[0] = loom_value_facts_make(0, 1, 1);
    return iree_ok_status();
  }

  const loom_encoding_match_result_t match = loom_encoding_match_requirements(
      summary, loom_encoding_matches_requirements(op));
  if (match == LOOM_ENCODING_MATCH_RESULT_UNKNOWN) {
    result_facts[0] = loom_value_facts_make(0, 1, 1);
  } else {
    result_facts[0] =
        loom_value_facts_exact_i64(match == LOOM_ENCODING_MATCH_RESULT_TRUE);
  }
  return iree_ok_status();
}

iree_status_t loom_encoding_assume_match_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary = {
      .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
  };
  (void)loom_value_facts_query_encoding_summary(context, operand_facts[0],
                                                &summary);
  summary.role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA;

  loom_value_fact_encoded_operand_schema_t* schema =
      &summary.storage_schema.encoded_operand;
  const loom_attribute_t requirements =
      loom_encoding_assume_match_requirements(op);
  if (loom_encoding_match_attr_has_element_format(requirements)) {
    const loom_encoding_numeric_format_t element_format =
        loom_encoding_match_attr_element_format(requirements);
    schema->element_format = loom_encoding_numeric_format_fact(element_format);
    if (element_format == LOOM_ENCODING_NUMERIC_FORMAT_NONE) {
      schema->flags |= LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ELEMENT_FORMAT_NONE;
    } else {
      schema->flags &=
          ~LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ELEMENT_FORMAT_NONE;
    }
  }
  if (loom_encoding_match_attr_has_payload_packing(requirements)) {
    schema->payload_packing = loom_encoding_payload_packing_fact(
        loom_encoding_match_attr_payload_packing(requirements));
  }
  if (loom_encoding_match_attr_has_affine(requirements)) {
    const loom_encoding_affine_policy_t affine =
        loom_encoding_match_attr_affine(requirements);
    schema->affine_policy = loom_encoding_affine_policy_fact(affine);
    if (affine == LOOM_ENCODING_AFFINE_POLICY_NONE) {
      schema->flags |= LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_AFFINE_NONE;
    } else {
      schema->flags &= ~LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_AFFINE_NONE;
    }
  }

  return loom_value_facts_make_encoding_summary(context, summary,
                                                &result_facts[0]);
}
