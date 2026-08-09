// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the encoding dialect.

#include <stdint.h>

#include "loom/ir/module.h"
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
  uint16_t spec_id = loom_encoding_assume_spec_spec(op);
  const loom_encoding_t* spec = loom_module_encoding(module, spec_id);
  if (!spec) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_type_t result_type =
      loom_module_value_type(module, loom_encoding_assume_spec_result(op));
  loom_encoding_role_t role = loom_type_is_encoding(result_type)
                                  ? loom_type_encoding_role(result_type)
                                  : LOOM_ENCODING_ROLE_UNKNOWN;
  if (role == LOOM_ENCODING_ROLE_UNKNOWN) {
    role = loom_encoding_static_role(module, spec);
  }

  loom_value_facts_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_encoding_family_summary_t family_summary;
  loom_encoding_summarize_verified_static(module, spec_id, static_strides,
                                          IREE_ARRAYSIZE(static_strides),
                                          &family_summary);

  family_summary.encoding.role = role;
  family_summary.encoding.static_spec_encoding_id = spec_id;
  return loom_value_facts_make_encoding_summary(
      context, family_summary.encoding, &result_facts[0]);
}
