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
#include "loom/util/fact_table.h"

static iree_string_view_t loom_encoding_facts_layout_param_name(void) {
  return IREE_SV("layout");
}

static iree_string_view_t loom_encoding_facts_schema_param_name(void) {
  return IREE_SV("schema");
}

static bool loom_encoding_facts_string_id_equal(const loom_module_t* module,
                                                loom_string_id_t string_id,
                                                iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static const loom_named_attr_t* loom_encoding_facts_find_param(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_encoding_facts_string_id_equal(module, entry->name_id, name)) {
      return entry;
    }
  }
  return NULL;
}

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

static iree_status_t loom_encoding_facts_make_unknown_address_layout(
    loom_fact_context_t* context, loom_value_facts_t* out) {
  return loom_encoding_facts_make_summary(
      context, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, /*static_spec_encoding_id=*/0,
      (loom_value_fact_address_layout_t){0},
      (loom_value_fact_storage_schema_t){0}, out);
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
  if (static_strides.kind != LOOM_ATTR_I64_ARRAY ||
      static_strides.count > UINT8_MAX) {
    return loom_encoding_facts_make_unknown_address_layout(context,
                                                           &result_facts[0]);
  }

  loom_value_slice_t dynamic_strides = loom_encoding_layout_strided_strides(op);
  uint16_t expected_dynamic_count = 0;
  for (uint16_t i = 0; i < static_strides.count; ++i) {
    if (static_strides.i64_array[i] == INT64_MIN) {
      ++expected_dynamic_count;
    }
  }
  if (expected_dynamic_count != dynamic_strides.count) {
    return loom_encoding_facts_make_unknown_address_layout(context,
                                                           &result_facts[0]);
  }

  loom_value_facts_t* strides = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_facts_scratch(
      context->table, static_strides.count, &strides));
  uint16_t dynamic_ordinal = 0;
  for (uint16_t i = 0; i < static_strides.count; ++i) {
    int64_t static_stride = static_strides.i64_array[i];
    if (static_stride == INT64_MIN) {
      strides[i] = operand_facts[dynamic_ordinal++];
    } else {
      if (static_stride < 0) {
        return loom_encoding_facts_make_unknown_address_layout(
            context, &result_facts[0]);
      }
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
  if (rank < 0 || rank > UINT8_MAX) {
    return loom_encoding_facts_make_summary(
        context, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        /*static_spec_encoding_id=*/0, (loom_value_fact_address_layout_t){0},
        (loom_value_fact_storage_schema_t){0}, &result_facts[0]);
  }

  loom_value_facts_t* strides = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_facts_scratch(
      context->table, (iree_host_size_t)rank, &strides));
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

  loom_type_t result_type =
      loom_module_value_type(module, loom_encoding_define_result(op));
  loom_encoding_role_t role = loom_type_is_encoding(result_type)
                                  ? loom_type_encoding_role(result_type)
                                  : LOOM_ENCODING_ROLE_UNKNOWN;
  if (role == LOOM_ENCODING_ROLE_UNKNOWN) {
    role = loom_encoding_static_role(module, params.spec);
  }

  loom_value_fact_address_layout_t address_layout = {0};
  loom_value_fact_storage_schema_t storage_schema = {0};
  loom_value_facts_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  if (role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
    (void)loom_encoding_query_static_address_layout(
        module, loom_encoding_define_spec(op), static_strides,
        IREE_ARRAYSIZE(static_strides), &address_layout);
  } else if (role == LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
    (void)loom_encoding_query_static_storage_schema(
        module, loom_encoding_define_spec(op), &storage_schema);
  } else if (role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    const loom_named_attr_t* dynamic_layout = loom_encoding_facts_find_param(
        module, params.dynamic_names, loom_encoding_facts_layout_param_name());
    uint16_t dynamic_ordinal = 0;
    if (loom_encoding_define_dynamic_param_ordinal(&params, dynamic_layout,
                                                   &dynamic_ordinal)) {
      loom_value_fact_encoding_summary_t layout_summary = {0};
      if (loom_value_facts_query_encoding_summary(
              context, operand_facts[dynamic_ordinal], &layout_summary)) {
        address_layout = layout_summary.address_layout;
      }
    } else {
      (void)loom_encoding_query_static_address_layout(
          module, loom_encoding_define_spec(op), static_strides,
          IREE_ARRAYSIZE(static_strides), &address_layout);
    }

    const loom_named_attr_t* dynamic_schema = loom_encoding_facts_find_param(
        module, params.dynamic_names, loom_encoding_facts_schema_param_name());
    if (loom_encoding_define_dynamic_param_ordinal(&params, dynamic_schema,
                                                   &dynamic_ordinal)) {
      loom_value_fact_encoding_summary_t schema_summary = {0};
      if (loom_value_facts_query_encoding_summary(
              context, operand_facts[dynamic_ordinal], &schema_summary)) {
        storage_schema = schema_summary.storage_schema;
      }
    } else {
      (void)loom_encoding_query_static_storage_schema(
          module, loom_encoding_define_spec(op), &storage_schema);
    }
  }

  return loom_encoding_facts_make_summary(
      context, role, loom_encoding_define_spec(op), address_layout,
      storage_schema, &result_facts[0]);
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

  loom_value_fact_address_layout_t address_layout = {0};
  loom_value_fact_storage_schema_t storage_schema = {0};
  loom_value_facts_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  if (role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT ||
      role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    (void)loom_encoding_query_static_address_layout(
        module, spec_id, static_strides, IREE_ARRAYSIZE(static_strides),
        &address_layout);
  }
  if (role == LOOM_ENCODING_ROLE_STORAGE_SCHEMA ||
      role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    (void)loom_encoding_query_static_storage_schema(module, spec_id,
                                                    &storage_schema);
  }

  return loom_encoding_facts_make_summary(
      context, role, spec_id, address_layout, storage_schema, &result_facts[0]);
}
