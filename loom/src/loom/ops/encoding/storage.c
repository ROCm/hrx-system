// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/storage.h"

#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/encoding/matrix_operand.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/util/fact_table.h"

static iree_string_view_t loom_encoding_physical_storage_name(void) {
  return IREE_SV("physical_storage");
}

static iree_string_view_t loom_encoding_matrix_operand_name(void) {
  return IREE_SV("matrix_operand");
}

static iree_string_view_t loom_encoding_ggml_q8_0_name(void) {
  return IREE_SV("ggml_q8_0");
}

static iree_string_view_t loom_encoding_layout_param_name(void) {
  return IREE_SV("layout");
}

static iree_string_view_t loom_encoding_schema_param_name(void) {
  return IREE_SV("schema");
}

static iree_string_view_t loom_encoding_block_elems_param_name(void) {
  return IREE_SV("block_elems");
}

static iree_string_view_t loom_encoding_storage_bytes_param_name(void) {
  return IREE_SV("storage_bytes");
}

static iree_string_view_t loom_encoding_stride_param_name(void) {
  return IREE_SV("stride");
}

static iree_string_view_t loom_encoding_strides_param_name(void) {
  return IREE_SV("strides");
}

static iree_string_view_t loom_encoding_element_format_param_name(void) {
  return IREE_SV("element_format");
}

static iree_string_view_t loom_encoding_payload_packing_param_name(void) {
  return IREE_SV("payload_packing");
}

static iree_string_view_t loom_encoding_payload_elements_param_name(void) {
  return IREE_SV("payload_elements");
}

static iree_string_view_t loom_encoding_payload_registers_param_name(void) {
  return IREE_SV("payload_registers");
}

static iree_string_view_t loom_encoding_scale_topology_param_name(void) {
  return IREE_SV("scale_topology");
}

static iree_string_view_t loom_encoding_scale_format_param_name(void) {
  return IREE_SV("scale_format");
}

static iree_string_view_t loom_encoding_secondary_scale_format_param_name(
    void) {
  return IREE_SV("secondary_scale_format");
}

static iree_string_view_t loom_encoding_scale_group_elements_param_name(void) {
  return IREE_SV("scale_group_elements");
}

static iree_string_view_t loom_encoding_scale_operands_param_name(void) {
  return IREE_SV("scale_operands");
}

static iree_string_view_t loom_encoding_affine_param_name(void) {
  return IREE_SV("affine");
}

static iree_string_view_t loom_encoding_rounding_param_name(void) {
  return IREE_SV("rounding");
}

static iree_string_view_t loom_encoding_codebook_param_name(void) {
  return IREE_SV("codebook");
}

static iree_string_view_t loom_encoding_sparsity_param_name(void) {
  return IREE_SV("sparsity");
}

static iree_string_view_t loom_encoding_zero_scale_fallback_param_name(void) {
  return IREE_SV("zero_scale_fallback");
}

static bool loom_encoding_string_id_equal(const loom_module_t* module,
                                          loom_string_id_t string_id,
                                          iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static bool loom_encoding_name_equal(const loom_module_t* module,
                                     const loom_encoding_t* encoding,
                                     iree_string_view_t expected) {
  if (!encoding) return false;
  return loom_encoding_string_id_equal(module, encoding->name_id, expected);
}

static const loom_named_attr_t* loom_encoding_find_param(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_encoding_string_id_equal(module, entry->name_id, name)) {
      return entry;
    }
  }
  return NULL;
}

static bool loom_encoding_static_u16_param(const loom_module_t* module,
                                           const loom_encoding_t* encoding,
                                           iree_string_view_t param_name,
                                           uint16_t* out_value) {
  *out_value = 0;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry || entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(entry->value);
  if (value <= 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_static_nonnegative_u16_param(
    const loom_module_t* module, const loom_encoding_t* encoding,
    iree_string_view_t param_name, uint16_t* out_value) {
  *out_value = 0;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry || entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(entry->value);
  if (value < 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_static_nonnegative_u16_param_or_default(
    const loom_module_t* module, const loom_encoding_t* encoding,
    iree_string_view_t param_name, uint16_t default_value,
    uint16_t* out_value) {
  *out_value = default_value;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry) return true;
  if (entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(entry->value);
  if (value < 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_static_positive_u16_param_or_default(
    const loom_module_t* module, const loom_encoding_t* encoding,
    iree_string_view_t param_name, uint16_t default_value,
    uint16_t* out_value) {
  *out_value = default_value;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry) return true;
  if (entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(entry->value);
  if (value <= 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_static_symbol_param(
    const loom_module_t* module, const loom_encoding_t* encoding,
    iree_string_view_t param_name,
    loom_encoding_matrix_operand_symbol_set_t symbol_set, uint64_t* out_value) {
  *out_value = 0;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry || entry->value.kind != LOOM_ATTR_STRING ||
      entry->value.string_id == LOOM_STRING_ID_INVALID ||
      entry->value.string_id >= module->strings.count) {
    return false;
  }
  iree_string_view_t symbol = module->strings.entries[entry->value.string_id];
  return loom_encoding_matrix_operand_lookup_symbol(symbol_set, symbol,
                                                    out_value);
}

static bool loom_encoding_static_symbol_param_or_default(
    const loom_module_t* module, const loom_encoding_t* encoding,
    iree_string_view_t param_name,
    loom_encoding_matrix_operand_symbol_set_t symbol_set,
    uint64_t default_value, uint64_t* out_value) {
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry) {
    *out_value = default_value;
    return true;
  }
  return loom_encoding_static_symbol_param(module, encoding, param_name,
                                           symbol_set, out_value);
}

static bool loom_encoding_static_bool_param_or_default(
    const loom_module_t* module, const loom_encoding_t* encoding,
    iree_string_view_t param_name, bool default_value, bool* out_value) {
  *out_value = default_value;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry) return true;
  if (entry->value.kind != LOOM_ATTR_BOOL) return false;
  *out_value = loom_attr_as_bool(entry->value);
  return true;
}

static loom_value_fact_address_layout_t loom_encoding_dense_address_layout(
    void) {
  return (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
      .rank = 0,
      .strides = NULL,
  };
}

static bool loom_encoding_static_dense_layout_isa(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  return loom_encoding_name_equal(module, encoding, IREE_SV("dense"));
}

static bool loom_encoding_static_strided_layout_isa(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  return loom_encoding_name_equal(module, encoding, IREE_SV("strided"));
}

static bool loom_encoding_static_strided_layout(
    const loom_module_t* module, const loom_encoding_t* encoding,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  const loom_named_attr_t* strides =
      loom_encoding_find_param(module, loom_encoding_attrs(encoding),
                               loom_encoding_strides_param_name());
  if (strides) {
    if (strides->value.kind != LOOM_ATTR_I64_ARRAY ||
        strides->value.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK ||
        strides->value.count > stride_capacity ||
        (strides->value.count > 0 && !stride_storage)) {
      return false;
    }
    for (uint16_t i = 0; i < strides->value.count; ++i) {
      int64_t stride = strides->value.i64_array[i];
      if (stride < 0) return false;
      stride_storage[i] = loom_value_facts_exact_i64(stride);
    }
    *out_layout = (loom_value_fact_address_layout_t){
        .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
        .rank = (uint8_t)strides->value.count,
        .strides = strides->value.count > 0 ? stride_storage : NULL,
    };
    return true;
  }

  const loom_named_attr_t* stride = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_stride_param_name());
  if (!stride || stride->value.kind != LOOM_ATTR_I64 || stride_capacity < 1 ||
      !stride_storage || stride->value.i64 < 0) {
    return false;
  }
  stride_storage[0] = loom_value_facts_exact_i64(stride->value.i64);
  *out_layout = (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .rank = 1,
      .strides = stride_storage,
  };
  return true;
}

static bool loom_encoding_static_matrix_operand_schema(
    const loom_module_t* module, uint16_t encoding_id,
    const loom_encoding_t* encoding,
    loom_value_fact_storage_schema_t* out_schema) {
  out_schema->static_spec_encoding_id = encoding_id;

  uint64_t element_format = 0;
  uint32_t payload_packing = 0;
  uint64_t scale_format = 0;
  uint64_t secondary_scale_format = 0;
  uint32_t scale_topology = 0;
  uint32_t affine_policy = 0;
  uint32_t rounding_policy = 0;
  uint32_t codebook_policy = 0;
  uint32_t sparsity_policy = 0;
  uint16_t payload_elements = 0;
  uint16_t payload_registers = 0;
  uint16_t scale_group_elements = 0;
  uint16_t scale_operands = 0;
  bool zero_scale_fallback = false;
  uint64_t payload_packing_value = 0;
  uint64_t scale_topology_value = 0;
  uint64_t affine_policy_value = 0;
  uint64_t rounding_policy_value = 0;
  uint64_t codebook_policy_value = 0;
  uint64_t sparsity_policy_value = 0;
  if (!loom_encoding_static_symbol_param(
          module, encoding, loom_encoding_element_format_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
          &element_format) ||
      element_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE ||
      !loom_encoding_static_symbol_param_or_default(
          module, encoding, loom_encoding_payload_packing_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_PAYLOAD_PACKING,
          LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT,
          &payload_packing_value) ||
      !loom_encoding_static_symbol_param_or_default(
          module, encoding, loom_encoding_scale_format_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
          LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, &scale_format) ||
      !loom_encoding_static_symbol_param_or_default(
          module, encoding, loom_encoding_secondary_scale_format_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
          LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, &secondary_scale_format) ||
      !loom_encoding_static_symbol_param_or_default(
          module, encoding, loom_encoding_scale_topology_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SCALE_TOPOLOGY,
          LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE, &scale_topology_value) ||
      !loom_encoding_static_u16_param(
          module, encoding, loom_encoding_payload_elements_param_name(),
          &payload_elements) ||
      !loom_encoding_static_nonnegative_u16_param(
          module, encoding, loom_encoding_payload_registers_param_name(),
          &payload_registers) ||
      !loom_encoding_static_nonnegative_u16_param_or_default(
          module, encoding, loom_encoding_scale_group_elements_param_name(),
          /*default_value=*/0, &scale_group_elements) ||
      !loom_encoding_static_nonnegative_u16_param_or_default(
          module, encoding, loom_encoding_scale_operands_param_name(),
          /*default_value=*/0, &scale_operands) ||
      !loom_encoding_static_symbol_param_or_default(
          module, encoding, loom_encoding_affine_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_AFFINE_POLICY,
          LOOM_VALUE_FACT_AFFINE_POLICY_NONE, &affine_policy_value) ||
      !loom_encoding_static_symbol_param_or_default(
          module, encoding, loom_encoding_rounding_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_ROUNDING_POLICY,
          LOOM_VALUE_FACT_ROUNDING_POLICY_NONE, &rounding_policy_value) ||
      !loom_encoding_static_symbol_param_or_default(
          module, encoding, loom_encoding_codebook_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_CODEBOOK_POLICY,
          LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE, &codebook_policy_value) ||
      !loom_encoding_static_symbol_param_or_default(
          module, encoding, loom_encoding_sparsity_param_name(),
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SPARSITY_POLICY,
          LOOM_VALUE_FACT_SPARSITY_POLICY_NONE, &sparsity_policy_value) ||
      !loom_encoding_static_bool_param_or_default(
          module, encoding, loom_encoding_zero_scale_fallback_param_name(),
          /*default_value=*/false, &zero_scale_fallback)) {
    return true;
  }
  payload_packing = (uint32_t)payload_packing_value;
  scale_topology = (uint32_t)scale_topology_value;
  affine_policy = (uint32_t)affine_policy_value;
  rounding_policy = (uint32_t)rounding_policy_value;
  codebook_policy = (uint32_t)codebook_policy_value;
  sparsity_policy = (uint32_t)sparsity_policy_value;

  loom_value_fact_encoded_operand_schema_t encoded_operand = {
      .element_format = element_format,
      .payload_packing = payload_packing,
      .scale_format = scale_format,
      .secondary_scale_format = secondary_scale_format,
      .scale_topology = scale_topology,
      .affine_policy = affine_policy,
      .rounding_policy = rounding_policy,
      .codebook_policy = codebook_policy,
      .sparsity_policy = sparsity_policy,
      .payload_register_count = payload_registers,
      .payload_element_count = payload_elements,
      .scale_group_element_count = scale_group_elements,
      .scale_operand_count = scale_operands,
  };
  if (zero_scale_fallback) {
    encoded_operand.flags |=
        LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK;
  }
  if (encoded_operand.payload_packing == 0) {
    return true;
  }

  if (!loom_value_fact_encoded_operand_schema_scale_is_complete(
          encoded_operand)) {
    return true;
  }

  out_schema->encoded_operand = encoded_operand;
  return true;
}

static bool loom_encoding_static_ggml_q8_0_schema(
    const loom_module_t* module, uint16_t encoding_id,
    const loom_encoding_t* encoding,
    loom_value_fact_storage_schema_t* out_schema) {
  uint16_t block_elements = 0;
  uint16_t storage_bytes = 0;
  if (!loom_encoding_static_positive_u16_param_or_default(
          module, encoding, loom_encoding_block_elems_param_name(),
          /*default_value=*/32, &block_elements) ||
      !loom_encoding_static_positive_u16_param_or_default(
          module, encoding, loom_encoding_storage_bytes_param_name(),
          /*default_value=*/34, &storage_bytes)) {
    return true;
  }
  if (storage_bytes != (uint16_t)(block_elements + 2)) {
    return true;
  }

  out_schema->static_spec_encoding_id = encoding_id;
  out_schema->encoded_operand = (loom_value_fact_encoded_operand_schema_t){
      .element_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8,
      .scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
      .payload_packing = LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES,
      .scale_topology = LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D,
      .affine_policy = LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY,
      .payload_element_count = block_elements,
      .scale_group_element_count = block_elements,
      .scale_operand_count = 1,
  };
  return true;
}

static bool loom_encoding_query_static_address_layout_rec(
    const loom_module_t* module, uint16_t encoding_id, uint8_t depth,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!module || depth > 4) return false;
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (loom_encoding_static_dense_layout_isa(module, encoding)) {
    *out_layout = loom_encoding_dense_address_layout();
    return true;
  }
  if (loom_encoding_static_strided_layout_isa(module, encoding)) {
    return loom_encoding_static_strided_layout(module, encoding, stride_storage,
                                               stride_capacity, out_layout);
  }
  if (!loom_encoding_name_equal(module, encoding,
                                loom_encoding_physical_storage_name())) {
    return false;
  }

  const loom_named_attr_t* layout = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_layout_param_name());
  if (!layout || layout->value.kind != LOOM_ATTR_ENCODING) return false;
  return loom_encoding_query_static_address_layout_rec(
      module, loom_attr_as_encoding_id(layout->value), (uint8_t)(depth + 1),
      stride_storage, stride_capacity, out_layout);
}

bool loom_encoding_query_static_address_layout(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!out_layout) return false;
  *out_layout = (loom_value_fact_address_layout_t){0};
  return loom_encoding_query_static_address_layout_rec(
      module, encoding_id, /*depth=*/0, stride_storage, stride_capacity,
      out_layout);
}

static bool loom_encoding_query_static_storage_schema_rec(
    const loom_module_t* module, uint16_t encoding_id, uint8_t depth,
    loom_value_fact_storage_schema_t* out_schema) {
  if (!module || depth > 4) return false;
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (!encoding) return false;
  if (loom_encoding_name_equal(module, encoding,
                               loom_encoding_physical_storage_name())) {
    const loom_named_attr_t* schema =
        loom_encoding_find_param(module, loom_encoding_attrs(encoding),
                                 loom_encoding_schema_param_name());
    if (!schema || schema->value.kind != LOOM_ATTR_ENCODING) return false;
    return loom_encoding_query_static_storage_schema_rec(
        module, loom_attr_as_encoding_id(schema->value), (uint8_t)(depth + 1),
        out_schema);
  }

  if (loom_encoding_static_role(module, encoding) !=
      LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
    return false;
  }

  *out_schema = (loom_value_fact_storage_schema_t){
      .static_spec_encoding_id = encoding_id,
  };
  if (loom_encoding_name_equal(module, encoding,
                               loom_encoding_matrix_operand_name())) {
    return loom_encoding_static_matrix_operand_schema(module, encoding_id,
                                                      encoding, out_schema);
  }
  if (loom_encoding_name_equal(module, encoding,
                               loom_encoding_ggml_q8_0_name())) {
    return loom_encoding_static_ggml_q8_0_schema(module, encoding_id, encoding,
                                                 out_schema);
  }
  return true;
}

bool loom_encoding_query_static_storage_schema(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_fact_storage_schema_t* out_schema) {
  if (!out_schema) return false;
  *out_schema = (loom_value_fact_storage_schema_t){0};
  return loom_encoding_query_static_storage_schema_rec(module, encoding_id,
                                                       /*depth=*/0, out_schema);
}

static bool loom_encoding_facts_address_layout(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_address_layout_t* out_layout) {
  *out_layout = (loom_value_fact_address_layout_t){0};
  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(context, facts, &summary)) {
    return false;
  }
  if (summary.address_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN) {
    return false;
  }
  *out_layout = summary.address_layout;
  return true;
}

static bool loom_encoding_value_storage_schema(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_storage_schema_t* out_schema) {
  *out_schema = (loom_value_fact_storage_schema_t){0};
  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(context, facts, &summary)) {
    return false;
  }
  if (summary.storage_schema.static_spec_encoding_id == 0 &&
      loom_value_fact_encoded_operand_schema_is_unknown(
          summary.storage_schema.encoded_operand)) {
    return false;
  }
  *out_schema = summary.storage_schema;
  return true;
}

bool loom_encoding_query_type_address_layout(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!out_layout) return false;
  *out_layout = (loom_value_fact_address_layout_t){0};
  if (!module || !loom_type_has_encoding(type)) return false;

  if (loom_type_has_static_encoding(type)) {
    return loom_encoding_query_static_address_layout(
        module, type.encoding_id, stride_storage, stride_capacity, out_layout);
  }

  if (!loom_type_has_ssa_encoding(type) || !context || !context->table) {
    return false;
  }
  loom_value_id_t value_id = loom_type_encoding_value_id(type);
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(context->table, value_id);
  return loom_encoding_facts_address_layout(context, facts, out_layout);
}

bool loom_encoding_query_type_storage_schema(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_fact_storage_schema_t* out_schema) {
  if (!out_schema) return false;
  *out_schema = (loom_value_fact_storage_schema_t){0};
  if (!module || !loom_type_has_encoding(type)) return false;

  if (loom_type_has_static_encoding(type)) {
    return loom_encoding_query_static_storage_schema(module, type.encoding_id,
                                                     out_schema);
  }

  if (!loom_type_has_ssa_encoding(type) || !context || !context->table) {
    return false;
  }
  loom_value_id_t value_id = loom_type_encoding_value_id(type);
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(context->table, value_id);
  return loom_encoding_value_storage_schema(context, facts, out_schema);
}

static iree_status_t loom_encoding_emit(iree_diagnostic_emitter_t emitter,
                                        const loom_op_t* op,
                                        const loom_error_def_t* error,
                                        const loom_diagnostic_param_t* params,
                                        iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_encoding_emit_param_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, iree_string_view_t param_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_static_kind_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, loom_attr_kind_t actual_kind,
    iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_u32(actual_kind),
      loom_param_string(expected_kind),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_010, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_dynamic_type_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t param_name,
    loom_value_id_t value_id) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("encoding")),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_009, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_role_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, iree_string_view_t expected_role) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_string(expected_role),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_011, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_physical_storage_verify_static_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, const loom_named_attr_t* entry,
    iree_string_view_t param_name, loom_encoding_role_t expected_role,
    iree_string_view_t expected_role_name) {
  if (entry->value.kind != LOOM_ATTR_ENCODING) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, param_name, (loom_attr_kind_t)entry->value.kind,
        IREE_SV("encoding"));
  }

  const loom_encoding_t* nested =
      loom_module_encoding(module, loom_attr_as_encoding_id(entry->value));
  loom_encoding_role_t actual_role = loom_encoding_static_role(module, nested);
  if (actual_role != expected_role) {
    return loom_encoding_emit_role_error(emitter, op, param_name,
                                         expected_role_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter,
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* entry, iree_string_view_t param_name,
    loom_encoding_role_t expected_role, iree_string_view_t expected_role_name) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_define_dynamic_param_value(params, entry, &value_id)) {
    return iree_ok_status();
  }

  if (!loom_type_is_encoding(loom_module_value_type(module, value_id))) {
    return loom_encoding_emit_dynamic_type_error(module, emitter, op,
                                                 param_name, value_id);
  }

  loom_encoding_role_t actual_role = loom_encoding_value_role(module, value_id);
  if (actual_role != expected_role) {
    return loom_encoding_emit_role_error(emitter, op, param_name,
                                         expected_role_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return loom_encoding_emit_param_error(emitter, op, LOOM_ERR_ENCODING_008,
                                            param_name);
    }
  }
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return loom_encoding_emit_param_error(emitter, op, LOOM_ERR_ENCODING_008,
                                            param_name);
    }
  }

  const loom_named_attr_t* static_layout = loom_encoding_find_param(
      module, params->static_attrs, loom_encoding_layout_param_name());
  const loom_named_attr_t* dynamic_layout = loom_encoding_find_param(
      module, params->dynamic_names, loom_encoding_layout_param_name());
  if (!static_layout && !dynamic_layout) {
    return loom_encoding_emit_param_error(emitter, op, LOOM_ERR_ENCODING_007,
                                          loom_encoding_layout_param_name());
  }

  const loom_named_attr_t* static_schema = loom_encoding_find_param(
      module, params->static_attrs, loom_encoding_schema_param_name());
  const loom_named_attr_t* dynamic_schema = loom_encoding_find_param(
      module, params->dynamic_names, loom_encoding_schema_param_name());
  if (!static_schema && !dynamic_schema) {
    return loom_encoding_emit_param_error(emitter, op, LOOM_ERR_ENCODING_007,
                                          loom_encoding_schema_param_name());
  }

  if (static_layout) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_static_param(
        module, op, emitter, static_layout, loom_encoding_layout_param_name(),
        LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }
  if (dynamic_layout) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_dynamic_param(
        module, op, emitter, params, dynamic_layout,
        loom_encoding_layout_param_name(), LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }

  if (static_schema) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_static_param(
        module, op, emitter, static_schema, loom_encoding_schema_param_name(),
        LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
  }
  if (dynamic_schema) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_dynamic_param(
        module, op, emitter, params, dynamic_schema,
        loom_encoding_schema_param_name(), LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
  }

  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_static(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  const loom_named_attr_t* layout = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_layout_param_name());
  const loom_named_attr_t* schema = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_schema_param_name());

  for (iree_host_size_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* entry = &encoding->attributes[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' does not support parameter '%.*s'",
          (int)param_name.size, param_name.data);
    }
  }

  if (layout && layout->value.kind != LOOM_ATTR_ENCODING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding 'physical_storage' parameter 'layout' must be an encoding");
  }
  if (schema && schema->value.kind != LOOM_ATTR_ENCODING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding 'physical_storage' parameter 'schema' must be an encoding");
  }

  if (layout) {
    const loom_encoding_t* layout_encoding =
        loom_module_encoding(module, loom_attr_as_encoding_id(layout->value));
    if (loom_encoding_static_role(module, layout_encoding) !=
        LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' parameter 'layout' must be an address "
          "layout encoding");
    }
  }

  if (schema) {
    const loom_encoding_t* schema_encoding =
        loom_module_encoding(module, loom_attr_as_encoding_id(schema->value));
    if (loom_encoding_static_role(module, schema_encoding) !=
        LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' parameter 'schema' must be a storage "
          "schema encoding");
    }
  }

  return iree_ok_status();
}

static const loom_encoding_vtable_t loom_encoding_physical_storage_vtable = {
    .name = IREE_SVL("physical_storage"),
    .role = LOOM_ENCODING_ROLE_PHYSICAL_STORAGE,
    .verify = loom_encoding_physical_storage_verify_static,
    .verify_define = loom_encoding_physical_storage_verify_define,
};

iree_status_t loom_encoding_register_physical_storage_family(
    loom_context_t* context) {
  return loom_context_register_encoding_vtable(
      context, &loom_encoding_physical_storage_vtable);
}
