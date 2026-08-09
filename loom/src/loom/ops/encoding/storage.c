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
#include "loom/util/numeric_format.h"

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

static bool loom_encoding_string_id_equal(const loom_module_t* module,
                                          loom_string_id_t string_id,
                                          iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
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

static bool loom_encoding_positive_u16_param(const loom_named_attr_t* parameter,
                                             uint16_t* out_value) {
  *out_value = 0;
  if (!parameter || parameter->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(parameter->value);
  if (value <= 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_nonnegative_u16_param(
    const loom_named_attr_t* parameter, uint16_t* out_value) {
  *out_value = 0;
  if (!parameter || parameter->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(parameter->value);
  if (value < 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_nonnegative_u16_param_or_default(
    const loom_named_attr_t* parameter, uint16_t default_value,
    uint16_t* out_value) {
  *out_value = default_value;
  if (!parameter) return true;
  if (parameter->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(parameter->value);
  if (value < 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_positive_u16_param_or_default(
    const loom_named_attr_t* parameter, uint16_t default_value,
    uint16_t* out_value) {
  *out_value = default_value;
  if (!parameter) return true;
  if (parameter->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(parameter->value);
  if (value <= 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_symbol_param(
    const loom_module_t* module, const loom_named_attr_t* parameter,
    loom_encoding_matrix_operand_symbol_set_t symbol_set, uint64_t* out_value) {
  *out_value = 0;
  if (!parameter || parameter->value.kind != LOOM_ATTR_STRING ||
      parameter->value.string_id == LOOM_STRING_ID_INVALID ||
      parameter->value.string_id >= module->strings.count) {
    return false;
  }
  iree_string_view_t symbol =
      module->strings.entries[parameter->value.string_id];
  return loom_encoding_matrix_operand_lookup_symbol(symbol_set, symbol,
                                                    out_value);
}

static bool loom_encoding_symbol_param_or_default(
    const loom_module_t* module, const loom_named_attr_t* parameter,
    loom_encoding_matrix_operand_symbol_set_t symbol_set,
    uint64_t default_value, uint64_t* out_value) {
  if (!parameter) {
    *out_value = default_value;
    return true;
  }
  return loom_encoding_symbol_param(module, parameter, symbol_set, out_value);
}

static bool loom_encoding_bool_param_or_default(
    const loom_named_attr_t* parameter, bool default_value, bool* out_value) {
  *out_value = default_value;
  if (!parameter) return true;
  if (parameter->value.kind != LOOM_ATTR_BOOL) return false;
  *out_value = loom_attr_as_bool(parameter->value);
  return true;
}

static bool loom_encoding_scale_group_shape(
    const loom_named_attr_t* parameter, uint32_t scale_topology,
    uint16_t scale_group_element_count,
    uint16_t out_shape[LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK]) {
  for (uint8_t i = 0; i < LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK; ++i) {
    out_shape[i] = 0;
  }

  const bool is_1d = iree_any_bit_set(
      scale_topology, LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D |
                          LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D);
  const bool is_2d =
      iree_any_bit_set(scale_topology, LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D);
  if (!parameter) {
    if (is_2d) return false;
    if (is_1d && scale_group_element_count != 0) {
      out_shape[0] = scale_group_element_count;
    }
    return true;
  }
  if (parameter->value.count == 0 ||
      parameter->value.count > LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK) {
    return false;
  }

  for (uint16_t i = 0; i < parameter->value.count; ++i) {
    out_shape[i] = (uint16_t)parameter->value.i64_array[i];
  }
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

static bool loom_encoding_static_strided_layout(
    const loom_encoding_t* encoding, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  const loom_named_attr_t* params[LOOM_ENCODING_STRIDED_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_STRIDED_PARAMETER_COUNT_, params);
  const loom_named_attr_t* strides =
      params[LOOM_ENCODING_STRIDED_PARAMETER_STRIDES];
  if (strides) {
    if (strides->value.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK ||
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

  const loom_named_attr_t* stride =
      params[LOOM_ENCODING_STRIDED_PARAMETER_STRIDE];
  if (!stride || stride_capacity < 1 || !stride_storage ||
      stride->value.i64 < 0) {
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
  const loom_named_attr_t*
      params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_COUNT_, params);

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
  uint16_t scale_group_shape[LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK] = {0};
  uint16_t scale_operands = 0;
  uint16_t sparsity_group_elements = 0;
  uint16_t sparsity_group_nonzero_elements = 0;
  bool zero_scale_fallback = false;
  uint64_t payload_packing_value = 0;
  uint64_t scale_topology_value = 0;
  uint64_t affine_policy_value = 0;
  uint64_t rounding_policy_value = 0;
  uint64_t codebook_policy_value = 0;
  uint64_t sparsity_policy_value = 0;
  if (!loom_encoding_symbol_param(
          module, params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_ELEMENT_FORMAT],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
          &element_format) ||
      element_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE ||
      !loom_encoding_symbol_param_or_default(
          module,
          params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_PAYLOAD_PACKING],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_PAYLOAD_PACKING,
          LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT,
          &payload_packing_value) ||
      !loom_encoding_symbol_param_or_default(
          module, params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_FORMAT],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
          LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, &scale_format) ||
      !loom_encoding_symbol_param_or_default(
          module,
          params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SECONDARY_SCALE_FORMAT],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
          LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, &secondary_scale_format) ||
      !loom_encoding_symbol_param_or_default(
          module, params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_TOPOLOGY],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SCALE_TOPOLOGY,
          LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE, &scale_topology_value) ||
      !loom_encoding_positive_u16_param(
          params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_PAYLOAD_ELEMENTS],
          &payload_elements) ||
      !loom_encoding_nonnegative_u16_param(
          params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_PAYLOAD_REGISTERS],
          &payload_registers) ||
      !loom_encoding_nonnegative_u16_param_or_default(
          params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_GROUP_ELEMENTS],
          /*default_value=*/0, &scale_group_elements) ||
      !loom_encoding_nonnegative_u16_param_or_default(
          params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_OPERANDS],
          /*default_value=*/0, &scale_operands) ||
      !loom_encoding_symbol_param_or_default(
          module, params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_AFFINE],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_AFFINE_POLICY,
          LOOM_VALUE_FACT_AFFINE_POLICY_NONE, &affine_policy_value) ||
      !loom_encoding_symbol_param_or_default(
          module, params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_ROUNDING],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_ROUNDING_POLICY,
          LOOM_VALUE_FACT_ROUNDING_POLICY_NONE, &rounding_policy_value) ||
      !loom_encoding_symbol_param_or_default(
          module, params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_CODEBOOK],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_CODEBOOK_POLICY,
          LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE, &codebook_policy_value) ||
      !loom_encoding_symbol_param_or_default(
          module, params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SPARSITY],
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SPARSITY_POLICY,
          LOOM_VALUE_FACT_SPARSITY_POLICY_NONE, &sparsity_policy_value) ||
      !loom_encoding_nonnegative_u16_param_or_default(
          params
              [LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SPARSITY_GROUP_ELEMENTS],
          /*default_value=*/0, &sparsity_group_elements) ||
      !loom_encoding_nonnegative_u16_param_or_default(
          params
              [LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SPARSITY_GROUP_NONZERO_ELEMENTS],
          /*default_value=*/0, &sparsity_group_nonzero_elements) ||
      !loom_encoding_bool_param_or_default(
          params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_ZERO_SCALE_FALLBACK],
          /*default_value=*/false, &zero_scale_fallback)) {
    return true;
  }
  payload_packing = (uint32_t)payload_packing_value;
  scale_topology = (uint32_t)scale_topology_value;
  affine_policy = (uint32_t)affine_policy_value;
  rounding_policy = (uint32_t)rounding_policy_value;
  codebook_policy = (uint32_t)codebook_policy_value;
  sparsity_policy = (uint32_t)sparsity_policy_value;
  if (!loom_encoding_scale_group_shape(
          params[LOOM_ENCODING_MATRIX_OPERAND_PARAMETER_SCALE_GROUP_SHAPE],
          scale_topology, scale_group_elements, scale_group_shape)) {
    return true;
  }

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
      .sparsity_group =
          {
              .nonzero_element_count = sparsity_group_nonzero_elements,
              .element_count = sparsity_group_elements,
          },
      .payload_register_count = payload_registers,
      .payload_element_count = payload_elements,
      .scale_group =
          {
              .element_count = scale_group_elements,
          },
      .scale_operand_count = scale_operands,
  };
  for (uint8_t i = 0; i < LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK; ++i) {
    encoded_operand.scale_group.shape[i] = scale_group_shape[i];
  }
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
  if (!loom_value_fact_encoded_operand_schema_sparsity_is_complete(
          encoded_operand)) {
    return true;
  }

  out_schema->encoded_operand = encoded_operand;
  return true;
}

static bool loom_encoding_static_ggml_q8_0_schema(
    uint16_t encoding_id, const loom_encoding_t* encoding,
    loom_value_fact_storage_schema_t* out_schema) {
  const loom_named_attr_t* params[LOOM_ENCODING_GGML_Q8_0_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_GGML_Q8_0_PARAMETER_COUNT_, params);
  uint16_t block_elements = 0;
  uint16_t storage_bytes = 0;
  if (!loom_encoding_positive_u16_param_or_default(
          params[LOOM_ENCODING_GGML_Q8_0_PARAMETER_BLOCK_ELEMS],
          /*default_value=*/32, &block_elements) ||
      !loom_encoding_positive_u16_param_or_default(
          params[LOOM_ENCODING_GGML_Q8_0_PARAMETER_STORAGE_BYTES],
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
      .scale_group =
          {
              .element_count = block_elements,
              .shape = {block_elements},
          },
      .scale_operand_count = 1,
  };
  return true;
}

typedef struct loom_encoding_named_fp8_schema_t {
  const loom_encoding_family_descriptor_t* descriptor;
  uint8_t rounding_parameter_index;
  loom_value_fact_numeric_format_flags_t element_format;
} loom_encoding_named_fp8_schema_t;

static const loom_encoding_named_fp8_schema_t kLoomEncodingNamedFp8Schemas[] = {
    {&loom_encoding_ieee_fp8_e4m3_family_descriptor,
     LOOM_ENCODING_IEEE_FP8_E4M3_PARAMETER_ROUNDING,
     LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3},
    {&loom_encoding_ieee_fp8_e5m2_family_descriptor,
     LOOM_ENCODING_IEEE_FP8_E5M2_PARAMETER_ROUNDING,
     LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2},
    {&loom_encoding_fp8_e4m3fn_family_descriptor,
     LOOM_ENCODING_FP8_E4M3FN_PARAMETER_ROUNDING,
     LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN},
    {&loom_encoding_fp8_e4m3fnuz_family_descriptor,
     LOOM_ENCODING_FP8_E4M3FNUZ_PARAMETER_ROUNDING,
     LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ},
    {&loom_encoding_fp8_e5m2fnuz_family_descriptor,
     LOOM_ENCODING_FP8_E5M2FNUZ_PARAMETER_ROUNDING,
     LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ},
};

static const loom_encoding_named_fp8_schema_t*
loom_encoding_find_named_fp8_schema(
    const loom_encoding_family_descriptor_t* descriptor) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLoomEncodingNamedFp8Schemas);
       ++i) {
    const loom_encoding_named_fp8_schema_t* entry =
        &kLoomEncodingNamedFp8Schemas[i];
    if (entry->descriptor == descriptor) return entry;
  }
  return NULL;
}

static bool loom_encoding_static_named_fp8_schema(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* family_descriptor,
    uint16_t encoding_id, const loom_encoding_t* encoding,
    loom_value_fact_storage_schema_t* out_schema) {
  const loom_encoding_named_fp8_schema_t* schema =
      loom_encoding_find_named_fp8_schema(family_descriptor);
  if (!schema) return false;

  const loom_named_attr_t* rounding_parameter = NULL;
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* parameter = &encoding->attributes[i];
    if (loom_encoding_parameter_descriptor_index(parameter) ==
        schema->rounding_parameter_index) {
      rounding_parameter = parameter;
      break;
    }
  }
  uint64_t rounding_policy_value = 0;
  if (!loom_encoding_symbol_param_or_default(
          module, rounding_parameter,
          LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_ROUNDING_POLICY,
          LOOM_VALUE_FACT_ROUNDING_POLICY_NONE, &rounding_policy_value)) {
    return true;
  }

  out_schema->static_spec_encoding_id = encoding_id;
  out_schema->encoded_operand = (loom_value_fact_encoded_operand_schema_t){
      .element_format = schema->element_format,
      .payload_packing = LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES,
      .rounding_policy = (uint32_t)rounding_policy_value,
      .payload_element_count = 1,
  };
  return true;
}

static bool loom_encoding_query_static_address_layout_rec(
    const loom_module_t* module, uint16_t encoding_id, uint8_t depth,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!module || depth > 4) return false;
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (!encoding || !loom_encoding_static_parameters_are_valid(encoding)) {
    return false;
  }
  const loom_encoding_family_descriptor_t* family_descriptor =
      loom_module_encoding_family_descriptor(module, encoding);
  if (family_descriptor == &loom_encoding_dense_family_descriptor) {
    *out_layout = loom_encoding_dense_address_layout();
    return true;
  }
  if (family_descriptor == &loom_encoding_strided_family_descriptor) {
    return loom_encoding_static_strided_layout(encoding, stride_storage,
                                               stride_capacity, out_layout);
  }
  if (family_descriptor != &loom_encoding_physical_storage_family_descriptor) {
    return false;
  }

  const loom_named_attr_t*
      params[LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_COUNT_, params);
  const loom_named_attr_t* layout =
      params[LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_LAYOUT];
  if (!layout) return false;
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
  if (!encoding || !loom_encoding_static_parameters_are_valid(encoding)) {
    return false;
  }
  const loom_encoding_family_descriptor_t* family_descriptor =
      loom_module_encoding_family_descriptor(module, encoding);
  if (family_descriptor == &loom_encoding_physical_storage_family_descriptor) {
    const loom_named_attr_t*
        params[LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_COUNT_];
    loom_encoding_collect_parameter_slots(
        encoding, LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_COUNT_, params);
    const loom_named_attr_t* schema =
        params[LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_SCHEMA];
    if (!schema) return false;
    return loom_encoding_query_static_storage_schema_rec(
        module, loom_attr_as_encoding_id(schema->value), (uint8_t)(depth + 1),
        out_schema);
  }

  if (!family_descriptor ||
      family_descriptor->role != LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
    return false;
  }

  *out_schema = (loom_value_fact_storage_schema_t){
      .static_spec_encoding_id = encoding_id,
  };
  if (family_descriptor == &loom_encoding_matrix_operand_family_descriptor) {
    return loom_encoding_static_matrix_operand_schema(module, encoding_id,
                                                      encoding, out_schema);
  }
  if (family_descriptor == &loom_encoding_ggml_q8_0_family_descriptor) {
    return loom_encoding_static_ggml_q8_0_schema(encoding_id, encoding,
                                                 out_schema);
  }
  if (loom_encoding_static_named_fp8_schema(
          module, family_descriptor, encoding_id, encoding, out_schema)) {
    return true;
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

bool loom_encoding_query_storage_schema_content_facts(
    const loom_value_fact_storage_schema_t* storage_schema,
    loom_scalar_type_t element_type, loom_value_facts_t* out_facts) {
  if (!out_facts) return false;
  *out_facts = loom_value_facts_unknown();
  if (!loom_scalar_type_is_float(element_type)) {
    return false;
  }
  const loom_value_fact_numeric_format_flags_t element_format =
      storage_schema != NULL &&
              storage_schema->encoded_operand.element_format !=
                  LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE
          ? storage_schema->encoded_operand.element_format
          : loom_numeric_format_from_scalar_type(element_type);
  if (storage_schema != NULL &&
      iree_any_bit_set(storage_schema->encoded_operand.rounding_policy,
                       LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY)) {
    out_facts->flags |= LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_NOT_INF |
                        LOOM_VALUE_FACT_FINITE;
  }
  const loom_numeric_format_info_t* element_format_info = NULL;
  if (loom_numeric_format_info(element_format, &element_format_info) &&
      element_format_info->kind == LOOM_NUMERIC_FORMAT_KIND_FLOAT) {
    if (!iree_any_bit_set(element_format_info->flags,
                          LOOM_NUMERIC_FORMAT_FLAG_HAS_NAN)) {
      out_facts->flags |= LOOM_VALUE_FACT_NOT_NAN;
    }
    if (!iree_any_bit_set(element_format_info->flags,
                          LOOM_NUMERIC_FORMAT_FLAG_HAS_INFINITY)) {
      out_facts->flags |= LOOM_VALUE_FACT_NOT_INF;
    }
    if (loom_value_facts_is_not_nan(*out_facts) &&
        loom_value_facts_is_not_inf(*out_facts)) {
      out_facts->flags |= LOOM_VALUE_FACT_FINITE;
    }
  }
  if (storage_schema != NULL &&
      iree_any_bit_set(storage_schema->encoded_operand.rounding_policy,
                       LOOM_VALUE_FACT_ROUNDING_POLICY_FLUSH_SUBNORMAL)) {
    out_facts->flags |= LOOM_VALUE_FACT_NOT_SUBNORMAL;
  }
  return out_facts->flags != 0;
}

bool loom_encoding_query_type_storage_content_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_facts_t* out_facts) {
  if (!out_facts) return false;
  *out_facts = loom_value_facts_unknown();
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  loom_value_fact_storage_schema_t storage_schema = {0};
  if (!loom_encoding_query_type_storage_schema(context, module, type,
                                               &storage_schema)) {
    return loom_encoding_query_storage_schema_content_facts(NULL, element_type,
                                                            out_facts);
  }
  return loom_encoding_query_storage_schema_content_facts(
      &storage_schema, element_type, out_facts);
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

typedef struct loom_encoding_physical_storage_static_params_t {
  // Optional statically bound address layout parameter.
  const loom_named_attr_t* layout;
  // Optional statically bound storage schema parameter.
  const loom_named_attr_t* schema;
} loom_encoding_physical_storage_static_params_t;

static loom_encoding_physical_storage_static_params_t
loom_encoding_physical_storage_static_params(const loom_encoding_t* encoding) {
  const loom_named_attr_t*
      slots[LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_COUNT_, slots);
  return (loom_encoding_physical_storage_static_params_t){
      .layout = slots[LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_LAYOUT],
      .schema = slots[LOOM_ENCODING_PHYSICAL_STORAGE_PARAMETER_SCHEMA],
  };
}

enum loom_encoding_physical_storage_violation_bits_e {
  LOOM_ENCODING_PHYSICAL_STORAGE_LAYOUT_ROLE_VIOLATION = 1u << 0,
  LOOM_ENCODING_PHYSICAL_STORAGE_SCHEMA_ROLE_VIOLATION = 1u << 1,
};
typedef uint8_t loom_encoding_physical_storage_violations_t;

static loom_encoding_physical_storage_violations_t
loom_encoding_physical_storage_static_violations(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  const loom_encoding_physical_storage_static_params_t params =
      loom_encoding_physical_storage_static_params(encoding);
  loom_encoding_physical_storage_violations_t violations = 0;
  if (params.layout) {
    const loom_encoding_t* nested = loom_module_encoding(
        module, loom_attr_as_encoding_id(params.layout->value));
    if (loom_encoding_static_role(module, nested) !=
        LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
      violations |= LOOM_ENCODING_PHYSICAL_STORAGE_LAYOUT_ROLE_VIOLATION;
    }
  }
  if (params.schema) {
    const loom_encoding_t* nested = loom_module_encoding(
        module, loom_attr_as_encoding_id(params.schema->value));
    if (loom_encoding_static_role(module, nested) !=
        LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
      violations |= LOOM_ENCODING_PHYSICAL_STORAGE_SCHEMA_ROLE_VIOLATION;
    }
  }
  return violations;
}

static bool loom_encoding_physical_storage_is_static_valid(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  return loom_encoding_physical_storage_static_violations(module, encoding) ==
         0;
}

static iree_status_t loom_encoding_physical_storage_diagnose_static(
    const loom_module_t* module, const loom_encoding_t* encoding,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  const loom_encoding_physical_storage_violations_t violations =
      loom_encoding_physical_storage_static_violations(module, encoding);
  if (iree_any_bit_set(violations,
                       LOOM_ENCODING_PHYSICAL_STORAGE_LAYOUT_ROLE_VIOLATION)) {
    IREE_RETURN_IF_ERROR(loom_encoding_emit_role_error(
        emitter, op, loom_encoding_layout_param_name(),
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }
  if (iree_any_bit_set(violations,
                       LOOM_ENCODING_PHYSICAL_STORAGE_SCHEMA_ROLE_VIOLATION)) {
    IREE_RETURN_IF_ERROR(loom_encoding_emit_role_error(
        emitter, op, loom_encoding_schema_param_name(),
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
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

  const loom_encoding_physical_storage_static_params_t static_params =
      loom_encoding_physical_storage_static_params(params->spec);
  const loom_named_attr_t* dynamic_layout = loom_encoding_find_param(
      module, params->dynamic_names, loom_encoding_layout_param_name());
  if (!static_params.layout && !dynamic_layout) {
    return loom_encoding_emit_param_error(emitter, op, LOOM_ERR_ENCODING_007,
                                          loom_encoding_layout_param_name());
  }

  const loom_named_attr_t* dynamic_schema = loom_encoding_find_param(
      module, params->dynamic_names, loom_encoding_schema_param_name());
  if (!static_params.schema && !dynamic_schema) {
    return loom_encoding_emit_param_error(emitter, op, LOOM_ERR_ENCODING_007,
                                          loom_encoding_schema_param_name());
  }

  if (dynamic_layout) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_dynamic_param(
        module, op, emitter, params, dynamic_layout,
        loom_encoding_layout_param_name(), LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }

  if (dynamic_schema) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_dynamic_param(
        module, op, emitter, params, dynamic_schema,
        loom_encoding_schema_param_name(), LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
  }

  return iree_ok_status();
}

static const loom_encoding_vtable_t loom_encoding_physical_storage_vtable = {
    .descriptor = &loom_encoding_physical_storage_family_descriptor,
    .is_static_valid = loom_encoding_physical_storage_is_static_valid,
    .diagnose_static = loom_encoding_physical_storage_diagnose_static,
    .verify_define = loom_encoding_physical_storage_verify_define,
};

iree_status_t loom_encoding_register_physical_storage_family(
    loom_context_t* context) {
  return loom_context_register_encoding_vtable(
      context, &loom_encoding_physical_storage_vtable);
}
