// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/families.h"

#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/encoding/numeric_transform.h"
#include "loom/ops/encoding/operand.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/encoding/summary.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

static iree_string_view_t loom_encoding_turboquant_kv_name(void) {
  return IREE_SV("turboquant_kv");
}

static iree_string_view_t loom_encoding_operand_name(void) {
  return IREE_SV("encoding.operand");
}

static iree_string_view_t loom_encoding_layout_strided_name(void) {
  return IREE_SV("encoding.layout.strided");
}

static iree_string_view_t loom_encoding_ggml_q8_0_name(void) {
  return IREE_SV("ggml_q8_0");
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
    iree_string_view_t encoding_name, const loom_error_def_t* error,
    iree_string_view_t param_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_static_value_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    iree_string_view_t actual_value, iree_string_view_t expected_values) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_string(actual_value),
      loom_param_string(expected_values),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_013, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_attribute_constraint_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(param_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_dynamic_type_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t encoding_name,
    iree_string_view_t param_name, loom_value_id_t value_id,
    iree_string_view_t expected_type) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(expected_type),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_009, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_role_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    iree_string_view_t expected_role) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_string(expected_role),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_011, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_mutually_exclusive_param_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_a,
    iree_string_view_t param_b) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_a),
      loom_param_string(param_b),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_014, params,
                            IREE_ARRAYSIZE(params));
}

typedef enum loom_encoding_operand_param_requirement_e {
  LOOM_ENCODING_OPERAND_PARAM_OPTIONAL = 0,
  LOOM_ENCODING_OPERAND_PARAM_REQUIRED = 1,
} loom_encoding_operand_param_requirement_t;

static iree_status_t loom_encoding_static_i64(
    const loom_named_attr_t* parameter, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t encoding_name,
    iree_string_view_t param_name,
    loom_encoding_operand_param_requirement_t requirement,
    int64_t default_value, int64_t* out_value, bool* out_ok) {
  *out_value = default_value;
  *out_ok = false;
  if (!parameter) {
    if (requirement == LOOM_ENCODING_OPERAND_PARAM_OPTIONAL) {
      *out_ok = true;
      return iree_ok_status();
    }
    return loom_encoding_emit_param_error(emitter, op, encoding_name,
                                          LOOM_ERR_ENCODING_007, param_name);
  }
  *out_value = loom_attr_as_i64(parameter->value);
  *out_ok = true;
  return iree_ok_status();
}

static bool loom_encoding_static_bool(const loom_named_attr_t* parameter,
                                      bool default_value) {
  return parameter ? loom_attr_as_bool(parameter->value) : default_value;
}

static uint8_t loom_encoding_static_enum(const loom_named_attr_t* parameter,
                                         uint8_t default_value) {
  return parameter ? loom_attr_as_enum(parameter->value) : default_value;
}

static iree_status_t loom_encoding_operand_verify_i64_range(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, int64_t value,
    iree_string_view_t expected_constraint) {
  return loom_encoding_emit_attribute_constraint_error(
      emitter, op, param_name, value, expected_constraint);
}

static iree_status_t loom_encoding_operand_verify_optional_static_i64(
    const loom_named_attr_t* parameter, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    int64_t default_value, int64_t* out_value, bool* out_ok) {
  IREE_RETURN_IF_ERROR(loom_encoding_static_i64(
      parameter, op, emitter, loom_encoding_operand_name(), param_name,
      LOOM_ENCODING_OPERAND_PARAM_OPTIONAL, default_value, out_value, out_ok));
  if (!*out_ok) return iree_ok_status();
  if (*out_value < 0) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, param_name, *out_value, IREE_SV("non-negative i64"));
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_operand_verify_scale_group_shape(
    const loom_named_attr_t* element_count_parameter,
    const loom_named_attr_t* shape_parameter, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, uint64_t scale_topology,
    uint16_t* out_element_count,
    uint16_t out_shape[LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK], bool* out_ok) {
  *out_element_count = 0;
  for (uint8_t i = 0; i < LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK; ++i) {
    out_shape[i] = 0;
  }
  *out_ok = false;

  if (element_count_parameter && shape_parameter) {
    return loom_encoding_emit_mutually_exclusive_param_error(
        emitter, op, loom_encoding_operand_name(),
        IREE_SV("scale_group_elements"), IREE_SV("scale_group_shape"));
  }

  int64_t element_count = 0;
  bool element_count_ok = false;
  IREE_RETURN_IF_ERROR(loom_encoding_operand_verify_optional_static_i64(
      element_count_parameter, op, emitter, IREE_SV("scale_group_elements"),
      /*default_value=*/0, &element_count, &element_count_ok));
  if (!element_count_ok) return iree_ok_status();
  if (element_count > UINT16_MAX) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("scale_group_elements"), element_count,
        IREE_SV("non-negative and <= 65535"));
  }
  *out_element_count = (uint16_t)element_count;

  const uint32_t topology = (uint32_t)scale_topology;
  const bool is_1d =
      iree_any_bit_set(topology, LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D |
                                     LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D);
  const bool is_2d =
      iree_any_bit_set(topology, LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D);
  if (!shape_parameter) {
    if (is_2d) {
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_operand_name(), LOOM_ERR_ENCODING_007,
          IREE_SV("scale_group_shape"));
    }
    if (is_1d && element_count > 0) {
      out_shape[0] = (uint16_t)element_count;
    }
    *out_ok = true;
    return iree_ok_status();
  }
  const uint16_t rank = shape_parameter->value.count;
  if (rank == 0 || rank > LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("scale_group_shape"), rank,
        IREE_SV("1 to 4 positive dimensions"));
  }
  if ((is_1d && rank != 1) || (is_2d && rank != 2)) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("scale_group_shape"), rank,
        is_1d ? IREE_SV("rank 1 for a 1D scale topology")
              : IREE_SV("rank 2 for a 2D scale topology"));
  }

  uint32_t shape_element_count = 1;
  for (uint16_t i = 0; i < rank; ++i) {
    const int64_t dimension = shape_parameter->value.i64_array[i];
    if (dimension <= 0 || dimension > UINT16_MAX) {
      return loom_encoding_operand_verify_i64_range(
          emitter, op, IREE_SV("scale_group_shape"), dimension,
          IREE_SV("positive dimensions <= 65535"));
    }
    if (shape_element_count > UINT16_MAX / (uint32_t)dimension) {
      return loom_encoding_operand_verify_i64_range(
          emitter, op, IREE_SV("scale_group_shape"), UINT16_MAX + 1,
          IREE_SV("product <= 65535"));
    }
    shape_element_count *= (uint32_t)dimension;
    out_shape[i] = (uint16_t)dimension;
  }
  *out_element_count = (uint16_t)shape_element_count;
  *out_ok = true;
  return iree_ok_status();
}

typedef enum loom_encoding_layout_strided_static_violation_e {
  LOOM_ENCODING_LAYOUT_STRIDED_STATIC_VALID = 0,
  LOOM_ENCODING_LAYOUT_STRIDED_STATIC_MISSING_STRIDES,
  LOOM_ENCODING_LAYOUT_STRIDED_STATIC_OVERSIZED_RANK,
  LOOM_ENCODING_LAYOUT_STRIDED_STATIC_NEGATIVE_STRIDE_ARRAY_ELEMENT,
} loom_encoding_layout_strided_static_violation_t;

typedef struct loom_encoding_layout_strided_static_validation_t {
  loom_encoding_layout_strided_static_violation_t violation;
  int64_t actual_value;
} loom_encoding_layout_strided_static_validation_t;

static loom_encoding_layout_strided_static_validation_t
loom_encoding_layout_strided_validate_static(const loom_encoding_t* encoding) {
  const loom_named_attr_t*
      params[LOOM_ENCODING_LAYOUT_STRIDED_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_LAYOUT_STRIDED_PARAMETER_COUNT_, params);
  const loom_named_attr_t* strides =
      params[LOOM_ENCODING_LAYOUT_STRIDED_PARAMETER_STRIDES];
  if (!strides) {
    return (loom_encoding_layout_strided_static_validation_t){
        .violation = LOOM_ENCODING_LAYOUT_STRIDED_STATIC_MISSING_STRIDES,
    };
  }
  if (strides->value.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    return (loom_encoding_layout_strided_static_validation_t){
        .violation = LOOM_ENCODING_LAYOUT_STRIDED_STATIC_OVERSIZED_RANK,
        .actual_value = strides->value.count,
    };
  }
  for (uint16_t i = 0; i < strides->value.count; ++i) {
    if (strides->value.i64_array[i] < 0) {
      return (loom_encoding_layout_strided_static_validation_t){
          .violation =
              LOOM_ENCODING_LAYOUT_STRIDED_STATIC_NEGATIVE_STRIDE_ARRAY_ELEMENT,
          .actual_value = strides->value.i64_array[i],
      };
    }
  }
  return (loom_encoding_layout_strided_static_validation_t){0};
}

static bool loom_encoding_layout_strided_is_static_valid(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  (void)module;
  return loom_encoding_layout_strided_validate_static(encoding).violation ==
         LOOM_ENCODING_LAYOUT_STRIDED_STATIC_VALID;
}

static iree_status_t loom_encoding_layout_strided_diagnose_static(
    const loom_module_t* module, const loom_encoding_t* encoding,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  (void)module;
  const loom_encoding_layout_strided_static_validation_t validation =
      loom_encoding_layout_strided_validate_static(encoding);
  switch (validation.violation) {
    case LOOM_ENCODING_LAYOUT_STRIDED_STATIC_VALID:
      return iree_ok_status();
    case LOOM_ENCODING_LAYOUT_STRIDED_STATIC_MISSING_STRIDES:
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_layout_strided_name(),
          LOOM_ERR_ENCODING_007, IREE_SV("strides"));
    case LOOM_ENCODING_LAYOUT_STRIDED_STATIC_OVERSIZED_RANK:
      return loom_encoding_emit_attribute_constraint_error(
          emitter, op, IREE_SV("strides"), validation.actual_value,
          IREE_SV("rank <= 8"));
    case LOOM_ENCODING_LAYOUT_STRIDED_STATIC_NEGATIVE_STRIDE_ARRAY_ELEMENT:
      return loom_encoding_emit_attribute_constraint_error(
          emitter, op, IREE_SV("strides"), validation.actual_value,
          IREE_SV("non-negative stride"));
  }
  IREE_ASSERT_UNREACHABLE("unknown strided static validation result");
  IREE_BUILTIN_UNREACHABLE();
}

typedef enum loom_encoding_ggml_q8_0_static_violation_e {
  LOOM_ENCODING_GGML_Q8_0_STATIC_VALID = 0,
  LOOM_ENCODING_GGML_Q8_0_STATIC_BLOCK_RANGE,
  LOOM_ENCODING_GGML_Q8_0_STATIC_STORAGE_RANGE,
  LOOM_ENCODING_GGML_Q8_0_STATIC_STORAGE_MISMATCH,
} loom_encoding_ggml_q8_0_static_violation_t;

typedef struct loom_encoding_ggml_q8_0_static_validation_t {
  loom_encoding_ggml_q8_0_static_violation_t violation;
  int64_t actual_value;
} loom_encoding_ggml_q8_0_static_validation_t;

static loom_encoding_ggml_q8_0_static_validation_t
loom_encoding_ggml_q8_0_validate_static(const loom_encoding_t* encoding) {
  const loom_named_attr_t* params[LOOM_ENCODING_GGML_Q8_0_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_GGML_Q8_0_PARAMETER_COUNT_, params);
  const int64_t block_elements =
      params[LOOM_ENCODING_GGML_Q8_0_PARAMETER_BLOCK_ELEMS]
          ? loom_attr_as_i64(
                params[LOOM_ENCODING_GGML_Q8_0_PARAMETER_BLOCK_ELEMS]->value)
          : 32;
  if (block_elements <= 0 || block_elements > UINT16_MAX - 2) {
    return (loom_encoding_ggml_q8_0_static_validation_t){
        .violation = LOOM_ENCODING_GGML_Q8_0_STATIC_BLOCK_RANGE,
        .actual_value = block_elements,
    };
  }
  const int64_t storage_bytes =
      params[LOOM_ENCODING_GGML_Q8_0_PARAMETER_STORAGE_BYTES]
          ? loom_attr_as_i64(
                params[LOOM_ENCODING_GGML_Q8_0_PARAMETER_STORAGE_BYTES]->value)
          : 34;
  if (storage_bytes <= 0 || storage_bytes > UINT16_MAX) {
    return (loom_encoding_ggml_q8_0_static_validation_t){
        .violation = LOOM_ENCODING_GGML_Q8_0_STATIC_STORAGE_RANGE,
        .actual_value = storage_bytes,
    };
  }
  if (storage_bytes != block_elements + 2) {
    return (loom_encoding_ggml_q8_0_static_validation_t){
        .violation = LOOM_ENCODING_GGML_Q8_0_STATIC_STORAGE_MISMATCH,
        .actual_value = storage_bytes,
    };
  }
  return (loom_encoding_ggml_q8_0_static_validation_t){0};
}

static bool loom_encoding_ggml_q8_0_is_static_valid(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  (void)module;
  return loom_encoding_ggml_q8_0_validate_static(encoding).violation ==
         LOOM_ENCODING_GGML_Q8_0_STATIC_VALID;
}

static iree_status_t loom_encoding_ggml_q8_0_diagnose_static(
    const loom_module_t* module, const loom_encoding_t* encoding,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  (void)module;
  const loom_encoding_ggml_q8_0_static_validation_t validation =
      loom_encoding_ggml_q8_0_validate_static(encoding);
  switch (validation.violation) {
    case LOOM_ENCODING_GGML_Q8_0_STATIC_VALID:
      return iree_ok_status();
    case LOOM_ENCODING_GGML_Q8_0_STATIC_BLOCK_RANGE:
      return loom_encoding_emit_attribute_constraint_error(
          emitter, op, IREE_SV("block_elems"), validation.actual_value,
          IREE_SV("positive and <= 65533"));
    case LOOM_ENCODING_GGML_Q8_0_STATIC_STORAGE_RANGE:
      return loom_encoding_emit_attribute_constraint_error(
          emitter, op, IREE_SV("storage_bytes"), validation.actual_value,
          IREE_SV("positive and <= 65535"));
    case LOOM_ENCODING_GGML_Q8_0_STATIC_STORAGE_MISMATCH:
      return loom_encoding_emit_attribute_constraint_error(
          emitter, op, IREE_SV("storage_bytes"), validation.actual_value,
          IREE_SV("block_elems + 2"));
  }
  IREE_ASSERT_UNREACHABLE("unknown ggml_q8_0 static validation result");
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_encoding_operand_static_i64_value(
    const loom_named_attr_t* parameter,
    loom_encoding_operand_param_requirement_t requirement, int64_t minimum,
    int64_t maximum, int64_t default_value, int64_t* out_value) {
  *out_value = default_value;
  if (!parameter) {
    return requirement == LOOM_ENCODING_OPERAND_PARAM_OPTIONAL;
  }
  *out_value = loom_attr_as_i64(parameter->value);
  return *out_value >= minimum && *out_value <= maximum;
}

static bool loom_encoding_operand_static_scale_group_shape_valid(
    const loom_named_attr_t* element_count_parameter,
    const loom_named_attr_t* shape_parameter, uint64_t scale_topology,
    uint16_t* out_element_count,
    uint16_t out_shape[LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK]) {
  *out_element_count = 0;
  memset(out_shape, 0,
         LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK * sizeof(out_shape[0]));
  if (element_count_parameter && shape_parameter) return false;

  if (element_count_parameter) {
    const int64_t element_count =
        loom_attr_as_i64(element_count_parameter->value);
    if (element_count < 0 || element_count > UINT16_MAX) return false;
    *out_element_count = (uint16_t)element_count;
  }

  const uint32_t topology = (uint32_t)scale_topology;
  const bool is_1d =
      iree_any_bit_set(topology, LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D |
                                     LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D);
  const bool is_2d =
      iree_any_bit_set(topology, LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D);
  if (!shape_parameter) {
    if (is_2d) return false;
    if (is_1d && *out_element_count > 0) {
      out_shape[0] = *out_element_count;
    }
    return true;
  }
  const uint16_t rank = shape_parameter->value.count;
  if (rank == 0 || rank > LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK ||
      (is_1d && rank != 1) || (is_2d && rank != 2)) {
    return false;
  }
  uint32_t shape_element_count = 1;
  for (uint16_t i = 0; i < rank; ++i) {
    const int64_t dimension = shape_parameter->value.i64_array[i];
    if (dimension <= 0 || dimension > UINT16_MAX ||
        shape_element_count > UINT16_MAX / (uint32_t)dimension) {
      return false;
    }
    shape_element_count *= (uint32_t)dimension;
    out_shape[i] = (uint16_t)dimension;
  }
  *out_element_count = (uint16_t)shape_element_count;
  return true;
}

static bool loom_encoding_operand_is_static_valid(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  (void)module;
  const loom_named_attr_t* params[LOOM_ENCODING_OPERAND_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_OPERAND_PARAMETER_COUNT_, params);

  const loom_named_attr_t* element_format_param =
      params[LOOM_ENCODING_OPERAND_PARAMETER_ELEMENT_FORMAT];
  if (!element_format_param ||
      loom_encoding_numeric_format_fact(
          (loom_encoding_numeric_format_t)loom_attr_as_enum(
              element_format_param->value)) ==
          LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    return false;
  }
  const uint32_t payload_packing = loom_encoding_payload_packing_fact(
      (loom_encoding_payload_packing_t)loom_encoding_static_enum(
          params[LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_PACKING],
          LOOM_ENCODING_PAYLOAD_PACKING_TARGET_FRAGMENT));
  const uint64_t scale_format = loom_encoding_numeric_format_fact(
      (loom_encoding_numeric_format_t)loom_encoding_static_enum(
          params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_FORMAT],
          LOOM_ENCODING_NUMERIC_FORMAT_NONE));
  const uint64_t secondary_scale_format = loom_encoding_numeric_format_fact(
      (loom_encoding_numeric_format_t)loom_encoding_static_enum(
          params[LOOM_ENCODING_OPERAND_PARAMETER_SECONDARY_SCALE_FORMAT],
          LOOM_ENCODING_NUMERIC_FORMAT_NONE));
  const uint32_t scale_topology = loom_encoding_scale_topology_fact(
      (loom_encoding_scale_topology_t)loom_encoding_static_enum(
          params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_TOPOLOGY],
          LOOM_ENCODING_SCALE_TOPOLOGY_NONE));

  int64_t payload_elements = 0;
  if (!loom_encoding_operand_static_i64_value(
          params[LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_ELEMENTS],
          LOOM_ENCODING_OPERAND_PARAM_REQUIRED, 1, UINT16_MAX,
          /*default_value=*/0, &payload_elements)) {
    return false;
  }
  int64_t payload_registers = 0;
  if (!loom_encoding_operand_static_i64_value(
          params[LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_REGISTERS],
          LOOM_ENCODING_OPERAND_PARAM_OPTIONAL, 0, UINT16_MAX,
          /*default_value=*/0, &payload_registers) ||
      (iree_any_bit_set((uint32_t)payload_packing,
                        LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) &&
       payload_registers == 0)) {
    return false;
  }
  uint16_t scale_group_elements = 0;
  uint16_t scale_group_shape[LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK];
  if (!loom_encoding_operand_static_scale_group_shape_valid(
          params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_GROUP_ELEMENTS],
          params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_GROUP_SHAPE],
          scale_topology, &scale_group_elements, scale_group_shape)) {
    return false;
  }
  int64_t scale_operands = 0;
  if (!loom_encoding_operand_static_i64_value(
          params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_OPERANDS],
          LOOM_ENCODING_OPERAND_PARAM_OPTIONAL, 0, UINT16_MAX,
          /*default_value=*/0, &scale_operands)) {
    return false;
  }

  const uint32_t sparsity = loom_encoding_sparsity_policy_fact(
      (loom_encoding_sparsity_policy_t)loom_encoding_static_enum(
          params[LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY],
          LOOM_ENCODING_SPARSITY_POLICY_NONE));
  int64_t sparsity_group_elements = 0;
  int64_t sparsity_group_nonzero_elements = 0;
  if (!loom_encoding_operand_static_i64_value(
          params[LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY_GROUP_ELEMENTS],
          LOOM_ENCODING_OPERAND_PARAM_OPTIONAL, 0, UINT16_MAX,
          /*default_value=*/0, &sparsity_group_elements) ||
      !loom_encoding_operand_static_i64_value(
          params
              [LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY_GROUP_NONZERO_ELEMENTS],
          LOOM_ENCODING_OPERAND_PARAM_OPTIONAL, 0, UINT16_MAX,
          /*default_value=*/0, &sparsity_group_nonzero_elements)) {
    return false;
  }

  loom_value_fact_encoded_operand_schema_t encoded_operand = {
      .scale_format = scale_format,
      .secondary_scale_format = secondary_scale_format,
      .scale_topology = (uint32_t)scale_topology,
      .scale_group = {.element_count = scale_group_elements},
      .scale_operand_count = (uint16_t)scale_operands,
      .sparsity_policy = (uint32_t)sparsity,
      .sparsity_group =
          {
              .nonzero_element_count =
                  (uint16_t)sparsity_group_nonzero_elements,
              .element_count = (uint16_t)sparsity_group_elements,
          },
  };
  for (uint8_t i = 0; i < LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK; ++i) {
    encoded_operand.scale_group.shape[i] = scale_group_shape[i];
  }
  return loom_value_fact_encoded_operand_schema_scale_is_complete(
             encoded_operand) &&
         loom_value_fact_encoded_operand_schema_sparsity_is_complete(
             encoded_operand);
}

static iree_status_t loom_encoding_operand_diagnose_static(
    const loom_module_t* module, const loom_encoding_t* encoding,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  (void)module;
  bool param_ok = false;
  const loom_named_attr_t*
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_OPERAND_PARAMETER_COUNT_, static_params);

  const loom_named_attr_t* element_format_param =
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_ELEMENT_FORMAT];
  if (!element_format_param) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_operand_name(), LOOM_ERR_ENCODING_007,
        IREE_SV("element_format"));
  }
  const uint64_t element_format = loom_encoding_numeric_format_fact(
      (loom_encoding_numeric_format_t)loom_attr_as_enum(
          element_format_param->value));
  if (element_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("element_format"), 0,
        IREE_SV("non-none numeric format symbol"));
  }

  const uint32_t payload_packing = loom_encoding_payload_packing_fact(
      (loom_encoding_payload_packing_t)loom_encoding_static_enum(
          static_params[LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_PACKING],
          LOOM_ENCODING_PAYLOAD_PACKING_TARGET_FRAGMENT));
  const uint64_t scale_format = loom_encoding_numeric_format_fact(
      (loom_encoding_numeric_format_t)loom_encoding_static_enum(
          static_params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_FORMAT],
          LOOM_ENCODING_NUMERIC_FORMAT_NONE));
  const uint64_t secondary_scale_format = loom_encoding_numeric_format_fact(
      (loom_encoding_numeric_format_t)loom_encoding_static_enum(
          static_params[LOOM_ENCODING_OPERAND_PARAMETER_SECONDARY_SCALE_FORMAT],
          LOOM_ENCODING_NUMERIC_FORMAT_NONE));
  const uint32_t scale_topology = loom_encoding_scale_topology_fact(
      (loom_encoding_scale_topology_t)loom_encoding_static_enum(
          static_params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_TOPOLOGY],
          LOOM_ENCODING_SCALE_TOPOLOGY_NONE));

  int64_t payload_elements = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_static_i64(
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_ELEMENTS], op,
      emitter, loom_encoding_operand_name(), IREE_SV("payload_elements"),
      LOOM_ENCODING_OPERAND_PARAM_REQUIRED,
      /*default_value=*/0, &payload_elements, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (payload_elements <= 0 || payload_elements > UINT16_MAX) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("payload_elements"), payload_elements,
        IREE_SV("positive and <= 65535"));
  }

  int64_t payload_registers = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_static_i64(
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_REGISTERS], op,
      emitter, loom_encoding_operand_name(), IREE_SV("payload_registers"),
      LOOM_ENCODING_OPERAND_PARAM_OPTIONAL,
      /*default_value=*/0, &payload_registers, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (payload_registers < 0 || payload_registers > UINT16_MAX) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("payload_registers"), payload_registers,
        IREE_SV("non-negative and <= 65535"));
  }
  if (iree_any_bit_set((uint32_t)payload_packing,
                       LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) &&
      payload_registers == 0) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("payload_registers"), payload_registers,
        IREE_SV("positive for target-fragment payloads"));
  }

  uint16_t scale_group_elements = 0;
  uint16_t scale_group_shape[LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK] = {0};
  IREE_RETURN_IF_ERROR(loom_encoding_operand_verify_scale_group_shape(
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_GROUP_ELEMENTS],
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_GROUP_SHAPE], op,
      emitter, scale_topology, &scale_group_elements, scale_group_shape,
      &param_ok));
  if (!param_ok) return iree_ok_status();

  int64_t scale_operands = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_operand_verify_optional_static_i64(
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_SCALE_OPERANDS], op,
      emitter, IREE_SV("scale_operands"),
      /*default_value=*/0, &scale_operands, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (scale_operands > UINT16_MAX) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("scale_operands"), scale_operands,
        IREE_SV("non-negative and <= 65535"));
  }

  const uint32_t sparsity = loom_encoding_sparsity_policy_fact(
      (loom_encoding_sparsity_policy_t)loom_encoding_static_enum(
          static_params[LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY],
          LOOM_ENCODING_SPARSITY_POLICY_NONE));

  int64_t sparsity_group_elements = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_operand_verify_optional_static_i64(
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY_GROUP_ELEMENTS],
      op, emitter, IREE_SV("sparsity_group_elements"),
      /*default_value=*/0, &sparsity_group_elements, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (sparsity_group_elements > UINT16_MAX) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("sparsity_group_elements"),
        sparsity_group_elements, IREE_SV("non-negative and <= 65535"));
  }

  int64_t sparsity_group_nonzero_elements = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_operand_verify_optional_static_i64(
      static_params
          [LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY_GROUP_NONZERO_ELEMENTS],
      op, emitter, IREE_SV("sparsity_group_nonzero_elements"),
      /*default_value=*/0, &sparsity_group_nonzero_elements, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (sparsity_group_nonzero_elements > UINT16_MAX) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("sparsity_group_nonzero_elements"),
        sparsity_group_nonzero_elements, IREE_SV("non-negative and <= 65535"));
  }

  const bool zero_scale_fallback = loom_encoding_static_bool(
      static_params[LOOM_ENCODING_OPERAND_PARAMETER_ZERO_SCALE_FALLBACK],
      /*default_value=*/false);

  loom_value_fact_encoded_operand_schema_t encoded_operand = {
      .scale_format = (uint64_t)scale_format,
      .secondary_scale_format = (uint64_t)secondary_scale_format,
      .scale_topology = (uint32_t)scale_topology,
      .scale_group =
          {
              .element_count = (uint16_t)scale_group_elements,
          },
      .scale_operand_count = (uint16_t)scale_operands,
      .sparsity_policy = (uint32_t)sparsity,
      .sparsity_group =
          {
              .nonzero_element_count =
                  (uint16_t)sparsity_group_nonzero_elements,
              .element_count = (uint16_t)sparsity_group_elements,
          },
      .flags = zero_scale_fallback
                   ? LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK
                   : 0,
  };
  for (uint8_t i = 0; i < LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK; ++i) {
    encoded_operand.scale_group.shape[i] = scale_group_shape[i];
  }
  if (!loom_value_fact_encoded_operand_schema_scale_is_complete(
          encoded_operand)) {
    return loom_encoding_operand_verify_i64_range(
        emitter, op, IREE_SV("scale"), 1,
        IREE_SV("all-zero for none or topology/group/operand count for "
                "scaled"));
  }
  if (!loom_value_fact_encoded_operand_schema_sparsity_is_complete(
          encoded_operand)) {
    const bool is_structured = encoded_operand.sparsity_policy ==
                               LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED;
    const bool has_group_elements = sparsity_group_elements != 0;
    return loom_encoding_operand_verify_i64_range(
        emitter, op,
        is_structured
            ? IREE_SV("sparsity_group_nonzero_elements")
            : (has_group_elements ? IREE_SV("sparsity_group_elements")
                                  : IREE_SV("sparsity_group_nonzero_elements")),
        is_structured ? sparsity_group_nonzero_elements
                      : (has_group_elements ? sparsity_group_elements
                                            : sparsity_group_nonzero_elements),
        is_structured
            ? IREE_SV("positive and less than sparsity_group_elements")
            : IREE_SV("zero unless sparsity is n_m_structured"));
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_static_i64(
    const loom_named_attr_t* parameter, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    int64_t* out_value, bool* out_ok) {
  *out_value = 0;
  *out_ok = false;
  if (!parameter) {
    return loom_encoding_emit_param_error(emitter, op,
                                          loom_encoding_turboquant_kv_name(),
                                          LOOM_ERR_ENCODING_007, param_name);
  }
  *out_value = loom_attr_as_i64(parameter->value);
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_static_string(
    const loom_module_t* module, const loom_named_attr_t* parameter,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t param_name, iree_string_view_t* out_value,
    bool* out_ok) {
  *out_value = iree_string_view_empty();
  *out_ok = false;
  if (!parameter) {
    return loom_encoding_emit_param_error(emitter, op,
                                          loom_encoding_turboquant_kv_name(),
                                          LOOM_ERR_ENCODING_007, param_name);
  }
  *out_value =
      module->strings.entries[loom_attr_as_string_id(parameter->value)];
  *out_ok = true;
  return iree_ok_status();
}

static bool loom_encoding_turboquant_transform_family_supported(
    iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("hadamard")) ||
         iree_string_view_equal(value, IREE_SV("hadamard_sign")) ||
         iree_string_view_equal(value, IREE_SV("sign_permute_hadamard"));
}

static bool loom_encoding_numeric_transform_family_supported(
    iree_string_view_t value) {
  return loom_encoding_numeric_transform_family_from_name(value) !=
         LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN;
}

static iree_string_view_t loom_encoding_numeric_transform_family_quoted(
    iree_string_view_t value) {
  if (iree_string_view_equal(value, IREE_SV("hadamard"))) {
    return IREE_SV("'hadamard'");
  }
  if (iree_string_view_equal(value, IREE_SV("hadamard_sign"))) {
    return IREE_SV("'hadamard_sign'");
  }
  if (iree_string_view_equal(value, IREE_SV("jl_dense"))) {
    return IREE_SV("'jl_dense'");
  }
  if (iree_string_view_equal(value, IREE_SV("sign_permute_hadamard"))) {
    return IREE_SV("'sign_permute_hadamard'");
  }
  return value;
}

static bool loom_encoding_numeric_transform_normalization_supported(
    iree_string_view_t value) {
  loom_encoding_numeric_transform_normalization_t normalization =
      LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_NONE;
  return loom_encoding_numeric_transform_normalization_from_name(
      value, &normalization);
}

static iree_status_t loom_encoding_numeric_transform_verify_static_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_named_attr_t* const
        static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_],
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (uint8_t parameter_index = 0;
       parameter_index < LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_;
       ++parameter_index) {
    const loom_named_attr_t* entry = static_params[parameter_index];
    if (!entry) continue;
    const iree_string_view_t param_name = loom_attr_descriptor_name(
        &loom_encoding_numeric_transform_family_descriptor
             .parameter_descriptors[parameter_index]);
    if (parameter_index ==
            LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_INPUT_ELEMS ||
        parameter_index ==
            LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_OUTPUT_ELEMS) {
      int64_t extent = loom_attr_as_i64(entry->value);
      if (extent <= 0) {
        return loom_encoding_emit_attribute_constraint_error(
            emitter, op, param_name, extent, IREE_SV("positive extent"));
      }
      continue;
    }

    const iree_string_view_t string_value =
        module->strings.entries[loom_attr_as_string_id(entry->value)];
    if (parameter_index == LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_FAMILY) {
      if (loom_encoding_numeric_transform_family_supported(string_value)) {
        continue;
      }
      return loom_encoding_emit_static_value_error(
          emitter, op, loom_encoding_numeric_transform_name(), param_name,
          string_value,
          IREE_SV("'hadamard', 'hadamard_sign', 'jl_dense', or "
                  "'sign_permute_hadamard'"));
    }

    IREE_ASSERT(parameter_index ==
                LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_NORMALIZATION);
    if (loom_encoding_numeric_transform_normalization_supported(string_value)) {
      continue;
    }
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_numeric_transform_name(), param_name,
        string_value, IREE_SV("'none' or 'orthonormal'"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_verify_dynamic_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  const loom_value_id_t signs = loom_encoding_define_dynamic_parameter(
      params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SIGNS);
  if (signs != LOOM_VALUE_ID_INVALID) {
    const loom_type_t actual_type = loom_module_value_type(module, signs);
    if (loom_type_element_type(actual_type) != LOOM_SCALAR_TYPE_I1) {
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          loom_encoding_define_dynamic_parameter_name(
              params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SIGNS),
          signs, IREE_SV("i1 vector"));
    }
  }

  const loom_value_id_t permutation = loom_encoding_define_dynamic_parameter(
      params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_PERMUTATION);
  if (permutation != LOOM_VALUE_ID_INVALID) {
    const loom_type_t actual_type = loom_module_value_type(module, permutation);
    if (!loom_type_satisfies_constraint(
            actual_type,
            LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT)) {
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          loom_encoding_define_dynamic_parameter_name(
              params,
              LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_PERMUTATION),
          permutation, IREE_SV("vector with index or non-i1 integer elements"));
    }
  }

  const loom_value_id_t matrix = loom_encoding_define_dynamic_parameter(
      params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_MATRIX);
  if (matrix != LOOM_VALUE_ID_INVALID) {
    const loom_type_t actual_type = loom_module_value_type(module, matrix);
    if (!loom_scalar_type_is_float(loom_type_element_type(actual_type))) {
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          loom_encoding_define_dynamic_parameter_name(
              params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_MATRIX),
          matrix, IREE_SV("floating-point vector"));
    }
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_require_static_param(
    const loom_named_attr_t* static_param, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool* out_ok) {
  *out_ok = false;
  if (static_param) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(emitter, op,
                                        loom_encoding_numeric_transform_name(),
                                        LOOM_ERR_ENCODING_007, param_name);
}

static iree_status_t
loom_encoding_numeric_transform_require_static_or_dynamic_param(
    const loom_op_t* op, const loom_encoding_define_resolved_params_t* params,
    const loom_named_attr_t* static_param, iree_diagnostic_emitter_t emitter,
    uint8_t dynamic_parameter_index, bool* out_ok) {
  *out_ok = false;
  if (static_param || loom_encoding_define_has_dynamic_parameter(
                          params, dynamic_parameter_index)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      LOOM_ERR_ENCODING_007,
      loom_encoding_define_dynamic_parameter_name(params,
                                                  dynamic_parameter_index));
}

static iree_status_t loom_encoding_numeric_transform_require_dynamic_param(
    const loom_op_t* op, const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, uint8_t dynamic_parameter_index,
    bool* out_ok) {
  *out_ok = false;
  if (loom_encoding_define_has_dynamic_parameter(params,
                                                 dynamic_parameter_index)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      LOOM_ERR_ENCODING_007,
      loom_encoding_define_dynamic_parameter_name(params,
                                                  dynamic_parameter_index));
}

static iree_status_t loom_encoding_numeric_transform_reject_dynamic_param(
    const loom_op_t* op, const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, uint8_t dynamic_parameter_index,
    bool* out_ok) {
  *out_ok = false;
  if (!loom_encoding_define_has_dynamic_parameter(params,
                                                  dynamic_parameter_index)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      LOOM_ERR_ENCODING_008,
      loom_encoding_define_dynamic_parameter_name(params,
                                                  dynamic_parameter_index));
}

static iree_status_t loom_encoding_numeric_transform_verify_no_dynamic_payload(
    const loom_op_t* op, const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      op, params, emitter,
      LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SIGNS, out_ok));
  if (!*out_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      op, params, emitter,
      LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_PERMUTATION, out_ok));
  if (!*out_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      op, params, emitter,
      LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_MATRIX, out_ok));
  if (!*out_ok) return iree_ok_status();
  return loom_encoding_numeric_transform_reject_dynamic_param(
      op, params, emitter,
      LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SEED, out_ok);
}

static iree_status_t loom_encoding_numeric_transform_verify_no_matrix(
    const loom_op_t* op, const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  return loom_encoding_numeric_transform_reject_dynamic_param(
      op, params, emitter,
      LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_MATRIX, out_ok);
}

static iree_status_t
loom_encoding_numeric_transform_verify_hadamard_sign_payload(
    const loom_op_t* op, const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  const bool has_signs = loom_encoding_define_has_dynamic_parameter(
      params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SIGNS);
  const bool has_seed = loom_encoding_define_has_dynamic_parameter(
      params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SEED);
  if (!has_signs && !has_seed) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_numeric_transform_name(),
        LOOM_ERR_ENCODING_007, IREE_SV("signs or seed"));
  }
  if (has_signs && has_seed) {
    return loom_encoding_emit_mutually_exclusive_param_error(
        emitter, op, loom_encoding_numeric_transform_name(), IREE_SV("signs"),
        IREE_SV("seed"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_verify_jl_normalization(
    const loom_module_t* module, const loom_op_t* op,
    const loom_named_attr_t* normalization, iree_diagnostic_emitter_t emitter,
    bool* out_ok) {
  *out_ok = false;
  if (!normalization) {
    *out_ok = true;
    return iree_ok_status();
  }

  const iree_string_view_t value =
      module->strings.entries[loom_attr_as_string_id(normalization->value)];
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_static_value_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      IREE_SV("normalization"), value, IREE_SV("'none'"));
}

static iree_status_t loom_encoding_numeric_transform_verify_family_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_resolved_params_t* params,
    const loom_named_attr_t* family, const loom_named_attr_t* normalization,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  if (!family) return iree_ok_status();

  const iree_string_view_t family_name =
      module->strings.entries[loom_attr_as_string_id(family->value)];

  switch (loom_encoding_numeric_transform_family_from_name(family_name)) {
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD:
      return loom_encoding_numeric_transform_verify_no_dynamic_payload(
          op, params, emitter, out_ok);
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_verify_hadamard_sign_payload(
              op, params, emitter, out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          op, params, emitter,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_PERMUTATION,
          out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_no_matrix(op, params,
                                                              emitter, out_ok);
    }
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              op, params, emitter,
              LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SIGNS, out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              op, params, emitter,
              LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_PERMUTATION,
              out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          op, params, emitter,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SEED, out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_no_matrix(op, params,
                                                              emitter, out_ok);
    }
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              op, params, emitter,
              LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_MATRIX,
              out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          op, params, emitter,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SIGNS, out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          op, params, emitter,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_PERMUTATION,
          out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          op, params, emitter,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SEED, out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_jl_normalization(
          module, op, normalization, emitter, out_ok);
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_encoding_numeric_transform_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter) {
  const loom_named_attr_t*
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      params->spec, LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_,
      static_params);

  bool param_ok = false;
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_static_params(
      module, op, static_params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_dynamic_params(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_require_static_param(
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_FAMILY], op,
      emitter, IREE_SV("family"), &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_encoding_numeric_transform_require_static_or_dynamic_param(
          op, params,
          static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_INPUT_ELEMS],
          emitter,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_INPUT_ELEMS,
          &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_encoding_numeric_transform_require_static_or_dynamic_param(
          op, params,
          static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_OUTPUT_ELEMS],
          emitter,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_OUTPUT_ELEMS,
          &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_family_params(
      module, op, params,
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_FAMILY],
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_NORMALIZATION],
      emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  return iree_ok_status();
}

static bool loom_encoding_turboquant_ceil_bits_to_bytes(
    int64_t bit_count, int64_t* out_byte_count) {
  *out_byte_count = 0;
  int64_t padded_bits = 0;
  if (!iree_checked_add_i64(bit_count, 7, &padded_bits)) return false;
  *out_byte_count = padded_bits / 8;
  return true;
}

static bool loom_encoding_turboquant_min_record_bytes(
    int64_t logical_elems, int64_t first_stage_bits, int64_t qjl_rows,
    int64_t residual_bits, int64_t* out_record_bytes) {
  *out_record_bytes = 0;
  int64_t code_bits = 0;
  int64_t code_bytes = 0;
  int64_t residual_total_bits = 0;
  int64_t residual_bytes = 0;
  if (!iree_checked_mul_i64(logical_elems, first_stage_bits, &code_bits) ||
      !iree_checked_mul_i64(qjl_rows, residual_bits, &residual_total_bits)) {
    return false;
  }
  if (!loom_encoding_turboquant_ceil_bits_to_bytes(code_bits, &code_bytes) ||
      !loom_encoding_turboquant_ceil_bits_to_bytes(residual_total_bits,
                                                   &residual_bytes)) {
    return false;
  }
  if (!iree_checked_add_i64(code_bytes, residual_bytes, out_record_bytes)) {
    return false;
  }
  return true;
}

static iree_status_t loom_encoding_turboquant_verify_dynamic_view(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, uint8_t dynamic_parameter_index,
    int64_t expected_static_length, bool* out_ok) {
  *out_ok = false;
  const iree_string_view_t param_name =
      loom_encoding_define_dynamic_parameter_name(params,
                                                  dynamic_parameter_index);
  const loom_value_id_t value_id =
      loom_encoding_define_dynamic_parameter(params, dynamic_parameter_index);
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return loom_encoding_emit_param_error(emitter, op,
                                          loom_encoding_turboquant_kv_name(),
                                          LOOM_ERR_ENCODING_007, param_name);
  }

  const loom_type_t actual_type = loom_module_value_type(module, value_id);
  if (loom_type_rank(actual_type) != 1 ||
      !loom_scalar_type_is_float(loom_type_element_type(actual_type))) {
    return loom_encoding_emit_dynamic_type_error(
        module, emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        value_id, IREE_SV("rank-1 floating-point view"));
  }

  if (!loom_type_dim_is_dynamic_at(actual_type, 0)) {
    int64_t actual_length = loom_type_dim_static_size_at(actual_type, 0);
    if (actual_length != expected_static_length) {
      return loom_encoding_emit_attribute_constraint_error(
          emitter, op, param_name, actual_length,
          IREE_SV("static view length matching schema"));
    }
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_dynamic_param(
    const loom_op_t* op, const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, uint8_t dynamic_parameter_index,
    bool* out_ok) {
  *out_ok = false;
  if (!loom_encoding_define_has_dynamic_parameter(params,
                                                  dynamic_parameter_index)) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_turboquant_kv_name(), LOOM_ERR_ENCODING_007,
        loom_encoding_define_dynamic_parameter_name(params,
                                                    dynamic_parameter_index));
  }
  *out_ok = true;
  return iree_ok_status();
}

static bool loom_encoding_try_get_numeric_transform_family(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_view_t* out_family) {
  *out_family = iree_string_view_empty();
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* define_op = loom_value_def_op(value);
  if (!define_op || !loom_encoding_define_isa(define_op)) return false;

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, define_op);
  if (!params.spec || !loom_encoding_static_parameters_are_valid(params.spec) ||
      loom_module_encoding_family_descriptor(module, params.spec) !=
          &loom_encoding_numeric_transform_family_descriptor) {
    return false;
  }
  const loom_named_attr_t*
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      params.spec, LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_,
      static_params);
  const loom_named_attr_t* family_entry =
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_FAMILY];
  if (!family_entry) return false;
  *out_family =
      module->strings.entries[loom_attr_as_string_id(family_entry->value)];
  return true;
}

static iree_status_t loom_encoding_turboquant_verify_transform_family(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter, uint8_t dynamic_parameter_index,
    iree_string_view_t expected_family, bool* out_ok) {
  *out_ok = false;
  const loom_value_id_t value_id =
      loom_encoding_define_dynamic_parameter(params, dynamic_parameter_index);
  if (value_id == LOOM_VALUE_ID_INVALID) {
    *out_ok = true;
    return iree_ok_status();
  }

  iree_string_view_t actual_family = iree_string_view_empty();
  if (!loom_encoding_try_get_numeric_transform_family(module, value_id,
                                                      &actual_family)) {
    *out_ok = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(actual_family, expected_family)) {
    *out_ok = true;
    return iree_ok_status();
  }

  iree_string_view_t expected =
      loom_encoding_numeric_transform_family_quoted(expected_family);
  return loom_encoding_emit_static_value_error(
      emitter, op, loom_encoding_turboquant_kv_name(),
      loom_encoding_define_dynamic_parameter_name(params,
                                                  dynamic_parameter_index),
      actual_family, expected);
}

static iree_status_t loom_encoding_turboquant_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter) {
  const loom_named_attr_t*
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      params->spec, LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_COUNT_,
      static_params);

  bool param_ok = false;

  int64_t first_stage_bits = 0;
  int64_t logical_elems = 0;
  int64_t qjl_rows = 0;
  int64_t record_bytes = 0;
  int64_t residual_bits = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_FIRST_STAGE_BITS], op,
      emitter, IREE_SV("first_stage_bits"), &first_stage_bits, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_LOGICAL_ELEMS], op,
      emitter, IREE_SV("logical_elems"), &logical_elems, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_QJL_ROWS], op,
      emitter, IREE_SV("qjl_rows"), &qjl_rows, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_RECORD_BYTES], op,
      emitter, IREE_SV("record_bytes"), &record_bytes, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_RESIDUAL_BITS], op,
      emitter, IREE_SV("residual_bits"), &residual_bits, &param_ok));
  if (!param_ok) return iree_ok_status();

  if (first_stage_bits < 1 || first_stage_bits > 8) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("first_stage_bits"), first_stage_bits,
        IREE_SV("between 1 and 8"));
  }
  if (logical_elems <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("logical_elems"), logical_elems,
        IREE_SV("positive"));
  }
  if (qjl_rows <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("qjl_rows"), qjl_rows, IREE_SV("positive"));
  }
  if (record_bytes <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("record_bytes"), record_bytes,
        IREE_SV("positive"));
  }
  if (residual_bits != 1) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("residual_bits"), residual_bits,
        IREE_SV("equal to 1"));
  }

  iree_string_view_t logical_element = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module,
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_LOGICAL_ELEMENT], op,
      emitter, IREE_SV("logical_element"), &logical_element, &param_ok));
  if (!param_ok) return iree_ok_status();
  loom_scalar_type_t logical_scalar = LOOM_SCALAR_TYPE_COUNT_;
  if (!loom_scalar_type_parse(logical_element, &logical_scalar) ||
      !loom_scalar_type_is_float(logical_scalar)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("logical_element"), logical_element,
        IREE_SV("floating-point scalar type"));
  }

  iree_string_view_t pack_order = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_PACK_ORDER],
      op, emitter, IREE_SV("pack_order"), &pack_order, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!iree_string_view_equal(pack_order, IREE_SV("lsb0"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(), IREE_SV("pack_order"),
        pack_order, IREE_SV("'lsb0'"));
  }

  iree_string_view_t scalar_quantizer = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module,
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_SCALAR_QUANTIZER], op,
      emitter, IREE_SV("scalar_quantizer"), &scalar_quantizer, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!iree_string_view_equal(scalar_quantizer, IREE_SV("lloyd_max"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("scalar_quantizer"), scalar_quantizer, IREE_SV("'lloyd_max'"));
  }

  iree_string_view_t transform_family = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module,
      static_params[LOOM_ENCODING_TURBOQUANT_KV_PARAMETER_TRANSFORM_FAMILY], op,
      emitter, IREE_SV("transform_family"), &transform_family, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!loom_encoding_turboquant_transform_family_supported(transform_family)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("transform_family"), transform_family,
        IREE_SV("'hadamard', 'hadamard_sign', or 'sign_permute_hadamard'"));
  }

  int64_t minimum_record_bytes = 0;
  if (!loom_encoding_turboquant_min_record_bytes(
          logical_elems, first_stage_bits, qjl_rows, residual_bits,
          &minimum_record_bytes)) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("record_bytes"), record_bytes,
        IREE_SV("representable size for first-stage codes and QJL residual "
                "bits"));
  }
  if (record_bytes < minimum_record_bytes) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("record_bytes"), record_bytes,
        IREE_SV("large enough for first-stage codes and QJL residual bits"));
  }

  int64_t centroid_count = 1LL << first_stage_bits;
  int64_t threshold_count = centroid_count - 1;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_view(
      module, op, params, emitter,
      LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_CENTROIDS, centroid_count,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_view(
      module, op, params, emitter,
      LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_THRESHOLDS, threshold_count,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_dynamic_param(
      op, params, emitter,
      LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_QJL_TRANSFORM, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_transform_family(
      module, op, params, emitter,
      LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_QJL_TRANSFORM,
      IREE_SV("jl_dense"), &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_dynamic_param(
      op, params, emitter,
      LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_TRANSFORM, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_transform_family(
      module, op, params, emitter,
      LOOM_ENCODING_TURBOQUANT_KV_DYNAMIC_PARAMETER_TRANSFORM, transform_family,
      &param_ok));
  if (!param_ok) return iree_ok_status();

  return iree_ok_status();
}

static const loom_encoding_vtable_t loom_encoding_layout_dense_vtable = {
    .descriptor = &loom_encoding_layout_dense_family_descriptor,
    .summarize = loom_encoding_layout_dense_summarize,
};

static const loom_encoding_vtable_t loom_encoding_layout_strided_vtable = {
    .descriptor = &loom_encoding_layout_strided_family_descriptor,
    .is_static_valid = loom_encoding_layout_strided_is_static_valid,
    .diagnose_static = loom_encoding_layout_strided_diagnose_static,
    .summarize = loom_encoding_layout_strided_summarize,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q4_0_vtable = {
    .descriptor = &loom_encoding_ggml_q4_0_family_descriptor,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q8_0_vtable = {
    .descriptor = &loom_encoding_ggml_q8_0_family_descriptor,
    .is_static_valid = loom_encoding_ggml_q8_0_is_static_valid,
    .diagnose_static = loom_encoding_ggml_q8_0_diagnose_static,
    .summarize = loom_encoding_ggml_q8_0_summarize,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q6_k_vtable = {
    .descriptor = &loom_encoding_ggml_q6_k_family_descriptor,
};

static const loom_encoding_vtable_t loom_encoding_operand_vtable = {
    .descriptor = &loom_encoding_operand_family_descriptor,
    .is_static_valid = loom_encoding_operand_is_static_valid,
    .diagnose_static = loom_encoding_operand_diagnose_static,
    .summarize = loom_encoding_operand_summarize,
};

static const loom_encoding_vtable_t loom_encoding_numeric_transform_vtable = {
    .descriptor = &loom_encoding_numeric_transform_family_descriptor,
    .verify_define = loom_encoding_numeric_transform_verify_define,
};

static const loom_encoding_vtable_t loom_encoding_turboquant_kv_vtable = {
    .descriptor = &loom_encoding_turboquant_kv_family_descriptor,
    .verify_define = loom_encoding_turboquant_verify_define,
};

static const loom_encoding_vtable_t* const loom_encoding_builtin_vtables[] = {
    &loom_encoding_layout_dense_vtable,
    &loom_encoding_layout_strided_vtable,
    &loom_encoding_ggml_q4_0_vtable,
    &loom_encoding_ggml_q8_0_vtable,
    &loom_encoding_ggml_q6_k_vtable,
    &loom_encoding_operand_vtable,
    &loom_encoding_numeric_transform_vtable,
    &loom_encoding_turboquant_kv_vtable,
};

iree_status_t loom_context_register_builtin_encoding_vtables(
    loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_encoding_register_storage_family(context));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_encoding_builtin_vtables); ++i) {
    IREE_RETURN_IF_ERROR(loom_context_register_encoding_vtable(
        context, loom_encoding_builtin_vtables[i]));
  }
  return iree_ok_status();
}
