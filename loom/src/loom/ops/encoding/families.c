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
#include "loom/ops/encoding/operand.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/encoding/summary.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

static iree_string_view_t loom_encoding_operand_name(void) {
  return IREE_SV("encoding.operand");
}

static iree_string_view_t loom_encoding_layout_strided_name(void) {
  return IREE_SV("encoding.layout.strided");
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
};

static const loom_encoding_vtable_t loom_encoding_ggml_q4_k_vtable = {
    .descriptor = &loom_encoding_ggml_q4_k_family_descriptor,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q6_k_vtable = {
    .descriptor = &loom_encoding_ggml_q6_k_family_descriptor,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q8_1_x4_vtable = {
    .descriptor = &loom_encoding_ggml_q8_1_x4_family_descriptor,
};

static const loom_encoding_vtable_t loom_encoding_operand_vtable = {
    .descriptor = &loom_encoding_operand_family_descriptor,
    .is_static_valid = loom_encoding_operand_is_static_valid,
    .diagnose_static = loom_encoding_operand_diagnose_static,
    .summarize = loom_encoding_operand_summarize,
};

static const loom_encoding_vtable_t loom_encoding_hadamard_vtable = {
    .descriptor = &loom_encoding_transform_hadamard_family_descriptor,
};

static const loom_encoding_vtable_t* const loom_encoding_builtin_vtables[] = {
    &loom_encoding_layout_dense_vtable, &loom_encoding_layout_strided_vtable,
    &loom_encoding_ggml_q4_0_vtable,    &loom_encoding_ggml_q8_0_vtable,
    &loom_encoding_ggml_q4_k_vtable,    &loom_encoding_ggml_q6_k_vtable,
    &loom_encoding_ggml_q8_1_x4_vtable, &loom_encoding_operand_vtable,
    &loom_encoding_hadamard_vtable,
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
