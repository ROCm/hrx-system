// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/storage.h"

#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/encoding/operand.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/summary.h"
#include "loom/util/fact_table.h"
#include "loom/util/numeric_format.h"

static iree_string_view_t loom_encoding_storage_name(void) {
  return IREE_SV("encoding.storage");
}

static iree_string_view_t loom_encoding_operand_name(void) {
  return IREE_SV("encoding.operand");
}

static iree_string_view_t loom_encoding_layout_param_name(void) {
  return IREE_SV("layout");
}

static iree_string_view_t loom_encoding_schema_param_name(void) {
  return IREE_SV("schema");
}

static uint16_t loom_encoding_u16_param(const loom_named_attr_t* parameter) {
  return (uint16_t)loom_attr_as_i64(parameter->value);
}

static loom_value_fact_address_layout_t
loom_encoding_layout_dense_address_layout(void) {
  return (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
      .rank = 0,
      .strides = NULL,
  };
}

void loom_encoding_layout_dense_summarize(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary) {
  (void)request;
  out_summary->encoding.address_layout =
      loom_encoding_layout_dense_address_layout();
}

static bool loom_encoding_static_strided_layout(
    const loom_encoding_t* encoding, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  const loom_named_attr_t*
      params[LOOM_ENCODING_LAYOUT_STRIDED_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_LAYOUT_STRIDED_PARAMETER_COUNT_, params);
  const loom_named_attr_t* strides =
      params[LOOM_ENCODING_LAYOUT_STRIDED_PARAMETER_STRIDES];
  if (strides->value.count > stride_capacity ||
      (strides->value.count > 0 && !stride_storage)) {
    return false;
  }
  for (uint16_t i = 0; i < strides->value.count; ++i) {
    stride_storage[i] = loom_value_facts_exact_i64(strides->value.i64_array[i]);
  }
  *out_layout = (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .rank = (uint8_t)strides->value.count,
      .strides = strides->value.count > 0 ? stride_storage : NULL,
  };
  return true;
}

void loom_encoding_layout_strided_summarize(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary) {
  (void)loom_encoding_static_strided_layout(
      request->params->spec, request->stride_storage, request->stride_capacity,
      &out_summary->encoding.address_layout);
}

static void loom_encoding_static_operand_schema(
    const loom_encoding_t* encoding,
    loom_value_fact_storage_schema_t* out_schema) {
  // Unscaled operands with only format, element count, packing, and optional
  // rounding are the common FP8 and integer-weight form. Their canonical
  // sparse parameters are adjacent, so summarize them directly without paying
  // for the general descriptor switch on every query.
  const loom_named_attr_t* parameters = encoding->attributes;
  const bool is_compact_unscaled_operand =
      (encoding->attribute_count == 3 || encoding->attribute_count == 4) &&
      loom_encoding_parameter_descriptor_index(&parameters[0]) ==
          LOOM_ENCODING_OPERAND_PARAMETER_ELEMENT_FORMAT &&
      loom_encoding_parameter_descriptor_index(&parameters[1]) ==
          LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_ELEMENTS &&
      loom_encoding_parameter_descriptor_index(&parameters[2]) ==
          LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_PACKING &&
      (encoding->attribute_count == 3 ||
       loom_encoding_parameter_descriptor_index(&parameters[3]) ==
           LOOM_ENCODING_OPERAND_PARAMETER_ROUNDING);
  if (is_compact_unscaled_operand) {
    *out_schema = (loom_value_fact_storage_schema_t){
        .encoded_operand =
            {
                .element_format = loom_encoding_numeric_format_fact(
                    (loom_encoding_numeric_format_t)loom_attr_as_enum(
                        parameters[0].value)),
                .payload_packing = loom_encoding_payload_packing_fact(
                    (loom_encoding_payload_packing_t)loom_attr_as_enum(
                        parameters[2].value)),
                .rounding_policy =
                    encoding->attribute_count == 4
                        ? loom_encoding_rounding_policy_fact(
                              (loom_encoding_rounding_policy_t)
                                  loom_attr_as_enum(parameters[3].value))
                        : LOOM_VALUE_FACT_ROUNDING_POLICY_NONE,
                .payload_element_count =
                    loom_encoding_u16_param(&parameters[1]),
            },
    };
    return;
  }

  loom_value_fact_encoded_operand_schema_t encoded_operand = {
      .payload_packing = LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT,
  };
  bool has_scale_group_shape = false;
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* parameter = &encoding->attributes[i];
    switch ((loom_encoding_operand_parameter_t)
                loom_encoding_parameter_descriptor_index(parameter)) {
      case LOOM_ENCODING_OPERAND_PARAMETER_AFFINE:
        encoded_operand.affine_policy = loom_encoding_affine_policy_fact(
            (loom_encoding_affine_policy_t)loom_attr_as_enum(parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_CODEBOOK:
        encoded_operand.codebook_policy = loom_encoding_codebook_policy_fact(
            (loom_encoding_codebook_policy_t)loom_attr_as_enum(
                parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_ELEMENT_FORMAT:
        encoded_operand.element_format = loom_encoding_numeric_format_fact((
            loom_encoding_numeric_format_t)loom_attr_as_enum(parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_ELEMENTS:
        encoded_operand.payload_element_count =
            loom_encoding_u16_param(parameter);
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_PACKING:
        encoded_operand.payload_packing = loom_encoding_payload_packing_fact(
            (loom_encoding_payload_packing_t)loom_attr_as_enum(
                parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_REGISTERS:
        encoded_operand.payload_register_count =
            loom_encoding_u16_param(parameter);
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_ROUNDING:
        encoded_operand.rounding_policy = loom_encoding_rounding_policy_fact(
            (loom_encoding_rounding_policy_t)loom_attr_as_enum(
                parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SCALE_FORMAT:
        encoded_operand.scale_format = loom_encoding_numeric_format_fact((
            loom_encoding_numeric_format_t)loom_attr_as_enum(parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SCALE_GROUP_ELEMENTS:
        encoded_operand.scale_group.element_count =
            loom_encoding_u16_param(parameter);
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SCALE_GROUP_SHAPE:
        has_scale_group_shape = true;
        encoded_operand.scale_group.element_count = 1;
        for (uint16_t dimension = 0; dimension < parameter->value.count;
             ++dimension) {
          const uint16_t extent =
              (uint16_t)parameter->value.i64_array[dimension];
          encoded_operand.scale_group.shape[dimension] = extent;
          encoded_operand.scale_group.element_count *= extent;
        }
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SCALE_OPERANDS:
        encoded_operand.scale_operand_count =
            loom_encoding_u16_param(parameter);
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SCALE_TOPOLOGY:
        encoded_operand.scale_topology = loom_encoding_scale_topology_fact((
            loom_encoding_scale_topology_t)loom_attr_as_enum(parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SECONDARY_SCALE_FORMAT:
        encoded_operand.secondary_scale_format =
            loom_encoding_numeric_format_fact(
                (loom_encoding_numeric_format_t)loom_attr_as_enum(
                    parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY:
        encoded_operand.sparsity_policy = loom_encoding_sparsity_policy_fact(
            (loom_encoding_sparsity_policy_t)loom_attr_as_enum(
                parameter->value));
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY_GROUP_ELEMENTS:
        encoded_operand.sparsity_group.element_count =
            loom_encoding_u16_param(parameter);
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY_GROUP_NONZERO_ELEMENTS:
        encoded_operand.sparsity_group.nonzero_element_count =
            loom_encoding_u16_param(parameter);
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_ZERO_SCALE_FALLBACK:
        if (loom_attr_as_bool(parameter->value)) {
          encoded_operand.flags |=
              LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK;
        }
        break;
      case LOOM_ENCODING_OPERAND_PARAMETER_COUNT_:
        IREE_ASSERT_UNREACHABLE("invalid encoded operand parameter ordinal");
        IREE_BUILTIN_UNREACHABLE();
    }
  }

  const bool has_1d_scale_topology =
      iree_any_bit_set(encoded_operand.scale_topology,
                       LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D |
                           LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D);
  if (!has_scale_group_shape && has_1d_scale_topology) {
    encoded_operand.scale_group.shape[0] =
        encoded_operand.scale_group.element_count;
  }
  out_schema->encoded_operand = encoded_operand;
}

void loom_encoding_operand_summarize(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary) {
  loom_encoding_static_operand_schema(request->params->spec,
                                      &out_summary->encoding.storage_schema);
}

bool loom_encoding_query_static_address_layout(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!module || !out_layout) return false;
  *out_layout = (loom_value_fact_address_layout_t){0};
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (!encoding || !loom_encoding_static_is_valid(encoding)) {
    return false;
  }
  loom_encoding_family_summary_t summary;
  loom_encoding_summarize_verified_static(module, encoding_id, stride_storage,
                                          stride_capacity, &summary);
  *out_layout = summary.encoding.address_layout;
  return out_layout->kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN;
}

bool loom_encoding_query_static_storage_schema(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_fact_storage_schema_t* out_schema) {
  if (!module || !out_schema) return false;
  *out_schema = (loom_value_fact_storage_schema_t){0};
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (!encoding || !loom_encoding_static_is_valid(encoding)) {
    return false;
  }
  // The canonical operand family can fill the requested schema directly
  // without materializing and copying a composite family summary.
  if (loom_module_encoding_family_descriptor(module, encoding) ==
      &loom_encoding_operand_family_descriptor) {
    loom_encoding_static_operand_schema(encoding, out_schema);
    out_schema->static_spec_encoding_id = encoding_id;
    return true;
  }
  loom_encoding_family_summary_t summary;
  loom_encoding_summarize_verified_static(module, encoding_id,
                                          /*stride_storage=*/NULL,
                                          /*stride_capacity=*/0, &summary);
  *out_schema = summary.encoding.storage_schema;
  return out_schema->static_spec_encoding_id != 0 ||
         !loom_value_fact_encoded_operand_schema_is_unknown(
             out_schema->encoded_operand);
}

bool loom_encoding_query_static_record_geometry(
    const loom_module_t* module, uint16_t encoding_id,
    loom_encoding_record_geometry_t* out_geometry) {
  if (!module || !out_geometry) return false;
  *out_geometry = (loom_encoding_record_geometry_t){0};
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (!encoding || !loom_encoding_static_is_valid(encoding)) return false;
  const loom_encoding_family_descriptor_t* descriptor =
      loom_module_encoding_family_descriptor(module, encoding);
  if (!descriptor || !descriptor->fixed_metadata ||
      descriptor->fixed_metadata->record.logical_element_count == 0) {
    return false;
  }
  *out_geometry = descriptor->fixed_metadata->record;
  return true;
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
  if (!loom_type_has_encoding(type)) {
    if (!loom_type_can_have_encoding(type)) return false;
    out_layout->kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE;
    return true;
  }
  if (!module) return false;

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

static bool loom_encoding_query_address_layout_operands_from_value(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_encoding_address_layout_operands_t* out_operands) {
  while (true) {
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) return false;
    const loom_op_t* op = loom_value_def_op(value);

    if (loom_encoding_layout_strided_isa(op)) {
      const loom_value_slice_t dynamic_strides =
          loom_encoding_layout_strided_strides(op);
      *out_operands = (loom_encoding_address_layout_operands_t){
          .static_strides = loom_encoding_layout_strided_static_strides(op),
          .dynamic_stride_values = dynamic_strides.values,
          .dynamic_stride_count = dynamic_strides.count,
      };
      return true;
    }
    if (loom_encoding_layout_assume_strided_isa(op)) {
      value_id = loom_encoding_layout_assume_strided_layout(op);
      continue;
    }
    if (loom_encoding_assume_spec_isa(op)) {
      value_id = loom_encoding_assume_spec_enc(op);
      continue;
    }
    if (loom_encoding_assume_match_isa(op)) {
      value_id = loom_encoding_assume_match_enc(op);
      continue;
    }
    if (!loom_encoding_define_isa(op)) return false;

    const loom_encoding_t* spec =
        loom_module_encoding(module, loom_encoding_define_spec(op));
    if (!spec || loom_module_encoding_family_descriptor(module, spec) !=
                     &loom_encoding_storage_family_descriptor) {
      return false;
    }
    const loom_encoding_define_param_view_t param_view =
        loom_encoding_define_param_view(module, op);
    loom_encoding_define_dynamic_binding_t
        dynamic_bindings[LOOM_ENCODING_STORAGE_DYNAMIC_PARAMETER_COUNT_];
    loom_encoding_define_resolved_params_t params;
    loom_encoding_define_resolve_verified_params(
        module, &loom_encoding_storage_family_descriptor, &param_view,
        dynamic_bindings, &params);
    value_id = loom_encoding_define_dynamic_parameter(
        &params, LOOM_ENCODING_STORAGE_DYNAMIC_PARAMETER_LAYOUT);
    if (value_id == LOOM_VALUE_ID_INVALID) return false;
  }
}

bool loom_encoding_query_type_address_layout_operands(
    const loom_module_t* module, loom_type_t type,
    loom_encoding_address_layout_operands_t* out_operands) {
  if (!out_operands) return false;
  *out_operands = (loom_encoding_address_layout_operands_t){0};
  if (!module || !loom_type_has_ssa_encoding(type)) return false;
  return loom_encoding_query_address_layout_operands_from_value(
      module, loom_type_encoding_value_id(type), out_operands);
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
      loom_param_string(loom_encoding_storage_name()),
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_role_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, iree_string_view_t expected_role) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_storage_name()),
      loom_param_string(param_name),
      loom_param_string(expected_role),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_011, params,
                            IREE_ARRAYSIZE(params));
}

typedef struct loom_encoding_storage_static_params_t {
  // Optional statically bound address layout parameter.
  const loom_named_attr_t* layout;
  // Optional statically bound storage schema parameter.
  const loom_named_attr_t* schema;
} loom_encoding_storage_static_params_t;

static loom_encoding_storage_static_params_t
loom_encoding_storage_static_params(const loom_encoding_t* encoding) {
  const loom_named_attr_t* slots[LOOM_ENCODING_STORAGE_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_STORAGE_PARAMETER_COUNT_, slots);
  return (loom_encoding_storage_static_params_t){
      .layout = slots[LOOM_ENCODING_STORAGE_PARAMETER_LAYOUT],
      .schema = slots[LOOM_ENCODING_STORAGE_PARAMETER_SCHEMA],
  };
}

enum loom_encoding_storage_violation_bits_e {
  LOOM_ENCODING_STORAGE_LAYOUT_ROLE_VIOLATION = 1u << 0,
  LOOM_ENCODING_STORAGE_SCHEMA_ROLE_VIOLATION = 1u << 1,
};
typedef uint8_t loom_encoding_storage_violations_t;

static loom_encoding_storage_violations_t
loom_encoding_storage_static_violations(const loom_module_t* module,
                                        const loom_encoding_t* encoding) {
  const loom_encoding_storage_static_params_t params =
      loom_encoding_storage_static_params(encoding);
  loom_encoding_storage_violations_t violations = 0;
  if (params.layout) {
    const loom_encoding_t* nested = loom_module_encoding(
        module, loom_attr_as_encoding_id(params.layout->value));
    if (loom_encoding_static_role(module, nested) !=
        LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
      violations |= LOOM_ENCODING_STORAGE_LAYOUT_ROLE_VIOLATION;
    }
  }
  if (params.schema) {
    const loom_encoding_t* nested = loom_module_encoding(
        module, loom_attr_as_encoding_id(params.schema->value));
    if (loom_encoding_static_role(module, nested) !=
        LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
      violations |= LOOM_ENCODING_STORAGE_SCHEMA_ROLE_VIOLATION;
    }
  }
  return violations;
}

static bool loom_encoding_storage_is_static_valid(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  return loom_encoding_storage_static_violations(module, encoding) == 0;
}

static iree_status_t loom_encoding_storage_diagnose_static(
    const loom_module_t* module, const loom_encoding_t* encoding,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  const loom_encoding_storage_violations_t violations =
      loom_encoding_storage_static_violations(module, encoding);
  if (iree_any_bit_set(violations,
                       LOOM_ENCODING_STORAGE_LAYOUT_ROLE_VIOLATION)) {
    IREE_RETURN_IF_ERROR(loom_encoding_emit_role_error(
        emitter, op, loom_encoding_layout_param_name(),
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }
  if (iree_any_bit_set(violations,
                       LOOM_ENCODING_STORAGE_SCHEMA_ROLE_VIOLATION)) {
    IREE_RETURN_IF_ERROR(loom_encoding_emit_role_error(
        emitter, op, loom_encoding_schema_param_name(),
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_storage_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_resolved_params_t* params,
    iree_diagnostic_emitter_t emitter) {
  const loom_encoding_storage_static_params_t static_params =
      loom_encoding_storage_static_params(params->spec);
  if (!static_params.layout &&
      !loom_encoding_define_has_dynamic_parameter(
          params, LOOM_ENCODING_STORAGE_DYNAMIC_PARAMETER_LAYOUT)) {
    return loom_encoding_emit_param_error(emitter, op, LOOM_ERR_ENCODING_007,
                                          loom_encoding_layout_param_name());
  }
  if (!static_params.schema &&
      !loom_encoding_define_has_dynamic_parameter(
          params, LOOM_ENCODING_STORAGE_DYNAMIC_PARAMETER_SCHEMA)) {
    return loom_encoding_emit_param_error(emitter, op, LOOM_ERR_ENCODING_007,
                                          loom_encoding_schema_param_name());
  }
  return iree_ok_status();
}

void loom_encoding_storage_summarize(
    const loom_encoding_family_summary_request_t* request,
    loom_encoding_family_summary_t* out_summary) {
  const loom_encoding_storage_static_params_t static_params =
      loom_encoding_storage_static_params(request->params->spec);

  if (loom_encoding_define_has_dynamic_parameter(
          request->params, LOOM_ENCODING_STORAGE_DYNAMIC_PARAMETER_LAYOUT)) {
    loom_value_fact_encoding_summary_t dynamic_summary;
    if (loom_encoding_summary_query_dynamic_encoding(
            request, LOOM_ENCODING_STORAGE_DYNAMIC_PARAMETER_LAYOUT,
            &dynamic_summary)) {
      out_summary->encoding.address_layout = dynamic_summary.address_layout;
    }
  } else if (static_params.layout) {
    out_summary->nested_static.address_layout_encoding_id =
        loom_attr_as_encoding_id(static_params.layout->value);
  }

  if (loom_encoding_define_has_dynamic_parameter(
          request->params, LOOM_ENCODING_STORAGE_DYNAMIC_PARAMETER_SCHEMA)) {
    loom_value_fact_encoding_summary_t dynamic_summary;
    if (loom_encoding_summary_query_dynamic_encoding(
            request, LOOM_ENCODING_STORAGE_DYNAMIC_PARAMETER_SCHEMA,
            &dynamic_summary)) {
      out_summary->encoding.storage_schema = dynamic_summary.storage_schema;
    }
  } else if (static_params.schema) {
    out_summary->nested_static.storage_schema_encoding_id =
        loom_attr_as_encoding_id(static_params.schema->value);
  }
}

static const loom_encoding_vtable_t loom_encoding_storage_vtable = {
    .descriptor = &loom_encoding_storage_family_descriptor,
    .is_static_valid = loom_encoding_storage_is_static_valid,
    .diagnose_static = loom_encoding_storage_diagnose_static,
    .verify_define = loom_encoding_storage_verify_define,
    .summarize = loom_encoding_storage_summarize,
};

iree_status_t loom_encoding_register_storage_family(loom_context_t* context) {
  return loom_context_register_encoding_vtable(context,
                                               &loom_encoding_storage_vtable);
}
