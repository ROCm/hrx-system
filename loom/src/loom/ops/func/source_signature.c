// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func/source_signature.h"

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"

enum { LOOM_FUNC_C_ABI_MAX_NESTING = 64 };

static iree_status_t loom_func_source_signature_emit_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    int64_t actual_value, iree_string_view_t expected_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("source_signature")),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_014,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_func_source_signature_emit_type_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t value_name, loom_type_t actual_type,
    loom_type_t expected_type) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(value_name),
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("source_signature projection")),
      loom_param_type(expected_type),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_001,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

bool loom_func_c_abi_signature_cast(
    loom_attribute_t attr, loom_func_c_abi_signature_t* out_signature) {
  IREE_ASSERT_ARGUMENT(out_signature);
  *out_signature = (loom_func_c_abi_signature_t){0};
  if (!loom_func_abi_c_signature_attr_isa(attr)) {
    return false;
  }
  *out_signature = (loom_func_c_abi_signature_t){
      .arguments = loom_func_abi_c_signature_attr_arguments(attr),
      .results = loom_func_abi_c_signature_attr_results(attr),
      .types = loom_func_abi_c_signature_attr_types(attr),
  };
  return true;
}

static bool loom_func_c_abi_type_has_only(loom_attribute_t type, bool bitwidth,
                                          bool format, bool address_space,
                                          bool element_type, bool element_count,
                                          bool element_stride, bool name,
                                          bool fields) {
  return loom_func_abi_c_type_attr_has_bitwidth(type) == bitwidth &&
         loom_func_abi_c_type_attr_has_format(type) == format &&
         loom_func_abi_c_type_attr_has_address_space(type) == address_space &&
         loom_func_abi_c_type_attr_has_element_type(type) == element_type &&
         loom_func_abi_c_type_attr_has_element_count(type) == element_count &&
         loom_func_abi_c_type_attr_has_element_stride(type) == element_stride &&
         loom_func_abi_c_type_attr_has_name(type) == name &&
         loom_func_abi_c_type_attr_has_fields(type) == fields;
}

static loom_scalar_type_t loom_func_c_abi_integer_type(int64_t bitwidth) {
  switch (bitwidth) {
    case 8:
      return LOOM_SCALAR_TYPE_I8;
    case 16:
      return LOOM_SCALAR_TYPE_I16;
    case 32:
      return LOOM_SCALAR_TYPE_I32;
    case 64:
      return LOOM_SCALAR_TYPE_I64;
    default:
      return LOOM_SCALAR_TYPE_NONE;
  }
}

static loom_scalar_type_t loom_func_c_abi_float_type(
    loom_func_abi_c_type_format_t format) {
  switch (format) {
    case LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E4M3FN:
      return LOOM_SCALAR_TYPE_F8E4M3;
    case LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E5M2:
      return LOOM_SCALAR_TYPE_F8E5M2;
    case LOOM_FUNC_ABI_C_TYPE_FORMAT_F16:
      return LOOM_SCALAR_TYPE_F16;
    case LOOM_FUNC_ABI_C_TYPE_FORMAT_BF16:
      return LOOM_SCALAR_TYPE_BF16;
    case LOOM_FUNC_ABI_C_TYPE_FORMAT_F32:
      return LOOM_SCALAR_TYPE_F32;
    case LOOM_FUNC_ABI_C_TYPE_FORMAT_F64:
      return LOOM_SCALAR_TYPE_F64;
    default:
      return LOOM_SCALAR_TYPE_NONE;
  }
}

static uint64_t loom_func_c_abi_scalar_byte_length(loom_scalar_type_t type) {
  switch (type) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      return 1;
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      return 2;
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32:
      return 4;
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F64:
      return 8;
    default:
      return 0;
  }
}

static loom_type_t loom_func_c_abi_logical_type(
    loom_parameterized_attr_array_t types, uint32_t type_index) {
  const loom_attribute_t type = types.values[type_index];
  switch (loom_func_abi_c_type_attr_kind(type)) {
    case LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    case LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER:
    case LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER:
      return loom_type_scalar(loom_func_c_abi_integer_type(
          loom_func_abi_c_type_attr_bitwidth(type)));
    case LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING:
      return loom_type_scalar(
          loom_func_c_abi_float_type(loom_func_abi_c_type_attr_format(type)));
    case LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR: {
      const loom_type_t element = loom_func_c_abi_logical_type(
          types, (uint32_t)loom_func_abi_c_type_attr_element_type(type));
      return loom_type_shaped_1d(
          LOOM_TYPE_VECTOR, loom_type_element_type(element),
          (uint64_t)loom_func_abi_c_type_attr_element_count(type), 0);
    }
    default:
      IREE_CHECK_UNREACHABLE();
  }
}

static iree_status_t loom_func_c_abi_verify_type(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_parameterized_attr_array_t types, uint32_t type_index) {
  const loom_attribute_t type = types.values[type_index];
  const int64_t byte_length = loom_func_abi_c_type_attr_byte_length(type);
  const int64_t byte_alignment = loom_func_abi_c_type_attr_byte_alignment(type);
  const int64_t component_count =
      loom_func_abi_c_type_attr_component_count(type);
  if (byte_length <= 0 || byte_alignment <= 0 ||
      (byte_alignment & (byte_alignment - 1)) != 0 ||
      byte_length % byte_alignment != 0 || component_count < 0 ||
      component_count > UINT16_MAX) {
    return loom_func_source_signature_emit_constraint(
        emitter, op, type_index,
        IREE_SV("a positive object size and power-of-two alignment dividing "
                "that size, with at most 65535 projected components"));
  }

  switch (loom_func_abi_c_type_attr_kind(type)) {
    case LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN:
      if (!loom_func_c_abi_type_has_only(type, false, false, false, false,
                                         false, false, false, false) ||
          byte_length != 1 || component_count != 1) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index, IREE_SV("a one-byte boolean type node"));
      }
      break;
    case LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER:
    case LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER: {
      if (!loom_func_c_abi_type_has_only(type, true, false, false, false, false,
                                         false, false, false)) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("an integer node with only bitwidth and object layout"));
      }
      const int64_t bitwidth = loom_func_abi_c_type_attr_bitwidth(type);
      const loom_scalar_type_t scalar_type =
          loom_func_c_abi_integer_type(bitwidth);
      if (scalar_type == LOOM_SCALAR_TYPE_NONE || bitwidth != byte_length * 8 ||
          component_count != 1) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("an 8-, 16-, 32-, or 64-bit integer matching its object "
                    "footprint"));
      }
      break;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING: {
      if (!loom_func_c_abi_type_has_only(type, false, true, false, false, false,
                                         false, false, false)) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a floating node with only format and object layout"));
      }
      const loom_scalar_type_t scalar_type =
          loom_func_c_abi_float_type(loom_func_abi_c_type_attr_format(type));
      if (scalar_type == LOOM_SCALAR_TYPE_NONE ||
          loom_func_c_abi_scalar_byte_length(scalar_type) !=
              (uint64_t)byte_length ||
          component_count != 1) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a supported floating format matching its object "
                    "footprint"));
      }
      break;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_POINTER:
      if (!loom_func_c_abi_type_has_only(
              type, false, false,
              loom_func_abi_c_type_attr_has_address_space(type), false, false,
              false, false, false) ||
          (byte_length != 4 && byte_length != 8) || component_count != 2) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a four- or eight-byte pointer with at most an address "
                    "space"));
      }
      if (loom_func_abi_c_type_attr_has_address_space(type) &&
          loom_func_abi_c_type_attr_address_space(type) < 0) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a nonnegative source address space"));
      }
      break;
    case LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR: {
      if (!loom_func_c_abi_type_has_only(type, false, false, false, true, true,
                                         false, false, false)) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a vector node with element type and lane count"));
      }
      const int64_t element_type = loom_func_abi_c_type_attr_element_type(type);
      const int64_t element_count =
          loom_func_abi_c_type_attr_element_count(type);
      if (element_type < 0 || element_type >= type_index ||
          element_count <= 0 || element_count > LOOM_DIM_MAX_STATIC_SIZE) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a preceding scalar element type and positive supported "
                    "lane count"));
      }
      const loom_attribute_t element = types.values[element_type];
      const loom_func_abi_c_type_kind_t element_kind =
          loom_func_abi_c_type_attr_kind(element);
      const bool scalar_element =
          element_kind == LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER ||
          element_kind == LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER ||
          element_kind == LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING;
      const loom_scalar_type_t scalar_type =
          scalar_element ? loom_type_element_type(loom_func_c_abi_logical_type(
                               types, (uint32_t)element_type))
                         : LOOM_SCALAR_TYPE_NONE;
      const uint64_t element_byte_length =
          (uint64_t)loom_func_abi_c_type_attr_byte_length(element);
      if (loom_func_abi_c_type_attr_component_count(element) != 1 ||
          scalar_type == LOOM_SCALAR_TYPE_NONE ||
          scalar_type == LOOM_SCALAR_TYPE_I1 ||
          (uint64_t)element_count > UINT64_MAX / element_byte_length ||
          (uint64_t)byte_length <
              (uint64_t)element_count * element_byte_length ||
          component_count != 1) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("non-boolean scalar lanes fitting in the vector object"));
      }
      break;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY: {
      if (!loom_func_c_abi_type_has_only(type, false, false, false, true, true,
                                         true, false, false)) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("an array node with element type, count, and stride"));
      }
      const int64_t element_type = loom_func_abi_c_type_attr_element_type(type);
      const int64_t element_count =
          loom_func_abi_c_type_attr_element_count(type);
      const int64_t element_stride =
          loom_func_abi_c_type_attr_element_stride(type);
      if (element_type < 0 || element_type >= type_index ||
          element_count <= 0 || element_stride <= 0) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a preceding element type and positive count and stride"));
      }
      const loom_attribute_t element = types.values[element_type];
      const uint64_t element_byte_length =
          (uint64_t)loom_func_abi_c_type_attr_byte_length(element);
      const uint64_t element_component_count =
          (uint64_t)loom_func_abi_c_type_attr_component_count(element);
      if ((uint64_t)element_stride != element_byte_length ||
          (uint64_t)element_count > UINT64_MAX / (uint64_t)element_stride ||
          (uint64_t)byte_length !=
              (uint64_t)element_count * (uint64_t)element_stride ||
          (element_component_count != 0 &&
           (uint64_t)element_count > UINT16_MAX / element_component_count) ||
          (uint64_t)component_count !=
              (uint64_t)element_count * element_component_count) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a dense C array layout with a representable logical "
                    "projection"));
      }
      break;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_RECORD: {
      const bool has_name = loom_func_abi_c_type_attr_has_name(type);
      const bool has_fields = loom_func_abi_c_type_attr_has_fields(type);
      if (!loom_func_c_abi_type_has_only(type, false, false, false, false,
                                         false, false, has_name, has_fields)) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a record node containing only a name and fields"));
      }
      const loom_parameterized_attr_array_t fields =
          loom_func_abi_c_type_attr_fields(type);
      uint64_t field_component_total = 0;
      uint64_t previous_end = 0;
      for (iree_host_size_t i = 0; i < fields.count; ++i) {
        const loom_attribute_t field = fields.values[i];
        const int64_t field_type = loom_func_abi_c_field_attr_type(field);
        const int64_t field_offset =
            loom_func_abi_c_field_attr_byte_offset(field);
        if (field_type < 0 || field_type >= type_index || field_offset < 0) {
          return loom_func_source_signature_emit_constraint(
              emitter, op, type_index,
              IREE_SV("record fields with preceding types and nonnegative "
                      "offsets"));
        }
        const loom_attribute_t field_type_attr = types.values[field_type];
        const uint64_t field_byte_length =
            (uint64_t)loom_func_abi_c_type_attr_byte_length(field_type_attr);
        const uint64_t field_component_count =
            (uint64_t)loom_func_abi_c_type_attr_component_count(
                field_type_attr);
        const uint64_t offset = (uint64_t)field_offset;
        if (offset < previous_end || offset > UINT64_MAX - field_byte_length ||
            offset + field_byte_length > (uint64_t)byte_length ||
            field_component_total > UINT16_MAX - field_component_count) {
          return loom_func_source_signature_emit_constraint(
              emitter, op, type_index,
              IREE_SV("ordered non-overlapping fields within the record "
                      "footprint and a representable logical projection"));
        }
        previous_end = offset + field_byte_length;
        field_component_total += field_component_count;
      }
      if (field_component_total !=
          (uint64_t)loom_func_abi_c_type_attr_component_count(type)) {
        return loom_func_source_signature_emit_constraint(
            emitter, op, type_index,
            IREE_SV("a record component count equal to the sum of its fields"));
      }
      break;
    }
    default:
      return loom_func_source_signature_emit_constraint(
          emitter, op, type_index, IREE_SV("a supported C ABI type kind"));
  }
  return iree_ok_status();
}

static iree_status_t loom_func_c_abi_projection_emit(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter,
    const loom_func_c_abi_signature_t* signature, uint32_t type_index,
    uint32_t depth, loom_value_slice_t values, iree_string_view_t value_name,
    uint32_t* inout_value_index) {
  const loom_attribute_t type = signature->types.values[type_index];
  if (loom_func_abi_c_type_attr_component_count(type) == 0) {
    return iree_ok_status();
  }
  if (depth == LOOM_FUNC_C_ABI_MAX_NESTING) {
    return loom_func_source_signature_emit_constraint(
        emitter, op, depth,
        IREE_SV("a C ABI source type nesting depth below 64"));
  }
  switch (loom_func_abi_c_type_attr_kind(type)) {
    case LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN:
    case LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER:
    case LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER:
    case LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING:
    case LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR: {
      const loom_type_t expected_type =
          loom_func_c_abi_logical_type(signature->types, type_index);
      const loom_type_t actual_type =
          loom_module_value_type(module, values.values[(*inout_value_index)++]);
      if (!loom_type_equal(actual_type, expected_type)) {
        return loom_func_source_signature_emit_type_mismatch(
            emitter, op, value_name, actual_type, expected_type);
      }
      return iree_ok_status();
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_POINTER: {
      const loom_type_t expected_types[] = {
          loom_type_buffer(),
          loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      };
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_types); ++i) {
        const loom_type_t actual_type = loom_module_value_type(
            module, values.values[(*inout_value_index)++]);
        if (!loom_type_equal(actual_type, expected_types[i])) {
          return loom_func_source_signature_emit_type_mismatch(
              emitter, op, value_name, actual_type, expected_types[i]);
        }
      }
      return iree_ok_status();
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY: {
      const uint32_t element_type =
          (uint32_t)loom_func_abi_c_type_attr_element_type(type);
      const uint64_t element_count =
          (uint64_t)loom_func_abi_c_type_attr_element_count(type);
      for (uint64_t i = 0; i < element_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_func_c_abi_projection_emit(
            module, op, emitter, signature, element_type, depth + 1, values,
            value_name, inout_value_index));
      }
      return iree_ok_status();
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_RECORD: {
      const loom_parameterized_attr_array_t fields =
          loom_func_abi_c_type_attr_fields(type);
      for (iree_host_size_t i = 0; i < fields.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_func_c_abi_projection_emit(
            module, op, emitter, signature,
            (uint32_t)loom_func_abi_c_field_attr_type(fields.values[i]),
            depth + 1, values, value_name, inout_value_index));
      }
      return iree_ok_status();
    }
    default:
      IREE_CHECK_UNREACHABLE();
  }
}

static iree_status_t loom_func_c_abi_projection_count(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_i64_array_t roots, loom_parameterized_attr_array_t types,
    uint32_t* out_count) {
  uint64_t count = 0;
  for (iree_host_size_t i = 0; i < roots.count; ++i) {
    const int64_t root = roots.values[i];
    if (root < 0 || (uint64_t)root >= types.count) {
      return loom_func_source_signature_emit_constraint(
          emitter, op, root, IREE_SV("a valid root type index"));
    }
    const uint64_t component_count =
        (uint64_t)loom_func_abi_c_type_attr_component_count(types.values[root]);
    if (count > UINT16_MAX - component_count) {
      return loom_func_source_signature_emit_constraint(
          emitter, op, root,
          IREE_SV("at most 65535 flattened signature components"));
    }
    count += component_count;
  }
  *out_count = (uint32_t)count;
  return iree_ok_status();
}

iree_status_t loom_func_source_signature_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_func_like_t function = loom_func_like_cast(module, (loom_op_t*)op);
  const loom_attribute_t source_signature =
      loom_func_like_source_signature(function);
  if (loom_attr_is_absent(source_signature)) {
    return iree_ok_status();
  }

  loom_func_c_abi_signature_t signature = {0};
  if (!loom_func_c_abi_signature_cast(source_signature, &signature)) {
    return iree_ok_status();
  }
  if (op->tied_result_count != 0) {
    return loom_func_source_signature_emit_constraint(
        emitter, op, op->tied_result_count,
        IREE_SV("a C source boundary without tied logical results"));
  }

  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < signature.types.count && iree_status_is_ok(status);
       ++i) {
    status = loom_func_c_abi_verify_type(op, emitter, signature.types, i);
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  const loom_value_slice_t argument_values = {
      .values = (loom_value_id_t*)arguments,
      .count = argument_count,
  };
  const loom_value_slice_t result_values = {
      .values = (loom_value_id_t*)loom_op_results(op),
      .count = op->result_count,
  };
  uint32_t projected_argument_count = 0;
  uint32_t projected_result_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_func_c_abi_projection_count(op, emitter, signature.arguments,
                                              signature.types,
                                              &projected_argument_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_func_c_abi_projection_count(op, emitter, signature.results,
                                              signature.types,
                                              &projected_result_count);
  }
  if (iree_status_is_ok(status) &&
      (projected_argument_count != argument_values.count ||
       projected_result_count != result_values.count)) {
    status = loom_func_source_signature_emit_constraint(
        emitter, op,
        ((int64_t)projected_argument_count << 32) | projected_result_count,
        IREE_SV("argument/result projections matching the callable's flat "
                "value counts"));
  }

  uint32_t value_index = 0;
  for (iree_host_size_t i = 0;
       i < signature.arguments.count && iree_status_is_ok(status); ++i) {
    status = loom_func_c_abi_projection_emit(
        module, op, emitter, &signature,
        (uint32_t)signature.arguments.values[i], /*depth=*/0, argument_values,
        IREE_SV("argument"), &value_index);
  }
  value_index = 0;
  for (iree_host_size_t i = 0;
       i < signature.results.count && iree_status_is_ok(status); ++i) {
    status = loom_func_c_abi_projection_emit(
        module, op, emitter, &signature, (uint32_t)signature.results.values[i],
        /*depth=*/0, result_values, IREE_SV("result"), &value_index);
  }
  return status;
}
