// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/numeric_transform.h"

#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"

iree_string_view_t loom_encoding_numeric_transform_name(void) {
  return loom_bstring_view(
      loom_encoding_numeric_transform_family_descriptor.name);
}

loom_encoding_numeric_transform_family_t
loom_encoding_numeric_transform_family_from_name(iree_string_view_t name) {
  if (iree_string_view_equal(name, IREE_SV("hadamard"))) {
    return LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD;
  }
  if (iree_string_view_equal(name, IREE_SV("hadamard_sign"))) {
    return LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN;
  }
  if (iree_string_view_equal(name, IREE_SV("sign_permute_hadamard"))) {
    return LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD;
  }
  if (iree_string_view_equal(name, IREE_SV("jl_dense"))) {
    return LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE;
  }
  return LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN;
}

bool loom_encoding_numeric_transform_family_is_hadamard_like(
    loom_encoding_numeric_transform_family_t family) {
  return family == LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD ||
         family == LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN ||
         family == LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD;
}

bool loom_encoding_numeric_transform_normalization_from_name(
    iree_string_view_t name,
    loom_encoding_numeric_transform_normalization_t* out_normalization) {
  if (iree_string_view_equal(name, IREE_SV("none"))) {
    *out_normalization = LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_NONE;
    return true;
  }
  if (iree_string_view_equal(name, IREE_SV("orthonormal"))) {
    *out_normalization =
        LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_ORTHONORMAL;
    return true;
  }
  return false;
}

bool loom_encoding_numeric_transform_has_signs(
    const loom_encoding_numeric_transform_descriptor_t* descriptor) {
  return descriptor->signs != LOOM_VALUE_ID_INVALID;
}

bool loom_encoding_numeric_transform_has_permutation(
    const loom_encoding_numeric_transform_descriptor_t* descriptor) {
  return descriptor->permutation != LOOM_VALUE_ID_INVALID;
}

bool loom_encoding_numeric_transform_has_matrix(
    const loom_encoding_numeric_transform_descriptor_t* descriptor) {
  return descriptor->matrix != LOOM_VALUE_ID_INVALID;
}

bool loom_encoding_numeric_transform_has_seed(
    const loom_encoding_numeric_transform_descriptor_t* descriptor) {
  return descriptor->seed != LOOM_VALUE_ID_INVALID;
}

bool loom_encoding_numeric_transform_seed_sign_bit(int64_t seed,
                                                   int64_t input_index,
                                                   bool* out_negate) {
  *out_negate = false;
  if (input_index < 0) return false;
  uint64_t mixed = (uint64_t)seed + (uint64_t)input_index;
  mixed += UINT64_C(0x9E3779B97F4A7C15);
  mixed ^= mixed >> 30;
  mixed *= UINT64_C(0xBF58476D1CE4E5B9);
  mixed ^= mixed >> 27;
  mixed *= UINT64_C(0x94D049BB133111EB);
  mixed ^= mixed >> 31;
  *out_negate = (mixed & 1) != 0;
  return true;
}

static bool loom_encoding_numeric_transform_try_find_definition(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_encoding_define_param_view_t* out_params) {
  if (!module || value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= module->values.count) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;

  const loom_op_t* define_op = loom_value_def_op(value);
  if (!define_op || !loom_encoding_define_isa(define_op)) return false;

  const loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, define_op);
  if (!params.spec ||
      loom_module_encoding_family_descriptor(module, params.spec) !=
          &loom_encoding_numeric_transform_family_descriptor) {
    return false;
  }

  *out_params = params;
  return true;
}

static void loom_encoding_numeric_transform_read_static_descriptor(
    const loom_module_t* module,
    const loom_named_attr_t* const
        static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_],
    loom_encoding_numeric_transform_descriptor_t* out_descriptor) {
  loom_encoding_numeric_transform_descriptor_t descriptor = {
      .family = LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN,
      .normalization = LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_NONE,
      .input_extent =
          {
              .static_value = 0,
              .dynamic_value = LOOM_VALUE_ID_INVALID,
          },
      .output_extent =
          {
              .static_value = 0,
              .dynamic_value = LOOM_VALUE_ID_INVALID,
          },
      .signs = LOOM_VALUE_ID_INVALID,
      .permutation = LOOM_VALUE_ID_INVALID,
      .matrix = LOOM_VALUE_ID_INVALID,
      .seed = LOOM_VALUE_ID_INVALID,
  };

  const loom_named_attr_t* family_param =
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_FAMILY];
  const iree_string_view_t family_name =
      module->strings.entries[loom_attr_as_string_id(family_param->value)];
  descriptor.family =
      loom_encoding_numeric_transform_family_from_name(family_name);

  const loom_named_attr_t* normalization_param =
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_NORMALIZATION];
  if (normalization_param) {
    const iree_string_view_t normalization_name =
        module->strings
            .entries[loom_attr_as_string_id(normalization_param->value)];
    loom_encoding_numeric_transform_normalization_from_name(
        normalization_name, &descriptor.normalization);
  }

  const loom_named_attr_t* input_extent_param =
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_INPUT_ELEMS];
  if (input_extent_param) {
    descriptor.input_extent.static_value =
        loom_attr_as_i64(input_extent_param->value);
  }
  const loom_named_attr_t* output_extent_param =
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_OUTPUT_ELEMS];
  if (output_extent_param) {
    descriptor.output_extent.static_value =
        loom_attr_as_i64(output_extent_param->value);
  }

  *out_descriptor = descriptor;
}

static void loom_encoding_numeric_transform_read_dynamic_descriptor(
    const loom_encoding_define_resolved_params_t* resolved_params,
    loom_encoding_numeric_transform_descriptor_t* out_descriptor) {
  out_descriptor->input_extent.dynamic_value =
      loom_encoding_define_dynamic_parameter(
          resolved_params,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_INPUT_ELEMS);
  out_descriptor->output_extent.dynamic_value =
      loom_encoding_define_dynamic_parameter(
          resolved_params,
          LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_OUTPUT_ELEMS);
  out_descriptor->signs = loom_encoding_define_dynamic_parameter(
      resolved_params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SIGNS);
  out_descriptor->permutation = loom_encoding_define_dynamic_parameter(
      resolved_params,
      LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_PERMUTATION);
  out_descriptor->matrix = loom_encoding_define_dynamic_parameter(
      resolved_params,
      LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_MATRIX);
  out_descriptor->seed = loom_encoding_define_dynamic_parameter(
      resolved_params, LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_SEED);
}

bool loom_encoding_numeric_transform_try_read_unverified_descriptor(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_encoding_numeric_transform_descriptor_t* out_descriptor) {
  loom_encoding_define_param_view_t params;
  if (!loom_encoding_numeric_transform_try_find_definition(module, value_id,
                                                           &params) ||
      !loom_encoding_static_is_valid(params.spec)) {
    return false;
  }
  const loom_named_attr_t*
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      params.spec, LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_,
      static_params);
  if (!static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_FAMILY]) {
    return false;
  }
  loom_encoding_numeric_transform_descriptor_t descriptor;
  loom_encoding_numeric_transform_read_static_descriptor(module, static_params,
                                                         &descriptor);

  loom_encoding_define_dynamic_binding_t dynamic_bindings
      [LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_COUNT_];
  loom_encoding_define_resolved_params_t resolved_params;
  if (!loom_encoding_define_try_resolve_unverified_params(
          module, &loom_encoding_numeric_transform_family_descriptor, &params,
          dynamic_bindings, &resolved_params)) {
    return false;
  }
  loom_encoding_numeric_transform_read_dynamic_descriptor(&resolved_params,
                                                          &descriptor);

  *out_descriptor = descriptor;
  return true;
}

bool loom_encoding_numeric_transform_try_read_verified_descriptor(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_encoding_numeric_transform_descriptor_t* out_descriptor) {
  loom_encoding_define_param_view_t params;
  if (!loom_encoding_numeric_transform_try_find_definition(module, value_id,
                                                           &params)) {
    return false;
  }
  const loom_named_attr_t*
      static_params[LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      params.spec, LOOM_ENCODING_NUMERIC_TRANSFORM_PARAMETER_COUNT_,
      static_params);
  loom_encoding_numeric_transform_descriptor_t descriptor;
  loom_encoding_numeric_transform_read_static_descriptor(module, static_params,
                                                         &descriptor);

  loom_encoding_define_dynamic_binding_t dynamic_bindings
      [LOOM_ENCODING_NUMERIC_TRANSFORM_DYNAMIC_PARAMETER_COUNT_];
  loom_encoding_define_resolved_params_t resolved_params;
  loom_encoding_define_resolve_verified_params(
      module, &loom_encoding_numeric_transform_family_descriptor, &params,
      dynamic_bindings, &resolved_params);
  loom_encoding_numeric_transform_read_dynamic_descriptor(&resolved_params,
                                                          &descriptor);

  *out_descriptor = descriptor;
  return true;
}
