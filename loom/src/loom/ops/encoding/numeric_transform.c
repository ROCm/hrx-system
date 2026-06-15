// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/numeric_transform.h"

#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"

iree_string_view_t loom_encoding_numeric_transform_name(void) {
  return IREE_SV("numeric_transform");
}

static iree_string_view_t loom_encoding_numeric_transform_family_param_name(
    void) {
  return IREE_SV("family");
}

static iree_string_view_t
loom_encoding_numeric_transform_normalization_param_name(void) {
  return IREE_SV("normalization");
}

static iree_string_view_t loom_encoding_numeric_transform_signs_param_name(
    void) {
  return IREE_SV("signs");
}

static iree_string_view_t
loom_encoding_numeric_transform_permutation_param_name(void) {
  return IREE_SV("permutation");
}

static iree_string_view_t loom_encoding_numeric_transform_matrix_param_name(
    void) {
  return IREE_SV("matrix");
}

static iree_string_view_t loom_encoding_numeric_transform_seed_param_name(
    void) {
  return IREE_SV("seed");
}

static bool loom_encoding_numeric_transform_string_id_equal(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static const loom_named_attr_t* loom_encoding_numeric_transform_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_encoding_numeric_transform_string_id_equal(module, entry->name_id,
                                                        name)) {
      return entry;
    }
  }
  return NULL;
}

static bool loom_encoding_numeric_transform_string_attr_value(
    const loom_module_t* module, loom_attribute_t attr,
    iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  if (attr.kind != LOOM_ATTR_STRING ||
      attr.string_id == LOOM_STRING_ID_INVALID ||
      attr.string_id >= module->strings.count) {
    return false;
  }
  *out_value = module->strings.entries[attr.string_id];
  return true;
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

static loom_encoding_numeric_transform_read_t
loom_encoding_numeric_transform_make_read(
    loom_encoding_numeric_transform_read_code_t code,
    iree_string_view_t parameter_name,
    loom_encoding_numeric_transform_descriptor_t descriptor) {
  return (loom_encoding_numeric_transform_read_t){
      .descriptor = descriptor,
      .code = code,
      .parameter_name = parameter_name,
  };
}

static loom_encoding_numeric_transform_read_t
loom_encoding_numeric_transform_read_dynamic_param(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* param, iree_string_view_t param_name,
    loom_value_id_t* out_value,
    loom_encoding_numeric_transform_descriptor_t descriptor) {
  if (!param) {
    *out_value = LOOM_VALUE_ID_INVALID;
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK, iree_string_view_empty(),
        descriptor);
  }
  if (!loom_encoding_define_dynamic_param_value(params, param, out_value)) {
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_MALFORMED_DYNAMIC_PARAM,
        param_name, descriptor);
  }
  return loom_encoding_numeric_transform_make_read(
      LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK, iree_string_view_empty(),
      descriptor);
}

loom_encoding_numeric_transform_read_t
loom_encoding_numeric_transform_read_descriptor(const loom_module_t* module,
                                                loom_value_id_t value_id) {
  loom_encoding_numeric_transform_descriptor_t descriptor = {
      .family = LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN,
      .normalization = LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_NONE,
      .signs = LOOM_VALUE_ID_INVALID,
      .permutation = LOOM_VALUE_ID_INVALID,
      .matrix = LOOM_VALUE_ID_INVALID,
      .seed = LOOM_VALUE_ID_INVALID,
  };

  if (!module || value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= module->values.count) {
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_VALUE_OUT_OF_RANGE,
        iree_string_view_empty(), descriptor);
  }

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_LOCALLY_DEFINED,
        iree_string_view_empty(), descriptor);
  }

  const loom_op_t* define_op = loom_value_def_op(value);
  if (!define_op || !loom_encoding_define_isa(define_op)) {
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_ENCODING_DEFINE,
        iree_string_view_empty(), descriptor);
  }

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, define_op);
  if (!params.spec || !loom_encoding_numeric_transform_string_id_equal(
                          module, params.spec->name_id,
                          loom_encoding_numeric_transform_name())) {
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_NUMERIC_TRANSFORM,
        iree_string_view_empty(), descriptor);
  }

  iree_string_view_t family_param_name =
      loom_encoding_numeric_transform_family_param_name();
  const loom_named_attr_t* family_param =
      loom_encoding_numeric_transform_find_named_attr(
          module, params.static_attrs, family_param_name);
  if (!family_param) {
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_MISSING_FAMILY, family_param_name,
        descriptor);
  }

  iree_string_view_t family_name = iree_string_view_empty();
  if (!loom_encoding_numeric_transform_string_attr_value(
          module, family_param->value, &family_name)) {
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NON_STRING_FAMILY,
        family_param_name, descriptor);
  }
  descriptor.family =
      loom_encoding_numeric_transform_family_from_name(family_name);
  if (descriptor.family == LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN) {
    return loom_encoding_numeric_transform_make_read(
        LOOM_ENCODING_NUMERIC_TRANSFORM_READ_UNKNOWN_FAMILY, family_name,
        descriptor);
  }

  iree_string_view_t normalization_param_name =
      loom_encoding_numeric_transform_normalization_param_name();
  const loom_named_attr_t* normalization_param =
      loom_encoding_numeric_transform_find_named_attr(
          module, params.static_attrs, normalization_param_name);
  if (normalization_param) {
    iree_string_view_t normalization_name = iree_string_view_empty();
    if (!loom_encoding_numeric_transform_string_attr_value(
            module, normalization_param->value, &normalization_name)) {
      return loom_encoding_numeric_transform_make_read(
          LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NON_STRING_NORMALIZATION,
          normalization_param_name, descriptor);
    }
    if (!loom_encoding_numeric_transform_normalization_from_name(
            normalization_name, &descriptor.normalization)) {
      return loom_encoding_numeric_transform_make_read(
          LOOM_ENCODING_NUMERIC_TRANSFORM_READ_UNKNOWN_NORMALIZATION,
          normalization_name, descriptor);
    }
  }

  iree_string_view_t signs_param_name =
      loom_encoding_numeric_transform_signs_param_name();
  loom_encoding_numeric_transform_read_t read =
      loom_encoding_numeric_transform_read_dynamic_param(
          &params,
          loom_encoding_numeric_transform_find_named_attr(
              module, params.dynamic_names, signs_param_name),
          signs_param_name, &descriptor.signs, descriptor);
  if (read.code != LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK) return read;

  iree_string_view_t permutation_param_name =
      loom_encoding_numeric_transform_permutation_param_name();
  read = loom_encoding_numeric_transform_read_dynamic_param(
      &params,
      loom_encoding_numeric_transform_find_named_attr(
          module, params.dynamic_names, permutation_param_name),
      permutation_param_name, &descriptor.permutation, descriptor);
  if (read.code != LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK) return read;

  iree_string_view_t matrix_param_name =
      loom_encoding_numeric_transform_matrix_param_name();
  read = loom_encoding_numeric_transform_read_dynamic_param(
      &params,
      loom_encoding_numeric_transform_find_named_attr(
          module, params.dynamic_names, matrix_param_name),
      matrix_param_name, &descriptor.matrix, descriptor);
  if (read.code != LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK) return read;

  iree_string_view_t seed_param_name =
      loom_encoding_numeric_transform_seed_param_name();
  read = loom_encoding_numeric_transform_read_dynamic_param(
      &params,
      loom_encoding_numeric_transform_find_named_attr(
          module, params.dynamic_names, seed_param_name),
      seed_param_name, &descriptor.seed, descriptor);
  if (read.code != LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK) return read;

  return loom_encoding_numeric_transform_make_read(
      LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK, iree_string_view_empty(),
      descriptor);
}
