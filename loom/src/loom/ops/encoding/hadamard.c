// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/hadamard.h"

#include "loom/ops/encoding/params.h"

static const loom_encoding_t* loom_encoding_hadamard_try_find_definition(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return NULL;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return NULL;

  const loom_op_t* define_op = loom_value_def_op(value);
  if (!define_op || !loom_encoding_define_isa(define_op)) return NULL;
  const loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, define_op);
  if (!params.spec ||
      loom_module_encoding_family_descriptor(module, params.spec) !=
          &loom_encoding_transform_hadamard_family_descriptor) {
    return NULL;
  }
  return params.spec;
}

bool loom_encoding_hadamard_isa(const loom_module_t* module,
                                loom_value_id_t value_id) {
  return loom_encoding_hadamard_try_find_definition(module, value_id) != NULL;
}

bool loom_encoding_hadamard_try_read_verified_descriptor(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_encoding_hadamard_descriptor_t* out_descriptor) {
  const loom_encoding_t* encoding =
      loom_encoding_hadamard_try_find_definition(module, value_id);
  if (!encoding) return false;

  const loom_named_attr_t*
      parameters[LOOM_ENCODING_TRANSFORM_HADAMARD_PARAMETER_COUNT_];
  loom_encoding_collect_parameter_slots(
      encoding, LOOM_ENCODING_TRANSFORM_HADAMARD_PARAMETER_COUNT_, parameters);
  const loom_named_attr_t* normalization =
      parameters[LOOM_ENCODING_TRANSFORM_HADAMARD_PARAMETER_NORMALIZATION];
  *out_descriptor = (loom_encoding_hadamard_descriptor_t){
      .normalization = normalization
                           ? (loom_encoding_transform_normalization_t)
                                 loom_attr_as_enum(normalization->value)
                           : LOOM_ENCODING_TRANSFORM_NORMALIZATION_NONE,
  };
  return true;
}
