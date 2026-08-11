// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/auxiliary.h"

#include <string.h>

#include "loom/util/stable_id.h"

//===----------------------------------------------------------------------===//
// Key vocabulary
//===----------------------------------------------------------------------===//

iree_string_view_t loom_encoding_auxiliary_key_name(
    loom_encoding_auxiliary_key_t key) {
  if (key >= LOOM_ENCODING_AUXILIARY_KEY_COUNT_ ||
      loom_encoding_auxiliary_key_descriptors[key].name == NULL) {
    return iree_string_view_empty();
  }
  return loom_bstring_view(loom_encoding_auxiliary_key_descriptors[key].name);
}

bool loom_encoding_auxiliary_key_lookup_stable_id(
    uint64_t stable_id, loom_encoding_auxiliary_key_t* out_key) {
  for (uint8_t i = 0; i < LOOM_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    const loom_encoding_auxiliary_key_descriptor_t* descriptor =
        &loom_encoding_auxiliary_key_descriptors[i];
    if (descriptor->name == NULL || stable_id != descriptor->stable_id) {
      continue;
    }
    *out_key = (loom_encoding_auxiliary_key_t)i;
    return true;
  }
  return false;
}

bool loom_encoding_auxiliary_key_lookup(
    iree_string_view_t name, loom_encoding_auxiliary_key_t* out_key) {
  return loom_encoding_auxiliary_key_lookup_stable_id(
      loom_stable_id_from_string(name), out_key);
}

bool loom_encoding_auxiliary_scale_key(uint16_t index,
                                       loom_encoding_auxiliary_key_t* out_key) {
  static const loom_encoding_auxiliary_key_t scale_keys[] = {
      LOOM_ENCODING_AUXILIARY_KEY_SCALE,
      LOOM_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE,
      LOOM_ENCODING_AUXILIARY_KEY_SCALE2,
      LOOM_ENCODING_AUXILIARY_KEY_SCALE3,
      LOOM_ENCODING_AUXILIARY_KEY_SCALE4,
      LOOM_ENCODING_AUXILIARY_KEY_SCALE5,
      LOOM_ENCODING_AUXILIARY_KEY_SCALE6,
      LOOM_ENCODING_AUXILIARY_KEY_SCALE7,
  };
  if (index >= IREE_ARRAYSIZE(scale_keys)) {
    return false;
  }
  *out_key = scale_keys[index];
  return true;
}

//===----------------------------------------------------------------------===//
// View resolution
//===----------------------------------------------------------------------===//

void loom_encoding_auxiliary_view_initialize(
    loom_encoding_auxiliary_view_t* out_view) {
  memset(out_view, 0, sizeof(*out_view));
  for (uint8_t i = 0; i < LOOM_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    out_view->values[i] = LOOM_VALUE_ID_INVALID;
  }
}

bool loom_encoding_auxiliary_view_resolve(
    const loom_module_t* module, loom_value_slice_t auxiliary_values,
    loom_named_attr_slice_t auxiliary_names,
    loom_encoding_auxiliary_view_t* out_view,
    iree_string_view_t* out_unknown_key) {
  loom_encoding_auxiliary_view_initialize(out_view);
  if (out_unknown_key) {
    *out_unknown_key = iree_string_view_empty();
  }
  for (iree_host_size_t i = 0; i < auxiliary_names.count; ++i) {
    const loom_named_attr_t* entry = &auxiliary_names.entries[i];
    if (entry->name_id == LOOM_STRING_ID_INVALID ||
        entry->name_id >= module->strings.count) {
      continue;
    }
    iree_string_view_t key_name = module->strings.entries[entry->name_id];
    loom_encoding_auxiliary_key_t key = 0;
    if (!loom_encoding_auxiliary_key_lookup_stable_id(
            loom_stable_id_from_string(key_name), &key)) {
      if (out_unknown_key) {
        *out_unknown_key = key_name;
      }
      return false;
    }
    int64_t ordinal =
        entry->value.kind == LOOM_ATTR_I64 ? entry->value.i64 : -1;
    iree_host_size_t ordinal_index = (iree_host_size_t)ordinal;
    if (ordinal >= 0 && ordinal_index < auxiliary_values.count) {
      out_view->values[key] = auxiliary_values.values[ordinal_index];
    }
    out_view->present_keys |= loom_encoding_auxiliary_key_flag(key);
  }
  return true;
}

bool loom_encoding_auxiliary_required_keys_from_schema(
    loom_value_fact_encoded_operand_schema_t schema,
    loom_encoding_auxiliary_key_flags_t* out_required_keys,
    uint16_t* out_unsupported_scale_index) {
  loom_encoding_auxiliary_key_flags_t required_keys = 0;
  if (out_unsupported_scale_index) {
    *out_unsupported_scale_index = UINT16_MAX;
  }

  for (uint16_t i = 0; i < schema.scale_operand_count; ++i) {
    loom_encoding_auxiliary_key_t key = 0;
    if (!loom_encoding_auxiliary_scale_key(i, &key)) {
      if (out_unsupported_scale_index) {
        *out_unsupported_scale_index = i;
      }
      *out_required_keys = required_keys;
      return false;
    }
    required_keys |= loom_encoding_auxiliary_key_flag(key);
  }

  if (iree_any_bit_set(schema.scale_topology,
                       LOOM_VALUE_FACT_SCALE_TOPOLOGY_RUNTIME_AMAX_DERIVED)) {
    required_keys |=
        loom_encoding_auxiliary_key_flag(LOOM_ENCODING_AUXILIARY_KEY_AMAX);
  }
  if (iree_any_bit_set(schema.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT)) {
    required_keys |= loom_encoding_auxiliary_key_flag(
        LOOM_ENCODING_AUXILIARY_KEY_ZERO_POINT);
  }
  if (iree_any_bit_set(schema.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN)) {
    required_keys |=
        loom_encoding_auxiliary_key_flag(LOOM_ENCODING_AUXILIARY_KEY_MINIMUM);
  }
  if (iree_any_bit_set(schema.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS)) {
    required_keys |=
        loom_encoding_auxiliary_key_flag(LOOM_ENCODING_AUXILIARY_KEY_BIAS);
  }
  if (iree_any_bit_set(schema.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SUM_CORRECTION)) {
    required_keys |= loom_encoding_auxiliary_key_flag(
        LOOM_ENCODING_AUXILIARY_KEY_SUM_CORRECTION);
  }
  if (iree_any_bit_set(schema.codebook_policy,
                       LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND)) {
    required_keys |=
        loom_encoding_auxiliary_key_flag(LOOM_ENCODING_AUXILIARY_KEY_CODEBOOK);
  }
  if (schema.sparsity_policy != 0) {
    required_keys |=
        loom_encoding_auxiliary_key_flag(LOOM_ENCODING_AUXILIARY_KEY_SPARSITY);
  }

  *out_required_keys = required_keys;
  return true;
}
