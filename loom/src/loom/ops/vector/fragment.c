// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/fragment.h"

#include <string.h>

#include "loom/util/stable_id.h"

// Stable key ID for the vector.fragment schema parameter.
static const uint64_t loom_vector_fragment_schema_key_id =
    UINT64_C(0x758d17ff764667b6);

loom_vector_fragment_role_flags_t loom_vector_fragment_role_flag(
    loom_vector_role_t role) {
  if (role >= LOOM_VECTOR_ROLE_COUNT_) {
    return 0;
  }
  return 1u << role;
}

loom_vector_fragment_role_flags_t loom_vector_fragment_fact_role_flags(
    loom_vector_role_t role) {
  loom_vector_fragment_role_flags_t role_flag =
      loom_vector_fragment_role_flag(role);
  if (role_flag == LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT ||
      role_flag == LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT) {
    return LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
           LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  }
  return role_flag;
}

void loom_vector_fragment_fact_initialize(
    loom_vector_fragment_fact_t* out_fact) {
  memset(out_fact, 0, sizeof(*out_fact));
}

bool loom_vector_fragment_fact_equal(loom_vector_fragment_fact_t lhs,
                                     loom_vector_fragment_fact_t rhs) {
  return memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

bool loom_vector_fragment_fact_is_unknown(loom_vector_fragment_fact_t fact) {
  loom_vector_fragment_fact_t unknown;
  loom_vector_fragment_fact_initialize(&unknown);
  return loom_vector_fragment_fact_equal(fact, unknown);
}

bool loom_vector_fragment_fact_is_accumulator_like(
    loom_vector_fragment_fact_t fact) {
  const loom_vector_fragment_role_flags_t accumulator_roles =
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  return (fact.shape_rank == 2 || fact.shape_rank == 3) &&
         fact.role_flags != 0 && (fact.role_flags & ~accumulator_roles) == 0;
}

bool loom_vector_fragment_fact_has_matrix_shape(
    loom_vector_fragment_fact_t fact) {
  return (fact.shape_rank == 2 || fact.shape_rank == 3) && fact.role_flags != 0;
}

loom_value_id_t loom_vector_fragment_fact_block_value(
    loom_vector_fragment_fact_t fact) {
  return fact.shape_rank == 3 ? fact.shape_value_ids[0] : LOOM_VALUE_ID_INVALID;
}

loom_value_id_t loom_vector_fragment_fact_row_value(
    loom_vector_fragment_fact_t fact) {
  if (fact.shape_rank != 2 && fact.shape_rank != 3) {
    return LOOM_VALUE_ID_INVALID;
  }
  return fact.shape_value_ids[fact.shape_rank - 2];
}

loom_value_id_t loom_vector_fragment_fact_column_value(
    loom_vector_fragment_fact_t fact) {
  if (fact.shape_rank != 2 && fact.shape_rank != 3) {
    return LOOM_VALUE_ID_INVALID;
  }
  return fact.shape_value_ids[fact.shape_rank - 1];
}

static bool loom_vector_fragment_facts_match_contract_except_native_storage(
    loom_vector_fragment_fact_t lhs, loom_vector_fragment_fact_t rhs) {
  lhs.flags &= ~LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  rhs.flags &= ~LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  return loom_vector_fragment_fact_equal(lhs, rhs);
}

bool loom_vector_fragment_facts_match_accumulator_contract(
    loom_vector_fragment_fact_t lhs, loom_vector_fragment_fact_t rhs) {
  if (!loom_vector_fragment_fact_is_accumulator_like(lhs) ||
      !loom_vector_fragment_fact_is_accumulator_like(rhs)) {
    return false;
  }
  lhs.role_flags = LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                   LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  rhs.role_flags = LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                   LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  return loom_vector_fragment_facts_match_contract_except_native_storage(lhs,
                                                                         rhs);
}

iree_status_t loom_vector_fragment_fact_make_value_facts(
    loom_fact_context_t* context, loom_vector_fragment_fact_t fact,
    loom_value_facts_t* out_facts) {
  if (loom_vector_fragment_fact_is_unknown(fact)) {
    *out_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }
  return loom_value_facts_make_extension_payload(
      context, LOOM_VECTOR_FRAGMENT_FACT_PAYLOAD_TAG_FRAGMENT, &fact,
      sizeof(fact), out_facts);
}

bool loom_vector_fragment_fact_query_value_facts(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_vector_fragment_fact_t* out_fact) {
  const void* payload = NULL;
  iree_host_size_t payload_length = 0;
  if (!loom_value_facts_query_extension_payload(
          context, facts, LOOM_VECTOR_FRAGMENT_FACT_PAYLOAD_TAG_FRAGMENT,
          &payload, &payload_length) ||
      payload_length != sizeof(*out_fact)) {
    return false;
  }
  if (out_fact) {
    memcpy(out_fact, payload, sizeof(*out_fact));
  }
  return true;
}

bool loom_vector_fragment_parameter_view_resolve(
    const loom_module_t* module, loom_value_slice_t parameter_values,
    loom_named_attr_slice_t parameter_names,
    loom_vector_fragment_parameter_view_t* out_view,
    iree_string_view_t* out_unknown_key) {
  memset(out_view, 0, sizeof(*out_view));
  out_view->schema_value_id = LOOM_VALUE_ID_INVALID;
  out_view->schema_parameter_ordinal = UINT16_MAX;
  loom_encoding_auxiliary_view_initialize(&out_view->auxiliary);
  if (out_unknown_key) {
    *out_unknown_key = iree_string_view_empty();
  }

  for (iree_host_size_t i = 0; i < parameter_names.count; ++i) {
    const loom_named_attr_t* entry = &parameter_names.entries[i];
    if (entry->name_id == LOOM_STRING_ID_INVALID ||
        entry->name_id >= module->strings.count) {
      continue;
    }

    int64_t ordinal =
        entry->value.kind == LOOM_ATTR_I64 ? entry->value.i64 : -1;
    if (ordinal < 0 || ordinal >= parameter_values.count) {
      continue;
    }

    iree_string_view_t key_name = module->strings.entries[entry->name_id];
    uint64_t stable_id = loom_stable_id_from_string(key_name);
    if (stable_id == loom_vector_fragment_schema_key_id) {
      out_view->has_schema = true;
      out_view->schema_value_id = parameter_values.values[ordinal];
      out_view->schema_parameter_ordinal = (uint16_t)ordinal;
      continue;
    }

    loom_encoding_auxiliary_key_t auxiliary_key = 0;
    if (!loom_encoding_auxiliary_key_lookup_stable_id(stable_id,
                                                      &auxiliary_key)) {
      if (out_unknown_key) {
        *out_unknown_key = key_name;
      }
      return false;
    }
    out_view->auxiliary.values[auxiliary_key] =
        parameter_values.values[ordinal];
    out_view->auxiliary.present_keys |=
        loom_encoding_auxiliary_key_flag(auxiliary_key);
  }
  return true;
}
