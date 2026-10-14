// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/cache.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"

loom_cache_policy_t loom_cache_policy_cast(const loom_module_t* module,
                                           const loom_op_t* op) {
  const loom_op_vtable_t* vtable = op ? loom_op_vtable(module, op) : NULL;
  if (!vtable || !vtable->cache_policy.available) {
    return (loom_cache_policy_t){0};
  }
  return (loom_cache_policy_t){
      .attributes = loom_op_const_attrs(op),
      .vtable = &vtable->cache_policy,
  };
}

bool loom_cache_scope_is_valid(uint8_t scope) {
  return scope < LOOM_CACHE_SCOPE_COUNT_;
}

bool loom_cache_temporal_is_valid(uint8_t temporal) {
  return temporal < LOOM_CACHE_TEMPORAL_COUNT_;
}

static bool loom_cache_temporal_is_load_compatible(uint8_t temporal) {
  return temporal != LOOM_CACHE_TEMPORAL_WRITEBACK &&
         temporal != LOOM_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK;
}

static bool loom_cache_temporal_is_store_compatible(uint8_t temporal) {
  return temporal != LOOM_CACHE_TEMPORAL_LAST_USE;
}

static bool loom_cache_temporal_is_atomic_compatible(uint8_t temporal) {
  return temporal == LOOM_CACHE_TEMPORAL_REGULAR ||
         temporal == LOOM_CACHE_TEMPORAL_NON_TEMPORAL;
}

loom_cache_policy_error_t loom_cache_policy_validate(
    uint8_t scope, uint8_t temporal, loom_cache_policy_access_t access) {
  if (!loom_cache_scope_is_valid(scope)) {
    return LOOM_CACHE_POLICY_ERROR_INVALID_SCOPE;
  }
  if (!loom_cache_temporal_is_valid(temporal)) {
    return LOOM_CACHE_POLICY_ERROR_INVALID_TEMPORAL;
  }

  switch (temporal) {
    case LOOM_CACHE_TEMPORAL_LAST_USE:
      if (scope == LOOM_CACHE_SCOPE_SYSTEM) {
        return LOOM_CACHE_POLICY_ERROR_LAST_USE_SYSTEM_SCOPE;
      }
      break;
    case LOOM_CACHE_TEMPORAL_BYPASS:
      if (scope != LOOM_CACHE_SCOPE_SYSTEM) {
        return LOOM_CACHE_POLICY_ERROR_BYPASS_NON_SYSTEM_SCOPE;
      }
      break;
    default:
      break;
  }

  switch (access) {
    case LOOM_CACHE_POLICY_ACCESS_LOAD:
      return loom_cache_temporal_is_load_compatible(temporal)
                 ? LOOM_CACHE_POLICY_ERROR_NONE
                 : LOOM_CACHE_POLICY_ERROR_LOAD_TEMPORAL;
    case LOOM_CACHE_POLICY_ACCESS_STORE:
      return loom_cache_temporal_is_store_compatible(temporal)
                 ? LOOM_CACHE_POLICY_ERROR_NONE
                 : LOOM_CACHE_POLICY_ERROR_STORE_TEMPORAL;
    case LOOM_CACHE_POLICY_ACCESS_ATOMIC:
      return loom_cache_temporal_is_atomic_compatible(temporal)
                 ? LOOM_CACHE_POLICY_ERROR_NONE
                 : LOOM_CACHE_POLICY_ERROR_ATOMIC_TEMPORAL;
  }

  return LOOM_CACHE_POLICY_ERROR_NONE;
}

static bool loom_cache_policy_error_is_scope(loom_cache_policy_error_t error) {
  switch (error) {
    case LOOM_CACHE_POLICY_ERROR_INVALID_SCOPE:
    case LOOM_CACHE_POLICY_ERROR_LAST_USE_SYSTEM_SCOPE:
    case LOOM_CACHE_POLICY_ERROR_BYPASS_NON_SYSTEM_SCOPE:
      return true;
    case LOOM_CACHE_POLICY_ERROR_INVALID_TEMPORAL:
    case LOOM_CACHE_POLICY_ERROR_LOAD_TEMPORAL:
    case LOOM_CACHE_POLICY_ERROR_STORE_TEMPORAL:
    case LOOM_CACHE_POLICY_ERROR_ATOMIC_TEMPORAL:
    case LOOM_CACHE_POLICY_ERROR_NONE:
      return false;
  }
  return false;
}

static iree_string_view_t loom_cache_policy_error_expected_constraint(
    loom_cache_policy_error_t error) {
  switch (error) {
    case LOOM_CACHE_POLICY_ERROR_INVALID_SCOPE:
      return IREE_SV("workgroup, cluster, device, or system");
    case LOOM_CACHE_POLICY_ERROR_INVALID_TEMPORAL:
      return IREE_SV("supported cache temporal hint");
    case LOOM_CACHE_POLICY_ERROR_LOAD_TEMPORAL:
      return IREE_SV("load-compatible temporal hint");
    case LOOM_CACHE_POLICY_ERROR_STORE_TEMPORAL:
      return IREE_SV("store-compatible temporal hint");
    case LOOM_CACHE_POLICY_ERROR_ATOMIC_TEMPORAL:
      return IREE_SV("regular or non_temporal temporal hint");
    case LOOM_CACHE_POLICY_ERROR_LAST_USE_SYSTEM_SCOPE:
      return IREE_SV("non-system scope for last_use temporal hint");
    case LOOM_CACHE_POLICY_ERROR_BYPASS_NON_SYSTEM_SCOPE:
      return IREE_SV("system scope for bypass temporal hint");
    case LOOM_CACHE_POLICY_ERROR_NONE:
      return iree_string_view_empty();
  }
  return iree_string_view_empty();
}

iree_status_t loom_cache_policy_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       loom_cache_policy_access_t access,
                                       iree_diagnostic_emitter_t emitter) {
  const loom_cache_policy_t policy = loom_cache_policy_cast(module, op);
  const loom_attribute_t scope = loom_cache_policy_scope(policy);
  const loom_attribute_t temporal = loom_cache_policy_temporal(policy);
  const bool has_scope = !loom_attr_is_absent(scope);
  const bool has_temporal = !loom_attr_is_absent(temporal);
  if (!has_scope && !has_temporal) {
    return iree_ok_status();
  }

  uint8_t attribute_index = 0;
  iree_string_view_t attribute_name = iree_string_view_empty();
  int64_t actual_value = 0;
  iree_string_view_t expected_constraint = iree_string_view_empty();
  if (!has_scope) {
    attribute_index = policy.vtable->scope_attr_index;
    attribute_name = IREE_SV("cache_scope");
    expected_constraint = IREE_SV("present when cache_temporal is present");
  } else if (!has_temporal) {
    attribute_index = policy.vtable->temporal_attr_index;
    attribute_name = IREE_SV("cache_temporal");
    expected_constraint = IREE_SV("present when cache_scope is present");
  } else {
    const loom_cache_policy_error_t error = loom_cache_policy_validate(
        loom_attr_as_enum(scope), loom_attr_as_enum(temporal), access);
    if (error == LOOM_CACHE_POLICY_ERROR_NONE) {
      return iree_ok_status();
    }
    const bool is_scope = loom_cache_policy_error_is_scope(error);
    attribute_index = is_scope ? policy.vtable->scope_attr_index
                               : policy.vtable->temporal_attr_index;
    attribute_name =
        is_scope ? IREE_SV("cache_scope") : IREE_SV("cache_temporal");
    actual_value = loom_attr_as_enum(is_scope ? scope : temporal);
    expected_constraint = loom_cache_policy_error_expected_constraint(error);
  }

  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(attribute_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attribute_index)),
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
