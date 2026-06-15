// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/cache.h"

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

iree_string_view_t loom_cache_policy_error_attr_name(
    loom_cache_policy_error_t error) {
  switch (error) {
    case LOOM_CACHE_POLICY_ERROR_INVALID_SCOPE:
    case LOOM_CACHE_POLICY_ERROR_LAST_USE_SYSTEM_SCOPE:
    case LOOM_CACHE_POLICY_ERROR_BYPASS_NON_SYSTEM_SCOPE:
      return IREE_SV("cache_scope");
    case LOOM_CACHE_POLICY_ERROR_INVALID_TEMPORAL:
    case LOOM_CACHE_POLICY_ERROR_LOAD_TEMPORAL:
    case LOOM_CACHE_POLICY_ERROR_STORE_TEMPORAL:
    case LOOM_CACHE_POLICY_ERROR_ATOMIC_TEMPORAL:
      return IREE_SV("cache_temporal");
    case LOOM_CACHE_POLICY_ERROR_NONE:
      return iree_string_view_empty();
  }
  return iree_string_view_empty();
}

iree_string_view_t loom_cache_policy_error_expected_constraint(
    loom_cache_policy_error_t error) {
  switch (error) {
    case LOOM_CACHE_POLICY_ERROR_INVALID_SCOPE:
      return IREE_SV("cu, se, device, or system");
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
