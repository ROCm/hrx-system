// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared cache-policy vocabulary used by kernel async copies, view scalar
// memory operations, and vector memory operations. Generated dialect APIs alias
// these enum types directly so verification, lowering, and builders use one
// semantic domain.

#ifndef LOOM_OPS_CACHE_H_
#define LOOM_OPS_CACHE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cache/coherency scope for memory operations.
typedef enum loom_cache_scope_e {
  // Cache/coherency scope is the compute unit.
  LOOM_CACHE_SCOPE_CU = 0,
  // Cache/coherency scope is the shader engine.
  LOOM_CACHE_SCOPE_SE = 1,
  // Cache/coherency scope is the current device.
  LOOM_CACHE_SCOPE_DEVICE = 2,
  // Cache/coherency scope is the full system.
  LOOM_CACHE_SCOPE_SYSTEM = 3,
  LOOM_CACHE_SCOPE_COUNT_,
} loom_cache_scope_t;

// Temporal cache policy for memory operations.
typedef enum loom_cache_temporal_e {
  // Regular temporal caching behavior.
  LOOM_CACHE_TEMPORAL_REGULAR = 0,
  // Data is expected to have little or no temporal reuse.
  LOOM_CACHE_TEMPORAL_NON_TEMPORAL = 1,
  // Data is expected to be reused and should be retained when possible.
  LOOM_CACHE_TEMPORAL_HIGH_TEMPORAL = 2,
  // Data is not expected to be used again after this memory operation.
  LOOM_CACHE_TEMPORAL_LAST_USE = 3,
  // Store-oriented high-temporal write-back policy.
  LOOM_CACHE_TEMPORAL_WRITEBACK = 4,
  // Non-temporal near-cache behavior with regular outer-cache behavior.
  LOOM_CACHE_TEMPORAL_NON_TEMPORAL_REGULAR = 5,
  // Regular near-cache behavior with non-temporal outer-cache behavior.
  LOOM_CACHE_TEMPORAL_REGULAR_NON_TEMPORAL = 6,
  // Non-temporal near-cache behavior with high-temporal outer-cache behavior.
  LOOM_CACHE_TEMPORAL_NON_TEMPORAL_HIGH_TEMPORAL = 7,
  // Store-oriented non-temporal near-cache behavior with write-back outer-cache
  // behavior.
  LOOM_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK = 8,
  // Bypass caches at the requested cache scope when the target supports it.
  LOOM_CACHE_TEMPORAL_BYPASS = 9,
  LOOM_CACHE_TEMPORAL_COUNT_,
} loom_cache_temporal_t;

// Memory operation kind used for cache-policy compatibility checks.
typedef enum loom_cache_policy_access_e {
  // Memory operation reads without writing.
  LOOM_CACHE_POLICY_ACCESS_LOAD = 0,
  // Memory operation writes without returning an old value.
  LOOM_CACHE_POLICY_ACCESS_STORE = 1,
  // Memory operation performs an atomic read-modify-write.
  LOOM_CACHE_POLICY_ACCESS_ATOMIC = 2,
} loom_cache_policy_access_t;

// Cache-policy validation result.
typedef enum loom_cache_policy_error_e {
  // Cache policy is valid for the requested access.
  LOOM_CACHE_POLICY_ERROR_NONE = 0,
  // Cache scope is outside the shared cache-scope vocabulary.
  LOOM_CACHE_POLICY_ERROR_INVALID_SCOPE = 1,
  // Cache temporal policy is outside the shared cache-temporal vocabulary.
  LOOM_CACHE_POLICY_ERROR_INVALID_TEMPORAL = 2,
  // Temporal policy is not valid on a read-like operation.
  LOOM_CACHE_POLICY_ERROR_LOAD_TEMPORAL = 3,
  // Temporal policy is not valid on a write-like operation.
  LOOM_CACHE_POLICY_ERROR_STORE_TEMPORAL = 4,
  // Temporal policy is not valid on an atomic read-modify-write operation.
  LOOM_CACHE_POLICY_ERROR_ATOMIC_TEMPORAL = 5,
  // Last-use policy cannot be requested at system scope.
  LOOM_CACHE_POLICY_ERROR_LAST_USE_SYSTEM_SCOPE = 6,
  // Bypass policy requires system scope.
  LOOM_CACHE_POLICY_ERROR_BYPASS_NON_SYSTEM_SCOPE = 7,
} loom_cache_policy_error_t;

// Returns true when |scope| is a known loom_cache_scope_t value.
bool loom_cache_scope_is_valid(uint8_t scope);

// Returns true when |temporal| is a known loom_cache_temporal_t value.
bool loom_cache_temporal_is_valid(uint8_t temporal);

// Validates a complete cache policy for a concrete memory operation kind.
loom_cache_policy_error_t loom_cache_policy_validate(
    uint8_t scope, uint8_t temporal, loom_cache_policy_access_t access);

// Returns the attribute name responsible for |error|.
iree_string_view_t loom_cache_policy_error_attr_name(
    loom_cache_policy_error_t error);

// Returns the expected constraint for |error|.
iree_string_view_t loom_cache_policy_error_expected_constraint(
    loom_cache_policy_error_t error);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_CACHE_H_
