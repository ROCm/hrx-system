// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Backend-independent source-to-low policy for the synthetic test-low target.
//
// The test-low target is intentionally tiny and deterministic. It lets generic
// low lowering, scheduling, allocation, packetization, and generated stress
// tests exercise a real descriptor-driven target without linking optional
// production backends.

#ifndef LOOM_TARGET_TEST_LOWER_H_
#define LOOM_TARGET_TEST_LOWER_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/lower/lower_rules.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps source semantic types to test-low register types.
iree_status_t loom_test_low_lower_map_type(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type);

// Maps source function arguments to direct low arguments or resource imports.
iree_status_t loom_test_low_lower_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument);

// Maps one source value into target-low register metadata for direct rule-table
// selection tests. This mirrors the policy map callbacks without mutating IR.
iree_status_t loom_test_low_lower_rule_match_map_value(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value);

// Maps one source value into target-low register metadata for read-only target
// contract queries.
iree_status_t loom_test_low_lower_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value);

// Returns the canonical test-low lowering policy.
const loom_low_lower_policy_t* loom_test_low_lower_policy(void);

// Initializes a registry containing the canonical test.low.core policy.
void loom_test_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TEST_LOWER_H_
