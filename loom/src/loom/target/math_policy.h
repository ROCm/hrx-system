// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target math legalization policy.
//
// Semantic math ops carry source-level intent such as logistic, SiLU, softplus,
// and GELU formula families. Targets decide whether those ops can remain in IR,
// must rewrite through a target-independent recipe, or are illegal for the
// selected contract set. The policy surface is intentionally target-neutral:
// target packages contribute compact contract-key tables and decision callbacks
// without leaking backend descriptors into the math pass.

#ifndef LOOM_TARGET_MATH_POLICY_H_
#define LOOM_TARGET_MATH_POLICY_H_

#include "iree/base/api.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_math_policy_t loom_target_math_policy_t;
typedef struct loom_target_math_policy_registry_t
    loom_target_math_policy_registry_t;

typedef enum loom_target_math_op_e {
  LOOM_TARGET_MATH_OP_UNKNOWN = 0,
  LOOM_TARGET_MATH_OP_LOGISTICF = 1,
  LOOM_TARGET_MATH_OP_SILUF = 2,
  LOOM_TARGET_MATH_OP_SOFTPLUSF = 3,
  LOOM_TARGET_MATH_OP_EXPF = 4,
  LOOM_TARGET_MATH_OP_ERFF = 5,
  LOOM_TARGET_MATH_OP_GELUF_ERF = 6,
  LOOM_TARGET_MATH_OP_GELUF_TANH = 7,
  LOOM_TARGET_MATH_OP_GELUF_LOGISTIC = 8,
  LOOM_TARGET_MATH_OP_ADDF = 9,
  LOOM_TARGET_MATH_OP_MULF = 10,
  LOOM_TARGET_MATH_OP_LOGF = 11,
  LOOM_TARGET_MATH_OP_LOG2F = 12,
  LOOM_TARGET_MATH_OP_SINF = 13,
  LOOM_TARGET_MATH_OP_COSF = 14,
  LOOM_TARGET_MATH_OP_SINTURNSF = 15,
  LOOM_TARGET_MATH_OP_COSTURNSF = 16,
  LOOM_TARGET_MATH_OP_TANHF = 17,
  LOOM_TARGET_MATH_OP_POWF = 18,
  LOOM_TARGET_MATH_OP_CEILF = 19,
  LOOM_TARGET_MATH_OP_FLOORF = 20,
  LOOM_TARGET_MATH_OP_ROUNDF = 21,
  LOOM_TARGET_MATH_OP_ROUNDEVENF = 22,
  LOOM_TARGET_MATH_OP_TRUNCF = 23,
} loom_target_math_op_t;

typedef enum loom_target_math_lane_domain_e {
  LOOM_TARGET_MATH_LANE_DOMAIN_UNKNOWN = 0,
  LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR = 1,
  LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR = 2,
} loom_target_math_lane_domain_t;

// Mirrors the scalar/vector FastMathFlags bit layout so target policy decisions
// can be projected into generated source ops without per-dialect tables.
typedef enum loom_target_math_fastmath_flag_bits_e {
  LOOM_TARGET_MATH_FASTMATH_FLAG_NONE = 0u,
  LOOM_TARGET_MATH_FASTMATH_FLAG_REASSOC = 1u << 0,
  LOOM_TARGET_MATH_FASTMATH_FLAG_NNAN = 1u << 1,
  LOOM_TARGET_MATH_FASTMATH_FLAG_NINF = 1u << 2,
  LOOM_TARGET_MATH_FASTMATH_FLAG_NSZ = 1u << 3,
  LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP = 1u << 4,
  LOOM_TARGET_MATH_FASTMATH_FLAG_CONTRACT = 1u << 5,
  LOOM_TARGET_MATH_FASTMATH_FLAG_AFN = 1u << 6,
  LOOM_TARGET_MATH_FASTMATH_FLAG_FAST = 0x7Fu,
} loom_target_math_fastmath_flag_bits_t;
typedef uint8_t loom_target_math_fastmath_flags_t;

typedef enum loom_target_math_policy_action_e {
  LOOM_TARGET_MATH_POLICY_ACTION_UNKNOWN = 0,
  LOOM_TARGET_MATH_POLICY_ACTION_KEEP = 1,
  LOOM_TARGET_MATH_POLICY_ACTION_REWRITE = 2,
  LOOM_TARGET_MATH_POLICY_ACTION_REJECT = 3,
} loom_target_math_policy_action_t;

typedef enum loom_target_math_recipe_e {
  LOOM_TARGET_MATH_RECIPE_UNKNOWN = 0,
  LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32 = 1,
  LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32 = 2,
  LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32 = 3,
  LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32 = 4,
  LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32 = 5,
  LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32 = 6,
  LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32 = 7,
  LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32 = 8,
  LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND_BF16 = 9,
  LOOM_TARGET_MATH_RECIPE_LOG_LOG2_F32 = 10,
  LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32 = 11,
  LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32 = 12,
  LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32 = 13,
  LOOM_TARGET_MATH_RECIPE_POW_LOG2_EXP2_F32 = 14,
  LOOM_TARGET_MATH_RECIPE_ROUND_AWAY_F32 = 15,
} loom_target_math_recipe_t;

typedef struct loom_target_math_query_t {
  // Semantic math operation requested by the source op.
  loom_target_math_op_t math_op;
  // Whether the source op computes one scalar lane or a vector of lanes.
  loom_target_math_lane_domain_t lane_domain;
  // Full source value type carried by the op result.
  loom_type_t value_type;
  // Scalar element type of |value_type|.
  loom_scalar_type_t element_type;
  // Fast-math permission bits projected into the target-neutral bit layout.
  loom_target_math_fastmath_flags_t fastmath_flags;
} loom_target_math_query_t;

typedef struct loom_target_math_policy_decision_t {
  // Action selected by the target policy.
  loom_target_math_policy_action_t action;
  // Recipe selected when |action| is REWRITE.
  loom_target_math_recipe_t recipe;
  // Extra fast-math permissions the selected target recipe may use internally.
  // Recipe builders apply these only to generated operations covered by the
  // recipe's error contract.
  loom_target_math_fastmath_flags_t recipe_fastmath_flags;
  // Stable structured constraint key for diagnostics.
  iree_string_view_t constraint_key;
} loom_target_math_policy_decision_t;

typedef void (*loom_target_math_policy_query_fn_t)(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision);

struct loom_target_math_policy_t {
  // Stable policy name used in diagnostics.
  iree_string_view_t name;
  // Target-owned decision callback.
  loom_target_math_policy_query_fn_t query;
  // Target-owned immutable payload forwarded to |query|.
  const void* user_data;
};

typedef struct loom_target_math_policy_registry_entry_t {
  // Target contract-set key that selects |policy|.
  iree_string_view_t contract_set_key;
  // Borrowed math policy used for bundles naming |contract_set_key|.
  const loom_target_math_policy_t* policy;
} loom_target_math_policy_registry_entry_t;

struct loom_target_math_policy_registry_t {
  // Borrowed contract-set key to policy table linked into the target package.
  const loom_target_math_policy_registry_entry_t* entries;
  // Number of rows in |entries|.
  iree_host_size_t entry_count;
};

// Capability type for loom_target_math_pass_capability_t.
extern const loom_pass_environment_capability_type_t
    loom_target_math_pass_capability_type;

typedef struct loom_target_math_pass_capability_t {
  // Base capability header. Must remain the first field.
  loom_pass_environment_capability_t base;
  // Linked target math policy registry.
  const loom_target_math_policy_registry_t* policy_registry;
} loom_target_math_pass_capability_t;

// Initializes |out_registry| from a target-owned entry table. The table is
// borrowed and must outlive |out_registry|.
void loom_target_math_policy_registry_initialize_from_entries(
    loom_target_math_policy_registry_t* out_registry,
    const loom_target_math_policy_registry_entry_t* entries,
    iree_host_size_t entry_count);

// Looks up the math policy for |contract_set_key|, or returns NULL when no row
// matches.
const loom_target_math_policy_t* loom_target_math_policy_registry_lookup(
    const loom_target_math_policy_registry_t* registry,
    iree_string_view_t contract_set_key);

// Looks up the math policy for |bundle|'s target-contract key, or returns NULL
// when no row matches.
const loom_target_math_policy_t*
loom_target_math_policy_registry_lookup_for_bundle(
    const loom_target_math_policy_registry_t* registry,
    const loom_target_bundle_t* bundle);

// Applies |policy| to |query|.
void loom_target_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision);

// Returns the assembly spelling for |math_op| used in diagnostics.
iree_string_view_t loom_target_math_op_name(loom_target_math_op_t math_op);

// Returns the assembly spelling for |domain| used in diagnostics.
iree_string_view_t loom_target_math_lane_domain_name(
    loom_target_math_lane_domain_t domain);

// Returns the assembly spelling for |recipe| used in diagnostics.
iree_string_view_t loom_target_math_recipe_name(
    loom_target_math_recipe_t recipe);

// Creates a borrowed target-math pass capability.
loom_target_math_pass_capability_t loom_target_math_pass_capability_make(
    const loom_target_math_policy_registry_t* policy_registry);

// Looks up the target math capability from |environment|. Returns NULL when
// absent.
const loom_target_math_pass_capability_t*
loom_target_math_pass_capability_from_environment(
    const loom_pass_environment_t* environment);

// Looks up the target math capability from |pass->environment|. Returns NULL
// when absent.
const loom_target_math_pass_capability_t*
loom_target_math_pass_capability_from_pass(const loom_pass_t* pass);

// Returns the policy registry selected by |capability|, or NULL.
const loom_target_math_policy_registry_t*
loom_target_math_pass_capability_policy_registry(
    const loom_target_math_pass_capability_t* capability);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_MATH_POLICY_H_
