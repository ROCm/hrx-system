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
// target packages contribute compact contract-key tables without leaking
// backend descriptors into the math pass.

#ifndef LOOM_TARGET_MATH_POLICY_H_
#define LOOM_TARGET_MATH_POLICY_H_

#include "iree/base/api.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/target/types.h"
#include "loom/util/string_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_math_policy_t loom_target_math_policy_t;
typedef struct loom_target_math_policy_registry_t
    loom_target_math_policy_registry_t;
typedef struct loom_target_compile_report_t loom_target_compile_report_t;

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
  LOOM_TARGET_MATH_OP_SUBF = 24,
  LOOM_TARGET_MATH_OP_SQRTF = 25,
  LOOM_TARGET_MATH_OP_RSQRTF = 26,
  LOOM_TARGET_MATH_OP_EXP2F = 27,
  LOOM_TARGET_MATH_OP_ASINHF = 28,
  LOOM_TARGET_MATH_OP_ACOSHF = 29,
  LOOM_TARGET_MATH_OP_ATANHF = 30,
} loom_target_math_op_t;

// Number of operation ordinals available to operation-indexed policy tables.
#define LOOM_TARGET_MATH_OP_COUNT ((int)LOOM_TARGET_MATH_OP_ATANHF + 1)

typedef enum loom_target_math_lane_domain_e {
  LOOM_TARGET_MATH_LANE_DOMAIN_UNKNOWN = 0,
  LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR = 1,
  LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR = 2,
} loom_target_math_lane_domain_t;

// Number of lane-domain ordinals available to name tables.
#define LOOM_TARGET_MATH_LANE_DOMAIN_COUNT \
  ((int)LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR + 1)

typedef uint8_t loom_target_math_lane_domain_set_t;
enum loom_target_math_lane_domain_set_bits_e {
  LOOM_TARGET_MATH_LANE_DOMAIN_SET_NONE = 0u,
  LOOM_TARGET_MATH_LANE_DOMAIN_SET_SCALAR =
      1u << LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
  LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR =
      1u << LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
  LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL =
      LOOM_TARGET_MATH_LANE_DOMAIN_SET_SCALAR |
      LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
};

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

// Positive source requirements, independent of fast-math permissions. The bit
// layout matches scalar/vector TrigonometryMathFlags in the existing flag byte.
typedef uint8_t loom_target_math_requirement_flags_t;
enum loom_target_math_requirement_flag_bits_e {
  // Preserve the source-defined signs of trigonometric zero results.
  LOOM_TARGET_MATH_REQUIREMENT_FLAG_PRESERVE = 1u << 7,
};

// Mirrors the scalar/vector FloatConversionFlags bit layout so target-selected
// evaluation policies can be projected into generated conversion ops.
typedef enum loom_target_math_float_conversion_flag_bits_e {
  LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_NONE = 0u,
  LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ = 1u << 0,
  LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_FTZ = 1u << 1,
} loom_target_math_float_conversion_flag_bits_t;
typedef uint8_t loom_target_math_float_conversion_flags_t;

typedef uint8_t loom_target_math_policy_action_t;
enum loom_target_math_policy_action_e {
  LOOM_TARGET_MATH_POLICY_ACTION_UNKNOWN = 0,
  LOOM_TARGET_MATH_POLICY_ACTION_KEEP = 1,
  LOOM_TARGET_MATH_POLICY_ACTION_REWRITE = 2,
  LOOM_TARGET_MATH_POLICY_ACTION_REJECT = 3,
};

typedef uint8_t loom_target_math_recipe_t;
enum loom_target_math_recipe_e {
  LOOM_TARGET_MATH_RECIPE_UNKNOWN = 0,
  LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32 = 1,
  LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32 = 2,
  LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32 = 3,
  LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32 = 4,
  LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32 = 5,
  LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32 = 6,
  LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32 = 7,
  LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32 = 8,
  LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND = 9,
  LOOM_TARGET_MATH_RECIPE_LOG_LOG2_F32 = 10,
  LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32 = 11,
  LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32 = 12,
  LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32 = 13,
  LOOM_TARGET_MATH_RECIPE_POW_LOG2_EXP2_F32 = 14,
  LOOM_TARGET_MATH_RECIPE_ROUND_AWAY = 15,
  LOOM_TARGET_MATH_RECIPE_LOGISTIC_TANH_F32 = 16,
  LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV = 17,
  LOOM_TARGET_MATH_RECIPE_APPROXIMATE_EXP2_F32 = 18,
  LOOM_TARGET_MATH_RECIPE_APPROXIMATE_SIN_TURNS_F32 = 19,
  LOOM_TARGET_MATH_RECIPE_APPROXIMATE_COS_TURNS_F32 = 20,
  LOOM_TARGET_MATH_RECIPE_INVERSE_HYPERBOLIC_F32 = 21,
  LOOM_TARGET_MATH_RECIPE_RSQRT_NEWTON_ONE = 22,
  LOOM_TARGET_MATH_RECIPE_RSQRT_NEWTON_TWO = 23,
  LOOM_TARGET_MATH_RECIPE_TRIGONOMETRIC_F32 = 24,
  LOOM_TARGET_MATH_RECIPE_PRESERVE_TRIGONOMETRIC_ZEROS = 25,
};

// Number of recipe ordinals available to recipe-indexed metadata tables.
#define LOOM_TARGET_MATH_RECIPE_COUNT \
  ((int)LOOM_TARGET_MATH_RECIPE_PRESERVE_TRIGONOMETRIC_ZEROS + 1)

// Packed policy-table sentinel selecting REJECT instead of KEEP or REWRITE.
#define LOOM_TARGET_MATH_POLICY_RECIPE_REJECT UINT8_C(31)

// Representation used to evaluate a target-neutral math recipe.
typedef uint8_t loom_target_math_evaluation_kind_t;
enum loom_target_math_evaluation_kind_e {
  // Evaluate every value in the authored scalar or vector representation.
  LOOM_TARGET_MATH_EVALUATION_SOURCE = 0,
  // Evaluate one vector packet with narrow grouped products accumulated into
  // wider lanes. Products use vector.dot2f's two-input grouping.
  LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT = 1,
  // Preserve scalar/vector shape while evaluating in wider element lanes.
  // Operands are extended once and the result is rounded once on export.
  LOOM_TARGET_MATH_EVALUATION_WIDENED = 2,
};

// Subnormal permissions applied at the representation boundaries of a
// target-selected math evaluation.
typedef struct loom_target_math_evaluation_conversion_t {
  // Permissions used when importing authored operands into accumulator lanes.
  loom_target_math_float_conversion_flags_t input_flags;
  // Permissions used when rounding accumulator values into product lanes.
  loom_target_math_float_conversion_flags_t product_flags;
  // Permissions used when restoring accumulator values to the authored type.
  loom_target_math_float_conversion_flags_t result_flags;
} loom_target_math_evaluation_conversion_t;

typedef struct loom_target_math_evaluation_t {
  // Physical accumulator lanes carried by one evaluation packet.
  uint16_t packet_lane_count;
  // Evaluation mechanism selected for the recipe.
  loom_target_math_evaluation_kind_t kind;
  // Element type to which multiplication operands are rounded.
  loom_scalar_type_t product_element_type;
  // Element type used for constants, sums, and nonlinear primitives.
  loom_scalar_type_t accumulator_element_type;
  // Conversion policy for the evaluation's representation boundaries.
  loom_target_math_evaluation_conversion_t conversion;
} loom_target_math_evaluation_t;

typedef struct loom_target_math_query_t {
  // Resolved target bundle owning the policy decision.
  const loom_target_bundle_t* target_bundle;
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
  // Positive requirements retained independently of approximation permissions.
  loom_target_math_requirement_flags_t requirement_flags;
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
  // Source fast-math permissions required to select an available target-native
  // form instead of this decision. Zero means no native alternative is known.
  loom_target_math_fastmath_flags_t native_required_fastmath_flags;
  // Representation used while evaluating the selected recipe. A zero-
  // initialized descriptor preserves the authored source representation.
  loom_target_math_evaluation_t evaluation;
  // One-based policy constraint ordinal for diagnostics, or zero.
  uint8_t constraint;
} loom_target_math_policy_decision_t;

// Interpretation of the lane-count bounds in a policy rule.
typedef uint8_t loom_target_math_policy_shape_t;
enum loom_target_math_policy_shape_e {
  // Match without inspecting the value shape.
  LOOM_TARGET_MATH_POLICY_SHAPE_ANY = 0,
  // Match a scalar as one lane or a static rank-one vector by lane count.
  LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE = 1,
  // Match a scalar as one lane or a static vector by total element count.
  LOOM_TARGET_MATH_POLICY_SHAPE_STATIC_ELEMENTS = 2,
  // Match a static rank-one vector with an even lane count.
  LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE_EVEN = 3,
};

typedef uint8_t loom_target_math_policy_feature_bits_t;

// Query-dependent predicates shared by operation and FMA policy rows.
typedef struct loom_target_math_policy_match_t {
  // Scalar element types accepted by the row.
  loom_scalar_type_set_t element_types;
  // Inclusive lower lane-count bound for shape-aware rows.
  uint8_t minimum_lane_count;
  // Inclusive upper lane-count bound, or zero when unbounded.
  uint8_t maximum_lane_count;
  // Scalar/vector source domains accepted by the row.
  loom_target_math_lane_domain_set_t lane_domains;
  // Interpretation of the lane-count bounds.
  loom_target_math_policy_shape_t shape;
  // Source fast-math permissions required by the row.
  loom_target_math_fastmath_flags_t required_fastmath_flags;
  // One-based policy/target feature-mask class required by the row, or zero.
  uint8_t feature_class;
} loom_target_math_policy_match_t;

// Contiguous rule range selected directly by a math operation ordinal.
typedef struct loom_target_math_policy_rule_span_t {
  // First row in the policy rule table.
  uint16_t rule_start;
  // Number of ordered rows belonging to the operation.
  uint8_t rule_count;
  // One-based constraint ordinal reported when no row matches.
  uint8_t fallback_constraint;
} loom_target_math_policy_rule_span_t;

// Match and result bytes loaded together for the common direct query path.
typedef struct loom_target_math_policy_direct_header_t {
  // Element types occupy bits 0-15, lane domains bits 16-23, and required
  // fast-math permissions bits 24-31.
  uint32_t match;
  // Action selected by the row.
  loom_target_math_policy_action_t action;
  // Rewrite recipe selected by the row, or UNKNOWN.
  loom_target_math_recipe_t recipe;
  // Extra fast-math permissions granted inside the selected recipe.
  loom_target_math_fastmath_flags_t recipe_fastmath_flags;
  // Permissions required to select a known native alternative.
  loom_target_math_fastmath_flags_t native_required_fastmath_flags;
} loom_target_math_policy_direct_header_t;

// Direct operation decision for the common element/lane/permission-only path.
// Rows requiring shape or target-feature matching use ordered policy rules.
typedef struct loom_target_math_policy_direct_rule_t {
  // Match and decision bytes consumed by every successful direct query.
  iree_alignas(16) loom_target_math_policy_direct_header_t header;
  // One-based evaluation descriptor ordinal, or zero for source evaluation.
  uint8_t evaluation;
  // One-based constraint ordinal naming the selected decision.
  uint8_t decision_constraint;
  // Immediate rejection constraint when required permissions are absent.
  uint8_t permission_constraint;
} loom_target_math_policy_direct_rule_t;

#define LOOM_TARGET_MATH_POLICY_DIRECT_RULE(                               \
    element_types_, lane_domains_, required_fastmath_flags_, recipe_,      \
    recipe_fastmath_flags_, native_required_fastmath_flags_, evaluation_,  \
    decision_constraint_, permission_constraint_)                          \
  {                                                                        \
      .header =                                                            \
          {                                                                \
              .match = (uint32_t)(element_types_) |                        \
                       ((uint32_t)(lane_domains_) << 16) |                 \
                       ((uint32_t)(required_fastmath_flags_) << 24),       \
              .action = (recipe_) == LOOM_TARGET_MATH_POLICY_RECIPE_REJECT \
                            ? LOOM_TARGET_MATH_POLICY_ACTION_REJECT        \
                        : (recipe_) == LOOM_TARGET_MATH_RECIPE_UNKNOWN     \
                            ? LOOM_TARGET_MATH_POLICY_ACTION_KEEP          \
                            : LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,      \
              .recipe = (recipe_) == LOOM_TARGET_MATH_POLICY_RECIPE_REJECT \
                            ? LOOM_TARGET_MATH_RECIPE_UNKNOWN              \
                            : (recipe_),                                   \
              .recipe_fastmath_flags = (recipe_fastmath_flags_),           \
              .native_required_fastmath_flags =                            \
                  (native_required_fastmath_flags_),                       \
          },                                                               \
      .evaluation = (evaluation_),                                         \
      .decision_constraint = (decision_constraint_),                       \
      .permission_constraint = (permission_constraint_),                   \
  }

// One ordered target math policy rule. Enum-valued fields use byte storage so
// target tables remain compact; generation and checked-in initializers own
// their validity. A zero recipe keeps the source operation, an ordinary recipe
// rewrites it, and the packed rejection sentinel rejects it.
typedef struct loom_target_math_policy_rule_t {
  // Query predicates selecting the rule.
  loom_target_math_policy_match_t match;
  // Rewrite recipe selected by the rule, or UNKNOWN to keep the operation.
  uint8_t recipe;
  // Extra fast-math permissions granted inside the selected recipe.
  loom_target_math_fastmath_flags_t recipe_fastmath_flags;
  // Permissions required to select a known native alternative.
  loom_target_math_fastmath_flags_t native_required_fastmath_flags;
  // One-based evaluation descriptor ordinal, or zero for source evaluation.
  uint8_t evaluation;
  // One-based constraint ordinal naming the selected decision.
  uint8_t decision_constraint;
  // Constraint selected when the element type does not match, or zero.
  uint8_t element_constraint;
  // Constraint selected when the value shape does not match, or zero.
  uint8_t shape_constraint;
  // Immediate rejection constraint when required permissions are absent. Zero
  // continues to the next ordered rule.
  uint8_t permission_constraint;
} loom_target_math_policy_rule_t;

// Immutable data interpreted by the shared target math policy query.
typedef struct loom_target_math_policy_table_t {
  // Dense operation-ordinal lookup into |rules|.
  const loom_target_math_policy_rule_span_t* operation_spans;
  // Ordered policy rules referenced by |operation_spans|.
  const loom_target_math_policy_rule_t* rules;
  // FMA preference forms scanned in declaration order.
  const loom_target_math_policy_match_t* fma_forms;
  // Number of rows in |fma_forms|.
  uint16_t fma_form_count;
  // One-based constraint ordinal reported for operations without a span.
  uint8_t default_constraint;
  // Source permissions required before ordered-rule and fallback queries.
  loom_target_math_fastmath_flags_t required_fastmath_flags;
  // Rejection constraint used when ordered-rule permissions are absent.
  uint8_t permission_constraint;
  // One-based diagnostic constraint strings referenced by rules and spans.
  const loom_string_ref_t* constraints;
  // Packed target-owned diagnostic constraint strings.
  loom_string_pool_t constraint_strings;
  // One-based evaluation descriptors referenced by rules.
  const loom_target_math_evaluation_t* evaluations;
  // One-based target feature masks referenced by rule matches.
  const uint64_t* target_feature_masks;
  // One-based policy feature masks referenced by rule matches.
  const loom_target_math_policy_feature_bits_t* policy_feature_masks;
} loom_target_math_policy_table_t;

struct loom_target_math_policy_t {
  // Stable policy name used in diagnostics.
  iree_string_view_t name;
  // Dense operation-ordinal decisions for the common query path.
  const loom_target_math_policy_direct_rule_t* direct_rules;
  // Immutable target rule table.
  const loom_target_math_policy_table_t* table;
  // Target-owned feature bits consumed by table rule predicates.
  loom_target_math_policy_feature_bits_t feature_bits;
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
  // Optional caller-owned compile report receiving math legalization rows.
  loom_target_compile_report_t* compile_report;
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

// Resolves a decision's one-based policy constraint ordinal to its stable key.
// Returns an empty view for ordinal zero.
iree_string_view_t loom_target_math_policy_constraint_key(
    const loom_target_math_policy_t* policy, uint8_t constraint);

// Returns true when the policy prefers contracting a multiply/add over
// |value_type| into FMA.
bool loom_target_math_policy_prefers_fma(
    const loom_target_math_policy_t* policy,
    const loom_target_bundle_t* target_bundle, loom_type_t value_type,
    loom_target_math_fastmath_flags_t fastmath_flags);

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
    const loom_target_math_policy_registry_t* policy_registry,
    loom_target_compile_report_t* compile_report);

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

// Returns the optional compile report selected by |capability|, or NULL.
loom_target_compile_report_t* loom_target_math_pass_capability_compile_report(
    const loom_target_math_pass_capability_t* capability);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_MATH_POLICY_H_
