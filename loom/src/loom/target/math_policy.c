// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/math_policy.h"

#include <stddef.h>
#include <string.h>

static_assert(LOOM_TARGET_MATH_OP_COUNT <= UINT8_MAX,
              "math operation ordinals must fit in one byte");
static_assert(LOOM_TARGET_MATH_RECIPE_COUNT <=
                  LOOM_TARGET_MATH_POLICY_RECIPE_REJECT,
              "math recipe ordinals must fit below the packed reject code");
static_assert(sizeof(loom_target_math_policy_match_t) == 8,
              "math policy matches must remain compact");
static_assert(sizeof(loom_target_math_policy_rule_span_t) == 4,
              "math policy rule spans must remain compact");
static_assert(sizeof(loom_target_math_policy_direct_header_t) == 8,
              "direct math policy headers must fit one machine word");
static_assert(offsetof(loom_target_math_policy_direct_header_t, action) == 4 &&
                  offsetof(loom_target_math_policy_direct_header_t, recipe) ==
                      5 &&
                  offsetof(loom_target_math_policy_direct_header_t,
                           recipe_fastmath_flags) == 6 &&
                  offsetof(loom_target_math_policy_direct_header_t,
                           native_required_fastmath_flags) == 7,
              "direct decisions must follow their match word");
static_assert(sizeof(loom_target_math_policy_direct_rule_t) == 16,
              "direct math policy rules must remain compact");
static_assert(offsetof(loom_target_math_policy_direct_rule_t, evaluation) == 8,
              "direct headers must occupy eight bytes");
static_assert(sizeof(loom_target_math_policy_rule_t) == 16,
              "math policy rules must remain compact");
static_assert(sizeof(loom_target_math_evaluation_t) == 8,
              "math evaluations must remain compact");
static_assert(sizeof(loom_target_math_policy_decision_t) == 14,
              "math policy decisions must remain compact");
static_assert(offsetof(loom_target_math_policy_decision_t, action) == 0 &&
                  offsetof(loom_target_math_policy_decision_t, recipe) == 1 &&
                  offsetof(loom_target_math_policy_decision_t,
                           recipe_fastmath_flags) == 2 &&
                  offsetof(loom_target_math_policy_decision_t,
                           native_required_fastmath_flags) == 3,
              "direct decision bytes must match policy decision bytes");

const loom_pass_environment_capability_type_t
    loom_target_math_pass_capability_type = {
        .name = IREE_SVL("target.math"),
};

void loom_target_math_policy_registry_initialize_from_entries(
    loom_target_math_policy_registry_t* out_registry,
    const loom_target_math_policy_registry_entry_t* entries,
    iree_host_size_t entry_count) {
  *out_registry = (loom_target_math_policy_registry_t){
      .entries = entries,
      .entry_count = entry_count,
  };
}

const loom_target_math_policy_t* loom_target_math_policy_registry_lookup(
    const loom_target_math_policy_registry_t* registry,
    iree_string_view_t contract_set_key) {
  if (registry == NULL) {
    return NULL;
  }
  contract_set_key = iree_string_view_trim(contract_set_key);
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_target_math_policy_registry_entry_t* entry =
        &registry->entries[i];
    if (iree_string_view_equal(entry->contract_set_key, contract_set_key)) {
      return entry->policy;
    }
  }
  return NULL;
}

const loom_target_math_policy_t*
loom_target_math_policy_registry_lookup_for_bundle(
    const loom_target_math_policy_registry_t* registry,
    const loom_target_bundle_t* bundle) {
  if (bundle == NULL || bundle->config == NULL) {
    return NULL;
  }
  return loom_target_math_policy_registry_lookup(
      registry, bundle->config->contract_set_key);
}

iree_string_view_t loom_target_math_policy_constraint_key(
    const loom_target_math_policy_t* policy, uint8_t constraint) {
  if (constraint == 0) {
    return iree_string_view_empty();
  }
  const loom_target_math_policy_table_t* table = policy->table;
  return loom_string_pool_get(&table->constraint_strings,
                              table->constraints[constraint - 1]);
}

static bool loom_target_math_policy_match_features(
    const loom_target_math_policy_t* policy,
    const loom_target_bundle_t* target_bundle,
    const loom_target_math_policy_match_t* match) {
  if (match->feature_class == 0) {
    return true;
  }
  const iree_host_size_t feature_index = match->feature_class - 1;
  const loom_target_math_policy_table_t* table = policy->table;
  if (table->policy_feature_masks != NULL &&
      !iree_all_bits_set(policy->feature_bits,
                         table->policy_feature_masks[feature_index])) {
    return false;
  }
  if (table->target_feature_masks != NULL) {
    const uint64_t target_feature_mask =
        table->target_feature_masks[feature_index];
    return target_feature_mask == 0 ||
           (target_bundle != NULL && target_bundle->config != NULL &&
            iree_all_bits_set(target_bundle->config->contract_feature_bits,
                              target_feature_mask));
  }
  return true;
}

static bool loom_target_math_policy_match_shape(
    loom_type_t value_type, loom_target_math_lane_domain_t lane_domain,
    const loom_target_math_policy_match_t* match) {
  if (match->shape == LOOM_TARGET_MATH_POLICY_SHAPE_ANY) {
    return true;
  }
  uint64_t lane_count = 0;
  if (lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR) {
    lane_count = 1;
  } else if (lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR &&
             loom_type_is_vector(value_type) &&
             loom_type_is_all_static(value_type)) {
    if (match->shape == LOOM_TARGET_MATH_POLICY_SHAPE_STATIC_ELEMENTS) {
      if (!loom_type_static_element_count(value_type, &lane_count)) {
        return false;
      }
    } else {
      if (loom_type_rank(value_type) != 1) {
        return false;
      }
      lane_count = (uint64_t)loom_type_dim_static_size_at(value_type, 0);
    }
  } else {
    return false;
  }
  if (match->shape == LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE_EVEN &&
      (lane_count & 1u) != 0) {
    return false;
  }
  return lane_count >= match->minimum_lane_count &&
         (match->maximum_lane_count == 0 ||
          lane_count <= match->maximum_lane_count);
}

typedef enum loom_target_math_policy_special_match_e {
  LOOM_TARGET_MATH_POLICY_SPECIAL_MATCH = 0,
  LOOM_TARGET_MATH_POLICY_SPECIAL_FEATURE_MISMATCH = 1,
  LOOM_TARGET_MATH_POLICY_SPECIAL_SHAPE_MISMATCH = 2,
} loom_target_math_policy_special_match_t;

static IREE_ATTRIBUTE_NOINLINE loom_target_math_policy_special_match_t
loom_target_math_policy_match_special(
    const loom_target_math_policy_t* policy,
    const loom_target_bundle_t* target_bundle, loom_type_t value_type,
    loom_target_math_lane_domain_t lane_domain,
    const loom_target_math_policy_match_t* match) {
  if (match->feature_class != 0 &&
      !loom_target_math_policy_match_features(policy, target_bundle, match)) {
    return LOOM_TARGET_MATH_POLICY_SPECIAL_FEATURE_MISMATCH;
  }
  if (match->shape != LOOM_TARGET_MATH_POLICY_SHAPE_ANY &&
      !loom_target_math_policy_match_shape(value_type, lane_domain, match)) {
    return LOOM_TARGET_MATH_POLICY_SPECIAL_SHAPE_MISMATCH;
  }
  return LOOM_TARGET_MATH_POLICY_SPECIAL_MATCH;
}

static void loom_target_math_policy_decision_from_rule(
    const loom_target_math_policy_table_t* table,
    const loom_target_math_policy_rule_t* rule,
    loom_target_math_policy_decision_t* out_decision) {
  const bool is_rejection =
      rule->recipe == LOOM_TARGET_MATH_POLICY_RECIPE_REJECT;
  *out_decision = (loom_target_math_policy_decision_t){
      .action = is_rejection ? LOOM_TARGET_MATH_POLICY_ACTION_REJECT
                : rule->recipe == LOOM_TARGET_MATH_RECIPE_UNKNOWN
                    ? LOOM_TARGET_MATH_POLICY_ACTION_KEEP
                    : LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
      .recipe = is_rejection ? LOOM_TARGET_MATH_RECIPE_UNKNOWN
                             : (loom_target_math_recipe_t)rule->recipe,
      .recipe_fastmath_flags = rule->recipe_fastmath_flags,
      .native_required_fastmath_flags = rule->native_required_fastmath_flags,
      .constraint = rule->decision_constraint,
  };
  if (rule->evaluation != 0) {
    out_decision->evaluation = table->evaluations[rule->evaluation - 1];
  }
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
loom_target_math_policy_decision_from_direct_rule(
    const loom_target_math_policy_t* policy,
    const loom_target_math_policy_direct_rule_t* rule,
    loom_target_math_policy_direct_header_t header,
    loom_target_math_policy_decision_t* out_decision) {
  *out_decision = (loom_target_math_policy_decision_t){0};
  memcpy(out_decision, &header.action, sizeof(uint32_t));
  out_decision->constraint = rule->decision_constraint;
  if (rule->evaluation != 0) {
    out_decision->evaluation = policy->table->evaluations[rule->evaluation - 1];
  }
}

static IREE_ATTRIBUTE_NOINLINE void loom_target_math_policy_query_rules(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  const loom_target_math_policy_table_t* table = policy->table;
  if (!iree_all_bits_set(query->fastmath_flags,
                         table->required_fastmath_flags)) {
    *out_decision = (loom_target_math_policy_decision_t){
        .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
        .constraint = table->permission_constraint,
    };
    return;
  }
  const loom_scalar_type_set_t query_element_type =
      (loom_scalar_type_set_t)(1u << query->element_type);
  const loom_target_math_lane_domain_set_t query_lane_domain =
      (loom_target_math_lane_domain_set_t)(1u << query->lane_domain);
  const loom_target_math_policy_rule_span_t span =
      table->operation_spans != NULL ? table->operation_spans[query->math_op]
                                     : (loom_target_math_policy_rule_span_t){0};
  uint8_t failure_constraint = span.fallback_constraint != 0
                                   ? span.fallback_constraint
                                   : table->default_constraint;
  bool matched_element_type = false;
  const uint16_t rule_end = (uint16_t)(span.rule_start + span.rule_count);
  for (uint16_t i = span.rule_start; i < rule_end; ++i) {
    const loom_target_math_policy_rule_t* rule = &table->rules[i];
    const loom_target_math_policy_match_t* match = &rule->match;
    if (!iree_any_bit_set(match->lane_domains, query_lane_domain)) {
      continue;
    }
    if (!iree_any_bit_set(match->element_types, query_element_type)) {
      if (!matched_element_type && rule->element_constraint != 0) {
        failure_constraint = rule->element_constraint;
      }
      continue;
    }
    matched_element_type = true;
    const loom_target_math_policy_special_match_t special_match =
        match->shape == LOOM_TARGET_MATH_POLICY_SHAPE_ANY &&
                match->feature_class == 0
            ? LOOM_TARGET_MATH_POLICY_SPECIAL_MATCH
            : loom_target_math_policy_match_special(
                  policy, query->target_bundle, query->value_type,
                  query->lane_domain, match);
    if (special_match != LOOM_TARGET_MATH_POLICY_SPECIAL_MATCH) {
      if (special_match == LOOM_TARGET_MATH_POLICY_SPECIAL_FEATURE_MISMATCH) {
        continue;
      }
      if (rule->shape_constraint != 0) {
        failure_constraint = rule->shape_constraint;
      }
      continue;
    }
    if (!iree_all_bits_set(query->fastmath_flags,
                           match->required_fastmath_flags)) {
      if (rule->permission_constraint != 0) {
        *out_decision = (loom_target_math_policy_decision_t){
            .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
            .constraint = rule->permission_constraint,
        };
        return;
      }
      continue;
    }
    loom_target_math_policy_decision_from_rule(table, rule, out_decision);
    return;
  }

  *out_decision = (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint = failure_constraint,
  };
}

void loom_target_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  if (IREE_LIKELY(policy->direct_rules != NULL)) {
    const loom_target_math_policy_direct_rule_t* direct_rule =
        &policy->direct_rules[query->math_op];
    const loom_target_math_policy_direct_header_t direct_header =
        direct_rule->header;
    const uint32_t direct_match = direct_header.match;
    const loom_scalar_type_set_t query_element_type =
        (loom_scalar_type_set_t)(1u << query->element_type);
    const uint32_t query_lane_domain = 1u << (query->lane_domain + 16u);
    if (iree_any_bit_set(direct_match, query_element_type) &&
        iree_any_bit_set(direct_match, query_lane_domain)) {
      if (iree_all_bits_set(query->fastmath_flags,
                            (uint8_t)(direct_match >> 24))) {
        loom_target_math_policy_decision_from_direct_rule(
            policy, direct_rule, direct_header, out_decision);
        return;
      }
      if (direct_rule->permission_constraint != 0) {
        *out_decision = (loom_target_math_policy_decision_t){
            .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
            .constraint = direct_rule->permission_constraint,
        };
        return;
      }
    }
  }
  loom_target_math_policy_query_rules(policy, query, out_decision);
}

bool loom_target_math_policy_prefers_fma(
    const loom_target_math_policy_t* policy,
    const loom_target_bundle_t* target_bundle, loom_type_t value_type,
    loom_target_math_fastmath_flags_t fastmath_flags) {
  if (policy == NULL) {
    return false;
  }
  const loom_target_math_policy_table_t* table = policy->table;
  const loom_scalar_type_t element_type = loom_type_element_type(value_type);
  const loom_target_math_lane_domain_t lane_domain =
      loom_type_is_scalar(value_type) ? LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR
                                      : LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR;
  for (uint16_t i = 0; i < table->fma_form_count; ++i) {
    const loom_target_math_policy_match_t* match = &table->fma_forms[i];
    if (iree_any_bit_set(match->element_types,
                         (loom_scalar_type_set_t)(1u << element_type)) &&
        iree_any_bit_set(
            match->lane_domains,
            (loom_target_math_lane_domain_set_t)(1u << lane_domain)) &&
        ((match->shape == LOOM_TARGET_MATH_POLICY_SHAPE_ANY &&
          match->feature_class == 0) ||
         loom_target_math_policy_match_special(
             policy, target_bundle, value_type, lane_domain, match) ==
             LOOM_TARGET_MATH_POLICY_SPECIAL_MATCH) &&
        iree_all_bits_set(fastmath_flags, match->required_fastmath_flags)) {
      return true;
    }
  }
  return false;
}

#define LOOM_TARGET_MATH_OP_NAMES(_)                     \
  _(UNKNOWN, op_unknown, "unknown")                      \
  _(LOGISTICF, op_logisticf, "logisticf")                \
  _(SILUF, op_siluf, "siluf")                            \
  _(SOFTPLUSF, op_softplusf, "softplusf")                \
  _(EXPF, op_expf, "expf")                               \
  _(ERFF, op_erff, "erff")                               \
  _(GELUF_ERF, op_geluf_erf, "geluf.erf")                \
  _(GELUF_TANH, op_geluf_tanh, "geluf.tanh")             \
  _(GELUF_LOGISTIC, op_geluf_logistic, "geluf.logistic") \
  _(ADDF, op_addf, "addf")                               \
  _(MULF, op_mulf, "mulf")                               \
  _(LOGF, op_logf, "logf")                               \
  _(LOG2F, op_log2f, "log2f")                            \
  _(SINF, op_sinf, "sinf")                               \
  _(COSF, op_cosf, "cosf")                               \
  _(SINTURNSF, op_sinturnsf, "sinturnsf")                \
  _(COSTURNSF, op_costurnsf, "costurnsf")                \
  _(TANHF, op_tanhf, "tanhf")                            \
  _(POWF, op_powf, "powf")                               \
  _(CEILF, op_ceilf, "ceilf")                            \
  _(FLOORF, op_floorf, "floorf")                         \
  _(ROUNDF, op_roundf, "roundf")                         \
  _(ROUNDEVENF, op_roundevenf, "roundevenf")             \
  _(TRUNCF, op_truncf, "truncf")                         \
  _(SUBF, op_subf, "subf")                               \
  _(SQRTF, op_sqrtf, "sqrtf")                            \
  _(RSQRTF, op_rsqrtf, "rsqrtf")                         \
  _(EXP2F, op_exp2f, "exp2f")                            \
  _(ASINHF, op_asinhf, "asinhf")                         \
  _(ACOSHF, op_acoshf, "acoshf")                         \
  _(ATANHF, op_atanhf, "atanhf")

#define LOOM_TARGET_MATH_LANE_DOMAIN_NAMES(_) \
  _(UNKNOWN, lane_unknown, "unknown")         \
  _(SCALAR, lane_scalar, "scalar")            \
  _(VECTOR, lane_vector, "vector")

#define LOOM_TARGET_MATH_RECIPE_NAMES(_)                                       \
  _(UNKNOWN, recipe_unknown, "unknown")                                        \
  _(LOGISTIC_EXP2_F32, recipe_logistic_exp2_f32, "logistic-exp2-f32")          \
  _(SILU_LOGISTIC_F32, recipe_silu_logistic_f32, "silu-logistic-f32")          \
  _(SOFTPLUS_EXP2_F32, recipe_softplus_exp2_f32, "softplus-exp2-f32")          \
  _(EXP_EXP2_F32, recipe_exp_exp2_f32, "exp-exp2-f32")                         \
  _(GELU_TANH_F32, recipe_gelu_tanh_f32, "gelu-tanh-f32")                      \
  _(GELU_LOGISTIC_F32, recipe_gelu_logistic_f32, "gelu-logistic-f32")          \
  _(ERF_RATIONAL_F32, recipe_erf_rational_f32, "erf-rational-f32")             \
  _(GELU_ERF_F32, recipe_gelu_erf_f32, "gelu-erf-f32")                         \
  _(WIDEN_F32_ROUND, recipe_widen_f32_round, "widen-f32-round")                \
  _(LOG_LOG2_F32, recipe_log_log2_f32, "log-log2-f32")                         \
  _(SIN_TURNS_F32, recipe_sin_turns_f32, "sin-turns-f32")                      \
  _(COS_TURNS_F32, recipe_cos_turns_f32, "cos-turns-f32")                      \
  _(TANH_LOGISTIC_F32, recipe_tanh_logistic_f32, "tanh-logistic-f32")          \
  _(POW_LOG2_EXP2_F32, recipe_pow_log2_exp2_f32, "pow-log2-exp2-f32")          \
  _(ROUND_AWAY, recipe_round_away, "round-away")                               \
  _(LOGISTIC_TANH_F32, recipe_logistic_tanh_f32, "logistic-tanh-f32")          \
  _(RSQRT_SQRT_DIV, recipe_rsqrt_sqrt_div, "rsqrt-sqrt-div")                   \
  _(APPROXIMATE_EXP2_F32, recipe_approximate_exp2_f32, "approximate-exp2-f32") \
  _(APPROXIMATE_SIN_TURNS_F32, recipe_approximate_sin_turns_f32,               \
    "approximate-sin-turns-f32")                                               \
  _(APPROXIMATE_COS_TURNS_F32, recipe_approximate_cos_turns_f32,               \
    "approximate-cos-turns-f32")                                               \
  _(INVERSE_HYPERBOLIC_F32, recipe_inverse_hyperbolic_f32,                     \
    "inverse-hyperbolic-f32")                                                  \
  _(RSQRT_NEWTON_ONE, recipe_rsqrt_newton_one, "rsqrt-newton-one")             \
  _(RSQRT_NEWTON_TWO, recipe_rsqrt_newton_two, "rsqrt-newton-two")             \
  _(TRIGONOMETRIC_F32, recipe_trigonometric_f32, "trigonometric-f32")

typedef struct loom_target_math_name_strings_t {
#define LOOM_TARGET_MATH_NAME_FIELD(name, field, value) \
  char field[sizeof(value)];
  LOOM_TARGET_MATH_OP_NAMES(LOOM_TARGET_MATH_NAME_FIELD)
  LOOM_TARGET_MATH_LANE_DOMAIN_NAMES(LOOM_TARGET_MATH_NAME_FIELD)
  LOOM_TARGET_MATH_RECIPE_NAMES(LOOM_TARGET_MATH_NAME_FIELD)
#undef LOOM_TARGET_MATH_NAME_FIELD
} loom_target_math_name_strings_t;

static const loom_target_math_name_strings_t kMathNameStrings = {
#define LOOM_TARGET_MATH_NAME_VALUE(name, field, value) .field = value,
    LOOM_TARGET_MATH_OP_NAMES(LOOM_TARGET_MATH_NAME_VALUE)
        LOOM_TARGET_MATH_LANE_DOMAIN_NAMES(LOOM_TARGET_MATH_NAME_VALUE)
            LOOM_TARGET_MATH_RECIPE_NAMES(LOOM_TARGET_MATH_NAME_VALUE)
#undef LOOM_TARGET_MATH_NAME_VALUE
};

#define LOOM_TARGET_MATH_NAME_REF(name, field, value)                  \
  LOOM_STRING_REF(0, offsetof(loom_target_math_name_strings_t, field), \
                  sizeof(value) - 1)

static const loom_string_ref_t kMathOpNames[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_TARGET_MATH_OP_NAME_REF(name, field, value) \
  [LOOM_TARGET_MATH_OP_##name] = LOOM_TARGET_MATH_NAME_REF(name, field, value),
    LOOM_TARGET_MATH_OP_NAMES(LOOM_TARGET_MATH_OP_NAME_REF)
#undef LOOM_TARGET_MATH_OP_NAME_REF
};

static const loom_string_ref_t
    kMathLaneDomainNames[LOOM_TARGET_MATH_LANE_DOMAIN_COUNT] = {
#define LOOM_TARGET_MATH_LANE_DOMAIN_NAME_REF(name, field, value) \
  [LOOM_TARGET_MATH_LANE_DOMAIN_##name] =                         \
      LOOM_TARGET_MATH_NAME_REF(name, field, value),
        LOOM_TARGET_MATH_LANE_DOMAIN_NAMES(
            LOOM_TARGET_MATH_LANE_DOMAIN_NAME_REF)
#undef LOOM_TARGET_MATH_LANE_DOMAIN_NAME_REF
};

static const loom_string_ref_t kMathRecipeNames[LOOM_TARGET_MATH_RECIPE_COUNT] =
    {
#define LOOM_TARGET_MATH_RECIPE_NAME_REF(name, field, value) \
  [LOOM_TARGET_MATH_RECIPE_##name] =                         \
      LOOM_TARGET_MATH_NAME_REF(name, field, value),
        LOOM_TARGET_MATH_RECIPE_NAMES(LOOM_TARGET_MATH_RECIPE_NAME_REF)
#undef LOOM_TARGET_MATH_RECIPE_NAME_REF
};

#undef LOOM_TARGET_MATH_NAME_REF
#undef LOOM_TARGET_MATH_RECIPE_NAMES
#undef LOOM_TARGET_MATH_LANE_DOMAIN_NAMES
#undef LOOM_TARGET_MATH_OP_NAMES

static const loom_string_pool_t kMathNameStringPool = {
    .data = {(const char*)&kMathNameStrings},
    .data_length = {sizeof(kMathNameStrings)},
};

iree_string_view_t loom_target_math_op_name(loom_target_math_op_t math_op) {
  const loom_target_math_op_t ordinal = math_op < LOOM_TARGET_MATH_OP_COUNT
                                            ? math_op
                                            : LOOM_TARGET_MATH_OP_UNKNOWN;
  return loom_string_pool_get(&kMathNameStringPool, kMathOpNames[ordinal]);
}

iree_string_view_t loom_target_math_lane_domain_name(
    loom_target_math_lane_domain_t domain) {
  const loom_target_math_lane_domain_t ordinal =
      domain < LOOM_TARGET_MATH_LANE_DOMAIN_COUNT
          ? domain
          : LOOM_TARGET_MATH_LANE_DOMAIN_UNKNOWN;
  return loom_string_pool_get(&kMathNameStringPool,
                              kMathLaneDomainNames[ordinal]);
}

iree_string_view_t loom_target_math_recipe_name(
    loom_target_math_recipe_t recipe) {
  const loom_target_math_recipe_t ordinal =
      recipe < LOOM_TARGET_MATH_RECIPE_COUNT ? recipe
                                             : LOOM_TARGET_MATH_RECIPE_UNKNOWN;
  return loom_string_pool_get(&kMathNameStringPool, kMathRecipeNames[ordinal]);
}

loom_target_math_pass_capability_t loom_target_math_pass_capability_make(
    const loom_target_math_policy_registry_t* policy_registry,
    loom_target_compile_report_t* compile_report) {
  return (loom_target_math_pass_capability_t){
      .base =
          {
              .type = &loom_target_math_pass_capability_type,
          },
      .policy_registry = policy_registry,
      .compile_report = compile_report,
  };
}

const loom_target_math_pass_capability_t*
loom_target_math_pass_capability_from_environment(
    const loom_pass_environment_t* environment) {
  return (const loom_target_math_pass_capability_t*)
      loom_pass_environment_lookup(environment,
                                   &loom_target_math_pass_capability_type);
}

const loom_target_math_pass_capability_t*
loom_target_math_pass_capability_from_pass(const loom_pass_t* pass) {
  return pass && pass->environment
             ? loom_target_math_pass_capability_from_environment(
                   pass->environment)
             : NULL;
}

const loom_target_math_policy_registry_t*
loom_target_math_pass_capability_policy_registry(
    const loom_target_math_pass_capability_t* capability) {
  return capability ? capability->policy_registry : NULL;
}

loom_target_compile_report_t* loom_target_math_pass_capability_compile_report(
    const loom_target_math_pass_capability_t* capability) {
  return capability ? capability->compile_report : NULL;
}
