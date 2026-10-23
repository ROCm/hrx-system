// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/configured.h"

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/special_values.h"
#include "loom/ops/view/ops.h"
#include "loom/transforms/cleanup/fact_refinement_policy.h"
#include "loom/transforms/cleanup/special_value_policy.h"
#include "loom/transforms/index/cleanup_patterns.h"
#include "loom/transforms/index/combine_patterns.h"
#include "loom/transforms/scalar/cleanup_patterns.h"
#include "loom/transforms/scalar/combine_patterns.h"
#include "loom/transforms/scf/branch_fact_patterns.h"
#include "loom/transforms/scf/combine_patterns.h"
#include "loom/transforms/vector/cleanup_patterns.h"
#include "loom/transforms/vector/combine_patterns.h"
#include "loom/transforms/view/combine_patterns.h"

static const loom_cleanup_special_value_policy_t kConfiguredSpecialValuePolicy =
    {
        .type_has_poison_materializer = loom_type_has_poison_materializer,
        .materialize_poison = loom_poison_build,
        .op_is_empty = loom_op_is_empty,
        .type_has_empty_materializer = loom_type_has_empty_materializer,
        .materialize_empty = loom_empty_build,
        .can_materialize_constant = loom_value_facts_can_materialize_constant,
        .materialize_constant = loom_constant_build,
};

static bool loom_configured_index_refinement_supports(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static iree_status_t loom_configured_materialize_index_refinement(
    loom_builder_t* builder, loom_value_id_t source,
    const loom_predicate_t* predicates, iree_host_size_t predicate_count,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(builder, &source, 1, predicates,
                                               predicate_count, &result_type, 1,
                                               location, &op));
  *out_value = loom_op_const_results(op)[0];
  return iree_ok_status();
}

static const loom_fact_refinement_value_provider_t
    kConfiguredIndexRefinementProvider = {
        .supports = loom_configured_index_refinement_supports,
        .materialize = loom_configured_materialize_index_refinement,
};

static bool loom_configured_scalar_refinement_supports(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type != LOOM_SCALAR_TYPE_INDEX &&
         scalar_type != LOOM_SCALAR_TYPE_OFFSET;
}

static iree_status_t loom_configured_materialize_scalar_refinement(
    loom_builder_t* builder, loom_value_id_t source,
    const loom_predicate_t* predicates, iree_host_size_t predicate_count,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_assume_build(builder, &source, 1, predicates,
                                                predicate_count, &result_type,
                                                1, location, &op));
  *out_value = loom_op_const_results(op)[0];
  return iree_ok_status();
}

static const loom_fact_refinement_value_provider_t
    kConfiguredScalarRefinementProvider = {
        .supports = loom_configured_scalar_refinement_supports,
        .materialize = loom_configured_materialize_scalar_refinement,
};

static bool loom_configured_view_refinement_supports(loom_type_t source_type,
                                                     loom_type_t target_type) {
  return loom_type_is_view(source_type) && loom_type_is_view(target_type) &&
         loom_type_rank_equals(source_type, target_type);
}

static iree_status_t loom_configured_materialize_view_refinement(
    loom_builder_t* builder, loom_value_id_t source, loom_type_t target_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_view_refine_build(builder, source, target_type, location, &op));
  *out_value = loom_view_refine_result(op);
  return iree_ok_status();
}

static const loom_fact_refinement_carrier_provider_t
    kConfiguredViewRefinementProvider = {
        .supports = loom_configured_view_refinement_supports,
        .materialize = loom_configured_materialize_view_refinement,
};

static const loom_fact_refinement_value_provider_t* const
    kConfiguredValueRefinementProviders[] = {
        &kConfiguredIndexRefinementProvider,
        &kConfiguredScalarRefinementProvider,
};

static const loom_fact_refinement_carrier_provider_t* const
    kConfiguredCarrierRefinementProviders[] = {
        &kConfiguredViewRefinementProvider,
};

static const loom_fact_refinement_policy_t kConfiguredFactRefinementPolicy = {
    .value_providers =
        {
            .values = kConfiguredValueRefinementProviders,
            .count = IREE_ARRAYSIZE(kConfiguredValueRefinementProviders),
        },
    .carrier_providers =
        {
            .values = kConfiguredCarrierRefinementProviders,
            .count = IREE_ARRAYSIZE(kConfiguredCarrierRefinementProviders),
        },
};

static const loom_rewrite_pattern_provider_t* const
    kConfiguredRegionInitializationPatternProviders[] = {
        &loom_scf_branch_fact_pattern_provider,
};

static const loom_rewrite_pattern_provider_t* const
    kConfiguredUniversalPreFoldPatternProviders[] = {
        &loom_scf_branch_fact_pattern_provider,
        &loom_vector_universal_pre_fold_pattern_provider,
};

static const loom_rewrite_pattern_provider_t* const
    kConfiguredUniversalPostTypePatternProviders[] = {
        &loom_index_universal_post_type_pattern_provider,
        &loom_scalar_universal_post_type_pattern_provider,
};

static const loom_rewrite_pattern_provider_t* const
    kConfiguredSourceCombinePatternProviders[] = {
        &loom_index_source_combine_pattern_provider,
        &loom_scalar_source_combine_pattern_provider,
        &loom_scf_source_combine_pattern_provider,
        &loom_view_source_combine_pattern_provider,
        &loom_vector_source_combine_pattern_provider,
};

static const loom_rewrite_pattern_provider_t* const
    kConfiguredFactIndependentPatternProviders[] = {
        &loom_vector_fact_independent_pattern_provider,
};

static const loom_cleanup_pattern_provider_set_t kConfiguredPatternProviders = {
    .fact_independent =
        {
            .count = IREE_ARRAYSIZE(kConfiguredFactIndependentPatternProviders),
            .values = kConfiguredFactIndependentPatternProviders,
        },
    .region_initialization =
        {
            .count =
                IREE_ARRAYSIZE(kConfiguredRegionInitializationPatternProviders),
            .values = kConfiguredRegionInitializationPatternProviders,
        },
    .universal_pre_fold =
        {
            .count =
                IREE_ARRAYSIZE(kConfiguredUniversalPreFoldPatternProviders),
            .values = kConfiguredUniversalPreFoldPatternProviders,
        },
    .universal_post_type =
        {
            .count =
                IREE_ARRAYSIZE(kConfiguredUniversalPostTypePatternProviders),
            .values = kConfiguredUniversalPostTypePatternProviders,
        },
    .source_combine =
        {
            .count = IREE_ARRAYSIZE(kConfiguredSourceCombinePatternProviders),
            .values = kConfiguredSourceCombinePatternProviders,
        },
    .special_value_policy = &kConfiguredSpecialValuePolicy,
    .fact_refinement_policy = &kConfiguredFactRefinementPolicy,
};

const loom_cleanup_pattern_provider_set_t*
loom_cleanup_configured_pattern_provider_set(void) {
  return &kConfiguredPatternProviders;
}
