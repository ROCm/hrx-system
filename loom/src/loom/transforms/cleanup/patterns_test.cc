// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/patterns.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/environment.h"
#include "loom/transforms/cleanup/configured.h"
#include "loom/transforms/cleanup/pass_environment.h"
#include "loom/transforms/cleanup/pass_requirements.h"

namespace loom {
namespace {

static iree_status_t IgnorePattern(const loom_rewrite_pattern_t*, void*,
                                   loom_op_t*, loom_rewriter_t*,
                                   bool* out_changed) {
  *out_changed = false;
  return iree_ok_status();
}

typedef struct CanonicalizerContextResolverState {
  // Target facts returned by the test resolver.
  const loom_target_facts_t* target_facts;
  // Math policy returned by the test resolver.
  const loom_target_math_policy_t* math_policy;
  // Pass received by the test resolver.
  const loom_pass_t* pass;
  // True when the resolver was invoked.
  bool called;
} CanonicalizerContextResolverState;

static iree_status_t ResolveCanonicalizerContext(
    void* user_data, const loom_pass_t* pass, const loom_module_t*,
    loom_func_like_t, loom_cleanup_canonicalizer_context_t* out_context) {
  CanonicalizerContextResolverState* state =
      static_cast<CanonicalizerContextResolverState*>(user_data);
  state->pass = pass;
  state->called = true;
  out_context->target_facts = state->target_facts;
  out_context->math_policy = state->math_policy;
  return iree_ok_status();
}

TEST(CleanupPatternsTest, KeepsPhaseRegistriesSeparate) {
  const loom_op_kind_t region_initialization_kind =
      LOOM_OP_KIND(LOOM_DIALECT_SCF, 0);
  const loom_op_kind_t pre_fold_kind = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 1);
  const loom_op_kind_t post_type_kind = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 2);
  const loom_op_kind_t source_combine_kind =
      LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 3);
  const loom_rewrite_pattern_t region_initialization_patterns[] = {
      {region_initialization_kind, IgnorePattern, nullptr},
  };
  const loom_rewrite_pattern_t pre_fold_patterns[] = {
      {pre_fold_kind, IgnorePattern, nullptr},
  };
  const loom_rewrite_pattern_t post_type_patterns[] = {
      {post_type_kind, IgnorePattern, nullptr},
  };
  const loom_rewrite_pattern_t source_combine_patterns[] = {
      {source_combine_kind, IgnorePattern, nullptr},
  };
  const loom_rewrite_pattern_provider_t region_initialization_provider = {
      .name = IREE_SVL("region-initialization"),
      .patterns = region_initialization_patterns,
      .pattern_count = IREE_ARRAYSIZE(region_initialization_patterns),
  };
  const loom_rewrite_pattern_provider_t pre_fold_provider = {
      .name = IREE_SVL("pre-fold"),
      .patterns = pre_fold_patterns,
      .pattern_count = IREE_ARRAYSIZE(pre_fold_patterns),
  };
  const loom_rewrite_pattern_provider_t post_type_provider = {
      .name = IREE_SVL("post-type"),
      .patterns = post_type_patterns,
      .pattern_count = IREE_ARRAYSIZE(post_type_patterns),
  };
  const loom_rewrite_pattern_provider_t source_combine_provider = {
      .name = IREE_SVL("source-combine"),
      .patterns = source_combine_patterns,
      .pattern_count = IREE_ARRAYSIZE(source_combine_patterns),
  };
  const loom_rewrite_pattern_provider_t* region_initialization_providers[] = {
      &region_initialization_provider,
  };
  const loom_rewrite_pattern_provider_t* pre_fold_providers[] = {
      &pre_fold_provider,
  };
  const loom_rewrite_pattern_provider_t* post_type_providers[] = {
      &post_type_provider,
  };
  const loom_rewrite_pattern_provider_t* source_combine_providers[] = {
      &source_combine_provider,
  };
  const loom_cleanup_pattern_provider_set_t provider_set = {
      .region_initialization = loom_rewrite_pattern_provider_list_make(
          region_initialization_providers,
          IREE_ARRAYSIZE(region_initialization_providers)),
      .universal_pre_fold = loom_rewrite_pattern_provider_list_make(
          pre_fold_providers, IREE_ARRAYSIZE(pre_fold_providers)),
      .universal_post_type = loom_rewrite_pattern_provider_list_make(
          post_type_providers, IREE_ARRAYSIZE(post_type_providers)),
      .source_combine = loom_rewrite_pattern_provider_list_make(
          source_combine_providers, IREE_ARRAYSIZE(source_combine_providers)),
      .special_value_policy = nullptr,
      .fact_refinement_policy = nullptr,
  };

  loom_cleanup_pattern_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_cleanup_pattern_registry_storage_initialize(
      &provider_set, iree_allocator_system(), &storage));
  const loom_cleanup_pattern_registry_t* registry =
      loom_cleanup_pattern_registry_storage_registry(&storage);

  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry->region_initialization, region_initialization_kind)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry->region_initialization, pre_fold_kind)
                .count,
            0u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry->universal_pre_fold, pre_fold_kind)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry->universal_pre_fold, post_type_kind)
                .count,
            0u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry->universal_post_type, post_type_kind)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(registry->source_combine,
                                                      source_combine_kind)
                .count,
            1u);

  const loom_cleanup_pass_capability_t capability =
      loom_cleanup_pass_capability_make(
          registry, (loom_cleanup_canonicalizer_context_resolver_t){});
  const loom_pass_environment_capability_t* capabilities[] = {
      &capability.base,
  };
  const loom_pass_environment_t environment =
      loom_pass_environment_make(capabilities, IREE_ARRAYSIZE(capabilities));
  EXPECT_EQ(loom_cleanup_pass_capability_pattern_registry(
                loom_cleanup_pass_capability_from_environment(&environment)),
            registry);
  EXPECT_TRUE(loom_pass_environment_capability_satisfies_requirement(
      &capability.base,
      IREE_SV(LOOM_CLEANUP_PASS_REQUIREMENT_SOURCE_COMBINE_PATTERNS)));

  loom_cleanup_pattern_registry_storage_deinitialize(&storage);
}

TEST(CleanupPatternsTest, ExplicitEmptyProviderSetSatisfiesComposition) {
  const loom_cleanup_pattern_provider_set_t provider_set = {};
  loom_cleanup_pattern_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_cleanup_pattern_registry_storage_initialize(
      &provider_set, iree_allocator_system(), &storage));
  const loom_cleanup_pattern_registry_t* registry =
      loom_cleanup_pattern_registry_storage_registry(&storage);
  ASSERT_NE(registry->region_initialization, nullptr);
  EXPECT_EQ(registry->region_initialization->pattern_count, 0u);
  ASSERT_NE(registry->source_combine, nullptr);
  EXPECT_EQ(registry->source_combine->pattern_count, 0u);
  EXPECT_EQ(registry->special_value_policy, nullptr);
  EXPECT_EQ(registry->fact_refinement_policy, nullptr);

  const loom_cleanup_pass_capability_t capability =
      loom_cleanup_pass_capability_make(
          registry, (loom_cleanup_canonicalizer_context_resolver_t){});
  EXPECT_TRUE(loom_pass_environment_capability_satisfies_requirement(
      &capability.base,
      IREE_SV(LOOM_CLEANUP_PASS_REQUIREMENT_SOURCE_COMBINE_PATTERNS)));
  const loom_cleanup_pass_capability_t missing_capability =
      loom_cleanup_pass_capability_make(
          nullptr, (loom_cleanup_canonicalizer_context_resolver_t){});
  EXPECT_FALSE(loom_pass_environment_capability_satisfies_requirement(
      &missing_capability.base,
      IREE_SV(LOOM_CLEANUP_PASS_REQUIREMENT_SOURCE_COMBINE_PATTERNS)));

  loom_cleanup_pattern_registry_storage_deinitialize(&storage);
}

TEST(CleanupPatternsTest, ResolvesCanonicalizerContextThroughCapability) {
  CanonicalizerContextResolverState state = {};
  state.target_facts =
      reinterpret_cast<const loom_target_facts_t*>(&state.target_facts);
  state.math_policy =
      reinterpret_cast<const loom_target_math_policy_t*>(&state.math_policy);
  const loom_cleanup_pass_capability_t capability =
      loom_cleanup_pass_capability_make(
          /*pattern_registry=*/nullptr,
          (loom_cleanup_canonicalizer_context_resolver_t){
              .fn = ResolveCanonicalizerContext,
              .user_data = &state,
          });
  const loom_pass_t pass = {};
  loom_cleanup_canonicalizer_context_t context = {};
  IREE_EXPECT_OK(loom_cleanup_pass_capability_resolve_canonicalizer_context(
      &capability, &pass, /*module=*/nullptr, /*function=*/{}, &context));
  EXPECT_TRUE(state.called);
  EXPECT_EQ(state.pass, &pass);
  EXPECT_EQ(context.target_facts, state.target_facts);
  EXPECT_EQ(context.math_policy, state.math_policy);

  context.target_facts = state.target_facts;
  context.math_policy = state.math_policy;
  IREE_EXPECT_OK(loom_cleanup_pass_capability_resolve_canonicalizer_context(
      /*capability=*/nullptr, &pass, /*module=*/nullptr, /*function=*/{},
      &context));
  EXPECT_EQ(context.target_facts, nullptr);
  EXPECT_EQ(context.math_policy, nullptr);
}

TEST(CleanupPatternsTest, ConfiguredProvidersCoverOwnedRoots) {
  const loom_cleanup_pattern_provider_set_t* provider_set =
      loom_cleanup_configured_pattern_provider_set();
  EXPECT_EQ(provider_set->region_initialization.count, 1u);
  EXPECT_EQ(provider_set->universal_pre_fold.count, 2u);
  EXPECT_EQ(provider_set->universal_post_type.count, 2u);
  EXPECT_EQ(provider_set->source_combine.count, 5u);
  EXPECT_NE(provider_set->special_value_policy, nullptr);
  ASSERT_NE(provider_set->fact_refinement_policy, nullptr);
  EXPECT_EQ(provider_set->fact_refinement_policy->value_providers.count, 2u);
  EXPECT_EQ(provider_set->fact_refinement_policy->carrier_providers.count, 1u);

  loom_cleanup_pattern_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_cleanup_pattern_registry_storage_initialize(
      provider_set, iree_allocator_system(), &storage));
  const loom_cleanup_pattern_registry_t* registries =
      loom_cleanup_pattern_registry_storage_registry(&storage);
  EXPECT_EQ(registries->special_value_policy,
            provider_set->special_value_policy);
  EXPECT_EQ(registries->fact_refinement_policy,
            provider_set->fact_refinement_policy);
  const loom_rewrite_pattern_registry_t* region_initialization =
      registries->region_initialization;
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(region_initialization,
                                                      LOOM_OP_SCF_IF)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(region_initialization,
                                                      LOOM_OP_SCF_SWITCH)
                .count,
            1u);
  EXPECT_EQ(region_initialization->pattern_count, 2u);
  const loom_rewrite_pattern_registry_t* pre_fold =
      registries->universal_pre_fold;
  EXPECT_EQ(
      loom_rewrite_pattern_registry_lookup_kind(pre_fold, LOOM_OP_SCF_IF).count,
      1u);
  EXPECT_EQ(
      loom_rewrite_pattern_registry_lookup_kind(pre_fold, LOOM_OP_SCF_SWITCH)
          .count,
      1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(pre_fold,
                                                      LOOM_OP_VECTOR_EXTRACT)
                .count,
            1u);
  EXPECT_EQ(
      loom_rewrite_pattern_registry_lookup_kind(pre_fold, LOOM_OP_VECTOR_REDUCE)
          .count,
      1u);
  EXPECT_EQ(
      loom_rewrite_pattern_registry_lookup_kind(pre_fold, LOOM_OP_VECTOR_DOTF)
          .count,
      1u);
  EXPECT_EQ(pre_fold->pattern_count, 5u);

  const loom_rewrite_pattern_registry_t* post_type =
      registries->universal_post_type;
  EXPECT_EQ(
      loom_rewrite_pattern_registry_lookup_kind(post_type, LOOM_OP_INDEX_SUB)
          .count,
      1u);
  EXPECT_EQ(
      loom_rewrite_pattern_registry_lookup_kind(post_type, LOOM_OP_INDEX_CMP)
          .count,
      1u);
  EXPECT_EQ(
      loom_rewrite_pattern_registry_lookup_kind(post_type, LOOM_OP_SCALAR_CMPI)
          .count,
      1u);
  EXPECT_EQ(post_type->pattern_count, 3u);

  const loom_rewrite_pattern_registry_t* source_combine =
      registries->source_combine;
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_INDEX_CAST)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_SCALAR_EXTF)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_SCALAR_FPTRUNC)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_SCALAR_EXTSI)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_SCALAR_EXTUI)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_SCALAR_TRUNCI)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_SCF_SELECT)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_VIEW_LOAD)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                source_combine, LOOM_OP_VECTOR_FROM_ELEMENTS)
                .count,
            2u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                source_combine, LOOM_OP_VECTOR_TABLE_LOOKUP)
                .count,
            2u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_VECTOR_EXTF)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_VECTOR_FPTRUNC)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_VECTOR_EXTSI)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_VECTOR_EXTUI)
                .count,
            1u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(source_combine,
                                                      LOOM_OP_VECTOR_TRUNCI)
                .count,
            1u);
  loom_cleanup_pattern_registry_storage_deinitialize(&storage);
}

}  // namespace
}  // namespace loom
