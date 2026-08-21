// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/provider_resolver.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/link/module_index.h"

namespace loom {
namespace {

TEST(ProviderResolverTest, EmptyResolverHasNoBindings) {
  loom_link_provider_resolver_t resolver = {0};
  IREE_ASSERT_OK(loom_link_provider_resolver_prepare(
      /*provider_count=*/0, /*bindings=*/nullptr, /*binding_count=*/0,
      &resolver));
  EXPECT_EQ(resolver.bindings, nullptr);
  EXPECT_EQ(resolver.binding_count, 0u);
  EXPECT_EQ(loom_link_provider_resolver_lookup(&resolver, IREE_SV("absent")),
            LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
}

TEST(ProviderResolverTest, SortsAndResolvesBindings) {
  loom_link_provider_binding_t bindings[] = {
      {IREE_SV("zeta"), 2},
      {IREE_SV("alpha"), 0},
      {IREE_SV("middle"), 1},
  };
  loom_link_provider_resolver_t resolver = {0};
  IREE_ASSERT_OK(loom_link_provider_resolver_prepare(
      /*provider_count=*/3, bindings, IREE_ARRAYSIZE(bindings), &resolver));

  EXPECT_TRUE(
      iree_string_view_equal(resolver.bindings[0].key, IREE_SV("alpha")));
  EXPECT_TRUE(
      iree_string_view_equal(resolver.bindings[1].key, IREE_SV("middle")));
  EXPECT_TRUE(
      iree_string_view_equal(resolver.bindings[2].key, IREE_SV("zeta")));
  EXPECT_EQ(loom_link_provider_resolver_lookup(&resolver, IREE_SV("alpha")),
            0u);
  EXPECT_EQ(loom_link_provider_resolver_lookup(&resolver, IREE_SV("middle")),
            1u);
  EXPECT_EQ(loom_link_provider_resolver_lookup(&resolver, IREE_SV("zeta")), 2u);
  EXPECT_EQ(loom_link_provider_resolver_lookup(&resolver, IREE_SV("absent")),
            LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
}

TEST(ProviderResolverTest, RejectsDuplicateKeys) {
  loom_link_provider_binding_t bindings[] = {
      {IREE_SV("provider"), 0},
      {IREE_SV("provider"), 0},
  };
  loom_link_provider_resolver_t resolver = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_link_provider_resolver_prepare(
          /*provider_count=*/1, bindings, IREE_ARRAYSIZE(bindings), &resolver));
}

TEST(ProviderResolverTest, RejectsInvalidBindings) {
  loom_link_provider_resolver_t resolver = {0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_link_provider_resolver_prepare(
                            /*provider_count=*/1, /*bindings=*/nullptr,
                            /*binding_count=*/1, &resolver));

  loom_link_provider_binding_t empty_key = {iree_string_view_empty(), 0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_link_provider_resolver_prepare(/*provider_count=*/1, &empty_key,
                                          /*binding_count=*/1, &resolver));

  loom_link_provider_binding_t invalid_key = {
      {/*.data=*/nullptr, /*.size=*/1},
      0,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_link_provider_resolver_prepare(/*provider_count=*/1, &invalid_key,
                                          /*binding_count=*/1, &resolver));

  loom_link_provider_binding_t invalid_ordinal = {IREE_SV("provider"), 1};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_link_provider_resolver_prepare(
                            /*provider_count=*/1, &invalid_ordinal,
                            /*binding_count=*/1, &resolver));
}

}  // namespace
}  // namespace loom
