// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptor_scope.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(LowDescriptorScopeTest, FamilyExactAndSharedScopes) {
  const loom_low_descriptor_set_t first = {
      .stable_id = 11,
      .target_stable_id = 1,
  };
  const loom_low_descriptor_set_t sibling = {
      .stable_id = 12,
      .target_stable_id = 1,
  };
  const loom_low_descriptor_set_t foreign = {
      .stable_id = 21,
      .target_stable_id = 2,
  };
  const loom_low_descriptor_set_t unowned = {.stable_id = 31};
  const loom_low_descriptor_set_t* descriptors[] = {&first, &sibling, &foreign,
                                                    &unowned, nullptr};
  const loom_low_descriptor_scope_t scopes[] = {
      {.target_stable_id = 1},
      {.descriptor_set_stable_id = 11},
      LOOM_LOW_DESCRIPTOR_SCOPE_SHARED,
  };
  const bool expected[][5] = {
      {true, true, false, false, false},
      {true, false, false, false, false},
      {true, true, true, true, true},
  };
  for (size_t scope_index = 0; scope_index < IREE_ARRAYSIZE(scopes);
       ++scope_index) {
    EXPECT_TRUE(loom_low_descriptor_scope_is_valid(scopes[scope_index]));
    for (size_t set_index = 0; set_index < IREE_ARRAYSIZE(descriptors);
         ++set_index) {
      EXPECT_EQ(loom_low_descriptor_scope_matches(scopes[scope_index],
                                                  descriptors[set_index]),
                expected[scope_index][set_index])
          << scope_index << ":" << set_index;
    }
  }
}

TEST(LowDescriptorScopeTest, RegistrationRequiresOneExplicitScope) {
  EXPECT_FALSE(loom_low_descriptor_scope_is_valid({}));
  EXPECT_FALSE(loom_low_descriptor_scope_is_valid(
      {.target_stable_id = 1, .descriptor_set_stable_id = 11}));
  EXPECT_FALSE(loom_low_descriptor_scope_is_valid(
      {.target_stable_id = UINT64_MAX, .descriptor_set_stable_id = 11}));
}

}  // namespace
}  // namespace loom
