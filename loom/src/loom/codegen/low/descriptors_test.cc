// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

static const uint8_t kEnumStrings[] =
    LOOM_BSTRING_LITERAL(3, "sge") LOOM_BSTRING_LITERAL(3, "ult");

enum {
  kEnumStringSge = 0,
  kEnumStringUlt = kEnumStringSge + sizeof("sge"),
};

static const loom_low_enum_value_t kEnumValues[] = {
    {/*.token_string_offset=*/kEnumStringSge, /*.value=*/7},
    {/*.token_string_offset=*/kEnumStringUlt, /*.value=*/11},
};

static const loom_low_enum_domain_t kEnumDomains[] = {
    {/*.name_string_offset=*/LOOM_LOW_STRING_OFFSET_NONE,
     /*.value_start=*/0,
     /*.value_count=*/IREE_ARRAYSIZE(kEnumValues),
     /*.reserved=*/0},
};

static loom_low_descriptor_set_t MakeEnumDescriptorSet() {
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.string_table.data = kEnumStrings;
  descriptor_set.string_table.data_length = sizeof(kEnumStrings) - 1;
  descriptor_set.enum_domains = kEnumDomains;
  descriptor_set.enum_domain_count = IREE_ARRAYSIZE(kEnumDomains);
  descriptor_set.enum_values = kEnumValues;
  descriptor_set.enum_value_count = IREE_ARRAYSIZE(kEnumValues);
  return descriptor_set;
}

TEST(LowDescriptorsTest, LooksUpEnumValuesAndTokens) {
  const loom_low_descriptor_set_t descriptor_set = MakeEnumDescriptorSet();
  int64_t value = 0;
  EXPECT_TRUE(loom_low_descriptor_set_lookup_enum_value_by_token(
      &descriptor_set, 0, IREE_SV("ult"), &value));
  EXPECT_EQ(value, 11);

  iree_string_view_t token = iree_string_view_empty();
  EXPECT_TRUE(loom_low_descriptor_set_lookup_enum_token_by_value(
      &descriptor_set, 0, 7, &token));
  EXPECT_TRUE(iree_string_view_equal(token, IREE_SV("sge")));
}

TEST(LowDescriptorsTest, RejectsUnknownEnumValuesAndDomains) {
  const loom_low_descriptor_set_t descriptor_set = MakeEnumDescriptorSet();
  int64_t value = 1;
  EXPECT_FALSE(loom_low_descriptor_set_lookup_enum_value_by_token(
      &descriptor_set, 0, IREE_SV("missing"), &value));
  EXPECT_EQ(value, 0);

  iree_string_view_t token = IREE_SV("sentinel");
  EXPECT_FALSE(loom_low_descriptor_set_lookup_enum_token_by_value(
      &descriptor_set, 1, 7, &token));
  EXPECT_TRUE(iree_string_view_is_empty(token));
}

}  // namespace
}  // namespace loom
