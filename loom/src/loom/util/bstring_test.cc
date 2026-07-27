// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/bstring.h"

#include "iree/testing/gtest.h"

namespace {

// Test B-strings covering different lengths.
static const uint8_t kEmpty[] = LOOM_BSTRING_LITERAL(0, "");
static const uint8_t kX[] = LOOM_BSTRING_LITERAL(1, "x");
static const uint8_t kTo[] = LOOM_BSTRING_LITERAL(2, "to");
static const uint8_t kNsw[] = LOOM_BSTRING_LITERAL(3, "nsw");
static const uint8_t kStep[] = LOOM_BSTRING_LITERAL(4, "step");
static const uint8_t kHello[] = LOOM_BSTRING_LITERAL(5, "hello");
static const uint8_t kIterArgs[] = LOOM_BSTRING_LITERAL(9, "iter_args");

TEST(BString, Length) {
  EXPECT_EQ(loom_bstring_length(kEmpty), 0);
  EXPECT_EQ(loom_bstring_length(kX), 1);
  EXPECT_EQ(loom_bstring_length(kTo), 2);
  EXPECT_EQ(loom_bstring_length(kNsw), 3);
  EXPECT_EQ(loom_bstring_length(kStep), 4);
  EXPECT_EQ(loom_bstring_length(kHello), 5);
  EXPECT_EQ(loom_bstring_length(kIterArgs), 9);
}

TEST(BString, Data) {
  EXPECT_EQ(std::string_view(loom_bstring_data(kHello), 5), "hello");
  EXPECT_EQ(std::string_view(loom_bstring_data(kTo), 2), "to");
  EXPECT_EQ(std::string_view(loom_bstring_data(kX), 1), "x");
  EXPECT_EQ(std::string_view(loom_bstring_data(kIterArgs), 9), "iter_args");
}

TEST(BString, ViewConvertsCorrectly) {
  EXPECT_TRUE(
      iree_string_view_equal(loom_bstring_view(kHello), IREE_SV("hello")));
  EXPECT_TRUE(iree_string_view_equal(loom_bstring_view(kTo), IREE_SV("to")));
  EXPECT_TRUE(iree_string_view_equal(loom_bstring_view(kX), IREE_SV("x")));
  EXPECT_TRUE(iree_string_view_equal(loom_bstring_view(kIterArgs),
                                     IREE_SV("iter_args")));
}

TEST(BString, ViewEmptyIsEmpty) {
  iree_string_view_t view = loom_bstring_view(kEmpty);
  EXPECT_TRUE(iree_string_view_is_empty(view));
  EXPECT_EQ(view.size, 0u);
}

TEST(BString, EqualMatchesIdentical) {
  EXPECT_TRUE(loom_bstring_equal(kHello, IREE_SV("hello")));
  EXPECT_TRUE(loom_bstring_equal(kTo, IREE_SV("to")));
  EXPECT_TRUE(loom_bstring_equal(kX, IREE_SV("x")));
  EXPECT_TRUE(loom_bstring_equal(kNsw, IREE_SV("nsw")));
  EXPECT_TRUE(loom_bstring_equal(kStep, IREE_SV("step")));
  EXPECT_TRUE(loom_bstring_equal(kIterArgs, IREE_SV("iter_args")));
  EXPECT_TRUE(loom_bstring_equal(kEmpty, iree_string_view_empty()));
  EXPECT_TRUE(loom_bstring_equal(kEmpty, IREE_SV("")));
}

TEST(BString, EqualRejectsDifferentContent) {
  EXPECT_FALSE(loom_bstring_equal(kHello, IREE_SV("world")));
  EXPECT_FALSE(loom_bstring_equal(kTo, IREE_SV("no")));
  EXPECT_FALSE(loom_bstring_equal(kNsw, IREE_SV("nuw")));
}

TEST(BString, EqualRejectsShorterView) {
  EXPECT_FALSE(loom_bstring_equal(kHello, IREE_SV("hell")));
  EXPECT_FALSE(loom_bstring_equal(kTo, IREE_SV("t")));
  EXPECT_FALSE(loom_bstring_equal(kIterArgs, IREE_SV("iter")));
}

TEST(BString, EqualRejectsLongerView) {
  EXPECT_FALSE(loom_bstring_equal(kHello, IREE_SV("helloo")));
  EXPECT_FALSE(loom_bstring_equal(kTo, IREE_SV("too")));
  EXPECT_FALSE(loom_bstring_equal(kX, IREE_SV("xy")));
}

TEST(BString, EqualRejectsPrefix) {
  EXPECT_FALSE(loom_bstring_equal(kHello, IREE_SV("hel")));
  EXPECT_FALSE(loom_bstring_equal(kStep, IREE_SV("ste")));
}

TEST(BString, EqualEmptyRejectsNonEmpty) {
  EXPECT_FALSE(loom_bstring_equal(kEmpty, IREE_SV("x")));
  EXPECT_FALSE(loom_bstring_equal(kEmpty, IREE_SV("hello")));
}

TEST(BString, EqualNonEmptyRejectsEmpty) {
  EXPECT_FALSE(loom_bstring_equal(kX, iree_string_view_empty()));
  EXPECT_FALSE(loom_bstring_equal(kHello, iree_string_view_empty()));
}

// Verify that B-strings work in array tables (the primary use case).
TEST(BString, ArrayTableLookup) {
  static const loom_bstring_t keywords[] = {
      kTo,
      kStep,
      kHello,
      kIterArgs,
  };
  // Simulate a parser token scan: find "step" in the table.
  iree_string_view_t token = IREE_SV("step");
  int found = -1;
  for (int i = 0; i < 4; ++i) {
    if (loom_bstring_equal(keywords[i], token)) {
      found = i;
      break;
    }
  }
  EXPECT_EQ(found, 1);

  // Verify miss.
  token = IREE_SV("missing");
  found = -1;
  for (int i = 0; i < 4; ++i) {
    if (loom_bstring_equal(keywords[i], token)) {
      found = i;
      break;
    }
  }
  EXPECT_EQ(found, -1);
}

TEST(BString, InlineReferenceLiteral) {
  static const loom_bstring_t kResult = LOOM_BSTRING_REF(6, "result");

  EXPECT_EQ(loom_bstring_length(kResult), 6);
  EXPECT_TRUE(loom_bstring_equal(kResult, IREE_SV("result")));
}

TEST(BString, InlineOpNameReferenceLiteral) {
  static const loom_bstring_t kOpName = LOOM_OP_NAME_REF(12, 6, "vector.empty");

  EXPECT_EQ(kOpName[0], 12);
  EXPECT_EQ(kOpName[1], 6);
  EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(kOpName + 2), 12),
            "vector.empty");
}

// clang-format off
static const uint8_t kPackedStrings[] =
    LOOM_BSTRING_LITERAL(0, "")
    LOOM_BSTRING_LITERAL(5, "hello")
    LOOM_BSTRING_LITERAL(4, "step");
// clang-format on

enum {
  kPackedStringEmpty = 0,
  kPackedStringHello = kPackedStringEmpty + sizeof(""),
  kPackedStringStep = kPackedStringHello + sizeof("hello"),
  kPackedStringEnd = kPackedStringStep + sizeof("step"),
};

static_assert(kPackedStringEnd == sizeof(kPackedStrings) - 1,
              "packed B-string offsets must cover the table payload");

TEST(BStringTable, ContainsValidOffsets) {
  const loom_bstring_table_t table = {
      /*.data=*/kPackedStrings,
      /*.data_length=*/sizeof(kPackedStrings) - 1,
  };
  const loom_bstring_table_offset_t hello_offset =
      static_cast<loom_bstring_table_offset_t>(kPackedStringHello);
  const loom_bstring_table_offset_t step_offset =
      static_cast<loom_bstring_table_offset_t>(kPackedStringStep);

  EXPECT_TRUE(loom_bstring_table_contains(&table, hello_offset));
  EXPECT_TRUE(loom_bstring_table_contains(&table, step_offset));
  EXPECT_TRUE(loom_bstring_equal(loom_bstring_table_get(&table, hello_offset),
                                 IREE_SV("hello")));

  loom_bstring_t step = nullptr;
  EXPECT_TRUE(loom_bstring_table_try_get(&table, step_offset, &step));
  EXPECT_TRUE(loom_bstring_equal(step, IREE_SV("step")));
}

TEST(BStringTable, RejectsInvalidOffsets) {
  static const uint8_t kTruncated[] = {5, 'h', 'e'};
  const loom_bstring_table_t table = {
      /*.data=*/kTruncated,
      /*.data_length=*/sizeof(kTruncated),
  };

  EXPECT_FALSE(loom_bstring_table_contains(&table, 0));
  EXPECT_FALSE(loom_bstring_table_contains(&table, table.data_length));
  EXPECT_FALSE(
      loom_bstring_table_contains(&table, LOOM_BSTRING_TABLE_OFFSET_NONE));

  loom_bstring_t value = reinterpret_cast<loom_bstring_t>(0x1);
  EXPECT_FALSE(loom_bstring_table_try_get(&table, 0, &value));
  EXPECT_EQ(value, nullptr);
}

}  // namespace
