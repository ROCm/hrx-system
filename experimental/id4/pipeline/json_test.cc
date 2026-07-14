// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/json.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::string EncodeJsonString(iree_string_view_t value) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_EXPECT_OK(id4_pipeline_json_append_string(&builder, value));
  const iree_string_view_t result = iree_string_builder_view(&builder);
  std::string encoded(result.data, result.size);
  iree_string_builder_deinitialize(&builder);
  return encoded;
}

TEST(JsonTest, QuotesAndEscapesStructuralCharacters) {
  EXPECT_EQ(EncodeJsonString(IREE_SV("path\\to/\"value\"")),
            "\"path\\\\to/\\\"value\\\"\"");
}

TEST(JsonTest, EscapesNamedControlCharacters) {
  EXPECT_EQ(EncodeJsonString(IREE_SV("\b\f\n\r\t")), "\"\\b\\f\\n\\r\\t\"");
}

TEST(JsonTest, EscapesRemainingControlBytes) {
  static const char value[] = {'a', '\x00', '\x01', '\x1f', 'z'};
  EXPECT_EQ(EncodeJsonString(iree_make_string_view(value, sizeof(value))),
            "\"a\\u0000\\u0001\\u001fz\"");
}

TEST(JsonTest, PreservesUtf8Bytes) {
  EXPECT_EQ(EncodeJsonString(IREE_SV("weight.\xE2\x88\x86")),
            "\"weight.\xE2\x88\x86\"");
}

}  // namespace
