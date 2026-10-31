// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/tool_schemas.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ToolSchemasTest, LookupBorrowsInputWithoutReorderingDeclarations) {
  const std::string declarations =
      R"([{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}},{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{}}}}])";
  std::string input = declarations;
  loom_serve_text_tool_schemas_t schemas;
  IREE_ASSERT_OK(loom_serve_text_tool_schemas_initialize(
      iree_make_string_view(input.data(), input.size()),
      iree_allocator_system(), &schemas));
  EXPECT_EQ(schemas.count, 2);
  const auto* write =
      loom_serve_text_tool_schemas_find(&schemas, IREE_SV("write"));
  ASSERT_NE(write, nullptr);
  EXPECT_EQ(write->name.data, input.data() + input.find("write"));
  EXPECT_TRUE(iree_string_view_equal(
      write->parameters,
      IREE_SV(
          R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})")));
  EXPECT_EQ(write->name_storage, nullptr);
  const auto* read =
      loom_serve_text_tool_schemas_find(&schemas, IREE_SV("read"));
  ASSERT_NE(read, nullptr);
  EXPECT_EQ(read->name_storage, nullptr);
  EXPECT_EQ(loom_serve_text_tool_schemas_find(&schemas, IREE_SV("unknown")),
            nullptr);
  EXPECT_EQ(input, declarations);
  loom_serve_text_tool_schemas_deinitialize(&schemas);
}

TEST(ToolSchemasTest,
     RejectsAmbiguousFieldsWithoutRetainingPartialDescriptors) {
  const char* invalid_functions[] = {
      R"({"name":"write","name":"other","parameters":{"type":"object","properties":{}}})",
      R"({"name":"write","parameters":{},"parameters":{}})",
  };
  for (const char* function : invalid_functions) {
    SCOPED_TRACE(function);
    const std::string input =
        std::string(
            R"([{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{}}}},{"type":"function","function":)") +
        function + "}]";
    loom_serve_text_tool_schemas_t schemas;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_text_tool_schemas_initialize(
                              iree_make_string_view(input.data(), input.size()),
                              iree_allocator_system(), &schemas));
    EXPECT_EQ(schemas.values, nullptr);
    EXPECT_EQ(schemas.count, 0);
    loom_serve_text_tool_schemas_deinitialize(&schemas);
  }
}

TEST(ToolSchemasTest, ParameterSchemasAreOpaqueAndNamesAreDecoded) {
  const char* functions[] = {
      R"({"name":"wr\u0069te"})",
      R"({"name":"write","parameters":{},"strict":false})",
      R"({"name":"write","parameters":{},"strict":true})",
      R"({"name":"write","parameters":true})",
      R"({"name":"write","parameters":[]})",
      R"({"name":"write","parameters":null,"strict":null})",
      R"({"name":"write","parameters":{"$ref":"#/$defs/args","$defs":{"args":{"oneOf":[{"type":"object"},{"type":"null"}]}}}})",
      R"({"name":"write","parameters":{"type":"array","properties":false,"required":42},"description":null,"metadata":{"tag":"pass through"}})",
  };
  for (const char* function : functions) {
    SCOPED_TRACE(function);
    const std::string input =
        std::string(R"([{"type":"function","function":)") + function + "}]";
    loom_serve_text_tool_schemas_t schemas;
    IREE_ASSERT_OK(loom_serve_text_tool_schemas_initialize(
        iree_make_string_view(input.data(), input.size()),
        iree_allocator_system(), &schemas));
    EXPECT_EQ(schemas.count, 1u);
    EXPECT_NE(loom_serve_text_tool_schemas_find(&schemas, IREE_SV("write")),
              nullptr);
    loom_serve_text_tool_schemas_deinitialize(&schemas);
  }
}

TEST(ToolSchemasTest, CatalogEntriesWithoutFunctionHintsNeedNoAllocation) {
  loom_serve_text_tool_schemas_t schemas;
  IREE_ASSERT_OK(loom_serve_text_tool_schemas_initialize(
      IREE_SV(R"([{"type":"custom","custom":{"name":"read"}},
                  {"type":"function","function":{"name":42}},
                  {"type":"function","function":{}},null])"),
      iree_allocator_null(), &schemas));
  EXPECT_EQ(schemas.count, 0);
  loom_serve_text_tool_schemas_deinitialize(&schemas);
}

TEST(ToolSchemasTest, RepeatedNamesHaveNoUnambiguousHint) {
  loom_serve_text_tool_schemas_t schemas;
  IREE_ASSERT_OK(loom_serve_text_tool_schemas_initialize(
      IREE_SV(
          R"([{"type":"function","function":{"name":"read","parameters":{}}},
                  {"type":"function","function":{"name":"re\u0061d","parameters":{"properties":{"path":{"type":"integer"}}}}},
                  {"type":"function","function":{"name":"write"}}])"),
      iree_allocator_system(), &schemas));
  EXPECT_EQ(schemas.count, 3);
  EXPECT_EQ(loom_serve_text_tool_schemas_find(&schemas, IREE_SV("read")),
            nullptr);
  EXPECT_NE(loom_serve_text_tool_schemas_find(&schemas, IREE_SV("write")),
            nullptr);
  loom_serve_text_tool_schemas_deinitialize(&schemas);
}

TEST(ToolSchemasTest, EmptyDeclarationsNeedNoAllocation) {
  loom_serve_text_tool_schemas_t schemas;
  IREE_ASSERT_OK(loom_serve_text_tool_schemas_initialize(
      IREE_SV("[]"), iree_allocator_null(), &schemas));
  EXPECT_EQ(schemas.count, 0);
  EXPECT_EQ(loom_serve_text_tool_schemas_find(&schemas, IREE_SV("read")),
            nullptr);
  loom_serve_text_tool_schemas_deinitialize(&schemas);
}

}  // namespace
