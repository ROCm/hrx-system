// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/tool_xml.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/program.h"

namespace {

class ToolXmlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(host_allocator_, &environment_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &types_));
    IREE_ASSERT_OK(loom_serve_text_tool_xml_module_create(
        environment_, host_allocator_, &module_));
    IREE_ASSERT_OK(loom_serve_program_link(module_, {nullptr, 0},
                                           host_allocator_, &program_));
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        loom_serve_program_process(program_), IREE_SV("tools"),
        IREE_SV("parse_xml"), &parse_));
  }

  void TearDown() override {
    loom_serve_program_free(program_);
    iree_vm_module_release(module_);
    iree_vm_environment_free(environment_);
  }

  iree_vm_variant_t Text(iree_string_view_t text) {
    iree_vm_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(
        iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_make_const_byte_span(text.data, text.size), 1,
                             host_allocator_, &buffer));
    return iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
  }

  iree_status_t Parse(iree_string_view_t text, std::string* out_calls,
                      bool truncated = false) {
    return Parse(text,
                 IREE_SV(
                     R"([{"type":"function","function":{"name":"search",
          "parameters":{"type":"object","properties":{
            "query":{"type":"string"},"limit":{"type":"integer"}},
            "required":["query"]}}}])"),
                 out_calls, truncated);
  }

  iree_status_t Parse(iree_string_view_t text, iree_string_view_t schemas,
                      std::string* out_calls, bool truncated = false) {
    out_calls->clear();
    iree_vm_variant_t arguments[] = {
        Text(schemas),
        Text(text),
        iree_vm_variant_from_i64(truncated),
    };
    iree_vm_variant_t result = iree_vm_variant_empty();
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(program_), parse_,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_ptr(&result, 1));
    if (iree_status_is_ok(status)) {
      iree_vm_buffer_t* buffer = nullptr;
      status =
          iree_vm_buffer_ptr_from_variant_borrowed(&types_, result, &buffer);
      iree_const_byte_span_t bytes = iree_const_byte_span_empty();
      if (iree_status_is_ok(status)) {
        status = iree_vm_buffer_map_read(buffer, 0,
                                         iree_vm_buffer_length(buffer), &bytes);
      }
      if (iree_status_is_ok(status)) {
        out_calls->assign(reinterpret_cast<const char*>(bytes.data),
                          bytes.data_length);
      }
    } else {
      EXPECT_TRUE(iree_vm_variant_is_empty(result));
    }
    iree_vm_variant_reset(&result);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    return status;
  }

  // Allocation policy for the real native module and its invocation.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Type owner outliving the module, program, and result buffers.
  iree_vm_environment_t* environment_ = nullptr;
  // Canonical identities used to exchange UTF-8 buffers with the native codec.
  iree_vm_ref_types_t types_ = {};
  // Production schema-aware codec, retained by the linked program.
  iree_vm_module_t* module_ = nullptr;
  // Shared invocation reused after both successful and failed completions.
  loom_serve_program_t* program_ = nullptr;
  // Native export resolved from the linked program.
  iree_vm_function_t parse_ = {};
};

TEST_F(ToolXmlTest, GeneratedArgumentsPreserveDeclaredTypesAndStringContent) {
  std::string calls;
  IREE_ASSERT_OK(
      Parse(IREE_SV("<tool_call><function=search>"
                    "<parameter=query>\n\"λ\"\n</parameter>"
                    "<parameter=limit>3</parameter></function></tool_call>"),
            &calls));
  EXPECT_EQ(
      calls,
      R"([{"type":"function","function":{"name":"search","arguments":"{\"query\":\"\\\"λ\\\"\",\"limit\":3}"}}])");
}

TEST_F(ToolXmlTest, InvalidLaterCallPublishesNothingAndInvocationIsReusable) {
  const std::string valid =
      "<tool_call><function=search><parameter=query>kept</parameter>"
      "</function></tool_call>";
  struct Failure {
    // Invalid second call following an otherwise valid first call.
    const char* text;
    // Diagnostic identifying the generated tool and invalid argument.
    const char* message;
  };
  const Failure failures[] = {
      {"<tool_call><function=search", "unterminated generated tool name"},
      {"<tool_call><function=search><parameter=query>cut off",
       "unterminated generated parameter"},
      {"<tool_call><function=search></function>",
       "generated tool call expected '</tool_call>'; got ''"},
  };
  for (const auto& failure : failures) {
    SCOPED_TRACE(failure.text);
    const std::string text = valid + failure.text;
    std::string calls;
    auto status =
        Parse(iree_make_string_view(text.data(), text.size()), &calls);
    EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
    const auto message = iree_status_message(status);
    const std::string diagnostic(message.data, message.size);
    EXPECT_NE(diagnostic.find(failure.message), std::string::npos);
    EXPECT_NE(diagnostic.find("tool call 1 at byte "), std::string::npos);
    iree_status_free(status);
    EXPECT_TRUE(calls.empty());
    IREE_ASSERT_OK(
        Parse(iree_make_string_view(valid.data(), valid.size()), &calls));
    EXPECT_EQ(
        calls,
        R"([{"type":"function","function":{"name":"search","arguments":"{\"query\":\"kept\"}"}}])");
  }
}

TEST_F(ToolXmlTest, EmptyOutputDoesNotInspectUnusedSchemas) {
  std::string calls;
  IREE_ASSERT_OK(Parse(IREE_SV("\r\n\t"), IREE_SV("unused"), &calls));
  EXPECT_EQ(calls, "[]");
}

TEST_F(ToolXmlTest, TokenLimitAtEveryBytePublishesOnlyCompleteCalls) {
  const std::string call =
      "<tool_call>\n<function=search>\n"
      "<parameter=query>\nhello\n</parameter>\n"
      "<parameter=limit>3</parameter>\n</function>\n</tool_call>";
  const std::string expected =
      R"({"type":"function","function":{"name":"search","arguments":"{\"query\":\"hello\",\"limit\":3}"}})";
  for (size_t length = 0; length <= call.size(); ++length) {
    SCOPED_TRACE(length);
    std::string calls;
    IREE_ASSERT_OK(
        Parse(iree_make_string_view(call.data(), length), &calls, true));
    EXPECT_EQ(calls, length == call.size() ? "[" + expected + "]" : "[]");
    // A complete prefix survives a later truncated call, without repairing or
    // publishing any of that last call's name or arguments.
    const std::string text = call + "\n" + call.substr(0, length);
    IREE_ASSERT_OK(
        Parse(iree_make_string_view(text.data(), text.size()), &calls, true));
    EXPECT_EQ(calls, "[" + expected +
                         (length == call.size() ? "," + expected : "") + "]");
  }
}

TEST_F(ToolXmlTest, TokenLimitDoesNotHideMalformedFraming) {
  const std::string valid =
      "<tool_call><function=search></function></tool_call>";
  for (const char* invalid :
       {"not a call", "<tool_cX", "<tool_call><funX",
        "<tool_call><function=search><paramX",
        "<tool_call><function=search></funcX",
        "<tool_call><function=search></function></tool_cX"}) {
    SCOPED_TRACE(invalid);
    for (const bool truncated : {false, true}) {
      SCOPED_TRACE(truncated);
      const std::string text = valid + invalid;
      std::string calls;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          Parse(iree_make_string_view(text.data(), text.size()), &calls,
                truncated));
      EXPECT_TRUE(calls.empty());
      IREE_ASSERT_OK(
          Parse(iree_make_cstring_view(valid.c_str()), &calls, truncated));
      EXPECT_EQ(
          calls,
          R"([{"type":"function","function":{"name":"search","arguments":"{}"}}])");
    }
  }
}

TEST_F(ToolXmlTest, NamesAreDataAndEscapesOnlyBelongToJson) {
  std::string calls;
  IREE_ASSERT_OK(
      Parse(IREE_SV("<tool_call><function=namespace/read>"
                    "<parameter=line count>3</parameter>"
                    "<parameter=λ>kept</parameter></function></tool_call>"),
            IREE_SV(R"([{"type":"function","function":{"name":"namespace\/read",
        "parameters":{"properties":{"line\u0020count":{"type":"integer"}}}}}])"),
            &calls));
  EXPECT_EQ(
      calls,
      R"([{"type":"function","function":{"name":"namespace/read","arguments":"{\"line count\":3,\"λ\":\"kept\"}"}}])");
}

TEST_F(ToolXmlTest, InvalidUtf8PublishesNoResultAndNextCallWorks) {
  std::string calls;
  const char invalid[] = {'<', '\xFF'};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(iree_make_string_view(invalid, sizeof(invalid)), &calls));
  EXPECT_TRUE(calls.empty());
  IREE_ASSERT_OK(Parse(
      IREE_SV("<tool_call><function=search></function></tool_call>"), &calls));
  EXPECT_EQ(
      calls,
      R"([{"type":"function","function":{"name":"search","arguments":"{}"}}])");
}

TEST_F(ToolXmlTest, SemanticMistakesReachTheHarnessWithoutRepair) {
  std::string calls;
  IREE_ASSERT_OK(Parse(
      IREE_SV("<tool_call><function=search><parameter=q>where?</parameter>"
              "</function></tool_call>"
              "<tool_call><function=search><parameter=query>first</parameter>"
              "<parameter=query>second</parameter></function></tool_call>"
              "<tool_call><function=undeclared><parameter=path>123</parameter>"
              "</function></tool_call>"
              "<tool_call><function=search><parameter=limit>many</parameter>"
              "</function></tool_call>"
              "<tool_call><function=search></function></tool_call>"),
      &calls));
  EXPECT_EQ(
      calls,
      R"([{"type":"function","function":{"name":"search","arguments":"{\"q\":\"where?\"}"}},{"type":"function","function":{"name":"search","arguments":"{\"query\":\"first\",\"query\":\"second\"}"}},{"type":"function","function":{"name":"undeclared","arguments":"{\"path\":\"123\"}"}},{"type":"function","function":{"name":"search","arguments":"{\"limit\":\"many\"}"}},{"type":"function","function":{"name":"search","arguments":"{}"}}])");
}

TEST_F(ToolXmlTest, CatalogPolicyDoesNotConstrainOutputTranslation) {
  const auto schemas = IREE_SV(R"([
    {"type":"custom","custom":{"name":"client_tool"}},
    {"type":"function","function":{"name":"read","strict":true,
      "parameters":{"properties":{"path":{"type":"string"}}}}},
    {"type":"function","function":{"name":"ambiguous",
      "parameters":{"properties":{"value":{"type":"integer"}}}}},
    {"type":"function","function":{"name":"ambiguous",
      "parameters":{"properties":{"value":{"type":"string"}}}}}
  ])");
  std::string calls;
  IREE_ASSERT_OK(Parse(
      IREE_SV("<tool_call><function=read><parameter=path>123</parameter>"
              "<parameter=extra>true</parameter></function></tool_call>"
              "<tool_call><function=ambiguous><parameter=value>123</parameter>"
              "</function></tool_call>"),
      schemas, &calls));
  EXPECT_EQ(
      calls,
      R"([{"type":"function","function":{"name":"read","arguments":"{\"path\":\"123\",\"extra\":\"true\"}"}},{"type":"function","function":{"name":"ambiguous","arguments":"{\"value\":\"123\"}"}}])");
}

TEST_F(ToolXmlTest,
       CodingAndImageArgumentsUseRepresentationHintsNotConstraints) {
  const auto schemas = IREE_SV(R"([
    {"type":"function","function":{"name":"edit","parameters":{
      "type":"object","properties":{
        "path":{"type":"string"},
        "replacement":{"type":"string"},
        "offset":{"anyOf":[{"type":"integer","minimum":1},{"type":"null"}]},
        "limit":{"type":["integer","null"],"maximum":100},
        "dry_run":{"type":"boolean"},
        "edits":{"type":"array","items":{"$ref":"#/$defs/edit"}}},
      "required":["path","replacement"],"additionalProperties":false}}},
    {"type":"function","function":{"name":"generate_image","parameters":{
      "properties":{
        "prompt":{"oneOf":[{"type":"string"},{"type":"null"}]},
        "size":{"type":"string","enum":["1024x1024"]},
        "seed":{"type":"integer"},"options":{"type":"object"}}}}}
  ])");
  std::string calls;
  IREE_ASSERT_OK(Parse(
      IREE_SV("<tool_call><function=edit>"
              "<parameter=path>123</parameter>"
              "<parameter=replacement>\n  return \"λ\";\n\n</parameter>"
              "<parameter=offset>-1</parameter>"
              "<parameter=limit>null</parameter>"
              "<parameter=dry_run>FALSE</parameter>"
              "<parameter=edits>[{\"old\":\"a\",\"new\":\"b\"}]</parameter>"
              "</function></tool_call>"
              "<tool_call><function=generate_image>"
              "<parameter=prompt>null</parameter>"
              "<parameter=size>1536x1024</parameter>"
              "<parameter=seed>18446744073709551616</parameter>"
              "<parameter=options>{\"scale\":1.25e2}</parameter>"
              "</function></tool_call>"),
      schemas, &calls));
  EXPECT_EQ(
      calls,
      R"([{"type":"function","function":{"name":"edit","arguments":"{\"path\":\"123\",\"replacement\":\"  return \\\"λ\\\";\\n\",\"offset\":-1,\"limit\":null,\"dry_run\":\"FALSE\",\"edits\":[{\"old\":\"a\",\"new\":\"b\"}]}"}},{"type":"function","function":{"name":"generate_image","arguments":"{\"prompt\":null,\"size\":\"1536x1024\",\"seed\":18446744073709551616,\"options\":{\"scale\":1.25e2}}"}}])");
}

TEST_F(ToolXmlTest, StringsAreLiteralAndNonJsonValuesAreNotGuessed) {
  const auto schemas = IREE_SV(R"([{"type":"function","function":{
    "name":"values","parameters":{"properties":{
      "string":{"type":"string"},"number":{"type":"number"},
      "object":{"type":"object"},"nullable":{"type":["string","null"]}
    }}}}])");
  std::string calls;
  IREE_ASSERT_OK(Parse(IREE_SV("<tool_call><function=values>"
                               "<parameter=string>null</parameter>"
                               "<parameter=string>{\"a\":1}</parameter>"
                               "<parameter=string>&lt;tag&gt;</parameter>"
                               "<parameter=number>3 trailing</parameter>"
                               "<parameter=object>{broken</parameter>"
                               "<parameter=nullable>123</parameter>"
                               "<parameter=unknown></parameter>"
                               "</function></tool_call>"),
                       schemas, &calls));
  EXPECT_EQ(
      calls,
      R"([{"type":"function","function":{"name":"values","arguments":"{\"string\":\"null\",\"string\":\"{\\\"a\\\":1}\",\"string\":\"&lt;tag&gt;\",\"number\":\"3 trailing\",\"object\":\"{broken\",\"nullable\":\"123\",\"unknown\":\"\"}"}}])");
}

TEST_F(ToolXmlTest, OpaqueSchemaStringsAreNotReparsedAsJson) {
  const char* parameters[] = {
      R"({"properties":"{opaque"})",
      R"({"properties":{"value":{"type":"[opaque","anyOf":"{opaque"}}})",
      R"({"properties":{"value":{"type":null}}})",
  };
  for (const char* parameter_schema : parameters) {
    SCOPED_TRACE(parameter_schema);
    const std::string schemas =
        std::string(
            R"([{"type":"function","function":{"name":"read","parameters":)") +
        parameter_schema + "}}]";
    std::string calls;
    IREE_ASSERT_OK(Parse(
        IREE_SV("<tool_call><function=read><parameter=value>null</parameter>"
                "</function></tool_call>"),
        iree_make_string_view(schemas.data(), schemas.size()), &calls));
    EXPECT_EQ(
        calls,
        R"([{"type":"function","function":{"name":"read","arguments":"{\"value\":\"null\"}"}}])");
  }
}

}  // namespace
