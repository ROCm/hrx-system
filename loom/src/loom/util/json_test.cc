// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/json.h"

#include <string>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

// Helper: escape a string through the JSON escape adapter and return
// the raw escaped content (no surrounding quotes).
std::string EscapeRaw(iree_string_view_t input) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(&stream, &escape_data, &escape_stream);

  iree_status_t status = loom_output_stream_write(&escape_stream, input);
  std::string result;
  if (iree_status_is_ok(status)) {
    iree_string_view_t view = iree_string_builder_view(&builder);
    result.assign(view.data, view.size);
  }
  IREE_EXPECT_OK(status);
  iree_string_builder_deinitialize(&builder);
  return result;
}

std::string EscapeRaw(const char* input) {
  return EscapeRaw(iree_make_cstring_view(input));
}

// Helper: escape through the quoted convenience function.
std::string EscapeQuoted(const char* input) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  iree_status_t status = loom_json_write_escaped_cstring(&stream, input);
  std::string result;
  if (iree_status_is_ok(status)) {
    iree_string_view_t view = iree_string_builder_view(&builder);
    result.assign(view.data, view.size);
  }
  IREE_EXPECT_OK(status);
  iree_string_builder_deinitialize(&builder);
  return result;
}

std::string WriteStatusObject(iree_status_code_t code,
                              iree_string_view_t message) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  iree_status_t status = loom_json_write_status_object(&stream, code, message);
  std::string result;
  if (iree_status_is_ok(status)) {
    iree_string_view_t view = iree_string_builder_view(&builder);
    result.assign(view.data, view.size);
  }
  IREE_EXPECT_OK(status);
  iree_string_builder_deinitialize(&builder);
  return result;
}

std::string WriteStructuredJson() {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  loom_json_object_writer_t object;
  IREE_EXPECT_OK(loom_json_object_begin(&stream, &object));
  IREE_EXPECT_OK(loom_json_object_write_string_field(
      &object, IREE_SV("required"), iree_string_view_empty()));
  IREE_EXPECT_OK(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("omitted"), iree_string_view_empty()));
  IREE_EXPECT_OK(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("present"), IREE_SV("value")));
  IREE_EXPECT_OK(
      loom_json_object_write_null_field(&object, IREE_SV("unknown")));
  IREE_EXPECT_OK(
      loom_json_object_write_uint32_field(&object, IREE_SV("answer"), 42));
  IREE_EXPECT_OK(
      loom_json_object_begin_field(&object, IREE_SV("escaped\"field")));
  loom_json_array_writer_t array;
  IREE_EXPECT_OK(loom_json_array_begin(object.stream, &array));
  IREE_EXPECT_OK(loom_json_array_write_int64_element(&array, -7));
  IREE_EXPECT_OK(loom_json_array_write_bool_element(&array, true));
  IREE_EXPECT_OK(loom_json_array_write_null_element(&array));
  IREE_EXPECT_OK(loom_json_array_begin_element(&array));
  loom_json_object_writer_t nested_object;
  IREE_EXPECT_OK(loom_json_object_begin(array.stream, &nested_object));
  IREE_EXPECT_OK(loom_json_object_write_string_field(
      &nested_object, IREE_SV("name"), IREE_SV("nested")));
  IREE_EXPECT_OK(loom_json_object_end(&nested_object));
  IREE_EXPECT_OK(loom_json_array_end(&array));
  IREE_EXPECT_OK(loom_json_object_end(&object));

  iree_string_view_t view = iree_string_builder_view(&builder);
  std::string result(view.data, view.size);
  iree_string_builder_deinitialize(&builder);
  return result;
}

//===----------------------------------------------------------------------===//
// Basic escaping
//===----------------------------------------------------------------------===//

TEST(JsonEscape, EmptyString) {
  EXPECT_EQ(EscapeRaw(""), "");
  EXPECT_EQ(EscapeQuoted(""), "\"\"");
}

TEST(JsonEscape, PlainAscii) {
  EXPECT_EQ(EscapeRaw("hello world"), "hello world");
  EXPECT_EQ(EscapeQuoted("hello"), "\"hello\"");
}

TEST(JsonEscape, DoubleQuote) {
  EXPECT_EQ(EscapeRaw("say \"hi\""), "say \\\"hi\\\"");
}

TEST(JsonEscape, Backslash) { EXPECT_EQ(EscapeRaw("a\\b"), "a\\\\b"); }

TEST(JsonEscape, Newline) { EXPECT_EQ(EscapeRaw("a\nb"), "a\\nb"); }

TEST(JsonEscape, CarriageReturn) { EXPECT_EQ(EscapeRaw("a\rb"), "a\\rb"); }

TEST(JsonEscape, Tab) { EXPECT_EQ(EscapeRaw("a\tb"), "a\\tb"); }

TEST(JsonEscape, Backspace) { EXPECT_EQ(EscapeRaw("a\bb"), "a\\bb"); }

TEST(JsonEscape, FormFeed) { EXPECT_EQ(EscapeRaw("a\fb"), "a\\fb"); }

//===----------------------------------------------------------------------===//
// Control characters
//===----------------------------------------------------------------------===//

TEST(JsonEscape, NullByte) {
  // NUL is a control character — must be escaped as \u0000.
  std::string input("a\0b", 3);
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(&stream, &escape_data, &escape_stream);
  IREE_ASSERT_OK(loom_output_stream_write(
      &escape_stream, iree_make_string_view(input.data(), input.size())));
  iree_string_view_t view = iree_string_builder_view(&builder);
  EXPECT_EQ(std::string(view.data, view.size), "a\\u0000b");
  iree_string_builder_deinitialize(&builder);
}

TEST(JsonEscape, AllControlChars) {
  // Every byte 0x00-0x1F must be escaped. The named ones (\b, \t, \n,
  // \f, \r) get their short form; the rest get \uNNNN.
  for (int c = 0; c < 0x20; ++c) {
    char input_byte = (char)c;
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    loom_json_escape_stream_t escape_data;
    loom_output_stream_t escape_stream;
    loom_json_escape_stream_init(&stream, &escape_data, &escape_stream);
    IREE_ASSERT_OK(loom_output_stream_write(
        &escape_stream, iree_make_string_view(&input_byte, 1)));
    iree_string_view_t view = iree_string_builder_view(&builder);
    std::string escaped(view.data, view.size);
    iree_string_builder_deinitialize(&builder);
    // Must not pass through unchanged.
    EXPECT_NE(escaped, std::string(&input_byte, 1))
        << "control char 0x" << std::hex << c << " was not escaped";
    // Must start with backslash.
    ASSERT_FALSE(escaped.empty())
        << "control char 0x" << std::hex << c << " produced empty output";
    EXPECT_EQ(escaped[0], '\\')
        << "control char 0x" << std::hex << c
        << " escape doesn't start with backslash: " << escaped;
  }
}

//===----------------------------------------------------------------------===//
// UTF-8 pass-through
//===----------------------------------------------------------------------===//

TEST(JsonEscape, ChinesePassThrough) {
  // U+4F60 U+597D = "you good" (nihao) in UTF-8: E4 BD A0 E5 A5 BD.
  EXPECT_EQ(EscapeRaw("\xe4\xbd\xa0\xe5\xa5\xbd"), "\xe4\xbd\xa0\xe5\xa5\xbd");
}

TEST(JsonEscape, EmojiPassThrough) {
  // U+1F600 (grinning face) in UTF-8: F0 9F 98 80.
  EXPECT_EQ(EscapeRaw("\xf0\x9f\x98\x80"), "\xf0\x9f\x98\x80");
}

TEST(JsonEscape, TwoByteUtf8PassThrough) {
  // U+00E9 (e with acute) in UTF-8: C3 A9.
  EXPECT_EQ(EscapeRaw("\xc3\xa9"), "\xc3\xa9");
}

//===----------------------------------------------------------------------===//
// U+2028/U+2029 escaping
//===----------------------------------------------------------------------===//

TEST(JsonEscape, LineSeparatorEscaped) {
  // U+2028 LINE SEPARATOR in UTF-8: E2 80 A8.
  EXPECT_EQ(EscapeRaw("a\xe2\x80\xa8"
                      "b"),
            "a\\u2028b");
}

TEST(JsonEscape, ParagraphSeparatorEscaped) {
  // U+2029 PARAGRAPH SEPARATOR in UTF-8: E2 80 A9.
  EXPECT_EQ(EscapeRaw("a\xe2\x80\xa9"
                      "b"),
            "a\\u2029b");
}

TEST(JsonEscape, BothSeparatorsInSequence) {
  EXPECT_EQ(EscapeRaw("\xe2\x80\xa8\xe2\x80\xa9"), "\\u2028\\u2029");
}

TEST(JsonEscape, E2NotFollowedBy80) {
  // E2 followed by something other than 80 — passes through as regular
  // UTF-8 (it's a valid 3-byte sequence start for U+2800-U+2FFF range).
  // E2 81 A0 = U+2060 (WORD JOINER) — should NOT be escaped.
  EXPECT_EQ(EscapeRaw("\xe2\x81\xa0"), "\xe2\x81\xa0");
}

TEST(JsonEscape, E280ButNotA8OrA9) {
  // E2 80 94 = U+2014 (EM DASH) — should NOT be escaped.
  EXPECT_EQ(EscapeRaw("\xe2\x80\x94"), "\xe2\x80\x94");
}

TEST(JsonEscape, InvalidUtf8BytesBecomeReplacementEscapes) {
  std::string input =
      "a\x80"
      "\xe2\x80"
      "b";
  EXPECT_EQ(EscapeRaw(iree_make_string_view(input.data(), input.size())),
            "a\\ufffd\\ufffd\\ufffdb");
}

//===----------------------------------------------------------------------===//
// Mixed content
//===----------------------------------------------------------------------===//

TEST(JsonEscape, MixedContent) {
  // ASCII + control chars + Chinese + U+2028 + emoji.
  std::string input = "hello\t\xe4\xbd\xa0\xe2\x80\xa8\xf0\x9f\x98\x80";
  std::string expected = "hello\\t\xe4\xbd\xa0\\u2028\xf0\x9f\x98\x80";
  EXPECT_EQ(EscapeRaw(input.c_str()), expected);
}

TEST(JsonEscape, QuotedStringWithEscapes) {
  EXPECT_EQ(EscapeQuoted("line1\nline2"), "\"line1\\nline2\"");
}

//===----------------------------------------------------------------------===//
// Streaming containers
//===----------------------------------------------------------------------===//

TEST(JsonWriter, WritesNestedContainersAndTypedValues) {
  std::string json = WriteStructuredJson();
  EXPECT_EQ(json,
            "{\"required\":\"\",\"present\":\"value\",\"unknown\":null,"
            "\"answer\":42,"
            "\"escaped\\\"field\":[-7,true,null,{\"name\":\"nested\"}]}");

  iree_string_view_t object = iree_make_string_view(json.data(), json.size());
  iree_string_view_t value;
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(object, IREE_SV("required"), &value));
  EXPECT_TRUE(iree_string_view_is_empty(value));
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(object, IREE_SV("unknown"), &value));
  EXPECT_TRUE(iree_string_view_equal(value, IREE_SV("null")));
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(object, IREE_SV("answer"), &value));
  uint64_t answer = 0;
  IREE_ASSERT_OK(iree_json_parse_uint64(value, &answer));
  EXPECT_EQ(answer, 42u);
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(object, IREE_SV("present"), &value));
  EXPECT_TRUE(iree_string_view_equal(value, IREE_SV("value")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_json_lookup_object_value(object, IREE_SV("omitted"), &value));
}

TEST(JsonWriter, WritesEmptyContainers) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  loom_json_object_writer_t object;
  IREE_ASSERT_OK(loom_json_object_begin(&stream, &object));
  IREE_ASSERT_OK(loom_json_object_begin_field(&object, IREE_SV("object")));
  loom_json_object_writer_t empty_object;
  IREE_ASSERT_OK(loom_json_object_begin(object.stream, &empty_object));
  IREE_ASSERT_OK(loom_json_object_end(&empty_object));
  IREE_ASSERT_OK(loom_json_object_begin_field(&object, IREE_SV("array")));
  loom_json_array_writer_t empty_array;
  IREE_ASSERT_OK(loom_json_array_begin(object.stream, &empty_array));
  IREE_ASSERT_OK(loom_json_array_end(&empty_array));
  IREE_ASSERT_OK(loom_json_object_end(&object));

  iree_string_view_t view = iree_string_builder_view(&builder);
  EXPECT_EQ(std::string(view.data, view.size), "{\"object\":{},\"array\":[]}");
  iree_string_builder_deinitialize(&builder);
}

TEST(JsonWriter, DefersValueListsForArrayEmbedding) {
  loom_json_value_list_t values;
  loom_json_value_list_initialize(iree_allocator_system(), &values);

  loom_output_stream_t value_stream;
  IREE_ASSERT_OK(loom_json_value_list_begin_value(&values, &value_stream));
  loom_json_object_writer_t first;
  IREE_ASSERT_OK(loom_json_object_begin(&value_stream, &first));
  IREE_ASSERT_OK(
      loom_json_object_write_uint32_field(&first, IREE_SV("index"), 1));
  IREE_ASSERT_OK(loom_json_object_end(&first));
  IREE_ASSERT_OK(loom_json_value_list_begin_value(&values, &value_stream));
  loom_json_object_writer_t second;
  IREE_ASSERT_OK(loom_json_object_begin(&value_stream, &second));
  IREE_ASSERT_OK(
      loom_json_object_write_uint32_field(&second, IREE_SV("index"), 2));
  IREE_ASSERT_OK(loom_json_object_end(&second));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_json_value_list_write_array(&values, &stream));
  EXPECT_EQ(std::string(iree_string_builder_buffer(&builder),
                        iree_string_builder_size(&builder)),
            "[{\"index\":1},{\"index\":2}]");

  iree_string_builder_deinitialize(&builder);
  loom_json_value_list_deinitialize(&values);
}

//===----------------------------------------------------------------------===//
// IREE status objects
//===----------------------------------------------------------------------===//

TEST(JsonStatus, WritesCanonicalObject) {
  EXPECT_EQ(WriteStatusObject(IREE_STATUS_UNAVAILABLE,
                              IREE_SV("profile \"decode\" failed")),
            "{\"code\":14,\"name\":\"UNAVAILABLE\","
            "\"message\":\"profile \\\"decode\\\" failed\"}");
}

TEST(JsonStatus, OmitsUnavailableMessage) {
  EXPECT_EQ(WriteStatusObject(IREE_STATUS_OK, iree_string_view_empty()),
            "{\"code\":0,\"name\":\"OK\"}");
}

//===----------------------------------------------------------------------===//
// Stream offset tracking
//===----------------------------------------------------------------------===//

TEST(JsonEscape, OffsetTracking) {
  // The outer stream's offset should reflect the escaped output length,
  // not the input length.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  IREE_ASSERT_OK(loom_json_write_escaped_cstring(&stream, "a\"b"));
  // Output: "a\"b" = 6 bytes (quote, a, backslash, quote, b, quote).
  EXPECT_EQ(stream.offset, 6u);
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace loom
