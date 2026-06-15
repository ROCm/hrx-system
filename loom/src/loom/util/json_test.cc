// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/json.h"

#include <string>

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
