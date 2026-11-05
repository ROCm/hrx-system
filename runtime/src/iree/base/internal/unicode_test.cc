// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/unicode.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

//===----------------------------------------------------------------------===//
// UTF-8 Codec Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeUtf8Test, DecodeAscii) {
  iree_string_view_t text = iree_make_cstring_view("Hello");
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'H');
  EXPECT_EQ(position, 1u);
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'e');
  EXPECT_EQ(position, 2u);
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'l');
  EXPECT_EQ(position, 3u);
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'l');
  EXPECT_EQ(position, 4u);
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 'o');
  EXPECT_EQ(position, 5u);
}

TEST(UnicodeUtf8Test, DecodeTwoByteSequence) {
  // "é" is U+00E9, encoded as 0xC3 0xA9
  const char data[] = "\xC3\xA9";
  iree_string_view_t text = iree_make_string_view(data, 2);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 0x00E9u);
  EXPECT_EQ(position, 2u);
}

TEST(UnicodeUtf8Test, DecodeThreeByteSequence) {
  // "中" is U+4E2D, encoded as 0xE4 0xB8 0xAD
  const char data[] = "\xE4\xB8\xAD";
  iree_string_view_t text = iree_make_string_view(data, 3);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 0x4E2Du);
  EXPECT_EQ(position, 3u);
}

TEST(UnicodeUtf8Test, DecodeFourByteSequence) {
  // U+1F600 (grinning face emoji), encoded as 0xF0 0x9F 0x98 0x80
  const char data[] = "\xF0\x9F\x98\x80";
  iree_string_view_t text = iree_make_string_view(data, 4);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position), 0x1F600u);
  EXPECT_EQ(position, 4u);
}

TEST(UnicodeUtf8Test, DecodeInvalidSequence) {
  // Invalid continuation byte.
  const char data[] = "\xC3\x00";
  iree_string_view_t text = iree_make_string_view(data, 2);
  iree_host_size_t position = 0;

  // Should return replacement character and advance by 1.
  EXPECT_EQ(iree_unicode_utf8_decode(text, &position),
            IREE_UNICODE_REPLACEMENT_CHAR);
  EXPECT_EQ(position, 1u);
}

TEST(UnicodeUtf8Test, DecodeOverlongSequence) {
  // Overlong encoding of ASCII 'A' (0x41) as 2 bytes: 0xC1 0x81
  const char data[] = "\xC1\x81";
  iree_string_view_t text = iree_make_string_view(data, 2);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position),
            IREE_UNICODE_REPLACEMENT_CHAR);
}

TEST(UnicodeUtf8Test, DecodeSurrogateHalf) {
  // U+D800 (surrogate) encoded as 0xED 0xA0 0x80 - invalid in UTF-8.
  const char data[] = "\xED\xA0\x80";
  iree_string_view_t text = iree_make_string_view(data, 3);
  iree_host_size_t position = 0;

  EXPECT_EQ(iree_unicode_utf8_decode(text, &position),
            IREE_UNICODE_REPLACEMENT_CHAR);
}

TEST(UnicodeUtf8Test, EncodeAscii) {
  char buffer[4];
  EXPECT_EQ(iree_unicode_utf8_encode('A', buffer), 1);
  EXPECT_EQ(buffer[0], 'A');
}

TEST(UnicodeUtf8Test, EncodeTwoByteSequence) {
  char buffer[4];
  EXPECT_EQ(iree_unicode_utf8_encode(0x00E9, buffer), 2);
  EXPECT_EQ((uint8_t)buffer[0], 0xC3);
  EXPECT_EQ((uint8_t)buffer[1], 0xA9);
}

TEST(UnicodeUtf8Test, EncodeThreeByteSequence) {
  char buffer[4];
  EXPECT_EQ(iree_unicode_utf8_encode(0x4E2D, buffer), 3);
  EXPECT_EQ((uint8_t)buffer[0], 0xE4);
  EXPECT_EQ((uint8_t)buffer[1], 0xB8);
  EXPECT_EQ((uint8_t)buffer[2], 0xAD);
}

TEST(UnicodeUtf8Test, EncodeFourByteSequence) {
  char buffer[4];
  EXPECT_EQ(iree_unicode_utf8_encode(0x1F600, buffer), 4);
  EXPECT_EQ((uint8_t)buffer[0], 0xF0);
  EXPECT_EQ((uint8_t)buffer[1], 0x9F);
  EXPECT_EQ((uint8_t)buffer[2], 0x98);
  EXPECT_EQ((uint8_t)buffer[3], 0x80);
}

TEST(UnicodeUtf8Test, EncodeInvalidCodepoint) {
  char buffer[4];
  // Beyond max codepoint.
  EXPECT_EQ(iree_unicode_utf8_encode(0x110000, buffer), 0);
  // Surrogate half.
  EXPECT_EQ(iree_unicode_utf8_encode(0xD800, buffer), 0);
}

TEST(UnicodeUtf8Test, RoundTrip) {
  std::vector<uint32_t> codepoints = {'A', 0x00E9, 0x4E2D, 0x1F600};
  for (uint32_t codepoint : codepoints) {
    char buffer[4];
    int length = iree_unicode_utf8_encode(codepoint, buffer);
    ASSERT_GT(length, 0);

    iree_string_view_t text = iree_make_string_view(buffer, length);
    iree_host_size_t position = 0;
    EXPECT_EQ(iree_unicode_utf8_decode(text, &position), codepoint);
    EXPECT_EQ(position, (iree_host_size_t)length);
  }
}

TEST(UnicodeUtf8Test, CodepointCount) {
  // ASCII only.
  EXPECT_EQ(iree_unicode_utf8_codepoint_count(iree_make_cstring_view("Hello")),
            5u);

  // Mixed: "Héllo中" - H, é (2 bytes), l, l, o, 中 (3 bytes)
  const char mixed[] = "H\xC3\xA9llo\xE4\xB8\xAD";
  EXPECT_EQ(iree_unicode_utf8_codepoint_count(
                iree_make_string_view(mixed, sizeof(mixed) - 1)),
            6u);
}

TEST(UnicodeUtf8Test, Validate) {
  EXPECT_TRUE(iree_unicode_utf8_validate(iree_make_cstring_view("Hello")));
  EXPECT_TRUE(iree_unicode_utf8_validate(iree_make_cstring_view("")));

  // Valid UTF-8 with multi-byte sequences.
  const char valid[] = "\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80";
  EXPECT_TRUE(iree_unicode_utf8_validate(
      iree_make_string_view(valid, sizeof(valid) - 1)));

  // Invalid: truncated sequence.
  const char invalid[] = "\xC3";
  EXPECT_FALSE(iree_unicode_utf8_validate(
      iree_make_string_view(invalid, sizeof(invalid) - 1)));
}

TEST(UnicodeUtf8Test, EncodedLength) {
  EXPECT_EQ(iree_unicode_utf8_encoded_length('A'), 1);
  EXPECT_EQ(iree_unicode_utf8_encoded_length(0x00E9), 2);
  EXPECT_EQ(iree_unicode_utf8_encoded_length(0x4E2D), 3);
  EXPECT_EQ(iree_unicode_utf8_encoded_length(0x1F600), 4);
  EXPECT_EQ(iree_unicode_utf8_encoded_length(0x110000), 0);  // Invalid
}

TEST(UnicodeUtf8Test, SequenceLength) {
  // ASCII (0x00-0x7F): 1 byte.
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0x00), 1u);  // Null byte.
  EXPECT_EQ(iree_unicode_utf8_sequence_length('A'), 1u);
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0x7F), 1u);

  // 2-byte lead (0xC0-0xDF): 2 bytes.
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xC3), 2u);  // Start of é.
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xDF), 2u);

  // 3-byte lead (0xE0-0xEF): 3 bytes.
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xE4), 3u);  // Start of 中.
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xEF), 3u);

  // 4-byte lead (0xF0-0xF7): 4 bytes.
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xF0), 4u);  // Start of emoji.
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xF4), 4u);

  // Continuation bytes (0x80-0xBF): return 1 (advance past invalid).
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0x80), 1u);
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xBF), 1u);
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xA9), 1u);  // Part of é.

  // Invalid lead bytes (0xF8+): return 1.
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xF8), 1u);
  EXPECT_EQ(iree_unicode_utf8_sequence_length(0xFF), 1u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthEmpty) {
  // Empty buffer has no incomplete tail.
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length("", 0), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthCompleteAscii) {
  // Complete ASCII strings have no incomplete tail.
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length("Hello", 5), 0u);
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length("x", 1), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthCompleteTwoByte) {
  // Complete 2-byte UTF-8 sequence "é" (U+00E9 = 0xC3 0xA9).
  const char complete[] = "\xC3\xA9";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(complete, 2), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthCompleteThreeByte) {
  // Complete 3-byte UTF-8 sequence "中" (U+4E2D = 0xE4 0xB8 0xAD).
  const char complete[] = "\xE4\xB8\xAD";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(complete, 3), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthCompleteFourByte) {
  // Complete 4-byte UTF-8 sequence U+1F600 (grinning face emoji).
  const char complete[] = "\xF0\x9F\x98\x80";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(complete, 4), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthIncompleteTwoByte) {
  // Incomplete 2-byte sequence: only lead byte present (0xC3).
  const char incomplete[] = "abc\xC3";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete, 4), 1u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthIncompleteThreeByte) {
  // Incomplete 3-byte sequence: lead + 1 continuation (0xE4 0xB8).
  const char incomplete1[] = "abc\xE4\xB8";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete1, 5), 2u);

  // Incomplete 3-byte sequence: only lead byte (0xE4).
  const char incomplete2[] = "abc\xE4";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete2, 4), 1u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthIncompleteFourByte) {
  // Incomplete 4-byte sequence: lead + 2 continuations (0xF0 0x9F 0x98).
  const char incomplete1[] = "abc\xF0\x9F\x98";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete1, 6), 3u);

  // Incomplete 4-byte sequence: lead + 1 continuation (0xF0 0x9F).
  const char incomplete2[] = "abc\xF0\x9F";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete2, 5), 2u);

  // Incomplete 4-byte sequence: only lead byte (0xF0).
  const char incomplete3[] = "abc\xF0";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(incomplete3, 4), 1u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthOrphanContinuations) {
  // Four continuation bytes with no lead byte - malformed, treated as complete.
  const char orphans[] = "\x80\x80\x80\x80";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(orphans, 4), 0u);
}

TEST(UnicodeUtf8Test, IncompleteTailLengthInvalidLeadByte) {
  // Invalid lead byte 0xFF followed by nothing - treated as complete.
  const char invalid[] = "abc\xFF";
  EXPECT_EQ(iree_unicode_utf8_incomplete_tail_length(invalid, 4), 0u);
}

//===----------------------------------------------------------------------===//
// Category Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeCategoryTest, AsciiLetters) {
  for (uint32_t codepoint = 'A'; codepoint <= 'Z'; ++codepoint) {
    EXPECT_TRUE(iree_unicode_is_letter(codepoint))
        << "U+" << std::hex << codepoint;
  }
  for (uint32_t codepoint = 'a'; codepoint <= 'z'; ++codepoint) {
    EXPECT_TRUE(iree_unicode_is_letter(codepoint))
        << "U+" << std::hex << codepoint;
  }
}

TEST(UnicodeCategoryTest, AsciiNumbers) {
  for (uint32_t codepoint = '0'; codepoint <= '9'; ++codepoint) {
    EXPECT_TRUE(iree_unicode_is_number(codepoint))
        << "U+" << std::hex << codepoint;
  }
}

TEST(UnicodeCategoryTest, AsciiPunctuation) {
  EXPECT_TRUE(iree_unicode_is_punctuation('.'));
  EXPECT_TRUE(iree_unicode_is_punctuation(','));
  EXPECT_TRUE(iree_unicode_is_punctuation('!'));
  EXPECT_TRUE(iree_unicode_is_punctuation('?'));
  EXPECT_TRUE(iree_unicode_is_punctuation(';'));
  EXPECT_TRUE(iree_unicode_is_punctuation(':'));
  EXPECT_TRUE(iree_unicode_is_punctuation('\''));
  EXPECT_TRUE(iree_unicode_is_punctuation('"'));
  EXPECT_TRUE(iree_unicode_is_punctuation('('));
  EXPECT_TRUE(iree_unicode_is_punctuation(')'));
  EXPECT_TRUE(iree_unicode_is_punctuation('['));
  EXPECT_TRUE(iree_unicode_is_punctuation(']'));
  EXPECT_TRUE(iree_unicode_is_punctuation('{'));
  EXPECT_TRUE(iree_unicode_is_punctuation('}'));
}

TEST(UnicodeCategoryTest, AsciiSymbols) {
  EXPECT_TRUE(iree_unicode_is_symbol('$'));
  EXPECT_TRUE(iree_unicode_is_symbol('+'));
  EXPECT_TRUE(iree_unicode_is_symbol('<'));
  EXPECT_TRUE(iree_unicode_is_symbol('='));
  EXPECT_TRUE(iree_unicode_is_symbol('>'));
  EXPECT_TRUE(iree_unicode_is_symbol('|'));
  EXPECT_TRUE(iree_unicode_is_symbol('~'));
}

TEST(UnicodeCategoryTest, AsciiControl) {
  EXPECT_TRUE(iree_unicode_is_control(0x00));  // NUL
  EXPECT_TRUE(iree_unicode_is_control(0x1F));  // US
  EXPECT_TRUE(iree_unicode_is_control(0x7F));  // DEL
  EXPECT_FALSE(iree_unicode_is_control(' '));
  EXPECT_FALSE(iree_unicode_is_control('A'));
}

TEST(UnicodeCategoryTest, AsciiWhitespace) {
  EXPECT_TRUE(iree_unicode_is_whitespace(' '));
  EXPECT_TRUE(iree_unicode_is_whitespace('\t'));
  EXPECT_TRUE(iree_unicode_is_whitespace('\n'));
  EXPECT_TRUE(iree_unicode_is_whitespace('\r'));
  EXPECT_TRUE(iree_unicode_is_whitespace('\f'));
  EXPECT_TRUE(iree_unicode_is_whitespace('\v'));
  EXPECT_FALSE(iree_unicode_is_whitespace('A'));
  EXPECT_FALSE(iree_unicode_is_whitespace('0'));
}

TEST(UnicodeCategoryTest, NonAsciiWhitespace) {
  EXPECT_TRUE(iree_unicode_is_whitespace(0x00A0));  // No-Break Space
  EXPECT_TRUE(iree_unicode_is_whitespace(0x2000));  // En Quad
  EXPECT_TRUE(iree_unicode_is_whitespace(0x2003));  // Em Space
  EXPECT_TRUE(iree_unicode_is_whitespace(0x3000));  // Ideographic Space
}

TEST(UnicodeCategoryTest, CjkLetters) {
  EXPECT_TRUE(iree_unicode_is_letter(0x4E00));  // 一 (first CJK)
  EXPECT_TRUE(iree_unicode_is_letter(0x4E2D));  // 中
  EXPECT_TRUE(iree_unicode_is_letter(0x9FFF));  // Last CJK in basic block
}

TEST(UnicodeCategoryTest, GreekLetters) {
  EXPECT_TRUE(iree_unicode_is_letter(0x0391));  // Α (Alpha)
  EXPECT_TRUE(iree_unicode_is_letter(0x03B1));  // α (alpha)
  EXPECT_TRUE(iree_unicode_is_letter(0x03A9));  // Ω (Omega)
  EXPECT_TRUE(iree_unicode_is_letter(0x03C9));  // ω (omega)
}

TEST(UnicodeCategoryTest, CombiningMarks) {
  EXPECT_TRUE(iree_unicode_is_mark(0x0300));  // Combining Grave Accent
  EXPECT_TRUE(iree_unicode_is_mark(0x0301));  // Combining Acute Accent
  EXPECT_TRUE(iree_unicode_is_mark(0x0302));  // Combining Circumflex Accent
}

TEST(UnicodeCategoryTest, MarkNonspacing) {
  // iree_unicode_is_mark_nonspacing() returns true for Mn (Mark, Nonspacing)
  // category, and false for Mc (Spacing Combining) and Me (Enclosing).

  // Combining Diacritical Marks (U+0300-U+036F) - all Mn.
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x0300));  // Grave accent
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x0301));  // Acute accent
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x0308));  // Diaeresis
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x036F));  // Last in block

  // Hebrew cantillation marks (Mn).
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x0591));  // Etnahta
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x05B0));  // Sheva

  // Arabic marks (Mn).
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x0610));  // Sallallahou
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x064B));  // Fathatan

  // Khmer combining marks (Mn).
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x17D2));  // Coeng (virama)

  // Combining Diacritical Marks Supplement (U+1DC0-U+1DFF) - Mn.
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x1DC0));
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x1DFF));

  // Combining Half Marks (U+FE20-U+FE2F) - Mn.
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0xFE20));
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0xFE2F));

  // Sharada Sign Virama (U+111B6) - Mn.
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x111B6));

  // Musical Symbol Combining Stem (U+1D165) - Mc (NOT Mn).
  EXPECT_FALSE(iree_unicode_is_mark_nonspacing(0x1D165));

  // Grantha Vowel Sign AU (U+1134C) - Mc (NOT Mn) - the original bug case.
  EXPECT_FALSE(iree_unicode_is_mark_nonspacing(0x1134C));

  // Devanagari vowel signs: some are Mn, some are Mc.
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x0941));   // Mn: Vowel Sign U
  EXPECT_FALSE(iree_unicode_is_mark_nonspacing(0x093E));  // Mc: Vowel Sign AA

  // Bengali: some Mn, some Mc.
  EXPECT_TRUE(iree_unicode_is_mark_nonspacing(0x09C1));   // Mn: Vowel Sign U
  EXPECT_FALSE(iree_unicode_is_mark_nonspacing(0x09BE));  // Mc: Vowel Sign AA

  // Non-marks should return false.
  EXPECT_FALSE(iree_unicode_is_mark_nonspacing('A'));
  EXPECT_FALSE(iree_unicode_is_mark_nonspacing(0x4E2D));  // CJK
  EXPECT_FALSE(iree_unicode_is_mark_nonspacing(' '));
}

TEST(UnicodeCategoryTest, C1ControlCharacters) {
  // C1 controls (0x80-0x9F) should be detected as control characters.
  EXPECT_TRUE(iree_unicode_is_control(0x80));
  EXPECT_TRUE(iree_unicode_is_control(0x85));  // NEL (Next Line)
  EXPECT_TRUE(iree_unicode_is_control(0x9F));
  EXPECT_FALSE(iree_unicode_is_control(0xA0));  // NBSP is not a control
}

TEST(UnicodeCategoryTest, InvisibleFormatCharacters) {
  // Zero-width characters (U+200B-U+200F).
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x200B));  // ZWSP
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x200C));  // ZWNJ
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x200D));  // ZWJ
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x200E));  // LRM
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x200F));  // RLM

  // Directional formatting (U+202A-U+202E).
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x202A));  // LRE
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x202B));  // RLE
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x202C));  // PDF
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x202D));  // LRO
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x202E));  // RLO

  // Word joiner and invisible operators (U+2060-U+2064).
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x2060));  // Word Joiner
  EXPECT_TRUE(
      iree_unicode_is_invisible_format(0x2061));  // Function Application
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x2062));  // Invisible Times
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x2063));  // Invisible Separator
  EXPECT_TRUE(iree_unicode_is_invisible_format(0x2064));  // Invisible Plus

  // Byte Order Mark (U+FEFF).
  EXPECT_TRUE(iree_unicode_is_invisible_format(0xFEFF));  // BOM

  // Characters that should NOT be invisible format.
  EXPECT_FALSE(iree_unicode_is_invisible_format(' '));  // Regular space
  EXPECT_FALSE(iree_unicode_is_invisible_format('A'));  // ASCII letter
  EXPECT_FALSE(
      iree_unicode_is_invisible_format(0x00A0));  // NBSP (visible space)
  EXPECT_FALSE(iree_unicode_is_invisible_format(0x3000));  // Ideographic Space
  EXPECT_FALSE(
      iree_unicode_is_invisible_format(0x00));  // NUL (control, not format)
  EXPECT_FALSE(iree_unicode_is_invisible_format(0x1F));  // Control character
  EXPECT_FALSE(iree_unicode_is_invisible_format(0x7F));  // DEL (control)
  EXPECT_FALSE(
      iree_unicode_is_invisible_format(0x0301));  // Combining Acute (mark)
}

TEST(UnicodeCategoryTest, HanCharacters) {
  // Han characters: CJK Unified Ideographs (U+4E00-U+9FFF).
  EXPECT_TRUE(iree_unicode_is_han(0x4E00));  // First CJK Unified Ideograph
  EXPECT_TRUE(iree_unicode_is_han(0x4E2D));  // 中 (middle)
  EXPECT_TRUE(iree_unicode_is_han(0x6587));  // 文 (text)
  EXPECT_TRUE(iree_unicode_is_han(0x9FFF));  // Last CJK Unified Ideograph

  // CJK Extension A (U+3400-U+4DBF).
  EXPECT_TRUE(iree_unicode_is_han(0x3400));
  EXPECT_TRUE(iree_unicode_is_han(0x4DBF));

  // CJK Extension B (U+20000-U+2A6DF).
  EXPECT_TRUE(iree_unicode_is_han(0x20000));
  EXPECT_TRUE(iree_unicode_is_han(0x2A6DF));

  // CJK Compatibility Ideographs (U+F900-U+FAFF).
  EXPECT_TRUE(iree_unicode_is_han(0xF900));
  EXPECT_TRUE(iree_unicode_is_han(0xFAFF));

  // Non-CJK characters.
  EXPECT_FALSE(iree_unicode_is_han('A'));     // ASCII
  EXPECT_FALSE(iree_unicode_is_han(0x3041));  // Hiragana あ
  EXPECT_FALSE(iree_unicode_is_han(0x30A1));  // Katakana ア
  EXPECT_FALSE(iree_unicode_is_han(0xAC00));  // Hangul 가
}

TEST(UnicodeCategoryTest, NonAsciiPunctuation) {
  EXPECT_TRUE(iree_unicode_is_punctuation(0x00A1));  // ¡ Inverted Exclamation
  EXPECT_TRUE(iree_unicode_is_punctuation(0x00BF));  // ¿ Inverted Question
  EXPECT_TRUE(iree_unicode_is_punctuation(0x2010));  // ‐ Hyphen
  EXPECT_TRUE(iree_unicode_is_punctuation(0x2014));  // — Em Dash
  EXPECT_TRUE(iree_unicode_is_punctuation(0x2018));  // ' Left Single Quote
  EXPECT_TRUE(iree_unicode_is_punctuation(0x201C));  // " Left Double Quote
}

TEST(UnicodeCategoryTest, NonAsciiSymbols) {
  EXPECT_TRUE(iree_unicode_is_symbol(0x00A2));  // ¢ Cent Sign
  EXPECT_TRUE(iree_unicode_is_symbol(0x00A3));  // £ Pound Sign
  EXPECT_TRUE(iree_unicode_is_symbol(0x00A5));  // ¥ Yen Sign
  EXPECT_TRUE(iree_unicode_is_symbol(0x00A9));  // © Copyright
  EXPECT_TRUE(iree_unicode_is_symbol(0x00AE));  // ® Registered
  EXPECT_TRUE(iree_unicode_is_symbol(0x2122));  // ™ Trade Mark
}

TEST(UnicodeCategoryTest, Separators) {
  EXPECT_TRUE(iree_unicode_is_separator(0x00A0));  // No-Break Space
  EXPECT_TRUE(iree_unicode_is_separator(0x2028));  // Line Separator
  EXPECT_TRUE(iree_unicode_is_separator(0x2029));  // Paragraph Separator
  EXPECT_TRUE(iree_unicode_is_separator(0x202F));  // Narrow No-Break Space
  EXPECT_TRUE(iree_unicode_is_separator(0x3000));  // Ideographic Space
}

TEST(UnicodeCategoryTest, OtherCategory) {
  // Format characters (Cf) are in the Other category.
  EXPECT_TRUE(iree_unicode_is_other(0x200B));  // Zero Width Space
  EXPECT_TRUE(iree_unicode_is_other(0x200C));  // Zero Width Non-Joiner
  EXPECT_TRUE(iree_unicode_is_other(0x200D));  // Zero Width Joiner
  EXPECT_TRUE(iree_unicode_is_other(0xFEFF));  // BOM / ZWNBSP
}

TEST(UnicodeCategoryTest, TableBoundaries) {
  // Test first and last entries in various Unicode blocks to ensure
  // binary search handles boundaries correctly.

  // Latin Extended-A block (0x0100-0x017F) - all letters.
  EXPECT_TRUE(iree_unicode_is_letter(0x0100));  // First
  EXPECT_TRUE(iree_unicode_is_letter(0x017F));  // Last

  // Latin Extended-B block (0x0180-0x024F) - all letters.
  EXPECT_TRUE(iree_unicode_is_letter(0x0180));  // First
  EXPECT_TRUE(iree_unicode_is_letter(0x024F));  // Last

  // Combining Diacritical Marks (0x0300-0x036F) - all marks.
  EXPECT_TRUE(iree_unicode_is_mark(0x0300));  // First
  EXPECT_TRUE(iree_unicode_is_mark(0x036F));  // Last

  // General Punctuation (0x2000-0x206F) - mixed categories.
  EXPECT_TRUE(iree_unicode_is_separator(0x2000));    // En Quad (space)
  EXPECT_TRUE(iree_unicode_is_punctuation(0x2010));  // Hyphen
}

TEST(UnicodeCategoryTest, HighCodepoints) {
  // Emoji are symbols (So category) in the Supplementary Multilingual Plane.
  EXPECT_TRUE(iree_unicode_is_symbol(0x1F600));  // 😀 Grinning Face
  EXPECT_TRUE(iree_unicode_is_symbol(0x1F4A9));  // 💩 Pile of Poo
  EXPECT_TRUE(iree_unicode_is_symbol(0x2764));   // ❤ Heavy Black Heart

  // Mathematical Alphanumeric Symbols (1D400-1D7FF) are letters.
  EXPECT_TRUE(
      iree_unicode_is_letter(0x1D400));  // 𝐀 Mathematical Bold Capital A
  EXPECT_TRUE(iree_unicode_is_letter(0x1D41A));  // 𝐚 Mathematical Bold Small A

  // Musical Symbols (1D100-1D1FF) are symbols.
  EXPECT_TRUE(iree_unicode_is_symbol(0x1D11E));  // 𝄞 Musical Symbol G Clef
}

TEST(UnicodeCategoryTest, UnassignedCodepoints) {
  // Unassigned codepoints (Cn) should return OTHER category.
  // These are gaps in the Unicode assignment that may be filled in future
  // versions.
  EXPECT_TRUE(iree_unicode_is_other(0x0378));    // Unassigned in Greek block
  EXPECT_TRUE(iree_unicode_is_other(0x0530));    // Unassigned in Armenian block
  EXPECT_TRUE(iree_unicode_is_other(0xFFFF));    // Noncharacter (still OTHER)
  EXPECT_TRUE(iree_unicode_is_other(0x10FFFF));  // Max codepoint (unassigned)
}

TEST(UnicodeCategoryTest, Latin1SupplementCategories) {
  // Verify the Latin-1 Supplement fast path table (U+0080-U+00FF) produces
  // correct categories for every codepoint in the range.

  // U+0080-009F: C1 Controls (Cc) → OTHER.
  for (uint32_t cp = 0x80; cp <= 0x9F; ++cp) {
    EXPECT_EQ(iree_unicode_category(cp), IREE_UNICODE_CATEGORY_OTHER)
        << "U+" << std::hex << cp;
  }

  // U+00A0: No-Break Space (Zs) → SEPARATOR.
  EXPECT_EQ(iree_unicode_category(0x00A0), IREE_UNICODE_CATEGORY_SEPARATOR);

  // U+00A1: ¡ Inverted Exclamation Mark (Po) → PUNCTUATION.
  EXPECT_EQ(iree_unicode_category(0x00A1), IREE_UNICODE_CATEGORY_PUNCTUATION);

  // U+00A2-00A6: Currency/Other symbols (Sc/So) → SYMBOL.
  for (uint32_t cp = 0xA2; cp <= 0xA6; ++cp) {
    EXPECT_EQ(iree_unicode_category(cp), IREE_UNICODE_CATEGORY_SYMBOL)
        << "U+" << std::hex << cp;
  }

  // U+00A7: § Section Sign (Po) → PUNCTUATION.
  EXPECT_EQ(iree_unicode_category(0x00A7), IREE_UNICODE_CATEGORY_PUNCTUATION);

  // U+00A8-00A9: Modifier/Other symbols (Sk/So) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00A8), IREE_UNICODE_CATEGORY_SYMBOL);
  EXPECT_EQ(iree_unicode_category(0x00A9), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00AA: ª Feminine Ordinal (Lo) → LETTER.
  EXPECT_EQ(iree_unicode_category(0x00AA), IREE_UNICODE_CATEGORY_LETTER);

  // U+00AB: « Left-Pointing Double Angle (Pi) → PUNCTUATION.
  EXPECT_EQ(iree_unicode_category(0x00AB), IREE_UNICODE_CATEGORY_PUNCTUATION);

  // U+00AC: ¬ Not Sign (Sm) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00AC), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00AD: Soft Hyphen (Cf) → OTHER.
  EXPECT_EQ(iree_unicode_category(0x00AD), IREE_UNICODE_CATEGORY_OTHER);

  // U+00AE-00AF: ® ¯ (So/Sk) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00AE), IREE_UNICODE_CATEGORY_SYMBOL);
  EXPECT_EQ(iree_unicode_category(0x00AF), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00B0-00B1: ° ± (So/Sm) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00B0), IREE_UNICODE_CATEGORY_SYMBOL);
  EXPECT_EQ(iree_unicode_category(0x00B1), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00B2-00B3: ² ³ (No) → NUMBER.
  EXPECT_EQ(iree_unicode_category(0x00B2), IREE_UNICODE_CATEGORY_NUMBER);
  EXPECT_EQ(iree_unicode_category(0x00B3), IREE_UNICODE_CATEGORY_NUMBER);

  // U+00B4: ´ Acute Accent (Sk) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00B4), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00B5: µ Micro Sign (Ll) → LETTER.
  EXPECT_EQ(iree_unicode_category(0x00B5), IREE_UNICODE_CATEGORY_LETTER);

  // U+00B6: ¶ Pilcrow Sign (So) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00B6), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00B7: · Middle Dot (Po) → PUNCTUATION.
  EXPECT_EQ(iree_unicode_category(0x00B7), IREE_UNICODE_CATEGORY_PUNCTUATION);

  // U+00B8: ¸ Cedilla (Sk) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00B8), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00B9: ¹ Superscript One (No) → NUMBER.
  EXPECT_EQ(iree_unicode_category(0x00B9), IREE_UNICODE_CATEGORY_NUMBER);

  // U+00BA: º Masculine Ordinal (Lo) → LETTER.
  EXPECT_EQ(iree_unicode_category(0x00BA), IREE_UNICODE_CATEGORY_LETTER);

  // U+00BB: » Right-Pointing Double Angle (Pf) → PUNCTUATION.
  EXPECT_EQ(iree_unicode_category(0x00BB), IREE_UNICODE_CATEGORY_PUNCTUATION);

  // U+00BC-00BE: ¼ ½ ¾ (No) → NUMBER.
  EXPECT_EQ(iree_unicode_category(0x00BC), IREE_UNICODE_CATEGORY_NUMBER);
  EXPECT_EQ(iree_unicode_category(0x00BD), IREE_UNICODE_CATEGORY_NUMBER);
  EXPECT_EQ(iree_unicode_category(0x00BE), IREE_UNICODE_CATEGORY_NUMBER);

  // U+00BF: ¿ Inverted Question Mark (Po) → PUNCTUATION.
  EXPECT_EQ(iree_unicode_category(0x00BF), IREE_UNICODE_CATEGORY_PUNCTUATION);

  // U+00C0-00D6: Latin uppercase letters (Lu) → LETTER.
  for (uint32_t cp = 0xC0; cp <= 0xD6; ++cp) {
    EXPECT_EQ(iree_unicode_category(cp), IREE_UNICODE_CATEGORY_LETTER)
        << "U+" << std::hex << cp;
  }

  // U+00D7: × Multiplication Sign (Sm) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00D7), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00D8-00F6: Latin letters (Lu/Ll) → LETTER.
  for (uint32_t cp = 0xD8; cp <= 0xF6; ++cp) {
    EXPECT_EQ(iree_unicode_category(cp), IREE_UNICODE_CATEGORY_LETTER)
        << "U+" << std::hex << cp;
  }

  // U+00F7: ÷ Division Sign (Sm) → SYMBOL.
  EXPECT_EQ(iree_unicode_category(0x00F7), IREE_UNICODE_CATEGORY_SYMBOL);

  // U+00F8-00FF: Latin lowercase letters (Ll) → LETTER.
  for (uint32_t cp = 0xF8; cp <= 0xFF; ++cp) {
    EXPECT_EQ(iree_unicode_category(cp), IREE_UNICODE_CATEGORY_LETTER)
        << "U+" << std::hex << cp;
  }
}

//===----------------------------------------------------------------------===//
// NFD Base Character Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeNfdTest, AsciiUnchanged) {
  EXPECT_EQ(iree_unicode_nfd_base('A'), 'A');
  EXPECT_EQ(iree_unicode_nfd_base('a'), 'a');
  EXPECT_EQ(iree_unicode_nfd_base('0'), '0');
}

TEST(UnicodeNfdTest, Latin1AccentStripping) {
  EXPECT_EQ(iree_unicode_nfd_base(0x00C0), 0x0041u);  // À → A
  EXPECT_EQ(iree_unicode_nfd_base(0x00C1), 0x0041u);  // Á → A
  EXPECT_EQ(iree_unicode_nfd_base(0x00C2), 0x0041u);  // Â → A
  EXPECT_EQ(iree_unicode_nfd_base(0x00C4), 0x0041u);  // Ä → A
  EXPECT_EQ(iree_unicode_nfd_base(0x00E0), 0x0061u);  // à → a
  EXPECT_EQ(iree_unicode_nfd_base(0x00E9), 0x0065u);  // é → e
  EXPECT_EQ(iree_unicode_nfd_base(0x00F1), 0x006Eu);  // ñ → n
  EXPECT_EQ(iree_unicode_nfd_base(0x00FC), 0x0075u);  // ü → u
}

TEST(UnicodeNfdTest, LatinExtendedA) {
  EXPECT_EQ(iree_unicode_nfd_base(0x0100), 0x0041u);  // Ā → A
  EXPECT_EQ(iree_unicode_nfd_base(0x0101), 0x0061u);  // ā → a
  EXPECT_EQ(iree_unicode_nfd_base(0x010C), 0x0043u);  // Č → C
  EXPECT_EQ(iree_unicode_nfd_base(0x010D), 0x0063u);  // č → c
  EXPECT_EQ(iree_unicode_nfd_base(0x0160), 0x0053u);  // Š → S
  EXPECT_EQ(iree_unicode_nfd_base(0x0161), 0x0073u);  // š → s
}

TEST(UnicodeNfdTest, NonDecomposable) {
  // Characters without NFD decomposition should return unchanged.
  EXPECT_EQ(iree_unicode_nfd_base(0x00C6), 0x00C6u);  // Æ (ligature, no decomp)
  EXPECT_EQ(iree_unicode_nfd_base(0x00D0), 0x00D0u);  // Ð (Eth)
  EXPECT_EQ(iree_unicode_nfd_base(0x00D8), 0x00D8u);  // Ø (O with stroke)
  EXPECT_EQ(iree_unicode_nfd_base(0x00DE), 0x00DEu);  // Þ (Thorn)
}

//===----------------------------------------------------------------------===//
// Canonical Combining Class (CCC) Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeCccTest, BaseCharactersHaveZeroCcc) {
  // Base characters (starters) have CCC=0.
  EXPECT_EQ(iree_unicode_ccc('A'), 0);
  EXPECT_EQ(iree_unicode_ccc('a'), 0);
  EXPECT_EQ(iree_unicode_ccc('0'), 0);
  EXPECT_EQ(iree_unicode_ccc(0x00E9), 0);  // é (precomposed, is a starter)
  EXPECT_EQ(iree_unicode_ccc(0x4E2D), 0);  // 中 (CJK)
}

TEST(UnicodeCccTest, CombiningDiacriticalMarks) {
  // Combining marks have non-zero CCC values.
  EXPECT_EQ(iree_unicode_ccc(0x0300), 230);  // Combining Grave Accent
  EXPECT_EQ(iree_unicode_ccc(0x0301), 230);  // Combining Acute Accent
  EXPECT_EQ(iree_unicode_ccc(0x0302), 230);  // Combining Circumflex Accent
  EXPECT_EQ(iree_unicode_ccc(0x0303), 230);  // Combining Tilde
  EXPECT_EQ(iree_unicode_ccc(0x0304), 230);  // Combining Macron
  EXPECT_EQ(iree_unicode_ccc(0x0308), 230);  // Combining Diaeresis
  EXPECT_EQ(iree_unicode_ccc(0x030A), 230);  // Combining Ring Above
  EXPECT_EQ(iree_unicode_ccc(0x030C), 230);  // Combining Caron
}

TEST(UnicodeCccTest, BelowMarks) {
  // Marks that attach below have lower CCC values.
  EXPECT_EQ(iree_unicode_ccc(0x0327), 202);  // Combining Cedilla
  EXPECT_EQ(iree_unicode_ccc(0x0328), 202);  // Combining Ogonek
}

TEST(UnicodeCccTest, DoubleMarks) {
  // Double diacritics (spanning two base characters).
  EXPECT_EQ(iree_unicode_ccc(0x035C), 233);  // Combining Double Breve Below
  EXPECT_EQ(iree_unicode_ccc(0x035D), 234);  // Combining Double Breve
  EXPECT_EQ(iree_unicode_ccc(0x035E), 234);  // Combining Double Macron
}

TEST(UnicodeCccTest, HebrewCombining) {
  // Hebrew combining marks have specific CCC values.
  EXPECT_EQ(iree_unicode_ccc(0x05B0), 10);  // Hebrew Point Sheva
  EXPECT_EQ(iree_unicode_ccc(0x05B1), 11);  // Hebrew Point Hataf Segol
  EXPECT_EQ(iree_unicode_ccc(0x05BC), 21);  // Hebrew Point Dagesh
  EXPECT_EQ(iree_unicode_ccc(0x05C1), 24);  // Hebrew Point Shin Dot
}

//===----------------------------------------------------------------------===//
// Composition Pair Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeComposePairTest, LatinCompositions) {
  // Common Latin letter + combining mark compositions.
  EXPECT_EQ(iree_unicode_compose_pair('A', 0x0300), 0x00C0u);  // A + grave → À
  EXPECT_EQ(iree_unicode_compose_pair('A', 0x0301), 0x00C1u);  // A + acute → Á
  EXPECT_EQ(iree_unicode_compose_pair('A', 0x0302),
            0x00C2u);  // A + circumflex → Â
  EXPECT_EQ(iree_unicode_compose_pair('A', 0x0303), 0x00C3u);  // A + tilde → Ã
  EXPECT_EQ(iree_unicode_compose_pair('A', 0x0308),
            0x00C4u);  // A + diaeresis → Ä
  EXPECT_EQ(iree_unicode_compose_pair('A', 0x030A),
            0x00C5u);  // A + ring above → Å
  EXPECT_EQ(iree_unicode_compose_pair('C', 0x0327),
            0x00C7u);  // C + cedilla → Ç
  EXPECT_EQ(iree_unicode_compose_pair('E', 0x0301), 0x00C9u);  // E + acute → É
  EXPECT_EQ(iree_unicode_compose_pair('N', 0x0303), 0x00D1u);  // N + tilde → Ñ
  EXPECT_EQ(iree_unicode_compose_pair('O', 0x0308),
            0x00D6u);  // O + diaeresis → Ö
  EXPECT_EQ(iree_unicode_compose_pair('U', 0x0308),
            0x00DCu);  // U + diaeresis → Ü
}

TEST(UnicodeComposePairTest, LowercaseLatinCompositions) {
  EXPECT_EQ(iree_unicode_compose_pair('a', 0x0300), 0x00E0u);  // a + grave → à
  EXPECT_EQ(iree_unicode_compose_pair('a', 0x0301), 0x00E1u);  // a + acute → á
  EXPECT_EQ(iree_unicode_compose_pair('e', 0x0301), 0x00E9u);  // e + acute → é
  EXPECT_EQ(iree_unicode_compose_pair('n', 0x0303), 0x00F1u);  // n + tilde → ñ
  EXPECT_EQ(iree_unicode_compose_pair('o', 0x0308),
            0x00F6u);  // o + diaeresis → ö
  EXPECT_EQ(iree_unicode_compose_pair('u', 0x0308),
            0x00FCu);  // u + diaeresis → ü
}

TEST(UnicodeComposePairTest, GreekCompositions) {
  // Greek letter + combining mark compositions.
  EXPECT_EQ(iree_unicode_compose_pair(0x0391, 0x0301),
            0x0386u);  // Α + acute → Ά
  EXPECT_EQ(iree_unicode_compose_pair(0x03B1, 0x0301),
            0x03ACu);  // α + acute → ά
  EXPECT_EQ(iree_unicode_compose_pair(0x03B5, 0x0301),
            0x03ADu);  // ε + acute → έ
}

TEST(UnicodeComposePairTest, NoComposition) {
  // Pairs that don't compose should return 0.
  EXPECT_EQ(iree_unicode_compose_pair('A', 'B'), 0u);  // Not a valid pair
  EXPECT_EQ(iree_unicode_compose_pair('X', 0x0301),
            0u);  // X doesn't compose with acute
  EXPECT_EQ(iree_unicode_compose_pair(0x4E2D, 0x0301),
            0u);  // CJK doesn't compose
}

TEST(UnicodeComposePairTest, LatinExtendedCompositions) {
  // Compositions in Latin Extended blocks.
  EXPECT_EQ(iree_unicode_compose_pair('A', 0x0304), 0x0100u);  // A + macron → Ā
  EXPECT_EQ(iree_unicode_compose_pair('a', 0x0304), 0x0101u);  // a + macron → ā
  EXPECT_EQ(iree_unicode_compose_pair('C', 0x030C), 0x010Cu);  // C + caron → Č
  EXPECT_EQ(iree_unicode_compose_pair('c', 0x030C), 0x010Du);  // c + caron → č
  EXPECT_EQ(iree_unicode_compose_pair('S', 0x030C), 0x0160u);  // S + caron → Š
  EXPECT_EQ(iree_unicode_compose_pair('s', 0x030C), 0x0161u);  // s + caron → š
}

//===----------------------------------------------------------------------===//
// Unicode Composition Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeComposeTest, AsciiPassthrough) {
  char buffer[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_unicode_compose(iree_make_cstring_view("Hello World"),
                                      sizeof(buffer), buffer, &length));
  EXPECT_EQ(std::string(buffer, length), "Hello World");
}

TEST(UnicodeComposeTest, EmptyString) {
  char buffer[64];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(iree_unicode_compose(iree_make_cstring_view(""),
                                      sizeof(buffer), buffer, &length));
  EXPECT_EQ(length, 0u);
}

TEST(UnicodeComposeTest, AlreadyComposed) {
  // Already-composed characters should pass through unchanged.
  char buffer[64];
  iree_host_size_t length = 0;

  // "café" with precomposed é (U+00E9).
  const char input[] = "caf\xC3\xA9";
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeComposeTest, DecomposedToComposed) {
  char buffer[64];
  iree_host_size_t length = 0;

  // "café" with decomposed é: 'e' (U+0065) + combining acute (U+0301).
  const char input[] = "cafe\xCC\x81";  // e + combining acute
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));

  // Should compose to "café" with precomposed é.
  const char expected[] = "caf\xC3\xA9";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeComposeTest, MultipleDecomposedCharacters) {
  char buffer[64];
  iree_host_size_t length = 0;

  // "résumé" with all decomposed accents.
  // r + e + acute + s + u + m + e + acute
  const char input[] = "re\xCC\x81sume\xCC\x81";
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));

  // Should compose to "résumé" with precomposed é.
  const char expected[] = "r\xC3\xA9sum\xC3\xA9";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeComposeTest, MultipleMarksOnSameBase) {
  char buffer[64];
  iree_host_size_t length = 0;

  // o + combining acute (U+0301) + combining diaeresis (U+0308).
  // Composition can only compose one mark, so result should be ó + diaeresis.
  const char input[] = "o\xCC\x81\xCC\x88";
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));

  // o + acute composes to ó (U+00F3), diaeresis remains.
  // Expected: ó (C3 B3) + combining diaeresis (CC 88).
  const char expected[] = "\xC3\xB3\xCC\x88";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeComposeTest, CanonicalOrdering) {
  char buffer[64];
  iree_host_size_t length = 0;

  // Test that combining marks are canonically ordered.
  // o + cedilla (CCC=202) + acute (CCC=230).
  // Cedilla is below (202), acute is above (230).
  // Since 202 < 230, order is already canonical.
  const char input1[] = "o\xCC\xA7\xCC\x81";  // cedilla + acute
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input1, sizeof(input1) - 1),
                           sizeof(buffer), buffer, &length));

  // o + acute (CCC=230) + cedilla (CCC=202) is NOT canonical order.
  // Compose should reorder to cedilla + acute, then compose.
  char buffer2[64];
  iree_host_size_t length2 = 0;
  const char input2[] = "o\xCC\x81\xCC\xA7";  // acute + cedilla (wrong order)
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input2, sizeof(input2) - 1),
                           sizeof(buffer2), buffer2, &length2));

  // Both should produce the same canonical result.
  EXPECT_EQ(std::string(buffer, length), std::string(buffer2, length2));
}

TEST(UnicodeComposeTest, NonComposingMarks) {
  char buffer[64];
  iree_host_size_t length = 0;

  // 'x' doesn't compose with acute accent, so mark should remain.
  const char input[] = "x\xCC\x81";  // x + combining acute
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));

  // Should remain unchanged (x doesn't compose with acute).
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeComposeTest, CjkUnchanged) {
  char buffer[64];
  iree_host_size_t length = 0;

  // CJK characters pass through unchanged.
  const char input[] = "\xE4\xB8\xAD\xE6\x96\x87";  // 中文
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeComposeTest, MixedContent) {
  char buffer[128];
  iree_host_size_t length = 0;

  // Mixed ASCII, decomposed accents, and CJK.
  // "Hello café 中文"
  // With decomposed é: e + combining acute.
  const char input[] = "Hello cafe\xCC\x81 \xE4\xB8\xAD\xE6\x96\x87";
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));

  // Expected: "Hello café 中文" with composed é.
  const char expected[] = "Hello caf\xC3\xA9 \xE4\xB8\xAD\xE6\x96\x87";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeComposeTest, BufferTooSmall) {
  char buffer[4];
  iree_host_size_t length = 0;

  // Buffer too small for output.
  iree_status_t status = iree_unicode_compose(
      iree_make_cstring_view("Hello World"), sizeof(buffer), buffer, &length);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));
}

TEST(UnicodeComposeTest, BufferCapacityIsInputSize) {
  // Compose only needs capacity >= input.size (output can only shrink).
  // Uses a small internal buffer for processing, not the output buffer.

  // "café" with decomposed é: 6 bytes input (4 ASCII + 2-byte combining mark),
  // 5 codepoints, composes to 5 bytes output.
  const char input[] = "cafe\xCC\x81";    // 6 bytes, 5 codepoints
  const char expected[] = "caf\xC3\xA9";  // 5 bytes output

  // Verify our understanding of the input/output sizes.
  EXPECT_EQ(sizeof(input) - 1, 6u);     // Input is 6 bytes.
  EXPECT_EQ(sizeof(expected) - 1, 5u);  // Output is 5 bytes.

  // Buffer exactly input size should work.
  char exact_buffer[6];
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(exact_buffer), exact_buffer, &length));
  EXPECT_EQ(length, 5u);
  EXPECT_EQ(std::string(exact_buffer, length),
            std::string(expected, sizeof(expected) - 1));

  // Buffer smaller than input size should fail.
  char small_buffer[5];
  length = 0;
  iree_status_t status =
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(small_buffer), small_buffer, &length);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));

  // Larger buffer also works.
  char large_buffer[64];
  length = 0;
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(large_buffer), large_buffer, &length));
  EXPECT_EQ(length, 5u);
  EXPECT_EQ(std::string(large_buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeComposeTest, GreekWithAccents) {
  char buffer[64];
  iree_host_size_t length = 0;

  // Greek alpha + combining acute.
  const char input[] = "\xCE\xB1\xCC\x81";  // α + acute
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));

  // Should compose to ά (U+03AC).
  const char expected[] = "\xCE\xAC";
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeComposeTest, Emoji) {
  char buffer[64];
  iree_host_size_t length = 0;

  // Emoji pass through unchanged.
  const char input[] = "\xF0\x9F\x98\x80";  // 😀
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeComposeTest, HangulUnchanged) {
  char buffer[64];
  iree_host_size_t length = 0;

  // Pre-composed Hangul syllables pass through unchanged.
  // We don't do Hangul algorithmic composition.
  const char input[] = "\xEA\xB0\x80";  // 가 (U+AC00)
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeComposeTest, MultipleCombiningSequences) {
  // Test that multiple combining sequences in one string are processed
  // correctly by the chunk-based algorithm.
  char buffer[64];
  iree_host_size_t length = 0;

  // "café naïve" with decomposed accents.
  // café: c a f e + combining acute
  // naïve: n a i + combining diaeresis v e
  const char input[] = "caf\x65\xCC\x81 na\x69\xCC\x88ve";
  const char expected[] = "caf\xC3\xA9 na\xC3\xAFve";

  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeComposeTest, ManyCombiningMarks) {
  // Test a sequence with many combining marks on one base character.
  // This exercises the internal buffer for combining sequences.
  char buffer[256];
  iree_host_size_t length = 0;

  // Build a string with a base character followed by 10 combining marks.
  // Using combining acute (U+0301) repeatedly - CCC 230.
  std::string input = "a";
  for (int i = 0; i < 10; ++i) {
    input += "\xCC\x81";  // Combining acute.
  }

  // Should succeed - 10 marks is well under the 32-codepoint limit.
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input.data(), input.size()),
                           sizeof(buffer), buffer, &length));

  // First acute should compose with 'a' to form 'á', rest remain.
  // á = U+00E1 = C3 A1
  std::string expected = "\xC3\xA1";
  for (int i = 0; i < 9; ++i) {
    expected += "\xCC\x81";  // Remaining acute marks.
  }
  EXPECT_EQ(std::string(buffer, length), expected);
}

TEST(UnicodeComposeTest, CombiningSequenceLimitExceeded) {
  // Test that exceeding 32 combining marks fails with RESOURCE_EXHAUSTED.
  char buffer[512];
  iree_host_size_t length = 0;

  // Build a string with a base character followed by 35 combining marks.
  std::string input = "a";
  for (int i = 0; i < 35; ++i) {
    input += "\xCC\x81";  // Combining acute.
  }

  // Should fail - 36 codepoints (1 base + 35 marks) exceeds 32 limit.
  iree_status_t status =
      iree_unicode_compose(iree_make_string_view(input.data(), input.size()),
                           sizeof(buffer), buffer, &length);
  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));
}

TEST(UnicodeComposeTest, LeadingCombiningMark) {
  // Test input that starts with a combining mark (no preceding base).
  // This is unusual but valid Unicode.
  char buffer[64];
  iree_host_size_t length = 0;

  // Combining acute followed by 'a' + combining acute.
  const char input[] =
      "\xCC\x81"
      "a\xCC\x81";
  // First mark has no base to combine with, second forms 'á'.
  const char expected[] = "\xCC\x81\xC3\xA1";

  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));
  EXPECT_EQ(std::string(buffer, length),
            std::string(expected, sizeof(expected) - 1));
}

TEST(UnicodeComposeTest, SingleCombiningMark) {
  // Test single combining mark with no base (edge case).
  char buffer[8];
  iree_host_size_t length = 0;

  const char input[] = "\xCC\x81";  // Just combining acute.

  // Should pass through unchanged.
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input, sizeof(input) - 1),
                           sizeof(buffer), buffer, &length));
  EXPECT_EQ(std::string(buffer, length), std::string(input, sizeof(input) - 1));
}

TEST(UnicodeComposeTest, CombiningSequenceLimitBoundary) {
  // Test exactly 32 codepoints (1 base + 31 combining marks) succeeds.
  // This is the maximum allowed combining sequence length.
  char buffer[256];
  iree_host_size_t length = 0;

  // Build a string with exactly 32 codepoints.
  std::string input = "a";  // 1 base character.
  for (int i = 0; i < 31; ++i) {
    input += "\xCC\x81";  // 31 combining acute marks.
  }

  // Should succeed - exactly at the 32-codepoint limit.
  IREE_ASSERT_OK(
      iree_unicode_compose(iree_make_string_view(input.data(), input.size()),
                           sizeof(buffer), buffer, &length));

  // First acute composes with 'a' to form 'á', rest remain.
  std::string expected = "\xC3\xA1";  // á
  for (int i = 0; i < 30; ++i) {
    expected += "\xCC\x81";  // 30 remaining acute marks.
  }
  EXPECT_EQ(std::string(buffer, length), expected);
}

TEST(UnicodeComposeTest, InvalidUtf8BehaviorUndefined) {
  // Invalid UTF-8 behavior is undefined per the API contract.
  // This test documents current behavior but does not guarantee it.
  // The decoder replaces invalid bytes with U+FFFD, which may cause
  // output to exceed input size.
  char buffer[64];
  iree_host_size_t length = 0;

  // Single invalid byte 0xFF - decoder returns U+FFFD (3 bytes: EF BF BD).
  // With capacity = 1, this would fail; with larger capacity it may succeed.
  const char invalid_input[] = "\xFF";

  // We document that behavior is undefined, so we just verify it doesn't crash.
  // The actual result depends on implementation details.
  iree_status_t status = iree_unicode_compose(
      iree_make_string_view(invalid_input, sizeof(invalid_input) - 1),
      sizeof(buffer), buffer, &length);

  // Either succeeds (with replacement char) or fails (capacity issues).
  // We don't assert on the result, just that it doesn't crash.
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// NFD Decomposition Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeDecomposeTest, AsciiUnchanged) {
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose('A', out), 1);
  EXPECT_EQ(out[0], 'A');
  EXPECT_EQ(iree_unicode_decompose('z', out), 1);
  EXPECT_EQ(out[0], 'z');
  EXPECT_EQ(iree_unicode_decompose('5', out), 1);
  EXPECT_EQ(out[0], '5');
}

TEST(UnicodeDecomposeTest, LatinPrecomposed) {
  uint32_t out[4];
  // é (U+00E9) -> e (U+0065) + combining acute accent (U+0301)
  EXPECT_EQ(iree_unicode_decompose(0x00E9, out), 2);
  EXPECT_EQ(out[0], 'e');
  EXPECT_EQ(out[1], 0x0301);  // Combining acute accent
  // ñ (U+00F1) -> n (U+006E) + combining tilde (U+0303)
  EXPECT_EQ(iree_unicode_decompose(0x00F1, out), 2);
  EXPECT_EQ(out[0], 'n');
  EXPECT_EQ(out[1], 0x0303);  // Combining tilde
}

TEST(UnicodeDecomposeTest, HangulSyllableLVT) {
  // 뮨 (U+BBA8) decomposes to ᄆ (U+1106) + ᅲ (U+1172) + ᆫ (U+11AB)
  // This is an LVT syllable (Leading + Vowel + Trailing).
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose(0xBBA8, out), 3);
  EXPECT_EQ(out[0], 0x1106);  // ᄆ (Choseong Mieum)
  EXPECT_EQ(out[1], 0x1172);  // ᅲ (Jungseong Yu)
  EXPECT_EQ(out[2], 0x11AB);  // ᆫ (Jongseong Nieun)
}

TEST(UnicodeDecomposeTest, HangulSyllableLV) {
  // 가 (U+AC00) decomposes to ᄀ (U+1100) + ᅡ (U+1161)
  // This is an LV syllable (Leading + Vowel, no Trailing).
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose(0xAC00, out), 2);
  EXPECT_EQ(out[0], 0x1100);  // ᄀ (Choseong Kiyeok)
  EXPECT_EQ(out[1], 0x1161);  // ᅡ (Jungseong A)
}

TEST(UnicodeDecomposeTest, HangulSyllableLastLVT) {
  // 힣 (U+D7A3) - last valid Hangul syllable.
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose(0xD7A3, out), 3);
  EXPECT_EQ(out[0], 0x1112);  // ᄒ (Choseong Hieuh)
  EXPECT_EQ(out[1], 0x1175);  // ᅵ (Jungseong I)
  EXPECT_EQ(out[2], 0x11C2);  // ᇂ (Jongseong Hieuh)
}

TEST(UnicodeDecomposeTest, HangulJamoUnchanged) {
  // Jamo characters (already decomposed) should pass through unchanged.
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose(0x1100, out), 1);  // ᄀ
  EXPECT_EQ(out[0], 0x1100);
  EXPECT_EQ(iree_unicode_decompose(0x1161, out), 1);  // ᅡ
  EXPECT_EQ(out[0], 0x1161);
}

TEST(UnicodeDecomposeTest, NonDecomposable) {
  uint32_t out[4];
  // CJK character - no decomposition
  EXPECT_EQ(iree_unicode_decompose(0x4E2D, out), 1);  // 中
  EXPECT_EQ(out[0], 0x4E2D);
  // Emoji - no decomposition
  EXPECT_EQ(iree_unicode_decompose(0x1F600, out), 1);  // 😀
  EXPECT_EQ(out[0], 0x1F600);
}

//===----------------------------------------------------------------------===//
// Vietnamese Character Recursive Decomposition Tests
//===----------------------------------------------------------------------===//

// Vietnamese characters require recursive decomposition because they have
// multiple levels of combining marks. For example:
// - ử (U+1EED) = u + horn + hook above
//   First level:  U+1EED → U+1B0 (ư - u with horn)
//   Second level: U+1B0 → U+0075 (u)
// - ệ (U+1EC7) = e + circumflex + dot below
//   First level:  U+1EC7 → U+1EB9 (ẹ - e with dot below)
//   Second level: U+1EB9 → U+0065 (e)

TEST(UnicodeNfdTest, VietnameseRecursiveDecomposition) {
  // ử (U+1EED) should decompose recursively to 'u' (U+0075)
  // Path: U+1EED → U+1B0 (ư) → U+0075 (u)
  EXPECT_EQ(iree_unicode_nfd_base(0x1EED), 0x0075u);  // ử → u

  // ệ (U+1EC7) should decompose recursively to 'e' (U+0065)
  // Path: U+1EC7 → U+1EB9 (ẹ) → U+0065 (e)
  EXPECT_EQ(iree_unicode_nfd_base(0x1EC7), 0x0065u);  // ệ → e

  // ư (U+1B0) should decompose to 'u' (U+0075)
  EXPECT_EQ(iree_unicode_nfd_base(0x1B0), 0x0075u);  // ư → u

  // ẹ (U+1EB9) should decompose to 'e' (U+0065)
  EXPECT_EQ(iree_unicode_nfd_base(0x1EB9), 0x0065u);  // ẹ → e

  // ơ (U+01A1) should decompose to 'o' (U+006F)
  EXPECT_EQ(iree_unicode_nfd_base(0x1A1), 0x006Fu);  // ơ → o

  // Ơ (U+01A0) should decompose to 'O' (U+004F)
  EXPECT_EQ(iree_unicode_nfd_base(0x1A0), 0x004Fu);  // Ơ → O
}

TEST(UnicodeDecomposeTest, VietnameseRecursiveDecomposition) {
  uint32_t out[4];

  // ử (U+1EED) = ư + hook above, ư = u + horn
  // Full NFD: u (U+0075) + horn (U+031B) + hook above (U+0309)
  EXPECT_EQ(iree_unicode_decompose(0x1EED, out), 3);
  EXPECT_EQ(out[0], 0x0075u);  // u
  EXPECT_EQ(out[1], 0x031Bu);  // horn
  EXPECT_EQ(out[2], 0x0309u);  // hook above

  // ệ (U+1EC7) decomposes to ẹ (U+1EB9) + circumflex (U+0302)
  // ẹ (U+1EB9) decomposes to e (U+0065) + dot below (U+0323)
  // Full NFD: e (U+0065) + dot below (U+0323) + circumflex (U+0302)
  // Note: The order follows the decomposition chain, not canonical order.
  EXPECT_EQ(iree_unicode_decompose(0x1EC7, out), 3);
  EXPECT_EQ(out[0], 0x0065u);  // e
  EXPECT_EQ(out[1], 0x0323u);  // dot below (from ẹ decomposition)
  EXPECT_EQ(out[2], 0x0302u);  // circumflex (from ệ decomposition)

  // ư (U+01B0) = u + horn
  // Full NFD: u (U+0075) + horn (U+031B)
  EXPECT_EQ(iree_unicode_decompose(0x1B0, out), 2);
  EXPECT_EQ(out[0], 0x0075u);  // u
  EXPECT_EQ(out[1], 0x031Bu);  // horn

  // ẹ (U+1EB9) = e + dot below
  // Full NFD: e (U+0065) + dot below (U+0323)
  EXPECT_EQ(iree_unicode_decompose(0x1EB9, out), 2);
  EXPECT_EQ(out[0], 0x0065u);  // e
  EXPECT_EQ(out[1], 0x0323u);  // dot below
}

TEST(UnicodeDecomposeTest, BufferSafetyCheck) {
  uint32_t out[4];

  // All decompositions should fit in a 4-element buffer.
  // Most return 1-3 codepoints; Hangul can return up to 3.

  // Vietnamese (recursive decomposition) - can return up to 3
  EXPECT_LE(iree_unicode_decompose(0x1EED, out), 4);
  EXPECT_LE(iree_unicode_decompose(0x1EC7, out), 4);

  // Latin Extended Additional - returns 1-2
  EXPECT_LE(iree_unicode_decompose(0x1E00, out), 4);  // Ḁ
  EXPECT_LE(iree_unicode_decompose(0x1EF9, out), 4);  // ỹ

  // Hangul (algorithmic) - returns 2 or 3
  EXPECT_LE(iree_unicode_decompose(0xAC00, out), 4);  // 가 (LV)
  EXPECT_LE(iree_unicode_decompose(0xBBA8, out), 4);  // 뮨 (LVT)
  EXPECT_LE(iree_unicode_decompose(0xD7A3, out), 4);  // 힣 (LVT)

  // Greek Extended - returns 1-2
  EXPECT_LE(iree_unicode_decompose(0x1F00, out), 4);  // ἀ
  EXPECT_LE(iree_unicode_decompose(0x1FFF, out), 4);

  // Non-decomposable characters - returns 1
  EXPECT_LE(iree_unicode_decompose(0x4E2D, out), 4);   // 中
  EXPECT_LE(iree_unicode_decompose(0x1F600, out), 4);  // 😀
}

//===----------------------------------------------------------------------===//
// NFC Canonical Decomposition Tests
//===----------------------------------------------------------------------===//

// Tests for iree_unicode_decompose_nfc_canonical() which handles all
// decompositions required for NFC normalization, including NFC_QC=No
// characters with multi-codepoint canonical decompositions.

TEST(UnicodeDecomposeNfcCanonicalTest, AsciiUnchanged) {
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical('A', out), 1u);
  EXPECT_EQ(out[0], (uint32_t)'A');
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical('z', out), 1u);
  EXPECT_EQ(out[0], (uint32_t)'z');
}

TEST(UnicodeDecomposeNfcCanonicalTest, PrecomposedUnchanged) {
  // Precomposed characters (NFC_QC=Yes) should NOT be decomposed.
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x00E9, out), 1u);  // é
  EXPECT_EQ(out[0], 0x00E9u);
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x00FC, out), 1u);  // ü
  EXPECT_EQ(out[0], 0x00FCu);
}

TEST(UnicodeDecomposeNfcCanonicalTest, SingletonNfcQcNo) {
  // Singleton NFC_QC=No characters (decompose to a single codepoint).
  uint32_t out[4];

  // U+0340 COMBINING GRAVE TONE MARK → U+0300
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x0340, out), 1u);
  EXPECT_EQ(out[0], 0x0300u);

  // U+0341 COMBINING ACUTE TONE MARK → U+0301
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x0341, out), 1u);
  EXPECT_EQ(out[0], 0x0301u);

  // U+0343 COMBINING GREEK KORONIS → U+0313
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x0343, out), 1u);
  EXPECT_EQ(out[0], 0x0313u);
}

TEST(UnicodeDecomposeNfcCanonicalTest, MultiCodepointNfcQcNo) {
  // U+0344 COMBINING GREEK DIALYTIKA TONOS → U+0308 + U+0301
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x0344, out), 2u);
  EXPECT_EQ(out[0], 0x0308u);
  EXPECT_EQ(out[1], 0x0301u);
}

TEST(UnicodeDecomposeNfcCanonicalTest, DevanagariNukta) {
  // U+0958 → U+0915 + U+093C (ka + nukta)
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x0958, out), 2u);
  EXPECT_EQ(out[0], 0x0915u);
  EXPECT_EQ(out[1], 0x093Cu);

  // U+095F → U+092F + U+093C (ya + nukta)
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x095F, out), 2u);
  EXPECT_EQ(out[0], 0x092Fu);
  EXPECT_EQ(out[1], 0x093Cu);
}

TEST(UnicodeDecomposeNfcCanonicalTest, TibetanSubjoined) {
  // U+0F43 → U+0F42 + U+0FB7 (ga + subjoined ha)
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x0F43, out), 2u);
  EXPECT_EQ(out[0], 0x0F42u);
  EXPECT_EQ(out[1], 0x0FB7u);

  // U+0F73 → U+0F71 + U+0F72 (Tibetan vowel sign II)
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x0F73, out), 2u);
  EXPECT_EQ(out[0], 0x0F71u);
  EXPECT_EQ(out[1], 0x0F72u);
}

TEST(UnicodeDecomposeNfcCanonicalTest, HebrewDagesh) {
  // U+FB1D → U+05D9 + U+05B4 (yod + hiriq)
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0xFB1D, out), 2u);
  EXPECT_EQ(out[0], 0x05D9u);
  EXPECT_EQ(out[1], 0x05B4u);

  // U+FB2A → U+05E9 + U+05C1 (shin + shin dot)
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0xFB2A, out), 2u);
  EXPECT_EQ(out[0], 0x05E9u);
  EXPECT_EQ(out[1], 0x05C1u);
}

TEST(UnicodeDecomposeNfcCanonicalTest, MusicalSymbolsRecursive) {
  // U+1D15E → U+1D157 + U+1D165 (2 codepoints, no recursion)
  uint32_t out[4];
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x1D15E, out), 2u);
  EXPECT_EQ(out[0], 0x1D157u);
  EXPECT_EQ(out[1], 0x1D165u);

  // U+1D160 → U+1D158 + U+1D165 + U+1D16E (3 codepoints, recursive expansion)
  // (U+1D160 → U+1D15F + U+1D16E, then U+1D15F → U+1D158 + U+1D165)
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x1D160, out), 3u);
  EXPECT_EQ(out[0], 0x1D158u);
  EXPECT_EQ(out[1], 0x1D165u);
  EXPECT_EQ(out[2], 0x1D16Eu);
}

TEST(UnicodeDecomposeNfcCanonicalTest, HangulSyllable) {
  // Hangul is handled algorithmically (same as decompose_singleton).
  uint32_t out[4];

  // U+AC00 → U+1100 + U+1161 (LV syllable)
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0xAC00, out), 2u);
  EXPECT_EQ(out[0], 0x1100u);
  EXPECT_EQ(out[1], 0x1161u);

  // U+AC01 → U+1100 + U+1161 + U+11A8 (LVT syllable)
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0xAC01, out), 3u);
  EXPECT_EQ(out[0], 0x1100u);
  EXPECT_EQ(out[1], 0x1161u);
  EXPECT_EQ(out[2], 0x11A8u);
}

TEST(UnicodeDecomposeNfcCanonicalTest, CjkCompatibility) {
  // CJK compatibility ideographs (handled via NFD singleton table).
  uint32_t out[4];

  // U+F900 → U+8C48
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0xF900, out), 1u);
  EXPECT_EQ(out[0], 0x8C48u);
}

TEST(UnicodeDecomposeNfcCanonicalTest, NonDecomposableUnchanged) {
  uint32_t out[4];

  // Standard CJK ideograph — not a compatibility character.
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x4E2D, out), 1u);
  EXPECT_EQ(out[0], 0x4E2Du);  // 中

  // Emoji.
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x1F600, out), 1u);
  EXPECT_EQ(out[0], 0x1F600u);

  // Combining mark (NFC_QC=Maybe, not No).
  EXPECT_EQ(iree_unicode_decompose_nfc_canonical(0x0301, out), 1u);
  EXPECT_EQ(out[0], 0x0301u);
}

TEST(UnicodeDecomposeNfcCanonicalTest, BufferSafetyCheck) {
  // Verify all returned counts fit within the 4-element buffer.
  uint32_t out[4];

  // All entries in the NFC decomposition table should return <= 3 codepoints.
  // Test the entries with 3-codepoint expansions explicitly.
  EXPECT_LE(iree_unicode_decompose_nfc_canonical(0x1D160, out), 4u);
  EXPECT_LE(iree_unicode_decompose_nfc_canonical(0x1D161, out), 4u);
  EXPECT_LE(iree_unicode_decompose_nfc_canonical(0x1D1BD, out), 4u);
  EXPECT_LE(iree_unicode_decompose_nfc_canonical(0x1D1BF, out), 4u);

  // Hangul max is 3.
  EXPECT_LE(iree_unicode_decompose_nfc_canonical(0xAC01, out), 4u);
  EXPECT_LE(iree_unicode_decompose_nfc_canonical(0xD7A3, out), 4u);
}

//===----------------------------------------------------------------------===//
// Full NFC Normalization Tests
//===----------------------------------------------------------------------===//

// Tests for iree_unicode_nfc() which performs full NFC normalization
// (NFD decomposition followed by canonical composition).
// Unlike iree_unicode_compose(), this handles characters that need
// decomposition first, such as CJK Compatibility Ideographs.

TEST(UnicodeNfcTest, AsciiUnchanged) {
  char buffer[64];
  iree_host_size_t length;
  IREE_ASSERT_OK(iree_unicode_nfc(iree_make_cstring_view("Hello World"),
                                  sizeof(buffer), buffer, &length));
  EXPECT_EQ(length, 11u);
  EXPECT_EQ(std::string(buffer, length), "Hello World");
}

TEST(UnicodeNfcTest, EmptyString) {
  char buffer[16];
  iree_host_size_t length;
  IREE_ASSERT_OK(iree_unicode_nfc(iree_make_cstring_view(""), sizeof(buffer),
                                  buffer, &length));
  EXPECT_EQ(length, 0u);
}

TEST(UnicodeNfcTest, CJKCompatibilityIdeograph) {
  // U+2FA15 (CJK Compatibility Ideograph 麻) should normalize to U+9EBB.
  char buffer[16];
  iree_host_size_t length;

  // U+2FA15 in UTF-8: F0 AF A8 95
  const char input[] = "\xF0\xAF\xA8\x95";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length));

  // U+9EBB in UTF-8: E9 BA BB (3 bytes)
  ASSERT_EQ(length, 3u);
  EXPECT_EQ((uint8_t)buffer[0], 0xE9);
  EXPECT_EQ((uint8_t)buffer[1], 0xBA);
  EXPECT_EQ((uint8_t)buffer[2], 0xBB);
}

TEST(UnicodeNfcTest, MultipleCJKCompatibilityIdeographs) {
  // Test several codepoints from the U+2F800-U+2FA1D range.
  struct TestCase {
    const char* utf8_input;
    iree_host_size_t input_length;
    uint32_t expected_codepoint;
  };

  // U+2F800 -> U+4E3D (丽), U+2FA14 -> U+2A291, U+2FA15 -> U+9EBB (麻)
  TestCase cases[] = {
      {"\xF0\xAF\xA0\x80", 4, 0x4E3D},  // U+2F800 -> U+4E3D
      {"\xF0\xAF\xA8\x95", 4, 0x9EBB},  // U+2FA15 -> U+9EBB
      {"\xF0\xAF\xA8\x96", 4, 0x4D56},  // U+2FA16 -> U+4D56
      {"\xF0\xAF\xA8\x97", 4, 0x9EF9},  // U+2FA17 -> U+9EF9
      {"\xF0\xAF\xA8\x98", 4, 0x9EFE},  // U+2FA18 -> U+9EFE
      {"\xF0\xAF\xA8\x9B", 4, 0x9F16},  // U+2FA1B -> U+9F16
      {"\xF0\xAF\xA8\x9C", 4, 0x9F3B},  // U+2FA1C -> U+9F3B (鼻)
  };

  for (const auto& tc : cases) {
    char buffer[16];
    iree_host_size_t length;
    IREE_ASSERT_OK(
        iree_unicode_nfc(iree_make_string_view(tc.utf8_input, tc.input_length),
                         sizeof(buffer), buffer, &length));

    // Decode output and verify.
    iree_host_size_t pos = 0;
    uint32_t output_cp =
        iree_unicode_utf8_decode(iree_make_string_view(buffer, length), &pos);
    EXPECT_EQ(output_cp, tc.expected_codepoint)
        << "Input bytes: " << std::hex << (int)(uint8_t)tc.utf8_input[0] << " "
        << (int)(uint8_t)tc.utf8_input[1] << " "
        << (int)(uint8_t)tc.utf8_input[2] << " "
        << (int)(uint8_t)tc.utf8_input[3];
  }
}

TEST(UnicodeNfcTest, StandardCJKUnchanged) {
  // Standard CJK characters should pass through unchanged.
  char buffer[16];
  iree_host_size_t length;

  // U+9EBB (麻) in UTF-8: E9 BA BB
  const char input[] = "\xE9\xBA\xBB";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length));

  ASSERT_EQ(length, 3u);
  EXPECT_EQ(memcmp(buffer, input, length), 0);
}

TEST(UnicodeNfcTest, MixedCJKAndAscii) {
  // Mix of standard CJK, compatibility ideographs, and ASCII.
  char buffer[64];
  iree_host_size_t length;

  // "a" + U+2FA15 + "b" -> "a" + U+9EBB + "b"
  const char input[] =
      "a\xF0\xAF\xA8\x95"
      "b";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length));

  // Output: "a" (1) + U+9EBB (3) + "b" (1) = 5 bytes
  EXPECT_EQ(length, 5u);
  EXPECT_EQ(buffer[0], 'a');
  EXPECT_EQ((uint8_t)buffer[1], 0xE9);  // U+9EBB
  EXPECT_EQ((uint8_t)buffer[2], 0xBA);
  EXPECT_EQ((uint8_t)buffer[3], 0xBB);
  EXPECT_EQ(buffer[4], 'b');
}

TEST(UnicodeNfcTest, HangulSyllableDecompositionAndRecomposition) {
  // Hangul syllables should be decomposed to Jamo and then recomposed.
  // 뮨 (U+BBA8) should remain as 뮨 (already NFC).
  char buffer[16];
  iree_host_size_t length;

  // U+BBA8 in UTF-8: EB AE A8
  const char input[] = "\xEB\xAE\xA8";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length));

  // Should be unchanged (or recomposed to the same character).
  ASSERT_EQ(length, 3u);
  EXPECT_EQ(memcmp(buffer, input, length), 0);
}

TEST(UnicodeNfcTest, DecomposedToComposed) {
  // Decomposed e + combining acute (U+0065 + U+0301) -> é (U+00E9)
  char buffer[16];
  iree_host_size_t length;

  // e (0x65) + combining acute (CC 81)
  const char input[] = "e\xCC\x81";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length));

  // Should compose to é (U+00E9): C3 A9
  ASSERT_EQ(length, 2u);
  EXPECT_EQ((uint8_t)buffer[0], 0xC3);
  EXPECT_EQ((uint8_t)buffer[1], 0xA9);
}

TEST(UnicodeNfcTest, BufferTooSmall) {
  char buffer[2];  // Too small for the output.
  iree_host_size_t length;

  // U+2FA15 needs 3 bytes when normalized to U+9EBB.
  const char input[] = "\xF0\xAF\xA8\x95";
  iree_status_t status =
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length);

  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kResourceExhausted));
}

TEST(UnicodeNfcTest, NfcQcNoSingletonGraveTone) {
  // U+0340 (Combining Grave Tone Mark) → U+0300 via NFC decomposition.
  // e + U+0340 should normalize to è (U+00E8 = C3 A8).
  // U+0340 = CD 80.
  char buffer[16];
  iree_host_size_t length;
  const char input[] = "e\xCD\x80";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length));
  EXPECT_EQ(length, 2u);
  EXPECT_EQ((uint8_t)buffer[0], 0xC3u);  // è = C3 A8
  EXPECT_EQ((uint8_t)buffer[1], 0xA8u);
}

TEST(UnicodeNfcTest, NfcQcNoMultiCodepointDialytikaTonos) {
  // U+0344 (Combining Greek Dialytika Tonos) → U+0308 + U+0301.
  // a + U+0344 = a + U+0308 + U+0301.
  // compose_pair('a', U+0308) = ä (U+00E4).
  // U+0301 has same CCC=230 as U+0308, so it's blocked from composing with ä.
  // Result: ä + U+0301 = C3 A4 CC 81.
  // U+0344 = CD 84.
  char buffer[16];
  iree_host_size_t length;
  const char input[] = "a\xCD\x84";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length));
  EXPECT_EQ(length, 4u);
  EXPECT_EQ((uint8_t)buffer[0], 0xC3u);  // ä = C3 A4
  EXPECT_EQ((uint8_t)buffer[1], 0xA4u);
  EXPECT_EQ((uint8_t)buffer[2], 0xCCu);  // U+0301 = CC 81
  EXPECT_EQ((uint8_t)buffer[3], 0x81u);
}

TEST(UnicodeNfcTest, NfcQcNoDevanagariNukta) {
  // U+0958 → U+0915 + U+093C (ka + nukta).
  // These should NOT recompose back (U+0958 is a composition exclusion).
  // U+0958 = E0 A5 98, U+0915 = E0 A4 95, U+093C = E0 A4 BC.
  char buffer[16];
  iree_host_size_t length;
  const char input[] = "\xE0\xA5\x98";
  IREE_ASSERT_OK(
      iree_unicode_nfc(iree_make_string_view(input, sizeof(input) - 1),
                       sizeof(buffer), buffer, &length));
  EXPECT_EQ(length, 6u);
  EXPECT_EQ((uint8_t)buffer[0], 0xE0u);  // U+0915 = E0 A4 95
  EXPECT_EQ((uint8_t)buffer[1], 0xA4u);
  EXPECT_EQ((uint8_t)buffer[2], 0x95u);
  EXPECT_EQ((uint8_t)buffer[3], 0xE0u);  // U+093C = E0 A4 BC
  EXPECT_EQ((uint8_t)buffer[4], 0xA4u);
  EXPECT_EQ((uint8_t)buffer[5], 0xBCu);
}

//===----------------------------------------------------------------------===//
// UTF-8 Sequence Validation Tests
//===----------------------------------------------------------------------===//

TEST(UnicodeUtf8Test, IsValidSequenceAscii) {
  // Valid ASCII (single byte).
  uint8_t bytes[] = {'A'};
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 1));

  bytes[0] = 0x00;  // Null byte is valid ASCII.
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 1));

  bytes[0] = 0x7F;  // DEL is valid ASCII.
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 1));
}

TEST(UnicodeUtf8Test, IsValidSequenceTwoByte) {
  // Valid 2-byte: é (U+00E9) = C3 A9
  uint8_t bytes[] = {0xC3, 0xA9};
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 2));

  // Valid 2-byte: U+0080 (first 2-byte codepoint) = C2 80
  bytes[0] = 0xC2;
  bytes[1] = 0x80;
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 2));
}

TEST(UnicodeUtf8Test, IsValidSequenceThreeByte) {
  // Valid 3-byte: 中 (U+4E2D) = E4 B8 AD
  uint8_t bytes[] = {0xE4, 0xB8, 0xAD};
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 3));

  // Valid 3-byte: U+FFFD (replacement char) = EF BF BD
  bytes[0] = 0xEF;
  bytes[1] = 0xBF;
  bytes[2] = 0xBD;
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 3));
}

TEST(UnicodeUtf8Test, IsValidSequenceFourByte) {
  // Valid 4-byte: 😀 (U+1F600) = F0 9F 98 80
  uint8_t bytes[] = {0xF0, 0x9F, 0x98, 0x80};
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 4));

  // Valid 4-byte: U+10FFFF (max codepoint) = F4 8F BF BF
  bytes[0] = 0xF4;
  bytes[1] = 0x8F;
  bytes[2] = 0xBF;
  bytes[3] = 0xBF;
  EXPECT_TRUE(iree_unicode_utf8_is_valid_sequence(bytes, 4));
}

TEST(UnicodeUtf8Test, IsValidSequenceInvalidContinuation) {
  // Invalid: continuation byte in 2-byte sequence.
  uint8_t bytes[] = {0xC3, 0x00};  // 0x00 is not 10xxxxxx
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 2));

  bytes[1] = 0xC0;  // 0xC0 is a lead byte, not continuation
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 2));
}

TEST(UnicodeUtf8Test, IsValidSequenceOverlong) {
  // Overlong 2-byte: encoding of ASCII 'A' (should be 1 byte).
  // U+0041 as 2 bytes would be C1 81, but C1 is not a valid lead byte.
  // Actually the overlong check is: (bytes[0] & 0x1E) == 0 for 2-byte.
  uint8_t bytes[] = {0xC0, 0x80};  // Overlong NUL
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 2));

  // Overlong 3-byte: U+007F as 3 bytes.
  uint8_t bytes3[] = {0xE0, 0x81, 0xBF};  // Would encode U+007F
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes3, 3));
}

TEST(UnicodeUtf8Test, IsValidSequenceSurrogate) {
  // Invalid: UTF-16 surrogate half U+D800 = ED A0 80
  uint8_t bytes[] = {0xED, 0xA0, 0x80};
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 3));

  // Invalid: UTF-16 surrogate half U+DFFF = ED BF BF
  bytes[0] = 0xED;
  bytes[1] = 0xBF;
  bytes[2] = 0xBF;
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 3));
}

TEST(UnicodeUtf8Test, IsValidSequenceOutOfRange) {
  // Invalid: codepoint > U+10FFFF = F4 90 80 80 (U+110000)
  uint8_t bytes[] = {0xF4, 0x90, 0x80, 0x80};
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 4));
}

TEST(UnicodeUtf8Test, IsValidSequenceZeroLength) {
  uint8_t bytes[] = {0x00};
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 0));
}

TEST(UnicodeUtf8Test, IsValidSequenceInvalidLength) {
  uint8_t bytes[] = {0x00};
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 5));  // Max is 4
}

TEST(UnicodeUtf8Test, IsValidSequenceHighBitAscii) {
  // High bit set in "ASCII" position is invalid.
  uint8_t bytes[] = {0x80};  // Continuation byte alone
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 1));

  bytes[0] = 0xFF;  // Invalid lead byte
  EXPECT_FALSE(iree_unicode_utf8_is_valid_sequence(bytes, 1));
}

}  // namespace
