// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/nfc.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer/normalizer_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedNormalizer;
using testing::ScopedNormalizerState;
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllChunkSizes;

//===----------------------------------------------------------------------===//
// Test fixture for NFC normalizer tests.
//===----------------------------------------------------------------------===//

class NFCNormalizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_nfc_allocate(
        iree_allocator_system(), &raw_normalizer));
    normalizer_ = ScopedNormalizer(raw_normalizer);
  }

  iree_tokenizer_normalizer_t* normalizer() { return normalizer_.get(); }

 private:
  ScopedNormalizer normalizer_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, CreateAndDestroy) {
  EXPECT_NE(normalizer(), nullptr);
}

TEST_F(NFCNormalizerTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer());
  EXPECT_GT(state_size, 0u);
  // State should be bounded (combining sequence buffer + emit tracking).
  EXPECT_LE(state_size, 256u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, EmptyInput) {
  std::string result = ProcessAndFinalize(
      normalizer(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(NFCNormalizerTest, AsciiOnly) {
  // Pure ASCII is always NFC — passes through unchanged.
  // The last ASCII byte is buffered (potential composition base), so
  // has_pending is true after process.
  TestWithAllChunkSizes(normalizer(), "HELLO WORLD 123!", "HELLO WORLD 123!",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, LongAsciiRun) {
  std::string input(500, 'A');
  TestWithAllChunkSizes(normalizer(), input, input,
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, MalformedUtf8ConsumesOneByteAtATime) {
  const std::string input = {static_cast<char>(0xC0), static_cast<char>(0x80),
                             static_cast<char>(0xF6), '@', 'A'};
  const std::string expected = "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD@A";
  TestWithAllChunkSizes(normalizer(), input, expected,
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Precomposed Characters (Already NFC)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, PrecomposedPassthrough) {
  // Already-NFC text passes through unchanged.
  // é = U+00E9 = C3 A9, ü = U+00FC = C3 BC.
  TestWithAllChunkSizes(normalizer(), "caf\xC3\xA9", "caf\xC3\xA9",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, MixedAsciiAndPrecomposed) {
  // "naïve" with precomposed ï (U+00EF = C3 AF).
  TestWithAllChunkSizes(normalizer(), "na\xC3\xAFve", "na\xC3\xAFve",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Combining Sequences → Composition
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, SingleCombiningMark) {
  // e + combining acute (U+0301 = CC 81) → é (U+00E9 = C3 A9).
  TestWithAllChunkSizes(normalizer(), "e\xCC\x81", "\xC3\xA9",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, CombiningMarkInContext) {
  // "cafe" + combining acute on the 'e' → "café".
  TestWithAllChunkSizes(normalizer(), "cafe\xCC\x81", "caf\xC3\xA9",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, MultipleCombiningMarks) {
  // e + combining acute + combining acute → é + combining acute.
  // First acute composes with 'e', second doesn't (blocked: same CCC=230).
  TestWithAllChunkSizes(normalizer(), "e\xCC\x81\xCC\x81", "\xC3\xA9\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NonComposingCombiningMark) {
  // a + combining tilde (U+0303 = CC 83) → ã (U+00E3 = C3 A3).
  TestWithAllChunkSizes(normalizer(), "a\xCC\x83", "\xC3\xA3",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, CombiningMarkNoComposition) {
  // b + combining acute → no composition exists for b+acute in Unicode.
  // Output: b + combining acute unchanged.
  TestWithAllChunkSizes(normalizer(), "b\xCC\x81", "b\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// NFC_QC=No Characters (Multi-codepoint Canonical Decompositions)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoSingleton) {
  // U+0340 (Combining Grave Tone Mark) → U+0300 (Combining Grave Accent).
  // With preceding 'a': a + U+0300 → à (U+00E0).
  // U+0340 = CD 80, U+00E0 = C3 A0.
  TestWithAllChunkSizes(normalizer(), "a\xCD\x80", "\xC3\xA0",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoSingletonAcuteTone) {
  // U+0341 (Combining Acute Tone Mark) → U+0301 (Combining Acute Accent).
  // With preceding 'e': e + U+0301 → é (U+00E9).
  // U+0341 = CD 81.
  TestWithAllChunkSizes(normalizer(), "e\xCD\x81", "\xC3\xA9",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoMultiCodepoint) {
  // U+0344 (Combining Greek Dialytika Tonos) → U+0308 + U+0301.
  // With preceding 'a': a + U+0308 + U+0301.
  // compose_pair('a', U+0308) = ä (U+00E4).
  // compose_pair(ä, U+0301) = no composition (blocked: both CCC=230).
  // Output: ä + U+0301 = C3 A4 CC 81.
  // U+0344 = CD 84.
  TestWithAllChunkSizes(normalizer(), "a\xCD\x84", "\xC3\xA4\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoStandalone) {
  // U+0344 alone (defective combining sequence): decomposes to U+0308 + U+0301.
  // No preceding starter, so no composition occurs.
  // U+0344 = CD 84, U+0308 = CC 88, U+0301 = CC 81.
  TestWithAllChunkSizes(normalizer(), "\xCD\x84", "\xCC\x88\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoDevanagariNukta) {
  // U+0958 → U+0915 + U+093C (ka + nukta).
  // Nukta (U+093C) has CCC=7, ka (U+0915) has CCC=0.
  // compose_pair(U+0915, U+093C) = U+0958? No! U+0958 is a composition
  // exclusion, so compose_pair should NOT produce it.
  // Output: U+0915 + U+093C = E0 A4 95 E0 A4 BC.
  // U+0958 = E0 A5 98.
  TestWithAllChunkSizes(normalizer(), "\xE0\xA5\x98",
                        "\xE0\xA4\x95\xE0\xA4\xBC",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Musical Symbols (Recursive Decomposition, Composition Exclusions)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoMusicalHalfNote) {
  // U+1D15E (Musical Symbol Half Note) → U+1D157 + U+1D165.
  // Recursive: 1D15E → 1D157 + 1D165 (via intermediate 1D15F → 1D158 + 1D165,
  // but the half note decomposes directly to stem + combining stem).
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xF0\x9D\x85\x9E",
                        "\xF0\x9D\x85\x97\xF0\x9D\x85\xA5",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoMusicalQuarterNote) {
  // U+1D15F (Musical Symbol Quarter Note) → U+1D158 + U+1D165.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xF0\x9D\x85\x9F",
                        "\xF0\x9D\x85\x98\xF0\x9D\x85\xA5",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoMusicalEighthNote) {
  // U+1D160 (Musical Symbol Eighth Note) → U+1D158 + U+1D165 + U+1D16E.
  // Recursive: 1D160 → 1D15F + 1D16E → (1D158 + 1D165) + 1D16E.
  // Three-codepoint expansion, all 4-byte UTF-8.
  TestWithAllChunkSizes(normalizer(), "\xF0\x9D\x85\xA0",
                        "\xF0\x9D\x85\x98\xF0\x9D\x85\xA5\xF0\x9D\x85\xAE",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoMusicalSixteenthNote) {
  // U+1D161 (Musical Symbol Sixteenth Note) → U+1D158 + U+1D165 + U+1D16F.
  // Same recursive pattern as eighth note but different flag.
  TestWithAllChunkSizes(normalizer(), "\xF0\x9D\x85\xA1",
                        "\xF0\x9D\x85\x98\xF0\x9D\x85\xA5\xF0\x9D\x85\xAF",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoMusicalHalfNoteAfterStarter) {
  // 'x' + U+1D15E → 'x' + U+1D157 + U+1D165.
  // The starter 'x' and U+1D157 (CCC=0, another starter) don't compose.
  TestWithAllChunkSizes(normalizer(), "x\xF0\x9D\x85\x9E",
                        "x\xF0\x9D\x85\x97\xF0\x9D\x85\xA5",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoMusicalEighthNoteAfterStarter) {
  // 'x' + U+1D160 → 'x' + U+1D158 + U+1D165 + U+1D16E.
  // Three-codepoint expansion after an ASCII starter.
  TestWithAllChunkSizes(normalizer(), "x\xF0\x9D\x85\xA0",
                        "x\xF0\x9D\x85\x98\xF0\x9D\x85\xA5\xF0\x9D\x85\xAE",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Tibetan (Composition Exclusions)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoTibetanVowelIi) {
  // U+0F73 (Tibetan Vowel Sign II) → U+0F71 + U+0F72.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xE0\xBD\xB3",
                        "\xE0\xBD\xB1\xE0\xBD\xB2",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoTibetanVowelUu) {
  // U+0F75 (Tibetan Vowel Sign UU) → U+0F71 + U+0F74.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xE0\xBD\xB5",
                        "\xE0\xBD\xB1\xE0\xBD\xB4",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoTibetanSubjoinedGha) {
  // U+0F43 (Tibetan Letter GHA) → U+0F42 + U+0FB7.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xE0\xBD\x83",
                        "\xE0\xBD\x82\xE0\xBE\xB7",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoTibetanVowelIiAfterKa) {
  // U+0F40 (Tibetan KA) + U+0F73 → U+0F40 + U+0F71 + U+0F72.
  // KA is a starter (CCC=0), vowel sign decomposes but doesn't recompose.
  TestWithAllChunkSizes(normalizer(), "\xE0\xBD\x80\xE0\xBD\xB3",
                        "\xE0\xBD\x80\xE0\xBD\xB1\xE0\xBD\xB2",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Hebrew Presentation Forms (Composition Exclusions)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoHebrewYodHiriq) {
  // U+FB1D (Hebrew Letter Yod With Hiriq) → U+05D9 + U+05B4.
  // Composition exclusion: stays decomposed.
  // Input: EF AC 9D → Output: D7 99 D6 B4.
  TestWithAllChunkSizes(normalizer(), "\xEF\xAC\x9D", "\xD7\x99\xD6\xB4",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoHebrewShinDot) {
  // U+FB2A (Hebrew Letter Shin With Shin Dot) → U+05E9 + U+05C1.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xEF\xAC\xAA", "\xD7\xA9\xD7\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoHebrewSinDot) {
  // U+FB2B (Hebrew Letter Shin With Sin Dot) → U+05E9 + U+05C2.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xEF\xAC\xAB", "\xD7\xA9\xD7\x82",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Bengali Nukta Forms (Composition Exclusions)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoBengaliDdaNukta) {
  // U+09DC (Bengali Letter RRA) → U+09A1 + U+09BC.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xE0\xA7\x9C",
                        "\xE0\xA6\xA1\xE0\xA6\xBC",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoBengaliDdhaNukta) {
  // U+09DD (Bengali Letter RHA) → U+09A2 + U+09BC.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xE0\xA7\x9D",
                        "\xE0\xA6\xA2\xE0\xA6\xBC",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoBengaliYaNukta) {
  // U+09DF (Bengali Letter YYA) → U+09AF + U+09BC.
  // Composition exclusion: stays decomposed.
  TestWithAllChunkSizes(normalizer(), "\xE0\xA7\x9F",
                        "\xE0\xA6\xAF\xE0\xA6\xBC",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Greek Koronis (Decomposes AND Recomposes With Preceding Alpha)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoGreekKoronisAfterAlpha) {
  // U+03B1 (α) + U+0343 (Combining Greek Koronis, NFC_QC=No) →
  // Decomposes: α + U+0313 (Combining Comma Above).
  // Composes: α + U+0313 → U+1F00 (ἀ, Greek Small Letter Alpha With Psili).
  // Output: E1 BC 80.
  TestWithAllChunkSizes(normalizer(), "\xCE\xB1\xCD\x83", "\xE1\xBC\x80",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Sequences of Multiple NFC_QC=No Marks
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoTwoGraveToneMarks) {
  // U+0340 + U+0340 → U+0300 + U+0300.
  // Both decompose to combining grave. Defective combining sequence (no
  // starter), no composition possible.
  TestWithAllChunkSizes(normalizer(), "\xCD\x80\xCD\x80", "\xCC\x80\xCC\x80",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoGraveThenAcuteTone) {
  // U+0340 + U+0341 → U+0300 + U+0301.
  // Both decompose. Same CCC (230), stable order. No starter, no composition.
  TestWithAllChunkSizes(normalizer(), "\xCD\x80\xCD\x81", "\xCC\x80\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoTwoMarksAfterStarter) {
  // 'a' + U+0340 + U+0341 → 'a' + U+0300 + U+0301.
  // a + U+0300 → à (U+00E0). U+0301 has same CCC (230) as previous mark,
  // so it's blocked from composing further.
  // Output: à + U+0301 = C3 A0 CC 81.
  TestWithAllChunkSizes(normalizer(), "a\xCD\x80\xCD\x81", "\xC3\xA0\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Interaction With Other Combining Marks (Blocking, Ordering)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoCedillaThenGraveTone) {
  // 'a' + U+0327 (Combining Cedilla, CCC=202) + U+0340 (→U+0300, CCC=230).
  // Canonical order: CCC 202 < 230, already correct.
  // Composition: a + U+0327 doesn't compose. But U+0300 has CCC=230 > CCC=202,
  // so cedilla does not block grave from reaching the starter.
  // a + U+0300 → à (U+00E0), then cedilla attaches.
  // Output: à + U+0327 = C3 A0 CC A7.
  TestWithAllChunkSizes(normalizer(), "a\xCC\xA7\xCD\x80", "\xC3\xA0\xCC\xA7",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoAcuteThenDialytikaTonos) {
  // 'a' + U+0301 (Combining Acute, CCC=230) + U+0344 (→U+0308+U+0301, CCC=230).
  // After decomposition: a + U+0301 + U+0308 + U+0301.
  // a + U+0301 → á (U+00E1). U+0308 has same CCC (230), blocked. U+0301 same.
  // Output: á + U+0308 + U+0301 = C3 A1 CC 88 CC 81.
  TestWithAllChunkSizes(normalizer(), "a\xCC\x81\xCD\x84",
                        "\xC3\xA1\xCC\x88\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Additional Starters With Singletons
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, NfcQcNoAcuteToneAfterA) {
  // 'a' + U+0341 → 'a' + U+0301 → á (U+00E1).
  TestWithAllChunkSizes(normalizer(), "a\xCD\x81", "\xC3\xA1",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoDialytikaTondsAfterE) {
  // 'e' + U+0344 → 'e' + U+0308 + U+0301.
  // e + U+0308 → ë (U+00EB). U+0301 blocked (same CCC).
  // Output: ë + U+0301 = C3 AB CC 81.
  TestWithAllChunkSizes(normalizer(), "e\xCD\x84", "\xC3\xAB\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, NfcQcNoDialytikaTondsAfterO) {
  // 'o' + U+0344 → 'o' + U+0308 + U+0301.
  // o + U+0308 → ö (U+00F6). U+0301 blocked (same CCC).
  // Output: ö + U+0301 = C3 B6 CC 81.
  TestWithAllChunkSizes(normalizer(), "o\xCD\x84", "\xC3\xB6\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// CJK Compatibility Ideographs
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, CjkCompatibilityIdeograph) {
  // U+F900 → U+8C48 (豈). NFC decomposes CJK compatibility ideographs.
  // U+F900 = EF A4 80, U+8C48 = E8 B1 88.
  TestWithAllChunkSizes(normalizer(), "\xEF\xA4\x80", "\xE8\xB1\x88",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, CjkCompatibilitySIP) {
  // U+2F800 → U+4E3D (丽). First CJK compat ideograph in SIP.
  // U+2F800 = F0 AF A0 80, U+4E3D = E4 B8 BD.
  TestWithAllChunkSizes(normalizer(), "\xF0\xAF\xA0\x80", "\xE4\xB8\xBD",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, CjkCompatibilityDecomposition) {
  // U+2F9BA → U+86E2 (蛢). CJK Compatibility Ideograph.
  // U+2F9BA = F0 AF A6 BA, U+86E2 = E8 9B A2.
  TestWithAllChunkSizes(normalizer(), "\xF0\xAF\xA6\xBA", "\xE8\x9B\xA2",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, MultipleCjkCompat) {
  // U+F900 (豈) + U+F901 (更) + U+F902 (車).
  // U+F901 → U+66F4 = E6 9B B4, U+F902 → U+8ECA = E8 BB 8A.
  TestWithAllChunkSizes(normalizer(), "\xEF\xA4\x80\xEF\xA4\x81\xEF\xA4\x82",
                        "\xE8\xB1\x88\xE6\x9B\xB4\xE8\xBB\x8A",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, MixedCjkAndStandard) {
  // Standard CJK 中 (U+4E2D) + compat ideograph U+F900.
  TestWithAllChunkSizes(normalizer(), "\xE4\xB8\xAD\xEF\xA4\x80",
                        "\xE4\xB8\xAD\xE8\xB1\x88",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Hangul Jamo Composition
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, HangulLV) {
  // Hangul L + V composition.
  // ㄱ (U+1100) + ㅏ (U+1161) → 가 (U+AC00).
  // U+1100 = E1 84 80, U+1161 = E1 85 A1, U+AC00 = EA B0 80.
  TestWithAllChunkSizes(normalizer(), "\xE1\x84\x80\xE1\x85\xA1",
                        "\xEA\xB0\x80",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, HangulLVT) {
  // Hangul L + V + T composition.
  // ㄱ + ㅏ + ㄱ (U+11A8) → 각 (U+AC01).
  // U+11A8 = E1 86 A8, U+AC01 = EA B0 81.
  TestWithAllChunkSizes(normalizer(), "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8",
                        "\xEA\xB0\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, HangulSyllableDecomposed) {
  // Pre-composed Hangul syllable 가 (U+AC00) should pass through.
  TestWithAllChunkSizes(normalizer(), "\xEA\xB0\x80", "\xEA\xB0\x80",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Emoji (Passthrough)
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, Emoji) {
  // 😀 (U+1F600) — 4-byte UTF-8, already NFC.
  TestWithAllChunkSizes(normalizer(), "\xF0\x9F\x98\x80", "\xF0\x9F\x98\x80",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, EmojiWithSkinTone) {
  // 👍🏽 = U+1F44D + U+1F3FD (already composed).
  TestWithAllChunkSizes(normalizer(), "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD",
                        "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Chunk Boundaries
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, CombiningMarkAcrossChunks) {
  // 'e' in one chunk, combining acute in next.
  // ProcessAndFinalize handles this with the chunked variant.
  TestWithAllChunkSizes(normalizer(), "e\xCC\x81", "\xC3\xA9",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, HangulAcrossChunks) {
  // Hangul Jamo L+V+T across chunks. TestWithAllChunkSizes tries all splits.
  TestWithAllChunkSizes(normalizer(), "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8",
                        "\xEA\xB0\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, CjkCompatAcrossChunks) {
  // CJK compat ideograph with ASCII context, split at various points.
  TestWithAllChunkSizes(normalizer(),
                        "A\xF0\xAF\xA6\xBA"
                        "B",
                        "A\xE8\x9B\xA2"
                        "B",
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Output Buffer Limits
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, LimitedOutputCapacityAscii) {
  TestLimitedOutputCapacity(normalizer(), "hello world", "hello world");
}

TEST_F(NFCNormalizerTest, LimitedOutputCapacityCombining) {
  // e + acute → é. With output capacity=1, must handle partial UTF-8.
  TestLimitedOutputCapacity(normalizer(), "e\xCC\x81", "\xC3\xA9");
}

TEST_F(NFCNormalizerTest, LimitedOutputCapacityCjk) {
  TestLimitedOutputCapacity(normalizer(), "\xEF\xA4\x80", "\xE8\xB1\x88");
}

TEST_F(NFCNormalizerTest, LimitedOutputCapacityHangul) {
  TestLimitedOutputCapacity(normalizer(), "\xE1\x84\x80\xE1\x85\xA1",
                            "\xEA\xB0\x80");
}

TEST_F(NFCNormalizerTest, LimitedOutputCapacityDefectiveSequence) {
  // Defective combining sequence (no starter) with reordering under tight
  // output. U+0315 (CCC=232) + U+0300 (CCC=230) → U+0300 + U+0315.
  TestLimitedOutputCapacity(normalizer(), "\xCC\x95\xCC\x80",
                            "\xCC\x80\xCC\x95");
}

TEST_F(NFCNormalizerTest, LimitedOutputCapacityMultiByteStarters) {
  // Multiple 2-byte starters in sequence. With capacity=1, each partial emit
  // creates pending_utf8. The next starter must NOT clobber the pending bytes.
  // é (C3 A9) + a + b — the partial emit of é must complete before 'a' emits.
  TestLimitedOutputCapacity(normalizer(),
                            "\xC3\xA9"
                            "ab",
                            "\xC3\xA9"
                            "ab");
  // à (C3 A0) + é (C3 A9) + ü (C3 BC) — three consecutive 2-byte starters.
  TestLimitedOutputCapacity(normalizer(), "\xC3\xA0\xC3\xA9\xC3\xBC",
                            "\xC3\xA0\xC3\xA9\xC3\xBC");
  // 3-byte starter 中 (E4 B8 AD) + 2-byte starter é (C3 A9) + ASCII.
  TestLimitedOutputCapacity(normalizer(), "\xE4\xB8\xAD\xC3\xA9x",
                            "\xE4\xB8\xAD\xC3\xA9x");
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, DefectiveCombiningSequence) {
  // Combining mark without preceding starter.
  // combining acute alone → passes through unchanged.
  TestWithAllChunkSizes(normalizer(), "\xCC\x81", "\xCC\x81",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, DefectiveCombiningSequenceReordering) {
  // Combining marks without preceding starter must still be canonically
  // ordered. U+0315 (CCC=232, Combining Comma Above Right) + U+0300 (CCC=230,
  // Combining Grave Accent) → reordered to U+0300 + U+0315. U+0315 = CC 95,
  // U+0300 = CC 80.
  TestWithAllChunkSizes(normalizer(), "\xCC\x95\xCC\x80", "\xCC\x80\xCC\x95",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, DefectiveCombiningSequenceMultipleMarks) {
  // Three combining marks without starter, all needing reorder.
  // U+0315 (CCC=232) + U+0327 (CCC=202, Combining Cedilla) +
  // U+0300 (CCC=230) → sorted: U+0327 (202) + U+0300 (230) + U+0315 (232).
  // U+0327 = CC A7.
  TestWithAllChunkSizes(normalizer(), "\xCC\x95\xCC\xA7\xCC\x80",
                        "\xCC\xA7\xCC\x80\xCC\x95",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, DefectiveCombiningSequenceFollowedByStarter) {
  // Defective combining sequence followed by a starter. The marks are emitted
  // first (reordered), then the starter begins a fresh sequence.
  // U+0315 (CCC=232) + U+0300 (CCC=230) + 'a'
  // → U+0300 + U+0315 + 'a'.
  TestWithAllChunkSizes(normalizer(),
                        "\xCC\x95\xCC\x80"
                        "a",
                        "\xCC\x80\xCC\x95"
                        "a",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, SingleCharacter) {
  // Single ASCII character.
  TestWithAllChunkSizes(normalizer(), "a", "a",
                        /*expect_pending_after_process=*/true);
}

TEST_F(NFCNormalizerTest, StateReuse) {
  // Multiple independent normalizations with the same normalizer.
  EXPECT_EQ(ProcessAndFinalize(normalizer(), "first",
                               /*expect_pending_after_process=*/true),
            "first");
  EXPECT_EQ(ProcessAndFinalize(normalizer(), "second",
                               /*expect_pending_after_process=*/true),
            "second");
  EXPECT_EQ(ProcessAndFinalize(normalizer(), "e\xCC\x81",
                               /*expect_pending_after_process=*/true),
            "\xC3\xA9");
}

//===----------------------------------------------------------------------===//
// Sequence Overflow
//===----------------------------------------------------------------------===//

TEST_F(NFCNormalizerTest, SequenceOverflow) {
  // Build a string with 33 combining marks (exceeds 32 limit).
  // Base 'a' + 32 combining acute marks = 33 codepoints total.
  std::string input = "a";
  for (int i = 0; i < 32; ++i) {
    input += "\xCC\x81";  // combining acute (CCC=230)
  }

  ScopedNormalizerState state(normalizer());
  char output_buffer[4096];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_tokenizer_normalizer_state_process(
          state.get(), iree_make_string_view(input.data(), input.size()),
          iree_make_mutable_string_view(output_buffer, sizeof(output_buffer)),
          IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
}

}  // namespace
}  // namespace iree::tokenizer
