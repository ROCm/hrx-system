// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for the loom IR text tokenizer.
//
// The tokenizer processes untrusted input (user-authored .loom files),
// so this fuzzer is designed to stress every boundary in the scanner.
//
// Strategies:
//   1. Raw tokenization: feed arbitrary bytes through the full
//      peek/next/consume API and verify structural invariants.
//   2. Grammar-aware generation: build syntactically plausible token
//      sequences from fuzzer input bytes, stressing transitions between
//      token kinds and interleaving of strings/brackets/comments.
//   3. Truncation stress: take a valid-ish prefix and truncate at every
//      interesting position (after prefixes, inside escapes, mid-number).
//   4. Angle bracket nesting: generate deeply nested balanced and
//      unbalanced angle bracket structures with embedded strings.
//   5. API stress: exercise peek/next/expect/try_consume/at interleaving
//      on the same tokenizer to verify lookahead consistency.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/unicode.h"
#include "loom/format/text/tokenizer.h"

//===----------------------------------------------------------------------===//
// Fuzzer input consumption helpers
//===----------------------------------------------------------------------===//

typedef struct {
  const uint8_t* data;
  size_t remaining;
} fuzz_input_t;

typedef struct {
  iree_arena_block_pool_t block_pool;
  iree_arena_allocator_t scratch_arena;
  loom_tokenizer_t tokenizer;
} fuzz_tokenizer_t;

static void fuzz_tokenizer_initialize(iree_string_view_t source,
                                      iree_string_view_t filename,
                                      fuzz_tokenizer_t* out_tokenizer) {
  iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                   &out_tokenizer->block_pool);
  iree_arena_initialize(&out_tokenizer->block_pool,
                        &out_tokenizer->scratch_arena);
  loom_tokenizer_initialize(source, filename, &out_tokenizer->scratch_arena,
                            &out_tokenizer->tokenizer);
}

static void fuzz_tokenizer_deinitialize(fuzz_tokenizer_t* tokenizer) {
  loom_tokenizer_deinitialize(&tokenizer->tokenizer);
  iree_arena_deinitialize(&tokenizer->scratch_arena);
  iree_arena_block_pool_deinitialize(&tokenizer->block_pool);
}

static uint8_t fuzz_consume_u8(fuzz_input_t* input) {
  if (input->remaining == 0) return 0;
  uint8_t value = input->data[0];
  input->data++;
  input->remaining--;
  return value;
}

static uint16_t fuzz_consume_u16(fuzz_input_t* input) {
  uint16_t value = fuzz_consume_u8(input);
  value |= (uint16_t)fuzz_consume_u8(input) << 8;
  return value;
}

//===----------------------------------------------------------------------===//
// Invariant checks
//===----------------------------------------------------------------------===//

// Verifies that the tokenizer's position never exceeds source size.
// This is the core safety invariant: every bounds check in the scanner
// guards on position < source.size, so a position > source.size would
// mean an out-of-bounds read has already happened or is imminent.
static void fuzz_assert_position_invariant(const loom_tokenizer_t* tokenizer) {
  if (tokenizer->position > tokenizer->source.size) {
    __builtin_trap();
  }
}

static bool fuzz_string_view_in_bounds(iree_string_view_t haystack,
                                       iree_string_view_t needle) {
  uintptr_t haystack_start = (uintptr_t)haystack.data;
  uintptr_t haystack_end = haystack_start + haystack.size;
  uintptr_t needle_start = (uintptr_t)needle.data;
  uintptr_t needle_end = needle_start + needle.size;
  return needle_start >= haystack_start && needle_start <= haystack_end &&
         needle_end >= needle_start && needle_end <= haystack_end;
}

// Verifies token slice invariants. |source_text| must always alias source
// bytes for diagnostics. |text| aliases |source_text| for non-string tokens;
// string payloads may instead point at decoded scratch storage when escapes
// were present.
static void fuzz_assert_token_slice_valid(const loom_tokenizer_t* tokenizer,
                                          const loom_token_t* token) {
  if (token->kind == LOOM_TOKEN_EOF || token->kind == LOOM_TOKEN_NONE) {
    return;
  }
  if (!fuzz_string_view_in_bounds(tokenizer->source, token->source_text)) {
    __builtin_trap();
  }

  if (token->kind == LOOM_TOKEN_ERROR) {
    if (token->text.data != token->source_text.data ||
        token->text.size != token->source_text.size) {
      __builtin_trap();
    }
    return;
  }

  if (token->kind == LOOM_TOKEN_STRING) {
    if (token->source_text.size < 2 || token->source_text.data[0] != '"' ||
        token->source_text.data[token->source_text.size - 1] != '"') {
      __builtin_trap();
    }
    if (token->text.size > token->source_text.size - 2) {
      __builtin_trap();
    }
    if (!iree_unicode_utf8_validate(token->text)) {
      __builtin_trap();
    }
    return;
  }

  if (iree_string_view_is_empty(token->text) ||
      !fuzz_string_view_in_bounds(token->source_text, token->text)) {
    __builtin_trap();
  }
}

// Verifies that line tracking is monotonically non-decreasing. The
// tokenizer processes source left-to-right and line numbers only
// increase at newlines — they should never decrease.
static void fuzz_assert_line_monotonic(uint32_t previous_line,
                                       const loom_token_t* token) {
  if (token->kind == LOOM_TOKEN_EOF || token->kind == LOOM_TOKEN_NONE) {
    return;
  }
  if (token->line < previous_line) {
    __builtin_trap();
  }
}

//===----------------------------------------------------------------------===//
// Strategy 1: Raw tokenization with invariant checking
//===----------------------------------------------------------------------===//
// Feeds the entire fuzz input as source text and drains all tokens,
// checking structural invariants at each step. This catches:
// - Position overflow past source.size (out-of-bounds reads)
// - Tokens pointing outside the source buffer
// - Infinite loops (via iteration cap)
// - Line tracking going backwards
// - Status leaks (unconsumed error status)

static void fuzz_strategy_raw_tokenize(const uint8_t* data, size_t size) {
  iree_string_view_t source = {reinterpret_cast<const char*>(data), size};
  fuzz_tokenizer_t tokenizer;
  fuzz_tokenizer_initialize(source, IREE_SV("<fuzz>"), &tokenizer);

  uint32_t previous_line = 0;
  // Cap iterations to prevent infinite loops from consuming the entire
  // fuzzer budget. A source of N bytes can produce at most N tokens
  // (single-character punctuation), plus one EOF. We add margin for
  // the peek/next interleaving.
  size_t max_iterations = size + 16;
  for (size_t i = 0; i < max_iterations; ++i) {
    fuzz_assert_position_invariant(&tokenizer.tokenizer);

    loom_token_t token = loom_tokenizer_next(&tokenizer.tokenizer);
    fuzz_assert_position_invariant(&tokenizer.tokenizer);
    fuzz_assert_token_slice_valid(&tokenizer.tokenizer, &token);
    fuzz_assert_line_monotonic(previous_line, &token);

    if (token.kind != LOOM_TOKEN_EOF && token.kind != LOOM_TOKEN_NONE) {
      previous_line = token.line;
    }
    if (token.kind == LOOM_TOKEN_EOF) break;
  }

  // Malformed input is represented by LOOM_TOKEN_ERROR. The status channel is
  // reserved for infrastructure failures.
  IREE_CHECK_OK(loom_tokenizer_consume_status(&tokenizer.tokenizer));
  fuzz_tokenizer_deinitialize(&tokenizer);
}

//===----------------------------------------------------------------------===//
// Strategy 2: Grammar-aware token sequence generation
//===----------------------------------------------------------------------===//
// Builds a source string from fuzzer input by selecting token fragments
// from a vocabulary of syntactically valid constructs. This biases the
// fuzzer toward inputs that reach deeper into the scanner's logic
// (multi-character tokens, escape sequences, nesting) rather than
// getting stuck on "unexpected character" errors.

static void fuzz_strategy_grammar_aware(fuzz_input_t* input) {
  // Build a source string from grammar fragments.
  char source_buffer[1024];
  size_t source_length = 0;

  // Grammar fragments organized by category. These are the building
  // blocks of loom IR syntax — the fuzzer selects and concatenates
  // them to produce inputs that exercise real scanning paths.
  static const char* const fragments[] = {
      // Punctuation (single and multi-char).
      "(",
      ")",
      "{",
      "}",
      "[",
      "]",
      "<",
      ">",
      "=",
      ":",
      ",",
      "->",
      "::",

      // Whitespace and comments.
      " ",
      "\t",
      "\n",
      "\r\n",
      "  ",
      "// comment\n",
      "//\n",

      // Integer literals.
      "0",
      "1",
      "42",
      "0xFF",
      "0x0",
      "-1",
      "-42",
      "999999",

      // Float literals.
      "3.14",
      "1.0e5",
      "1.0E-5",
      "-0.5",
      "1e10",
      "0.0",

      // String literals (with escape sequences).
      "\"hello\"",
      "\"\"",
      "\"a\\\"b\"",
      "\"\\\\\"",
      "\"\\n\"",
      "\"\\r\\t\\b\\f\\/\"",
      "\"\\u0041\"",
      "\"\\u03BB\"",
      "\"\\uD83D\\uDD25\"",
      "\"\\uD83D\"",
      "\"\\uDD25\"",
      "\"\\u12G4\"",
      "\"line1\nline2\"",

      // SSA values.
      "%x",
      "%0",
      "%arg0",
      "%M",
      "%tile",
      "%contract0",
      "%_",

      // Symbols.
      "@main",
      "@mod",
      "@_init",

      // Hash attrs.
      "#q8_0",
      "#enc",

      // Block labels.
      "^bb0",
      "^entry",
      "^loop",

      // Bare identifiers.
      "tile",
      "tensor",
      "i32",
      "f32",
      "to",
      "step",

      // Op names (with dot).
      "test.addi",
      "tile.contract",
      "scalar.addf",
      "func.def",
      "func.decl",
      "func.template",
      "func.ukernel",
      "hal.buffer",

      // Boundary triggers: prefix characters alone.
      "%",
      "@",
      "#",
      "^",
      "\"",

      // Negative number vs arrow ambiguity.
      "->",
      "-1",
      "- 1",
      "-",
  };
  static constexpr size_t kFragmentCount =
      sizeof(fragments) / sizeof(fragments[0]);

  uint8_t count = fuzz_consume_u8(input) % 64 + 1;
  for (uint8_t i = 0; i < count && input->remaining > 0; ++i) {
    uint8_t index = fuzz_consume_u8(input) % kFragmentCount;
    const char* fragment = fragments[index];
    size_t fragment_length = strlen(fragment);
    if (source_length + fragment_length >= sizeof(source_buffer)) break;
    memcpy(source_buffer + source_length, fragment, fragment_length);
    source_length += fragment_length;
  }

  // Tokenize the constructed source.
  iree_string_view_t source = {source_buffer, source_length};
  fuzz_tokenizer_t tokenizer;
  fuzz_tokenizer_initialize(source, IREE_SV("<fuzz-grammar>"), &tokenizer);

  for (size_t i = 0; i < source_length + 16; ++i) {
    fuzz_assert_position_invariant(&tokenizer.tokenizer);
    loom_token_t token = loom_tokenizer_next(&tokenizer.tokenizer);
    fuzz_assert_position_invariant(&tokenizer.tokenizer);
    fuzz_assert_token_slice_valid(&tokenizer.tokenizer, &token);
    if (token.kind == LOOM_TOKEN_EOF) break;
  }

  IREE_CHECK_OK(loom_tokenizer_consume_status(&tokenizer.tokenizer));
  fuzz_tokenizer_deinitialize(&tokenizer);
}

//===----------------------------------------------------------------------===//
// Strategy 3: Truncation stress
//===----------------------------------------------------------------------===//
// Takes a well-formed source string and truncates it at a position chosen
// by the fuzzer. This targets the most dangerous code paths in the scanner:
// - Escape character at EOF (backslash with nothing after it)
// - Prefix character at EOF (%, @, #, ^ with no following identifier)
// - Number scanning interrupted mid-exponent
// - String literal interrupted mid-escape
// - Hex literal interrupted after 0x
// - Arrow interrupted after '-'
// - Cross-module symbol interrupted after ::
//
// The truncation point is drawn from fuzzer input, so libFuzzer can learn
// which positions trigger interesting behavior.

static void fuzz_strategy_truncation(fuzz_input_t* input) {
  // A syntactically rich source covering all token kinds and edge cases.
  // The interesting truncation points are distributed throughout.
  static const char kRichSource[] =
      "%result = test.addi %lhs, %rhs : i32\n"
      "test.func @main(%a: f32) -> (f32) {\n"
      "  %x = scalar.addf %a, %a : f32\n"
      "  ^bb0:\n"
      "  test.return %x : f32\n"
      "}\n"
      "// Comment at end\n"
      "@mod\n"
      "#q8_0 #enc\n"
      "\"escaped \\\"quotes\\\" here\"\n"
      "\"unterminated\\\n"
      "\"unicode \\u0041 \\u03BB \\uD83D\\uDD25\"\n"
      "\"bad unicode \\uD83D \\uDD25 \\u12G4\"\n"
      "0xFF 0x 0x0\n"
      "1.0e-5 1.0E 1.0e\n"
      "-42 ->  -\n"
      ": :\n"
      "<nested<deep<est>>>\n"
      "<\"str with \\\"esc\\\" in angles\">\n"
      "%  @  #  ^  \"\n";
  static constexpr size_t kRichSourceSize = sizeof(kRichSource) - 1;

  // Pick a truncation point from the fuzzer input.
  uint16_t raw_position = fuzz_consume_u16(input);
  size_t truncation_point = raw_position % (kRichSourceSize + 1);

  iree_string_view_t source = {kRichSource, truncation_point};
  fuzz_tokenizer_t tokenizer;
  fuzz_tokenizer_initialize(source, IREE_SV("<fuzz-trunc>"), &tokenizer);

  for (size_t i = 0; i < truncation_point + 16; ++i) {
    fuzz_assert_position_invariant(&tokenizer.tokenizer);
    loom_token_t token = loom_tokenizer_next(&tokenizer.tokenizer);
    fuzz_assert_position_invariant(&tokenizer.tokenizer);
    fuzz_assert_token_slice_valid(&tokenizer.tokenizer, &token);
    if (token.kind == LOOM_TOKEN_EOF) break;
  }

  IREE_CHECK_OK(loom_tokenizer_consume_status(&tokenizer.tokenizer));
  fuzz_tokenizer_deinitialize(&tokenizer);
}

//===----------------------------------------------------------------------===//
// Strategy 4: Angle bracket nesting with embedded strings
//===----------------------------------------------------------------------===//
// Generates deeply nested angle bracket structures. This stresses
// loom_tokenizer_scan_angle_interior, which must handle:
// - Arbitrary nesting depth
// - String literals inside brackets (with their own escape handling)
// - Balanced vs unbalanced brackets
// - Interleaved strings containing '>' that shouldn't close brackets
// - EOF inside nested brackets (unterminated)

static void fuzz_strategy_angle_nesting(fuzz_input_t* input) {
  char source_buffer[512];
  size_t source_length = 0;

  // Start with '<' to trigger angle bracket scanning.
  source_buffer[source_length++] = '<';

  uint8_t element_count = fuzz_consume_u8(input) % 48 + 1;
  int nesting_depth = 1;

  for (uint8_t i = 0; i < element_count && input->remaining > 0; ++i) {
    if (source_length >= sizeof(source_buffer) - 32) break;

    uint8_t action = fuzz_consume_u8(input) % 8;
    switch (action) {
      case 0:
        // Open nested angle bracket.
        source_buffer[source_length++] = '<';
        ++nesting_depth;
        break;
      case 1:
        // Close angle bracket.
        if (nesting_depth > 0) {
          source_buffer[source_length++] = '>';
          --nesting_depth;
        }
        break;
      case 2:
        // Embed a simple string.
        source_buffer[source_length++] = '"';
        source_buffer[source_length++] = 'a';
        source_buffer[source_length++] = '"';
        break;
      case 3:
        // Embed a string containing '>'. The scanner must recognize
        // this '>' is inside a string, not bracket closure.
        source_buffer[source_length++] = '"';
        source_buffer[source_length++] = '>';
        source_buffer[source_length++] = '"';
        break;
      case 4:
        // Embed a string with escaped quote. The scanner must not
        // be fooled into thinking the string ended early.
        source_buffer[source_length++] = '"';
        source_buffer[source_length++] = '\\';
        source_buffer[source_length++] = '"';
        source_buffer[source_length++] = 'x';
        source_buffer[source_length++] = '"';
        break;
      case 5:
        // Embed content text.
        source_buffer[source_length++] = 'i';
        source_buffer[source_length++] = '3';
        source_buffer[source_length++] = '2';
        break;
      case 6:
        // Newline inside brackets (tests line tracking).
        source_buffer[source_length++] = '\n';
        break;
      case 7:
        // Embed a string with escape at end (stress boundary).
        source_buffer[source_length++] = '"';
        source_buffer[source_length++] = '\\';
        break;
    }
  }

  // Consume the opening '<' and test scan_angle_interior.
  iree_string_view_t source = {source_buffer, source_length};
  fuzz_tokenizer_t tokenizer;
  fuzz_tokenizer_initialize(source, IREE_SV("<fuzz-angles>"), &tokenizer);

  loom_token_t opening = loom_tokenizer_next(&tokenizer.tokenizer);
  if (opening.kind == LOOM_TOKEN_LANGLE) {
    iree_string_view_t interior;
    IREE_CHECK_OK(
        loom_tokenizer_scan_angle_interior(&tokenizer.tokenizer, &interior));
    if (tokenizer.tokenizer.peeked.kind != LOOM_TOKEN_ERROR) {
      // Interior must be within the source buffer.
      if (interior.data < source_buffer ||
          interior.data + interior.size > source_buffer + source_length) {
        __builtin_trap();
      }
    }
  }

  fuzz_assert_position_invariant(&tokenizer.tokenizer);
  IREE_CHECK_OK(loom_tokenizer_consume_status(&tokenizer.tokenizer));
  fuzz_tokenizer_deinitialize(&tokenizer);
}

//===----------------------------------------------------------------------===//
// Strategy 5: API interleaving stress
//===----------------------------------------------------------------------===//
// Exercises the public API in unusual call sequences. The tokenizer
// supports one-token lookahead via peek/next, and several convenience
// methods (at, try_consume, expect, at_keyword, try_consume_keyword).
// This strategy calls them in arbitrary order driven by fuzzer input
// to verify that:
// - peek() is idempotent (repeated calls return same token)
// - next() after peek() returns the peeked token
// - try_consume(wrong_kind) doesn't advance
// - expect(wrong_kind) returns an error without corrupting state
// - All combinations leave the tokenizer in a consistent state

static void fuzz_strategy_api_interleave(const uint8_t* data, size_t size,
                                         fuzz_input_t* input) {
  iree_string_view_t source = {reinterpret_cast<const char*>(data), size};
  fuzz_tokenizer_t tokenizer;
  fuzz_tokenizer_initialize(source, IREE_SV("<fuzz-api>"), &tokenizer);

  // Use the original source as both the source text and the control
  // sequence. The first half is source, the second half drives API calls.
  uint8_t call_count = fuzz_consume_u8(input) % 128 + 1;

  for (uint8_t i = 0; i < call_count && input->remaining > 0; ++i) {
    fuzz_assert_position_invariant(&tokenizer.tokenizer);

    uint8_t action = fuzz_consume_u8(input) % 10;
    switch (action) {
      case 0: {
        // peek — should be idempotent.
        loom_token_t first = loom_tokenizer_peek(&tokenizer.tokenizer);
        loom_token_t second = loom_tokenizer_peek(&tokenizer.tokenizer);
        if (first.kind != second.kind) __builtin_trap();
        fuzz_assert_token_slice_valid(&tokenizer.tokenizer, &first);
        break;
      }
      case 1: {
        // next — consume a token.
        loom_token_t token = loom_tokenizer_next(&tokenizer.tokenizer);
        fuzz_assert_token_slice_valid(&tokenizer.tokenizer, &token);
        break;
      }
      case 2: {
        // peek then next — must return same token.
        loom_token_t peeked = loom_tokenizer_peek(&tokenizer.tokenizer);
        loom_token_t consumed = loom_tokenizer_next(&tokenizer.tokenizer);
        if (peeked.kind != consumed.kind) __builtin_trap();
        if (peeked.text.data != consumed.text.data) __builtin_trap();
        if (peeked.text.size != consumed.text.size) __builtin_trap();
        break;
      }
      case 3: {
        // at — check without consuming.
        uint8_t kind_byte = fuzz_consume_u8(input);
        loom_token_kind_t kind =
            static_cast<loom_token_kind_t>(kind_byte % (LOOM_TOKEN_EOF + 1));
        loom_tokenizer_at(&tokenizer.tokenizer, kind);
        break;
      }
      case 4: {
        // try_consume — should not advance on mismatch.
        loom_token_t before = loom_tokenizer_peek(&tokenizer.tokenizer);
        uint8_t kind_byte = fuzz_consume_u8(input);
        loom_token_kind_t kind =
            static_cast<loom_token_kind_t>(kind_byte % (LOOM_TOKEN_ERROR + 1));
        bool consumed = loom_tokenizer_try_consume(&tokenizer.tokenizer, kind);
        if (!consumed) {
          // Should not have advanced.
          loom_token_t after = loom_tokenizer_peek(&tokenizer.tokenizer);
          if (before.kind != after.kind) __builtin_trap();
        }
        break;
      }
      case 5: {
        // try_consume — should return false on mismatch without corrupting.
        uint8_t kind_byte = fuzz_consume_u8(input);
        loom_token_kind_t kind =
            static_cast<loom_token_kind_t>(kind_byte % (LOOM_TOKEN_ERROR + 1));
        loom_tokenizer_try_consume(&tokenizer.tokenizer, kind);
        break;
      }
      case 6: {
        // at_keyword — check for a specific keyword.
        static const iree_string_view_t keywords[] = {
            IREE_SV("to"),   IREE_SV("step"),  IREE_SV("tile"),
            IREE_SV("else"), IREE_SV("group"),
        };
        uint8_t keyword_index = fuzz_consume_u8(input) % 5;
        loom_tokenizer_at_keyword(&tokenizer.tokenizer,
                                  keywords[keyword_index]);
        break;
      }
      case 7: {
        // try_consume_keyword.
        static const iree_string_view_t keywords[] = {
            IREE_SV("to"),   IREE_SV("step"),  IREE_SV("tile"),
            IREE_SV("else"), IREE_SV("group"),
        };
        uint8_t keyword_index = fuzz_consume_u8(input) % 5;
        loom_tokenizer_try_consume_keyword(&tokenizer.tokenizer,
                                           keywords[keyword_index]);
        break;
      }
      case 8: {
        // Consume '<' then scan angle interior if we're at '<'.
        if (loom_tokenizer_try_consume(&tokenizer.tokenizer,
                                       LOOM_TOKEN_LANGLE)) {
          iree_string_view_t interior;
          IREE_CHECK_OK(loom_tokenizer_scan_angle_interior(&tokenizer.tokenizer,
                                                           &interior));
        }
        break;
      }
      case 9: {
        // consume_status — verify no infrastructure failure is pending.
        IREE_CHECK_OK(loom_tokenizer_consume_status(&tokenizer.tokenizer));
        break;
      }
    }
  }

  IREE_CHECK_OK(loom_tokenizer_consume_status(&tokenizer.tokenizer));
  fuzz_tokenizer_deinitialize(&tokenizer);
}

//===----------------------------------------------------------------------===//
// Main fuzzer entry point
//===----------------------------------------------------------------------===//

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size == 0) return 0;

  // Use the first byte to select which strategy gets the most attention
  // for this input, but always run the raw tokenizer (it's cheap and
  // catches the most fundamental issues).
  fuzz_input_t input = {data, size};
  uint8_t strategy = fuzz_consume_u8(&input);

  // Strategy 1 always runs: raw tokenize the entire input.
  fuzz_strategy_raw_tokenize(data, size);

  // Select a directed strategy based on the first byte.
  switch (strategy % 4) {
    case 0:
      fuzz_strategy_grammar_aware(&input);
      break;
    case 1:
      fuzz_strategy_truncation(&input);
      break;
    case 2:
      fuzz_strategy_angle_nesting(&input);
      break;
    case 3:
      fuzz_strategy_api_interleave(data, size, &input);
      break;
  }

  return 0;
}
