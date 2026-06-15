// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/tokenizer.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"

namespace loom {
namespace {

// RAII wrapper that deinitializes on scope exit.
class ScopedTokenizer {
 public:
  explicit ScopedTokenizer(const char* text) {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_tokenizer_initialize(iree_make_cstring_view(text), IREE_SV("<test>"),
                              &scratch_arena_, &t_);
  }
  ~ScopedTokenizer() {
    loom_tokenizer_deinitialize(&t_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }
  loom_tokenizer_t* get() { return &t_; }
  loom_token_t next() { return loom_tokenizer_next(&t_); }
  loom_token_t peek() { return loom_tokenizer_peek(&t_); }
  iree_host_size_t total_allocation_size() const {
    return scratch_arena_.total_allocation_size;
  }

 private:
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  loom_tokenizer_t t_;
};

static void ExpectNoInfrastructureStatus(loom_tokenizer_t* tokenizer) {
  IREE_EXPECT_OK(loom_tokenizer_consume_status(tokenizer));
}

static void ExpectErrorStringParam(const loom_tokenizer_error_t& error,
                                   iree_host_size_t param_index,
                                   iree_string_view_t expected) {
  ASSERT_LT(param_index, error.param_count);
  EXPECT_EQ(error.params[param_index].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(
      iree_string_view_equal(error.params[param_index].string, expected));
}

static void ExpectErrorU32Param(const loom_tokenizer_error_t& error,
                                iree_host_size_t param_index,
                                uint32_t expected) {
  ASSERT_LT(param_index, error.param_count);
  EXPECT_EQ(error.params[param_index].kind, LOOM_PARAM_U32);
  EXPECT_EQ(error.params[param_index].u32, expected);
}

static loom_token_t ExpectPeekedErrorToken(loom_tokenizer_t* tokenizer,
                                           const loom_error_def_t* error,
                                           iree_string_view_t source_text,
                                           uint32_t line, uint32_t column,
                                           uint32_t end_column) {
  loom_token_t token = loom_tokenizer_peek(tokenizer);
  EXPECT_EQ(token.kind, LOOM_TOKEN_ERROR);
  EXPECT_EQ(token.line, line);
  EXPECT_EQ(token.column, column);
  EXPECT_EQ(token.end_column, end_column);
  EXPECT_TRUE(iree_string_view_equal(token.text, source_text));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, source_text));
  EXPECT_EQ(tokenizer->error.error, error);
  EXPECT_EQ(tokenizer->error.line, line);
  EXPECT_EQ(tokenizer->error.column, column);
  EXPECT_EQ(tokenizer->error.end_column, end_column);
  return token;
}

static loom_token_t ExpectNextErrorToken(loom_tokenizer_t* tokenizer,
                                         const loom_error_def_t* error,
                                         iree_string_view_t source_text,
                                         uint32_t line, uint32_t column,
                                         uint32_t end_column) {
  ExpectPeekedErrorToken(tokenizer, error, source_text, line, column,
                         end_column);
  loom_token_t token = loom_tokenizer_next(tokenizer);
  EXPECT_EQ(token.kind, LOOM_TOKEN_ERROR);
  EXPECT_EQ(token.line, line);
  EXPECT_EQ(token.column, column);
  EXPECT_EQ(token.end_column, end_column);
  EXPECT_TRUE(iree_string_view_equal(token.text, source_text));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, source_text));
  EXPECT_EQ(tokenizer->error.error, error);
  return token;
}

//===----------------------------------------------------------------------===//
// Punctuation
//===----------------------------------------------------------------------===//

TEST(Tokenizer, SingleCharPunctuation) {
  ScopedTokenizer t("( ) { } [ ] < > = : ,");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LBRACE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RBRACE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LBRACKET);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RBRACKET);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RANGLE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EQUALS);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COMMA);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, Arrow) {
  ScopedTokenizer t("->");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_ARROW);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("->")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("->")));
}

TEST(Tokenizer, ColonBeforeColon) {
  ScopedTokenizer t(": :");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
}

//===----------------------------------------------------------------------===//
// Numbers
//===----------------------------------------------------------------------===//

TEST(Tokenizer, Integer) {
  ScopedTokenizer t("42");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("42")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("42")));
}

TEST(Tokenizer, NegativeInteger) {
  ScopedTokenizer t("-1");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("-1")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("-1")));
}

TEST(Tokenizer, HexInteger) {
  ScopedTokenizer t("0xFF");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("0xFF")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("0xFF")));
}

TEST(Tokenizer, Float) {
  ScopedTokenizer t("3.14");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_FLOAT);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("3.14")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("3.14")));
}

TEST(Tokenizer, FloatExponent) {
  ScopedTokenizer t("1.0e-5");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_FLOAT);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("1.0e-5")));
}

TEST(Tokenizer, NegativeFloat) {
  ScopedTokenizer t("-0.5");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_FLOAT);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("-0.5")));
}

TEST(Tokenizer, NegativeSpecialFloat) {
  ScopedTokenizer t("-inf -nan");
  loom_token_t inf_token = t.next();
  EXPECT_EQ(inf_token.kind, LOOM_TOKEN_FLOAT);
  EXPECT_TRUE(iree_string_view_equal(inf_token.text, IREE_SV("-inf")));
  loom_token_t nan_token = t.next();
  EXPECT_EQ(nan_token.kind, LOOM_TOKEN_FLOAT);
  EXPECT_TRUE(iree_string_view_equal(nan_token.text, IREE_SV("-nan")));
}

TEST(Tokenizer, NegativeArrowDisambiguation) {
  ScopedTokenizer t("-> -1");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_ARROW);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
}

//===----------------------------------------------------------------------===//
// Strings
//===----------------------------------------------------------------------===//

TEST(Tokenizer, SimpleString) {
  ScopedTokenizer t("\"hello\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("hello")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("\"hello\"")));
  EXPECT_EQ(token.text.data, token.source_text.data + 1);
}

TEST(Tokenizer, StringWithEscapes) {
  ScopedTokenizer t("\"has \\\"quotes\\\"\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("has \"quotes\"")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text,
                                     IREE_SV("\"has \\\"quotes\\\"\"")));
  EXPECT_NE(token.text.data, token.source_text.data + 1);
}

TEST(Tokenizer, StringWithNewlineAndTabEscapes) {
  ScopedTokenizer t("\"row\\n\\tcol\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("row\n\tcol")));
  EXPECT_TRUE(
      iree_string_view_equal(token.source_text, IREE_SV("\"row\\n\\tcol\"")));
}

TEST(Tokenizer, StringWithJsonEscapes) {
  ScopedTokenizer t("\"slash:\\/ backspace:\\b formfeed:\\f return:\\r\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(
      token.text, IREE_SV("slash:/ backspace:\b formfeed:\f return:\r")));
  EXPECT_TRUE(iree_string_view_equal(
      token.source_text,
      IREE_SV("\"slash:\\/ backspace:\\b formfeed:\\f return:\\r\"")));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, StringWithUnicodeEscapes) {
  ScopedTokenizer t("\"A=\\u0041 lambda=\\u03BB fire=\\uD83D\\uDD25\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(
      token.text, IREE_SV("A=A lambda=\xCE\xBB fire=\xF0\x9F\x94\xA5")));
  EXPECT_TRUE(iree_string_view_equal(
      token.source_text,
      IREE_SV("\"A=\\u0041 lambda=\\u03BB fire=\\uD83D\\uDD25\"")));
  EXPECT_NE(token.text.data, token.source_text.data + 1);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, EmptyString) {
  ScopedTokenizer t("\"\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("\"\"")));
}

TEST(Tokenizer, EscapedStringDecodeScratchIsReused) {
  ScopedTokenizer t(
      "\"long \\\"escaped\\\" string\" \"x\\n\" \"y\\t\" \"z\\\\\"");

  loom_token_t first = t.next();
  EXPECT_EQ(first.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(
      iree_string_view_equal(first.text, IREE_SV("long \"escaped\" string")));
  EXPECT_EQ(first.text.data, t.get()->decoded_string_data);
  char* decoded_string_data = t.get()->decoded_string_data;
  iree_host_size_t decoded_string_capacity = t.get()->decoded_string_capacity;
  iree_host_size_t total_allocation_size = t.total_allocation_size();
  EXPECT_NE(decoded_string_data, nullptr);
  EXPECT_GE(decoded_string_capacity, first.text.size);

  loom_token_t second = t.next();
  EXPECT_EQ(second.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(second.text, IREE_SV("x\n")));
  EXPECT_EQ(second.text.data, decoded_string_data);
  EXPECT_EQ(t.get()->decoded_string_data, decoded_string_data);
  EXPECT_EQ(t.get()->decoded_string_capacity, decoded_string_capacity);
  EXPECT_EQ(t.total_allocation_size(), total_allocation_size);

  loom_token_t third = t.next();
  EXPECT_EQ(third.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(third.text, IREE_SV("y\t")));
  EXPECT_EQ(third.text.data, decoded_string_data);
  EXPECT_EQ(t.total_allocation_size(), total_allocation_size);

  loom_token_t fourth = t.next();
  EXPECT_EQ(fourth.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(fourth.text, IREE_SV("z\\")));
  EXPECT_EQ(fourth.text.data, decoded_string_data);
  EXPECT_EQ(t.total_allocation_size(), total_allocation_size);

  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, UnescapedStringRemainsZeroCopyAfterEscapedScratchReuse) {
  ScopedTokenizer t("\"a\\n\" \"plain\"");

  loom_token_t escaped = t.next();
  EXPECT_EQ(escaped.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(escaped.text, IREE_SV("a\n")));
  EXPECT_EQ(escaped.text.data, t.get()->decoded_string_data);
  iree_host_size_t total_allocation_size = t.total_allocation_size();

  loom_token_t plain = t.next();
  EXPECT_EQ(plain.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(plain.text, IREE_SV("plain")));
  EXPECT_TRUE(iree_string_view_equal(plain.source_text, IREE_SV("\"plain\"")));
  EXPECT_EQ(plain.text.data, plain.source_text.data + 1);
  EXPECT_EQ(t.total_allocation_size(), total_allocation_size);

  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

//===----------------------------------------------------------------------===//
// References
//===----------------------------------------------------------------------===//

TEST(Tokenizer, SSAValue) {
  ScopedTokenizer t("%x %0 %arg0 %M %contract0");
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE) << "token " << i;
  }
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, SSAValueText) {
  ScopedTokenizer t("%tile");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("tile")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("%tile")));
}

TEST(Tokenizer, Symbol) {
  ScopedTokenizer t("@main");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SYMBOL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("main")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("@main")));
}

TEST(Tokenizer, DottedSymbol) {
  ScopedTokenizer t("@model36.model.hidden_size");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SYMBOL);
  EXPECT_TRUE(
      iree_string_view_equal(token.text, IREE_SV("model36.model.hidden_size")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text,
                                     IREE_SV("@model36.model.hidden_size")));
}

TEST(Tokenizer, HashAttr) {
  ScopedTokenizer t("#q8_0");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_HASH_ATTR);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("q8_0")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("#q8_0")));
}

TEST(Tokenizer, HashAttrRejectsNumericName) {
  ScopedTokenizer t("#0 #1");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("#"), 1, 1, 2);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("#"));
  loom_token_t value = t.next();
  EXPECT_EQ(value.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(value.text, IREE_SV("0")));
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("#"), 1, 4, 5);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("#"));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, BlockLabel) {
  ScopedTokenizer t("^bb0");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_BLOCK_LABEL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("bb0")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("^bb0")));
}

//===----------------------------------------------------------------------===//
// Identifiers and op names
//===----------------------------------------------------------------------===//

TEST(Tokenizer, BareIdent) {
  ScopedTokenizer t("tile tensor group to step else");
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(t.next().kind, LOOM_TOKEN_BARE_IDENT) << "token " << i;
  }
}

TEST(Tokenizer, BareIdentText) {
  ScopedTokenizer t("tile");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("tile")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("tile")));
}

TEST(Tokenizer, OpName) {
  ScopedTokenizer t("test.addi tile.contract scalar.addf");
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(t.next().kind, LOOM_TOKEN_OP_NAME) << "token " << i;
  }
}

TEST(Tokenizer, OpNameText) {
  ScopedTokenizer t("test.addi");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_OP_NAME);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("test.addi")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("test.addi")));
}

TEST(Tokenizer, TokenKindName) {
  EXPECT_TRUE(iree_string_view_equal(loom_token_kind_name(LOOM_TOKEN_INTEGER),
                                     IREE_SV("integer")));
  EXPECT_TRUE(iree_string_view_equal(loom_token_kind_name(LOOM_TOKEN_ARROW),
                                     IREE_SV("'->'")));
  EXPECT_TRUE(iree_string_view_equal(loom_token_kind_name(LOOM_TOKEN_EOF),
                                     IREE_SV("end of file")));
}

//===----------------------------------------------------------------------===//
// Whitespace and comments
//===----------------------------------------------------------------------===//

TEST(Tokenizer, SkipsWhitespace) {
  ScopedTokenizer t("  42  ");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, SkipsComments) {
  ScopedTokenizer t("42 // comment\n99");
  loom_token_t token0 = t.next();
  EXPECT_EQ(token0.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token0.text, IREE_SV("42")));
  loom_token_t token1 = t.next();
  EXPECT_EQ(token1.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token1.text, IREE_SV("99")));
}

TEST(Tokenizer, CollectsPendingCommentsExactly) {
  ScopedTokenizer t("// first\n//second\n99");
  loom_token_t token = t.peek();
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  const iree_string_view_t* comments = NULL;
  iree_host_size_t comment_count = 0;
  loom_tokenizer_take_pending_comments(t.get(), &comments, &comment_count);
  ASSERT_EQ(comment_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(comments[0], IREE_SV(" first")));
  EXPECT_TRUE(iree_string_view_equal(comments[1], IREE_SV("second")));
  loom_tokenizer_take_pending_comments(t.get(), &comments, &comment_count);
  EXPECT_EQ(comment_count, 0u);
}

TEST(Tokenizer, EmptyInput) {
  ScopedTokenizer t("");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_EOF);
  EXPECT_TRUE(iree_string_view_is_empty(token.text));
  EXPECT_TRUE(iree_string_view_is_empty(token.source_text));
}

TEST(Tokenizer, OnlyWhitespace) {
  ScopedTokenizer t("   \n\t  ");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, OnlyComment) {
  ScopedTokenizer t("// nothing here");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

//===----------------------------------------------------------------------===//
// Location tracking
//===----------------------------------------------------------------------===//

TEST(Tokenizer, LineAndColumn) {
  ScopedTokenizer t("42\n  99");
  loom_token_t token0 = t.next();
  EXPECT_EQ(token0.line, 1u);
  EXPECT_EQ(token0.column, 1u);
  loom_token_t token1 = t.next();
  EXPECT_EQ(token1.line, 2u);
  EXPECT_EQ(token1.column, 3u);
}

//===----------------------------------------------------------------------===//
// Peek and consume helpers
//===----------------------------------------------------------------------===//

TEST(Tokenizer, PeekDoesNotConsume) {
  ScopedTokenizer t("42");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, At) {
  ScopedTokenizer t("42");
  EXPECT_TRUE(loom_tokenizer_at(t.get(), LOOM_TOKEN_INTEGER));
  EXPECT_FALSE(loom_tokenizer_at(t.get(), LOOM_TOKEN_FLOAT));
}

TEST(Tokenizer, TryConsume) {
  ScopedTokenizer t("42 3.14");
  EXPECT_FALSE(loom_tokenizer_try_consume(t.get(), LOOM_TOKEN_FLOAT));
  EXPECT_TRUE(loom_tokenizer_try_consume(t.get(), LOOM_TOKEN_INTEGER));
  EXPECT_TRUE(loom_tokenizer_try_consume(t.get(), LOOM_TOKEN_FLOAT));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, TryConsumeKeyword) {
  ScopedTokenizer t("to step");
  EXPECT_FALSE(loom_tokenizer_try_consume_keyword(t.get(), IREE_SV("step")));
  EXPECT_TRUE(loom_tokenizer_try_consume_keyword(t.get(), IREE_SV("to")));
  EXPECT_TRUE(loom_tokenizer_try_consume_keyword(t.get(), IREE_SV("step")));
}

TEST(Tokenizer, NextReturnsExpectedKind) {
  ScopedTokenizer t("42");
  loom_token_t token = loom_tokenizer_next(t.get());
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("42")));
}

TEST(Tokenizer, PeekDoesNotMatchWrongKind) {
  ScopedTokenizer t("42");
  loom_token_t token = loom_tokenizer_peek(t.get());
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_NE(token.kind, LOOM_TOKEN_FLOAT);
}

//===----------------------------------------------------------------------===//
// Error propagation
//===----------------------------------------------------------------------===//

TEST(Tokenizer, UnterminatedStringError) {
  ScopedTokenizer t("\"unterminated");
  ExpectPeekedErrorToken(t.get(),
                         loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5),
                         IREE_SV("\""), 1, 1, 2);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, ErrorSurvivesMultiplePeeks) {
  ScopedTokenizer t("\"unterminated");
  ExpectPeekedErrorToken(t.get(),
                         loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5),
                         IREE_SV("\""), 1, 1, 2);
  ExpectPeekedErrorToken(t.get(),
                         loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5),
                         IREE_SV("\""), 1, 1, 2);
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5),
                       IREE_SV("\""), 1, 1, 2);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, ScanErrorSurfacedOnNext) {
  ScopedTokenizer t("\"unterminated");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5),
                       IREE_SV("\""), 1, 1, 2);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, ErrorAfterValidTokens) {
  ScopedTokenizer t("42 \"\\x\" %ok");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\\x"), 1, 5, 7);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("unknown escape sequence"));
  loom_token_t value = t.next();
  EXPECT_EQ(value.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_TRUE(iree_string_view_equal(value.text, IREE_SV("ok")));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, UnexpectedCharacterError) {
  ScopedTokenizer t("~");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 25),
                       IREE_SV("~"), 1, 1, 2);
  EXPECT_EQ(t.get()->error.param_count, 0u);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidStringEscapeError) {
  ScopedTokenizer t("\"\\x\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\\x"), 1, 2, 4);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("unknown escape sequence"));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidStringEscapeMessage) {
  ScopedTokenizer t("\"\\x\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\\x"), 1, 2, 4);
  ASSERT_EQ(t.get()->error.param_count, 1u);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("unknown escape sequence"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, TruncatedUnicodeEscapeError) {
  ScopedTokenizer t("\"\\u12\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\""), 1, 6, 7);
  ExpectErrorStringParam(t.get()->error, 0,
                         IREE_SV("truncated unicode escape"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidUnicodeEscapeHexDigitError) {
  ScopedTokenizer t("\"\\u12G4\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("G"), 1, 6, 7);
  ExpectErrorStringParam(t.get()->error, 0,
                         IREE_SV("invalid hex digit in unicode escape"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, LoneHighSurrogateEscapeError) {
  ScopedTokenizer t("\"\\uD83D\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\\uD83D"), 1, 2, 8);
  ExpectErrorStringParam(
      t.get()->error, 0,
      IREE_SV("high surrogate not followed by low surrogate"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidLowSurrogateEscapeError) {
  ScopedTokenizer t("\"\\uD83D\\u0041\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("0041"), 1, 10, 14);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("invalid low surrogate"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, UnexpectedLowSurrogateEscapeError) {
  ScopedTokenizer t("\"\\uDD25\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("DD25"), 1, 4, 8);
  ExpectErrorStringParam(t.get()->error, 0,
                         IREE_SV("unexpected low surrogate"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, RawControlCharacterInStringError) {
  ScopedTokenizer t("\"\x01\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\x01"), 1, 2, 3);
  ExpectErrorStringParam(
      t.get()->error, 0,
      IREE_SV("unescaped control character in string literal"));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, RawNewlineInStringError) {
  ScopedTokenizer t("\"line1\nline2\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\n"), 1, 7, 8);
  ExpectErrorStringParam(
      t.get()->error, 0,
      IREE_SV("unescaped control character in string literal"));
  loom_token_t value = t.next();
  EXPECT_EQ(value.kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_TRUE(iree_string_view_equal(value.text, IREE_SV("line2")));
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5),
                       IREE_SV("\""), 2, 6, 7);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, BarePercentError) {
  ScopedTokenizer t("% ");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("%"), 1, 1, 2);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("%"));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, BareAtSignError) {
  ScopedTokenizer t("@ ");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("@"), 1, 1, 2);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("@"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, BareHashError) {
  ScopedTokenizer t("# ");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("#"), 1, 1, 2);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("#"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, BareCaretError) {
  ScopedTokenizer t("^ ");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("^"), 1, 1, 2);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("^"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, BarePercentAtEOFError) {
  ScopedTokenizer t("%");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("%"), 1, 1, 2);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("%"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, BareAtSignAtEOFError) {
  ScopedTokenizer t("@");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("@"), 1, 1, 2);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("@"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, PrefixStrippedFromSSAValue) {
  // Token text should be the name without the '%' prefix.
  ScopedTokenizer t("%arg0");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("arg0")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("%arg0")));
  // Source location still points at the '%'.
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 6u);
}

TEST(Tokenizer, PrefixStrippedFromSymbol) {
  // Token text should be the name without the '@' prefix.
  ScopedTokenizer t("@main");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SYMBOL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("main")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("@main")));
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 6u);
}

TEST(Tokenizer, PrefixStrippedFromHashAttr) {
  // Token text should be the name without the '#' prefix.
  ScopedTokenizer t("#enc");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_HASH_ATTR);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("enc")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("#enc")));
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 5u);
}

TEST(Tokenizer, HashAttrNumericNameAtEOFIsInvalid) {
  ScopedTokenizer t("#42");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 24),
                       IREE_SV("#"), 1, 1, 2);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("#"));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, PrefixStrippedFromBlockLabel) {
  // Token text should be the label without the '^' prefix.
  ScopedTokenizer t("^entry");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_BLOCK_LABEL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("entry")));
  EXPECT_TRUE(iree_string_view_equal(token.source_text, IREE_SV("^entry")));
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 7u);
}

//===----------------------------------------------------------------------===//
// Angle bracket scanning
//===----------------------------------------------------------------------===//

TEST(Tokenizer, AngleInterior) {
  ScopedTokenizer t("<4x4xf32>");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_equal(interior, IREE_SV("4x4xf32")));
}

TEST(Tokenizer, NestedAngleBrackets) {
  ScopedTokenizer t("<vm.ref<hal.buffer>>");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_equal(interior, IREE_SV("vm.ref<hal.buffer>")));
}

TEST(Tokenizer, AngleInteriorWithEscapedQuotes) {
  // Encoding parameter with escaped quotes: #enc<label="a\"b">
  ScopedTokenizer t("<label=\"a\\\"b\">");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_equal(interior, IREE_SV("label=\"a\\\"b\"")));
}

TEST(Tokenizer, AngleInteriorInvalidStringEscapeError) {
  ScopedTokenizer t("<label=\"\\x\">");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_is_empty(interior));
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\\x"), 1, 9, 11);
  ExpectErrorStringParam(t.get()->error, 0, IREE_SV("unknown escape sequence"));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, AngleInteriorEscapeAtEOF) {
  // Unterminated string ending with escape at EOF.
  ScopedTokenizer t("<\"foo\\");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_is_empty(interior));
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 5),
                       IREE_SV("\\"), 1, 6, 7);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, AngleInteriorStringWithNewline) {
  // JSON string escapes inside angle brackets are validated by the same
  // scanner as top-level string tokens.
  ScopedTokenizer t("<\"line1\\nline2\">");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_equal(interior, IREE_SV("\"line1\\nline2\"")));
  loom_token_t eof = t.next();
  EXPECT_EQ(eof.kind, LOOM_TOKEN_EOF);
  EXPECT_EQ(eof.line, 1u);
}

TEST(Tokenizer, AngleInteriorStringWithRawNewlineIsInvalid) {
  ScopedTokenizer t("<\"line1\nline2\">");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_is_empty(interior));
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 23),
                       IREE_SV("\n"), 1, 8, 9);
  ExpectErrorStringParam(
      t.get()->error, 0,
      IREE_SV("unescaped control character in string literal"));
  ExpectNoInfrastructureStatus(t.get());
}

//===----------------------------------------------------------------------===//
// Complex sequences
//===----------------------------------------------------------------------===//

TEST(Tokenizer, BinaryOp) {
  ScopedTokenizer t("%result = test.addi %lhs, %rhs : i32");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EQUALS);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_OP_NAME);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COMMA);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, FunctionSignature) {
  ScopedTokenizer t("test.func @main(%a: f32) -> (f32)");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_OP_NAME);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SYMBOL);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_ARROW);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

//===----------------------------------------------------------------------===//
// Unicode and UTF-8
//===----------------------------------------------------------------------===//

TEST(Tokenizer, ChineseStringLiteral) {
  // "你好" — two CJK characters, each 3 bytes in UTF-8.
  // U+4F60 = E4 BD A0, U+597D = E5 A5 BD.
  ScopedTokenizer t("\"\xe4\xbd\xa0\xe5\xa5\xbd\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  // Column tracking: " (col 1), 你 (col 2), 好 (col 3), " (col 4).
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 5u);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, EmojiStringLiteral) {
  // "🔥" — one emoji, 4 bytes in UTF-8 (U+1F525 = F0 9F 94 A5).
  ScopedTokenizer t("\"\xf0\x9f\x94\xa5\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  // Column tracking: " (col 1), 🔥 (col 2), " (col 3).
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 4u);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, MixedAsciiAndUtf8InString) {
  // "hello 世界 🌍" — ASCII, two CJK (3 bytes each), space, globe emoji (4
  // bytes).
  ScopedTokenizer t("\"hello \xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x8c\x8d\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  // Columns: " h e l l o   世 界   🌍 "
  //           1 2 3 4 5 6 7 8  9 10 11 12
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 13u);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, TwoByteUtf8ColumnTracking) {
  // "aä" — 'a' is 1 byte, 'ä' (U+00E4) is 2 bytes (C3 A4).
  // Columns: " (col 1), a (col 2), ä (col 3), " (col 4).
  ScopedTokenizer t("\"a\xc3\xa4\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 5u);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, ChineseInComment) {
  // Comment with Chinese, followed by a token to verify column tracking.
  ScopedTokenizer t(
      "// \xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c\n"
      "%x");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(token.line, 2u);
  EXPECT_EQ(token.column, 1u);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, CommentColumnTrackingAfterContent) {
  // "42 // comment\n%x" — verify %x is at line 2, column 1.
  ScopedTokenizer t("42 // comment\n%x");
  loom_token_t num = t.next();
  EXPECT_EQ(num.kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(num.line, 1u);
  EXPECT_EQ(num.column, 1u);
  loom_token_t val = t.next();
  EXPECT_EQ(val.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(val.line, 2u);
  EXPECT_EQ(val.column, 1u);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidUtf8InString) {
  // 0xFF is never valid as a UTF-8 lead byte.
  ScopedTokenizer t("\"\xff\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 19),
                       IREE_SV("\xff"), 1, 2, 3);
  ExpectErrorU32Param(t.get()->error, 0, 1u);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidUtf8TruncatedSequence) {
  // Start of a 3-byte sequence (0xE4) followed by only 1 continuation byte,
  // then the closing quote. The closing quote is not a valid continuation byte,
  // so the sequence is invalid.
  ScopedTokenizer t("\"\xe4\xbd\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 19),
                       IREE_SV("\xe4"), 1, 2, 3);
  ExpectErrorU32Param(t.get()->error, 0, 1u);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidUtf8OverlongSequence) {
  // Overlong encoding of '/' (U+002F): 0xC0 0xAF.
  ScopedTokenizer t("\"\xc0\xaf\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 19),
                       IREE_SV("\xc0"), 1, 2, 3);
  ExpectErrorU32Param(t.get()->error, 0, 1u);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidUtf8SurrogateRejected) {
  // UTF-8 encoding of surrogate U+D800: 0xED 0xA0 0x80.
  ScopedTokenizer t("\"\xed\xa0\x80\"");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 19),
                       IREE_SV("\xed"), 1, 2, 3);
  ExpectErrorU32Param(t.get()->error, 0, 1u);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, InvalidUtf8BareBytes) {
  // Bare continuation byte outside a string — not valid UTF-8.
  ScopedTokenizer t("\x80");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 19),
                       IREE_SV("\x80"), 1, 1, 2);
  ExpectErrorU32Param(t.get()->error, 0, 0u);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, ValidUnicodeOutsideStringError) {
  // Chinese character outside a string — valid UTF-8 but not a valid token.
  // Should produce an error mentioning the codepoint.
  ScopedTokenizer t("\xe4\xbd\xa0");
  ExpectNextErrorToken(t.get(),
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 25),
                       IREE_SV("\xe4\xbd\xa0"), 1, 1, 2);
  EXPECT_EQ(t.get()->error.param_count, 0u);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, Utf8StringFollowedByToken) {
  // Verify tokenizer recovers correctly after a UTF-8 string.
  ScopedTokenizer t("\"\xe4\xbd\xa0\" %x");
  loom_token_t str = t.next();
  EXPECT_EQ(str.kind, LOOM_TOKEN_STRING);
  loom_token_t val = t.next();
  EXPECT_EQ(val.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_TRUE(iree_string_view_equal(val.text, IREE_SV("x")));
  ExpectNoInfrastructureStatus(t.get());
}

TEST(Tokenizer, Utf8InStringEscapeInteraction) {
  // Escaped quote followed by UTF-8: "\"\xe4\xbd\xa0" inside string.
  ScopedTokenizer t("\"\\\"\xe4\xbd\xa0\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  ExpectNoInfrastructureStatus(t.get());
}

//===----------------------------------------------------------------------===//
// Dimension separator (in_dim_list mode)
//===----------------------------------------------------------------------===//

TEST(Tokenizer, DimXInDimList) {
  ScopedTokenizer t("4x4xf32");
  t.get()->in_dim_list = true;
  loom_token_t t0 = t.next();
  EXPECT_EQ(t0.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(t0.text, IREE_SV("4")));
  loom_token_t t1 = t.next();
  EXPECT_EQ(t1.kind, LOOM_TOKEN_DIM_X);
  EXPECT_TRUE(iree_string_view_equal(t1.text, IREE_SV("x")));
  loom_token_t t2 = t.next();
  EXPECT_EQ(t2.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(t2.text, IREE_SV("4")));
  loom_token_t t3 = t.next();
  EXPECT_EQ(t3.kind, LOOM_TOKEN_DIM_X);
  EXPECT_TRUE(iree_string_view_equal(t3.text, IREE_SV("x")));
  loom_token_t t4 = t.next();
  EXPECT_EQ(t4.kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_TRUE(iree_string_view_equal(t4.text, IREE_SV("f32")));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, DimXNotActiveOutsideDimList) {
  // Without in_dim_list, 'x' starts an identifier as before.
  ScopedTokenizer t("4x4xf32");
  loom_token_t t0 = t.next();
  EXPECT_EQ(t0.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(t0.text, IREE_SV("4")));
  loom_token_t t1 = t.next();
  EXPECT_EQ(t1.kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_TRUE(iree_string_view_equal(t1.text, IREE_SV("x4xf32")));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, DimXWithDynamicDims) {
  ScopedTokenizer t("[%M]x4xf32");
  t.get()->in_dim_list = true;
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LBRACKET);
  loom_token_t ssa = t.next();
  EXPECT_EQ(ssa.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_TRUE(iree_string_view_equal(ssa.text, IREE_SV("M")));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RBRACKET);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_DIM_X);
  loom_token_t dim = t.next();
  EXPECT_EQ(dim.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(dim.text, IREE_SV("4")));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_DIM_X);
  loom_token_t elem = t.next();
  EXPECT_EQ(elem.kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_TRUE(iree_string_view_equal(elem.text, IREE_SV("f32")));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, DimXPositionTracking) {
  // "4x[%M]xf32" — verify each token has correct column.
  // Columns are 1-indexed.
  ScopedTokenizer t("4x[%M]xf32");
  t.get()->in_dim_list = true;
  loom_token_t t0 = t.next();  // '4' at column 1
  EXPECT_EQ(t0.kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t0.column, 1u);
  loom_token_t t1 = t.next();  // 'x' at column 2
  EXPECT_EQ(t1.kind, LOOM_TOKEN_DIM_X);
  EXPECT_EQ(t1.column, 2u);
  loom_token_t t2 = t.next();  // '[' at column 3
  EXPECT_EQ(t2.kind, LOOM_TOKEN_LBRACKET);
  EXPECT_EQ(t2.column, 3u);
  loom_token_t t3 = t.next();  // '%M' at column 4 (column of '%')
  EXPECT_EQ(t3.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t3.column, 4u);
  loom_token_t t4 = t.next();  // ']' at column 6
  EXPECT_EQ(t4.kind, LOOM_TOKEN_RBRACKET);
  EXPECT_EQ(t4.column, 6u);
  loom_token_t t5 = t.next();  // 'x' at column 7
  EXPECT_EQ(t5.kind, LOOM_TOKEN_DIM_X);
  EXPECT_EQ(t5.column, 7u);
  loom_token_t t6 = t.next();  // 'f32' at column 8
  EXPECT_EQ(t6.kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_EQ(t6.column, 8u);
}

TEST(Tokenizer, DimXKindName) {
  EXPECT_TRUE(iree_string_view_equal(loom_token_kind_name(LOOM_TOKEN_DIM_X),
                                     IREE_SV("'x'")));
}

TEST(Tokenizer, DimXZeroDimNotHex) {
  // "0xf32" in dim list mode: '0' is a static dim, 'x' is DIM_X,
  // 'f32' is the element type. Must NOT scan '0xf32' as hex integer.
  ScopedTokenizer t("0xf32");
  t.get()->in_dim_list = true;
  loom_token_t t0 = t.next();
  EXPECT_EQ(t0.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(t0.text, IREE_SV("0")));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_DIM_X);
  loom_token_t t2 = t.next();
  EXPECT_EQ(t2.kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_TRUE(iree_string_view_equal(t2.text, IREE_SV("f32")));
}

TEST(Tokenizer, DimXClearedBeforeElementType) {
  // Simulates the parser lifecycle: set in_dim_list for dims, clear
  // before scanning element type. Verifies that clearing mid-stream
  // produces BARE_IDENT for the element type.
  ScopedTokenizer t("4xf32");
  t.get()->in_dim_list = true;
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);  // 4
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_DIM_X);    // x
  // Parser clears in_dim_list after consuming DIM_X.
  t.get()->in_dim_list = false;
  loom_token_t elem = t.next();  // f32
  EXPECT_EQ(elem.kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_TRUE(iree_string_view_equal(elem.text, IREE_SV("f32")));
}

}  // namespace
}  // namespace loom
