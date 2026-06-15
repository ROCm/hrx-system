# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the hand-written loom IR tokenizer."""

import pytest

from loom.format.text.tokenizer import (
    ParseError,
    Token,
    Tokenizer,
    TokenKind,
)

# ============================================================================
# Helpers
# ============================================================================


def _tokens(source: str) -> list[Token]:
    """Tokenize a string and return all non-EOF tokens."""
    tokenizer = Tokenizer(source)
    result = []
    while True:
        token = tokenizer.next()
        if token.kind == TokenKind.EOF:
            break
        result.append(token)
    return result


def _kinds(source: str) -> list[TokenKind]:
    """Return just the token kinds."""
    return [t.kind for t in _tokens(source)]


def _texts(source: str) -> list[str]:
    """Return just the token texts."""
    return [t.text for t in _tokens(source)]


# ============================================================================
# Individual token types
# ============================================================================


class TestSSAValue:
    def test_named(self) -> None:
        assert _texts("%x") == ["x"]
        assert _kinds("%x") == [TokenKind.SSA_VALUE]

    def test_numeric(self) -> None:
        assert _texts("%0") == ["0"]
        assert _texts("%42") == ["42"]

    def test_complex_name(self) -> None:
        assert _texts("%arg0") == ["arg0"]
        assert _texts("%contract0") == ["contract0"]
        assert _texts("%M") == ["M"]

    def test_underscore(self) -> None:
        assert _texts("%_foo") == ["_foo"]
        assert _texts("%foo_bar") == ["foo_bar"]

    def test_dollar(self) -> None:
        assert _texts("%$var") == ["$var"]

    def test_error_bare_percent(self) -> None:
        with pytest.raises(ParseError, match="expected identifier"):
            _tokens("% ")


class TestSymbol:
    def test_simple(self) -> None:
        assert _texts("@main") == ["main"]
        assert _kinds("@main") == [TokenKind.SYMBOL]

    def test_complex(self) -> None:
        assert _texts("@dn_layer") == ["dn_layer"]

    def test_error_bare_at(self) -> None:
        with pytest.raises(ParseError, match="expected identifier"):
            _tokens("@ ")


class TestHashAttr:
    def test_simple(self) -> None:
        assert _texts("#q8_0") == ["q8_0"]
        assert _kinds("#q8_0") == [TokenKind.HASH_ATTR]

    def test_encoding_name(self) -> None:
        assert _texts("#enc") == ["enc"]
        assert _texts("#q6_k") == ["q6_k"]

    def test_error_numeric_name(self) -> None:
        with pytest.raises(ParseError, match="expected identifier after '#'"):
            _tokens("#0")


class TestBlockLabel:
    def test_simple(self) -> None:
        assert _texts("^bb0") == ["bb0"]
        assert _kinds("^bb0") == [TokenKind.BLOCK_LABEL]

    def test_named(self) -> None:
        assert _texts("^entry") == ["entry"]


class TestInteger:
    def test_simple(self) -> None:
        assert _texts("42") == ["42"]
        assert _kinds("42") == [TokenKind.INTEGER]

    def test_zero(self) -> None:
        assert _texts("0") == ["0"]

    def test_negative(self) -> None:
        assert _texts("-1") == ["-1"]
        assert _kinds("-1") == [TokenKind.INTEGER]

    def test_hex(self) -> None:
        assert _texts("0xFF") == ["0xFF"]
        assert _kinds("0xFF") == [TokenKind.INTEGER]

    def test_negative_hex(self) -> None:
        assert _texts("-0x1A") == ["-0x1A"]


class TestFloat:
    def test_simple(self) -> None:
        assert _texts("3.14") == ["3.14"]
        assert _kinds("3.14") == [TokenKind.FLOAT]

    def test_negative(self) -> None:
        assert _texts("-0.5") == ["-0.5"]

    def test_exponent(self) -> None:
        assert _texts("1.0e-5") == ["1.0e-5"]
        assert _kinds("1.0e-5") == [TokenKind.FLOAT]

    def test_exponent_only(self) -> None:
        assert _texts("1e5") == ["1e5"]
        assert _kinds("1e5") == [TokenKind.FLOAT]

    def test_negative_exponent(self) -> None:
        assert _texts("2.5E+10") == ["2.5E+10"]


class TestString:
    def test_simple(self) -> None:
        assert _texts('"hello"') == ["hello"]
        assert _kinds('"hello"') == [TokenKind.STRING]

    def test_empty(self) -> None:
        assert _texts('""') == [""]

    def test_with_spaces(self) -> None:
        assert _texts('"hello world"') == ["hello world"]

    def test_unterminated(self) -> None:
        with pytest.raises(ParseError, match="unterminated string"):
            _tokens('"hello')

    def test_escaped_quote(self) -> None:
        tokens = _tokens(r'"has \"quotes\""')
        assert len(tokens) == 1
        assert tokens[0].kind == TokenKind.STRING
        assert tokens[0].text == 'has "quotes"'

    def test_escaped_backslash(self) -> None:
        tokens = _tokens(r'"path\\to\\file"')
        assert len(tokens) == 1
        assert tokens[0].kind == TokenKind.STRING
        assert tokens[0].text == r"path\to\file"

    def test_json_escapes(self) -> None:
        tokens = _tokens(r'"\/\b\f\n\r\t"')
        assert len(tokens) == 1
        assert tokens[0].text == "/\b\f\n\r\t"

    def test_unicode_escapes(self) -> None:
        tokens = _tokens(r'"A=\u0041 \u03BB \uD83D\uDD25"')
        assert len(tokens) == 1
        assert tokens[0].text == "A=A λ 🔥"

    def test_escaped_newline(self) -> None:
        tokens = _tokens(r'"line1\nline2"')
        assert len(tokens) == 1
        assert tokens[0].text == "line1\nline2"

    def test_unterminated_escape(self) -> None:
        with pytest.raises(ParseError, match="unterminated string literal"):
            _tokens('"trailing\\')

    def test_invalid_escape(self) -> None:
        with pytest.raises(ParseError, match=r"invalid escape sequence '\\x'"):
            _tokens(r'"\x"')

    def test_truncated_unicode_escape(self) -> None:
        with pytest.raises(ParseError, match="truncated unicode escape"):
            _tokens(r'"\u12"')

    def test_invalid_unicode_escape_hex_digit(self) -> None:
        with pytest.raises(ParseError, match="invalid hex digit 'G'"):
            _tokens(r'"\u12G4"')

    def test_lone_high_surrogate_escape(self) -> None:
        with pytest.raises(
            ParseError, match="high surrogate not followed by low surrogate"
        ):
            _tokens(r'"\uD83D"')

    def test_invalid_low_surrogate_escape(self) -> None:
        with pytest.raises(ParseError, match="invalid low surrogate U\\+0041"):
            _tokens(r'"\uD83D\u0041"')

    def test_unexpected_low_surrogate_escape(self) -> None:
        with pytest.raises(ParseError, match="unexpected low surrogate U\\+DD25"):
            _tokens(r'"\uDD25"')

    def test_raw_control_character(self) -> None:
        with pytest.raises(ParseError, match="unescaped control character U\\+0001"):
            _tokens('"\x01"')

    def test_raw_newline(self) -> None:
        with pytest.raises(ParseError, match="unescaped control character U\\+000A"):
            _tokens('"line1\nline2"')


class TestBareIdent:
    def test_keyword(self) -> None:
        assert _texts("to") == ["to"]
        assert _kinds("to") == [TokenKind.BARE_IDENT]

    def test_type_keyword(self) -> None:
        assert _texts("f32") == ["f32"]
        assert _kinds("f32") == [TokenKind.BARE_IDENT]

    def test_index(self) -> None:
        assert _texts("index") == ["index"]

    def test_true_false(self) -> None:
        assert _texts("true") == ["true"]
        assert _texts("false") == ["false"]

    def test_various_keywords(self) -> None:
        for kw in [
            "step",
            "iter_args",
            "where",
            "as",
            "else",
            "public",
            "host",
            "device",
        ]:
            assert _texts(kw) == [kw]
            assert _kinds(kw) == [TokenKind.BARE_IDENT]


class TestOpName:
    def test_simple(self) -> None:
        assert _texts("tile.contract") == ["tile.contract"]
        assert _kinds("tile.contract") == [TokenKind.OP_NAME]

    def test_three_segments(self) -> None:
        assert _texts("loom.tile.contract") == ["loom.tile.contract"]
        assert _kinds("loom.tile.contract") == [TokenKind.OP_NAME]

    def test_scalar(self) -> None:
        assert _texts("scalar.addi") == ["scalar.addi"]

    def test_scf(self) -> None:
        assert _texts("scf.for") == ["scf.for"]
        assert _texts("scf.yield") == ["scf.yield"]

    def test_disambiguation_from_type_keyword(self) -> None:
        """'tile' before '.' is OP_NAME start, 'tile' before '<' is BARE_IDENT."""
        assert _kinds("tile.contract") == [TokenKind.OP_NAME]
        assert _kinds("tile") == [TokenKind.BARE_IDENT]


class TestPunctuation:
    def test_all_single_char(self) -> None:
        source = "( ) { } [ ] < > = : ,"
        expected = [
            TokenKind.LPAREN,
            TokenKind.RPAREN,
            TokenKind.LBRACE,
            TokenKind.RBRACE,
            TokenKind.LBRACKET,
            TokenKind.RBRACKET,
            TokenKind.LANGLE,
            TokenKind.RANGLE,
            TokenKind.EQUALS,
            TokenKind.COLON,
            TokenKind.COMMA,
        ]
        assert _kinds(source) == expected

    def test_arrow(self) -> None:
        assert _texts("->") == ["->"]
        assert _kinds("->") == [TokenKind.ARROW]

    def test_arrow_vs_negative(self) -> None:
        """'->' is ARROW, '-1' is negative INTEGER."""
        tokens = _tokens("-> -1")
        assert tokens[0].kind == TokenKind.ARROW
        assert tokens[1].kind == TokenKind.INTEGER
        assert tokens[1].text == "-1"

    def test_consecutive_colons(self) -> None:
        tokens = _tokens(": :")
        assert tokens[0].kind == TokenKind.COLON
        assert tokens[1].kind == TokenKind.COLON


# ============================================================================
# Comments
# ============================================================================


class TestComments:
    def test_line_comment_skipped(self) -> None:
        tokens = _tokens("// this is a comment\n%x")
        assert len(tokens) == 1
        assert tokens[0].kind == TokenKind.SSA_VALUE

    def test_comment_collected(self) -> None:
        tokenizer = Tokenizer("// hello\n%x")
        tokenizer.next()  # consume %x
        # Comments were already collected during scanning.
        # Need to call collect before consuming %x.
        # Let me re-test with proper ordering.
        tokenizer2 = Tokenizer("// hello\n%x")
        tokenizer2.peek()  # triggers scan, collects comment
        comments = tokenizer2.collect_pending_comments()
        assert comments == [" hello"]

    def test_multiple_comments(self) -> None:
        tokenizer = Tokenizer("// first\n// second\n%x")
        tokenizer.peek()
        comments = tokenizer.collect_pending_comments()
        assert comments == [" first", " second"]

    def test_inline_comment(self) -> None:
        tokens = _tokens("%x // comment\n%y")
        assert len(tokens) == 2
        assert tokens[0].text == "x"
        assert tokens[1].text == "y"


# ============================================================================
# Whitespace
# ============================================================================


class TestWhitespace:
    def test_spaces(self) -> None:
        assert _texts("%a   %b") == ["a", "b"]

    def test_tabs(self) -> None:
        assert _texts("%a\t%b") == ["a", "b"]

    def test_newlines(self) -> None:
        assert _texts("%a\n%b") == ["a", "b"]

    def test_mixed(self) -> None:
        assert _texts("  %a \t\n  %b  ") == ["a", "b"]


# ============================================================================
# Source locations
# ============================================================================


class TestSourceLocations:
    def test_first_token(self) -> None:
        token = _tokens("%x")[0]
        assert token.location.line == 1
        assert token.location.column == 1

    def test_second_token_same_line(self) -> None:
        tokens = _tokens("%a %b")
        assert tokens[1].location.line == 1
        assert tokens[1].location.column == 4

    def test_second_line(self) -> None:
        tokens = _tokens("%a\n%b")
        assert tokens[0].location.line == 1
        assert tokens[1].location.line == 2
        assert tokens[1].location.column == 1

    def test_after_comment(self) -> None:
        tokens = _tokens("// comment\n%x")
        assert tokens[0].location.line == 2


# ============================================================================
# Multi-token sequences (real IR fragments)
# ============================================================================


class TestSequences:
    def test_binary_op(self) -> None:
        kinds = _kinds("%r = test.addi %a, %b : i32")
        assert kinds == [
            TokenKind.SSA_VALUE,
            TokenKind.EQUALS,
            TokenKind.OP_NAME,
            TokenKind.SSA_VALUE,
            TokenKind.COMMA,
            TokenKind.SSA_VALUE,
            TokenKind.COLON,
            TokenKind.BARE_IDENT,
        ]

    def test_constant(self) -> None:
        kinds = _kinds("%c42 = test.constant 42 : i32")
        assert kinds == [
            TokenKind.SSA_VALUE,
            TokenKind.EQUALS,
            TokenKind.OP_NAME,
            TokenKind.INTEGER,
            TokenKind.COLON,
            TokenKind.BARE_IDENT,
        ]

    def test_call(self) -> None:
        kinds = _kinds("%r = func.call @compute(%x, %y) : (f32, i32) -> (f32)")
        assert kinds == [
            TokenKind.SSA_VALUE,
            TokenKind.EQUALS,
            TokenKind.OP_NAME,
            TokenKind.SYMBOL,
            TokenKind.LPAREN,
            TokenKind.SSA_VALUE,
            TokenKind.COMMA,
            TokenKind.SSA_VALUE,
            TokenKind.RPAREN,
            TokenKind.COLON,
            TokenKind.LPAREN,
            TokenKind.BARE_IDENT,
            TokenKind.COMMA,
            TokenKind.BARE_IDENT,
            TokenKind.RPAREN,
            TokenKind.ARROW,
            TokenKind.LPAREN,
            TokenKind.BARE_IDENT,
            TokenKind.RPAREN,
        ]

    def test_yield(self) -> None:
        kinds = _kinds("test.yield %a, %b : f32, i32")
        assert kinds == [
            TokenKind.OP_NAME,
            TokenKind.SSA_VALUE,
            TokenKind.COMMA,
            TokenKind.SSA_VALUE,
            TokenKind.COLON,
            TokenKind.BARE_IDENT,
            TokenKind.COMMA,
            TokenKind.BARE_IDENT,
        ]

    def test_slice_with_index_list(self) -> None:
        kinds = _kinds("%t = tile.slice %src[0, %off]")
        assert kinds == [
            TokenKind.SSA_VALUE,
            TokenKind.EQUALS,
            TokenKind.OP_NAME,
            TokenKind.SSA_VALUE,
            TokenKind.LBRACKET,
            TokenKind.INTEGER,
            TokenKind.COMMA,
            TokenKind.SSA_VALUE,
            TokenKind.RBRACKET,
        ]

    def test_tied_result(self) -> None:
        texts = _texts("-> (%tensor as tensor<[%M]xf32>)")
        assert texts[0] == "->"
        assert texts[1] == "("
        assert texts[2] == "tensor"
        assert texts[3] == "as"

    def test_function_signature(self) -> None:
        kinds = _kinds("func.def public @f(%a: f32) -> (f32)")
        assert kinds == [
            TokenKind.OP_NAME,  # func.def
            TokenKind.BARE_IDENT,  # public
            TokenKind.SYMBOL,  # @f
            TokenKind.LPAREN,
            TokenKind.SSA_VALUE,
            TokenKind.COLON,
            TokenKind.BARE_IDENT,
            TokenKind.RPAREN,  # (%a: f32)
            TokenKind.ARROW,
            TokenKind.LPAREN,
            TokenKind.BARE_IDENT,
            TokenKind.RPAREN,
        ]

    def test_region(self) -> None:
        kinds = _kinds("{\n  test.yield %a : f32\n}")
        assert kinds == [
            TokenKind.LBRACE,
            TokenKind.OP_NAME,
            TokenKind.SSA_VALUE,
            TokenKind.COLON,
            TokenKind.BARE_IDENT,
            TokenKind.RBRACE,
        ]


# ============================================================================
# Tokenizer interface methods
# ============================================================================


class TestInterface:
    def test_peek_does_not_consume(self) -> None:
        tokenizer = Tokenizer("%x %y")
        first = tokenizer.peek()
        assert first.text == "x"
        second = tokenizer.peek()
        assert second.text == "x"  # same token

    def test_next_consumes(self) -> None:
        tokenizer = Tokenizer("%x %y")
        first = tokenizer.next()
        assert first.text == "x"
        second = tokenizer.next()
        assert second.text == "y"

    def test_at(self) -> None:
        tokenizer = Tokenizer("%x")
        assert tokenizer.at(TokenKind.SSA_VALUE)
        assert tokenizer.at(TokenKind.SSA_VALUE, "x")
        assert not tokenizer.at(TokenKind.INTEGER)
        assert not tokenizer.at(TokenKind.SSA_VALUE, "y")

    def test_expect_success(self) -> None:
        tokenizer = Tokenizer("%x")
        token = tokenizer.expect(TokenKind.SSA_VALUE)
        assert token.text == "x"

    def test_expect_failure(self) -> None:
        tokenizer = Tokenizer("%x")
        with pytest.raises(ParseError, match="expected INTEGER"):
            tokenizer.expect(TokenKind.INTEGER)

    def test_try_consume_success(self) -> None:
        tokenizer = Tokenizer("%x %y")
        token = tokenizer.try_consume(TokenKind.SSA_VALUE)
        assert token is not None
        assert token.text == "x"

    def test_try_consume_failure(self) -> None:
        tokenizer = Tokenizer("%x")
        token = tokenizer.try_consume(TokenKind.INTEGER)
        assert token is None
        # Token not consumed.
        assert tokenizer.peek().text == "x"

    def test_current_location(self) -> None:
        tokenizer = Tokenizer("%x\n%y")
        location = tokenizer.current_location()
        assert location.line == 1
        assert location.column == 1
        assert location.offset == 0
        tokenizer.next()  # %x
        tokenizer.peek()  # triggers scan past newline
        location = tokenizer.current_location()
        assert location.line == 2
        assert location.column == 3
        assert location.offset == 5

    def test_token_end_location(self) -> None:
        tokenizer = Tokenizer("%x\nop.name")

        value_token = tokenizer.next()
        assert value_token.location.line == 1
        assert value_token.location.column == 1
        assert value_token.location.offset == 0
        assert value_token.end_location.line == 1
        assert value_token.end_location.column == 3
        assert value_token.end_location.offset == 2

        op_token = tokenizer.next()
        assert op_token.location.line == 2
        assert op_token.location.column == 1
        assert op_token.location.offset == 3
        assert op_token.end_location.line == 2
        assert op_token.end_location.column == 8
        assert op_token.end_location.offset == 10

    def test_eof(self) -> None:
        tokenizer = Tokenizer("")
        assert tokenizer.at(TokenKind.EOF)
        token = tokenizer.next()
        assert token.kind == TokenKind.EOF
        assert token.location == token.end_location


# ============================================================================
# Angle bracket scanning (for type parser)
# ============================================================================


class TestAngleBracketScan:
    def test_simple(self) -> None:
        tokenizer = Tokenizer("<4xf32>")
        tokenizer.expect(TokenKind.LANGLE)
        interior = tokenizer.scan_to_matching_angle_bracket()
        assert interior == "4xf32"

    def test_nested(self) -> None:
        tokenizer = Tokenizer("<256xi8, #q8_0<block=32>>")
        tokenizer.expect(TokenKind.LANGLE)
        interior = tokenizer.scan_to_matching_angle_bracket()
        assert interior == "256xi8, #q8_0<block=32>"

    def test_deeply_nested(self) -> None:
        tokenizer = Tokenizer("<a<b<c>>>")
        tokenizer.expect(TokenKind.LANGLE)
        interior = tokenizer.scan_to_matching_angle_bracket()
        assert interior == "a<b<c>>"

    def test_string_with_brackets(self) -> None:
        """Brackets inside strings are ignored during scanning."""
        tokenizer = Tokenizer('<"<tag>">>')
        tokenizer.expect(TokenKind.LANGLE)
        interior = tokenizer.scan_to_matching_angle_bracket()
        assert interior == '"<tag>"'

    def test_string_with_nested_brackets(self) -> None:
        tokenizer = Tokenizer('<a<">>">>')
        tokenizer.expect(TokenKind.LANGLE)
        interior = tokenizer.scan_to_matching_angle_bracket()
        # a< starts depth 2, ">>" is inside string (ignored),
        # > closes inner (depth 1), > closes outer (depth 0).
        assert interior == 'a<">>">'

    def test_string_with_escaped_quote(self) -> None:
        tokenizer = Tokenizer(r'<label="a\"b">')
        tokenizer.expect(TokenKind.LANGLE)
        interior = tokenizer.scan_to_matching_angle_bracket()
        assert interior == r'label="a\"b"'

    def test_string_with_raw_newline_is_invalid(self) -> None:
        tokenizer = Tokenizer('<"line1\nline2">')
        tokenizer.expect(TokenKind.LANGLE)
        with pytest.raises(ParseError, match="unescaped control character U\\+000A"):
            tokenizer.scan_to_matching_angle_bracket()

    def test_unterminated(self) -> None:
        tokenizer = Tokenizer("<no closing")
        tokenizer.expect(TokenKind.LANGLE)
        with pytest.raises(ParseError, match="unterminated"):
            tokenizer.scan_to_matching_angle_bracket()


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_adjacent_punctuation(self) -> None:
        """No spaces between punctuation."""
        kinds = _kinds("(%a)")
        assert kinds == [
            TokenKind.LPAREN,
            TokenKind.SSA_VALUE,
            TokenKind.RPAREN,
        ]

    def test_type_in_angle_brackets(self) -> None:
        """Type keywords inside angle brackets are BARE_IDENT."""
        kinds = _kinds("tile < f32 >")
        assert kinds == [
            TokenKind.BARE_IDENT,  # tile
            TokenKind.LANGLE,
            TokenKind.BARE_IDENT,  # f32
            TokenKind.RANGLE,
        ]

    def test_op_name_vs_type(self) -> None:
        """tile.contract is OP_NAME, tile alone is BARE_IDENT."""
        tokens = _tokens("tile.contract tile")
        assert tokens[0].kind == TokenKind.OP_NAME
        assert tokens[0].text == "tile.contract"
        assert tokens[1].kind == TokenKind.BARE_IDENT
        assert tokens[1].text == "tile"

    def test_negative_sign_error(self) -> None:
        """'-' not followed by '>' or digit is an error."""
        with pytest.raises(ParseError, match="unexpected '-'"):
            _tokens("- x")

    def test_hash_error(self) -> None:
        """'#' not followed by an identifier is an error."""
        with pytest.raises(ParseError, match="expected identifier after '#'"):
            _tokens("# ")

    def test_unexpected_character(self) -> None:
        with pytest.raises(ParseError, match="unexpected character"):
            _tokens("~")

    def test_float_dot_followed_by_non_digit(self) -> None:
        """'4 . x' — dot is not a valid standalone token, so errors."""
        with pytest.raises(ParseError, match="unexpected character"):
            _tokens("4 . x")

    def test_negative_special_float(self) -> None:
        tokens = _tokens("-inf -nan")
        assert [(token.kind, token.text) for token in tokens] == [
            (TokenKind.FLOAT, "-inf"),
            (TokenKind.FLOAT, "-nan"),
        ]

    def test_empty_input(self) -> None:
        tokens = _tokens("")
        assert tokens == []

    def test_only_whitespace(self) -> None:
        tokens = _tokens("   \n\t  ")
        assert tokens == []

    def test_only_comments(self) -> None:
        tokens = _tokens("// just a comment\n// another")
        assert tokens == []
