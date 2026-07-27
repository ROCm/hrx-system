# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Hand-written character scanner for loom IR text format.

Produces a lazy stream of tokens from source text. Each token carries
its kind, text, and source location (line:col) for error reporting.

This is a single-pass scanner with one character of lookahead. No
regex. The scanner handles all disambiguation:
  - '#' letter → HASH_ATTR, '#' digit is invalid
  - '-' '>' → ARROW, '-' digit → negative number
  - identifier '.' identifier → OP_NAME, bare identifier → BARE_IDENT
  - '-' may continue identifiers for descriptor keys such as pass names
  - 'tile' before '<' → BARE_IDENT (type keyword), 'tile' before '.' → OP_NAME

Comments (//) are collected separately and not emitted as tokens.
The parser retrieves them via collect_pending_comments() and attaches
them to the next operation for round-trip preservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique

__all__ = [
    "TokenKind",
    "Token",
    "SourceLocation",
    "ParseError",
    "Tokenizer",
]


# ============================================================================
# Source location
# ============================================================================


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Position in source text."""

    line: int
    column: int
    offset: int = 0


# ============================================================================
# Token types
# ============================================================================


@unique
class TokenKind(IntEnum):
    """Token kinds produced by the scanner."""

    # Literals.
    INTEGER = 0
    FLOAT = 1
    STRING = 2

    # References.
    SSA_VALUE = 3  # %name, %0, %arg0
    SYMBOL = 4  # @name
    HASH_ATTR = 5  # #q8_0, #enc
    BLOCK_LABEL = 6  # ^bb0, ^entry

    # Identifiers.
    BARE_IDENT = 7  # keyword, type name, or other bare identifier
    OP_NAME = 8  # dotted name: tile.contract, scalar.addi

    # Punctuation.
    LPAREN = 9
    RPAREN = 10
    LBRACE = 11
    RBRACE = 12
    LBRACKET = 13
    RBRACKET = 14
    LANGLE = 15
    RANGLE = 16
    EQUALS = 17
    COLON = 18
    COMMA = 19
    ARROW = 20  # ->
    DIM_X = 21  # 'x' dimension separator (only when in_dim_list)
    PIPE = 22  # |

    # Special.
    EOF = 23


@dataclass(frozen=True, slots=True)
class Token:
    """A lexical token with source location."""

    kind: TokenKind
    text: str
    location: SourceLocation
    end_location: SourceLocation

    def __repr__(self) -> str:
        return (
            f"Token({self.kind.name}, {self.text!r}, "
            f"{self.location.line}:{self.location.column})"
        )


# ============================================================================
# Parse error
# ============================================================================


class ParseError(Exception):
    """Error with source location."""

    def __init__(
        self, message: str, location: SourceLocation, filename: str = "<input>"
    ) -> None:
        self.location = location
        self.filename = filename
        super().__init__(
            f"{filename}:{location.line}:{location.column}: error: {message}"
        )


# ============================================================================
# Character classification
# ============================================================================


def _is_ident_start(character: str) -> bool:
    """Characters that can start an identifier: [a-zA-Z_$]."""
    return character.isalpha() or character == "_" or character == "$"


def _is_ident_continue(character: str) -> bool:
    """Characters that can continue an identifier: [a-zA-Z0-9_$.-]."""
    return (
        character.isalnum()
        or character == "_"
        or character == "$"
        or character == "."
        or character == "-"
    )


def _is_ident_continue_no_dot(character: str) -> bool:
    """Identifier continuation without dot (for bare identifiers)."""
    return (
        character.isalnum() or character == "_" or character == "$" or character == "-"
    )


_JSON_ESCAPE_CHARS: dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _is_unicode_high_surrogate(codepoint: int) -> bool:
    """Returns true for UTF-16 high surrogate code units."""
    return 0xD800 <= codepoint <= 0xDBFF


def _is_unicode_low_surrogate(codepoint: int) -> bool:
    """Returns true for UTF-16 low surrogate code units."""
    return 0xDC00 <= codepoint <= 0xDFFF


# ============================================================================
# Tokenizer
# ============================================================================


class Tokenizer:
    """Hand-written character scanner for loom IR.

    Usage:
        tokenizer = Tokenizer(source, "model.loom")
        while not tokenizer.at(TokenKind.EOF):
            token = tokenizer.next()
            ...
    """

    __slots__ = (
        "_source",
        "_filename",
        "_position",
        "_line",
        "_column",
        "_peeked",
        "_comments",
        "in_dim_list",
    )

    def __init__(self, source: str, filename: str = "<input>") -> None:
        self._source = source
        self._filename = filename
        self._position = 0
        self._line = 1
        self._column = 1
        self._peeked: Token | None = None
        self._comments: list[str] = []
        self.in_dim_list: bool = False

    # --- Public interface ---

    def peek(self) -> Token:
        """Return the next token without consuming it."""
        if self._peeked is None:
            self._peeked = self._scan_token()
        return self._peeked

    def peek_n(self, offset: int) -> Token:
        """Return the token at |offset| without consuming scanner state."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        saved_position = self._position
        saved_line = self._line
        saved_column = self._column
        saved_peeked = self._peeked
        saved_comments = list(self._comments)
        saved_in_dim_list = self.in_dim_list
        token = self.peek()
        try:
            for _ in range(offset + 1):
                token = self.next()
            return token
        finally:
            self._position = saved_position
            self._line = saved_line
            self._column = saved_column
            self._peeked = saved_peeked
            self._comments = saved_comments
            self.in_dim_list = saved_in_dim_list

    def next(self) -> Token:
        """Consume and return the next token."""
        if self._peeked is not None:
            token = self._peeked
            self._peeked = None
            return token
        return self._scan_token()

    def expect(self, kind: TokenKind, text: str | None = None) -> Token:
        """Consume the next token, raising ParseError if it doesn't match."""
        token = self.next()
        if token.kind != kind:
            expected = f"{kind.name}"
            if text is not None:
                expected = repr(text)
            raise ParseError(
                f"expected {expected}, got {token.kind.name} {token.text!r}",
                token.location,
                self._filename,
            )
        if text is not None and token.text != text:
            raise ParseError(
                f"expected {text!r}, got {token.text!r}", token.location, self._filename
            )
        return token

    def at(self, kind: TokenKind, text: str | None = None) -> bool:
        """Check if the next token matches without consuming."""
        token = self.peek()
        if token.kind != kind:
            return False
        if text is not None and token.text != text:
            return False
        return True

    def try_consume(self, kind: TokenKind, text: str | None = None) -> Token | None:
        """Consume if matches, return None otherwise."""
        if self.at(kind, text):
            return self.next()
        return None

    def collect_pending_comments(self) -> list[str]:
        """Return and clear accumulated comment lines."""
        comments = self._comments
        self._comments = []
        return comments

    def current_location(self) -> SourceLocation:
        """Current scanner position (for range tracking by the parser)."""
        return SourceLocation(self._line, self._column, self._position)

    def scan_to_matching_angle_bracket(self) -> str:
        """Scan from current position to the matching '>'.

        The opening '<' must already have been consumed. Handles
        nested angle brackets (for encodings like #q8_0<block=32>>)
        and ignores brackets inside string literals.
        Returns the text between the brackets (exclusive).
        """
        start = self._position
        depth = 1
        while self._position < len(self._source) and depth > 0:
            character = self._source[self._position]
            if character == '"':
                string_location = SourceLocation(
                    self._line, self._column, self._position
                )
                self._advance()
                self._scan_string_content(string_location, decode=False)
                continue
            if character == "<":
                depth += 1
            elif character == ">":
                depth -= 1
                if depth == 0:
                    interior = self._source[start : self._position]
                    self._advance()  # consume the closing '>'
                    return interior
            if character == "\n":
                self._line += 1
                self._column = 0  # advance() will increment to 1
            self._advance()
        raise ParseError(
            "unterminated '<' — no matching '>'",
            SourceLocation(self._line, self._column, self._position),
            self._filename,
        )

    # --- Scanning ---

    def _advance(self) -> None:
        """Advance position by one character."""
        self._position += 1
        self._column += 1

    def _char(self) -> str:
        """Current character, or NUL at EOF.

        Returns '\\0' (not empty string) at EOF so that `_char() in "abc"`
        and `_char().isdigit()` etc. behave correctly. Empty string is a
        substring of any string in Python, which causes infinite loops.
        """
        if self._position < len(self._source):
            return self._source[self._position]
        return "\0"

    def _peek_char(self, offset: int = 1) -> str:
        """Character at position + offset, or NUL at EOF."""
        index = self._position + offset
        if index < len(self._source):
            return self._source[index]
        return "\0"

    def _make_token(
        self, kind: TokenKind, text: str, location: SourceLocation
    ) -> Token:
        """Create a token."""
        return Token(kind, text, location, self.current_location())

    def _skip_whitespace_and_comments(self) -> None:
        """Skip whitespace and collect comments."""
        while self._position < len(self._source):
            character = self._char()

            # Whitespace.
            if character in " \t\r":
                self._advance()
                continue
            if character == "\n":
                self._line += 1
                self._column = 0
                self._advance()
                continue

            # Comment: // to end of line.
            if character == "/" and self._peek_char() == "/":
                comment_start = self._position + 2
                self._advance()  # skip first /
                self._advance()  # skip second /
                while self._position < len(self._source) and self._char() != "\n":
                    self._advance()
                comment_text = self._source[comment_start : self._position]
                if comment_text.endswith("\r"):
                    comment_text = comment_text[:-1]
                self._comments.append(comment_text)
                continue

            break

    def _scan_token(self) -> Token:
        """Scan and return the next token."""
        self._skip_whitespace_and_comments()

        if self._position >= len(self._source):
            return self._make_token(
                TokenKind.EOF,
                "",
                SourceLocation(self._line, self._column, self._position),
            )

        location = SourceLocation(self._line, self._column, self._position)
        character = self._char()

        # SSA value: %name or %digits.
        if character == "%":
            return self._scan_ssa_value(location)

        # Symbol: @name.
        if character == "@":
            return self._scan_symbol(location)

        # Hash attr.
        if character == "#":
            return self._scan_hash(location)

        # Block label: ^name.
        if character == "^":
            return self._scan_block_label(location)

        # String literal.
        if character == '"':
            return self._scan_string(location)

        # Negative number or arrow.
        if character == "-":
            if self._peek_char() == ">":
                self._advance()
                self._advance()
                return self._make_token(TokenKind.ARROW, "->", location)
            if self._peek_char().isdigit():
                return self._scan_number(location)
            if self._source.startswith(
                "-inf", self._position
            ) and not _is_ident_continue(self._peek_char(4)):
                self._advance()
                self._advance()
                self._advance()
                self._advance()
                return self._make_token(TokenKind.FLOAT, "-inf", location)
            if self._source.startswith(
                "-nan", self._position
            ) and not _is_ident_continue(self._peek_char(4)):
                self._advance()
                self._advance()
                self._advance()
                self._advance()
                return self._make_token(TokenKind.FLOAT, "-nan", location)
            raise ParseError(
                "unexpected '-' (not followed by '>' or digit)",
                location,
                self._filename,
            )

        # Number.
        if character.isdigit():
            return self._scan_number(location)

        # Dimension separator: in a dim list, 'x' is a single-character
        # separator token rather than an identifier start.
        if self.in_dim_list and character == "x":
            self._advance()
            return self._make_token(TokenKind.DIM_X, "x", location)

        # Identifier (may become BARE_IDENT or OP_NAME).
        if _is_ident_start(character):
            return self._scan_identifier(location)

        # Colon.
        if character == ":":
            self._advance()
            return self._make_token(TokenKind.COLON, ":", location)

        # Single-character punctuation.
        punctuation = {
            "(": TokenKind.LPAREN,
            ")": TokenKind.RPAREN,
            "{": TokenKind.LBRACE,
            "}": TokenKind.RBRACE,
            "[": TokenKind.LBRACKET,
            "]": TokenKind.RBRACKET,
            "<": TokenKind.LANGLE,
            ">": TokenKind.RANGLE,
            "=": TokenKind.EQUALS,
            ",": TokenKind.COMMA,
            "|": TokenKind.PIPE,
        }
        if character in punctuation:
            self._advance()
            return self._make_token(punctuation[character], character, location)

        raise ParseError(
            f"unexpected character {character!r}", location, self._filename
        )

    # --- Individual token scanners ---

    def _scan_ssa_value(self, location: SourceLocation) -> Token:
        """Scan %name or %digits.

        Token text is the bare name without the '%' sigil: %foo -> "foo".
        """
        self._advance()  # skip %
        name_start = self._position
        if self._char().isdigit():
            while self._char().isdigit():
                self._advance()
        elif _is_ident_start(self._char()):
            while _is_ident_continue_no_dot(self._char()):
                self._advance()
        else:
            raise ParseError(
                "expected identifier or digit after '%'", location, self._filename
            )
        text = self._source[name_start : self._position]
        return self._make_token(TokenKind.SSA_VALUE, text, location)

    def _scan_symbol(self, location: SourceLocation) -> Token:
        """Scan @name.

        Token text is the bare name without the '@' sigil: @foo -> "foo".
        """
        self._advance()  # skip @
        name_start = self._position
        if not _is_ident_start(self._char()):
            raise ParseError("expected identifier after '@'", location, self._filename)
        while _is_ident_continue_no_dot(self._char()):
            self._advance()
        text = self._source[name_start : self._position]
        return self._make_token(TokenKind.SYMBOL, text, location)

    def _scan_hash(self, location: SourceLocation) -> Token:
        """Scan #name (hash attr).

        Token text is the bare name without the '#' sigil:
        #q8_0 -> "q8_0".
        """
        self._advance()  # skip #
        name_start = self._position
        if _is_ident_start(self._char()):
            while _is_ident_continue_no_dot(self._char()):
                self._advance()
            text = self._source[name_start : self._position]
            return self._make_token(TokenKind.HASH_ATTR, text, location)
        raise ParseError("expected identifier after '#'", location, self._filename)

    def _scan_block_label(self, location: SourceLocation) -> Token:
        """Scan ^name.

        Token text is the bare name without the '^' sigil: ^bb0 -> "bb0".
        """
        self._advance()  # skip ^
        name_start = self._position
        if not _is_ident_start(self._char()) and not self._char().isdigit():
            raise ParseError("expected identifier after '^'", location, self._filename)
        while _is_ident_continue_no_dot(self._char()):
            self._advance()
        text = self._source[name_start : self._position]
        return self._make_token(TokenKind.BLOCK_LABEL, text, location)

    def _scan_string(self, location: SourceLocation) -> Token:
        """Scan "..." string literal with escape sequences.

        Token text is the decoded payload without surrounding quotes, matching
        the C tokenizer contract and keeping parser callsites escape-agnostic.
        """
        self._advance()  # skip opening "
        text = self._scan_string_content(location, decode=True)
        assert text is not None
        return self._make_token(TokenKind.STRING, text, location)

    def _scan_string_content(
        self,
        location: SourceLocation,
        *,
        decode: bool,
    ) -> str | None:
        """Scan a string body after the opening quote has been consumed."""
        decoded_chunks: list[str] | None = [] if decode else None
        while self._position < len(self._source):
            character = self._char()
            if character == "\\":
                decoded = self._scan_string_escape()
                if decoded_chunks is not None:
                    decoded_chunks.append(decoded)
                continue
            if character == '"':
                self._advance()
                return "".join(decoded_chunks) if decoded_chunks is not None else None
            if ord(character) < 0x20:
                raise ParseError(
                    f"unescaped control character U+{ord(character):04X} "
                    f"in string literal",
                    SourceLocation(self._line, self._column, self._position),
                    self._filename,
                )
            if decoded_chunks is not None:
                decoded_chunks.append(character)
            self._advance()
        raise ParseError(
            "unterminated string literal at end of input", location, self._filename
        )

    def _scan_string_escape(self) -> str:
        """Scan one JSON-compatible escape and return its decoded payload."""
        assert self._char() == "\\"
        self._advance()
        if self._position >= len(self._source):
            raise ParseError(
                "unterminated string literal",
                SourceLocation(self._line, self._column, self._position),
                self._filename,
            )

        escaped = self._char()
        if escaped in _JSON_ESCAPE_CHARS:
            self._advance()
            return _JSON_ESCAPE_CHARS[escaped]
        if escaped != "u":
            raise ParseError(
                f"invalid escape sequence '\\{escaped}' in string literal",
                SourceLocation(self._line, self._column, self._position),
                self._filename,
            )

        self._advance()
        codepoint_location = SourceLocation(self._line, self._column, self._position)
        codepoint = self._scan_unicode_escape_codepoint()
        if _is_unicode_high_surrogate(codepoint):
            if self._char() != "\\" or self._peek_char() != "u":
                raise ParseError(
                    "high surrogate not followed by low surrogate",
                    SourceLocation(self._line, self._column, self._position),
                    self._filename,
                )
            self._advance()
            self._advance()
            low_location = SourceLocation(self._line, self._column, self._position)
            low = self._scan_unicode_escape_codepoint()
            if not _is_unicode_low_surrogate(low):
                raise ParseError(
                    f"invalid low surrogate U+{low:04X}",
                    low_location,
                    self._filename,
                )
            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
        elif _is_unicode_low_surrogate(codepoint):
            raise ParseError(
                f"unexpected low surrogate U+{codepoint:04X}",
                codepoint_location,
                self._filename,
            )
        return chr(codepoint)

    def _scan_unicode_escape_codepoint(self) -> int:
        """Scan 4 hex digits after '\\u' and return the decoded code unit."""
        value = 0
        for _ in range(4):
            if self._position >= len(self._source):
                raise ParseError(
                    "truncated unicode escape in string literal",
                    SourceLocation(self._line, self._column, self._position),
                    self._filename,
                )
            character = self._char()
            if character == '"':
                raise ParseError(
                    "truncated unicode escape in string literal",
                    SourceLocation(self._line, self._column, self._position),
                    self._filename,
                )
            if "0" <= character <= "9":
                digit = ord(character) - ord("0")
            elif "a" <= character <= "f":
                digit = ord(character) - ord("a") + 10
            elif "A" <= character <= "F":
                digit = ord(character) - ord("A") + 10
            else:
                raise ParseError(
                    f"invalid hex digit '{character}' in unicode escape",
                    SourceLocation(self._line, self._column, self._position),
                    self._filename,
                )
            value = (value << 4) | digit
            self._advance()
        return value

    def _scan_number(self, location: SourceLocation) -> Token:
        """Scan integer or float literal."""
        start = self._position
        is_negative = self._char() == "-"
        if is_negative:
            self._advance()

        # Hex integer: 0x... (but not when in_dim_list — 'x' is a
        # dimension separator, so '0' is a static dim of size 0).
        if (
            self._char() == "0"
            and self._peek_char() in ("x", "X")
            and not self.in_dim_list
        ):
            self._advance()  # 0
            self._advance()  # x
            if self._char() not in "0123456789abcdefABCDEF":
                raise ParseError(
                    "expected hex digits after '0x'", location, self._filename
                )
            while self._char() in "0123456789abcdefABCDEF":
                self._advance()
            text = self._source[start : self._position]
            return self._make_token(TokenKind.INTEGER, text, location)

        # Decimal digits.
        while self._char().isdigit():
            self._advance()

        # Check for float: '.' or 'e'/'E'.
        is_float = False
        if self._char() == "." and self._peek_char().isdigit():
            is_float = True
            self._advance()  # skip '.'
            while self._char().isdigit():
                self._advance()

        if self._char() in ("e", "E"):
            is_float = True
            self._advance()  # skip e/E
            if self._char() in ("+", "-"):
                self._advance()
            if not self._char().isdigit():
                raise ParseError(
                    "expected digits in exponent", location, self._filename
                )
            while self._char().isdigit():
                self._advance()

        text = self._source[start : self._position]
        kind = TokenKind.FLOAT if is_float else TokenKind.INTEGER
        return self._make_token(kind, text, location)

    def _scan_identifier(self, location: SourceLocation) -> Token:
        """Scan a bare identifier or dotted op name.

        If the identifier contains dots (like tile.contract), it's
        an OP_NAME. Otherwise it's a BARE_IDENT.

        Special case: 'tile', 'tensor', 'group' followed by '<' are
        type keywords (BARE_IDENT), not the start of an op name.
        """
        start = self._position
        # Scan the first segment (no dots).
        while _is_ident_continue_no_dot(self._char()):
            self._advance()

        first_segment = self._source[start : self._position]

        # Check if there's a dot following — could be an op name.
        if self._char() == ".":
            # Scan the rest including dots.
            while self._char() == ".":
                self._advance()  # skip dot
                if not _is_ident_start(self._char()) and not self._char().isdigit():
                    # Trailing dot without continuation — back up.
                    self._position -= 1
                    self._column -= 1
                    break
                while _is_ident_continue_no_dot(self._char()):
                    self._advance()
            text = self._source[start : self._position]
            if "." in text:
                return self._make_token(TokenKind.OP_NAME, text, location)

        return self._make_token(TokenKind.BARE_IDENT, first_segment, location)
