# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Applies exact algebraic rewrites to the embedded FlashAttention source."""

import argparse
from pathlib import Path

_EXPECTED_EXPORT = (
    'export("qwen3_moe_flash_attention_f32_f16_wmma") '
    "@qwen3_moe_flash_attention_f32_f16_wmma"
)
_TAIL_SUBTRACTION = (
    "  %tail_key_value_token_count = index.sub "
    "%bounded_key_value_token_count, %full_key_value_token_count : index"
)
_TAIL_REMAINDER = (
    "  %tail_key_value_token_count = index.rem "
    "%bounded_key_value_token_count, %sixtyfour : index"
)
_FULL_TILE_BOUND_SUBTRACTION = (
    "    %last_full_key_tile_start = index.sub "
    "%bounded_key_value_token_count, %fifteen : index"
)
_BOUNDED_FULL_TILE_SUBTRACTION = (
    "    %full_tile_key_value_token_count = index.assume "
    "%bounded_key_value_token_count "
    "[range(%bounded_key_value_token_count, 64, 32768)] : index\n"
    "    %last_full_key_tile_start = index.sub "
    "%full_tile_key_value_token_count, %fifteen : index"
)
_TAIL_SCORE_WAVE = "  %tail_score_wave = index.cmp ult, %subgroup, %two : index"
_TAIL_SCORE_WAVE_WITH_COUNTS = (
    f"{_TAIL_SCORE_WAVE}\n"
    "  %first_tail_key_count = index.min "
    "%tail_key_value_token_count, %thirtytwo : index\n"
    "  %second_tail_key_count = index.rem "
    "%tail_key_value_token_count, %thirtytwo : index"
)
_TAIL_REMAINING_SUBTRACTION = (
    "    %tail_remaining = index.sub "
    "%bounded_key_value_token_count, %tail_key_origin : index\n"
    "    %tail_key_count = index.min %tail_remaining, %thirtytwo : index"
)
_TAIL_TILE_COUNT_SELECTION = (
    "    %is_first_tail_tile = index.cmp eq, "
    "%tail_key_origin, %full_key_value_token_count : index\n"
    "    %tail_key_count = scf.select %is_first_tail_tile, "
    "%first_tail_key_count, %second_tail_key_count : index"
)


def _require_exactly_once(source: str, text: str, description: str) -> None:
    occurrence_count = source.count(text)
    if occurrence_count != 1:
        raise ValueError(
            f"expected exactly one {description}, found {occurrence_count}"
        )


def rewrite_flash_attention_source(source: str) -> str:
    """Rewrites bounded expressions while rejecting unexpected source drift."""
    _require_exactly_once(source, _EXPECTED_EXPORT, "FlashAttention export")
    subtraction_count = source.count("index.sub")
    if subtraction_count != 3:
        raise ValueError(
            "expected exactly three FlashAttention index.sub operations, "
            f"found {subtraction_count}"
        )

    rewrites = (
        (
            _TAIL_SUBTRACTION,
            _TAIL_REMAINDER,
            "FlashAttention tail subtraction",
        ),
        (
            _FULL_TILE_BOUND_SUBTRACTION,
            _BOUNDED_FULL_TILE_SUBTRACTION,
            "FlashAttention full-tile bound subtraction",
        ),
        (
            _TAIL_SCORE_WAVE,
            _TAIL_SCORE_WAVE_WITH_COUNTS,
            "FlashAttention tail-tile count anchor",
        ),
        (
            _TAIL_REMAINING_SUBTRACTION,
            _TAIL_TILE_COUNT_SELECTION,
            "FlashAttention tail-remaining subtraction",
        ),
    )
    rewritten_source = source
    for expected_text, replacement_text, description in rewrites:
        _require_exactly_once(rewritten_source, expected_text, description)
        rewritten_source = rewritten_source.replace(expected_text, replacement_text, 1)

    rewritten_subtraction_count = rewritten_source.count("index.sub")
    if rewritten_subtraction_count != 1:
        raise ValueError(
            "expected exactly one bounded FlashAttention index.sub operation "
            f"after rewriting, found {rewritten_subtraction_count}"
        )
    return rewritten_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    try:
        rewritten_source = rewrite_flash_attention_source(source)
    except ValueError as error:
        raise SystemExit(f"FlashAttention source rewrite failed: {error}") from error
    args.output.write_text(rewritten_source, encoding="utf-8")


if __name__ == "__main__":
    main()
