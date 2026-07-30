# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from experimental.qwen.kernels import flash_attention_source_rewrite

_EXPORT = (
    'kernel.def export("qwen3_moe_flash_attention_f32_f16_wmma") '
    "@qwen3_moe_flash_attention_f32_f16_wmma"
)
_TAIL_SUBTRACTION = (
    "  %tail_key_value_token_count = index.sub "
    "%bounded_key_value_token_count, %full_key_value_token_count : index"
)
_FULL_TILE_BOUND_SUBTRACTION = (
    "    %last_full_key_tile_start = index.sub "
    "%bounded_key_value_token_count, %fifteen : index"
)
_TAIL_SCORE_WAVE = "  %tail_score_wave = index.cmp ult, %subgroup, %two : index"
_TAIL_REMAINING_SUBTRACTION = (
    "    %tail_remaining = index.sub "
    "%bounded_key_value_token_count, %tail_key_origin : index\n"
    "    %tail_key_count = index.min %tail_remaining, %thirtytwo : index"
)
_COMPLETE_SOURCE = "\n".join(
    (
        _EXPORT,
        _TAIL_SUBTRACTION,
        _FULL_TILE_BOUND_SUBTRACTION,
        _TAIL_SCORE_WAVE,
        _TAIL_REMAINING_SUBTRACTION,
        "",
    )
)


class FlashAttentionSourceRewriteTest(unittest.TestCase):
    def test_rewrites_every_address_width_invalid_subtraction(self):
        rewritten_source = (
            flash_attention_source_rewrite.rewrite_flash_attention_source(
                _COMPLETE_SOURCE
            )
        )

        self.assertIn(
            "%tail_key_value_token_count = index.rem "
            "%bounded_key_value_token_count, %sixtyfour : index",
            rewritten_source,
        )
        self.assertIn(
            "%full_tile_key_value_token_count = index.assume "
            "%bounded_key_value_token_count "
            "[range(%bounded_key_value_token_count, 64, 32768)] : index",
            rewritten_source,
        )
        self.assertIn(
            "%last_full_key_tile_start = index.sub "
            "%full_tile_key_value_token_count, %fifteen : index",
            rewritten_source,
        )
        self.assertIn(
            "  %first_tail_key_count = index.min "
            "%tail_key_value_token_count, %thirtytwo : index",
            rewritten_source,
        )
        self.assertIn(
            "  %second_tail_key_count = index.rem "
            "%tail_key_value_token_count, %thirtytwo : index",
            rewritten_source,
        )
        self.assertIn(
            "%tail_key_count = scf.select %is_first_tail_tile, "
            "%first_tail_key_count, %second_tail_key_count : index",
            rewritten_source,
        )
        self.assertEqual(rewritten_source.count("index.sub"), 1)
        self.assertIn(_EXPORT, rewritten_source)

    def test_rejects_missing_export(self):
        with self.assertRaisesRegex(ValueError, "FlashAttention export, found 0"):
            flash_attention_source_rewrite.rewrite_flash_attention_source(
                _COMPLETE_SOURCE.replace(_EXPORT, "")
            )

    def test_rejects_unexpected_subtraction_count(self):
        with self.assertRaisesRegex(
            ValueError,
            "three FlashAttention index.sub operations, found 4",
        ):
            flash_attention_source_rewrite.rewrite_flash_attention_source(
                _COMPLETE_SOURCE + "  %unexpected = index.sub %a, %b : index\n"
            )

    def test_rejects_source_drift_in_each_rewrite(self):
        expected_failures = (
            (_TAIL_SUBTRACTION, "tail subtraction"),
            (_FULL_TILE_BOUND_SUBTRACTION, "full-tile bound subtraction"),
            (_TAIL_SCORE_WAVE, "tail-tile count anchor"),
            (_TAIL_REMAINING_SUBTRACTION, "tail-remaining subtraction"),
        )
        for original_text, description in expected_failures:
            with self.subTest(description=description):
                drifted_text = (
                    original_text.replace("index.sub", "index.add")
                    if "index.sub" in original_text
                    else original_text.replace("%two", "%three")
                )
                drifted_source = _COMPLETE_SOURCE.replace(original_text, drifted_text)
                if "index.sub" in original_text:
                    drifted_source += "  %replacement = index.sub %a, %b : index\n"
                with self.assertRaisesRegex(ValueError, f"{description}, found 0"):
                    flash_attention_source_rewrite.rewrite_flash_attention_source(
                        drifted_source
                    )


if __name__ == "__main__":
    unittest.main()
