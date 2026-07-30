# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from experimental.qwen.kernels import flash_attention_bringup_workaround

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
_DYNAMIC_TAIL_STAGE_ALLOCATION = (
    "  %tail_key_value_stage = buffer.alloca "
    "%tail_key_value_stage_bytes "
    "{base_alignment = 16, memory_space = workgroup} : buffer"
)
_COMPLETE_SOURCE = "\n".join(
    (
        _EXPORT,
        _TAIL_SUBTRACTION,
        _FULL_TILE_BOUND_SUBTRACTION,
        _TAIL_SCORE_WAVE,
        _TAIL_REMAINING_SUBTRACTION,
        _DYNAMIC_TAIL_STAGE_ALLOCATION,
        "",
    )
)


class FlashAttentionBringupWorkaroundTest(unittest.TestCase):
    def test_applies_every_required_patch(self):
        patched_source = (
            flash_attention_bringup_workaround.apply_flash_attention_bringup_workaround(
                _COMPLETE_SOURCE
            )
        )

        self.assertIn(
            "%tail_key_value_token_count = index.rem "
            "%bounded_key_value_token_count, %sixtyfour : index",
            patched_source,
        )
        self.assertIn(
            "%full_tile_key_value_token_count = index.assume "
            "%bounded_key_value_token_count "
            "[range(%bounded_key_value_token_count, 64, 32768)] : index",
            patched_source,
        )
        self.assertIn(
            "%last_full_key_tile_start = index.sub "
            "%full_tile_key_value_token_count, %fifteen : index",
            patched_source,
        )
        self.assertIn(
            "  %first_tail_key_count = index.min "
            "%tail_key_value_token_count, %thirtytwo : index",
            patched_source,
        )
        self.assertIn(
            "  %second_tail_key_count = index.rem "
            "%tail_key_value_token_count, %thirtytwo : index",
            patched_source,
        )
        self.assertIn(
            "%tail_key_count = scf.select %is_first_tail_tile, "
            "%first_tail_key_count, %second_tail_key_count : index",
            patched_source,
        )
        self.assertIn(
            "%tail_key_value_stage = buffer.alloca "
            "%tail_key_value_stage_capacity "
            "{base_alignment = 16, memory_space = workgroup} : buffer",
            patched_source,
        )
        self.assertEqual(patched_source.count("index.sub"), 1)
        self.assertIn(_EXPORT, patched_source)

    def test_rejects_missing_export(self):
        with self.assertRaisesRegex(ValueError, "FlashAttention export, found 0"):
            flash_attention_bringup_workaround.apply_flash_attention_bringup_workaround(
                _COMPLETE_SOURCE.replace(_EXPORT, "")
            )

    def test_rejects_unexpected_subtraction_count(self):
        with self.assertRaisesRegex(
            ValueError,
            "three FlashAttention index.sub operations, found 4",
        ):
            flash_attention_bringup_workaround.apply_flash_attention_bringup_workaround(
                _COMPLETE_SOURCE + "  %unexpected = index.sub %a, %b : index\n"
            )

    def test_rejects_source_drift_in_each_patch(self):
        expected_failures = (
            (_TAIL_SUBTRACTION, "tail subtraction"),
            (_FULL_TILE_BOUND_SUBTRACTION, "full-tile bound subtraction"),
            (_TAIL_SCORE_WAVE, "tail-tile count anchor"),
            (_TAIL_REMAINING_SUBTRACTION, "tail-remaining subtraction"),
            (_DYNAMIC_TAIL_STAGE_ALLOCATION, "tail-stage allocation"),
        )
        for original_text, description in expected_failures:
            with self.subTest(description=description):
                if "index.sub" in original_text:
                    drifted_text = original_text.replace("index.sub", "index.add")
                elif "tail_key_value_stage_bytes" in original_text:
                    drifted_text = original_text.replace(
                        "%tail_key_value_stage_bytes",
                        "%tail_key_value_stage_capacity",
                    )
                else:
                    drifted_text = original_text.replace("%two", "%three")
                drifted_source = _COMPLETE_SOURCE.replace(original_text, drifted_text)
                if "index.sub" in original_text:
                    drifted_source += "  %replacement = index.sub %a, %b : index\n"
                with self.assertRaisesRegex(ValueError, f"{description}, found 0"):
                    flash_attention_bringup_workaround.apply_flash_attention_bringup_workaround(
                        drifted_source
                    )


if __name__ == "__main__":
    unittest.main()
