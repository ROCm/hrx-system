# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the Loom programming guide site contract."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_STYLESHEET_PATH = Path(__file__).resolve().parent / "src/stylesheets/extra.css"
_COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}")


def _selector_variables(stylesheet: str, selector: str) -> dict[str, str]:
    match = re.search(rf"{re.escape(selector)}\s*\{{([^}}]+)\}}", stylesheet)
    assert match is not None, selector
    return {
        name: value
        for name, value in re.findall(
            rf"(--[a-z0-9-]+):\s*({_COLOR_PATTERN.pattern})\s*;",
            match.group(1),
        )
    }


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear_channels = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return (
        0.2126 * linear_channels[0]
        + 0.7152 * linear_channels[1]
        + 0.0722 * linear_channels[2]
    )


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


class SiteTest(unittest.TestCase):
    def test_body_links_are_readable_in_both_color_schemes(self) -> None:
        stylesheet = _STYLESHEET_PATH.read_text(encoding="utf-8")

        for selector in (":root", '[data-md-color-scheme="slate"]'):
            with self.subTest(selector=selector):
                variables = _selector_variables(stylesheet, selector)
                self.assertGreaterEqual(
                    _contrast_ratio(
                        variables["--md-typeset-a-color"],
                        variables["--md-default-bg-color"],
                    ),
                    4.5,
                )


if __name__ == "__main__":
    unittest.main()
