#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Proves the checked-in wire projection exactly matches its authority."""

from __future__ import annotations

import pathlib
import unittest

from generate import write_or_check_wire


class GenerationTest(unittest.TestCase):
    def test_checked_in_outputs_are_current(self) -> None:
        bytecode_directory = pathlib.Path(__file__).resolve().parents[1]
        self.assertTrue(
            write_or_check_wire(
                bytecode_directory,
                check_only=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
