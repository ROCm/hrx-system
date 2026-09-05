# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P configured vector data-path control words."""


def vector_data_path_control(
    *,
    sign_x: bool,
    sign_y: bool,
    accumulator_mode: int,
    multiplication_mode: int,
    compute_mode: int,
) -> int:
    """Encodes the shared AIE2P vector multiply/accumulate control word."""

    if accumulator_mode < 0 or accumulator_mode > 0b11:
        raise ValueError("AIE2P accumulator mode must fit two bits")
    if multiplication_mode < 0 or multiplication_mode > 0b11:
        raise ValueError("AIE2P multiplication mode must fit two bits")
    if compute_mode < 0 or compute_mode > 0b111:
        raise ValueError("AIE2P compute mode must fit three bits")
    return (
        int(sign_x) << 9
        | int(sign_y) << 8
        | accumulator_mode << 1
        | multiplication_mode << 3
        | compute_mode << 5
    )
