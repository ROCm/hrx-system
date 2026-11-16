# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Admission invariants for typed scalar projections."""

from dataclasses import replace

import pytest

from loom.dialect.scalar.comparison import scalar_isfinitef, scalar_isinff, scalar_isnanf
from loom.dialect.scalar.math import scalar_frexpf, scalar_ldexpf
from loom.dsl import I1, Operand, SameType
from loom.format.cxx.expressions import SCALAR_PROJECTION_OPERATIONS
from loom.gen.cxx.intrinsics import scalar_projections, validate_scalar_projection


def test_exponent_family_has_one_floating_type_and_fixed_exponents() -> None:
    projections = scalar_projections()
    assert scalar_frexpf in projections
    assert scalar_ldexpf in projections


def test_classification_family_has_fixed_boolean_results() -> None:
    projections = scalar_projections()
    assert scalar_isfinitef in projections
    assert scalar_isinff in projections
    assert scalar_isnanf in projections


def test_renderer_covers_every_generated_scalar_projection() -> None:
    assert SCALAR_PROJECTION_OPERATIONS == {operation.name for operation in scalar_projections()}


def test_fixed_fields_cannot_join_the_variable_type_relationship() -> None:
    invalid = replace(scalar_ldexpf, constraints=[SameType("input", "exponent", "result")])
    with pytest.raises(ValueError, match="one scalar type relationship"):
        validate_scalar_projection(invalid)


def test_fixed_boolean_operands_are_rejected_at_generation() -> None:
    invalid = replace(scalar_ldexpf, operands=[scalar_ldexpf.operands[0], Operand("exponent", I1)])
    with pytest.raises(ValueError, match="fixed boolean operands"):
        validate_scalar_projection(invalid)
