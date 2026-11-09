# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source expression availability at the complete-value binding boundary."""

import pytest

from loom.format.cxx.context import CxxRenderContext
from loom.format.cxx.errors import CxxRenderError
from loom.ir import (
    ENCODING_LAYOUT_TYPE,
    F32,
    INDEX,
    DynamicDim,
    DynamicEncoding,
    Module,
    ShapedType,
    TypeKind,
    Value,
)


@pytest.mark.parametrize("component", ["dimension", "layout"])
def test_hidden_dependency_requires_an_independent_expression(component: str) -> None:
    module = Module()
    dimension = module.add_value(Value("dimension", INDEX))
    layout = module.add_value(Value("layout", ENCODING_LAYOUT_TYPE))
    view = module.add_value(
        Value(
            "input",
            ShapedType(
                TypeKind.VIEW, F32, (DynamicDim(dimension),), DynamicEncoding(layout)
            ),
        )
    )
    context = CxxRenderContext(module)
    dependency = dimension if component == "dimension" else layout
    context.hidden_source_dependencies[dependency] = view
    with pytest.raises(
        CxxRenderError,
        match=f"dependency %{component} carried by %input has no independent C\\+\\+ expression",
    ):
        context.expression(dependency)

    # The same SSA identity can also have an independently bound source use.
    context.expressions[dependency] = "independent"
    assert context.expression(dependency) == "independent"
