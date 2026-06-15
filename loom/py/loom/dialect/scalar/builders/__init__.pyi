# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from loom.builders import DialectBuilder
from loom.dialect.scalar.builders.analysis import ScalarAnalysisMixin
from loom.dialect.scalar.builders.arithmetic import ScalarArithmeticMixin
from loom.dialect.scalar.builders.bitwise import ScalarBitwiseMixin
from loom.dialect.scalar.builders.comparison import ScalarComparisonMixin
from loom.dialect.scalar.builders.conversion import ScalarConversionMixin
from loom.dialect.scalar.builders.math import ScalarMathMixin

class ScalarBuilder(
    ScalarArithmeticMixin,
    ScalarMathMixin,
    ScalarComparisonMixin,
    ScalarConversionMixin,
    ScalarBitwiseMixin,
    ScalarAnalysisMixin,
    DialectBuilder,
): ...
