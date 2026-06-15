# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.python.builders_pyi.
# Regenerate: python3 loom/py/loom/gen/run.py builders_pyi --in-place

from __future__ import annotations

from loom.builders import DialectBuilder
from loom.dialect.vector.builders.aggregate import VectorAggregateMixin
from loom.dialect.vector.builders.atomic import VectorAtomicMixin
from loom.dialect.vector.builders.bitpack import VectorBitpackMixin
from loom.dialect.vector.builders.cast import VectorCastMixin
from loom.dialect.vector.builders.compare import VectorCompareMixin
from loom.dialect.vector.builders.construction import VectorConstructionMixin
from loom.dialect.vector.builders.contraction import VectorContractionMixin
from loom.dialect.vector.builders.encoding import VectorEncodingMixin
from loom.dialect.vector.builders.float_arithmetic import VectorFloatArithmeticMixin
from loom.dialect.vector.builders.integer_arithmetic import VectorIntegerArithmeticMixin
from loom.dialect.vector.builders.math import VectorMathMixin
from loom.dialect.vector.builders.memory import VectorMemoryMixin
from loom.dialect.vector.builders.reduction import VectorReductionMixin
from loom.dialect.vector.builders.table import VectorTableMixin

class VectorBuilder(
    VectorConstructionMixin,
    VectorAggregateMixin,
    VectorTableMixin,
    VectorMemoryMixin,
    VectorAtomicMixin,
    VectorCompareMixin,
    VectorFloatArithmeticMixin,
    VectorIntegerArithmeticMixin,
    VectorMathMixin,
    VectorCastMixin,
    VectorBitpackMixin,
    VectorContractionMixin,
    VectorReductionMixin,
    VectorEncodingMixin,
    DialectBuilder,
): ...
