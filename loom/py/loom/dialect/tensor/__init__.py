"""Tensor dialect: logical shaped values."""

from loom.builtin_types import tensor_type
from loom.dialect.tensor.defs import (
    ALL_TENSOR_OPS,
    ALL_TENSOR_TYPES,
    tensor_duplicate,
    tensor_ops,
)

__all__ = [
    "tensor_ops",
    "tensor_type",
    "tensor_duplicate",
    "ALL_TENSOR_OPS",
    "ALL_TENSOR_TYPES",
]
