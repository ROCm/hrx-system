# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pipeline dialect: portable persistent dataflow programs."""

from loom.dialect.pipeline.defs import (
    ALL_PIPELINE_OPS,
    ALL_PIPELINE_TYPES,
    PipelineScope,
    pipeline_compose,
    pipeline_decl,
    pipeline_def,
    pipeline_end,
    pipeline_finish,
    pipeline_memory,
    pipeline_ops,
    pipeline_serial,
    pipeline_strand,
    pipeline_yield,
)

__all__ = [
    "pipeline_ops",
    "PipelineScope",
    "pipeline_compose",
    "pipeline_memory",
    "pipeline_def",
    "pipeline_decl",
    "pipeline_strand",
    "pipeline_end",
    "pipeline_finish",
    "pipeline_serial",
    "pipeline_yield",
    "ALL_PIPELINE_OPS",
    "ALL_PIPELINE_TYPES",
]
