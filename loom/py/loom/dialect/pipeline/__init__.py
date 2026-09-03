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
    pipeline_buffer,
    pipeline_def,
    pipeline_flow_type,
    pipeline_ops,
    pipeline_read,
    pipeline_reduce,
    pipeline_return,
    pipeline_scatter,
    pipeline_stage,
    pipeline_write,
)

__all__ = [
    "pipeline_ops",
    "pipeline_flow_type",
    "PipelineScope",
    "pipeline_def",
    "pipeline_scatter",
    "pipeline_read",
    "pipeline_stage",
    "pipeline_buffer",
    "pipeline_reduce",
    "pipeline_write",
    "pipeline_return",
    "ALL_PIPELINE_OPS",
    "ALL_PIPELINE_TYPES",
]
