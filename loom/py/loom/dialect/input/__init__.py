# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Managed streaming tokenizer state for source-defined input processing."""

from loom.dsl import ReferenceTypeKey, TypeDef

input_encoder_type = TypeDef(
    name="input.encoder",
    doc="Independently owned incremental tokenizer encoder state.",
    reference=ReferenceTypeKey("input", "encoder"),
)

input_decoder_type = TypeDef(
    name="input.decoder",
    doc="Independently owned incremental tokenizer decoder state.",
    reference=ReferenceTypeKey("input", "decoder"),
)

ALL_INPUT_TYPES: tuple[TypeDef, ...] = (input_encoder_type, input_decoder_type)

__all__ = ["input_encoder_type", "input_decoder_type", "ALL_INPUT_TYPES"]
