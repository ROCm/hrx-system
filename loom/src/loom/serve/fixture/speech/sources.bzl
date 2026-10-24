# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source closure for the incremental waveform serving fixture."""

SOURCE_FILES = ["control.loom", "generate.loom", "model.json", "prepare.loom", "voice_card.json", "voice_service.loom", "voice_generate.loom"]
VARIANT_FILES = ["variants/identity.loom", "variants/policy.loom", "variants/prepare.loom", "variants/tokenizer.json", "variants/weights.loom"]
SOURCES = ["//loom/src/loom/serve/fixture/speech:" + path for path in SOURCE_FILES]
VARIANT_SOURCES = ["//loom/src/loom/serve/fixture/speech:" + path for path in VARIANT_FILES]
