# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable model descriptions and explicit program/asset selections."""

from .manifest import ManifestError, ModelCard, ModelFile, Selection

__all__ = ["ManifestError", "ModelCard", "ModelFile", "Selection"]
