# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lazy TileLang fixture harness."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from loom.importers.tilelang.model import TileLangImportInput


class TileLangHarnessError(RuntimeError):
    """Raised when the TileLang fixture harness cannot be prepared."""


@dataclass(frozen=True, slots=True)
class TileLangModules:
    """Heavy TileLang/TVM modules loaded for one fixture run."""

    tilelang: ModuleType
    tvm: ModuleType
    tir: ModuleType
    T: ModuleType


class TileLangHarness:
    """Provides lazily imported TileLang objects to Python check fixtures."""

    def __init__(self) -> None:
        self.modules = self._import_modules()

    @property
    def T(self) -> ModuleType:
        return self.modules.T

    @property
    def tilelang(self) -> ModuleType:
        return self.modules.tilelang

    @property
    def tvm(self) -> ModuleType:
        return self.modules.tvm

    @property
    def tir(self) -> ModuleType:
        return self.modules.tir

    def input(
        self,
        source: Any,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        target: str | None = None,
        name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TileLangImportInput:
        return TileLangImportInput(
            source=source,
            args=args,
            kwargs={} if kwargs is None else kwargs,
            target=target,
            name=name,
            metadata={} if metadata is None else metadata,
        )

    @staticmethod
    def _import_modules() -> TileLangModules:
        try:
            tilelang = importlib.import_module("tilelang")
            tvm = importlib.import_module("tvm")
            tir = importlib.import_module("tvm.tir")
            language = importlib.import_module("tilelang.language")
        except Exception as exc:
            raise TileLangHarnessError(
                f"failed to import TileLang fixture dependencies: {exc}"
            ) from exc
        return TileLangModules(
            tilelang=tilelang,
            tvm=tvm,
            tir=tir,
            T=language,
        )
