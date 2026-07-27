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
    tir: TileLangTirCompat
    T: ModuleType


class TileLangTirCompat:
    """Compatibility facade for TileLang's current vendored TIR module."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        if name == "Block":
            return self._module.SBlock
        if name == "BlockRealize":
            return self._module.SBlockRealize
        return getattr(self._module, name)

    def Call(
        self,
        dtype: str,
        op: object,
        args: list[object],
        annotations: dict[object, object] | None = None,
        span: object | None = None,
    ) -> object:
        return self._module.Call(
            dtype,
            _normalize_tir_op_name(op),
            args,
            annotations=annotations,
            span=span,
        )

    def call_intrin(
        self,
        dtype: str,
        func_name: object,
        *args: object,
        annotations: dict[object, object] | None = None,
        span: object | None = None,
    ) -> object:
        return self._module.call_intrin(
            dtype,
            _normalize_tir_op_name(func_name),
            *args,
            annotations=annotations,
            span=span,
        )


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
    def tir(self) -> TileLangTirCompat:
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
            tir = importlib.import_module("tvm.tirx")
            language = importlib.import_module("tilelang.language")
            print_op = importlib.import_module("tilelang.language.print_op")
        except Exception as exc:
            raise TileLangHarnessError(
                f"failed to import TileLang fixture dependencies: {exc}"
            ) from exc
        # TileLang currently gates T.device_assert macro emission on CUDA
        # discovery at import time. Importer checks inspect frontend IR and do
        # not execute CUDA, so keep debug/assert fixture output independent of
        # whether the host test environment exposes nvcc.
        print_op._IS_CUDA_AVAILABLE = True
        return TileLangModules(
            tilelang=tilelang,
            tvm=tvm,
            tir=TileLangTirCompat(tir),
            T=language,
        )


def _normalize_tir_op_name(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value.startswith("tir."):
        return "tirx." + value[len("tir.") :]
    return value
