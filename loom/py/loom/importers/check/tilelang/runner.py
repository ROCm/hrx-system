# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs TileLang importer checks from Python fixtures."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loom.importers.check.loom_verifier import LoomOutputVerifier
from loom.importers.check.python import (
    PythonCheckCase,
    PythonCheckInvocation,
    PythonCheckOptions,
    PythonCheckSkip,
    run_python_check,
)
from loom.importers.check.results import CheckResult
from loom.importers.check.tilelang.cases import (
    get_tilelang_case_metadata,
    is_tilelang_case,
)
from loom.importers.check.tilelang.harness import TileLangHarness
from loom.importers.core import kernel_module_ops, print_loom_module
from loom.importers.tilelang.importer import TileLangImportOptions, import_tilelang
from loom.importers.tilelang.model import TileLangImportInput
from loom.importers.tilelang.oracle import (
    TileLangCodeObjectOracle,
    TileLangGeneratedSource,
    TileLangOracleError,
    TileLangOracleUnavailable,
    capture_generated_source,
    compile_hip_code_object,
    target_arch,
)

_ORACLE_OFF = "off"
_ORACLE_SOURCE = "source"
_ORACLE_CODE_OBJECT = "code-object"
TILELANG_ORACLE_MODES = (_ORACLE_OFF, _ORACLE_SOURCE, _ORACLE_CODE_OBJECT)


@dataclass(frozen=True, slots=True)
class TileLangOracleCheckOptions:
    """Options for optional TileLang oracle capture during checks."""

    mode: str = _ORACLE_OFF
    output_directory: Path | None = None


@dataclass(frozen=True, slots=True)
class TileLangCheckOptions:
    """Options for Python TileLang importer checks."""

    update: bool = False
    target_preset: str | None = None
    case_filter: str | None = None
    verify_structure: bool = True
    output_verifier: LoomOutputVerifier | None = None
    oracle: TileLangOracleCheckOptions = field(
        default_factory=TileLangOracleCheckOptions
    )


def run_tilelang_check(
    path: Path,
    *,
    options: TileLangCheckOptions,
) -> tuple[CheckResult, ...]:
    return run_python_check(
        path,
        options=PythonCheckOptions(
            update=options.update,
            case_filter=options.case_filter,
            output_verifier=options.output_verifier,
        ),
        is_case=is_tilelang_case,
        invoke=lambda case: _invoke_tilelang_case(case, options),
        case_labels=_tilelang_case_labels,
    )


def _invoke_tilelang_case(
    case: PythonCheckCase,
    options: TileLangCheckOptions,
) -> str | PythonCheckInvocation:
    value = case.function(**_case_kwargs(case.function))
    metadata = get_tilelang_case_metadata(case.function)
    if not isinstance(value, TileLangImportInput):
        value = TileLangImportInput(
            source=value,
            target=options.target_preset,
            name=metadata.name if metadata is not None else None,
        )
    target_preset = options.target_preset or value.target or "tilelang.generic"
    result = import_tilelang(
        value,
        options=TileLangImportOptions(
            target_preset=options.target_preset,
            verify_structure=options.verify_structure,
        ),
    )
    stdout = print_loom_module(result.module, ops=kernel_module_ops(target_preset))
    oracle_metadata = _capture_oracle_metadata(
        case,
        value,
        target_preset=target_preset,
        options=options.oracle,
    )
    if not oracle_metadata:
        return stdout
    return PythonCheckInvocation(
        stdout=stdout,
        metadata={"tilelang_oracle": oracle_metadata},
    )


def _case_kwargs(function: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    if not signature.parameters:
        return {}
    harness: TileLangHarness | None = None
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if harness is None:
            harness = TileLangHarness()
        if name == "h":
            kwargs[name] = harness
        elif name == "T":
            kwargs[name] = harness.T
        elif name == "tilelang":
            kwargs[name] = harness.tilelang
        elif name == "tvm":
            kwargs[name] = harness.tvm
        elif name == "tir":
            kwargs[name] = harness.tir
        else:
            raise TypeError(f"unsupported TileLang fixture parameter `{name}`")
    return kwargs


def _tilelang_case_labels(case: PythonCheckCase) -> tuple[str, ...]:
    metadata = get_tilelang_case_metadata(case.function)
    if metadata is None:
        return ()
    labels = [metadata.category, *metadata.tags]
    if metadata.name:
        labels.append(metadata.name)
    return tuple(labels)


def _capture_oracle_metadata(
    case: PythonCheckCase,
    import_input: TileLangImportInput,
    *,
    target_preset: str,
    options: TileLangOracleCheckOptions,
) -> dict[str, object]:
    if options.mode == _ORACLE_OFF:
        return {}
    target_text = target_preset if target_preset != "tilelang.generic" else None
    if target_text is None:
        raise PythonCheckSkip(
            "TileLang oracle capture requires explicit target text",
            metadata={
                "tilelang_oracle": {
                    "status": "unavailable",
                    "dependency": "explicit-target",
                    "reason": "TileLang oracle capture requires explicit target text",
                }
            },
        )
    try:
        target_arch(target_text)
    except TileLangOracleError as exc:
        raise PythonCheckSkip(
            str(exc),
            metadata={
                "tilelang_oracle": {
                    "status": "unavailable",
                    "dependency": "explicit-target",
                    "reason": str(exc),
                    "target": target_text,
                }
            },
        ) from exc
    try:
        generated_source = capture_generated_source(
            import_input,
            target_text=target_text,
        )
        if options.mode == _ORACLE_CODE_OBJECT:
            oracle = _compile_case_code_object(case, generated_source, options)
            return dict(oracle.metadata())
        return _capture_case_source(case, generated_source, options)
    except TileLangOracleUnavailable as exc:
        raise PythonCheckSkip(
            exc.reason,
            metadata={"tilelang_oracle": exc.metadata()},
        ) from exc


def _compile_case_code_object(
    case: PythonCheckCase,
    generated_source: TileLangGeneratedSource,
    options: TileLangOracleCheckOptions,
) -> TileLangCodeObjectOracle:
    if options.output_directory is None:
        raise ValueError(
            "TileLang code-object oracle capture requires --oracle-output-dir "
            "or --dump-temp-dir"
        )
    return compile_hip_code_object(
        generated_source,
        output_directory=_oracle_case_directory(case, options.output_directory),
        stem=_oracle_case_stem(case),
    )


def _capture_case_source(
    case: PythonCheckCase,
    generated_source: TileLangGeneratedSource,
    options: TileLangOracleCheckOptions,
) -> dict[str, object]:
    metadata = dict(generated_source.metadata())
    if options.output_directory is None:
        return metadata
    case_directory = _oracle_case_directory(case, options.output_directory)
    case_directory.mkdir(parents=True, exist_ok=True)
    stem = _oracle_case_stem(case)
    source_path = case_directory / f"{stem}.{generated_source.arch}.hip.cpp"
    metadata_path = case_directory / f"{stem}.{generated_source.arch}.metadata.json"
    source_path.write_text(generated_source.source, encoding="utf-8")
    metadata["source_path"] = str(source_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _oracle_case_directory(case: PythonCheckCase, output_directory: Path) -> Path:
    case_name = get_tilelang_case_metadata(case.function)
    if case_name is not None and case_name.name:
        name = case_name.name
    else:
        name = case.function.__name__
    return (
        output_directory
        / _safe_fragment(case.check_case.path.stem)
        / (f"case{case.check_case.index}_{_safe_fragment(name)}")
    )


def _oracle_case_stem(case: PythonCheckCase) -> str:
    metadata = get_tilelang_case_metadata(case.function)
    if metadata is not None and metadata.name:
        return _safe_fragment(metadata.name)
    return _safe_fragment(case.function.__name__)


def _safe_fragment(text: str) -> str:
    return (
        "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text
        ).strip("_")
        or "case"
    )
