# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs simple pytest-style tests through Bazel without depending on pytest.

This runner is intentionally tiny. It supports the pytest surface used by Loom's
unit tests: module-level test functions, class-based test methods,
pytest.raises, pytest.mark.parametrize, tmp_path, and capsys. Unsupported
fixtures or parameters fail during discovery instead of silently skipping tests.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import io
import itertools
import re
import shutil
import sys
import tempfile
import traceback
import unittest
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


class TestRunnerError(ValueError):
    """Raised when a test module uses unsupported runner features."""


@dataclass(frozen=True)
class ParametrizeSpec:
    names: tuple[str, ...]
    values: tuple[Any, ...]
    ids: Any = None


@dataclass(frozen=True)
class TestCase:
    module_name: str
    test_name: str
    function: Callable[..., object]
    parameters: dict[str, object]


@dataclass
class _CapturedOutput:
    out: str
    err: str


class _CapsysFixture:
    def __init__(self):
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()

    @property
    def stdout(self) -> io.StringIO:
        return self._stdout

    @property
    def stderr(self) -> io.StringIO:
        return self._stderr

    def readouterr(self) -> _CapturedOutput:
        captured = _CapturedOutput(
            out=self._stdout.getvalue(),
            err=self._stderr.getvalue(),
        )
        self._stdout.seek(0)
        self._stdout.truncate(0)
        self._stderr.seek(0)
        self._stderr.truncate(0)
        return captured


class _RaisesContext:
    def __init__(
        self,
        expected_exception: type[BaseException] | tuple[type[BaseException], ...],
        *,
        match: str | None = None,
    ):
        self.expected_exception = expected_exception
        self.match = match
        self.value: BaseException | None = None

    def __enter__(self) -> _RaisesContext:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del traceback
        if exc_type is None or exc is None:
            raise AssertionError(f"expected {self._expected_name()} to be raised")
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match is not None and re.search(self.match, str(exc)) is None:
            raise AssertionError(
                f"exception message {str(exc)!r} did not match {self.match!r}"
            )
        self.value = exc
        return True

    def _expected_name(self) -> str:
        if isinstance(self.expected_exception, tuple):
            return " or ".join(
                exception.__name__ for exception in self.expected_exception
            )
        return self.expected_exception.__name__


class _Mark:
    def parametrize(
        self,
        argnames: str | tuple[str, ...] | list[str],
        argvalues: Any,
        *,
        ids: Any = None,
    ):
        names = _parse_argnames(argnames)
        values = tuple(argvalues)

        def decorator(function):
            specs = list(getattr(function, "_loom_parametrize_specs", ()))
            specs.append(ParametrizeSpec(names=names, values=values, ids=ids))
            function._loom_parametrize_specs = tuple(specs)
            return function

        return decorator


class _CaptureFixtureType:
    def __class_getitem__(cls, item):
        del item
        return cls


def _raises(
    expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    *,
    match: str | None = None,
) -> _RaisesContext:
    return _RaisesContext(expected_exception, match=match)


def _parse_argnames(argnames: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(argnames, str):
        return tuple(name.strip() for name in argnames.split(",") if name.strip())
    return tuple(argnames)


def _install_pytest_compat() -> None:
    pytest_module = ModuleType("pytest")
    pytest_module.raises = _raises  # type: ignore[attr-defined]
    pytest_module.mark = _Mark()  # type: ignore[attr-defined]
    pytest_module.CaptureFixture = _CaptureFixtureType  # type: ignore[attr-defined]
    sys.modules["pytest"] = pytest_module


_install_pytest_compat()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        module_names = _parse_runner_args(tuple(sys.argv[1:] if argv is None else argv))
    except TestRunnerError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    if not module_names:
        sys.stderr.write("error: at least one test module is required\n")
        return 2

    failures = 0
    skips = 0
    test_count = 0
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            for test_case in _iter_test_cases(module):
                test_count += 1
                try:
                    _run_test_case(test_case)
                except unittest.SkipTest as exc:
                    skips += 1
                    sys.stdout.write(
                        f"SKIP {test_case.module_name}.{test_case.test_name}: {exc}\n"
                    )
                except Exception:
                    failures += 1
                    sys.stderr.write(
                        f"\nFAILED {test_case.module_name}.{test_case.test_name}\n"
                    )
                    traceback.print_exc(file=sys.stderr)
        except Exception:
            failures += 1
            sys.stderr.write(f"\nFAILED loading {module_name}\n")
            traceback.print_exc(file=sys.stderr)

    sys.stdout.write(
        "loom pytest-style runner: "
        f"{test_count} tests, {failures} failures, {skips} skipped\n"
    )
    if test_count == 0:
        sys.stderr.write("error: no tests were discovered\n")
        return 2
    return 1 if failures else 0


def _iter_test_cases(module: ModuleType) -> Iterable[TestCase]:
    for name, value in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if value.__module__ != module.__name__:
            continue
        yield from _expand_test_callable(module.__name__, name, value)
    for class_name, class_value in inspect.getmembers(module, inspect.isclass):
        if not class_name.startswith("Test"):
            continue
        if class_value.__module__ != module.__name__:
            continue
        for method_name, method in inspect.getmembers(class_value, inspect.isfunction):
            if not method_name.startswith("test_"):
                continue
            bound_method = _bind_test_method(class_value, method)
            yield from _expand_test_callable(
                module.__name__,
                f"{class_name}.{method_name}",
                bound_method,
            )


def _bind_test_method(
    test_class: type[object],
    method: Callable[..., object],
) -> Callable[..., object]:
    @functools.wraps(method)
    def bound_method(**kwargs):
        return method(test_class(), **kwargs)

    parameters = tuple(inspect.signature(method).parameters.values())
    bound_method.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters[1:]
    )
    bound_method._loom_parametrize_specs = tuple(  # type: ignore[attr-defined]
        getattr(method, "_loom_parametrize_specs", ())
    )
    return bound_method


def _expand_test_callable(
    module_name: str,
    test_name: str,
    function: Callable[..., object],
) -> Iterable[TestCase]:
    signature = inspect.signature(function)
    specs = tuple(getattr(function, "_loom_parametrize_specs", ()))
    for parameter_names in _parameter_name_sets(specs):
        for name in parameter_names:
            if name not in signature.parameters:
                raise TestRunnerError(
                    f"{module_name}.{test_name} parametrizes unknown argument {name!r}"
                )
    fixture_names = [
        name
        for name in signature.parameters
        if name not in _parameter_name_union(specs)
    ]
    for fixture_name in fixture_names:
        if fixture_name not in ("tmp_path", "capsys"):
            raise TestRunnerError(
                f"{module_name}.{test_name} requests unsupported fixture "
                f"{fixture_name!r}"
            )
    for index, parameters in enumerate(_expand_parameters(specs)):
        suffix = _parameter_suffix(parameters, index)
        yield TestCase(
            module_name=module_name,
            test_name=f"{test_name}{suffix}",
            function=function,
            parameters=parameters,
        )


def _parameter_name_sets(
    specs: Sequence[ParametrizeSpec],
) -> Iterable[tuple[str, ...]]:
    for spec in specs:
        yield spec.names


def _parameter_name_union(
    specs: Sequence[ParametrizeSpec],
) -> set[str]:
    names: set[str] = set()
    for spec in specs:
        names.update(spec.names)
    return names


def _expand_parameters(
    specs: Sequence[ParametrizeSpec],
) -> Iterable[dict[str, object]]:
    if not specs:
        yield {}
        return
    expanded_specs = [
        [_parameter_mapping(spec, value) for value in spec.values] for spec in specs
    ]
    for combination in itertools.product(*expanded_specs):
        parameters: dict[str, object] = {}
        for mapping in combination:
            duplicate_names = parameters.keys() & mapping.keys()
            if duplicate_names:
                duplicate_list = ", ".join(sorted(duplicate_names))
                raise TestRunnerError(
                    f"duplicate parametrized arguments: {duplicate_list}"
                )
            parameters.update(mapping)
        yield parameters


def _parameter_mapping(
    spec: ParametrizeSpec,
    value: object,
) -> dict[str, object]:
    if len(spec.names) == 1:
        return {spec.names[0]: value}
    if not isinstance(value, tuple):
        raise TestRunnerError(
            f"parametrize for {spec.names!r} expects tuple values, got {value!r}"
        )
    if len(value) != len(spec.names):
        raise TestRunnerError(
            f"parametrize for {spec.names!r} expected {len(spec.names)} values, "
            f"got {len(value)}"
        )
    return dict(zip(spec.names, value, strict=True))


def _parameter_suffix(parameters: dict[str, object], index: int) -> str:
    if not parameters:
        return ""
    parts = [f"{name}={_short_value(value)}" for name, value in parameters.items()]
    return f"[{index}:{','.join(parts)}]"


def _short_value(value: object) -> str:
    text = repr(value)
    if len(text) > 48:
        return text[:45] + "..."
    return text


def _run_test_case(test_case: TestCase) -> None:
    sys.stdout.write(f"RUN {test_case.module_name}.{test_case.test_name}\n")
    kwargs = dict(test_case.parameters)
    temporary_directory: str | None = None
    capsys: _CapsysFixture | None = None
    try:
        signature = inspect.signature(test_case.function)
        if "tmp_path" in signature.parameters:
            temporary_directory = tempfile.mkdtemp(prefix="loom-pytest-")
            kwargs["tmp_path"] = Path(temporary_directory)
        if "capsys" in signature.parameters:
            capsys = _CapsysFixture()
            kwargs["capsys"] = capsys
            with (
                contextlib.redirect_stdout(capsys.stdout),
                contextlib.redirect_stderr(capsys.stderr),
            ):
                test_case.function(**kwargs)
        else:
            test_case.function(**kwargs)
    finally:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory)


def _parse_runner_args(args: Sequence[str]) -> tuple[str, ...]:
    module_names: list[str] = []
    for arg in args:
        if arg == "--update":
            continue
        if arg.startswith("--"):
            raise TestRunnerError(f"unsupported runner flag {arg!r}")
        module_names.append(arg)
    return tuple(module_names)


if __name__ == "__main__":
    raise SystemExit(main())
