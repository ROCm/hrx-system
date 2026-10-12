# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Protected, bounded pure-function oracle for the local pi coding experiment."""

import argparse
import ast
import builtins
import json
import math
import sys
from pathlib import Path


def equal(actual, expected):
    if isinstance(expected, list):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(equal(a, b) for a, b in zip(actual, expected))
        )
    return type(actual) in (int, float) and math.isclose(
        actual, expected, rel_tol=1e-6, abs_tol=1e-7
    )


def evaluate(task_id, candidate):
    task = next(
        item
        for item in json.loads((Path(__file__).parent / "tasks.json").read_text())
        if item["id"] == task_id
    )
    source = Path(candidate).read_text()
    if len(source) > 32768:
        raise ValueError("candidate exceeds the 32-KiB source budget")
    tree = ast.parse(source)
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].name != "solve"
    ):
        raise ValueError(
            "candidate must contain exactly one function named solve; no imports or top-level execution"
        )
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.With,
        ast.Global,
        ast.Nonlocal,
        ast.Try,
        ast.Raise,
        ast.Lambda,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
    )
    functions = 0
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"unsupported pure-function syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef):
            functions += 1
            if functions > 1 or node.decorator_list:
                raise ValueError(
                    "nested functions and decorators are not part of this workload"
                )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("private names are not part of this workload")
        if isinstance(node, ast.Attribute):
            if node.attr == "append":
                continue
            if (
                not isinstance(node.value, ast.Name)
                or node.value.id != "math"
                or node.attr not in ("exp", "sqrt", "isfinite", "fabs")
            ):
                raise ValueError(
                    "only list.append and selected math operations are available"
                )
    namespace = {
        "__builtins__": {
            name: getattr(builtins, name)
            for name in (
                "abs",
                "all",
                "any",
                "bool",
                "dict",
                "enumerate",
                "float",
                "int",
                "len",
                "list",
                "max",
                "min",
                "range",
                "reversed",
                "sorted",
                "sum",
                "tuple",
                "zip",
            )
        },
        "math": math,
    }
    exec(compile(tree, str(candidate), "exec"), namespace)
    checks = [*task["cases"], [task["followup"]["args"], task["followup"]["expected"]]]
    outcomes = []
    for args, expected in checks:
        try:
            actual = namespace["solve"](*args)
            outcomes.append(
                {
                    "args": args,
                    "expected": expected,
                    "actual": actual,
                    "passed": equal(actual, expected),
                }
            )
        except Exception as error:
            outcomes.append(
                {
                    "args": args,
                    "expected": expected,
                    "error": f"{type(error).__name__}: {error}",
                    "passed": False,
                }
            )
    return {
        "task": task_id,
        "passed": all(item["passed"] for item in outcomes),
        "cases": outcomes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("candidate", type=Path)
    options = parser.parse_args(argv)
    try:
        import resource
    except ImportError:
        parser.error("the bounded checker requires POSIX resource limits")
    # These limits bound erroneous candidate code, never inference or HAL work.
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    try:
        result = evaluate(options.task, options.candidate)
    except Exception as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
