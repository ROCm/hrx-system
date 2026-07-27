# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: op declarations to dynamic builder `.pyi` stubs."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.builder_model import (
    BuilderParam,
    BuilderParamKind,
    BuilderSignature,
    python_name,
    signatures_for_ops,
)
from loom.builders import default_ops
from loom.dsl import Op
from loom.gen import bootstrap as _bootstrap
from loom.gen.support.generated_file import line_comment_header

type CategoryGroups = Mapping[str, Sequence[tuple[Any, Sequence[Op]]]]


@dataclass(frozen=True, slots=True)
class _ShardStub:
    module_name: str
    class_name: str
    signatures: tuple[BuilderSignature, ...]


@dataclass(frozen=True, slots=True)
class _DialectStub:
    attr_name: str
    ir_name: str
    package_name: str
    class_name: str
    shards: tuple[_ShardStub, ...]


def generate_builders_pyi(ops: Sequence[Op]) -> str:
    """Generate the root public stub for dynamic Loom builders."""
    return generate_builder_stub_files(ops)["loom/py/loom/builders.pyi"]


def generate_builder_stub_files(
    ops: Sequence[Op],
    *,
    category_groups: CategoryGroups | None = None,
) -> dict[str, str]:
    """Generate every dynamic builder stub file, keyed by repo-relative path."""
    dialects = _dialect_stubs(ops, category_groups=category_groups or {})
    files: dict[str, str] = {
        "loom/py/loom/builders.pyi": _generate_root_builders_pyi(dialects),
    }
    for dialect in dialects:
        dialect_root = f"loom/py/loom/dialect/{dialect.package_name}/builders"
        files[f"{dialect_root}/__init__.pyi"] = _generate_dialect_init_pyi(dialect)
        if len(dialect.shards) > 1:
            for shard in dialect.shards:
                files[f"{dialect_root}/{shard.module_name}.pyi"] = _generate_shard_pyi(shard)
    return files


def _generate_root_builders_pyi(dialects: Sequence[_DialectStub]) -> str:
    lines: list[str] = []
    lines.extend(_header())
    lines.extend(
        [
            "",
            "from __future__ import annotations",
            "",
            "from collections.abc import Sequence",
            "from typing import Any",
            "",
            "from loom.builder import IRBuilder, ValueRef",
        ]
    )
    lines.extend(f"from loom.dialect.{dialect.package_name}.builders import {dialect.class_name}" for dialect in dialects)
    lines.extend(
        [
            "from loom.dsl import Op, TypeDef",
            "from loom.ir import Block, Module, Region, Type",
            "",
            "def module_builder(",
            "    *,",
            "    module: Module | None = ...,",
            "    insertion_block: Block | None = ...,",
            "    ops: Sequence[Op] | None = ...,",
            "    types: Sequence[TypeDef] | None = ...,",
            ") -> tuple[Module, LoomBuilder]: ...",
            "def default_ops() -> tuple[Op, ...]: ...",
            "def default_types() -> tuple[TypeDef, ...]: ...",
            "",
            "class LoomBuilder:",
            "    @property",
            "    def module(self) -> Module: ...",
            "    @property",
            "    def ir(self) -> IRBuilder: ...",
            "    def value(self, name: str, value_type: Type, **kwargs: Any) -> ValueRef: ...",
            "    def region(self, args: Sequence[tuple[str, Type]] = ...) -> Region: ...",
            "    def insertion_block(self, block: Block | None) -> Any: ...",
            "    def location(self, location_id: int) -> Any: ...",
        ]
    )
    for dialect in dialects:
        lines.extend(
            [
                "    @property",
                f"    def {dialect.attr_name}(self) -> {dialect.class_name}: ...",
            ]
        )
    lines.extend(
        [
            "",
            "class DialectBuilder:",
            "    @property",
            "    def name(self) -> str: ...",
            "",
            "class OpCallable:",
            "    @property",
            "    def op_name(self) -> str: ...",
            "    def __call__(self, **kwargs: Any) -> ValueRef | list[ValueRef] | None: ...",
        ]
    )
    return "\n".join(lines) + "\n"


def _generate_dialect_init_pyi(dialect: _DialectStub) -> str:
    lines: list[str] = []
    lines.extend(_header())
    lines.extend(["", "from __future__ import annotations", ""])
    if len(dialect.shards) == 1:
        lines.extend(
            _method_imports(
                dialect.shards[0].signatures,
                extra_loom_imports=("from loom.builders import DialectBuilder",),
            )
        )
        lines.extend(["", f"class {dialect.class_name}(DialectBuilder):"])
        lines.extend(_class_body(dialect.shards[0].signatures))
        return "\n".join(lines) + "\n"

    lines.append("from loom.builders import DialectBuilder")
    lines.extend(sorted(f"from loom.dialect.{dialect.package_name}.builders.{shard.module_name} import {shard.class_name}" for shard in dialect.shards))
    bases = [shard.class_name for shard in dialect.shards] + ["DialectBuilder"]
    lines.extend(["", f"class {dialect.class_name}("])
    lines.extend(f"    {base}," for base in bases)
    lines.append("): ...")
    return "\n".join(lines) + "\n"


def _generate_shard_pyi(shard: _ShardStub) -> str:
    lines: list[str] = []
    lines.extend(_header())
    lines.extend(["", "from __future__ import annotations", ""])
    lines.extend(_method_imports(shard.signatures))
    lines.extend(["", f"class {shard.class_name}:"])
    lines.extend(_class_body(shard.signatures))
    return "\n".join(lines) + "\n"


def _class_body(signatures: Sequence[BuilderSignature]) -> list[str]:
    if not signatures:
        return ["    pass"]
    lines: list[str] = []
    for signature in signatures:
        lines.extend(_method_stub(signature))
    return lines


def _method_stub(signature: BuilderSignature) -> list[str]:
    params = _method_params(signature)
    lines = [f"    def {signature.method_name}("]
    lines.extend(f"        {param}," for param in params)
    lines.append(f"    ) -> {signature.return_hint}: ...")
    return lines


def _method_params(signature: BuilderSignature) -> list[str]:
    keyword_params = [_format_param(param) for param in signature.params]
    params = ["self"]
    params.append("*")
    params.extend(keyword_params)
    if signature.op.results:
        params.extend(
            [
                "name: str | None = ...",
                "names: Sequence[str] | None = ...",
                "result_names: Sequence[str] | None = ...",
            ]
        )
    params.append("location_id: int | None = ...")
    return params


def _format_param(param: BuilderParam) -> str:
    type_hint = _public_type_hint(param)
    if param.required:
        return f"{param.py_name}: {type_hint}"
    return f"{param.py_name}: {type_hint} = ..."


def _public_type_hint(param: BuilderParam) -> str:
    if param.kind == BuilderParamKind.ATTR and not param.required:
        return f"{param.type_hint} | None"
    if param.kind == BuilderParamKind.REGION:
        return "Region | None"
    return param.type_hint


def _dialect_stubs(
    ops: Sequence[Op],
    *,
    category_groups: CategoryGroups,
) -> list[_DialectStub]:
    grouped_ops: dict[str, list[Op]] = {}
    for op in ops:
        dialect_name = op.name.split(".", 1)[0]
        grouped_ops.setdefault(dialect_name, []).append(op)

    dialects: list[_DialectStub] = []
    for dialect_name, dialect_ops in sorted(grouped_ops.items()):
        attr_name = python_name(dialect_name)
        package_name = _package_name(dialect_name)
        class_name = _builder_class_name(attr_name)
        all_signatures = signatures_for_ops(dialect_ops)
        signature_by_op_name = {signature.op.name: signature for signature in all_signatures.values()}
        groups = category_groups.get(dialect_name)
        if groups:
            shards = tuple(
                _ShardStub(
                    module_name=_module_name(category.key),
                    class_name=_mixin_class_name(attr_name, category.key),
                    signatures=tuple(signature_by_op_name[op.name] for op in category_ops),
                )
                for category, category_ops in groups
                if category_ops
            )
        else:
            shards = (
                _ShardStub(
                    module_name="__init__",
                    class_name=class_name,
                    signatures=tuple(all_signatures.values()),
                ),
            )
        dialects.append(
            _DialectStub(
                attr_name=attr_name,
                ir_name=dialect_name,
                package_name=package_name,
                class_name=class_name,
                shards=shards,
            )
        )
    return dialects


def _builder_class_name(*parts: str) -> str:
    return _class_name(*parts, suffix="Builder")


def _mixin_class_name(*parts: str) -> str:
    return _class_name(*parts, suffix="Mixin")


def _class_name(*parts: str, suffix: str) -> str:
    words: list[str] = []
    for part in parts:
        words.extend(piece for piece in part.rstrip("_").replace("-", "_").replace(".", "_").split("_") if piece)
    return "".join(word.capitalize() for word in words) + suffix


def _module_name(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def _package_name(dialect_name: str) -> str:
    if dialect_name == "global":
        return "globals"
    return python_name(dialect_name)


def _method_imports(
    signatures: Sequence[BuilderSignature],
    *,
    extra_loom_imports: Sequence[str] = (),
) -> list[str]:
    required_names = _method_import_names(signatures)
    lines: list[str] = []
    stdlib_lines: list[str] = []
    collection_names = sorted(required_names & {"Mapping", "Sequence"})
    if collection_names:
        stdlib_lines.append(f"from collections.abc import {', '.join(collection_names)}")
    if "Any" in required_names:
        stdlib_lines.append("from typing import Any")
    if stdlib_lines:
        lines.extend(stdlib_lines)
        lines.append("")

    loom_lines: list[str] = []
    builder_names = sorted(required_names & {"TiedResultSpec", "ValueRef"})
    if builder_names:
        loom_lines.append(f"from loom.builder import {', '.join(builder_names)}")
    ir_names = sorted(required_names & {"Block", "Predicate", "Region", "Type"})
    if ir_names:
        loom_lines.append(f"from loom.ir import {', '.join(ir_names)}")
    loom_lines.extend(extra_loom_imports)
    lines.extend(sorted(loom_lines))
    return lines


def _method_import_names(signatures: Sequence[BuilderSignature]) -> set[str]:
    hints: list[str] = []
    for signature in signatures:
        hints.append(signature.return_hint)
        hints.extend(_public_type_hint(param) for param in signature.params)
        if signature.op.results:
            hints.append("Sequence")
    return {
        name
        for name in (
            "Any",
            "Block",
            "Mapping",
            "Predicate",
            "Region",
            "Sequence",
            "TiedResultSpec",
            "Type",
            "ValueRef",
        )
        if any(re.search(rf"\b{re.escape(name)}\b", hint) for hint in hints)
    }


def _header() -> list[str]:
    return line_comment_header(
        "#",
        generator="loom.gen.python.builders_pyi",
        regenerate="python3 loom/py/loom/gen/run.py builders_pyi --in-place",
    )


def _default_category_groups() -> CategoryGroups:
    from loom.dialect.scalar import SCALAR_OP_CATEGORY_GROUPS
    from loom.dialect.vector import VECTOR_OP_CATEGORY_GROUPS

    return {
        "scalar": SCALAR_OP_CATEGORY_GROUPS,
        "vector": VECTOR_OP_CATEGORY_GROUPS,
    }


def _output_files() -> dict[str, str]:
    return generate_builder_stub_files(
        default_ops(),
        category_groups=_default_category_groups(),
    )


def _write_files(files: Mapping[str, str]) -> None:
    for generated_root in _generated_roots():
        if generated_root.exists():
            shutil.rmtree(generated_root)
    for rel_path, content in sorted(files.items()):
        path = _bootstrap.REPO_ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  {rel_path}")


def _check_files(files: Mapping[str, str]) -> int:
    expected_paths = {_bootstrap.REPO_ROOT / rel_path for rel_path in files}
    stale_or_changed: list[str] = []
    for rel_path, expected in sorted(files.items()):
        path = _bootstrap.REPO_ROOT / rel_path
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            stale_or_changed.append(rel_path)
    for generated_root in _generated_roots():
        if not generated_root.exists():
            continue
        stale_or_changed.extend(str(path.relative_to(_bootstrap.REPO_ROOT)) for path in generated_root.rglob("*.pyi") if path not in expected_paths)
    if stale_or_changed:
        print(
            "error: generated builder stubs are stale; regenerate with python3 loom/py/loom/gen/run.py builders_pyi --in-place",
            file=sys.stderr,
        )
        for rel_path in stale_or_changed:
            print(f"  {rel_path}", file=sys.stderr)
        return 1
    return 0


def _generated_roots() -> tuple[Path, ...]:
    dialect_root = _bootstrap.REPO_ROOT / "loom" / "py" / "loom" / "dialect"
    return tuple(path for path in dialect_root.glob("*/builders") if path.is_dir())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)

    files = _output_files()
    if args.in_place:
        _write_files(files)
        return 0
    return _check_files(files)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
