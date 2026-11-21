# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Access-qualified C++ view types reconstructed from Loom value flow."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from loom.builtin_types import ChannelReadMode
from loom.dialect.func.defs import ArgumentAccess
from loom.dsl import EffectKind
from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.registry import source_ops
from loom.format.cxx.value_transport import lookup_columns
from loom.ir import Module, Operation, ParameterizedType, ShapedType, TypeKind

_CHANNEL_READ_MUTABLE_MODE = ChannelReadMode.case("mutable").value
_READ_ACCESS = ArgumentAccess.case("read").value
_WRITE_ACCESS = ArgumentAccess.case("write").value
_READ_WRITE_ACCESS = ArgumentAccess.case("read_write").value
_NONE_ACCESS = ArgumentAccess.case("none").value


def _walk(
    block: Any, callable_operation: Operation | None = None
) -> Iterable[tuple[Operation, Operation | None]]:
    for operation in block.ops:
        owner = operation if operation.name == "func.def" else callable_operation
        yield operation, owner
        for region in operation.regions:
            for child in region.blocks:
                yield from _walk(child, owner)


class ViewAccessPlan:
    """One access qualifier for each source-connected family of view values."""

    def __init__(self, module: Module):
        self._module = module
        self._parent = {
            value_id: value_id
            for value_id, value in enumerate(module.values)
            if isinstance(value.type, ShapedType)
            and value.type.type_kind == TypeKind.VIEW
        }
        self._mutable_sources: dict[int, Operation] = {}
        self._immutable_sources: dict[int, Operation] = {}
        self._mutable_dependencies: list[tuple[int, int, Operation]] = []
        self._mutable_roots: set[int] | None = None

    def _find(self, value_id: int) -> int:
        parent = self._parent[value_id]
        if parent != value_id:
            parent = self._find(parent)
            self._parent[value_id] = parent
        return parent

    def connect(self, *value_ids: int) -> None:
        views = [value_id for value_id in value_ids if value_id in self._parent]
        if len(views) < 2:
            return
        root = self._find(views[0])
        for value_id in views[1:]:
            other = self._find(value_id)
            if other != root:
                self._parent[other] = root

    def require_mutable(self, value_id: int, operation: Operation) -> None:
        if value_id in self._parent:
            self._mutable_sources.setdefault(value_id, operation)

    def require_immutable(self, value_id: int, operation: Operation) -> None:
        if value_id in self._parent:
            self._immutable_sources.setdefault(value_id, operation)

    def require_mutability_from(
        self, source_id: int, target_id: int, operation: Operation
    ) -> None:
        """Makes target mutable whenever source requires mutable access."""

        if source_id in self._parent and target_id in self._parent:
            self._mutable_dependencies.append((source_id, target_id, operation))

    def finish(self) -> None:
        mutable_roots: dict[int, tuple[int, Operation]] = {}
        immutable_roots: dict[int, tuple[int, Operation]] = {}
        for value_id, operation in self._mutable_sources.items():
            mutable_roots.setdefault(self._find(value_id), (value_id, operation))
        dependencies: dict[int, list[tuple[int, int, Operation]]] = {}
        for source_id, target_id, operation in self._mutable_dependencies:
            source_root = self._find(source_id)
            target_root = self._find(target_id)
            dependencies.setdefault(source_root, []).append(
                (target_root, target_id, operation)
            )
        pending = list(mutable_roots)
        while pending:
            source_root = pending.pop()
            for target_root, target_id, operation in dependencies.get(source_root, ()):
                if target_root in mutable_roots:
                    continue
                mutable_roots[target_root] = (target_id, operation)
                pending.append(target_root)
        for value_id, operation in self._immutable_sources.items():
            immutable_roots.setdefault(self._find(value_id), (value_id, operation))
        conflicts = mutable_roots.keys() & immutable_roots.keys()
        if conflicts:
            root = next(iter(conflicts))
            mutable_value, mutable_operation = mutable_roots[root]
            immutable_value, immutable_operation = immutable_roots[root]
            raise CxxRenderError(
                mutable_operation,
                "mutable access through "
                f"%{self._module.values[mutable_value].name} aliases immutable "
                f"%{self._module.values[immutable_value].name} produced by "
                f"{immutable_operation.name}",
            )
        self._mutable_roots = set(mutable_roots)

    def is_mutable(self, value_id: int) -> bool:
        if self._mutable_roots is None:
            raise RuntimeError("view access plan has not been finalized")
        return value_id in self._parent and self._find(value_id) in self._mutable_roots


def _connect_columns(plan: ViewAccessPlan, columns: Iterable[Iterable[int]]) -> None:
    for column in columns:
        plan.connect(*column)


def _connect_selection(plan: ViewAccessPlan, operation: Operation) -> None:
    if operation.name == "scf.lookup":
        for result, alternatives in lookup_columns(operation):
            plan.connect(result, *alternatives)
        return
    for region in operation.regions:
        if len(region.blocks) == 1 and region.blocks[0].ops:
            terminator = region.blocks[0].ops[-1]
            if terminator.name == "scf.yield":
                _connect_columns(
                    plan, zip(operation.results, terminator.operands, strict=True)
                )


def _function_signatures(
    plan: ViewAccessPlan,
    module: Module,
    resolve: Callable[[Operation], Any],
    flat_source_parameters: Callable[[Operation, list[int]], list[Any]],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Plans function view contracts and returns source signature mappings."""

    functions = {
        str(resolve(operation).attr("callee")): operation
        for operation in module.body.ops
        if operation.name in {"func.def", "func.decl"}
    }
    function_parameters: dict[str, list[Any]] = {}
    function_results: dict[str, list[Any]] = {}
    for symbol, function in functions.items():
        body = resolve(function).region("body") if function.name == "func.def" else None
        if function.name == "func.def" and (body is None or len(body.blocks) != 1):
            continue
        arguments = (
            list(body.blocks[0].arg_ids)
            if body is not None
            else list(function.operands)
        )
        function_parameters[symbol] = flat_source_parameters(function, arguments)
        function_results[symbol] = flat_source_parameters(
            function, list(function.results)
        )
        signature_flags = function.attributes.get("signature_flags", b"")
        for position, argument in enumerate(arguments):
            argument_type = module.values[argument].type
            if not (
                isinstance(argument_type, ShapedType)
                and argument_type.type_kind == TypeKind.VIEW
            ):
                continue
            access = signature_flags[position] if signature_flags else 0
            if access in {_READ_ACCESS, _NONE_ACCESS}:
                plan.require_immutable(argument, function)
            elif access in {_WRITE_ACCESS, _READ_WRITE_ACCESS} or (
                function.name == "func.decl" and access == 0
            ):
                plan.require_mutable(argument, function)
    return function_parameters, function_results


def plan_view_access(
    module: Module,
    resolve: Callable[[Operation], Any],
    flat_source_parameters: Callable[[Operation, list[int]], list[Any]],
) -> ViewAccessPlan:
    """Builds the source-level access plan for all supported view values."""

    plan = ViewAccessPlan(module)
    descriptors = {operation.name: operation for operation in source_ops()}
    operations = list(_walk(module.body))

    for operation, _ in operations:
        resolved = resolve(operation)
        descriptor = descriptors.get(operation.name)
        if descriptor is not None:
            for effect in descriptor.effects:
                if effect.operand is None or effect.kind == EffectKind.READ:
                    continue
                for value_id in resolved.value_ids(effect.operand):
                    plan.require_mutable(value_id, operation)

        if operation.name in {"view.subview", "view.refine", "view.bitcast"}:
            plan.connect(resolved.value_id("source"), resolved.value_id("result"))
        elif operation.name == "scf.select":
            plan.connect(
                resolved.value_id("true_value"),
                resolved.value_id("false_value"),
                resolved.value_id("result"),
            )
        elif operation.name in {"scf.if", "scf.switch", "scf.lookup"}:
            _connect_selection(plan, operation)
        elif operation.name == "scf.for":
            body = resolved.region("body")
            if body is None or len(body.blocks) != 1 or not body.blocks[0].ops:
                continue
            block = body.blocks[0]
            terminator = block.ops[-1]
            if terminator.name != "scf.yield":
                continue
            _connect_columns(
                plan,
                zip(
                    resolved.value_ids("iter_args"),
                    block.arg_ids[1:],
                    terminator.operands,
                    resolved.value_ids("results"),
                    strict=True,
                ),
            )
        elif operation.name == "scf.while":
            before_region = resolved.region("before")
            after_region = resolved.region("after")
            if (
                before_region is None
                or len(before_region.blocks) != 1
                or after_region is None
                or len(after_region.blocks) != 1
                or not before_region.blocks[0].ops
                or not after_region.blocks[0].ops
            ):
                continue
            before = before_region.blocks[0]
            after = after_region.blocks[0]
            condition = before.ops[-1]
            terminator = after.ops[-1]
            if condition.name != "scf.condition" or terminator.name != "scf.yield":
                continue
            _connect_columns(
                plan,
                zip(
                    resolved.value_ids("iter_args"),
                    before.arg_ids,
                    terminator.operands,
                    strict=True,
                ),
            )
            _connect_columns(
                plan,
                zip(
                    resolve(condition).value_ids("forwarded"),
                    after.arg_ids,
                    resolved.value_ids("results"),
                    strict=True,
                ),
            )
        elif operation.name == "channel.bind":
            plan.require_mutable(resolved.value_id("storage"), operation)
        elif operation.name == "channel.reserve":
            plan.require_mutable(resolved.value_id("view"), operation)
        elif operation.name == "channel.acquire":
            result = resolved.value_id("view")
            if resolved.attr("mode") == "mutable":
                plan.require_mutable(result, operation)
            else:
                plan.require_immutable(result, operation)
        elif operation.name == "channel.wait":
            result = resolved.value_id("view")
            read_type = module.values[resolved.value_id("read")].type
            mutable = (
                isinstance(read_type, ParameterizedType)
                and read_type.family_name == "read"
                and read_type.get("mode") == _CHANNEL_READ_MUTABLE_MODE
            )
            if mutable:
                plan.require_mutable(result, operation)
            else:
                plan.require_immutable(result, operation)

    function_parameters, function_results = _function_signatures(
        plan, module, resolve, flat_source_parameters
    )

    for operation, owner in operations:
        if operation.name == "func.call":
            symbol = str(resolve(operation).attr("callee"))
            formals = function_parameters.get(symbol)
            results = function_results.get(symbol)
            if formals is None or results is None:
                continue
            actuals = flat_source_parameters(operation, list(operation.operands))
            actual_results = flat_source_parameters(operation, list(operation.results))
            for actual, formal in zip(actuals, formals, strict=True):
                plan.require_mutability_from(
                    formal.value_id, actual.value_id, operation
                )
            _connect_columns(
                plan,
                (
                    (actual.value_id, formal.value_id)
                    for actual, formal in zip(actual_results, results, strict=True)
                ),
            )
        elif operation.name == "func.return" and owner is not None:
            symbol = str(resolve(owner).attr("callee"))
            results = function_results.get(symbol)
            if results is None:
                continue
            returned = flat_source_parameters(operation, list(operation.operands))
            _connect_columns(
                plan,
                (
                    (actual.value_id, formal.value_id)
                    for actual, formal in zip(returned, results, strict=True)
                ),
            )

    # Staged callable view parameters are externally supplied launch bindings.
    # The High IR does not retain their C++ access qualification, so preserve
    # the broad mutable contract instead of inferring one from a local body.
    for operation in module.body.ops:
        if operation.name not in {
            "command.def",
            "command.decl",
            "pipeline.def",
            "pipeline.decl",
        }:
            continue
        resolved = resolve(operation)
        if operation.name.endswith(".decl"):
            arguments = operation.operands
        else:
            body = resolved.region("body")
            arguments = body.blocks[0].arg_ids if body and len(body.blocks) == 1 else ()
        for argument in arguments:
            plan.require_mutable(argument, operation)

    plan.finish()
    return plan
