# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Call-sensitive immutable buffer origins for C++ source projection."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loom.ir import BufferType, Module, Operation

# Other origins are formal buffer parameter value IDs in the owning callable.
_IMMUTABLE_STORAGE = -1


@dataclass(eq=False)
class BufferProjection:
    """One ordinary callable with concrete source buffer qualification."""

    # Original callable; source projection does not mutate its IR.
    function: Operation
    # Formal buffer parameters receiving immutable storage in this projection.
    immutable_arguments: frozenset[int]
    # Callee projections selected once for the calls in this body.
    calls: dict[int, BufferProjection] = field(default_factory=dict)


@dataclass
class _Flow:
    """A union of origins, optionally substituted across one call boundary."""

    # Buffer value receiving the union.
    target: int
    # Buffer values whose origins flow to the target.
    sources: tuple[int, ...]
    # Formal-to-actual buffer mapping for a callable result summary.
    arguments: dict[int, int] | None = None


class BufferAccessPlan:
    """Retained buffer origins and reachable qualified callable projections.

    Summaries name the owning function's formal inputs, not all callers' roots.
    Substitution at each call therefore keeps an immutable identity invocation
    independent from a writable invocation of the same helper. Loops and
    recursion converge by monotone union over the finite origin sets.
    """

    def __init__(self, module: Module, resolve: Callable[[Operation], Any]):
        self._origins: dict[int, frozenset[int]] = {
            value_id: frozenset()
            for value_id, value in enumerate(module.values)
            if isinstance(value.type, BufferType)
        }
        self._functions = {
            str(resolve(operation).attr("callee")): operation
            for operation in module.body.ops
            if operation.name in {"func.def", "func.decl"}
        }
        self._parameters: dict[str, tuple[int, ...]] = {}
        self._calls: dict[str | None, list[Operation]] = defaultdict(list)
        self._projections: dict[tuple[str, frozenset[int]], BufferProjection] = {}
        self.projections: dict[str, list[BufferProjection]] = defaultdict(list)
        self.calls: dict[int, BufferProjection] = {}
        flows: list[_Flow] = []

        def connect(targets: Any, sources: Any) -> None:
            for target, source in zip(targets, sources, strict=True):
                if target in self._origins:
                    flows.append(_Flow(target, (source,)))

        for symbol, function in self._functions.items():
            parameters = (
                function.operands
                if function.name == "func.decl"
                else resolve(function).region("body").blocks[0].arg_ids
            )
            self._parameters[symbol] = tuple(parameters)
            for parameter in parameters:
                if parameter in self._origins:
                    self._origins[parameter] = frozenset({parameter})

        def constant(operation: Operation, resolved: Any, owner: str | None) -> None:
            self._origins[operation.results[0]] = frozenset({_IMMUTABLE_STORAGE})

        def forward(operation: Operation, resolved: Any, owner: str | None) -> None:
            connect(operation.results, operation.operands)

        def select(operation: Operation, resolved: Any, owner: str | None) -> None:
            result = resolved.value_id("result")
            if result in self._origins:
                flows.append(
                    _Flow(
                        result,
                        (
                            resolved.value_id("true_value"),
                            resolved.value_id("false_value"),
                        ),
                    )
                )

        def branch(operation: Operation, resolved: Any, owner: str | None) -> None:
            for region in operation.regions:
                connect(operation.results, region.blocks[0].ops[-1].operands)

        def counted_loop(
            operation: Operation, resolved: Any, owner: str | None
        ) -> None:
            body = resolved.region("body").blocks[0]
            connect(body.arg_ids[1:], resolved.value_ids("iter_args"))
            connect(body.arg_ids[1:], body.ops[-1].operands)
            connect(operation.results, body.arg_ids[1:])

        def conditional_loop(
            operation: Operation, resolved: Any, owner: str | None
        ) -> None:
            before = resolved.region("before").blocks[0]
            after = resolved.region("after").blocks[0]
            connect(before.arg_ids, resolved.value_ids("iter_args"))
            connect(before.arg_ids, after.ops[-1].operands)
            forwarded = resolve(before.ops[-1]).value_ids("forwarded")
            connect(after.arg_ids, forwarded)
            connect(operation.results, forwarded)

        def return_values(
            operation: Operation, resolved: Any, owner: str | None
        ) -> None:
            if owner is not None:
                connect(self._functions[owner].results, operation.operands)

        def call(operation: Operation, resolved: Any, owner: str | None) -> None:
            self._calls[owner].append(operation)
            symbol = str(resolved.attr("callee"))
            function = self._functions.get(symbol)
            if function is None:
                return  # The call renderer owns unresolved symbols.
            arguments = {
                formal: actual
                for formal, actual in zip(
                    self._parameters[symbol], operation.operands, strict=True
                )
                if formal in self._origins
            }
            for actual, formal in zip(operation.results, function.results, strict=True):
                if actual in self._origins:
                    flows.append(_Flow(actual, (formal,), arguments))

        collectors = {
            "buffer.constant": constant,
            "buffer.assume.alignment": forward,
            "buffer.assume.memory_space": forward,
            "buffer.assume.noalias": forward,
            "scf.select": select,
            "scf.if": branch,
            "scf.switch": branch,
            "scf.for": counted_loop,
            "scf.while": conditional_loop,
            "func.return": return_values,
            "func.call": call,
        }

        def visit(block: Any, owner: str | None = None) -> None:
            for operation in block.ops:
                resolved = resolve(operation)
                body_owner = (
                    str(resolved.attr("callee"))
                    if operation.name == "func.def"
                    else owner
                )
                for region in operation.regions:
                    for child in region.blocks:
                        visit(child, body_owner)
                collector = collectors.get(operation.name)
                if collector is not None:
                    collector(operation, resolved, owner)

        visit(module.body)
        self._solve(flows)
        self._plan_projections(resolve)

    def _solve(self, flows: list[_Flow]) -> None:
        users: dict[int, list[int]] = defaultdict(list)
        for index, flow in enumerate(flows):
            inputs = set(flow.sources)
            if flow.arguments is not None:
                inputs.update(flow.arguments.values())
            for source in inputs:
                users[source].append(index)
        pending = deque(range(len(flows)))
        queued = set(pending)
        while pending:
            index = pending.popleft()
            queued.remove(index)
            flow = flows[index]
            origins = frozenset().union(
                *(self._origins[source] for source in flow.sources)
            )
            if flow.arguments is not None:
                origins = frozenset().union(
                    *(
                        frozenset({_IMMUTABLE_STORAGE})
                        if origin == _IMMUTABLE_STORAGE
                        else self._origins[flow.arguments[origin]]
                        for origin in origins
                    )
                )
            merged = self._origins[flow.target] | origins
            if merged == self._origins[flow.target]:
                continue
            self._origins[flow.target] = merged
            for user in users[flow.target]:
                if user not in queued:
                    pending.append(user)
                    queued.add(user)

    def is_immutable(self, value: int, projection: BufferProjection | None) -> bool:
        origins = self._origins.get(value, frozenset())
        return _IMMUTABLE_STORAGE in origins or (
            projection is not None
            and not origins.isdisjoint(projection.immutable_arguments)
        )

    def _plan_projections(self, resolve: Callable[[Operation], Any]) -> None:
        pending: deque[BufferProjection] = deque()

        def get(symbol: str, immutable: frozenset[int]) -> BufferProjection:
            key = (symbol, immutable)
            if key not in self._projections:
                projection = BufferProjection(self._functions[symbol], immutable)
                self._projections[key] = projection
                self.projections[symbol].append(projection)
                pending.append(projection)
            return self._projections[key]

        def bind_calls(owner: str | None, projection: BufferProjection | None) -> None:
            bindings = projection.calls if projection is not None else self.calls
            for call in self._calls[owner]:
                symbol = str(resolve(call).attr("callee"))
                function = self._functions.get(symbol)
                if function is None:
                    continue
                immutable = frozenset(
                    formal
                    for formal, actual in zip(
                        self._parameters[symbol], call.operands, strict=True
                    )
                    if self.is_immutable(actual, projection)
                )
                # Bodyless declarations have an opaque, fixed source ABI.
                if function.name == "func.decl":
                    immutable = frozenset()
                bindings[id(call)] = get(symbol, immutable)

        def drain() -> None:
            while pending:
                projection = pending.popleft()
                symbol = str(resolve(projection.function).attr("callee"))
                bind_calls(symbol, projection)

        # Preserve linkable entry points independently of internal call sites.
        for symbol, function in self._functions.items():
            if (
                function.name == "func.decl"
                or resolve(function).attr("visibility") == "public"
            ):
                get(symbol, frozenset())
        bind_calls(None, None)
        drain()
        # Unreferenced private libraries are still part of the source projection.
        for symbol in self._functions:
            if symbol not in self.projections:
                get(symbol, frozenset())
                drain()
