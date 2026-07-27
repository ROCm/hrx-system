# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target contract fragment records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loom.target.contracts.descriptors import _validate_descriptor_set_keys
from loom.target.contracts.materializers import ValueMaterializer
from loom.target.contracts.rules import ContractCase
from loom.target.low_descriptors import DescriptorSet


@dataclass(frozen=True, slots=True)
class ContractFragment:
    """Target contract fragment authored and validated in Python."""

    name: str
    descriptor_set: DescriptorSet
    cases: tuple[ContractCase, ...] = ()
    public_header: str = ""
    c_source_includes: tuple[str, ...] = ()
    target_contract_query: bool = True
    materializers: tuple[ValueMaterializer, ...] = ()

    def __init__(
        self,
        *,
        name: str,
        descriptor_set: DescriptorSet,
        cases: Sequence[ContractCase] = (),
        public_header: str = "",
        c_source_includes: Sequence[str] = (),
        target_contract_query: bool = True,
        materializers: Sequence[ValueMaterializer] = (),
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "descriptor_set", descriptor_set)
        object.__setattr__(self, "cases", tuple(cases))
        object.__setattr__(self, "public_header", public_header)
        object.__setattr__(self, "c_source_includes", tuple(c_source_includes))
        object.__setattr__(self, "target_contract_query", target_contract_query)
        object.__setattr__(self, "materializers", tuple(materializers))
        if not name:
            raise ValueError("contract fragment name must be non-empty")
        _validate_descriptor_set_keys(descriptor_set)
        _validate_c_source_includes(self.c_source_includes)
        _validate_materializer_names(self.materializers)
        for case in self.cases:
            case.validate(descriptor_set)


def contract_fragment_public_header(fragment: ContractFragment) -> str:
    """Returns the public C header path for a contract fragment."""

    if fragment.public_header:
        return fragment.public_header
    raise ValueError(f"contract fragment '{fragment.name}' requires public_header")


def _validate_c_source_includes(includes: tuple[str, ...]) -> None:
    for include in includes:
        if not include:
            raise ValueError("contract fragment C source include must be non-empty")


def _validate_materializer_names(
    materializers: tuple[ValueMaterializer, ...],
) -> None:
    seen = set[str]()
    for materializer in materializers:
        if materializer.name in seen:
            raise ValueError(
                f"contract fragment materializer '{materializer.name}' is duplicated"
            )
        seen.add(materializer.name)
