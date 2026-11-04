# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.gen.target.contracts.contract_index import ContractCasePool
from loom.target.contracts import ContractSystem
from loom.target.contracts.compile import (
    CONTRACT_ROW_NONE,
    CompiledContractIndex,
    CompiledContractOpEntry,
    CompiledIndexCase,
)


def _index(
    cases: tuple[CompiledIndexCase, ...],
    *spans: tuple[int, int],
) -> CompiledContractIndex:
    return CompiledContractIndex(
        7,
        (tuple(CompiledContractOpEntry(start, count, count) if count else CompiledContractOpEntry(CONTRACT_ROW_NONE, 0, 0) for start, count in spans),),
        cases,
    )


def test_contract_case_pool_interns_rebased_operation_spans() -> None:
    pool = ContractCasePool("kTestCases")
    first = pool.pool_index(
        _index(
            (
                CompiledIndexCase(ContractSystem.VALUE_ALIAS, 0, 10),
                CompiledIndexCase(ContractSystem.VALUE_ALIAS, 1, 20),
                CompiledIndexCase(ContractSystem.VALUE_ELIDE, 1, 21),
            ),
            (0, 2),
            (2, 1),
            (CONTRACT_ROW_NONE, 0),
        ),
        (1, 3),
    )
    second = pool.pool_index(
        _index(
            (
                CompiledIndexCase(ContractSystem.VALUE_ALIAS, 0, 10),
                CompiledIndexCase(ContractSystem.VALUE_ALIAS, 1, 20),
                CompiledIndexCase(ContractSystem.VALUE_ALIAS, 0, 99),
            ),
            (0, 2),
            (2, 1),
        ),
        (1, 3),
    )

    assert first.case_index_bound == 3
    assert [entry.case_start for entry in first.dialects[0]] == [
        0,
        2,
        CONTRACT_ROW_NONE,
    ]
    assert second.case_index_bound == 4
    assert [entry.case_start for entry in second.dialects[0]] == [0, 3]
    assert pool.cases == (
        CompiledIndexCase(ContractSystem.VALUE_ALIAS, 1, 10),
        CompiledIndexCase(ContractSystem.VALUE_ALIAS, 3, 20),
        CompiledIndexCase(ContractSystem.VALUE_ELIDE, 3, 21),
        CompiledIndexCase(ContractSystem.VALUE_ALIAS, 1, 99),
    )
