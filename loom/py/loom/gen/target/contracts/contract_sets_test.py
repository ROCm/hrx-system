# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.target.contracts.contract_sets import (
    ContractSetGenerationInput,
    generate_contract_sets,
)
from loom.target.contracts import (
    CompiledContractFragment,
    ContractFragment,
    compile_contract_set,
)
from loom.target.test.descriptors import TEST_LOW_CORE_DESCRIPTOR_SET


def _contract_fragment(
    name: str,
) -> tuple[ContractFragment, CompiledContractFragment]:
    authored = ContractFragment(
        name=name,
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=f"loom/target/test/contracts/{name}.h",
    )
    compiled = CompiledContractFragment(
        name=name,
        target_contract_query=True,
        op_spans=(),
        cases=(),
        descriptor_rules=(),
        descriptor_matrices=(),
    )
    return authored, compiled


def test_generate_contract_sets_materializes_single_fragment_index() -> None:
    authored, compiled_fragment = _contract_fragment("test.alpha")
    compiled_set = compile_contract_set(
        "test.single",
        (compiled_fragment,),
    )

    generated = generate_contract_sets(
        (ContractSetGenerationInput(compiled_set, (authored,)),),
        public_header="loom/target/test/contracts/sets.h",
    )

    assert "loom_test_single_contract_set" in generated.header
    assert "kTestSingleContractSetIndex" in generated.source
    assert "&loom_test_alpha_contract_index" not in generated.source


def test_generate_contract_sets_materializes_composite_index() -> None:
    first_authored, first_compiled = _contract_fragment("test.first")
    second_authored, second_compiled = _contract_fragment("test.second")
    compiled_set = compile_contract_set(
        "test.composite",
        (first_compiled, second_compiled),
    )

    generated = generate_contract_sets(
        (
            ContractSetGenerationInput(
                compiled_set,
                (first_authored, second_authored),
            ),
        ),
        public_header="loom/target/test/contracts/sets.h",
    )

    first_binding = "{&loom_test_first_contract_fragment, LOOM_TARGET_CONTRACT_ROW_NONE}"
    second_binding = "{&loom_test_second_contract_fragment, LOOM_TARGET_CONTRACT_ROW_NONE}"
    assert generated.source.index(first_binding) < generated.source.index(second_binding)
    assert "kTestCompositeContractSetIndex" in generated.source
    assert "&loom_test_first_contract_index" not in generated.source
