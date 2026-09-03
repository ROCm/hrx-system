# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.gen.target.arch.vm import generate


def test_generated_source_is_compact_data() -> None:
    source = generate.generate_source()
    assert "loom_vm_instruction_descriptors[256]" in source
    assert "loom_vm_instruction_fields[]" in source
    assert "loom_vm_source_lowerings[]" in source
    for instruction in generate.SPECIFICATION.instructions:
        assert source.count(f"// {instruction.mnemonic}\n") == len(instruction.fields)
    assert "switch (" not in source
    assert "iree_status_t" not in source
