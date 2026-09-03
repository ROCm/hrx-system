# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.gen.target.arch.vm.generate import generate_source


def test_generated_source_is_compact_data() -> None:
    source = generate_source()
    assert "loom_vm_instruction_descriptors[256]" in source
    assert "loom_vm_instruction_fields[]" in source
    assert "loom_vm_source_lowerings[]" in source
    assert "LOOM_OP_SCALAR_ADDI" in source
    assert "LOOM_SCALAR_TYPE_I64" in source
    assert "IREE_VM_BYTECODE_OPCODE_INTEGER_ADD_I32" in source
    assert "switch (" not in source
    assert "iree_status_t" not in source
