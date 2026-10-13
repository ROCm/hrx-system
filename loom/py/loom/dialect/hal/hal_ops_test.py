# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL source authoring preserves grouping without native ABI attributes."""

from loom.builders import module_builder
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.hal import ALL_HAL_OPS, ALL_HAL_TYPES
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.format.text.printer import Printer
from loom.ir import I64, DialectType


def test_queue_copy_builder_and_roundtrips() -> None:
    ops = (*ALL_FUNC_OPS, *ALL_HAL_OPS)
    types = (*ALL_BUILTIN_TYPES, *ALL_HAL_TYPES)
    module, builder = module_builder(ops=ops, types=types)
    queue = builder.value("queue", DialectType("hal.queue"))
    semaphore = builder.value("semaphore", DialectType("hal.semaphore"))
    buffer = builder.value("buffer", DialectType("hal.buffer"))
    offset = builder.value("offset", I64)
    length = builder.value("length", I64)
    value = builder.value("value", I64)
    flags = builder.value("flags", I64)
    body = builder.region()
    builder.func.def_(callee="copy", args=[queue, semaphore, buffer, offset, length, value, flags], results=[], body=body)
    with builder.insertion_block(body.blocks[0]):
        builder.hal.queue_copy(arguments=(queue, [(semaphore, value)], [], (buffer, offset), (buffer, offset), length, (None, []), flags))
        builder.func.return_()

    operation = body.blocks[0].ops[0]
    assert set(operation.attributes) == {"shape"}
    assert operation.operands == [queue.id, semaphore.id, value.id, buffer.id, offset.id, buffer.id, offset.id, length.id, flags.id]
    printer = Printer()
    printer.register_ops(ops)
    printer.register_types(types)
    expected = (
        "func.def @copy(%queue: hal.queue, %semaphore: hal.semaphore, %buffer: hal.buffer, %offset: i64, %length: i64, %value: i64, %flags: i64) {\n"
        "  hal.queue.copy (%queue, [(%semaphore, %value)], [], (%buffer, %offset), (%buffer, %offset), %length, (none, []), %flags)\n"
        "  func.return\n"
        "}\n"
    )
    assert printer.print_module(module) == expected
    parser = Parser()
    parser.register_ops(ops)
    parser.register_types(types)
    assert printer.print_module(parser.parse(expected)) == expected
    assert printer.print_module(read_module(write_module(module))) == expected
