// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tagged bytecode attribute serialization.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_ATTRIBUTE_H_
#define LOOM_FORMAT_BYTECODE_WRITER_ATTRIBUTE_H_

#include "loom/format/bytecode/writer/encoder.h"
#include "loom/format/bytecode/writer/numbering.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes one tagged attribute directly to a streaming bytecode section.
iree_status_t loom_bytecode_write_attr_value(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor);

// Writes a scoped enum using the active Low representation contract.
iree_status_t loom_bytecode_write_scoped_enum(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr);

// Appends one tagged attribute to a buffered bytecode section.
iree_status_t loom_bytecode_emit_attr_value(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_ATTRIBUTE_H_
