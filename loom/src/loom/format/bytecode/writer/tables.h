// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared bytecode catalog table serialization.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_TABLES_H_
#define LOOM_FORMAT_BYTECODE_WRITER_TABLES_H_

#include "loom/format/bytecode/writer/encoder.h"
#include "loom/format/bytecode/writer/numbering.h"

#ifdef __cplusplus
extern "C" {
#endif

// Streams the first-use-ordered string catalog.
iree_status_t loom_bytecode_write_strings_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering);

// Streams module-local source identifier rows.
iree_status_t loom_bytecode_write_sources_section(
    loom_bytecode_page_writer_t* page_writer, const loom_module_t* module);

// Streams the topologically ordered structural type catalog.
iree_status_t loom_bytecode_write_types_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering);

// Streams encoding kind and parameterized instance rows.
iree_status_t loom_bytecode_write_encodings_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering);

// Streams operation kind registry rows.
iree_status_t loom_bytecode_write_ops_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering);

// Streams module-local source location rows.
iree_status_t loom_bytecode_write_locations_section(
    loom_bytecode_page_writer_t* page_writer, const loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_TABLES_H_
