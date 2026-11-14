// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source coordinates from validated indexed bytecode, without IR construction.

#ifndef LOOM_FORMAT_BYTECODE_INDEXED_SOURCE_H_
#define LOOM_FORMAT_BYTECODE_INDEXED_SOURCE_H_

#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/index.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves a validated location against its retained index. Filenames borrow
// from |metadata|. Tagged locations preserve their origin; fused and opaque
// locations have no designated primary source range. Performs no allocation.
bool loom_bytecode_index_resolve_source(
    iree_const_byte_span_t bytecode,
    const loom_bytecode_module_metadata_t* metadata,
    loom_location_id_t location, loom_source_range_t* out_range);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_INDEXED_SOURCE_H_
