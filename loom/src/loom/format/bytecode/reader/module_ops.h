// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MODULE_OPS section framing and allocation-summary validation.

#ifndef LOOM_FORMAT_BYTECODE_READER_MODULE_OPS_H_
#define LOOM_FORMAT_BYTECODE_READER_MODULE_OPS_H_

#include "loom/format/bytecode/module_summary.h"
#include "loom/format/bytecode/reader/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decodes and validates the allocation summary prefix of MODULE_OPS.
iree_status_t loom_bytecode_module_ops_summary_read(
    loom_bytecode_reader_decoder_t* decoder,
    iree_const_byte_span_t payload_bytes, uint64_t payload_absolute_offset,
    loom_bytecode_module_ops_summary_t* out_summary);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_MODULE_OPS_H_
