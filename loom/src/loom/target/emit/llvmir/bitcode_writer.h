// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM bitcode writer.
//
// The bitcode writer is the binary-format peer to text_writer. Both consume the
// same structured loom_llvmir_module_t; neither owns lowering decisions or
// repairs invalid modules while serializing.

#ifndef LOOM_TARGET_LLVMIR_BITCODE_WRITER_H_
#define LOOM_TARGET_LLVMIR_BITCODE_WRITER_H_

#include "iree/io/stream.h"
#include "loom/target/emit/llvmir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_llvmir_bitcode_write_module(
    const loom_llvmir_module_t* module, iree_io_stream_t* stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_BITCODE_WRITER_H_
