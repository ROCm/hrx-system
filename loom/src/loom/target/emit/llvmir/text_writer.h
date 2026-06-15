// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Textual LLVM IR writer.
//
// The text writer is a serializer over loom_llvmir_module_t. It must not create
// declarations, invent attributes, or parse textual instruction fragments.

#ifndef LOOM_TARGET_LLVMIR_TEXT_WRITER_H_
#define LOOM_TARGET_LLVMIR_TEXT_WRITER_H_

#include "loom/target/emit/llvmir/module.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_llvmir_text_write_module(const loom_llvmir_module_t* module,
                                            loom_output_stream_t* stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TEXT_WRITER_H_
