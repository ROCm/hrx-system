// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Small growable byte writer for WebAssembly binary emission.
//
// This is target-local plumbing, not a general Loom bytecode writer. It writes
// raw Wasm bytes and Wasm's LEB128 integer encodings.

#ifndef LOOM_TARGET_EMIT_WASM_BINARY_WRITER_H_
#define LOOM_TARGET_EMIT_WASM_BINARY_WRITER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_wasm_binary_writer_t {
  // Allocator used for byte storage.
  iree_allocator_t allocator;
  // Mutable byte storage.
  uint8_t* data;
  // Number of initialized bytes in |data|.
  iree_host_size_t length;
  // Allocated byte capacity of |data|.
  iree_host_size_t capacity;
} loom_wasm_binary_writer_t;

// Initializes |writer| with empty allocator-owned storage.
void loom_wasm_binary_writer_initialize(iree_allocator_t allocator,
                                        loom_wasm_binary_writer_t* out_writer);

// Releases storage owned by |writer|. Safe to call on a zero-initialized
// writer.
void loom_wasm_binary_writer_deinitialize(loom_wasm_binary_writer_t* writer);

// Appends one byte to |writer|.
iree_status_t loom_wasm_binary_write_u8(loom_wasm_binary_writer_t* writer,
                                        uint8_t value);

// Appends |data_length| raw bytes to |writer|.
iree_status_t loom_wasm_binary_write_bytes(loom_wasm_binary_writer_t* writer,
                                           const uint8_t* data,
                                           iree_host_size_t data_length);

// Appends an unsigned 32-bit integer using Wasm unsigned LEB128 encoding.
iree_status_t loom_wasm_binary_write_u32_leb(loom_wasm_binary_writer_t* writer,
                                             uint32_t value);

// Appends a signed 32-bit integer using Wasm signed LEB128 encoding.
iree_status_t loom_wasm_binary_write_i32_leb(loom_wasm_binary_writer_t* writer,
                                             int32_t value);

// Appends a little-endian 64-bit payload.
iree_status_t loom_wasm_binary_write_u64_le(loom_wasm_binary_writer_t* writer,
                                            uint64_t value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_BINARY_WRITER_H_
