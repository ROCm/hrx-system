// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Small growable word writer for SPIR-V binary emission.
//
// SPIR-V binaries are streams of little-endian 32-bit words. The writer stores
// host-order words because Loom currently emits only for little-endian hosts
// and all consumers at this layer inspect or hand off word-aligned binary
// payloads.

#ifndef LOOM_TARGET_EMIT_SPIRV_BINARY_WRITER_H_
#define LOOM_TARGET_EMIT_SPIRV_BINARY_WRITER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_binary_writer_t {
  // Allocator used for word storage.
  iree_allocator_t allocator;
  // Mutable SPIR-V word storage.
  uint32_t* words;
  // Number of initialized words in |words|.
  iree_host_size_t word_count;
  // Allocated word capacity of |words|.
  iree_host_size_t word_capacity;
} loom_spirv_binary_writer_t;

// Initializes |writer| with empty allocator-owned word storage.
void loom_spirv_binary_writer_initialize(
    iree_allocator_t allocator, loom_spirv_binary_writer_t* out_writer);

// Releases storage owned by |writer|. Safe to call on a zero-initialized
// writer.
void loom_spirv_binary_writer_deinitialize(loom_spirv_binary_writer_t* writer);

// Moves word storage out of |writer|. The caller owns the returned words and
// must release them with |writer|'s allocator.
void loom_spirv_binary_writer_steal_words(loom_spirv_binary_writer_t* writer,
                                          uint32_t** out_words,
                                          iree_host_size_t* out_word_count);

// Appends one raw word to |writer|.
iree_status_t loom_spirv_binary_write_word(loom_spirv_binary_writer_t* writer,
                                           uint32_t word);

// Appends |word_count| raw words to |writer|.
iree_status_t loom_spirv_binary_write_words(loom_spirv_binary_writer_t* writer,
                                            const uint32_t* words,
                                            iree_host_size_t word_count);

// Appends a SPIR-V instruction header and raw word operands. |operand_count|
// excludes the instruction header word.
iree_status_t loom_spirv_binary_write_instruction(
    loom_spirv_binary_writer_t* writer, uint16_t opcode,
    const uint32_t* operands, iree_host_size_t operand_count);

// Appends a SPIR-V instruction that contains one literal string between two
// raw operand spans. String bytes are copied with the required trailing NUL and
// padded to a 32-bit word boundary.
iree_status_t loom_spirv_binary_write_string_instruction(
    loom_spirv_binary_writer_t* writer, uint16_t opcode,
    const uint32_t* prefix_operands, iree_host_size_t prefix_operand_count,
    iree_string_view_t string, const uint32_t* suffix_operands,
    iree_host_size_t suffix_operand_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_BINARY_WRITER_H_
