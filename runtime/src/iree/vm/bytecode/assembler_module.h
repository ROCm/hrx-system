// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_ASSEMBLER_MODULE_H_
#define IREE_VM_BYTECODE_ASSEMBLER_MODULE_H_

#include "iree/base/api.h"
#include "iree/io/stream.h"
#include "iree/vm/bytecode/assembler_native.h"
#include "iree/vm/bytecode/wire/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Borrowed random-access view of the records produced by one assembly pass.
//
// Record geometry is trusted assembler-owned state. The view exists only to
// validate cross-record semantic relationships that canonical text cannot make
// true by construction, such as byte-sorted names and callable compatibility.
typedef struct iree_vm_bytecode_assembler_module_view_t {
  // Readable and seekable vector stream containing the complete image.
  iree_io_stream_t* stream;
  // Absolute stream offset of each physical record array.
  const uint64_t* record_offsets;
  // Exact number of append-mode rows in each physical record array.
  // Fixed singleton presence is represented by its non-UINT64_MAX offset.
  const uint32_t* record_counts;
  // Absolute stream offset of the string byte tail.
  uint64_t string_data_offset;
  // Grouped native operands in callsite ordinal order, awaiting encoding.
  const iree_vm_bytecode_assembler_native_call_t* native_calls;
} iree_vm_bytecode_assembler_module_view_t;

// Finalizes derived declaration geometry and validates semantic relationships
// in an assembled module before publication.
// All local field, extent, and symbolic-reference constraints must already have
// been established by parsing, layout, and fixup resolution.
iree_status_t iree_vm_bytecode_assembler_module_finalize(
    const iree_vm_bytecode_assembler_module_view_t* view,
    iree_allocator_t scratch_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_ASSEMBLER_MODULE_H_
