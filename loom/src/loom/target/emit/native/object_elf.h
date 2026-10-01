// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ELF relocatable objects from compiler-owned native contributions.

#ifndef LOOM_TARGET_EMIT_NATIVE_OBJECT_ELF_H_
#define LOOM_TARGET_EMIT_NATIVE_OBJECT_ELF_H_

#include "loom/target/emit/native/elf.h"
#include "loom/target/emit/native/object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Joins contribution storage and writes a detached ELF64LE relocatable object.
// The contribution is compiler-owned: symbol names are nonempty and contain no
// NUL, definitions name valid contribution offsets, and local symbols precede
// global/weak symbols. The producer establishes its external symbol namespace.
// This writer owns ELF section/string/symbol tables, including the undefined
// symbol and non-executable-stack declaration. It does not interpret code or
// select a calling convention. Relocation kinds require an explicit ELF mapping
// and are rejected until supplied by a product supporting them.
//
// Scratch arrays and joined sections live in |arena|; the stream owns the
// output independently. Failures are allocation, output, or ELF
// representability.
iree_status_t loom_native_object_write_elf64le(
    const loom_native_object_contribution_t* contribution,
    loom_native_elf_machine_t machine, iree_io_stream_t* stream,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_OBJECT_ELF_H_
