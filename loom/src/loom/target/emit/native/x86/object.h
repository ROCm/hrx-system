// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86-64 SysV relocatable object emission.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_OBJECT_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_OBJECT_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonical relocatable ELF object emitter for ordinary x86 functions.
extern const loom_target_emitter_t loom_x86_elf_object_emitter;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_OBJECT_H_
