// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ELF payload-section preparation from format-independent native storage.
// Product builders own this adaptation. The ELF serializer consumes completed
// ELF records and has no dependency on native contributions.

#ifndef LOOM_TARGET_EMIT_NATIVE_ELF_SECTIONS_H_
#define LOOM_TARGET_EMIT_NATIVE_ELF_SECTIONS_H_

#include "loom/target/emit/native/contribution.h"
#include "loom/target/emit/native/elf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Translates placed native bytes/reservations into an ELF payload section.
// The result borrows the name and contents from |section|. Format-specific
// tables are built separately by the product builder. Reservations become
// SHT_NOBITS; the target loading contract determines their initialization.
loom_native_elf_section_t loom_native_elf_section_from_native(
    const loom_native_section_t* section);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_ELF_SECTIONS_H_
