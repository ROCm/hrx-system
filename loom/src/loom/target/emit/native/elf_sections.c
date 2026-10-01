// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/elf_sections.h"

loom_native_elf_section_t loom_native_elf_section_from_native(
    const loom_native_section_t* section) {
  uint64_t flags = section->access != LOOM_NATIVE_SECTION_ACCESS_NONE
                       ? LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC
                       : 0;
  if (iree_any_bit_set(section->access, LOOM_NATIVE_SECTION_ACCESS_WRITE)) {
    flags |= LOOM_NATIVE_ELF_SECTION_FLAG_WRITE;
  }
  if (iree_any_bit_set(section->access, LOOM_NATIVE_SECTION_ACCESS_EXECUTE)) {
    flags |= LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR;
  }
  return (loom_native_elf_section_t){
      .name = section->name,
      .type = section->storage == LOOM_NATIVE_SECTION_STORAGE_RESERVATION
                  ? LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS
                  : LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .flags = flags,
      .address = section->address,
      .alignment = section->alignment,
      .contents = section->contents,
      .zero_fill_length = section->reservation_length,
  };
}
