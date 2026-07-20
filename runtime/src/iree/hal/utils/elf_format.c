// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/elf_format.h"

#include <string.h>

static const uint8_t iree_hal_elf_magic[] = {0x7F, 'E', 'L', 'F'};

bool iree_hal_elf_data_starts_with_magic(iree_const_byte_span_t elf_data) {
  return elf_data.data_length >= sizeof(iree_hal_elf_magic) &&
         memcmp(elf_data.data, iree_hal_elf_magic,
                sizeof(iree_hal_elf_magic)) == 0;
}
