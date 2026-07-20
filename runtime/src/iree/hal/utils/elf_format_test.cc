// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/elf_format.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace iree::hal {
namespace {

TEST(ElfFormatTest, DetectsBoundedMagic) {
  const uint8_t elf_data[] = {0x7F, 'E', 'L', 'F', 0};
  EXPECT_TRUE(iree_hal_elf_data_starts_with_magic(
      iree_make_const_byte_span(elf_data, sizeof(elf_data))));
  EXPECT_FALSE(iree_hal_elf_data_starts_with_magic(
      iree_make_const_byte_span(elf_data, 3)));
  EXPECT_FALSE(
      iree_hal_elf_data_starts_with_magic(iree_const_byte_span_empty()));

  const uint8_t non_elf_data[] = {0x7F, 'E', 'L', 'X'};
  EXPECT_FALSE(iree_hal_elf_data_starts_with_magic(
      iree_make_const_byte_span(non_elf_data, sizeof(non_elf_data))));
}

}  // namespace
}  // namespace iree::hal
