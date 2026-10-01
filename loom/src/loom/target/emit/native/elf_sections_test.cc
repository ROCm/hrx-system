// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/elf_sections.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(NativeElfSectionsTest, PreservesPlacedContentsAndAccess) {
  const uint8_t bytes[] = {0x01, 0x02, 0x03};
  const struct {
    // Runtime access required by native storage.
    loom_native_section_access_t access;
    // Corresponding ELF section-header flags.
    uint64_t elf_flags;
  } cases[] = {
      {LOOM_NATIVE_SECTION_ACCESS_NONE, 0},
      {LOOM_NATIVE_SECTION_ACCESS_READ, LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC},
      {LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_WRITE,
       LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC | LOOM_NATIVE_ELF_SECTION_FLAG_WRITE},
      {LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
       LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
           LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR},
  };
  for (const auto& test_case : cases) {
    const loom_native_section_t native = {
        /*.name=*/IREE_SV(".payload"),
        /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
        /*.access=*/test_case.access,
        /*.address=*/0x100004000ull,
        /*.alignment=*/64,
        /*.contents=*/iree_make_const_byte_span(bytes, sizeof(bytes)),
    };
    const loom_native_elf_section_t elf =
        loom_native_elf_section_from_native(&native);
    EXPECT_EQ(elf.name.data, native.name.data);
    EXPECT_EQ(elf.name.size, native.name.size);
    EXPECT_EQ(elf.type, LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS);
    EXPECT_EQ(elf.flags, test_case.elf_flags);
    EXPECT_EQ(elf.address, 0x100004000ull);
    EXPECT_EQ(elf.alignment, 64u);
    EXPECT_EQ(elf.contents.data, bytes);
    EXPECT_EQ(elf.contents.data_length, sizeof(bytes));
    EXPECT_EQ(elf.zero_fill_length, 0u);
    EXPECT_EQ(elf.entry_size, 0u);
    EXPECT_EQ(elf.link, 0u);
    EXPECT_EQ(elf.info, 0u);
  }
}

TEST(NativeElfSectionsTest, RepresentsReservationWithoutPayload) {
  const loom_native_section_t native = {
      /*.name=*/IREE_SV(".storage"),
      /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
      /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
          LOOM_NATIVE_SECTION_ACCESS_WRITE,
      /*.address=*/0x70000,
      /*.alignment=*/32,
      /*.contents=*/{},
      /*.reservation_length=*/320,
  };
  const loom_native_elf_section_t elf =
      loom_native_elf_section_from_native(&native);
  EXPECT_EQ(elf.type, LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS);
  EXPECT_EQ(elf.flags, LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                           LOOM_NATIVE_ELF_SECTION_FLAG_WRITE);
  EXPECT_EQ(elf.address, 0x70000u);
  EXPECT_EQ(elf.alignment, 32u);
  EXPECT_EQ(elf.contents.data, nullptr);
  EXPECT_EQ(elf.contents.data_length, 0u);
  EXPECT_EQ(loom_native_elf_section_byte_length(&elf), 320u);
}

}  // namespace
}  // namespace loom
