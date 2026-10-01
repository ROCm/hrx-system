// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/contribution.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  // Block pool backing the test arena.
  iree_arena_block_pool_t block_pool_ = {0};
  // Arena receiving transient contribution assembly storage.
  iree_arena_allocator_t arena_ = {0};
};

TEST(NativeContributionTest, AssemblesAlignedSectionsAndCopiesStorage) {
  char text_section_name[] = ".text";
  char rodata_section_name[] = ".rodata";
  uint8_t text0[] = {0x01, 0x02, 0x03};
  const uint8_t rodata0[] = {0xa0, 0xa1};
  const uint8_t text1[] = {0x10, 0x11};
  const loom_native_section_contribution_t contributions[] = {
      {
          /*.section_name=*/iree_make_string_view(
              text_section_name, sizeof(text_section_name) - 1u),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          /*.contribution_alignment=*/4,
          /*.contents=*/iree_make_const_byte_span(text0, sizeof(text0)),
      },
      {
          /*.section_name=*/iree_make_string_view(
              rodata_section_name, sizeof(rodata_section_name) - 1u),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ,
          /*.contribution_alignment=*/1,
          /*.contents=*/iree_make_const_byte_span(rodata0, sizeof(rodata0)),
      },
      {
          /*.section_name=*/iree_make_string_view(
              text_section_name, sizeof(text_section_name) - 1u),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          /*.contribution_alignment=*/8,
          /*.contents=*/iree_make_const_byte_span(text1, sizeof(text1)),
      },
  };

  TestArena arena;
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_ASSERT_OK(loom_native_assemble_section_contributions(
      contributions, IREE_ARRAYSIZE(contributions), &assembly, arena.arena()));
  text0[0] = 0xff;
  text_section_name[1] = 'X';
  rodata_section_name[1] = 'X';

  ASSERT_EQ(assembly.section_count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(assembly.sections[0].name, IREE_SV(".text")));
  EXPECT_TRUE(
      iree_string_view_equal(assembly.sections[1].name, IREE_SV(".rodata")));
  ASSERT_EQ(assembly.contribution_layout_count, IREE_ARRAYSIZE(contributions));
  EXPECT_EQ(assembly.contribution_layouts[0].section_index, 0u);
  EXPECT_EQ(assembly.contribution_layouts[0].section_offset, 0u);
  EXPECT_EQ(assembly.contribution_layouts[1].section_index, 1u);
  EXPECT_EQ(assembly.contribution_layouts[1].section_offset, 0u);
  EXPECT_EQ(assembly.contribution_layouts[2].section_index, 0u);
  EXPECT_EQ(assembly.contribution_layouts[2].section_offset, 8u);

  ASSERT_EQ(assembly.sections[0].contents.data_length, 10u);
  EXPECT_EQ(std::string((const char*)assembly.sections[0].contents.data,
                        assembly.sections[0].contents.data_length),
            std::string("\x01\x02\x03\x00\x00\x00\x00\x00\x10\x11", 10));
  EXPECT_EQ(assembly.sections[0].alignment, 8u);
  ASSERT_EQ(assembly.sections[1].contents.data_length, 2u);
  EXPECT_EQ(std::string((const char*)assembly.sections[1].contents.data,
                        assembly.sections[1].contents.data_length),
            std::string("\xa0\xa1", 2));
}

TEST(NativeContributionTest, AssemblesAlignedReservationsWithoutPayloadBytes) {
  const loom_native_section_contribution_t contributions[] = {
      {
          /*.section_name=*/IREE_SV(".bss"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_WRITE,
          /*.contribution_alignment=*/4,
          /*.contents=*/{},
          /*.reservation_length=*/12,
      },
      {
          /*.section_name=*/IREE_SV(".bss"),
          /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
          /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
              LOOM_NATIVE_SECTION_ACCESS_WRITE,
          /*.contribution_alignment=*/16,
          /*.contents=*/{},
          /*.reservation_length=*/8,
      },
  };

  TestArena arena;
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_ASSERT_OK(loom_native_assemble_section_contributions(
      contributions, IREE_ARRAYSIZE(contributions), &assembly, arena.arena()));

  ASSERT_EQ(assembly.section_count, 1u);
  EXPECT_EQ(assembly.sections[0].storage,
            LOOM_NATIVE_SECTION_STORAGE_RESERVATION);
  EXPECT_EQ(assembly.sections[0].alignment, 16u);
  EXPECT_EQ(assembly.sections[0].contents.data, nullptr);
  EXPECT_EQ(assembly.sections[0].contents.data_length, 0u);
  EXPECT_EQ(assembly.sections[0].reservation_length, 24u);
  ASSERT_EQ(assembly.contribution_layout_count, 2u);
  EXPECT_EQ(assembly.contribution_layouts[0].section_index, 0u);
  EXPECT_EQ(assembly.contribution_layouts[0].section_offset, 0u);
  EXPECT_EQ(assembly.contribution_layouts[1].section_index, 0u);
  EXPECT_EQ(assembly.contribution_layouts[1].section_offset, 16u);
}

TEST(NativeContributionTest, RejectsReservationLayoutOverflow) {
  // Both records are valid reservations. Their combined address-space extent
  // cannot be represented, either by addition or by aligning the second one.
  for (uint64_t alignment : {1u, 8u}) {
    const loom_native_section_contribution_t contributions[] = {
        {
            /*.section_name=*/IREE_SV(".storage"),
            /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
            /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
                LOOM_NATIVE_SECTION_ACCESS_WRITE,
            /*.contribution_alignment=*/1,
            /*.contents=*/{},
            /*.reservation_length=*/UINT64_MAX,
        },
        {
            /*.section_name=*/IREE_SV(".storage"),
            /*.storage=*/LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
            /*.access=*/LOOM_NATIVE_SECTION_ACCESS_READ |
                LOOM_NATIVE_SECTION_ACCESS_WRITE,
            /*.contribution_alignment=*/alignment,
            /*.contents=*/{},
            /*.reservation_length=*/1,
        },
    };
    TestArena arena;
    loom_native_section_contribution_assembly_t assembly = {};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          loom_native_assemble_section_contributions(
                              contributions, IREE_ARRAYSIZE(contributions),
                              &assembly, arena.arena()));
    EXPECT_EQ(assembly.sections, nullptr);
    EXPECT_EQ(assembly.section_count, 0u);
    EXPECT_EQ(assembly.contribution_layouts, nullptr);
    EXPECT_EQ(assembly.contribution_layout_count, 0u);
  }
}

}  // namespace
}  // namespace loom
