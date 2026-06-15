// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/object.h"

#include <limits>

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

TEST(NativeObjectTest, ResolvesSymbolsAndFixupsThroughContributionLayout) {
  const uint8_t first_text[] = {0x55, 0x48, 0x89, 0xe5};
  const uint8_t rodata[] = {0x2a};
  const uint8_t second_text[] = {0xc5, 0xf8, 0x77, 0xc3};
  const loom_native_section_contribution_t sections[] = {
      {
          /*.section_name=*/IREE_SV(".text"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          /*.contribution_alignment=*/4,
          /*.entry_size=*/{},
          /*.link=*/{},
          /*.info=*/{},
          /*.contents=*/
          iree_make_const_byte_span(first_text, sizeof(first_text)),
      },
      {
          /*.section_name=*/IREE_SV(".rodata"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
          /*.contribution_alignment=*/1,
          /*.entry_size=*/{},
          /*.link=*/{},
          /*.info=*/{},
          /*.contents=*/iree_make_const_byte_span(rodata, sizeof(rodata)),
      },
      {
          /*.section_name=*/IREE_SV(".text"),
          /*.section_type=*/LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
          /*.section_flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
              LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
          /*.contribution_alignment=*/8,
          /*.entry_size=*/{},
          /*.link=*/{},
          /*.info=*/{}, /*.contents=*/
          iree_make_const_byte_span(second_text, sizeof(second_text)),
      },
  };

  TestArena arena;
  loom_native_section_contribution_assembly_t assembly = {0};
  IREE_ASSERT_OK(loom_native_assemble_section_contributions(
      sections, IREE_ARRAYSIZE(sections), &assembly, arena.arena()));

  const loom_native_object_symbol_t symbols[] = {
      {
          /*.name=*/IREE_SV("entry"),
          /*.section_contribution_index=*/0,
          /*.section_offset=*/1,
          /*.size=*/3,
          /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
          /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
          /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
      },
      {
          /*.name=*/IREE_SV("second"),
          /*.section_contribution_index=*/2,
          /*.section_offset=*/2,
          /*.size=*/2,
          /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL,
          /*.visibility=*/LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN,
          /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
      },
  };
  loom_native_object_symbol_layout_t layouts[IREE_ARRAYSIZE(symbols)] = {};
  IREE_ASSERT_OK(loom_native_object_resolve_symbol_layouts(
      symbols, IREE_ARRAYSIZE(symbols), assembly.contribution_layouts,
      assembly.contribution_layout_count, layouts));

  EXPECT_EQ(layouts[0].section_index, 0u);
  EXPECT_EQ(layouts[0].section_offset, 1u);
  EXPECT_EQ(layouts[1].section_index, 0u);
  EXPECT_EQ(layouts[1].section_offset, 10u);

  const loom_native_object_fixup_t fixups[] = {{
      /*.section_contribution_index=*/2,
      /*.section_offset=*/3,
      /*.relocation_kind=*/1,
      /*.target_symbol_index=*/0,
      /*.addend=*/-4,
  }};
  loom_native_object_fixup_layout_t fixup_layouts[IREE_ARRAYSIZE(fixups)] = {};
  IREE_ASSERT_OK(loom_native_object_resolve_fixup_layouts(
      fixups, IREE_ARRAYSIZE(fixups), IREE_ARRAYSIZE(symbols),
      assembly.contribution_layouts, assembly.contribution_layout_count,
      fixup_layouts));
  EXPECT_EQ(fixup_layouts[0].section_index, 0u);
  EXPECT_EQ(fixup_layouts[0].section_offset, 11u);
}

TEST(NativeObjectTest, RejectsMissingSymbolName) {
  const loom_native_section_contribution_layout_t section_layouts[] = {{
      /*.section_index=*/0,
      /*.section_offset=*/0,
  }};
  const loom_native_object_symbol_t symbol = {
      /*.name=*/iree_string_view_empty(),
      /*.section_contribution_index=*/0,
      /*.section_offset=*/{},
      /*.size=*/{},
      /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      /*.visibility=*/{},
      /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  loom_native_object_symbol_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_native_object_resolve_symbol_layouts(
                            &symbol, 1, section_layouts,
                            IREE_ARRAYSIZE(section_layouts), &layout));
}

TEST(NativeObjectTest, RejectsInvalidSectionContributionIndex) {
  const loom_native_section_contribution_layout_t section_layouts[] = {{
      /*.section_index=*/0,
      /*.section_offset=*/0,
  }};
  const loom_native_object_symbol_t symbol = {
      /*.name=*/IREE_SV("bad"),
      /*.section_contribution_index=*/1,
      /*.section_offset=*/{},
      /*.size=*/{},
      /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      /*.visibility=*/{},
      /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  loom_native_object_symbol_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_native_object_resolve_symbol_layouts(
                            &symbol, 1, section_layouts,
                            IREE_ARRAYSIZE(section_layouts), &layout));
}

TEST(NativeObjectTest, RejectsOffsetOverflow) {
  const loom_native_section_contribution_layout_t section_layouts[] = {{
      /*.section_index=*/0,
      /*.section_offset=*/std::numeric_limits<uint64_t>::max(),
  }};
  const loom_native_object_symbol_t symbol = {
      /*.name=*/IREE_SV("overflow"),
      /*.section_contribution_index=*/0,
      /*.section_offset=*/1,
      /*.size=*/{},
      /*.binding=*/LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      /*.visibility=*/{},
      /*.kind=*/LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  loom_native_object_symbol_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_native_object_resolve_symbol_layouts(
                            &symbol, 1, section_layouts,
                            IREE_ARRAYSIZE(section_layouts), &layout));
}

TEST(NativeObjectTest, RejectsFixupWithoutRelocationKind) {
  const loom_native_section_contribution_layout_t section_layouts[] = {{
      /*.section_index=*/0,
      /*.section_offset=*/0,
  }};
  const loom_native_object_fixup_t fixup = {
      /*.section_contribution_index=*/0,
      /*.section_offset=*/{},
      /*.relocation_kind=*/0,
      /*.target_symbol_index=*/0,
  };
  loom_native_object_fixup_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_native_object_resolve_fixup_layouts(
                            &fixup, 1, 1, section_layouts,
                            IREE_ARRAYSIZE(section_layouts), &layout));
}

TEST(NativeObjectTest, RejectsFixupInvalidTargetSymbol) {
  const loom_native_section_contribution_layout_t section_layouts[] = {{
      /*.section_index=*/0,
      /*.section_offset=*/0,
  }};
  const loom_native_object_fixup_t fixup = {
      /*.section_contribution_index=*/0,
      /*.section_offset=*/{},
      /*.relocation_kind=*/1,
      /*.target_symbol_index=*/1,
  };
  loom_native_object_fixup_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_native_object_resolve_fixup_layouts(
                            &fixup, 1, 1, section_layouts,
                            IREE_ARRAYSIZE(section_layouts), &layout));
}

}  // namespace
}  // namespace loom
