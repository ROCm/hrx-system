// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/descriptors/entry.h"

#include <string>

#include "iree/testing/gtest.h"
#include "loom/target/arch/x86/descriptors/avx2_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx2_features_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_descriptors.h"
#include "loom/target/arch/x86/descriptors/avx512_features_descriptors.h"
#include "loom/target/arch/x86/descriptors/scalar_descriptors.h"

namespace loom {
namespace {

TEST(EntryDescriptorsTest, OrdinalsResolveInEachRepresentationView) {
  struct Instruction {
    // Field of the compact entry vocabulary used by the materializer.
    uint32_t loom_x86_entry_descriptors_t::* ordinal;
    // Semantic identity that must survive view-local ordinal renumbering.
    iree_string_view_t key;
  };
  const Instruction instructions[] = {
      {&loom_x86_entry_descriptors_t::load_u8,
       IREE_SV("x86.scalar.movzx.load.u8.gpr32")},
      {&loom_x86_entry_descriptors_t::load_u16,
       IREE_SV("x86.scalar.movzx.load.u16.gpr32")},
      {&loom_x86_entry_descriptors_t::load_u32,
       IREE_SV("x86.scalar.mov.load.gpr32")},
      {&loom_x86_entry_descriptors_t::load_u64,
       IREE_SV("x86.scalar.mov.load.gpr64")},
      {&loom_x86_entry_descriptors_t::widen_u32,
       IREE_SV("x86.scalar.movzx.gpr64.gpr32")},
      {&loom_x86_entry_descriptors_t::constant_u32,
       IREE_SV("x86.scalar.movimm.gpr32")},
      {&loom_x86_entry_descriptors_t::constant_u64,
       IREE_SV("x86.scalar.movimm.gpr64")},
      {&loom_x86_entry_descriptors_t::compare_uge_u32,
       IREE_SV("x86.scalar.cmp.uge.imm.gpr32")},
      {&loom_x86_entry_descriptors_t::symbol_address,
       IREE_SV("x86.scalar.lea.symbol.gpr64")},
  };
  for (const auto* set : {loom_x86_scalar_core_descriptor_set(),
                          loom_x86_avx2_core_descriptor_set(),
                          loom_x86_avx2_features_core_descriptor_set(),
                          loom_x86_avx512_core_descriptor_set(),
                          loom_x86_avx512_features_core_descriptor_set()}) {
    const auto* entry = loom_x86_entry_descriptors(set);
    ASSERT_NE(entry, nullptr);
    for (const auto& instruction : instructions) {
      SCOPED_TRACE(std::string(instruction.key.data, instruction.key.size));
      EXPECT_EQ(
          entry->*instruction.ordinal,
          loom_low_descriptor_set_lookup_descriptor(set, instruction.key));
    }
  }
}

}  // namespace
}  // namespace loom
