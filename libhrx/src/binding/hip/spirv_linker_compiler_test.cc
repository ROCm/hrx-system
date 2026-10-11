// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/spirv_linker_compiler.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(SpirvLinkerCompilerTest, CompilesAndLinksForGfx942) {
  static constexpr uint32_t kNoopSpirv[] = {
      0x07230203u, 0x00010000u, 0x00000000u, 0x00000005u, 0x00000000u,
      0x00020011u, 0x00000004u, 0x00020011u, 0x00000006u, 0x0003000eu,
      0x00000002u, 0x00000002u, 0x0007000fu, 0x00000006u, 0x00000003u,
      0x5f787268u, 0x6b6e696cu, 0x6e5f6465u, 0x00706f6fu, 0x00060010u,
      0x00000003u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u,
      0x00020013u, 0x00000001u, 0x00030021u, 0x00000002u, 0x00000001u,
      0x00050036u, 0x00000001u, 0x00000003u, 0x00000000u, 0x00000002u,
      0x000200f8u, 0x00000004u, 0x000100fdu, 0x00010038u,
  };
  const iree_hip_spirv_linker_input_t input = {
      .data = iree_make_const_byte_span(kNoopSpirv, sizeof(kNoopSpirv)),
      .name = "linked_noop.spv",
  };
  iree_byte_span_t executable = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_hip_spirv_linker_compile(
      iree_make_cstring_view("amdgcn-amd-amdhsa--gfx942"), 1, &input,
      /*option_count=*/0, /*options=*/nullptr, iree_allocator_system(),
      &executable));
  EXPECT_NE(nullptr, executable.data);
  EXPECT_GT(executable.data_length, 0u);
  iree_allocator_free(iree_allocator_system(), executable.data);
}

}  // namespace
