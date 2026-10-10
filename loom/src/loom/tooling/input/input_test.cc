// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/input/input.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"

namespace {

TEST(InputOptionsTest, RejectsUnlinkedSuffixesAndSupportsExplicitText) {
  const loom_input_provider_t* provider = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_input_provider_select({}, {}, IREE_SV("source.cc"), &provider));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_input_provider_select(
                            {}, {}, IREE_SV("source.cxx-test"), &provider));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_input_provider_select({}, IREE_SV("cxx"), {}, &provider));
  IREE_ASSERT_OK(loom_input_provider_select({}, IREE_SV("loom"),
                                            IREE_SV("source.txt"), &provider));
  EXPECT_EQ(provider, &loom_input_text_provider);
  IREE_ASSERT_OK(
      loom_input_provider_select({}, {}, IREE_SV("<stdin>"), &provider));
  EXPECT_EQ(provider, &loom_input_text_provider);
}

TEST(InputOptionsTest, KeepsLanguageOptionsIndependent) {
  const iree_string_view_t entries[] = {IREE_SV("cxx:std=c++20 D=VALUE=7"),
                                        IREE_SV("other:mode=fast")};
  iree_string_view_t options;
  IREE_ASSERT_OK(loom_input_options_for_provider(
      {IREE_ARRAYSIZE(entries), entries}, IREE_SV("cxx"), &options));
  EXPECT_TRUE(iree_string_view_equal(options, IREE_SV("std=c++20 D=VALUE=7")));
  IREE_ASSERT_OK(loom_input_options_for_provider(
      {IREE_ARRAYSIZE(entries), entries}, IREE_SV("loom"), &options));
  EXPECT_TRUE(iree_string_view_is_empty(options));
  const iree_string_view_t duplicates[] = {IREE_SV("cxx:"),
                                           IREE_SV("cxx:std=c++20")};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_input_options_for_provider({IREE_ARRAYSIZE(duplicates), duplicates},
                                      IREE_SV("cxx"), &options));
  const iree_string_view_t malformed[] = {IREE_SV("std=c++20")};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_input_options_for_provider({IREE_ARRAYSIZE(malformed), malformed},
                                      IREE_SV("cxx"), &options));
}

TEST(InputModuleTest, AllowsLogicalLocationsMatchingRemappedMainSource) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  const iree_string_view_t prefix_maps[] = {IREE_SV("/workspace/=")};
  loom_input_request_t request = {
      .source = IREE_SV(R"(
func.def @entry() {
  func.return loc("logical/source.loom":2:3)
} loc("logical/source.loom":1:1 to 3:2)
)"),
      .path = IREE_SV("/workspace/logical/source.loom"),
  };
  request.source_path_options.prefix_maps = {IREE_ARRAYSIZE(prefix_maps),
                                             prefix_maps};
  loom_input_module_t input = {};
  IREE_ASSERT_OK(loom_input_module_load(&loom_input_text_provider, &request,
                                        &context, &block_pool,
                                        iree_allocator_system(), &input));
  ASSERT_NE(input.module, nullptr);
  ASSERT_EQ(input.module->sources.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(input.module->sources.entries[0],
                                     IREE_SV("logical/source.loom")));
  EXPECT_TRUE(iree_string_view_equal(input.module->sources.entries[1],
                                     IREE_SV("logical/source.loom")));
  ASSERT_EQ(input.sources.table.count, 1u);
  EXPECT_EQ(input.sources.table.entries[0].source_id, 0u);
  EXPECT_TRUE(iree_string_view_equal(input.sources.table.entries[0].source,
                                     request.source));

  loom_input_module_deinitialize(&input);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
