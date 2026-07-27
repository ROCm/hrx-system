// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/test/registry.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/test/harness.h"
#include "loom/pass/testing/registry_verify.h"

namespace loom {
namespace {

static const loom_pass_descriptor_t* LookupTestPass(iree_string_view_t name) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_OK(
      loom_pass_registry_lookup(loom_test_pass_registry(), name, &descriptor));
  return descriptor;
}

class PassTestRegistryHarness : public PassTestHarness {};

TEST(PassTestRegistryTest, Verifies) {
  IREE_ASSERT_OK(loom_pass_registry_verify(loom_test_pass_registry()));
}

TEST(PassTestRegistryTest, ContainsExpectedPasses) {
  const loom_pass_registry_t* registry = loom_test_pass_registry();
  const iree_string_view_t expected_keys[] = {
      IREE_SV("test.fail"),
      IREE_SV("test.mark-changed"),
      IREE_SV("test.module-noop"),
      IREE_SV("test.noop"),
      IREE_SV("test.options"),
      IREE_SV("test.required"),
      IREE_SV("test.requires-target"),
      IREE_SV("test.resource-lifetime"),
      IREE_SV("test.unavailable"),
  };
  ASSERT_EQ(registry->descriptor_count, IREE_ARRAYSIZE(expected_keys));
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    EXPECT_TRUE(
        iree_string_view_equal(registry->descriptors[i].key, expected_keys[i]))
        << "unexpected test pass registry entry " << i;
  }
}

TEST(PassTestRegistryTest, ValidatesOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("count=4,mode=beta,string=payload")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("count=0")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("mode=gamma")));
}

TEST(PassTestRegistryTest, ValidatesRequiredOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.required"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("required=value")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, iree_string_view_empty()));
}

TEST(PassTestRegistryTest, DecodesOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_ASSERT_OK(loom_pass_descriptor_decode_options(
      descriptor, IREE_SV("count=7,mode=beta,string=payload"), &arena,
      &decoded_options));

  ASSERT_EQ(decoded_options.option_count, 3u);
  EXPECT_EQ(decoded_options.options[0].uint32_value, 7u);
  EXPECT_EQ(decoded_options.options[1].enum_value_index, 1u);
  EXPECT_TRUE(iree_string_view_equal(decoded_options.options[2].string_value,
                                     IREE_SV("payload")));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST_F(PassTestRegistryHarness, RunsThroughToolingInterpreter) {
  loom_module_t* module = AllocateModule();
  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunFlatPipeline(module, IREE_SV("test.module-noop"), &trace));

  EXPECT_EQ(trace.module_noop_invocation_count, 1);
}

TEST_F(PassTestRegistryHarness, FailingPassPropagatesStatusThroughTooling) {
  loom_module_t* module = AllocateModule();
  loom_test_pass_trace_t trace = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        RunFlatPipeline(module, IREE_SV("test.fail"), &trace));
}

TEST_F(PassTestRegistryHarness, UnavailablePassFailsThroughTooling) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.unavailable"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_FALSE(loom_pass_descriptor_is_available(descriptor));

  loom_module_t* module = AllocateModule();
  loom_test_pass_trace_t trace = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      RunFlatPipeline(module, IREE_SV("test.unavailable"), &trace));
}

}  // namespace
}  // namespace loom
