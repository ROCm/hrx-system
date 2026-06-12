// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/builtin_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/pass/testing/registry_verify.h"

namespace loom {
namespace {

static const loom_pass_descriptor_t* LookupBuiltinPass(
    iree_string_view_t name) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_OK(loom_pass_registry_lookup(loom_pass_builtin_registry(), name,
                                           &descriptor));
  return descriptor;
}

static iree_status_t CreateBuiltinPass(const loom_pass_descriptor_t* descriptor,
                                       iree_string_view_t options) {
  iree_arena_block_pool_t block_pool = {};
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t instance_arena = {};
  iree_arena_initialize(&block_pool, &instance_arena);

  loom_pass_t pass = {};
  pass.info = descriptor->info();
  pass.instance_arena = &instance_arena;
  iree_status_t status = descriptor->create ? descriptor->create(&pass, options)
                                            : iree_ok_status();
  if (iree_status_is_ok(status) && descriptor->destroy) {
    descriptor->destroy(&pass);
  }

  iree_arena_deinitialize(&instance_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

TEST(PassBuiltinRegistryTest, BuiltinRegistryVerifies) {
  IREE_ASSERT_OK(loom_pass_registry_verify(loom_pass_builtin_registry()));
}

TEST(PassBuiltinRegistryTest, LookupKnownAndUnknownPasses) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(loom_pass_registry_lookup(
      loom_pass_builtin_registry(), IREE_SV("canonicalize"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(descriptor->key, IREE_SV("canonicalize")));
  ASSERT_NE(descriptor->info, nullptr);
  EXPECT_EQ(descriptor->info()->kind, LOOM_PASS_FUNCTION);
  EXPECT_NE(descriptor->function_run, nullptr);
  EXPECT_NE(descriptor->create, nullptr);

  descriptor = reinterpret_cast<const loom_pass_descriptor_t*>(0x1);
  IREE_ASSERT_OK(loom_pass_registry_lookup(loom_pass_builtin_registry(),
                                           IREE_SV("definitely-not-a-pass"),
                                           &descriptor));
  EXPECT_EQ(descriptor, nullptr);
}

TEST(PassBuiltinRegistryTest, ValidatesBuiltinOptionSchemas) {
  const loom_pass_descriptor_t* canonicalize =
      LookupBuiltinPass(IREE_SV("canonicalize"));
  ASSERT_NE(canonicalize, nullptr);
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      canonicalize, IREE_SV("max-iterations=4")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            canonicalize, IREE_SV("max-iterations=0")));

  const loom_pass_descriptor_t* allocation =
      LookupBuiltinPass(IREE_SV("low-materialize-allocation"));
  ASSERT_NE(allocation, nullptr);
  ASSERT_EQ(allocation->requirement_count, 1u);
  EXPECT_EQ(allocation->requirement_defs[0].capability_type,
            &loom_low_pass_capability_type);
  EXPECT_TRUE(
      iree_string_view_equal(allocation->requirement_defs[0].key,
                             IREE_SV("target.low-descriptor-registry")));
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      allocation, IREE_SV("diagnostics=spills")));
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      allocation, IREE_SV("budgets=vm.i32=2;vm.ref=1")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            allocation, IREE_SV("diagnostics=verbose")));
  IREE_ASSERT_OK(CreateBuiltinPass(
      allocation, IREE_SV("spill-storage-spaces=stack;private")));
  IREE_ASSERT_OK(
      CreateBuiltinPass(allocation, IREE_SV("spill-storage-spaces=all")));
  IREE_ASSERT_OK(
      CreateBuiltinPass(allocation, IREE_SV("spill-storage-spaces=none")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateBuiltinPass(allocation,
                        IREE_SV("spill-storage-spaces=private;private")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateBuiltinPass(allocation, IREE_SV("spill-storage-spaces=scratch;")));

  const loom_pass_descriptor_t* operand_forms =
      LookupBuiltinPass(IREE_SV("low-select-operand-forms"));
  ASSERT_NE(operand_forms, nullptr);
  ASSERT_EQ(operand_forms->requirement_count, 1u);
  EXPECT_EQ(operand_forms->requirement_defs[0].capability_type,
            &loom_low_pass_capability_type);
  EXPECT_TRUE(
      iree_string_view_equal(operand_forms->requirement_defs[0].key,
                             IREE_SV("target.low-descriptor-registry")));

  const loom_pass_descriptor_t* low_dce = LookupBuiltinPass(IREE_SV("low-dce"));
  ASSERT_NE(low_dce, nullptr);
  ASSERT_EQ(low_dce->requirement_count, 1u);
  EXPECT_EQ(low_dce->requirement_defs[0].capability_type,
            &loom_low_pass_capability_type);
  EXPECT_TRUE(
      iree_string_view_equal(low_dce->requirement_defs[0].key,
                             IREE_SV("target.low-descriptor-registry")));

  const loom_pass_descriptor_t* source_to_low =
      LookupBuiltinPass(IREE_SV("source-to-low"));
  ASSERT_NE(source_to_low, nullptr);
  ASSERT_EQ(source_to_low->requirement_count, 2u);
  EXPECT_EQ(source_to_low->requirement_defs[0].capability_type,
            &loom_low_pass_capability_type);
  EXPECT_TRUE(
      iree_string_view_equal(source_to_low->requirement_defs[0].key,
                             IREE_SV("target.low-descriptor-registry")));
  EXPECT_TRUE(
      iree_string_view_equal(source_to_low->requirement_defs[1].key,
                             IREE_SV("target.low-lower-policy-registry")));
  EXPECT_EQ(source_to_low->requirement_defs[1].capability_type,
            &loom_low_pass_capability_type);
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      source_to_low, IREE_SV("max-errors=0")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            source_to_low, IREE_SV("max-errors=-1")));
}

}  // namespace
}  // namespace loom
