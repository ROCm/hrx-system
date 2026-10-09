// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/types.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/type_registry.h"

namespace {

static const loom_type_reference_key_t kResourceReference = {
    IREE_SVL("test.provider"), IREE_SVL("resource")};
static const loom_type_descriptor_t kResourceDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(8, "test.ref<"),
    /*.ir_kind=*/LOOM_TYPE_DIALECT,
    /*.param_count=*/0,
    /*.fact_domain=*/nullptr,
    /*.semantics=*/
    {
        .semantic = LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE,
        .contract_families = 0,
    },
    /*.format_elements=*/nullptr,
    /*.format_element_count=*/0,
    /*.parameterized=*/nullptr,
    /*.reference=*/&kResourceReference,
};
static const loom_type_registry_entry_t kTypeRegistry[] = {
    {IREE_SV("test.ref"), &kResourceDescriptor},
};

class VMTypesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_type_registry_register_types(
        &context_, kTypeRegistry, IREE_ARRAYSIZE(kTypeRegistry)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("types"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_type_t InternOpaqueType(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    return loom_type_dialect_opaque(name_id);
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
};

TEST_F(VMTypesTest, MapsBuiltinBuffer) {
  const loom_type_reference_key_t* key =
      loom_vm_type_reference_key(module_, loom_type_buffer());
  ASSERT_NE(key, nullptr);
  EXPECT_TRUE(iree_string_view_equal(key->namespace_name, IREE_SV("vm")));
  EXPECT_TRUE(iree_string_view_equal(key->type_name, IREE_SV("buffer")));
}

TEST_F(VMTypesTest, MapsDeclaredManagedReference) {
  EXPECT_EQ(loom_vm_type_reference_key(module_,
                                       InternOpaqueType(IREE_SV("test.ref"))),
            &kResourceReference);
}

TEST_F(VMTypesTest, RejectsTypesWithoutReferenceABI) {
  EXPECT_EQ(loom_vm_type_reference_key(module_,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_I32)),
            nullptr);
  EXPECT_EQ(loom_vm_type_reference_key(
                module_, InternOpaqueType(IREE_SV("test.unknown"))),
            nullptr);
}

}  // namespace
