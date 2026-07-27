// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for production dialect hook profiles layered on the structured IR
// generator. Core generator tests stay synthetic; this file opts into each
// production dialect profile explicitly.

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/testing/gen.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

static bool block_contains_dialect(const loom_block_t* block,
                                   loom_dialect_id_t dialect_id) {
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_op_dialect_id(op->kind) == dialect_id) {
      return true;
    }
    loom_region_t* const* regions = loom_op_regions(op);
    for (uint8_t region_index = 0; region_index < op->region_count;
         ++region_index) {
      const loom_region_t* region = regions[region_index];
      for (iree_host_size_t block_index = 0; block_index < region->block_count;
           ++block_index) {
        if (block_contains_dialect(loom_region_const_block(region, block_index),
                                   dialect_id)) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool module_contains_dialect(loom_module_t* module,
                                    loom_dialect_id_t dialect_id) {
  return block_contains_dialect(loom_module_block(module), dialect_id);
}

class GenDialectProfilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    register_test_dialect();
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void register_test_dialect() {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
  }

  void register_scalar_dialect() {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_scalar_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, vtables, (uint16_t)vtable_count));
  }

  void register_vector_dialect() {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_vector_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_VECTOR, vtables, (uint16_t)vtable_count));
  }

  iree_status_t generate_profile_module(uint64_t seed,
                                        const loom_test_gen_op_hook_t* hooks,
                                        iree_host_size_t hook_count,
                                        loom_module_t** out_module) {
    loom_test_gen_body_config_t body =
        loom_test_gen_body_config_representative(1);
    IREE_ASSERT(hook_count <= IREE_ARRAYSIZE(body.hooks));
    for (iree_host_size_t hook_index = 0; hook_index < hook_count;
         ++hook_index) {
      body.hooks[hook_index] = hooks[hook_index];
    }
    body.hook_count = hook_count;
    body.op_count = 30;
    body.block_arg_count = 0;

    loom_test_gen_module_config_t config = {0};
    config.function_count = 1;
    config.body_config = body;

    loom_test_gen_t gen;
    loom_test_gen_initialize_seeded(seed, &gen);
    IREE_RETURN_IF_ERROR(loom_test_gen_module(&gen, &config, &context_,
                                              &block_pool_, out_module));

    loom_verify_options_t options = {};
    options.sink = {loom_diagnostic_stderr_sink, NULL};
    loom_verify_result_t result = {};
    IREE_RETURN_IF_ERROR(loom_verify_module(*out_module, &options, &result));
    if (result.error_count > 0) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "generated profile module failed verification with %d errors",
          result.error_count);
    }
    return iree_ok_status();
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(GenDialectProfilesTest, ScalarProfileEmitsScalarDialectOps) {
  register_scalar_dialect();
  IREE_ASSERT_OK(loom_context_finalize(&context_));

  iree_host_size_t hook_count = 0;
  const loom_test_gen_op_hook_t* hooks =
      loom_test_gen_scalar_hooks(&hook_count);
  ASSERT_GT(hook_count, 0u);

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_profile_module(550, hooks, hook_count, &module));
  EXPECT_TRUE(module_contains_dialect(module, LOOM_DIALECT_SCALAR));
  loom_module_free(module);
}

TEST_F(GenDialectProfilesTest, VectorProfileEmitsVectorDialectOps) {
  register_vector_dialect();
  IREE_ASSERT_OK(loom_context_finalize(&context_));

  iree_host_size_t hook_count = 0;
  const loom_test_gen_op_hook_t* hooks =
      loom_test_gen_vector_hooks(&hook_count);
  ASSERT_GT(hook_count, 0u);

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_profile_module(551, hooks, hook_count, &module));
  EXPECT_TRUE(module_contains_dialect(module, LOOM_DIALECT_VECTOR));
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
