// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/input_contract.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class InputContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("inputs"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Allocate values before arranging them in signature order. This exercises
  // the source coordinate contract independently of SSA allocation order.
  std::array<loom_value_id_t, 3> Signature(loom_type_kind_t kind,
                                           bool specialized) {
    loom_block_t* block = nullptr;
    IREE_CHECK_OK(loom_module_allocate_block(module_, &block));
    loom_value_id_t width, encoding, shaped;
    IREE_CHECK_OK(
        loom_module_define_value(module_, loom_type_encoding(), &encoding));
    IREE_CHECK_OK(loom_module_define_value(
        module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &width));
    IREE_CHECK_OK(loom_block_add_arg(module_, block, width));
    IREE_CHECK_OK(loom_block_add_arg(module_, block, encoding));
    auto type = loom_type_shaped_2d(
        kind,
        kind == LOOM_TYPE_GROUP ? LOOM_SCALAR_TYPE_NONE : LOOM_SCALAR_TYPE_F32,
        specialized ? loom_dim_pack_static(16) : loom_dim_pack_dynamic(width),
        loom_dim_pack_dynamic(LOOM_VALUE_ID_INVALID), 0);
    if (!specialized && kind != LOOM_TYPE_GROUP && kind != LOOM_TYPE_VECTOR) {
      type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
      type.encoding_id = encoding;
    }
    IREE_CHECK_OK(loom_module_define_value(module_, type, &shaped));
    IREE_CHECK_OK(loom_block_add_arg(module_, block, shaped));
    return {width, encoding, shaped};
  }

  std::vector<loom_input_requirement_t> Project(
      const std::array<loom_value_id_t, 3>& arguments,
      const loom_value_id_t* inherited = nullptr) {
    const auto count = loom_input_contract_project(
        module_, arguments.data(), arguments.size(), inherited, nullptr);
    std::vector<loom_input_requirement_t> requirements(count);
    EXPECT_EQ(
        loom_input_contract_project(module_, arguments.data(), arguments.size(),
                                    inherited, requirements.data()),
        count);
    return requirements;
  }

  iree_arena_block_pool_t pool_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
};

TEST_F(InputContractTest, SourcePositionsAndInheritedObligations) {
  for (auto kind : {LOOM_TYPE_GROUP, LOOM_TYPE_TENSOR, LOOM_TYPE_TILE,
                    LOOM_TYPE_VECTOR, LOOM_TYPE_VIEW}) {
    SCOPED_TRACE(kind);
    const auto family = Signature(kind, false);
    const auto identical = Signature(kind, false);
    const auto specialized = Signature(kind, true);
    const bool has_encoding =
        kind != LOOM_TYPE_GROUP && kind != LOOM_TYPE_VECTOR;
    const auto obligations = Project(family);
    ASSERT_EQ(obligations.size(), has_encoding ? 2 : 1);
    EXPECT_EQ(obligations[0].argument, 2);
    EXPECT_EQ(obligations[0].axis, 0);
    EXPECT_TRUE(obligations[0].expected_is_argument);
    EXPECT_EQ(obligations[0].expected, 0);
    if (has_encoding) {
      EXPECT_EQ(obligations[1].argument, 2);
      EXPECT_EQ(obligations[1].axis, UINT8_MAX);
      EXPECT_TRUE(obligations[1].expected_is_argument);
      EXPECT_EQ(obligations[1].expected, 1);
    }
    EXPECT_TRUE(Project(identical, family.data()).empty());
    const auto additional = Project(specialized, family.data());
    ASSERT_EQ(additional.size(), has_encoding ? 2 : 1);
    EXPECT_EQ(additional[0].argument, 2);
    EXPECT_EQ(additional[0].axis, 0);
    EXPECT_FALSE(additional[0].expected_is_argument);
    EXPECT_EQ(additional[0].expected, 16);
    if (has_encoding) {
      EXPECT_EQ(additional[1].axis, UINT8_MAX);
      EXPECT_FALSE(additional[1].expected_is_argument);
      EXPECT_EQ(additional[1].expected, 0);
    }
  }
}

TEST_F(InputContractTest, MaximumRankPreservesEveryAxis) {
  alignas(loom_input_shape_t)
      uint8_t storage[sizeof(loom_input_shape_t) +
                      LOOM_TYPE_MAX_RANK * sizeof(uint64_t)] = {};
  auto* shape = reinterpret_cast<loom_input_shape_t*>(storage);
  shape->kind = LOOM_TYPE_VECTOR;
  shape->rank = LOOM_TYPE_MAX_RANK;
  shape->dynamic_axes = UINT16_MAX;
  for (uint8_t axis = 0; axis < shape->rank; ++axis) {
    shape->dimensions[axis] = axis + 1;
  }
  std::array<loom_input_requirement_t, LOOM_TYPE_MAX_RANK> requirements;
  ASSERT_EQ(
      loom_input_shape_requirements(shape, nullptr, 31, requirements.data()),
      LOOM_TYPE_MAX_RANK);
  for (uint8_t axis = 0; axis < shape->rank; ++axis) {
    EXPECT_EQ(requirements[axis].argument, 31);
    EXPECT_EQ(requirements[axis].axis, axis);
    EXPECT_TRUE(requirements[axis].expected_is_argument);
    EXPECT_EQ(requirements[axis].expected, axis);
  }
}

}  // namespace
}  // namespace loom
