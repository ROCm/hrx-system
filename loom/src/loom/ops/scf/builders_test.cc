// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/scf/ops.h"

namespace loom {
namespace {

enum class LoopKind { kCounted, kCondition };

class ScfBuilderTest : public ::testing::TestWithParam<LoopKind> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const auto* tables = loom_scf_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCF,
                                                 tables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("builders"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t Input(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_module_block(module_), type, &value));
    return value;
  }

  iree_status_t Build(loom_value_id_t bound, const loom_value_id_t* initial,
                      uint16_t count, const loom_type_t* header_types,
                      const loom_type_t* result_types, loom_op_t** out_loop) {
    if (GetParam() == LoopKind::kCondition) {
      return loom_scf_while_build(&builder_, initial, count, header_types,
                                  result_types, count, nullptr, 0,
                                  LOOM_LOCATION_UNKNOWN, out_loop);
    }
    return loom_scf_for_build(&builder_, 0, bound, bound, bound, initial, count,
                              result_types, nullptr, 0, LOOM_VALUE_ID_INVALID,
                              LOOM_VALUE_ID_INVALID, 0, 0,
                              LOOM_LOCATION_UNKNOWN, out_loop);
  }

  // Shared backing storage for the module and short-lived remaps.
  iree_arena_block_pool_t pool_;
  // Registered source operation metadata.
  loom_context_t context_;
  // Module owning constructed values and operations.
  loom_module_t* module_ = nullptr;
  // Builder positioned at the fixture's input block.
  loom_builder_t builder_ = {};
};

TEST_P(ScfBuilderTest, ProjectsReservedResultSchemeToEachEntry) {
  const auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto layout_type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  const auto size = Input(index_type);
  const auto layout = Input(layout_type);
  const uint64_t initial_dims[] = {loom_dim_pack_dynamic(size),
                                   loom_dim_pack_dynamic(size),
                                   loom_dim_pack_dynamic(size)};
  loom_type_t view_type = {};  // NOLINT(iree-cpp-designated-initializer) --
                               // Assignment sequencing spans intervening work.
  view_type.header =
      loom_type_make_header(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32, 3, 0);
  view_type.dims[0] = (uint64_t)(uintptr_t)initial_dims;
  view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  view_type.encoding_id = (uint16_t)layout;
  const auto initial_view = Input(view_type);
  const loom_value_id_t initial[] = {initial_view, size, size, layout};

  const uint16_t identity_count = GetParam() == LoopKind::kCondition ? 8 : 4;
  loom_value_id_t identities[8];
  IREE_ASSERT_OK(
      loom_builder_reserve_values(&builder_, identity_count, identities));
  const loom_value_id_t* headers = identities;
  const loom_value_id_t* results =
      identities + (GetParam() == LoopKind::kCondition ? 4 : 0);
  const uint64_t header_dims[] = {loom_dim_pack_dynamic(headers[1]),
                                  loom_dim_pack_dynamic(headers[2]),
                                  loom_dim_pack_dynamic(size)};
  loom_type_t header_view_type = view_type;
  header_view_type.dims[0] = (uint64_t)(uintptr_t)header_dims;
  header_view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  header_view_type.encoding_id = (uint16_t)headers[3];
  const loom_type_t header_types[] = {header_view_type, index_type, index_type,
                                      layout_type};
  const uint64_t result_dims[] = {loom_dim_pack_dynamic(results[1]),
                                  loom_dim_pack_dynamic(results[2]),
                                  loom_dim_pack_dynamic(size)};
  loom_type_t result_view_type = view_type;
  result_view_type.dims[0] = (uint64_t)(uintptr_t)result_dims;
  result_view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  result_view_type.encoding_id = (uint16_t)results[3];
  const loom_type_t types[] = {result_view_type, index_type, index_type,
                               layout_type};
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(Build(size, initial, 4, header_types, types, &loop));
  for (uint16_t i = 0; i < 4; ++i) {
    EXPECT_EQ(loom_op_results(loop)[i], results[i]);
    EXPECT_EQ(loom_value_def_op(loom_module_value(module_, results[i])), loop);
  }
  EXPECT_EQ(builder_.reserved_value_count, 0u);
  for (uint8_t i = 0; i < loop->region_count; ++i) {
    const auto* entry = loom_region_entry_block(loom_op_regions(loop)[i]);
    const uint16_t offset = GetParam() == LoopKind::kCounted ? 1 : 0;
    ASSERT_EQ(entry->arg_count, 4 + offset);
    const auto* arguments = entry->arg_ids + offset;
    const auto type = loom_module_value_type(module_, arguments[0]);
    EXPECT_EQ(loom_type_dim(type, 0), loom_dim_pack_dynamic(arguments[1]));
    EXPECT_EQ(loom_type_dim(type, 1), loom_dim_pack_dynamic(arguments[2]));
    EXPECT_EQ(loom_type_dim(type, 2), loom_dim_pack_dynamic(size));
    EXPECT_EQ(type.encoding_id, arguments[3]);
    for (uint16_t j = 0; j < 4; ++j) {
      EXPECT_NE(arguments[j], results[j]);
    }
  }
  EXPECT_TRUE(loom_type_equal(loom_module_value_type(module_, initial_view),
                              view_type));
}

TEST_P(ScfBuilderTest, InferredTypesPreserveReservedResultOwnership) {
  const auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto initial = Input(index_type);
  loom_value_id_t identities[2] = {};
  const uint16_t identity_count = GetParam() == LoopKind::kCondition ? 2 : 1;
  IREE_ASSERT_OK(
      loom_builder_reserve_values(&builder_, identity_count, identities));
  const loom_value_id_t result = identities[identity_count - 1];
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(Build(initial, &initial, 1, nullptr, nullptr, &loop));
  EXPECT_EQ(loom_op_results(loop)[0], result);
  EXPECT_TRUE(
      loom_type_equal(loom_module_value_type(module_, result), index_type));
  for (uint8_t i = 0; i < loop->region_count; ++i) {
    const auto* entry = loom_region_entry_block(loom_op_regions(loop)[i]);
    for (uint16_t j = 0; j < entry->arg_count; ++j) {
      EXPECT_NE(entry->arg_ids[j], result);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Loops, ScfBuilderTest,
                         ::testing::Values(LoopKind::kCounted,
                                           LoopKind::kCondition));

}  // namespace
}  // namespace loom
