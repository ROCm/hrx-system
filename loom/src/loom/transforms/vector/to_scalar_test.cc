// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar.h"

#include <stdint.h>

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/contract.h"
#include "loom/analysis/matrix_fragment_layout.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class VectorToScalarTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_module_t* Parse(iree_string_view_t source) {
    const loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("to_scalar_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    modules_.push_back(module);
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  std::vector<loom_module_t*> modules_;
};

static const loom_matrix_fragment_layout_t kTinyDistributedMmaLayout = {
    /*.kind=*/1,
    /*.name=*/IREE_SVL("test.tiny.distributed.mma"),
    /*.wave_size=*/2,
    /*.tile_shape=*/
    {
        /*.result_row_count=*/2,
        /*.result_column_count=*/2,
        /*.reduction_count=*/2,
    },
    /*.lhs=*/
    {
        /*.role=*/LOOM_CONTRACT_OPERAND_ROLE_LHS,
        /*.map_kind=*/LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION,
        /*.register_count=*/2,
        /*.elements_per_register=*/1,
        /*.element_bit_count=*/16,
        /*.coordinate_flags=*/LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
            LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION,
    },
    /*.rhs=*/
    {
        /*.role=*/LOOM_CONTRACT_OPERAND_ROLE_RHS,
        /*.map_kind=*/
        LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION,
        /*.register_count=*/2,
        /*.elements_per_register=*/1,
        /*.element_bit_count=*/16,
        /*.coordinate_flags=*/LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN |
            LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION,
    },
    /*.accumulator=*/
    {
        /*.role=*/LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
        /*.map_kind=*/LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN,
        /*.register_count=*/2,
        /*.elements_per_register=*/1,
        /*.element_bit_count=*/32,
        /*.coordinate_flags=*/LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
            LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN,
    },
    /*.result=*/
    {
        /*.role=*/LOOM_CONTRACT_OPERAND_ROLE_RESULT,
        /*.map_kind=*/LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN,
        /*.register_count=*/2,
        /*.elements_per_register=*/1,
        /*.element_bit_count=*/32,
        /*.coordinate_flags=*/LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
            LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN,
    },
};

static loom_op_t* FindFirstOp(loom_region_t* region, loom_op_kind_t kind) {
  loom_block_t* block = nullptr;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) {
        return op;
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint16_t i = 0; i < op->region_count; ++i) {
        loom_op_t* nested_op = FindFirstOp(regions[i], kind);
        if (nested_op != nullptr) {
          return nested_op;
        }
      }
    }
  }
  return nullptr;
}

static uint32_t CountOps(loom_region_t* region, loom_op_kind_t kind) {
  uint32_t count = 0;
  loom_block_t* block = nullptr;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) {
        ++count;
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint16_t i = 0; i < op->region_count; ++i) {
        count += CountOps(regions[i], kind);
      }
    }
  }
  return count;
}

static void ExpectModuleVerifies(const loom_module_t* module) {
  const loom_verify_options_t verify_options = {
      /*.sink=*/{},
      /*.max_errors=*/20,
  };
  loom_verify_result_t verify_result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &verify_options, &verify_result));
  EXPECT_EQ(verify_result.error_count, 0u);
}

TEST_F(VectorToScalarTest, TargetFragmentLayoutEnablesDistributedMmaFallback) {
  loom_module_t* module = Parse(
      IREE_SV("kernel.def @physical_mma() {\n"
              "  %c1 = index.constant 1 : index\n"
              "  %c2 = index.constant 2 : index\n"
              "  kernel.launch.config workgroups(%c1, %c1, %c1) "
              "workgroup_size(%c2, %c1, %c1) : index\n"
              "} launch(%lhs_data: vector<2xf16>, %rhs_data: vector<2xf16>, "
              "%init_data: vector<2xf32>) {\n"
              "  %m = index.constant 2 : index\n"
              "  %n = index.constant 2 : index\n"
              "  %k = index.constant 2 : index\n"
              "  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] : "
              "vector<2xf16>\n"
              "  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] : "
              "vector<2xf16>\n"
              "  %init = vector.fragment<init> %init_data shape [%m, %n] : "
              "vector<2xf32>\n"
              "  %result = vector.mma %lhs, %rhs, %init : vector<2xf16>, "
              "vector<2xf16>, vector<2xf32>\n"
              "  kernel.return\n"
              "}\n"));
  ExpectModuleVerifies(module);

  loom_op_t* kernel_op = loom_block_op(loom_module_block(module), 0);
  ASSERT_NE(kernel_op, nullptr);
  ASSERT_TRUE(loom_kernel_def_isa(kernel_op));
  loom_func_like_t function = loom_func_like_cast(module, kernel_op);
  loom_op_t* mma_op =
      FindFirstOp(loom_kernel_def_body(kernel_op), LOOM_OP_VECTOR_MMA);
  ASSERT_NE(mma_op, nullptr);

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_fact_owner = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_fact_owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_fact_owner, module, loom_pass_value_fact_scope_function(function),
      &facts));
  ASSERT_NE(facts, nullptr);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module, &pass_arena));
  IREE_ASSERT_OK(loom_rewriter_enable_analysis(&rewriter, function, facts));

  const loom_pass_info_t* pass_info = loom_vector_to_scalar_pass_info();
  std::vector<uint8_t> statistic_storage(
      pass_info->statistic_layout->storage_size, 0);
  loom_pass_t pass = {};
  pass.info = pass_info;
  pass.arena = &pass_arena;
  pass.statistic_storage = statistic_storage.data();

  const loom_vector_mma_to_scalar_options_t empty_options =
      loom_vector_mma_to_scalar_options_empty();
  EXPECT_TRUE(
      iree_any_bit_set(loom_vector_mma_to_scalar_reference_rejection_bits(
                           &pass, &rewriter, mma_op, empty_options),
                       LOOM_CONTRACT_REJECTION_FRAGMENT));

  const loom_vector_mma_to_scalar_options_t distributed_options = {
      /*.matrix_fragment_layout=*/&kTinyDistributedMmaLayout,
      /*.flags=*/LOOM_VECTOR_TO_SCALAR_FLAG_ALLOW_SUBGROUP_COMMUNICATION,
  };
  EXPECT_EQ(loom_vector_mma_to_scalar_reference_rejection_bits(
                &pass, &rewriter, mma_op, distributed_options),
            LOOM_CONTRACT_REJECTION_NONE);

  bool rewritten = false;
  IREE_ASSERT_OK(loom_vector_mma_to_scalar_rewrite_op(
      &pass, &rewriter, mma_op, distributed_options, &rewritten));
  EXPECT_TRUE(rewritten);
  EXPECT_EQ(CountOps(module->body, LOOM_OP_VECTOR_MMA), 0u);
  EXPECT_GT(CountOps(module->body, LOOM_OP_KERNEL_SUBGROUP_LANE_ID), 0u);
  EXPECT_EQ(CountOps(module->body, LOOM_OP_KERNEL_SUBGROUP_BROADCAST), 8u);
  EXPECT_EQ(CountOps(module->body, LOOM_OP_SCF_SELECT), 0u);
  ExpectModuleVerifies(module);

  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&value_fact_owner);
  iree_arena_deinitialize(&pass_arena);
}

}  // namespace
}  // namespace loom
