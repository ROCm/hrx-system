// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/kernel/launch_config.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;

class KernelLaunchConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* tables = loom_kernel_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_KERNEL, tables, static_cast<uint16_t>(count)));
    tables = loom_index_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_INDEX, tables, static_cast<uint16_t>(count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("launch"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("entry"), &name));
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    const loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    const loom_symbol_ref_t reference = {/*.module_id=*/0,
                                         /*.symbol_id=*/symbol};
    IREE_ASSERT_OK(loom_kernel_def_build(
        &builder_, /*build_flags=*/0, /*retain=*/0, loom_symbol_ref_null(),
        LOOM_STRING_ID_INVALID, /*export_linkage=*/0, reference, &offset, 1,
        /*arg_types=*/nullptr, /*arg_types_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &kernel_));
    loom_builder_set_block(
        &builder_, loom_region_entry_block(loom_kernel_def_body(kernel_)));
    IREE_ASSERT_OK(
        loom_kernel_return_build(&builder_, LOOM_LOCATION_UNKNOWN, &return_));
    loom_block_t* config =
        loom_region_entry_block(loom_kernel_def_config(kernel_));
    loom_builder_set_block(&builder_, config);
    request_ = loom_block_arg_id(config, 0);
    loom_op_t* constant = nullptr;
    IREE_ASSERT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(1), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &constant));
    unit_ = loom_index_constant_result(constant);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void BuildConfig(loom_kernel_launch_config_build_flags_t flags,
                   loom_value_id_t request, loom_op_t** out_config) {
    IREE_ASSERT_OK(loom_kernel_launch_config_build(
        &builder_, flags, unit_, unit_, unit_, unit_, unit_, unit_, unit_,
        unit_, unit_, request, LOOM_LOCATION_UNKNOWN, out_config));
  }

  // Shared storage backing the module and fact tables.
  iree_arena_block_pool_t block_pool_ = {};
  // Registered source dialect metadata.
  loom_context_t context_ = {};
  // Owned module used by the generated builders and launch queries.
  loom_module_t* module_ = nullptr;
  // Builder positioned in the kernel configuration region.
  loom_builder_t builder_ = {};
  // Kernel definition owning the configuration.
  loom_op_t* kernel_ = nullptr;
  // Existing body terminator, retained as a valid insertion boundary.
  loom_op_t* return_ = nullptr;
  // Dynamic offset-typed workload argument.
  loom_value_id_t request_ = LOOM_VALUE_ID_INVALID;
  // Constant index one for dimensions.
  loom_value_id_t unit_ = LOOM_VALUE_ID_INVALID;
};

class KernelLaunchClausesTest : public KernelLaunchConfigTest,
                                public ::testing::WithParamInterface<uint32_t> {
};

TEST_P(KernelLaunchClausesTest, OptionalClausesHaveIndependentPresence) {
  const uint32_t clauses = GetParam();
  constexpr auto kClusterFlags =
      LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_X |
      LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_Y |
      LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_Z;
  const uint32_t flags =
      (clauses & 1 ? kClusterFlags : 0) |
      (clauses & 2
           ? LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_DYNAMIC_WORKGROUP_STORAGE
           : 0);
  loom_op_t* config = nullptr;
  ASSERT_NO_FATAL_FAILURE(BuildConfig(flags, request_, &config));
  DiagnosticEmissionCapture capture;
  IREE_EXPECT_OK(
      loom_kernel_launch_config_verify(module_, config, capture.emitter()));
  EXPECT_TRUE(capture.emissions.empty());
  EXPECT_EQ(loom_kernel_launch_config_has_workgroup_cluster_size(config),
            (clauses & 1) != 0);
  EXPECT_EQ(loom_kernel_launch_config_dynamic_workgroup_storage(config),
            clauses & 2 ? request_ : LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(loom_kernel_launch_config_workgroup_size_x(config), unit_);
}

INSTANTIATE_TEST_SUITE_P(Presence, KernelLaunchClausesTest,
                         ::testing::Values(0u, 1u, 2u, 3u));

class KernelPartialClusterTest
    : public KernelLaunchConfigTest,
      public ::testing::WithParamInterface<uint32_t> {};

TEST_P(KernelPartialClusterTest, RejectsPartialClusterWithIndependentRequest) {
  const uint32_t dimensions = GetParam();
  uint32_t flags =
      LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_X |
      LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_DYNAMIC_WORKGROUP_STORAGE;
  if (dimensions == 2) {
    flags |= LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_WORKGROUP_CLUSTER_SIZE_Y;
  }
  loom_op_t* config = nullptr;
  ASSERT_NO_FATAL_FAILURE(BuildConfig(flags, request_, &config));
  DiagnosticEmissionCapture capture;
  IREE_EXPECT_OK(
      loom_kernel_launch_config_verify(module_, config, capture.emitter()));
  ASSERT_EQ(capture.emissions.size(), 1u);
  EXPECT_EQ(capture.emissions[0].error, LOOM_ERR_STRUCTURE_001);
  ASSERT_EQ(capture.emissions[0].u32_params.size(), 2u);
  EXPECT_EQ(capture.emissions[0].u32_params[0], 7u + dimensions);
  EXPECT_EQ(capture.emissions[0].u32_params[1], 10u);
}

INSTANTIATE_TEST_SUITE_P(Cardinality, KernelPartialClusterTest,
                         ::testing::Values(1u, 2u));

TEST_F(KernelLaunchConfigTest, AbsentStorageResolvesToZero) {
  loom_op_t* config = nullptr;
  ASSERT_NO_FATAL_FAILURE(BuildConfig(0, LOOM_VALUE_ID_INVALID, &config));
  uint64_t bytes = 1;
  EXPECT_TRUE(loom_kernel_def_static_dynamic_workgroup_storage_from_facts(
      module_, kernel_, nullptr, &bytes));
  EXPECT_EQ(bytes, 0u);
}

class KernelStorageConstantTest
    : public KernelLaunchConfigTest,
      public ::testing::WithParamInterface<int64_t> {};

TEST_P(KernelStorageConstantTest, ExplicitConstantKeepsZeroAndWideExtents) {
  const int64_t extent = GetParam();
  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(
      loom_index_constant_build(&builder_, loom_attr_i64(extent),
                                loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                LOOM_LOCATION_UNKNOWN, &constant));
  loom_op_t* config = nullptr;
  ASSERT_NO_FATAL_FAILURE(BuildConfig(
      LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_DYNAMIC_WORKGROUP_STORAGE,
      loom_index_constant_result(constant), &config));
  EXPECT_TRUE(
      loom_kernel_launch_config_dynamic_workgroup_storage_is_present(config));
  uint64_t bytes = 1;
  EXPECT_TRUE(loom_kernel_def_static_dynamic_workgroup_storage_from_facts(
      module_, kernel_, nullptr, &bytes));
  EXPECT_EQ(bytes, static_cast<uint64_t>(extent));
}

INSTANTIATE_TEST_SUITE_P(Extent, KernelStorageConstantTest,
                         ::testing::Values(INT64_C(0), INT64_C(1) << 34));

TEST_F(KernelLaunchConfigTest, PresentUnknownAndNegativeAreNotAbsent) {
  loom_op_t* config = nullptr;
  ASSERT_NO_FATAL_FAILURE(BuildConfig(
      LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_DYNAMIC_WORKGROUP_STORAGE,
      request_, &config));
  uint64_t bytes = 1;
  EXPECT_FALSE(loom_kernel_def_static_dynamic_workgroup_storage_from_facts(
      module_, kernel_, nullptr, &bytes));
  EXPECT_EQ(bytes, 0u);
  loom_value_fact_table_t facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &module_->arena,
                                                  module_->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &facts, request_, loom_value_facts_exact_i64(1536)));
  EXPECT_TRUE(loom_kernel_def_static_dynamic_workgroup_storage_from_facts(
      module_, kernel_, &facts, &bytes));
  EXPECT_EQ(bytes, 1536u);
  IREE_ASSERT_OK(loom_value_fact_table_define(&facts, request_,
                                              loom_value_facts_exact_i64(-1)));
  EXPECT_FALSE(loom_kernel_def_static_dynamic_workgroup_storage_from_facts(
      module_, kernel_, &facts, &bytes));
  EXPECT_EQ(bytes, 0u);
}

TEST_F(KernelLaunchConfigTest, WorkgroupRootRetainsBorrowedEntryFacts) {
  loom_op_t* config = nullptr;
  ASSERT_NO_FATAL_FAILURE(BuildConfig(
      LOOM_KERNEL_LAUNCH_CONFIG_BUILD_FLAG_HAS_DYNAMIC_WORKGROUP_STORAGE,
      request_, &config));
  loom_builder_set_before(&builder_, return_);
  loom_op_t* extent = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(
      &builder_, loom_attr_i64(1536), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      LOOM_LOCATION_UNKNOWN, &extent));
  loom_op_t* root = nullptr;
  IREE_ASSERT_OK(loom_kernel_workgroup_storage_build(
      &builder_, 64, loom_index_constant_result(extent), loom_type_buffer(),
      LOOM_LOCATION_UNKNOWN, &root));
  loom_value_fact_table_t facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &module_->arena,
                                                  module_->values.count));
  facts.context.reference_origin = {
      /*.function_symbol_id=*/loom_kernel_def_callee(kernel_).symbol_id,
      /*.region_index=*/1,
      /*.kind=*/LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY,
      /*.entry_value_id=*/LOOM_VALUE_ID_INVALID,
  };
  loom_value_facts_t operand = loom_value_facts_make(0, 1536, 512);
  loom_value_facts_mark_workgroup_uniform(&operand);
  loom_value_facts_t result = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_kernel_workgroup_storage_facts(&facts.context, module_,
                                                     root, &operand, &result));
  loom_value_fact_buffer_reference_t reference = {};
  ASSERT_TRUE(loom_value_facts_query_buffer_reference(&facts.context, result,
                                                      &reference));
  EXPECT_EQ(reference.maximum_byte_extent.range_lo, 0);
  EXPECT_EQ(reference.maximum_byte_extent.range_hi, 1536);
  EXPECT_EQ(reference.maximum_byte_extent.known_divisor, 512);
  EXPECT_EQ(reference.minimum_alignment, 64u);
  EXPECT_EQ(reference.memory_space, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(reference.root_value_id,
            loom_kernel_workgroup_storage_result(root));
  EXPECT_EQ(reference.alias_scope_id, LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE);
  EXPECT_EQ(reference.origin.kind, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY);
  EXPECT_EQ(reference.origin.entry_value_id, LOOM_VALUE_ID_INVALID);
  EXPECT_TRUE(loom_value_facts_is_workgroup_uniform(result));
}

}  // namespace
}  // namespace loom
