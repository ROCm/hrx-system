// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/runtime_requirements.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/test/ops.h"

namespace {

enum class SiteKind {
  kKernelAssert,
  kAssertValue,
  kAssertOp,
  kAssertLayout,
  kAssertAccess,
  kAssertAccesses,
  kRaceAccess,
  kRaceFragmentAccess,
  kRaceSync,
};

struct SiteRequirementCase {
  SiteKind kind;
  loom_sanitizer_runtime_requirements_t default_requirements;
  loom_sanitizer_runtime_requirements_t trap_requirements;
};

class SanitizerRuntimeRequirementsTest
    : public ::testing::TestWithParam<SiteRequirementCase> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SANITIZER, loom_sanitizer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, /*source_resolver=*/NULL,
                                        iree_allocator_system(), &module_));

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder, IREE_SV("entry"),
                                              &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_op_t* function = NULL;
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
        (loom_symbol_ref_t){/*module_id=*/0, /*symbol_id=*/symbol_id},
        /*arg_types=*/NULL, /*arg_types_count=*/0, /*result_types=*/NULL,
        /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
        /*predicates=*/NULL, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function));
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_test_func_body(function)), &builder_);
    builder_.ip.parent_op = function;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  template <typename VtableFn>
  void RegisterDialect(uint8_t dialect, VtableFn vtable_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = vtable_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect, vtables,
                                                 (uint16_t)count));
  }

  loom_value_id_t BuildConstant(int64_t value, loom_scalar_type_t type) {
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_test_constant_build(&builder_, loom_attr_i64(value),
                                           loom_type_scalar(type),
                                           LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_constant_result(op);
  }

  loom_value_id_t BuildConvertedValue(loom_value_id_t input,
                                      loom_type_t result_type) {
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_test_convert_build(&builder_, input, result_type,
                                          LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_convert_result(op);
  }

  loom_predicate_t MakeRangePredicate(loom_value_id_t value) {
    loom_predicate_t predicate = {.kind = LOOM_PREDICATE_RANGE, .arg_count = 3};
    predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
    predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
    predicate.arg_tags[2] = LOOM_PRED_ARG_CONST;
    predicate.args[0] = (int64_t)value;
    predicate.args[1] = 0;
    predicate.args[2] = 16;
    return predicate;
  }

  void BuildSite(SiteKind kind) {
    const loom_value_id_t index = BuildConstant(0, LOOM_SCALAR_TYPE_INDEX);
    const loom_value_id_t condition = BuildConstant(1, LOOM_SCALAR_TYPE_I1);
    const loom_type_t view_type = loom_type_shaped_1d(
        LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
    const loom_value_id_t view = BuildConvertedValue(index, view_type);
    const int64_t static_indices[] = {0};

    loom_op_t* op = NULL;
    switch (kind) {
      case SiteKind::kKernelAssert:
        IREE_ASSERT_OK(loom_kernel_assert_build(
            &builder_, /*build_flags=*/0, condition, LOOM_STRING_ID_INVALID,
            LOOM_LOCATION_UNKNOWN, &op));
        break;
      case SiteKind::kAssertValue: {
        const loom_predicate_t predicate = MakeRangePredicate(index);
        const loom_type_t result_type =
            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
        IREE_ASSERT_OK(loom_sanitizer_assert_value_build(
            &builder_, &index, 1, &predicate, 1, &result_type, 1,
            LOOM_LOCATION_UNKNOWN, &op));
        break;
      }
      case SiteKind::kAssertOp: {
        const loom_predicate_t predicate = MakeRangePredicate(index);
        IREE_ASSERT_OK(loom_sanitizer_assert_op_build(
            &builder_, &index, 1, &predicate, 1, LOOM_LOCATION_UNKNOWN, &op));
        break;
      }
      case SiteKind::kAssertLayout:
        IREE_ASSERT_OK(loom_sanitizer_assert_layout_build(
            &builder_, view, view_type, LOOM_LOCATION_UNKNOWN, &op));
        break;
      case SiteKind::kAssertAccess:
        IREE_ASSERT_OK(loom_sanitizer_assert_access_build(
            &builder_, /*build_flags=*/0,
            LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ, view, /*indices=*/NULL,
            /*indices_count=*/0, static_indices, IREE_ARRAYSIZE(static_indices),
            /*static_extents=*/NULL, /*static_extents_count=*/0,
            LOOM_LOCATION_UNKNOWN, &op));
        break;
      case SiteKind::kAssertAccesses: {
        const int64_t static_extents[] = {1};
        const int64_t static_strides[] = {1};
        IREE_ASSERT_OK(loom_sanitizer_assert_accesses_build(
            &builder_, LOOM_SANITIZER_ASSERT_ACCESSES_KIND_WRITE, view,
            /*indices=*/NULL, /*indices_count=*/0, static_indices,
            IREE_ARRAYSIZE(static_indices), static_extents,
            IREE_ARRAYSIZE(static_extents), static_strides,
            IREE_ARRAYSIZE(static_strides), /*static_count=*/2,
            LOOM_LOCATION_UNKNOWN, &op));
        break;
      }
      case SiteKind::kRaceAccess:
        IREE_ASSERT_OK(loom_sanitizer_race_access_build(
            &builder_, /*build_flags=*/0, LOOM_SANITIZER_RACE_ACCESS_KIND_READ,
            view, /*indices=*/NULL,
            /*indices_count=*/0, static_indices, IREE_ARRAYSIZE(static_indices),
            /*atomic=*/false, /*ordering=*/0, /*scope=*/0,
            LOOM_LOCATION_UNKNOWN, &op));
        break;
      case SiteKind::kRaceFragmentAccess: {
        const loom_type_t fragment_type = loom_type_shaped_1d(
            LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
        const loom_value_id_t fragment =
            BuildConvertedValue(index, fragment_type);
        const loom_value_id_t rows = BuildConstant(4, LOOM_SCALAR_TYPE_INDEX);
        const loom_value_id_t columns =
            BuildConstant(1, LOOM_SCALAR_TYPE_INDEX);
        IREE_ASSERT_OK(loom_sanitizer_race_fragment_access_build(
            &builder_, /*build_flags=*/0,
            LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_KIND_READ, fragment, view,
            /*indices=*/NULL, /*indices_count=*/0, static_indices,
            IREE_ARRAYSIZE(static_indices), /*blocks=*/LOOM_VALUE_ID_INVALID,
            rows, columns, LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_ROLE_LHS,
            LOOM_LOCATION_UNKNOWN, &op));
        break;
      }
      case SiteKind::kRaceSync:
        IREE_ASSERT_OK(loom_sanitizer_race_sync_build(
            &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
            LOOM_ATOMIC_SCOPE_DEVICE, LOOM_ATOMIC_ORDERING_ACQ_REL,
            LOOM_LOCATION_UNKNOWN, &op));
        break;
    }
    ASSERT_NE(op, nullptr);
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, /*values=*/NULL,
                                         /*values_count=*/0,
                                         LOOM_LOCATION_UNKNOWN, &op));
  }

  loom_sanitizer_runtime_requirements_t Query(
      loom_sanitizer_reporting_mode_t reporting_mode) {
    const loom_sanitizer_options_t options = {
        .checks = 0,
        .flags = 0,
        .reporting_mode = reporting_mode,
    };
    loom_sanitizer_runtime_requirements_t requirements =
        LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE;
    IREE_EXPECT_OK(loom_sanitizer_runtime_requirements_query(module_, &options,
                                                             &requirements));
    return requirements;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_ = {};
};

TEST(SanitizerRuntimeRequirementsOptionsTest, DisabledRequiresNothing) {
  const loom_sanitizer_options_t options = {};
  EXPECT_EQ(loom_sanitizer_runtime_requirements_from_options(&options),
            LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE);
}

TEST(SanitizerRuntimeRequirementsOptionsTest, CombinesEnabledServices) {
  const loom_sanitizer_options_t options = {
      .checks = LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_RACE,
      .flags = 0,
      .reporting_mode = LOOM_SANITIZER_REPORTING_MODE_DEFAULT,
  };
  EXPECT_EQ(loom_sanitizer_runtime_requirements_from_options(&options),
            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK |
                LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW |
                LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW);
}

TEST(SanitizerRuntimeRequirementsOptionsTest, TrapOmitsFeedback) {
  const loom_sanitizer_options_t options = {
      .checks = LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_RACE,
      .flags = 0,
      .reporting_mode = LOOM_SANITIZER_REPORTING_MODE_TRAP,
  };
  EXPECT_EQ(loom_sanitizer_runtime_requirements_from_options(&options),
            LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW |
                LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW);
}

TEST_P(SanitizerRuntimeRequirementsTest, DerivesRequirementsFromAuthoredSite) {
  const SiteRequirementCase& test_case = GetParam();
  BuildSite(test_case.kind);
  EXPECT_EQ(Query(LOOM_SANITIZER_REPORTING_MODE_DEFAULT),
            test_case.default_requirements);
  EXPECT_EQ(Query(LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY),
            test_case.default_requirements);
  EXPECT_EQ(Query(LOOM_SANITIZER_REPORTING_MODE_TRAP),
            test_case.trap_requirements);
}

INSTANTIATE_TEST_SUITE_P(
    AuthoredOperations, SanitizerRuntimeRequirementsTest,
    ::testing::Values(
        SiteRequirementCase{SiteKind::kKernelAssert,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE},
        SiteRequirementCase{SiteKind::kAssertValue,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE},
        SiteRequirementCase{SiteKind::kAssertOp,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE},
        SiteRequirementCase{SiteKind::kAssertLayout,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE},
        SiteRequirementCase{
            SiteKind::kAssertAccess,
            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK |
                LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW,
            LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW},
        SiteRequirementCase{
            SiteKind::kAssertAccesses,
            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK |
                LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW,
            LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW},
        SiteRequirementCase{SiteKind::kRaceAccess,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK |
                                LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW},
        SiteRequirementCase{SiteKind::kRaceFragmentAccess,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK |
                                LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW},
        SiteRequirementCase{SiteKind::kRaceSync,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW,
                            LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW}));

}  // namespace
