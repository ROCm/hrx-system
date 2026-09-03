// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/environment.h"
#include "iree/vm/module_test_provider.h"

namespace {

class VMModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    const iree_vm_ref_type_table_t* table =
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm"));
    ASSERT_NE(table, nullptr);
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(table, &types_));
    IREE_ASSERT_OK(iree_vm_module_test_provider_initialize(
        types_.buffer, &destroy_count_, &provider_));
    module_ = &provider_.base;
  }

  void TearDown() override {
    iree_vm_module_release(module_);
    iree_vm_environment_free(environment_);
  }

  iree_vm_environment_t* environment_ = nullptr;
  iree_vm_ref_types_t types_ = {};
  iree_vm_module_test_provider_t provider_ = {};
  iree_vm_module_t* module_ = nullptr;
  int destroy_count_ = 0;
};

TEST_F(VMModuleTest, PublishesStableStructureAndLifetime) {
  EXPECT_TRUE(
      iree_string_view_equal(iree_vm_module_name(module_), IREE_SV("fixture")));
  EXPECT_EQ(iree_vm_module_import_count(module_), 1u);
  EXPECT_EQ(iree_vm_module_export_count(module_), 2u);
  EXPECT_EQ(iree_vm_module_function_count(module_), 1u);
  EXPECT_EQ(iree_vm_module_ref_type_count(module_), 1u);

  iree_vm_ref_type_t ref_type = nullptr;
  IREE_ASSERT_OK(iree_vm_module_ref_type_by_ordinal(module_, 0, &ref_type));
  EXPECT_EQ(ref_type, types_.buffer);

  iree_vm_module_retain(module_);
  iree_vm_module_release(module_);
  EXPECT_EQ(destroy_count_, 0);
  iree_vm_module_release(module_);
  module_ = nullptr;
  EXPECT_EQ(destroy_count_, 1);
}

TEST_F(VMModuleTest, QueriesSortedPublicDeclarations) {
  iree_vm_import_t import_value = {};
  IREE_ASSERT_OK(iree_vm_module_import_by_ordinal(module_, 0, &import_value));
  const iree_vm_import_target_t target = iree_vm_import_target(import_value);
  EXPECT_TRUE(iree_string_view_equal(target.module_name, IREE_SV("support")));
  EXPECT_TRUE(iree_string_view_equal(target.export_name, IREE_SV("apply")));

  iree_vm_export_t export_value = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_export(module_, IREE_SV("increment"),
                                              &export_value));
  EXPECT_EQ(export_value.ordinal, 1u);
  EXPECT_TRUE(iree_string_view_equal(iree_vm_export_name(export_value),
                                     IREE_SV("increment")));

  iree_vm_module_export_declaration_t export_declaration = {};
  IREE_ASSERT_OK(iree_vm_module_query_export(module_, export_value.ordinal,
                                             &export_declaration));
  EXPECT_EQ(export_declaration.function_ordinal, 0u);
  EXPECT_EQ(export_declaration.callable_type_ordinal, 0u);
}

TEST_F(VMModuleTest, CheckedQueriesLeaveOutputsUntouched) {
  iree_vm_export_t export_value = {
      reinterpret_cast<iree_vm_module_t*>(uintptr_t{1}),
      99,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_vm_module_export_by_ordinal(module_, 3, &export_value));
  EXPECT_EQ(export_value.module,
            reinterpret_cast<iree_vm_module_t*>(uintptr_t{1}));
  EXPECT_EQ(export_value.ordinal, 99u);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_vm_module_lookup_export(module_, IREE_SV("missing"), &export_value));
  EXPECT_EQ(export_value.ordinal, 99u);
}

TEST_F(VMModuleTest, CallsNativeValueFunctionThroughGenericABI) {
  iree_vm_export_t export_value = {};
  IREE_ASSERT_OK(
      iree_vm_module_lookup_export(module_, IREE_SV("add_one"), &export_value));
  iree_vm_module_export_declaration_t declaration = {};
  IREE_ASSERT_OK(
      iree_vm_module_query_export(module_, export_value.ordinal, &declaration));
  uint64_t argument = 41;
  uint64_t result = 0;
  iree_vm_call_packet_t call = {};
  call.value_arguments.direct = &argument;
  call.value_results.direct = &result;
  iree_vm_module_function_start_params_t params = {};
  params.function_ordinal = declaration.function_ordinal;
  params.call = call;
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_ASSERT_OK(module_->vtable->function_start(module_, &params, &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  EXPECT_EQ(result, 42u);
}

}  // namespace
