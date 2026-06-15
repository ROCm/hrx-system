// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace loom {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

iree_vm_instance_t* hal_testbench_actual_test_vm_instance = nullptr;

class HalTestbenchActualTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(iree_vm_instance_create(
        IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(),
        &hal_testbench_actual_test_vm_instance));
  }

  static void TearDownTestSuite() {
    iree_vm_instance_release(hal_testbench_actual_test_vm_instance);
    hal_testbench_actual_test_vm_instance = nullptr;
  }
};

static const loom_run_hal_artifact_provider_t kFakeHalArtifactProvider = {
    /*.name=*/IREE_SVL("fake-hal"),
    /*.hal_driver_name=*/IREE_SVL("fake"),
    /*.target_family_name=*/IREE_SVL("fake-target"),
};

TEST_F(HalTestbenchActualTest, RequiresExplicitDeviceWhenHalProviderExists) {
  const loom_run_hal_artifact_provider_t* artifact_providers[] = {
      &kFakeHalArtifactProvider,
  };
  loom_run_hal_artifact_provider_registry_t registry = {};
  loom_run_hal_artifact_provider_registry_initialize_from_entries(
      artifact_providers, IREE_ARRAYSIZE(artifact_providers), &registry);

  loom_run_hal_testbench_context_t context = {};
  loom_run_hal_testbench_context_initialize(&registry, iree_allocator_system(),
                                            &context);

  iree::Status status = iree::internal::ConsumeForTest(
      loom_run_hal_testbench_context_ensure_runtime(&context));
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(), HasSubstr("explicit --device= URI"));
  EXPECT_THAT(status.ToString(), HasSubstr("fake-hal"));

  loom_run_hal_testbench_context_deinitialize(&context);
}

TEST_F(HalTestbenchActualTest, ScalarInputsPackDispatchConstantWords) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(iree_vm_value_make_i32(0x12345678)),
      iree_vm_make_variant_value(
          iree_vm_value_make_i64(static_cast<int64_t>(0x1122334455667788ull))),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 3u);
  EXPECT_EQ(options.constants[0], 0x12345678u);
  EXPECT_EQ(options.constants[1], 0x55667788u);
  EXPECT_EQ(options.constants[2], 0x11223344u);

  iree_vm_list_release(bindings);
}

TEST_F(HalTestbenchActualTest, F64InputsPackDispatchConstantWords) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(iree_vm_value_make_f64(1.0)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x00000000u);
  EXPECT_EQ(options.constants[1], 0x3ff00000u);

  iree_vm_list_release(bindings);
}

TEST_F(HalTestbenchActualTest, IndexInputPacksAsOneDispatchConstantWord) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(iree_vm_value_make_i64(3584)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 1u);
  EXPECT_EQ(options.constants[0], 3584u);

  iree_vm_list_release(bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputPacksAsTwoDispatchConstantWords) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(
          iree_vm_value_make_i64(static_cast<int64_t>(0x1122334455667788ull))),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x55667788u);
  EXPECT_EQ(options.constants[1], 0x11223344u);

  iree_vm_list_release(bindings);
}

}  // namespace
}  // namespace loom
