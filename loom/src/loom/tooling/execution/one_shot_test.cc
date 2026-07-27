// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/one_shot.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/op_registry.h"
#include "loom/target/low_descriptor_registry_core_test.h"

namespace loom {
namespace {

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  return loom_op_registry_register_all_dialects(context);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_target_core_test_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

class OneShotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_run_session_options_t options = {};
    loom_run_session_options_initialize(&options);
    options.register_context = (loom_run_register_context_callback_t){
        /*.fn=*/RegisterContext,
    };
    options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            /*.fn=*/InitializeLowDescriptorRegistry,
        };
    IREE_ASSERT_OK(loom_run_session_initialize(&options, &session_));
  }

  void TearDown() override { loom_run_session_deinitialize(&session_); }

  iree_status_t Parse(iree_string_view_t source,
                      loom_run_module_t* out_module) {
    loom_run_module_parse_options_t options = {};
    loom_run_module_parse_options_initialize(&options);
    options.filename = IREE_SV("one_shot_test.loom");
    options.source = source;
    return loom_run_module_parse(&session_, &options, out_module);
  }

  loom_run_session_t session_ = {};
};

TEST_F(OneShotTest, AppliesStaticHalWorkgroupCountFromSingleKernel) {
  static const char kSource[] =
      "kernel.def @entry() {\n"
      "  %workgroups_z = index.constant 4 : index\n"
      "  %workgroups_y = index.constant 3 : index\n"
      "  %workgroups_x = index.constant 2 : index\n"
      "  %workgroup_size_z = index.constant 1 : index\n"
      "  %workgroup_size_y = index.constant 6 : index\n"
      "  %workgroup_size_x = index.constant 5 : index\n"
      "  kernel.launch.config workgroups(%workgroups_x, %workgroups_y, "
      "%workgroups_z) workgroup_size(%workgroup_size_x, %workgroup_size_y, "
      "%workgroup_size_z) : index\n"
      "} launch() {\n"
      "  kernel.return\n"
      "}\n";

  loom_run_module_t module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kSource), &module));

  loom_run_one_shot_options_t options = {};
  loom_run_one_shot_options_initialize(&options);
  EXPECT_TRUE(loom_run_one_shot_options_apply_static_hal_workgroup_count(
      module.module, iree_string_view_empty(), &options));
  EXPECT_EQ(options.hal_workgroup_count[0], 2u);
  EXPECT_EQ(options.hal_workgroup_count[1], 3u);
  EXPECT_EQ(options.hal_workgroup_count[2], 4u);

  loom_run_module_deinitialize(&module);
}

TEST_F(OneShotTest, AppliesStaticHalWorkgroupCountFromNamedKernel) {
  static const char kSource[] =
      "kernel.def @first() {\n"
      "  %unit = index.constant 1 : index\n"
      "  kernel.launch.config workgroups(%unit, %unit, %unit) "
      "workgroup_size(%unit, %unit, %unit) : index\n"
      "} launch() {\n"
      "  kernel.return\n"
      "}\n"
      "\n"
      "kernel.def @second() {\n"
      "  %unit = index.constant 1 : index\n"
      "  %workgroups_x = index.constant 7 : index\n"
      "  %workgroups_y = index.constant 8 : index\n"
      "  %workgroups_z = index.constant 9 : index\n"
      "  kernel.launch.config workgroups(%workgroups_x, %workgroups_y, "
      "%workgroups_z) workgroup_size(%unit, %unit, %unit) : index\n"
      "} launch() {\n"
      "  kernel.return\n"
      "}\n";

  loom_run_module_t module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kSource), &module));

  loom_run_one_shot_options_t options = {};
  loom_run_one_shot_options_initialize(&options);
  EXPECT_TRUE(loom_run_one_shot_options_apply_static_hal_workgroup_count(
      module.module, IREE_SV("@second"), &options));
  EXPECT_EQ(options.hal_workgroup_count[0], 7u);
  EXPECT_EQ(options.hal_workgroup_count[1], 8u);
  EXPECT_EQ(options.hal_workgroup_count[2], 9u);

  loom_run_module_deinitialize(&module);
}

TEST_F(OneShotTest, LeavesDefaultForAmbiguousImplicitKernel) {
  static const char kSource[] =
      "kernel.def @first() {\n"
      "  %unit = index.constant 1 : index\n"
      "  kernel.launch.config workgroups(%unit, %unit, %unit) "
      "workgroup_size(%unit, %unit, %unit) : index\n"
      "} launch() {\n"
      "  kernel.return\n"
      "}\n"
      "\n"
      "kernel.def @second() {\n"
      "  %extent = index.constant 2 : index\n"
      "  kernel.launch.config workgroups(%extent, %extent, %extent) "
      "workgroup_size(%extent, %extent, %extent) : index\n"
      "} launch() {\n"
      "  kernel.return\n"
      "}\n";

  loom_run_module_t module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kSource), &module));

  loom_run_one_shot_options_t options = {};
  loom_run_one_shot_options_initialize(&options);
  EXPECT_FALSE(loom_run_one_shot_options_apply_static_hal_workgroup_count(
      module.module, iree_string_view_empty(), &options));
  EXPECT_EQ(options.hal_workgroup_count[0], 1u);
  EXPECT_EQ(options.hal_workgroup_count[1], 1u);
  EXPECT_EQ(options.hal_workgroup_count[2], 1u);

  loom_run_module_deinitialize(&module);
}

}  // namespace
}  // namespace loom
