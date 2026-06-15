// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/preflight.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;

const loom_low_descriptor_set_t* LookupAmdgpuCdna3DescriptorSet() {
  loom_target_low_descriptor_registry_t registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  return loom_low_descriptor_registry_lookup(&registry.registry,
                                             IREE_SV("amdgpu.cdna3.core"));
}

uint16_t FindRegisterClassId(const loom_low_descriptor_set_t* descriptor_set,
                             iree_string_view_t name) {
  for (uint16_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* reg_class = &descriptor_set->reg_classes[i];
    const iree_string_view_t reg_class_name = loom_low_descriptor_set_string(
        descriptor_set, reg_class->name_string_offset);
    if (iree_string_view_equal(reg_class_name, name)) {
      return i;
    }
  }
  return LOOM_LOW_REG_CLASS_NONE;
}

loom_low_resolved_target_t ResolvedTarget(
    const loom_low_descriptor_set_t* descriptor_set) {
  loom_low_resolved_target_t target = {};
  target.target_name = IREE_SV("gfx942_target");
  target.descriptor_set_key = IREE_SV("amdgpu.cdna3.core");
  target.descriptor_set = descriptor_set;
  return target;
}

TEST(AmdgpuNativePreflightTest, AgprNativeMetadataUnsupportedEmitsDiagnostic) {
  const loom_low_descriptor_set_t* descriptor_set =
      LookupAmdgpuCdna3DescriptorSet();
  if (descriptor_set == nullptr) {
    GTEST_SKIP() << "amdgpu.cdna3.core is not linked in this build";
  }
  const uint16_t agpr_reg_class_id =
      FindRegisterClassId(descriptor_set, IREE_SV("amdgpu.agpr"));
  ASSERT_NE(agpr_reg_class_id, LOOM_LOW_REG_CLASS_NONE);

  loom_module_t module = {};
  loom_op_t function_op = {};
  const loom_low_resolved_target_t target = ResolvedTarget(descriptor_set);

  loom_low_schedule_table_t schedule = {};
  schedule.module = &module;
  schedule.function_op = &function_op;
  schedule.target = target;

  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = 0;
  assignment.value_class.type_kind = LOOM_TYPE_REGISTER;
  assignment.value_class.register_descriptor_set_stable_id =
      descriptor_set->stable_id;
  assignment.value_class.register_class_id = agpr_reg_class_id;
  assignment.descriptor_reg_class_id = agpr_reg_class_id;
  assignment.unit_count = 4;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_count = 4;

  loom_low_allocation_table_t allocation = {};
  allocation.module = &module;
  allocation.function_op = &function_op;
  allocation.target = target;
  allocation.assignments = &assignment;
  allocation.assignment_count = 1;

  DiagnosticEmissionCapture capture;
  loom_amdgpu_native_preflight_options_t options = {};
  options.emitter = capture.emitter();
  loom_amdgpu_native_preflight_t preflight = {};
  IREE_ASSERT_OK(loom_amdgpu_native_preflight_analyze(&schedule, &allocation,
                                                      &options, &preflight));

  EXPECT_EQ(preflight.error_count, 1u);
  ASSERT_EQ(capture.emissions.size(), 1u);
  const testing::CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_AMDGPU_035);
  ASSERT_EQ(emission.string_params.size(), 8u);
  EXPECT_EQ(emission.string_params[0], "gfx942_target");
  EXPECT_EQ(emission.string_params[3], "<unnamed>");
  EXPECT_EQ(emission.string_params[4], "<unknown>");
  EXPECT_EQ(emission.string_params[5], "amdgpu.agpr");
  EXPECT_EQ(emission.string_params[6], "amdgpu.agpr");
  EXPECT_EQ(emission.string_params[7], "AGPR kernel-descriptor");
}

}  // namespace
}  // namespace loom
