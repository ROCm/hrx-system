// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/profile.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

const loom_target_profile_type_t kBaseProfileType = {
    .name = IREE_SVL("base"),
    .equal = loom_target_profile_equal_common,
};

TEST(TargetProfileTest,
     ComparesCommonFieldsWithoutProjectingOrPointerIdentity) {
  loom_target_bundle_storage_t lhs_storage = {};
  loom_target_bundle_storage_rebind(&lhs_storage);
  const loom_target_profile_t lhs = {
      .type = &kBaseProfileType,
      .target_bundle = &lhs_storage.bundle,
  };
  loom_target_bundle_storage_t rhs_storage = lhs_storage;
  loom_target_bundle_storage_rebind(&rhs_storage);
  const loom_target_profile_t rhs = {
      .type = &kBaseProfileType,
      .target_bundle = &rhs_storage.bundle,
  };
  EXPECT_TRUE(loom_target_profile_equal(&lhs, &rhs));

  struct Mutation {
    // Common payload member whose change must separate compilation domains.
    const char* name;
    // Changes that member without changing any pointer identity.
    void (*apply)(loom_target_bundle_storage_t&);
  };
#define SCALAR(field)                                                         \
  {                                                                           \
    #field, [](loom_target_bundle_storage_t& storage) { storage.field ^= 1; } \
  }
#define NAME(field)                                     \
  {                                                     \
    #field, [](loom_target_bundle_storage_t& storage) { \
      storage.field = IREE_SV("changed");               \
    }                                                   \
  }
  const Mutation mutations[] = {
      NAME(bundle.name),
      NAME(snapshot.name),
      SCALAR(snapshot.codegen_format),
      SCALAR(snapshot.artifact_format),
      SCALAR(snapshot.subgroup_partition),
      SCALAR(snapshot.maximum_index_bitwidth),
      SCALAR(snapshot.default_pointer_bitwidth),
      SCALAR(snapshot.index_bitwidth),
      SCALAR(snapshot.offset_bitwidth),
      SCALAR(snapshot.max_workgroup_size.x),
      SCALAR(snapshot.max_workgroup_size.y),
      SCALAR(snapshot.max_workgroup_size.z),
      SCALAR(snapshot.max_flat_workgroup_size),
      SCALAR(snapshot.max_workgroup_storage_bytes),
      SCALAR(snapshot.subgroup_size),
      SCALAR(snapshot.max_grid_size.x),
      SCALAR(snapshot.max_grid_size.y),
      SCALAR(snapshot.max_grid_size.z),
      SCALAR(snapshot.max_flat_grid_size),
      SCALAR(snapshot.max_workgroup_count.x),
      SCALAR(snapshot.max_workgroup_count.y),
      SCALAR(snapshot.max_workgroup_count.z),
      SCALAR(snapshot.memory_spaces.generic),
      SCALAR(snapshot.memory_spaces.global),
      SCALAR(snapshot.memory_spaces.workgroup),
      SCALAR(snapshot.memory_spaces.constant),
      SCALAR(snapshot.memory_spaces.private_memory),
      SCALAR(snapshot.memory_spaces.host),
      SCALAR(snapshot.memory_spaces.descriptor),
      NAME(export_plan.name),
      NAME(export_plan.export_symbol),
      NAME(export_plan.calling_convention),
      SCALAR(export_plan.abi_kind),
      SCALAR(export_plan.linkage),
      SCALAR(export_plan.hal_kernel.required_workgroup_size.x),
      SCALAR(export_plan.hal_kernel.required_workgroup_size.y),
      SCALAR(export_plan.hal_kernel.required_workgroup_size.z),
      SCALAR(export_plan.hal_kernel.flat_workgroup_size_min),
      SCALAR(export_plan.hal_kernel.flat_workgroup_size_max),
      NAME(config.name),
      SCALAR(config.schedule_model_stable_id),
      SCALAR(config.contract_set.stable_id),
      NAME(config.contract_set.spelling),
      SCALAR(config.contract_feature_bits),
  };
#undef NAME
#undef SCALAR
  for (const auto& mutation : mutations) {
    SCOPED_TRACE(mutation.name);
    rhs_storage = lhs_storage;
    loom_target_bundle_storage_rebind(&rhs_storage);
    mutation.apply(rhs_storage);
    EXPECT_FALSE(loom_target_profile_equal(&lhs, &rhs));
    EXPECT_FALSE(loom_target_profile_equal(&rhs, &lhs));
  }
}

TEST(TargetProfileTest, PreservesPartialFieldsAndFamilyIdentity) {
  loom_target_bundle_t first_bundle = {};
  loom_target_bundle_t second_bundle = {};
  loom_target_profile_t first = {
      .type = &kBaseProfileType,
      .target_bundle = &first_bundle,
  };
  loom_target_profile_t second = {
      .type = &kBaseProfileType,
      .target_bundle = &second_bundle,
  };
  EXPECT_TRUE(loom_target_profile_equal(nullptr, nullptr));
  EXPECT_FALSE(loom_target_profile_equal(&first, nullptr));
  EXPECT_FALSE(loom_target_profile_equal(nullptr, &first));
  EXPECT_TRUE(loom_target_profile_equal(&first, &first));
  EXPECT_TRUE(loom_target_profile_equal(&first, &second));
  second.explicit_fields = UINT64_C(1) << LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE;
  EXPECT_FALSE(loom_target_profile_equal(&first, &second));
  second.explicit_fields = 0;
  second.target_bundle = nullptr;
  EXPECT_FALSE(loom_target_profile_equal(&first, &second));
  second.target_bundle = &second_bundle;
  const loom_target_snapshot_t snapshot = {};
  second_bundle.snapshot = &snapshot;
  EXPECT_FALSE(loom_target_profile_equal(&first, &second));
  second_bundle.snapshot = nullptr;
  const loom_target_export_plan_t export_plan = {};
  second_bundle.export_plan = &export_plan;
  EXPECT_FALSE(loom_target_profile_equal(&first, &second));
  second_bundle.export_plan = nullptr;
  const loom_target_config_t config = {};
  second_bundle.config = &config;
  EXPECT_FALSE(loom_target_profile_equal(&first, &second));
  second_bundle.config = nullptr;
  const loom_target_profile_type_t other_type = kBaseProfileType;
  second.type = &other_type;
  EXPECT_FALSE(loom_target_profile_equal(&first, &second));
}

}  // namespace
}  // namespace loom
