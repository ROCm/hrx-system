// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/product/kernel.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_byte_sequence_t* MakeContents() {
  static const uint8_t kData[] = {1, 2, 3, 4};
  iree_byte_span_t data = iree_byte_span_empty();
  data.data_length = sizeof(kData);
  IREE_CHECK_OK(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), data.data_length, (void**)&data.data));
  memcpy(data.data, kData, sizeof(kData));
  iree_byte_sequence_t* contents = nullptr;
  IREE_CHECK_OK(iree_byte_sequence_create_from_span_move(
      &data, iree_allocator_system(), &contents));
  return contents;
}

TEST(KernelProductTest, OwnsTargetAndArtifactMetadata) {
  char target_key[] = "gfx1151";
  char bundle_name[] = "bundle";
  char snapshot_name[] = "snapshot";
  char export_plan_name[] = "export-plan";
  char export_symbol[] = "entry";
  char config_name[] = "config";
  char contract_set_key[] = "contract";
  loom_target_snapshot_t snapshot = {
      /*.name=*/iree_make_cstring_view(snapshot_name),
  };
  loom_target_export_plan_t export_plan = {
      /*.name=*/iree_make_cstring_view(export_plan_name),
      /*.export_symbol=*/iree_make_cstring_view(export_symbol),
  };
  loom_target_config_t config = {
      /*.name=*/iree_make_cstring_view(config_name),
      /*.contract_set_key=*/iree_make_cstring_view(contract_set_key),
  };
  loom_target_bundle_t bundle = {
      /*.name=*/iree_make_cstring_view(bundle_name),
      /*.snapshot=*/&snapshot,
      /*.export_plan=*/&export_plan,
      /*.config=*/&config,
  };
  char role[] = LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL;
  char format[] = "amdgpu-hsaco";
  char identifier[] = "kernel.hsaco";
  iree_byte_sequence_t* contents = MakeContents();
  loom_product_artifact_t artifact = {
      /*.role=*/iree_make_cstring_view(role),
      /*.format=*/iree_make_cstring_view(format),
      /*.identifier=*/iree_make_cstring_view(identifier),
      /*.contents=*/contents,
  };

  const loom_kernel_product_options_t options = {
      /*.target_key=*/iree_make_cstring_view(target_key),
      /*.target_bundle=*/&bundle,
      /*.artifacts=*/&artifact,
      /*.artifact_count=*/1,
      /*.loadable_artifact_ordinal=*/0,
      /*.export_count=*/3,
      /*.requirement_count=*/2,
  };
  loom_product_t* product = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_product_create(&options, iree_allocator_system(), &product));

  memset(target_key, 0, sizeof(target_key));
  memset(bundle_name, 0, sizeof(bundle_name));
  memset(snapshot_name, 0, sizeof(snapshot_name));
  memset(export_plan_name, 0, sizeof(export_plan_name));
  memset(export_symbol, 0, sizeof(export_symbol));
  memset(config_name, 0, sizeof(config_name));
  memset(contract_set_key, 0, sizeof(contract_set_key));
  memset(role, 0, sizeof(role));
  memset(format, 0, sizeof(format));
  memset(identifier, 0, sizeof(identifier));
  iree_byte_sequence_release(contents);

  EXPECT_TRUE(loom_product_isa(product, &loom_kernel_product_descriptor));
  EXPECT_TRUE(iree_string_view_equal(loom_kernel_product_operation.name,
                                     IREE_SV("kernel")));
  EXPECT_TRUE(iree_string_view_equal(loom_kernel_product_target_key(product),
                                     IREE_SV("gfx1151")));
  const loom_target_bundle_t* retained_bundle =
      loom_kernel_product_target_bundle(product);
  ASSERT_NE(retained_bundle, nullptr);
  EXPECT_TRUE(iree_string_view_equal(retained_bundle->name, IREE_SV("bundle")));
  EXPECT_TRUE(iree_string_view_equal(retained_bundle->snapshot->name,
                                     IREE_SV("snapshot")));
  EXPECT_TRUE(iree_string_view_equal(retained_bundle->export_plan->name,
                                     IREE_SV("export-plan")));
  EXPECT_TRUE(
      iree_string_view_equal(retained_bundle->config->name, IREE_SV("config")));
  EXPECT_EQ(loom_product_export_count(product), 3u);
  EXPECT_EQ(loom_product_requirement_count(product), 2u);
  const loom_product_artifact_t* retained_artifact =
      loom_kernel_product_loadable_artifact(product);
  ASSERT_NE(retained_artifact, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      retained_artifact->role, IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL)));
  EXPECT_TRUE(iree_string_view_equal(retained_artifact->format,
                                     IREE_SV("amdgpu-hsaco")));
  EXPECT_TRUE(iree_string_view_equal(retained_artifact->identifier,
                                     IREE_SV("kernel.hsaco")));
  EXPECT_EQ(iree_byte_sequence_length(retained_artifact->contents), 4u);

  loom_product_release(product);
}

TEST(KernelProductTest, RejectsInvalidLoadableArtifact) {
  const loom_target_snapshot_t snapshot = {};
  const loom_target_export_plan_t export_plan = {};
  const loom_target_config_t config = {};
  const loom_target_bundle_t bundle = {
      /*.name=*/IREE_SV("bundle"),
      /*.snapshot=*/&snapshot,
      /*.export_plan=*/&export_plan,
      /*.config=*/&config,
  };
  iree_byte_sequence_t* contents = MakeContents();
  const loom_product_artifact_t artifact = {
      /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_LISTING),
      /*.format=*/IREE_SV("text"),
      /*.identifier=*/IREE_SV("listing.txt"),
      /*.contents=*/contents,
  };
  const loom_kernel_product_options_t options = {
      /*.target_key=*/IREE_SV("target"),
      /*.target_bundle=*/&bundle,
      /*.artifacts=*/&artifact,
      /*.artifact_count=*/1,
      /*.loadable_artifact_ordinal=*/0,
  };
  loom_product_t* product = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_kernel_product_create(&options, iree_allocator_system(), &product));
  EXPECT_EQ(product, nullptr);
  iree_byte_sequence_release(contents);
}

}  // namespace
}  // namespace loom
