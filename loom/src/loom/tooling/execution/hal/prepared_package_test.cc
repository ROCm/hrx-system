// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/prepared_package.h"

#include <string_view>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

static constexpr std::string_view kManifest = R"json({
  "schema": "loom.testbench.prepared_actuals.v0",
  "target_profile": "amdgpu:gfx942",
  "artifact_format": "amdgpu-hsaco",
  "products": [
    {
      "subject": "private_reuse",
      "prepared_module": "products/0.loombc",
      "launch_config": "products/0.launch.loombc"
    },
    {
      "subject": "escaped_\u006bernel",
      "prepared_module": "products/1.loombc",
      "launch_config": "products/1.launch.loombc"
    }
  ]
})json";

class PreparedPackageTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_run_hal_testbench_prepared_package_manifest_deinitialize(&manifest_);
  }

  iree_status_t Parse(std::string_view json) {
    loom_run_hal_testbench_prepared_package_manifest_deinitialize(&manifest_);
    return loom_run_hal_testbench_prepared_package_manifest_parse(
        iree_make_string_view(json.data(), json.size()),
        iree_allocator_system(), &manifest_);
  }

  loom_run_hal_testbench_prepared_package_manifest_t manifest_ = {};
};

TEST_F(PreparedPackageTest, ParsesAndFindsProducts) {
  IREE_ASSERT_OK(Parse(kManifest));
  EXPECT_EQ(manifest_.product_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(manifest_.target_profile,
                                     IREE_SV("amdgpu:gfx942")));
  EXPECT_TRUE(iree_string_view_equal(manifest_.artifact_format,
                                     IREE_SV("amdgpu-hsaco")));

  const auto* first = loom_run_hal_testbench_prepared_package_manifest_find(
      &manifest_, IREE_SV("private_reuse"));
  ASSERT_NE(first, nullptr);
  EXPECT_TRUE(iree_string_view_equal(first->prepared_module_path,
                                     IREE_SV("products/0.loombc")));
  EXPECT_TRUE(iree_string_view_equal(first->launch_config_path,
                                     IREE_SV("products/0.launch.loombc")));
  EXPECT_NE(loom_run_hal_testbench_prepared_package_manifest_find(
                &manifest_, IREE_SV("escaped_kernel")),
            nullptr);
  EXPECT_EQ(loom_run_hal_testbench_prepared_package_manifest_find(
                &manifest_, IREE_SV("missing")),
            nullptr);
}

TEST_F(PreparedPackageTest, WritesCanonicalManifest) {
  IREE_ASSERT_OK(Parse(kManifest));
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_run_hal_testbench_prepared_package_manifest_write(
      &manifest_, &stream));
  EXPECT_TRUE(iree_string_view_equal(
      iree_string_builder_view(&builder),
      IREE_SV("{\"schema\":\"loom.testbench.prepared_actuals.v0\","
              "\"target_profile\":\"amdgpu:gfx942\","
              "\"artifact_format\":\"amdgpu-hsaco\",\"products\":[{"
              "\"subject\":\"private_reuse\","
              "\"prepared_module\":\"products/0.loombc\","
              "\"launch_config\":\"products/0.launch.loombc\"},{"
              "\"subject\":\"escaped_kernel\","
              "\"prepared_module\":\"products/1.loombc\","
              "\"launch_config\":\"products/1.launch.loombc\"}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST_F(PreparedPackageTest, RejectsMalformedRootFields) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(R"json({"schema":"loom.testbench.prepared_actuals.v0"})json"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[]})json"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[],"surprise":true})json"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(std::string(kManifest) + " false"));
}

TEST_F(PreparedPackageTest, RejectsUnsupportedSchemaAndEmptyProducts) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v1","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[{"subject":"k","prepared_module":"p.loombc","launch_config":"l.loombc"}]})json"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[]})json"));
}

TEST_F(PreparedPackageTest, RejectsMalformedProducts) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[false]})json"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[{"subject":"k","prepared_module":"p.loombc"}]})json"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[{"subject":"k","prepared_module":"p.loombc","launch_config":"l.loombc","extra":"x"}]})json"));
}

TEST_F(PreparedPackageTest, RejectsTraversalAndCollisions) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[{"subject":"k","prepared_module":"../p.loombc","launch_config":"l.loombc"}]})json"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[{"subject":"k","prepared_module":"p.loombc","launch_config":"l.loombc"},{"subject":"k","prepared_module":"q.loombc","launch_config":"m.loombc"}]})json"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Parse(
          R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[{"subject":"a","prepared_module":"shared.loombc","launch_config":"a.launch.loombc"},{"subject":"b","prepared_module":"b.loombc","launch_config":"shared.loombc"}]})json"));
}

TEST_F(PreparedPackageTest, AllowsSharedProductArtifacts) {
  IREE_ASSERT_OK(Parse(
      R"json({"schema":"loom.testbench.prepared_actuals.v0","target_profile":"amdgpu:gfx942","artifact_format":"amdgpu-hsaco","products":[{"subject":"a","prepared_module":"products/all.loombc","launch_config":"products/all.launch.loombc"},{"subject":"b","prepared_module":"products/all.loombc","launch_config":"products/all.launch.loombc"}]})json"));
  EXPECT_EQ(manifest_.product_count, 2u);
}

}  // namespace
}  // namespace loom
