// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/resources.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ResourcesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &output_);
  }
  void TearDown() override {
    iree_string_builder_deinitialize(&output_);
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
  }
  std::string Write(const loom_serve_resources_t& resources) {
    iree_string_builder_reset(&output_);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&output_, &stream);
    IREE_EXPECT_OK(loom_serve_resources_write_json(&resources, &stream));
    auto value = iree_string_builder_view(&output_);
    return std::string(value.data, value.size);
  }

  // Application owner outliving all borrowed records.
  loom_serve_host_t* host_ = nullptr;
  // One immutable group with independently identifiable logical devices.
  loom_serve_device_group_t* group_ = nullptr;
  // Response storage detached from all resource owners.
  iree_string_builder_t output_;
};

TEST_F(ResourcesTest, NoPublishedResourcesDoesNotInventHardware) {
  EXPECT_EQ(Write({}), R"({"listeners":[],"devices":[]})");
}

TEST_F(ResourcesTest, EveryRecordOfOneCpuGroupIsPublishedWithoutModelWork) {
  IREE_ASSERT_OK(loom_serve_host_create(1, &host_, iree_allocator_system()));
  const loom_serve_device_options_t options[] = {{.uri = IREE_SV("task")},
                                                 {.uri = IREE_SV("task")}};
  IREE_ASSERT_OK(loom_serve_device_group_create(host_, IREE_ARRAYSIZE(options),
                                                options, &group_,
                                                iree_allocator_system()));
  const loom_serve_resources_t resources = {.device_group = group_};
  EXPECT_EQ(
      Write(resources),
      R"({"listeners":[],"devices":[{"index":0,"id":"task","name":"task","driver":"task","backend":"task","physical_devices":[{"ordinal":0,"name":"task","path":"task"}],"memory":{"backing":"fixed","retained":null,"workspace":{"reserved_bytes":0,"committed_bytes":0,"free_bytes":0,"quarantined_bytes":0,"budget_limit":0,"reservation_count":0}}},{"index":1,"id":"task","name":"task","driver":"task","backend":"task","physical_devices":[{"ordinal":0,"name":"task","path":"task"}],"memory":{"backing":"fixed","retained":null,"workspace":{"reserved_bytes":0,"committed_bytes":0,"free_bytes":0,"quarantined_bytes":0,"budget_limit":0,"reservation_count":0}}}]})");
}

}  // namespace
