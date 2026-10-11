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
  iree_status_t CreateCpuGroup() {
    IREE_RETURN_IF_ERROR(
        loom_serve_host_create(1, &host_, iree_allocator_system()));
    const loom_serve_device_options_t options[] = {{.uri = IREE_SV("task")},
                                                   {.uri = IREE_SV("task")}};
    return loom_serve_device_group_create(host_, IREE_ARRAYSIZE(options),
                                          options, &group_,
                                          iree_allocator_system());
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
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t resources = {.device_group = group_};
  EXPECT_EQ(
      Write(resources),
      R"({"listeners":[],"devices":[{"index":0,"id":"task","name":"task","driver":"task","backend":"task","physical_devices":[{"ordinal":0,"name":"task","path":"task"}],"memory":{"backing":"fixed","retained":null,"workspace":{"reserved_bytes":0,"committed_bytes":0,"free_bytes":0,"quarantined_bytes":0,"budget_limit":0,"reservation_count":0}}},{"index":1,"id":"task","name":"task","driver":"task","backend":"task","physical_devices":[{"ordinal":0,"name":"task","path":"task"}],"memory":{"backing":"fixed","retained":null,"workspace":{"reserved_bytes":0,"committed_bytes":0,"free_bytes":0,"quarantined_bytes":0,"budget_limit":0,"reservation_count":0}}}]})");
}

TEST_F(ResourcesTest, TrimRequiresExplicitDeviceAndIntegerTarget) {
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t resources = {.device_group = group_};
  for (const auto* value : {
           "",
           "[]",
           "null",
           "{}",
           R"({"device":0})",
           R"({"target_bytes":0})",
           R"({"device":"0","target_bytes":0})",
           R"({"device":0,"target_bytes":null})",
           R"({"device":true,"target_bytes":0})",
           R"({"device":0,"target_bytes":[]})",
           R"({"device":0,"target_bytes":0,"unknown":1})",
           R"({"device":0,"device":1,"target_bytes":0})",
           R"({"device":0,"target_bytes":0,"target_bytes":1})",
           R"({"device":0.0,"target_bytes":0})",
           R"({"device":-1,"target_bytes":0})",
           R"({"device":0,"target_bytes":-1})",
           R"({"device":0,"target_bytes":0.5})",
           R"({"device":0,"target_bytes":1e3})",
           R"({"device":0,"target_bytes":0} trailing)",
       }) {
    SCOPED_TRACE(value);
    iree_host_size_t index = 99;
    uint64_t target = 99;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_resources_parse_trim(
            &resources, iree_make_cstring_view(value), &index, &target));
    EXPECT_EQ(index, 99u);
    EXPECT_EQ(target, 99u);
  }
}

TEST_F(ResourcesTest, TrimResolvesOnlyPublishedDeviceIndices) {
  IREE_ASSERT_OK(CreateCpuGroup());
  for (const loom_serve_resources_t resources :
       {loom_serve_resources_t{},
        loom_serve_resources_t{.device_group = group_}}) {
    for (const auto* value : {
             R"({"device":2,"target_bytes":0})",
             R"({"device":18446744073709551615,"target_bytes":0})",
             R"({"device":18446744073709551616,"target_bytes":0})",
             R"({"device":0,"target_bytes":18446744073709551616})",
         }) {
      SCOPED_TRACE(value);
      iree_host_size_t index = 99;
      uint64_t target = 99;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_OUT_OF_RANGE,
          loom_serve_resources_parse_trim(
              &resources, iree_make_cstring_view(value), &index, &target));
      EXPECT_EQ(index, 99u);
      EXPECT_EQ(target, 99u);
    }
  }
  const loom_serve_resources_t empty = {};
  iree_host_size_t index = 0;
  uint64_t target = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_resources_parse_trim(
                            &empty, IREE_SV(R"({"device":0,"target_bytes":0})"),
                            &index, &target));
}

TEST_F(ResourcesTest, TrimReportsSelectedResourceWithoutModelOwnership) {
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t resources = {.device_group = group_};
  iree_host_size_t index = 0;
  uint64_t target = 0;
  IREE_ASSERT_OK(loom_serve_resources_parse_trim(
      &resources,
      IREE_SV(R"({"target_bytes":18446744073709551615,"device":1})"), &index,
      &target));
  EXPECT_EQ(index, 1u);
  EXPECT_EQ(target, UINT64_MAX);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&output_, &stream);
  IREE_ASSERT_OK(loom_serve_resources_trim(&resources, index, target, &stream));
  auto value = iree_string_builder_view(&output_);
  EXPECT_EQ(
      std::string(value.data, value.size),
      R"({"target_bytes":18446744073709551615,"device":{"index":1,"id":"task","name":"task","driver":"task","backend":"task","physical_devices":[{"ordinal":0,"name":"task","path":"task"}],"memory":{"backing":"fixed","retained":null,"workspace":{"reserved_bytes":0,"committed_bytes":0,"free_bytes":0,"quarantined_bytes":0,"budget_limit":0,"reservation_count":0}}}})");
}

}  // namespace
