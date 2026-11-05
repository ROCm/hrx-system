// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/device_configuration.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DeviceConfigurationTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_serve_device_configuration_deinitialize(&configuration_);
  }
  iree_status_t Parse(const std::string& json) {
    loom_serve_device_configuration_deinitialize(&configuration_);
    return loom_serve_device_configuration_initialize(
        iree_make_string_view(json.data(), json.size()),
        iree_allocator_system(), &configuration_);
  }
  iree_status_t ParsePool(const std::string& fields) {
    return Parse(R"({"devices":{"gpu":{"uri":"task"}},"pools":{"agents":{)" +
                 fields + "}}}");
  }
  static std::string String(iree_string_view_t value) {
    return std::string(value.data, value.size);
  }
  void ExpectEmpty() {
    EXPECT_EQ(configuration_.count, 0u);
    EXPECT_EQ(configuration_.pool_count, 0u);
    EXPECT_EQ(configuration_.strings, nullptr);
    EXPECT_EQ(configuration_.pools, nullptr);
    EXPECT_EQ(configuration_.idle, nullptr);
  }
  // Owned declarations, independent of the input document lifetime.
  loom_serve_device_configuration_t configuration_ = {};
};

TEST_F(DeviceConfigurationTest, OwnsIndependentDeviceAndPoolInventories) {
  std::string json = R"({
    "pools":{
      "agents":{"devices":["gpu","npu"],"memory_bytes":"64gib",
        "weight_slab_bytes":2097152,"state_slab_bytes":65536,
        "workspace_idle_ms":2000,"parameter_idle_ms":60000},
      "images":{"devices":["gpu"],"memory_bytes":"16gib"},
      "audio":{"devices":["npu"],"pool_backing":"fixed"}},
    "devices":{
      "g\u0070u":{"uri":"amdgpu://0?queue_count=4"},
      "npu":{"uri":"xdna://0"}}
  })";
  IREE_ASSERT_OK(Parse(json));
  json.assign(json.size(), 'x');
  ASSERT_EQ(configuration_.count, 2u);
  ASSERT_EQ(configuration_.pool_count, 3u);
  EXPECT_EQ(String(configuration_.options[0].name), "gpu");
  EXPECT_EQ(String(configuration_.options[0].uri), "amdgpu://0?queue_count=4");
  EXPECT_EQ(configuration_.options[0].driver, nullptr);
  EXPECT_EQ(configuration_.options[0].completion_wait_flags,
            IREE_ASYNC_WAIT_FLAG_ACTIVE);
  EXPECT_EQ(String(configuration_.options[1].name), "npu");
  EXPECT_EQ(String(configuration_.options[1].uri), "xdna://0");
  EXPECT_EQ(configuration_.options[1].driver, nullptr);
  EXPECT_EQ(configuration_.options[1].completion_wait_flags,
            IREE_ASYNC_WAIT_FLAG_ACTIVE);
  const auto& agents = configuration_.pools[0];
  EXPECT_EQ(String(agents.binding.name), "agents");
  EXPECT_EQ(agents.binding.device_mask, 3u);
  EXPECT_TRUE(agents.options.elastic);
  EXPECT_EQ(agents.options.memory_limit, UINT64_C(64) << 30);
  EXPECT_EQ(agents.options.weight_slab_size, 2097152u);
  EXPECT_EQ(agents.options.state_slab_size, 65536u);
  EXPECT_EQ(configuration_.idle[0].workspace_timeout,
            iree_make_duration_ms(2000));
  EXPECT_EQ(configuration_.idle[0].parameter_timeout,
            iree_make_duration_ms(60000));
  EXPECT_EQ(configuration_.pools[1].binding.device_mask, 1u);
  EXPECT_EQ(configuration_.pools[1].options.memory_limit, UINT64_C(16) << 30);
  EXPECT_EQ(configuration_.pools[2].binding.device_mask, 2u);
  EXPECT_FALSE(configuration_.pools[2].options.elastic);
  EXPECT_EQ(configuration_.pools[2].options.memory_limit, 0u);
  EXPECT_EQ(configuration_.idle[2].workspace_timeout, 0);
}

TEST_F(DeviceConfigurationTest, UriConvenienceCreatesOneExplicitPool) {
  IREE_ASSERT_OK(Parse(R"({"devices":{"device":{"uri":"task"}},
    "pools":{"pool":{"devices":["device"]}}})"));
  loom_serve_device_configuration_t convenience = {};
  IREE_ASSERT_OK(loom_serve_device_configuration_initialize_uri(
      IREE_SV("task"), iree_allocator_system(), &convenience));
  EXPECT_EQ(convenience.count, 1u);
  EXPECT_EQ(convenience.pool_count, 1u);
  EXPECT_EQ(String(convenience.options[0].name), "device");
  EXPECT_EQ(String(convenience.options[0].uri), "task");
  EXPECT_EQ(convenience.options[0].completion_wait_flags,
            configuration_.options[0].completion_wait_flags);
  EXPECT_EQ(String(convenience.pools[0].binding.name), "pool");
  EXPECT_EQ(convenience.pools[0].binding.device_mask, 1u);
  const auto& actual = convenience.pools[0].options;
  EXPECT_EQ(actual.elastic, configuration_.pools[0].options.elastic);
  EXPECT_EQ(actual.weight_slab_size, UINT64_C(1) << 30);
  EXPECT_EQ(actual.state_slab_size, UINT64_C(64) << 20);
  EXPECT_EQ(actual.memory_limit, 0u);
  EXPECT_EQ(actual.placement.kind, LOOM_SERVE_POOL_PLACEMENT_AUTOMATIC);
  loom_serve_device_configuration_deinitialize(&convenience);
}

TEST_F(DeviceConfigurationTest,
       WaitModeAppliesToEveryDeviceWithoutChangingUris) {
  IREE_ASSERT_OK(Parse(R"({
    "devices":{"gpu":{"uri":"amdgpu://0?completion_spin_ns=50000"},
               "cpu":{"uri":"task"}},
    "pools":{"shared":{"devices":["gpu","cpu"]}},
    "wait_mode":"block\u0069ng"
  })"));
  ASSERT_EQ(configuration_.count, 2u);
  for (iree_host_size_t i = 0; i < configuration_.count; ++i) {
    EXPECT_EQ(configuration_.options[i].completion_wait_flags,
              IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_ASSERT_OK(loom_serve_device_configuration_set_wait_mode(
      &configuration_, IREE_SV("active")));
  for (iree_host_size_t i = 0; i < configuration_.count; ++i) {
    EXPECT_EQ(configuration_.options[i].completion_wait_flags,
              IREE_ASYNC_WAIT_FLAG_ACTIVE);
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_device_configuration_set_wait_mode(
                            &configuration_, IREE_SV("busy")));
  for (iree_host_size_t i = 0; i < configuration_.count; ++i) {
    EXPECT_EQ(configuration_.options[i].completion_wait_flags,
              IREE_ASYNC_WAIT_FLAG_ACTIVE);
  }
  IREE_ASSERT_OK(loom_serve_device_configuration_set_wait_mode(
      &configuration_, IREE_SV("blocking")));
  for (iree_host_size_t i = 0; i < configuration_.count; ++i) {
    EXPECT_EQ(configuration_.options[i].completion_wait_flags,
              IREE_ASYNC_WAIT_FLAG_NONE);
  }
  EXPECT_EQ(String(configuration_.options[0].uri),
            "amdgpu://0?completion_spin_ns=50000");
  EXPECT_EQ(String(configuration_.options[1].uri), "task");
}

TEST_F(DeviceConfigurationTest, ExplicitActiveWaitMode) {
  IREE_ASSERT_OK(Parse(R"({"wait_mode":"active",
    "devices":{"cpu":{"uri":"task"}},
    "pools":{"pool":{"devices":["cpu"]}}})"));
  EXPECT_EQ(configuration_.options[0].completion_wait_flags,
            IREE_ASYNC_WAIT_FLAG_ACTIVE);
}

TEST_F(DeviceConfigurationTest, RejectsMalformedOrDuplicateWaitMode) {
  for (const char* field :
       {R"("wait_mode":"")", R"("wait_mode":"wait")", R"("wait_mode":"ACTIVE")",
        R"("wait_mode":"active\u0000")", R"("wait_mode":null)",
        R"("wait_mode":true)", R"("wait_mode":1)", R"("wait_mode":[])",
        R"("wait_mode":{})",
        R"("wait_mode":"active","wait_mode":"blocking")"}) {
    SCOPED_TRACE(field);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Parse(std::string(R"({"devices":{"cpu":{"uri":"task"}},
          "pools":{"pool":{"devices":["cpu"]}},)") +
              field + "}"));
    ExpectEmpty();
  }
}

TEST_F(DeviceConfigurationTest, PlacementIsIndependentOfConsumerMembership) {
  IREE_ASSERT_OK(Parse(R"({
    "devices":{"cpu":{"uri":"task"},"gpu":{"uri":"amdgpu://0"}},
    "pools":{
      "remote":{"devices":["cpu"],"placement":{"device":"g\u0070u"}},
      "shared":{"devices":["cpu","gpu"],"placement":{"host_numa":3}},
      "automatic":{"devices":["gpu"]}}
  })"));
  ASSERT_EQ(configuration_.pool_count, 3u);
  EXPECT_EQ(configuration_.pools[0].binding.device_mask, 1u);
  EXPECT_EQ(configuration_.pools[0].options.placement.kind,
            LOOM_SERVE_POOL_PLACEMENT_DEVICE);
  EXPECT_EQ(configuration_.pools[0].options.placement.ordinal, 1u);
  EXPECT_EQ(configuration_.pools[1].binding.device_mask, 3u);
  EXPECT_EQ(configuration_.pools[1].options.placement.kind,
            LOOM_SERVE_POOL_PLACEMENT_HOST_NUMA);
  EXPECT_EQ(configuration_.pools[1].options.placement.ordinal, 3u);
  EXPECT_EQ(configuration_.pools[2].options.placement.kind,
            LOOM_SERVE_POOL_PLACEMENT_AUTOMATIC);
}

TEST_F(DeviceConfigurationTest, RejectsInvalidOrAmbiguousPlacement) {
  for (const char* placement :
       {R"({})", R"(null)", R"([])", R"("gpu")", R"(0)", R"({"device":0})",
        R"({"device":"missing"})", R"({"device":""})",
        R"({"device":"gpu","host_numa":0})",
        R"({"device":"gpu","device":"gpu"})",
        R"({"host_numa":0,"host_numa":0})", R"({"unknown":0})",
        R"({"host_numa":"0"})", R"({"host_numa":-1})", R"({"host_numa":1.5})",
        R"({"host_numa":1e3})"}) {
    SCOPED_TRACE(placement);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        ParsePool(std::string(R"("devices":["gpu"],"placement":)") +
                  placement));
    ExpectEmpty();
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      ParsePool(R"("devices":["gpu"],"placement":{"device":"gpu"},
                   "placement":{"device":"gpu"})"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      ParsePool(R"("devices":["gpu"],"placement":{"host_numa":4294967296})"));
  ExpectEmpty();
}

TEST_F(DeviceConfigurationTest, ByteSizeStringsDecodeToExactCounts) {
  IREE_ASSERT_OK(ParsePool(R"("devices":["gpu"],
    "pool_backing":"el\u0061stic","memory_bytes":"64\u0067ib",
    "weight_slab_bytes":"1gb","state_slab_bytes":"64mib",
    "workspace_idle_ms":1000)"));
  EXPECT_EQ(configuration_.pools[0].options.memory_limit, UINT64_C(64) << 30);
  EXPECT_EQ(configuration_.pools[0].options.weight_slab_size,
            UINT64_C(1000000000));
  EXPECT_EQ(configuration_.pools[0].options.state_slab_size, UINT64_C(64)
                                                                 << 20);
  EXPECT_EQ(configuration_.idle[0].workspace_timeout,
            iree_make_duration_ms(1000));
  IREE_ASSERT_OK(ParsePool(R"("devices":["gpu"],
    "memory_bytes":"9007199254740993","weight_slab_bytes":4096,
    "state_slab_bytes":"004096")"));
  EXPECT_EQ(configuration_.pools[0].options.memory_limit,
            UINT64_C(9007199254740993));
  EXPECT_EQ(configuration_.pools[0].options.weight_slab_size, 4096u);
  EXPECT_EQ(configuration_.pools[0].options.state_slab_size, 4096u);
}

TEST_F(DeviceConfigurationTest, ByteSizeErrorsNamePoolAndField) {
  const iree::Status status(
      ParsePool(R"("devices":["gpu"],"memory_bytes":"123fb")"));
  EXPECT_EQ(status.code(), iree::StatusCode::kInvalidArgument);
  EXPECT_NE(status.ToString().find("pool 'agents' field 'memory_bytes'"),
            std::string::npos);
  ExpectEmpty();
}

TEST_F(DeviceConfigurationTest, RejectsMalformedPoolFieldsAndConsumerSets) {
  const char* invalid[] = {
      "",
      R"("devices":[])",
      R"("devices":"gpu")",
      R"("devices":[0])",
      R"("devices":["missing"])",
      R"("devices":["gpu","g\u0070u"])",
      R"("devices":["gpu"],"devices":["gpu"])",
      R"("devices":["gpu"],"memory_bytes":1,"pool_backing":"fixed")",
      R"("devices":["gpu"],"memory_bytes":"1gib","pool_backing":"fixed")",
      R"("devices":["gpu"],"pool_backing":"automatic")",
      R"("devices":["gpu"],"memory_bytes":1,"memory_bytes":"4kib")",
      R"("devices":["gpu"],"parameter_idle_ms":false)",
      R"("devices":["gpu"],"parameter_idle_ms":"1000")",
      R"("devices":["gpu"],"workspace_idle_ms":-1)",
      R"("devices":["gpu"],"unknown":1)",
  };
  for (const char* fields : invalid) {
    SCOPED_TRACE(fields);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, ParsePool(fields));
    ExpectEmpty();
  }
  for (const char* field :
       {"weight_slab_bytes", "state_slab_bytes", "memory_bytes"}) {
    for (const char* value :
         {R"("")", R"("-1")", R"("1.5gib")", R"("1e3")", R"("123fb")",
          R"("64gib extra")", R"("1\u0000gb")", "-1", "1.5", "1e3", "true",
          "null", "[]", "{}"}) {
      const std::string fields =
          std::string(R"("devices":["gpu"],")") + field + "\":" + value;
      SCOPED_TRACE(fields);
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, ParsePool(fields));
      ExpectEmpty();
    }
  }
}

TEST_F(DeviceConfigurationTest,
       RejectsMalformedInventoriesBeforeOpeningDevices) {
  for (const char* devices :
       {R"([])", R"({})", R"({"gpu":{}})", R"({"gpu":[]})",
        R"({"":{"uri":"task"}})", R"({"bad key":{"uri":"task"}})",
        R"({"gpu":{"uri":"task"},"g\u0070u":{"uri":"task"}})",
        R"({"gpu":{"uri":""}})", R"({"gpu":{"uri":0}})",
        R"({"gpu":{"uri":"task\u0000"}})",
        R"({"gpu":{"uri":"task","uri":"xdna"}})",
        R"({"gpu":{"uri":"task","memory_bytes":"64gib"}})"}) {
    const std::string json = std::string(R"({"devices":)") + devices +
                             R"(,"pools":{"pool":{"devices":["gpu"]}}})";
    SCOPED_TRACE(json);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
    ExpectEmpty();
  }
  for (const char* pools :
       {R"([])", R"({})", R"({"pool":0})", R"({"bad key":{"devices":["gpu"]}})",
        R"({"pool":{"devices":["gpu"]},"p\u006fol":{"devices":["gpu"]}})"}) {
    const std::string json =
        std::string(R"({"devices":{"gpu":{"uri":"task"}},"pools":)") + pools +
        "}";
    SCOPED_TRACE(json);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
    ExpectEmpty();
  }
  for (const char* json :
       {"", "[]", "{}", R"({"devices":{}})", R"({"pools":{}})",
        R"({"devices":{},"devices":{},"pools":{}})",
        R"({"devices":{},"pools":{},"pools":{}})",
        R"({"devices":{},"pools":{},"unknown":0})",
        R"({"devices":{},"pools":{}} trailing)"}) {
    SCOPED_TRACE(json);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
    ExpectEmpty();
  }
}

TEST_F(DeviceConfigurationTest, RejectsOverflowAndAcceptsExplicitZero) {
  for (const char* fields : {R"("memory_bytes":18446744073709551616)",
                             R"("memory_bytes":"18446744073709551616")",
                             R"("memory_bytes":"17179869184gib")",
                             R"("workspace_idle_ms":9223372036855)",
                             R"("parameter_idle_ms":9223372036855)"}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        ParsePool(std::string(R"("devices":["gpu"],)") + fields));
    ExpectEmpty();
  }
  IREE_ASSERT_OK(ParsePool(R"("devices":["gpu"],
    "weight_slab_bytes":0,"state_slab_bytes":0,"memory_bytes":0,
    "parameter_idle_ms":0,"workspace_idle_ms":0)"));
  EXPECT_EQ(configuration_.pools[0].options.weight_slab_size, 0u);
  EXPECT_EQ(configuration_.pools[0].options.state_slab_size, 0u);
  IREE_ASSERT_OK(ParsePool(R"("devices":["gpu"],
    "pool_backing":"fixed","weight_slab_bytes":"0b",
    "state_slab_bytes":"0mib","memory_bytes":"0")"));
  EXPECT_FALSE(configuration_.pools[0].options.elastic);
  EXPECT_EQ(configuration_.pools[0].options.memory_limit, 0u);
}

}  // namespace
