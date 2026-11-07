// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-serve/options.h"

#include <string>

#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/server/device_configuration.h"

namespace {

class DeviceConfigurationOptionsTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_serve_device_configuration_deinitialize(&configuration_);
  }

  iree_status_t Initialize(
      const loom_serve_device_configuration_options_t& options) {
    loom_serve_device_configuration_deinitialize(&configuration_);
    return loom_serve_device_configuration_initialize_from_options(
        &options, iree_allocator_system(), &configuration_);
  }

  void ExpectFailure(const loom_serve_device_configuration_options_t& options,
                     iree::StatusCode code, const char* diagnostic) {
    const iree::Status status(Initialize(options));
    EXPECT_EQ(status.code(), code);
    EXPECT_NE(status.ToString().find(diagnostic), std::string::npos)
        << status.ToString();
    ExpectEmpty();
  }

  void ExpectEmpty() {
    EXPECT_EQ(configuration_.count, 0u);
    EXPECT_EQ(configuration_.pool_count, 0u);
    EXPECT_EQ(configuration_.strings, nullptr);
    EXPECT_EQ(configuration_.pools, nullptr);
    EXPECT_EQ(configuration_.idle, nullptr);
  }

  static std::string String(iree_string_view_t value) {
    return std::string(value.data, value.size);
  }

  // Owned decoded configuration reset before each initialization.
  loom_serve_device_configuration_t configuration_ = {};
};

TEST_F(DeviceConfigurationOptionsTest, RequiresExactlyOneSource) {
  loom_serve_device_configuration_options_t options = {};
  ExpectFailure(options, iree::StatusCode::kInvalidArgument,
                "--device is required");

  iree_string_view_t device_uris[] = {IREE_SV("task"), IREE_SV("task")};
  options.device_uris = {
      .count = IREE_ARRAYSIZE(device_uris),
      .values = device_uris,
  };
  ExpectFailure(options, iree::StatusCode::kInvalidArgument,
                "multiple devices require named entries");
}

TEST_F(DeviceConfigurationOptionsTest,
       NamedConfigurationRejectsEverySingleDeviceOption) {
  iree_string_view_t device_uri = IREE_SV("task");
  loom_serve_device_configuration_options_t cases[7] = {};
  for (auto& options : cases) {
    options.configuration_path = IREE_SV("not-opened.json");
  }
  cases[0].device_uris = {
      .count = 1,
      .values = &device_uri,
  };
  cases[1].pool_backing = IREE_SV("elastic");
  cases[2].weight_slab_bytes = IREE_SV("0");
  cases[3].state_slab_bytes = IREE_SV("0");
  cases[4].memory_bytes = IREE_SV("0");
  cases[5].workspace_idle_ms = IREE_SV("0");
  cases[6].parameter_idle_ms = IREE_SV("0");
  const char* names[] = {
      "device",
      "pool_backing",
      "weight_slab_bytes",
      "state_slab_bytes",
      "memory_bytes",
      "workspace_idle_ms",
      "parameter_idle_ms",
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(cases); ++i) {
    SCOPED_TRACE(names[i]);
    ExpectFailure(cases[i], iree::StatusCode::kInvalidArgument,
                  "--devices cannot be combined");
  }
}

TEST_F(DeviceConfigurationOptionsTest,
       NamedConfigurationLoadsAndAcceptsWaitOverride) {
  const std::string json = R"({
    "wait_mode":"blocking",
    "devices":{"primary":{"uri":"task"},"secondary":{"uri":"task"}},
    "pools":{"shared":{"devices":["primary","secondary"],
                         "pool_backing":"fixed"}}
  })";
  iree::testing::TempFilePath path("loom-serve-devices", ".json");
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(json.data(), json.size()),
      iree_allocator_system()));
  const loom_serve_device_configuration_options_t options = {
      .configuration_path = path.path_view(),
      .wait_mode = IREE_SV("active"),
  };
  IREE_ASSERT_OK(Initialize(options));
  EXPECT_TRUE(path.Remove());

  ASSERT_EQ(configuration_.count, 2u);
  ASSERT_EQ(configuration_.pool_count, 1u);
  EXPECT_EQ(String(configuration_.options[0].name), "primary");
  EXPECT_EQ(String(configuration_.options[1].name), "secondary");
  EXPECT_EQ(configuration_.options[0].completion_wait_flags,
            IREE_ASYNC_WAIT_FLAG_ACTIVE);
  EXPECT_EQ(configuration_.options[1].completion_wait_flags,
            IREE_ASYNC_WAIT_FLAG_ACTIVE);
  EXPECT_EQ(configuration_.pools[0].binding.device_mask, 3u);
  EXPECT_FALSE(configuration_.pools[0].options.elastic);
}

TEST_F(DeviceConfigurationOptionsTest, MapsSingleDeviceValuesExactly) {
  iree_string_view_t device_uri = IREE_SV("task");
  const loom_serve_device_configuration_options_t options = {
      .device_uris = {.count = 1, .values = &device_uri},
      .wait_mode = IREE_SV("blocking"),
      .pool_backing = IREE_SV("elastic"),
      .weight_slab_bytes = IREE_SV("1gb"),
      .state_slab_bytes = IREE_SV("64mib"),
      .memory_bytes = IREE_SV("64gib"),
      .workspace_idle_ms = IREE_SV("2000"),
      .parameter_idle_ms = IREE_SV("60000"),
  };
  IREE_ASSERT_OK(Initialize(options));

  ASSERT_EQ(configuration_.count, 1u);
  ASSERT_EQ(configuration_.pool_count, 1u);
  EXPECT_EQ(String(configuration_.options[0].name), "device");
  EXPECT_EQ(String(configuration_.options[0].uri), "task");
  EXPECT_EQ(configuration_.options[0].completion_wait_flags,
            IREE_ASYNC_WAIT_FLAG_NONE);
  const loom_runtime_pool_options_t& pool_options =
      configuration_.pools[0].options;
  EXPECT_TRUE(pool_options.elastic);
  EXPECT_EQ(pool_options.weight_slab_size, UINT64_C(1000000000));
  EXPECT_EQ(pool_options.state_slab_size, UINT64_C(64) << 20);
  EXPECT_EQ(pool_options.memory_limit, UINT64_C(64) << 30);
  EXPECT_EQ(configuration_.idle[0].workspace_timeout,
            iree_make_duration_ms(2000));
  EXPECT_EQ(configuration_.idle[0].parameter_timeout,
            iree_make_duration_ms(60000));
}

TEST_F(DeviceConfigurationOptionsTest, ErrorsNameTheCommandOption) {
  enum class Field {
    kPoolBacking,
    kWeightSlabBytes,
    kStateSlabBytes,
    kMemoryBytes,
    kWorkspaceIdleMs,
    kParameterIdleMs,
    kWaitMode,
  };
  struct InvalidCase {
    // Option field receiving the invalid value.
    Field field;
    // Invalid command-line spelling passed to the production decoder.
    iree_string_view_t value;
    // Expected status code produced by the field parser.
    iree::StatusCode code;
    // Stable option name or diagnostic fragment.
    const char* diagnostic;
  };
  const InvalidCase cases[] = {
      {Field::kPoolBacking, IREE_SV("automatic"),
       iree::StatusCode::kInvalidArgument, "pool_backing"},
      {Field::kWeightSlabBytes, IREE_SV("123fb"),
       iree::StatusCode::kInvalidArgument, "--weight_slab_bytes"},
      {Field::kStateSlabBytes, IREE_SV("1.5gib"),
       iree::StatusCode::kInvalidArgument, "--state_slab_bytes"},
      {Field::kMemoryBytes, IREE_SV("17179869184gib"),
       iree::StatusCode::kOutOfRange, "--memory_bytes"},
      {Field::kWorkspaceIdleMs, IREE_SV("-1"),
       iree::StatusCode::kInvalidArgument, "--workspace_idle_ms"},
      {Field::kParameterIdleMs, IREE_SV("invalid"),
       iree::StatusCode::kInvalidArgument, "--parameter_idle_ms"},
      {Field::kWaitMode, IREE_SV("busy"), iree::StatusCode::kInvalidArgument,
       "--wait_mode"},
  };
  iree_string_view_t device_uri = IREE_SV("task");
  for (const InvalidCase& invalid : cases) {
    SCOPED_TRACE(invalid.diagnostic);
    loom_serve_device_configuration_options_t options = {
        .device_uris = {.count = 1, .values = &device_uri},
    };
    switch (invalid.field) {
      case Field::kPoolBacking:
        options.pool_backing = invalid.value;
        break;
      case Field::kWeightSlabBytes:
        options.weight_slab_bytes = invalid.value;
        break;
      case Field::kStateSlabBytes:
        options.state_slab_bytes = invalid.value;
        break;
      case Field::kMemoryBytes:
        options.memory_bytes = invalid.value;
        break;
      case Field::kWorkspaceIdleMs:
        options.workspace_idle_ms = invalid.value;
        break;
      case Field::kParameterIdleMs:
        options.parameter_idle_ms = invalid.value;
        break;
      case Field::kWaitMode:
        options.wait_mode = invalid.value;
        break;
    }
    ExpectFailure(options, invalid.code, invalid.diagnostic);
  }
}

TEST_F(DeviceConfigurationOptionsTest, RejectsIdleIntervalOverflow) {
  iree_string_view_t device_uri = IREE_SV("task");
  const loom_serve_device_configuration_options_t options = {
      .device_uris = {.count = 1, .values = &device_uri},
      .workspace_idle_ms = IREE_SV("9223372036855"),
  };
  ExpectFailure(options, iree::StatusCode::kOutOfRange,
                "pool idle interval is out of range");
}

TEST_F(DeviceConfigurationOptionsTest,
       FixedBackingRejectsOnlyNonzeroPhysicalBudget) {
  iree_string_view_t device_uri = IREE_SV("task");
  loom_serve_device_configuration_options_t options = {
      .device_uris = {.count = 1, .values = &device_uri},
      .pool_backing = IREE_SV("fixed"),
      .memory_bytes = IREE_SV("1"),
  };
  ExpectFailure(options, iree::StatusCode::kInvalidArgument,
                "a physical budget requires elastic backing");

  options.memory_bytes = IREE_SV("0");
  IREE_ASSERT_OK(Initialize(options));
  ASSERT_EQ(configuration_.pool_count, 1u);
  EXPECT_FALSE(configuration_.pools[0].options.elastic);
  EXPECT_EQ(configuration_.pools[0].options.memory_limit, 0u);
}

}  // namespace
