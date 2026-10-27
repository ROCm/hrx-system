// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/telemetry.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/platform/linux/endpoint.h"

namespace {

class TelemetryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* temporary = std::getenv("TEST_TMPDIR");
    ASSERT_NE(temporary, nullptr);
    std::string pattern = std::string(temporary) + "/telemetry-XXXXXX";
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    root_ = pattern;
    sensor_ = root_ / "dev/char/226:128/device/hwmon/hwmon42";
    ASSERT_TRUE(std::filesystem::create_directories(sensor_));
    Write("name", "amdgpu\n");
    Write("freq1_input", "600000000\n");
    Write("freq2_input", "1000000000\n");
    Write("power1_average", "32942000\n");
    Write("temp1_input", "-2500\n");
    instance_.sysfs_descriptor = open(root_.c_str(), O_RDONLY | O_DIRECTORY);
    ASSERT_GE(instance_.sysfs_descriptor, 0);
    endpoint_.instance = &instance_;
    endpoint_.info.id.words[0] = (UINT64_C(226) << 32) | 128;
    endpoint_.info.native_identity.type =
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE;
    endpoint_.info.native_identity.value.linux_device.major = 226;
    endpoint_.info.native_identity.value.linux_device.minor = 128;
  }

  void TearDown() override {
    if (instance_.sysfs_descriptor >= 0) {
      EXPECT_EQ(close(instance_.sysfs_descriptor), 0);
    }
    if (!root_.empty()) {
      std::filesystem::remove_all(root_);
    }
  }

  void Write(const char* name, const char* value) {
    std::ofstream file(sensor_ / name);
    file << value;
    ASSERT_TRUE(file.good());
  }

  std::vector<amdf_telemetry_metric_info_t> Enumerate() {
    std::array<amdf_telemetry_metric_info_t, 16> metrics = {};
    uint32_t count = 0;
    EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(
                  nullptr, &endpoint_, metrics.size(), metrics.data(), &count),
              AMDF_STATUS_OK);
    return {metrics.begin(), metrics.begin() + count};
  }

  // Private sysfs fixture containing actual attribute files.
  std::filesystem::path root_;
  // Selected amdgpu hwmon directory under the fixture's device identity.
  std::filesystem::path sensor_;
  // Native instance facade borrowing only the fixture root descriptor.
  amdf_platform_instance_t instance_ = {.sysfs_descriptor = -1};
  // Native endpoint identity used by the actual provider path.
  amdf_platform_endpoint_t endpoint_ = {};
};

TEST_F(TelemetryTest, PreservesUnitsScopeSignedValuesAndSeparateSources) {
  const auto metrics = Enumerate();
  ASSERT_EQ(metrics.size(), 4u);
  const std::array<uint64_t, 3> unsigned_values = {600000000, 1000000000,
                                                   32942000};
  for (size_t i = 0; i < metrics.size(); ++i) {
    SCOPED_TRACE(metrics[i].name);
    EXPECT_EQ(
        metrics[i].source_flags & AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_SNAPSHOT,
        0u);
    if (i != 0) {
      EXPECT_NE(metrics[i].source_id, metrics[i - 1].source_id);
    }
    const amdf_telemetry_metric_id_t ids[] = {metrics[i].metric_id,
                                              metrics[i].metric_id};
    amdf_telemetry_value_t values[2] = {};
    amdf_telemetry_sample_t sample = {};
    ASSERT_EQ(
        amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metrics[i].source_id,
                                    2, ids, values, &sample),
        AMDF_STATUS_OK);
    EXPECT_EQ(values[0].status, AMDF_STATUS_OK);
    EXPECT_EQ(values[0].value.u64, values[1].value.u64);
    if (i < unsigned_values.size()) {
      EXPECT_EQ(values[0].value.u64, unsigned_values[i]);
    } else {
      EXPECT_EQ(values[0].value.i64, -2500);
      EXPECT_EQ(metrics[i].format.kind, AMDF_OBSERVATION_VALUE_KIND_I64);
      EXPECT_EQ(metrics[i].format.scale_denominator, 1000u);
    }
    EXPECT_LE(sample.host_before, sample.host_after);
    EXPECT_EQ(sample.driver_time.domain, AMDF_TELEMETRY_CLOCK_NONE);
    EXPECT_EQ(sample.source_time.domain, AMDF_TELEMETRY_CLOCK_NONE);
    EXPECT_EQ(values[0].duration_kind, AMDF_TELEMETRY_DURATION_UNKNOWN);
  }
  EXPECT_EQ(metrics[2].kind, AMDF_TELEMETRY_METRIC_POWER);
  EXPECT_EQ(metrics[2].domain, AMDF_TELEMETRY_DOMAIN_PACKAGE);
  EXPECT_EQ(metrics[2].format.scale_denominator, 1000000u);
  EXPECT_EQ(metrics[2].format.semantic,
            AMDF_OBSERVATION_VALUE_SEMANTIC_UNSPECIFIED);
}

TEST_F(TelemetryTest, CountAndTruncationAreExplicitProtocols) {
  uint32_t count = 99;
  EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(nullptr, &endpoint_, 0,
                                                     nullptr, &count),
            AMDF_STATUS_OK);
  EXPECT_EQ(count, 4u);
  amdf_telemetry_metric_info_t prefix = {};
  EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(nullptr, &endpoint_, 1,
                                                     &prefix, &count),
            amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
  EXPECT_EQ(count, 4u);
  EXPECT_STREQ(prefix.origin.field, "freq1_input");
}

TEST_F(TelemetryTest, ZeroIsValidAndMissingReadingIsUnavailable) {
  const auto metrics = Enumerate();
  ASSERT_FALSE(metrics.empty());
  Write("freq1_input", "0\n");
  amdf_telemetry_value_t value = {};
  amdf_telemetry_sample_t sample = {};
  EXPECT_EQ(
      amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metrics[0].source_id, 1,
                                  &metrics[0].metric_id, &value, &sample),
      AMDF_STATUS_OK);
  EXPECT_EQ(value.status, AMDF_STATUS_OK);
  EXPECT_EQ(value.value.u64, 0u);
  ASSERT_TRUE(std::filesystem::remove(sensor_ / "freq1_input"));
  EXPECT_EQ(
      amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metrics[0].source_id, 1,
                                  &metrics[0].metric_id, &value, &sample),
      AMDF_STATUS_OK);
  EXPECT_EQ(value.status, amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(value.value.u64, 0u);
}

TEST_F(TelemetryTest, LateNativeFailurePublishesNoCatalogPrefix) {
  Write("temp1_input", "not a temperature\n");
  std::array<amdf_telemetry_metric_info_t, 8> metrics;
  std::memset(metrics.data(), 0xa5, sizeof(metrics));
  const auto original = metrics;
  uint32_t count = 123;
  EXPECT_NE(amdf_gpu_umd_enumerate_telemetry_metrics(
                nullptr, &endpoint_, metrics.size(), metrics.data(), &count),
            AMDF_STATUS_OK);
  EXPECT_EQ(count, 123u);
  EXPECT_EQ(std::memcmp(metrics.data(), original.data(), sizeof(metrics)), 0);
}

TEST_F(TelemetryTest, InvalidNumberOrIdentityPreservesEveryOutput) {
  const auto metrics = Enumerate();
  ASSERT_FALSE(metrics.empty());
  for (const char* text :
       {"-1\n", " -1\n", "+1\n", "18446744073709551616\n", "1\n2\n", ""}) {
    SCOPED_TRACE(text);
    Write("freq1_input", text);
    amdf_telemetry_value_t value;
    amdf_telemetry_sample_t sample;
    std::memset(&value, 0xa5, sizeof(value));
    std::memset(&sample, 0xa5, sizeof(sample));
    const auto original_value = value;
    const auto original_sample = sample;
    EXPECT_NE(
        amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metrics[0].source_id,
                                    1, &metrics[0].metric_id, &value, &sample),
        AMDF_STATUS_OK);
    EXPECT_EQ(std::memcmp(&value, &original_value, sizeof(value)), 0);
    EXPECT_EQ(std::memcmp(&sample, &original_sample, sizeof(sample)), 0);
  }
  // A known metric from another attribute cannot be used to combine sources.
  const amdf_telemetry_metric_id_t ids[] = {metrics[0].metric_id,
                                            metrics[1].metric_id};
  amdf_telemetry_value_t values[2] = {};
  amdf_telemetry_sample_t sample = {};
  EXPECT_EQ(
      amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metrics[0].source_id, 2,
                                  ids, values, &sample),
      amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
}

TEST_F(TelemetryTest, MissingServiceIsNotMissingDevice) {
  std::filesystem::remove_all(sensor_.parent_path());
  uint32_t count = 99;
  EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(nullptr, &endpoint_, 0,
                                                     nullptr, &count),
            AMDF_STATUS_OK);
  EXPECT_EQ(count, 0u);
  std::filesystem::remove_all(root_ / "dev/char/226:128/device");
  count = 99;
  EXPECT_NE(amdf_gpu_umd_enumerate_telemetry_metrics(nullptr, &endpoint_, 0,
                                                     nullptr, &count),
            AMDF_STATUS_OK);
  EXPECT_EQ(count, 99u);
}

}  // namespace
