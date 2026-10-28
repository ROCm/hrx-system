// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/telemetry.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"

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
    SetSmuVersion(13, 0, 0);
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

  void SetSmuVersion(uint32_t major, uint32_t minor, uint32_t revision) {
    const auto directory =
        root_ / "dev/char/226:128/device/ip_discovery/die/0/1/0";
    std::filesystem::create_directories(directory);
    const char* names[] = {"major", "minor", "revision"};
    const uint32_t values[] = {major, minor, revision};
    for (size_t i = 0; i < 3; ++i) {
      std::ofstream file(directory / names[i]);
      file << values[i] << '\n';
      ASSERT_TRUE(file.good());
    }
  }

  std::vector<amdf_telemetry_metric_info_t> Enumerate() {
    uint32_t count = 0;
    EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(nullptr, &endpoint_, 0,
                                                       nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_telemetry_metric_info_t> metrics(count);
    EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(
                  nullptr, &endpoint_, metrics.size(), metrics.data(), &count),
              AMDF_STATUS_OK);
    return metrics;
  }

  void WriteSnapshot(const std::vector<uint8_t>& bytes) {
    std::ofstream file(root_ / "dev/char/226:128/device/gpu_metrics",
                       std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    ASSERT_TRUE(file.good());
  }

  std::vector<uint8_t> ApuSnapshot() {
    std::vector<uint8_t> bytes(264, 0xff);
    // Native v3.0: 264 bytes, GPU temperature 51 C, APU package power 23 W,
    // driver CLOCK_BOOTTIME timestamp 123 ns. The other fields are unavailable.
    bytes[0] = 8;
    bytes[1] = 1;
    bytes[2] = 3;
    bytes[3] = 0;
    const uint16_t temperature = 5100;
    const uint32_t power = 23000;
    const uint64_t timestamp = 123;
    std::memcpy(bytes.data() + 4, &temperature, sizeof(temperature));
    std::memcpy(bytes.data() + 120, &power, sizeof(power));
    std::memcpy(bytes.data() + 104, &timestamp, sizeof(timestamp));
    return bytes;
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
    EXPECT_EQ(metrics[i].instance_scope, AMDF_TELEMETRY_INSTANCE_SCOPE_DEVICE);
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

TEST_F(TelemetryTest, Smu14ApuTargetsAreNotMeasuredClocks) {
  Write("power1_input", "32942000\n");
  Write("temp2_input", "41000\n");
  // The SMU14 APU producer writes these frequency targets, SocketPower's
  // APU+dGPU power budget, and separate GPU/SoC TSEN temperature values.
  // Deliberately malformed target attributes prove enumeration never reads
  // them as if they were actual operating clocks.
  Write("freq1_input", "a configured target is not a measured clock\n");
  Write("freq2_input", "a configured target is not a measured clock\n");
  for (uint32_t revision : {0, 1, 4, 5}) {
    SCOPED_TRACE(revision);
    SetSmuVersion(14, 0, revision);
    const auto metrics = Enumerate();
    ASSERT_EQ(metrics.size(), 4u);
    for (const auto& metric : metrics) {
      EXPECT_STREQ(metric.origin.schema, "hwmon-abi:smu14-apu");
      const std::string field = metric.origin.field;
      if (field == "power1_average" || field == "power1_input") {
        EXPECT_EQ(metric.kind, AMDF_TELEMETRY_METRIC_POWER);
        EXPECT_EQ(metric.domain, AMDF_TELEMETRY_DOMAIN_POWER_BUDGET);
        EXPECT_EQ(metric.format.semantic,
                  AMDF_OBSERVATION_VALUE_SEMANTIC_FILTERED);
      } else {
        EXPECT_EQ(metric.kind, AMDF_TELEMETRY_METRIC_TEMPERATURE);
        EXPECT_EQ(metric.domain, field == "temp1_input"
                                     ? AMDF_TELEMETRY_DOMAIN_GPU
                                     : AMDF_TELEMETRY_DOMAIN_SOC);
      }
    }
  }
}

TEST_F(TelemetryTest, RenoirMemorySensorNamesItsActualFabricClock) {
  for (uint32_t revision : {0, 1}) {
    SCOPED_TRACE(revision);
    SetSmuVersion(12, 0, revision);
    const auto metrics = Enumerate();
    ASSERT_EQ(metrics.size(), 4u);
    ASSERT_STREQ(metrics[1].origin.field, "freq2_input");
    EXPECT_EQ(metrics[1].kind, AMDF_TELEMETRY_METRIC_CLOCK_FABRIC);
    EXPECT_EQ(metrics[1].domain, AMDF_TELEMETRY_DOMAIN_SOC);
  }
}

TEST_F(TelemetryTest, ApuTemperatureAndPowerRetainTheirNativeRegions) {
  Write("temp2_input", "41000\n");
  const std::array<std::array<uint32_t, 3>, 9> versions = {{
      {11, 0, 8},
      {11, 5, 0},
      {11, 5, 2},
      {13, 0, 1},
      {13, 0, 3},
      {13, 0, 4},
      {13, 0, 5},
      {13, 0, 8},
      {13, 0, 11},
  }};
  for (const auto& version : versions) {
    SCOPED_TRACE(::testing::PrintToString(version));
    SetSmuVersion(version[0], version[1], version[2]);
    const auto metrics = Enumerate();
    ASSERT_EQ(metrics.size(), 5u);
    EXPECT_EQ(metrics[2].domain, AMDF_TELEMETRY_DOMAIN_POWER_BUDGET);
    EXPECT_EQ(metrics[3].kind, AMDF_TELEMETRY_METRIC_TEMPERATURE);
    EXPECT_EQ(metrics[3].domain, AMDF_TELEMETRY_DOMAIN_GPU);
    EXPECT_EQ(metrics[4].kind, AMDF_TELEMETRY_METRIC_TEMPERATURE);
    EXPECT_EQ(metrics[4].domain, AMDF_TELEMETRY_DOMAIN_SOC);
  }
}

TEST_F(TelemetryTest, MultiDieClockRequiresNativeFirmwareCapability) {
  Write("temp2_input", "41000\n");
  const auto directory = root_ / "dev/char/226:128/device/fw_version";
  ASSERT_TRUE(std::filesystem::create_directories(directory));
  for (uint32_t revision : {6, 12}) {
    SCOPED_TRACE(revision);
    SetSmuVersion(13, 0, revision);
    const uint32_t minimum = revision == 6 ? 0x00552f00 : 0x00561900;
    for (const uint32_t firmware : {minimum - 1, minimum}) {
      SCOPED_TRACE(firmware);
      char text[16];
      std::snprintf(text, sizeof(text), "0x%08x\n", firmware);
      {
        std::ofstream file(directory / "smc_fw_version");
        file << text;
        ASSERT_TRUE(file.good());
      }
      const auto metrics = Enumerate();
      const bool has_clock = firmware == minimum;
      ASSERT_EQ(metrics.size(), has_clock ? 3u : 2u);
      EXPECT_EQ(metrics.front().kind, has_clock
                                          ? AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE
                                          : AMDF_TELEMETRY_METRIC_CLOCK_MEMORY);
      EXPECT_EQ(metrics.back().kind, AMDF_TELEMETRY_METRIC_TEMPERATURE_HOTSPOT);
      EXPECT_EQ(metrics.back().domain, AMDF_TELEMETRY_DOMAIN_PACKAGE);
    }
    ASSERT_TRUE(std::filesystem::remove(directory / "smc_fw_version"));
    EXPECT_EQ(Enumerate().size(), 2u);
  }
}

TEST_F(TelemetryTest, UnestablishedProducerNeverInheritsAnotherMeaning) {
  SetSmuVersion(99, 0, 0);
  EXPECT_TRUE(Enumerate().empty());
  std::filesystem::remove_all(root_ / "dev/char/226:128/device/ip_discovery");
  EXPECT_TRUE(Enumerate().empty());
}

TEST_F(TelemetryTest, MalformedProducerIdentityPreservesCatalog) {
  const auto directory =
      root_ / "dev/char/226:128/device/ip_discovery/die/0/1/0";
  for (const char* text : {"256\n", "0x0d\n", "-1\n", "13\n0\n"}) {
    SCOPED_TRACE(text);
    {
      std::ofstream file(directory / "major");
      file << text;
      ASSERT_TRUE(file.good());
    }
    amdf_telemetry_metric_info_t metric = {.metric_id = 42};
    uint32_t count = 123;
    EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(nullptr, &endpoint_, 1,
                                                       &metric, &count),
              amdf_linux_error(EPROTO));
    EXPECT_EQ(metric.metric_id, 42u);
    EXPECT_EQ(count, 123u);
  }
}

TEST_F(TelemetryTest, ChangedMeaningRejectsOldSourceIdentity) {
  const auto metrics = Enumerate();
  ASSERT_EQ(metrics.size(), 4u);
  const auto& power = metrics[2];
  ASSERT_EQ(power.kind, AMDF_TELEMETRY_METRIC_POWER);
  SetSmuVersion(14, 0, 0);
  amdf_telemetry_value_t value = {.value = {.u64 = 123}};
  amdf_telemetry_sample_t sample = {.host_before = 456};
  EXPECT_EQ(amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, power.source_id, 1,
                                        &power.metric_id, &value, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH));
  EXPECT_EQ(value.value.u64, 123u);
  EXPECT_EQ(sample.host_before, 456u);
}

TEST_F(TelemetryTest, CombinedSnapshotUsesTheSameSelectedSourceContract) {
  SetSmuVersion(14, 0, 0);
  WriteSnapshot(ApuSnapshot());
  const auto catalog = Enumerate();
  std::vector<amdf_telemetry_metric_id_t> selected;
  amdf_telemetry_source_id_t source = 0;
  for (const auto& metric : catalog) {
    if (std::strcmp(metric.source_name, "gpu_metrics") != 0) {
      continue;
    }
    if (std::strcmp(metric.origin.field, "temperature_gfx") == 0 ||
        std::strcmp(metric.origin.field, "average_apu_power") == 0) {
      if (source != 0) {
        EXPECT_EQ(source, metric.source_id);
      }
      source = metric.source_id;
      selected.push_back(metric.metric_id);
      EXPECT_NE(
          metric.source_flags & AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_SNAPSHOT, 0u);
    }
  }
  ASSERT_EQ(selected.size(), 2u);
  selected.push_back(selected.front());
  amdf_telemetry_value_t values[3] = {};
  amdf_telemetry_sample_t sample = {};
  ASSERT_EQ(amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, source, 3,
                                        selected.data(), values, &sample),
            AMDF_STATUS_OK);
  EXPECT_EQ(values[0].value.u64, 5100u);
  EXPECT_EQ(values[1].value.u64, 23000u);
  EXPECT_EQ(values[2].value.u64, values[0].value.u64);
  EXPECT_EQ(sample.driver_time.domain, AMDF_TELEMETRY_CLOCK_SYSTEM);
  EXPECT_EQ(sample.driver_time.value, 123u);
  EXPECT_LE(sample.host_before, sample.host_after);
  amdf_telemetry_metric_info_t prefix;
  uint32_t count = 0;
  EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(nullptr, &endpoint_, 1,
                                                     &prefix, &count),
            amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
  EXPECT_EQ(count, catalog.size());
  EXPECT_EQ(prefix.source_id, catalog.front().source_id);
  EXPECT_EQ(prefix.metric_id, catalog.front().metric_id);
}

TEST_F(TelemetryTest, SnapshotWorksWithoutAScalarSensorRoute) {
  SetSmuVersion(15, 0, 0);
  std::filesystem::remove_all(sensor_.parent_path());
  WriteSnapshot(ApuSnapshot());
  const auto catalog = Enumerate();
  ASSERT_FALSE(catalog.empty());
  for (const auto& metric : catalog) {
    EXPECT_STREQ(metric.source_name, "gpu_metrics");
  }
}

TEST_F(TelemetryTest, MalformedSnapshotPublishesNeitherScalarPrefixNorValues) {
  SetSmuVersion(14, 0, 0);
  WriteSnapshot(ApuSnapshot());
  const auto catalog = Enumerate();
  ASSERT_FALSE(catalog.empty());
  const auto metric = catalog.back();
  ASSERT_STREQ(metric.source_name, "gpu_metrics");
  auto bytes = ApuSnapshot();
  bytes.pop_back();
  WriteSnapshot(bytes);
  std::array<amdf_telemetry_metric_info_t, 4> output;
  std::memset(output.data(), 0xa5, sizeof(output));
  const auto original = output;
  uint32_t count = 123;
  EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(
                nullptr, &endpoint_, output.size(), output.data(), &count),
            amdf_linux_error(EPROTO));
  EXPECT_EQ(count, 123u);
  EXPECT_EQ(std::memcmp(output.data(), original.data(), sizeof(output)), 0);
  amdf_telemetry_value_t value = {.value = {.u64 = 456}};
  amdf_telemetry_sample_t sample = {.host_before = 789};
  EXPECT_EQ(amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metric.source_id,
                                        1, &metric.metric_id, &value, &sample),
            amdf_linux_error(EPROTO));
  EXPECT_EQ(value.value.u64, 456u);
  EXPECT_EQ(sample.host_before, 789u);
}

TEST_F(TelemetryTest, InvalidSnapshotSelectionPrecedesNativeRead) {
  SetSmuVersion(14, 0, 0);
  WriteSnapshot(ApuSnapshot());
  const auto catalog = Enumerate();
  ASSERT_FALSE(catalog.empty());
  const auto metric = catalog.back();
  ASSERT_STREQ(metric.source_name, "gpu_metrics");
  WriteSnapshot(
      {});  // A valid selection now encounters an absent native table.
  const amdf_telemetry_metric_id_t ids[] = {metric.metric_id, UINT64_MAX};
  amdf_telemetry_value_t values[2] = {};
  amdf_telemetry_sample_t sample = {};
  EXPECT_EQ(amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metric.source_id,
                                        2, ids, values, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  auto bytes = ApuSnapshot();
  bytes[3] = 1;
  WriteSnapshot(bytes);
  EXPECT_EQ(amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metric.source_id,
                                        1, ids, values, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH));
  // Changing the producer with the same wire layout changes interpretation.
  WriteSnapshot(ApuSnapshot());
  SetSmuVersion(15, 0, 0);
  EXPECT_EQ(amdf_gpu_umd_read_telemetry(nullptr, &endpoint_, metric.source_id,
                                        1, ids, values, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH));
}

}  // namespace
