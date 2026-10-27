// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/telemetry.h"

#include <array>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/gpu/umd/wddm/device.h"

namespace {

// Native dependency responses; the production provider performs selection,
// unit/domain description, output publication, and timestamp bracketing.
struct NativeTelemetry {
  // Result of the next native read.
  amdf_status_t status = AMDF_STATUS_OK;
  // Raw native response copied only on success.
  amdf_wkmi_bridge_gpu_telemetry_t snapshot = {};
  // Number of explicit native queries.
  uint32_t read_count = 0;
} native;

amdf_status_t AMDF_WKMI_BRIDGE_CALL ReadNativeTelemetry(
    uint32_t adapter, uint32_t device, uint32_t physical_adapter_index,
    amdf_wkmi_bridge_gpu_telemetry_t* out_snapshot) {
  EXPECT_EQ(adapter, 100u);
  EXPECT_EQ(device, 200u);
  EXPECT_EQ(physical_adapter_index, 0u);
  ++native.read_count;
  if (amdf_status_is_ok(native.status)) {
    *out_snapshot = native.snapshot;
  }
  return native.status;
}

class TelemetryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native = {};
    for (uint32_t id : {1, 2, 8, 19, 20, 23, 27}) {
      native.snapshot.supported[id] = 1;
    }
    api_.gpu_telemetry_read = ReadNativeTelemetry;
    device_.wkmi_adapter.api = &api_;
    device_.adapter = 100;
    device_.device = 200;
  }

  std::array<amdf_telemetry_metric_info_t, 7> Enumerate() {
    std::array<amdf_telemetry_metric_info_t, 7> metrics = {};
    uint32_t count = 0;
    EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(
                  &device_, nullptr, metrics.size(), metrics.data(), &count),
              AMDF_STATUS_OK);
    EXPECT_EQ(count, metrics.size());
    return metrics;
  }

  // Native service dependency table borrowed by the provider under test.
  amdf_wkmi_bridge_api_t api_ = {};
  // Already-live native object facade; no driver handles are acquired here.
  amdf_gpu_umd_device_t device_ = {};
};

TEST_F(TelemetryTest, DescribesPhysicalMeasurementsWithoutSensorControls) {
  const auto metrics = Enumerate();
  EXPECT_EQ(native.read_count, 1u);
  for (const auto& metric : metrics) {
    EXPECT_EQ(metric.source_id, metrics[0].source_id);
    EXPECT_EQ(metric.instance_scope, AMDF_TELEMETRY_INSTANCE_SCOPE_DEVICE);
    EXPECT_NE(metric.source_flags & AMDF_TELEMETRY_SOURCE_FLAG_NATIVE_SNAPSHOT,
              0u);
    EXPECT_EQ(metric.format.semantic,
              AMDF_OBSERVATION_VALUE_SEMANTIC_UNSPECIFIED);
    EXPECT_EQ(metric.format.kind, AMDF_OBSERVATION_VALUE_KIND_I64);
  }
  EXPECT_EQ(metrics[0].kind, AMDF_TELEMETRY_METRIC_CLOCK_COMPUTE);
  EXPECT_EQ(metrics[0].format.unit, AMDF_OBSERVATION_UNIT_HERTZ);
  EXPECT_EQ(metrics[0].format.scale_numerator, 1000000u);
  EXPECT_EQ(metrics[1].kind, AMDF_TELEMETRY_METRIC_CLOCK_MEMORY);
  EXPECT_EQ(metrics[2].kind, AMDF_TELEMETRY_METRIC_TEMPERATURE_EDGE);
  EXPECT_EQ(metrics[3].format.unit, AMDF_OBSERVATION_UNIT_RATIO);
  EXPECT_EQ(metrics[3].format.scale_denominator, 100u);
  EXPECT_EQ(metrics[5].kind, AMDF_TELEMETRY_METRIC_POWER);
  EXPECT_EQ(metrics[5].domain, AMDF_TELEMETRY_DOMAIN_PACKAGE);
  EXPECT_EQ(metrics[5].format.unit, AMDF_OBSERVATION_UNIT_WATTS);
  EXPECT_EQ(metrics[6].kind, AMDF_TELEMETRY_METRIC_TEMPERATURE_HOTSPOT);
}

TEST_F(TelemetryTest, SelectionAndDuplicatesUseOneNativeSnapshot) {
  const auto metrics = Enumerate();
  native.read_count = 0;
  native.snapshot.values[1] = 600;
  native.snapshot.values[23] = 37;
  const amdf_telemetry_metric_id_t ids[] = {
      metrics[5].metric_id, metrics[0].metric_id, metrics[5].metric_id};
  amdf_telemetry_value_t values[3] = {};
  amdf_telemetry_sample_t sample = {};
  ASSERT_EQ(amdf_gpu_umd_read_telemetry(&device_, nullptr, metrics[0].source_id,
                                        3, ids, values, &sample),
            AMDF_STATUS_OK);
  EXPECT_EQ(native.read_count, 1u);
  EXPECT_EQ(values[0].value.u64, 37u);
  EXPECT_EQ(values[1].value.u64, 600u);
  EXPECT_EQ(values[2].value.u64, 37u);
  EXPECT_EQ(values[0].status, AMDF_STATUS_OK);
  EXPECT_EQ(values[0].duration_kind, AMDF_TELEMETRY_DURATION_UNKNOWN);
  EXPECT_LE(sample.host_before, sample.host_after);
  EXPECT_EQ(sample.driver_time.domain, AMDF_TELEMETRY_CLOCK_NONE);
  EXPECT_EQ(sample.source_time.domain, AMDF_TELEMETRY_CLOCK_NONE);
}

TEST_F(TelemetryTest, UnsupportedAndUnavailableRemainDistinctFromZero) {
  const auto metrics = Enumerate();
  native.snapshot.supported[2] = 0;
  native.snapshot.values[2] = 123;
  native.snapshot.supported[8] = 0;
  native.snapshot.values[8] = UINT32_MAX;
  const amdf_telemetry_metric_id_t ids[] = {
      metrics[0].metric_id, metrics[1].metric_id, metrics[2].metric_id};
  amdf_telemetry_value_t values[3] = {};
  amdf_telemetry_sample_t sample = {};
  ASSERT_EQ(amdf_gpu_umd_read_telemetry(&device_, nullptr, metrics[0].source_id,
                                        3, ids, values, &sample),
            AMDF_STATUS_OK);
  EXPECT_EQ(values[0].status, AMDF_STATUS_OK);
  EXPECT_EQ(values[0].value.u64, 0u);
  for (size_t i = 1; i < 3; ++i) {
    EXPECT_EQ(values[i].status,
              amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
    EXPECT_EQ(values[i].value.u64, 0u);
  }
  uint32_t count = 99;
  native.status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(amdf_gpu_umd_enumerate_telemetry_metrics(&device_, nullptr, 0,
                                                     nullptr, &count),
            AMDF_STATUS_OK);
  EXPECT_EQ(count, 0u);
}

TEST_F(TelemetryTest, NativeSupportDistinguishesMinusOneFromUnavailable) {
  const auto metrics = Enumerate();
  native.snapshot.values[8] = UINT32_MAX;
  native.snapshot.values[27] = static_cast<uint32_t>(-5);
  const amdf_telemetry_metric_id_t ids[] = {metrics[2].metric_id,
                                            metrics[6].metric_id};
  amdf_telemetry_value_t values[2] = {};
  amdf_telemetry_sample_t sample = {};
  ASSERT_EQ(amdf_gpu_umd_read_telemetry(&device_, nullptr, metrics[0].source_id,
                                        2, ids, values, &sample),
            AMDF_STATUS_OK);
  EXPECT_EQ(values[0].status, AMDF_STATUS_OK);
  EXPECT_EQ(values[1].status, AMDF_STATUS_OK);
  EXPECT_EQ(values[0].value.i64, -1);
  EXPECT_EQ(values[1].value.i64, -5);
}

TEST_F(TelemetryTest, NativeFailureAndInvalidSelectionPublishNothing) {
  const auto metrics = Enumerate();
  native.read_count = 0;
  amdf_telemetry_value_t values[2];
  amdf_telemetry_sample_t sample;
  std::memset(values, 0xa5, sizeof(values));
  std::memset(&sample, 0xa5, sizeof(sample));
  const std::array<amdf_telemetry_value_t, 2> original = {values[0], values[1]};
  const auto original_sample = sample;
  const amdf_telemetry_metric_id_t ids[] = {metrics[0].metric_id, UINT64_MAX};
  EXPECT_EQ(amdf_gpu_umd_read_telemetry(&device_, nullptr, metrics[0].source_id,
                                        2, ids, values, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(native.read_count, 0u);
  native.status = amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, 0xc0000022);
  EXPECT_EQ(amdf_gpu_umd_read_telemetry(&device_, nullptr, metrics[0].source_id,
                                        1, ids, values, &sample),
            native.status);
  EXPECT_EQ(std::memcmp(values, original.data(), sizeof(values)), 0);
  EXPECT_EQ(std::memcmp(&sample, &original_sample, sizeof(sample)), 0);
  auto descriptors = metrics;
  uint32_t count = 99;
  EXPECT_EQ(
      amdf_gpu_umd_enumerate_telemetry_metrics(
          &device_, nullptr, descriptors.size(), descriptors.data(), &count),
      native.status);
  EXPECT_EQ(count, 99u);
  EXPECT_EQ(std::memcmp(descriptors.data(), metrics.data(), sizeof(metrics)),
            0);
}

}  // namespace
