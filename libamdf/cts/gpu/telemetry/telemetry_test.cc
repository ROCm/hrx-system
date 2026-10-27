// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/telemetry.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <vector>

#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "util/observation.h"

namespace {

class GpuTelemetryTest : public GpuDeviceFixture {
 protected:
  void SetUp() override {
    GpuDeviceFixture::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const void* extension = nullptr;
    ASSERT_EQ(api_->query_extension(AMDF_EXTENSION_TELEMETRY, 1, 1, &extension),
              AMDF_STATUS_OK);
    telemetry_ = static_cast<const amdf_telemetry_api_t*>(extension);
    ASSERT_NO_FATAL_FAILURE(clock_.Initialize(api_));
    uint32_t count = 0;
    ASSERT_EQ(telemetry_->device_enumerate_metrics(device_, 0, nullptr, &count),
              AMDF_STATUS_OK);
    if (count == 0) {
      GTEST_SKIP() << "native device has no supported telemetry";
    }
    metrics_.resize(count);
    ASSERT_EQ(telemetry_->device_enumerate_metrics(device_, metrics_.size(),
                                                   metrics_.data(), &count),
              AMDF_STATUS_OK);
    metrics_.resize(count);
  }

  // Immutable public table borrowed from the runtime-loaded provider.
  const amdf_telemetry_api_t* telemetry_ = nullptr;
  // Caller-owned descriptor catalog; no sensor ownership is retained.
  std::vector<amdf_telemetry_metric_info_t> metrics_;
  // Common profiling timebase used for independent enclosing host reads.
  ObservationHostClock clock_;
};

TEST_F(GpuTelemetryTest, SamplesPhysicalQuantitiesThroughOnePortableCaller) {
  std::map<amdf_telemetry_source_id_t, std::vector<size_t>> sources;
  std::set<std::pair<amdf_telemetry_source_id_t, amdf_telemetry_metric_id_t>>
      ids;
  for (size_t i = 0; i < metrics_.size(); ++i) {
    const auto& metric = metrics_[i];
    EXPECT_NE(metric.source_id, 0u);
    EXPECT_NE(metric.metric_id, 0u);
    ASSERT_NE(metric.name, nullptr);
    ASSERT_NE(metric.source_name, nullptr);
    ASSERT_NE(metric.origin.interface_name, nullptr);
    ASSERT_NE(metric.origin.schema, nullptr);
    ASSERT_NE(metric.origin.field, nullptr);
    EXPECT_EQ(metric.origin.reserved, 0u);
    EXPECT_GT(metric.format.scale_numerator, 0u);
    EXPECT_GT(metric.format.scale_denominator, 0u);
    EXPECT_NE(metric.format.unit, AMDF_OBSERVATION_UNIT_NATIVE);
    EXPECT_TRUE(ids.emplace(metric.source_id, metric.metric_id).second);
    sources[metric.source_id].push_back(i);
  }
  size_t available_count = 0;
  for (const auto& [source, indices] : sources) {
    std::vector<amdf_telemetry_metric_id_t> selected;
    for (size_t index : indices) {
      selected.push_back(metrics_[index].metric_id);
    }
    // One duplicated selection must reflect the same native source query.
    selected.push_back(selected.front());
    std::vector<amdf_telemetry_value_t> values(selected.size());
    amdf_telemetry_sample_t sample = {};
    uint64_t before = 0, after = 0;
    ASSERT_EQ(clock_.Read(&before), AMDF_STATUS_OK);
    ASSERT_EQ(telemetry_->device_read(device_, source, selected.size(),
                                      selected.data(), values.data(), &sample),
              AMDF_STATUS_OK);
    ASSERT_EQ(clock_.Read(&after), AMDF_STATUS_OK);
    EXPECT_LE(before, sample.host_before);
    EXPECT_LE(sample.host_before, sample.host_after);
    EXPECT_LE(sample.host_after, after);
    EXPECT_EQ(values.front().status, values.back().status);
    EXPECT_EQ(std::memcmp(&values.front().value, &values.back().value,
                          sizeof(amdf_observation_value_t)),
              0);
    for (size_t i = 0; i < indices.size(); ++i) {
      const auto& metric = metrics_[indices[i]];
      const auto& value = values[i];
      SCOPED_TRACE(metric.name);
      if (value.status == amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND)) {
        EXPECT_EQ(value.value.u64, 0u);
        continue;
      }
      ASSERT_EQ(value.status, AMDF_STATUS_OK);
      ++available_count;
      double native_value = 0;
      switch (metric.format.kind) {
        case AMDF_OBSERVATION_VALUE_KIND_U64:
          native_value = static_cast<double>(value.value.u64);
          break;
        case AMDF_OBSERVATION_VALUE_KIND_I64:
          native_value = static_cast<double>(value.value.i64);
          break;
        case AMDF_OBSERVATION_VALUE_KIND_F64:
          native_value = value.value.f64;
          break;
        default:
          FAIL() << "unrecognized numeric representation";
      }
      const double physical = native_value * metric.format.scale_numerator /
                              metric.format.scale_denominator;
      EXPECT_TRUE(std::isfinite(physical));
      std::printf("telemetry %s/%s %s: %.9g unit=%u domain=%u instance=%u\n",
                  metric.origin.interface_name, metric.origin.schema,
                  metric.name, physical, metric.format.unit, metric.domain,
                  metric.domain_instance);
      RecordProperty("telemetry_" + std::to_string(indices[i]) + "_value",
                     std::to_string(physical));
    }
  }
  RecordProperty("telemetry_metric_count", std::to_string(metrics_.size()));
  RecordProperty("telemetry_available_count", std::to_string(available_count));
  // Availability is a reading property; sensors can be temporarily unavailable
  // without fabricating zero or making the catalog itself invalid.
}

TEST_F(GpuTelemetryTest, RejectedSelectionPreservesCallerStorage) {
  ASSERT_FALSE(metrics_.empty());
  const auto& metric = metrics_.front();
  const amdf_telemetry_metric_id_t selected[] = {metric.metric_id, UINT64_MAX};
  amdf_telemetry_value_t values[2];
  amdf_telemetry_sample_t sample;
  std::memset(values, 0xa5, sizeof(values));
  std::memset(&sample, 0xa5, sizeof(sample));
  uint8_t original_values[sizeof(values)];
  uint8_t original_sample[sizeof(sample)];
  std::memcpy(original_values, values, sizeof(values));
  std::memcpy(original_sample, &sample, sizeof(sample));
  EXPECT_EQ(telemetry_->device_read(device_, metric.source_id, 2, selected,
                                    values, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(std::memcmp(values, original_values, sizeof(values)), 0);
  EXPECT_EQ(std::memcmp(&sample, original_sample, sizeof(sample)), 0);
  uint32_t count = 123;
  EXPECT_EQ(telemetry_->device_enumerate_metrics(device_, 1, nullptr, &count),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(count, 123u);
}

}  // namespace
