// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/metrics.h"

#include <errno.h>

#include <array>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace {

// Wire facts below were checked against independently compiled Linux native
// declarations at f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931. The fixtures
// contain bytes consumed by the production decoder, not its private descriptor
// tables.
void Store(std::vector<uint8_t>& bytes, size_t offset, size_t width,
           uint64_t value) {
  ASSERT_LE(offset + width, bytes.size());
  for (size_t i = 0; i < width; ++i) {
    bytes[offset + i] = value >> (8 * i);
  }
}

std::vector<uint8_t> Fixed(uint8_t major, uint8_t minor, size_t size) {
  std::vector<uint8_t> bytes(size, 0xff);
  Store(bytes, 0, 2, size);
  bytes[2] = major;
  bytes[3] = minor;
  return bytes;
}

std::vector<amdf_telemetry_metric_info_t> Describe(
    const amdf_gpu_kfd_metrics_t& snapshot) {
  const uint32_t count = amdf_gpu_kfd_metrics_describe(&snapshot, 0, nullptr);
  std::vector<amdf_telemetry_metric_info_t> metrics(count);
  EXPECT_EQ(amdf_gpu_kfd_metrics_describe(&snapshot, count, metrics.data()),
            count);
  return metrics;
}

const amdf_telemetry_metric_info_t* Find(
    const std::vector<amdf_telemetry_metric_info_t>& metrics,
    std::string_view name, uint32_t index = UINT32_MAX) {
  for (const auto& metric : metrics) {
    if (metric.origin.field == name && metric.origin.field_index == index) {
      return &metric;
    }
  }
  return nullptr;
}

TEST(MetricsTest, FixedLayoutFamiliesPreserveValuesAndNativeTime) {
  struct Case {
    // Native format revision.
    uint8_t major;
    // Native content revision.
    uint8_t minor;
    // Native complete structure size.
    uint16_t size;
    // Actual producer used to interpret the fixture.
    uint32_t mp1;
    // Offset of an established temperature field.
    uint16_t temperature;
    // Offset of the driver timestamp.
    uint16_t timestamp;
    // Exact native field name at temperature.
    const char* name;
  };
  const Case cases[] = {
      {1, 0, 80, 0x0b0000, 18, 8, "temperature_hotspot"},
      {1, 1, 96, 0x0b0000, 6, 32, "temperature_hotspot"},
      {1, 2, 104, 0x0d0002, 6, 32, "temperature_hotspot"},
      {1, 3, 120, 0x0e0002, 6, 32, "temperature_hotspot"},
      {1, 4, 288, 0x0d0006, 4, 32, "temperature_hotspot"},
      {1, 5, 360, 0x0d0006, 4, 96, "temperature_hotspot"},
      {1, 6, 1664, 0x0d0006, 4, 24, "temperature_hotspot"},
      {1, 7, 2208, 0x0f0008, 4, 32, "temperature_hotspot"},
      {1, 8, 3872, 0x0f0008, 4, 32, "temperature_hotspot"},
      {2, 0, 120, 0x0b0500, 16, 8, "temperature_gfx"},
      {2, 1, 120, 0x0d0001, 4, 32, "temperature_gfx"},
      {2, 2, 128, 0x0c0000, 4, 32, "temperature_gfx"},
      {2, 3, 152, 0x0b0500, 4, 32, "temperature_gfx"},
      {2, 4, 168, 0x0b0502, 4, 32, "temperature_gfx"},
      {3, 0, 264, 0x0e0000, 4, 104, "temperature_gfx"},
  };
  for (const auto& item : cases) {
    SCOPED_TRACE(::testing::Message()
                 << unsigned(item.major) << '.' << unsigned(item.minor));
    auto bytes = Fixed(item.major, item.minor, item.size);
    Store(bytes, item.temperature, 2, 57);
    Store(bytes, item.timestamp, 8, UINT64_C(0x1020304050607080));
    amdf_gpu_kfd_metrics_t snapshot = {};
    ASSERT_EQ(amdf_gpu_kfd_metrics_decode(item.mp1, bytes.size(), bytes.data(),
                                          &snapshot),
              AMDF_STATUS_OK);
    const auto metrics = Describe(snapshot);
    const auto* metric = Find(metrics, item.name);
    ASSERT_NE(metric, nullptr);
    EXPECT_EQ(metric->format.scale_denominator, item.major == 1 ? 1u : 100u);
    EXPECT_EQ(amdf_gpu_kfd_metrics_validate_selection(metric->source_id, 1,
                                                      &metric->metric_id),
              AMDF_STATUS_OK);
    amdf_telemetry_value_t value = {};
    amdf_telemetry_sample_t sample = {.host_before = 123, .host_after = 456};
    ASSERT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, 1, &metric->metric_id,
                                        &value, &sample),
              AMDF_STATUS_OK);
    EXPECT_EQ(value.status, AMDF_STATUS_OK);
    EXPECT_EQ(value.value.u64, 57u);
    EXPECT_EQ(sample.host_before, 123u);
    EXPECT_EQ(sample.host_after, 456u);
    EXPECT_EQ(sample.driver_time.value, UINT64_C(0x1020304050607080));
    EXPECT_EQ(sample.driver_time.domain, AMDF_TELEMETRY_CLOCK_SYSTEM);
    EXPECT_EQ(sample.driver_time.clock.frequency_hz, 1000000000u);
    EXPECT_EQ(sample.driver_time.point,
              AMDF_TELEMETRY_TIMESTAMP_POINT_DRIVER_READ);
    // The same known schema with a shorter declared size is malformed even
    // when every selected field would still fit in the supplied prefix.
    bytes.pop_back();
    Store(bytes, 0, 2, bytes.size());
    EXPECT_EQ(amdf_gpu_kfd_metrics_decode(item.mp1, bytes.size(), bytes.data(),
                                          &snapshot),
              amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EPROTO));
  }
}

TEST(MetricsTest, Smu14RetainsNpuTrafficFilterAndPowerDomainsWithoutTargets) {
  auto bytes = Fixed(3, 0, 264);
  Store(bytes, 4, 2, 5200);     // GFX TSEN, centi-C.
  Store(bytes, 46, 2, 75);      // First NPU column, percent.
  Store(bytes, 62, 2, 80);      // First CPU core C0 residency, percent.
  Store(bytes, 94, 2, 18000);   // Aggregate DRAM reads, MB/s.
  Store(bytes, 98, 2, 9000);    // NPU reads, MB/s.
  Store(bytes, 112, 4, 37000);  // APU+dGPU budget power, mW.
  Store(bytes, 116, 2, 2400);   // NPU power, mW.
  Store(bytes, 120, 4, 32000);  // APU package power, mW.
  Store(bytes, 174, 2, 600);    // GFX clock TARGET, MHz.
  Store(bytes, 190, 2, 3200);   // CPU core clock TARGET, MHz.
  Store(bytes, 256, 4, 200);    // Filter time constant, us.
  for (uint32_t mp1 : {0x0e0000u, 0x0e0001u, 0x0e0004u, 0x0e0005u}) {
    amdf_gpu_kfd_metrics_t snapshot = {};
    ASSERT_EQ(
        amdf_gpu_kfd_metrics_decode(mp1, bytes.size(), bytes.data(), &snapshot),
        AMDF_STATUS_OK);
    const auto metrics = Describe(snapshot);
    EXPECT_EQ(Find(metrics, "average_gfxclk_frequency"), nullptr);
    EXPECT_EQ(Find(metrics, "current_coreclk", 0), nullptr);
    EXPECT_EQ(Find(metrics, "average_dgpu_power"), nullptr);
    const auto* npu = Find(metrics, "average_ipu_activity", 0);
    const auto* core = Find(metrics, "average_core_c0_activity", 0);
    const auto* traffic = Find(metrics, "average_ipu_reads");
    const auto* dram = Find(metrics, "average_dram_reads");
    const auto* budget = Find(metrics, "average_socket_power");
    const auto* package = Find(metrics, "average_apu_power");
    const auto* temperature = Find(metrics, "temperature_gfx");
    ASSERT_NE(npu, nullptr);
    ASSERT_NE(core, nullptr);
    ASSERT_NE(traffic, nullptr);
    ASSERT_NE(dram, nullptr);
    ASSERT_NE(budget, nullptr);
    ASSERT_NE(package, nullptr);
    ASSERT_NE(temperature, nullptr);
    EXPECT_EQ(npu->domain, AMDF_TELEMETRY_DOMAIN_NPU_COLUMN);
    EXPECT_EQ(core->kind, AMDF_TELEMETRY_METRIC_ACTIVITY_ACTIVE);
    EXPECT_EQ(traffic->domain, AMDF_TELEMETRY_DOMAIN_NPU);
    EXPECT_EQ(traffic->format.unit, AMDF_OBSERVATION_UNIT_BYTES_PER_SECOND);
    EXPECT_EQ(traffic->format.scale_numerator, 1000000u);
    EXPECT_EQ(dram->domain, AMDF_TELEMETRY_DOMAIN_MEMORY_CONTROLLER);
    EXPECT_EQ(dram->domain_instance, AMDF_TELEMETRY_ALL_INSTANCES);
    EXPECT_EQ(budget->domain, AMDF_TELEMETRY_DOMAIN_POWER_BUDGET);
    EXPECT_EQ(package->domain, AMDF_TELEMETRY_DOMAIN_PACKAGE);
    EXPECT_EQ(package->format.scale_denominator, 1000u);
    const amdf_telemetry_metric_id_t ids[] = {
        npu->metric_id, traffic->metric_id, temperature->metric_id,
        package->metric_id, traffic->metric_id};
    amdf_telemetry_value_t values[5] = {};
    amdf_telemetry_sample_t sample = {};
    ASSERT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, 5, ids, values, &sample),
              AMDF_STATUS_OK);
    EXPECT_EQ(values[0].value.u64, 75u);
    EXPECT_EQ(values[1].value.u64, 9000u);
    EXPECT_EQ(values[2].value.u64, 5200u);
    EXPECT_EQ(values[3].value.u64, 32000u);
    EXPECT_EQ(values[4].value.u64, values[1].value.u64);
    for (size_t i : {0u, 1u, 3u, 4u}) {
      EXPECT_EQ(values[i].duration_kind,
                AMDF_TELEMETRY_DURATION_FILTER_TIME_CONSTANT);
      EXPECT_EQ(values[i].duration_ns, 200000u);
    }
    EXPECT_EQ(values[2].duration_kind, AMDF_TELEMETRY_DURATION_UNKNOWN);
  }
}

TEST(MetricsTest, Smu15PreservesOnlyMeaningSharedByBothVersionThreeProducers) {
  auto bytes = Fixed(3, 0, 264);
  Store(bytes, 8, 2, 6100);    // First reported core temperature.
  Store(bytes, 62, 2, 78);     // The same core's C0 residency.
  Store(bytes, 136, 2, 3500);  // The same core's power.
  Store(bytes, 190, 2, 4200);  // Target OR effective clock: same native field.
  Store(bytes, 174, 2, 800);   // Target OR effective GFX clock.
  Store(bytes, 42, 2, 0);      // Valid zero GFX activity.
  for (uint32_t mp1 : {0x0f0000u, 0x0f0005u, 0x0f0009u}) {
    amdf_gpu_kfd_metrics_t snapshot = {};
    ASSERT_EQ(
        amdf_gpu_kfd_metrics_decode(mp1, bytes.size(), bytes.data(), &snapshot),
        AMDF_STATUS_OK);
    const auto metrics = Describe(snapshot);
    const auto* activity = Find(metrics, "average_core_c0_activity", 0);
    const auto* power = Find(metrics, "average_core_power", 0);
    const auto* temperature = Find(metrics, "temperature_core", 0);
    const auto* absent = Find(metrics, "average_core_power", 1);
    const auto* gpu = Find(metrics, "average_gfx_activity");
    ASSERT_NE(activity, nullptr);
    ASSERT_NE(power, nullptr);
    ASSERT_NE(temperature, nullptr);
    ASSERT_NE(absent, nullptr);
    ASSERT_NE(gpu, nullptr);
    EXPECT_EQ(Find(metrics, "average_ipu_power"), nullptr);
    EXPECT_EQ(Find(metrics, "average_all_core_power"), nullptr);
    EXPECT_EQ(Find(metrics, "average_gfxclk_frequency"), nullptr);
    EXPECT_EQ(Find(metrics, "current_coreclk", 0), nullptr);
    for (const auto* core : {activity, power, temperature}) {
      EXPECT_EQ(core->domain, AMDF_TELEMETRY_DOMAIN_CPU_CORE);
      EXPECT_EQ(core->domain_instance, 0u);
      EXPECT_EQ(core->instance_scope, AMDF_TELEMETRY_INSTANCE_SCOPE_SNAPSHOT);
      EXPECT_EQ(core->format.semantic,
                AMDF_OBSERVATION_VALUE_SEMANTIC_UNSPECIFIED);
    }
    EXPECT_EQ(gpu->instance_scope, AMDF_TELEMETRY_INSTANCE_SCOPE_DEVICE);
    const amdf_telemetry_metric_id_t ids[] = {
        temperature->metric_id, activity->metric_id, power->metric_id,
        absent->metric_id, gpu->metric_id};
    amdf_telemetry_value_t values[5] = {};
    amdf_telemetry_sample_t sample = {};
    ASSERT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, 5, ids, values, &sample),
              AMDF_STATUS_OK);
    EXPECT_EQ(values[0].value.u64, 6100u);
    EXPECT_EQ(values[1].value.u64, 78u);
    EXPECT_EQ(values[2].value.u64, 3500u);
    EXPECT_EQ(values[3].status,
              amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
    EXPECT_EQ(values[3].value.u64, 0u);
    EXPECT_EQ(values[4].status, AMDF_STATUS_OK);
    EXPECT_EQ(values[4].value.u64, 0u);
    for (const auto& value : values) {
      EXPECT_EQ(value.duration_kind, AMDF_TELEMETRY_DURATION_UNKNOWN);
    }
  }
}

TEST(MetricsTest,
     VersionThreeUsesFieldSpecificUnmeasuredTemperatureConvention) {
  // AMD SMI 906f061fe55d4714ada216df2a264a2ce383434b establishes this
  // native v3.0 core/skin convention. It does not extend to other zero values.
  for (uint32_t mp1 : {0x0e0000u, 0x0e0001u, 0x0e0004u, 0x0e0005u, 0x0f0000u,
                       0x0f0005u, 0x0f0009u}) {
    auto bytes = Fixed(3, 0, 264);
    Store(bytes, 8, 2, 0);    // Unmeasured core.
    Store(bytes, 10, 2, 1);   // A nonzero native reading, without thresholding.
    Store(bytes, 40, 2, 0);   // Unmeasured skin.
    Store(bytes, 4, 2, 0);    // GFX temperature has no zero sentinel contract.
    Store(bytes, 62, 2, 0);   // Valid idle CPU C0 activity.
    Store(bytes, 136, 2, 0);  // Valid zero core power.
    amdf_gpu_kfd_metrics_t snapshot = {};
    ASSERT_EQ(
        amdf_gpu_kfd_metrics_decode(mp1, bytes.size(), bytes.data(), &snapshot),
        AMDF_STATUS_OK);
    const auto metrics = Describe(snapshot);
    const std::pair<const char*, uint32_t> fields[] = {
        {"temperature_core", 0},         {"temperature_skin", UINT32_MAX},
        {"temperature_core", 1},         {"temperature_core", 2},
        {"temperature_gfx", UINT32_MAX}, {"average_core_c0_activity", 0},
        {"average_core_power", 0}};
    std::vector<amdf_telemetry_metric_id_t> ids;
    for (const auto& [name, index] : fields) {
      const auto* metric = Find(metrics, name, index);
      ASSERT_NE(metric, nullptr);
      ids.push_back(metric->metric_id);
    }
    amdf_telemetry_value_t values[7] = {};
    amdf_telemetry_sample_t sample = {};
    ASSERT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, ids.size(), ids.data(),
                                        values, &sample),
              AMDF_STATUS_OK);
    for (size_t i : {0u, 1u, 3u}) {
      EXPECT_EQ(values[i].status,
                amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
      EXPECT_EQ(values[i].value.u64, 0u);
    }
    EXPECT_EQ(values[2].status, AMDF_STATUS_OK);
    EXPECT_EQ(values[2].value.u64, 1u);
    for (size_t i : {4u, 5u, 6u}) {
      EXPECT_EQ(values[i].status, AMDF_STATUS_OK);
      EXPECT_EQ(values[i].value.u64, 0u);
    }
  }
  auto bytes = Fixed(2, 2, 128);
  Store(bytes, 8, 2,
        0);  // Version 2 does not establish the v3 zero convention.
  amdf_gpu_kfd_metrics_t snapshot = {};
  ASSERT_EQ(amdf_gpu_kfd_metrics_decode(0x0c0000, bytes.size(), bytes.data(),
                                        &snapshot),
            AMDF_STATUS_OK);
  const auto metrics = Describe(snapshot);
  const auto* metric = Find(metrics, "temperature_core", 0);
  ASSERT_NE(metric, nullptr);
  amdf_telemetry_value_t value = {};
  amdf_telemetry_sample_t sample = {};
  ASSERT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, 1, &metric->metric_id, &value,
                                      &sample),
            AMDF_STATUS_OK);
  EXPECT_EQ(value.status, AMDF_STATUS_OK);
  EXPECT_EQ(value.value.u64, 0u);
}

TEST(MetricsTest, ZeroFilterConstantRemainsDifferentFromAnAbsentConstant) {
  for (uint32_t alpha : {0u, 200u, UINT32_MAX}) {
    auto bytes = Fixed(3, 0, 264);
    Store(bytes, 116, 2, 0);  // NPU power is a filtered quantity, even at zero.
    Store(bytes, 256, 4, alpha);
    amdf_gpu_kfd_metrics_t snapshot = {};
    ASSERT_EQ(amdf_gpu_kfd_metrics_decode(0x0e0001, bytes.size(), bytes.data(),
                                          &snapshot),
              AMDF_STATUS_OK);
    const auto metrics = Describe(snapshot);
    const auto* metric = Find(metrics, "average_ipu_power");
    ASSERT_NE(metric, nullptr);
    amdf_telemetry_value_t value = {};
    amdf_telemetry_sample_t sample = {};
    ASSERT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, 1, &metric->metric_id,
                                        &value, &sample),
              AMDF_STATUS_OK);
    EXPECT_EQ(value.status, AMDF_STATUS_OK);
    EXPECT_EQ(value.value.u64, 0u);
    EXPECT_EQ(value.duration_kind,
              alpha == UINT32_MAX
                  ? AMDF_TELEMETRY_DURATION_UNKNOWN
                  : AMDF_TELEMETRY_DURATION_FILTER_TIME_CONSTANT);
    EXPECT_EQ(value.duration_ns,
              alpha == UINT32_MAX ? 0 : uint64_t{alpha} * 1000);
  }
}

TEST(MetricsTest, ApuProducerSelectionRetainsCoreExtentAndActualMemoryClock) {
  for (const auto& [mp1, cores] :
       std::array<std::pair<uint32_t, uint32_t>, 11>{{{0x0b0008, 6},
                                                      {0x0b0500, 4},
                                                      {0x0b0502, 4},
                                                      {0x0c0000, 8},
                                                      {0x0c0001, 8},
                                                      {0x0d0001, 8},
                                                      {0x0d0003, 8},
                                                      {0x0d0008, 8},
                                                      {0x0d0004, 8},
                                                      {0x0d000b, 8},
                                                      {0x0d0005, 0}}}) {
    auto bytes = Fixed(2, 2, 128);
    amdf_gpu_kfd_metrics_t snapshot = {};
    ASSERT_EQ(
        amdf_gpu_kfd_metrics_decode(mp1, bytes.size(), bytes.data(), &snapshot),
        AMDF_STATUS_OK);
    const auto metrics = Describe(snapshot);
    EXPECT_EQ(Find(metrics, "temperature_core", cores), nullptr);
    if (cores != 0) {
      ASSERT_NE(Find(metrics, "temperature_core", cores - 1), nullptr);
    }
    const auto* memory = Find(metrics, "current_uclk");
    if (mp1 == 0x0c0000 || mp1 == 0x0c0001) {
      ASSERT_NE(memory, nullptr);
      EXPECT_EQ(memory->kind, AMDF_TELEMETRY_METRIC_CLOCK_MEMORY_CONTROLLER);
      EXPECT_NE(Find(metrics, "average_fclk_frequency"), nullptr);
      EXPECT_EQ(Find(metrics, "average_gfx_power"), nullptr);
    } else {
      EXPECT_EQ(Find(metrics, "average_fclk_frequency"), nullptr);
    }
  }
}

TEST(MetricsTest, SecondaryDieSyntheticZeroIsNotAnEnergyOrPowerMeasurement) {
  auto bytes = Fixed(1, 3, 120);
  Store(bytes, 22, 2, 0);
  Store(bytes, 24, 8, 0);
  amdf_gpu_kfd_metrics_t snapshot = {};
  ASSERT_EQ(amdf_gpu_kfd_metrics_decode(0x0d0002, bytes.size(), bytes.data(),
                                        &snapshot),
            AMDF_STATUS_OK);
  const auto metrics = Describe(snapshot);
  EXPECT_EQ(Find(metrics, "average_socket_power"), nullptr);
  EXPECT_EQ(Find(metrics, "energy_accumulator"), nullptr);
  EXPECT_NE(Find(metrics, "current_gfxclk"), nullptr);
}

// Complete native v1.9 objects serialized by DECLARE_SMU_METRICS_CLASS, with
// hotspot=37, memory temperature=42, socket power=77 and GFX clock[0]=1234.
constexpr char kSmu13[] =
    "ea040109320000000100200200000000250001042002000000002a000108200200000000ff"
    "ff010c2003000000004d00"
    "0110200a00000000ffff0114200a00000000ffff0118600800000000ffffffffffffffff01"
    "1c600000000000ffffffff"
    "ffffffff0120600400000000ffffffffffffffff0124600000000000ffffffffffffffff01"
    "28600000000000ffffffff"
    "ffffffff012c600000000000ffffffffffffffff0130600000000000ffffffffffffffff01"
    "34600000000000ffffffff"
    "ffffffff0138600000000000ffffffffffffffff013c400000000000ffffffff0140200000"
    "000000ffff014420070000"
    "0000ffff0148200000000000ffff014c200600000000ffff0150600000000000ffffffffff"
    "ffffff0154600000000000"
    "ffffffffffffffff0158600000000000ffffffffffffffff015c600800000000ffffffffff"
    "ffffff0160600000000000"
    "ffffffffffffffff0164600000000000ffffffffffffffff0168600000000000ffffffffff"
    "ffffff016c600000000000"
    "ffffffffffffffff0170600000000000ffffffffffffffff0874600900000000ffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "0878600900000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffff087c200000000000ffffffffff"
    "ffffffffffffffffffffff"
    "0180600500000000ffffffffffffffff0884200100000000d204ffffffffffffffffffffff"
    "ffffff0488200100000000"
    "ffffffffffffffff048c200100000000ffffffffffffffff0490200100000000ffffffffff"
    "ffffff0194200100000000"
    "ffff019c600000000000ffffffffffffffff08a0400a00000000ffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffff28a4200a00000000ffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffff04a8200a00000000ffffffffffffffff08ac600000000000ffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffff08b060000000"
    "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffff08b4600000000000ffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffff08b860000000"
    "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffff08bc600000000000ffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffff08c020020000"
    "0000ffffffffffffffffffffffffffffffff04c8200200000000ffffffffffffffff08cc20"
    "0200000000ffffffffffff"
    "ffffffffffffffffffff";

constexpr char kSmu15[] =
    "98040109330000000100200200000000250001042002000000002a000108200200000000ff"
    "ff0cc0200200000000ffff"
    "ffffffffffffffffffffffffffffffffffffffffffff02c4200200000000ffffffff02c820"
    "0200000000ffffffff08cc"
    "200200000000ffffffffffffffffffffffffffffffff010c2003000000004d000110200a00"
    "000000ffff0114200a0000"
    "0000ffff0118600800000000ffffffffffffffff011c600000000000ffffffffffffffff01"
    "20600400000000ffffffff"
    "ffffffff0124600000000000ffffffffffffffff0128600000000000ffffffffffffffff01"
    "2c600000000000ffffffff"
    "ffffffff0130600000000000ffffffffffffffff0134600000000000ffffffffffffffff01"
    "38600000000000ffffffff"
    "ffffffff013c400000000000ffffffff0140200000000000ffff0144200700000000ffff01"
    "48200000000000ffff014c"
    "200600000000ffff0150600000000000ffffffffffffffff0154600000000000ffffffffff"
    "ffffff0258600000000000"
    "ffffffffffffffffffffffffffffffff025c600800000000ffffffffffffffffffffffffff"
    "ffffff0160600000000000"
    "ffffffffffffffff0164600000000000ffffffffffffffff0168600000000000ffffffffff"
    "ffffff016c600000000000"
    "ffffffffffffffff0170600000000000ffffffffffffffff087c200000000000ffffffffff"
    "ffffffffffffffffffffff"
    "0174600900000000ffffffffffffffff0178600900000000ffffffffffffffff0180600500"
    "000000ffffffffffffffff"
    "0884200100000000d204ffffffffffffffffffffffffffff0288200100000000ffffffff04"
    "8c200100000000ffffffff"
    "ffffffff0490200100000000ffffffffffffffff0294200100000000ffffffff019c600000"
    "000000ffffffffffffffff"
    "08a0400a00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffff28a4200a00000000"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff04a8200a00"
    "000000ffffffffffffffff"
    "08ac600000000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffff08b0600000000000ffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "08b4600000000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffff08b8600000000000ffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "08bc600000000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffff";

std::vector<uint8_t> FromHex(std::string_view hex) {
  auto nibble = [](char c) -> uint8_t {
    return c >= 'a' ? static_cast<uint8_t>(c - 'a' + 10)
                    : static_cast<uint8_t>(c - '0');
  };
  std::vector<uint8_t> bytes(hex.size() / 2);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = (nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]);
  }
  return bytes;
}

TEST(MetricsTest, ActualTaggedNativeObjectsPreserveInlineArraysAndSentinels) {
  for (const auto& [mp1, hex] :
       std::array<std::pair<uint32_t, const char*>, 4>{{{0x0d0006, kSmu13},
                                                        {0x0d000c, kSmu13},
                                                        {0x0d000e, kSmu13},
                                                        {0x0f0008, kSmu15}}}) {
    auto bytes = FromHex(hex);
    amdf_gpu_kfd_metrics_t snapshot = {};
    ASSERT_EQ(
        amdf_gpu_kfd_metrics_decode(mp1, bytes.size(), bytes.data(), &snapshot),
        AMDF_STATUS_OK);
    const auto metrics = Describe(snapshot);
    const auto* hotspot = Find(metrics, "temperature_hotspot");
    const auto* memory = Find(metrics, "temperature_mem");
    const auto* power = Find(metrics, "curr_socket_power");
    const auto* clock = Find(metrics, "current_gfxclk", 0);
    const auto* energy = Find(metrics, "energy_accumulator");
    ASSERT_NE(hotspot, nullptr);
    ASSERT_NE(memory, nullptr);
    ASSERT_NE(power, nullptr);
    ASSERT_NE(clock, nullptr);
    ASSERT_NE(energy, nullptr);
    EXPECT_EQ(hotspot->domain, AMDF_TELEMETRY_DOMAIN_PACKAGE);
    EXPECT_EQ(energy->format.scale_denominator, 65536u);
    EXPECT_EQ(energy->format.counter_bit_count, 64u);
    EXPECT_EQ(energy->format.semantic,
              AMDF_OBSERVATION_VALUE_SEMANTIC_CUMULATIVE);
    const auto* memory_clock =
        Find(metrics, "current_uclk", mp1 == 0x0f0008 ? 0 : UINT32_MAX);
    ASSERT_NE(memory_clock, nullptr);
    const amdf_telemetry_metric_id_t ids[] = {
        hotspot->metric_id, memory->metric_id, power->metric_id,
        clock->metric_id, energy->metric_id};
    EXPECT_EQ(
        amdf_gpu_kfd_metrics_validate_selection(snapshot.source_id, 5, ids),
        AMDF_STATUS_OK);
    amdf_telemetry_value_t values[5] = {};
    amdf_telemetry_sample_t sample = {};
    ASSERT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, 5, ids, values, &sample),
              AMDF_STATUS_OK);
    EXPECT_EQ(values[0].value.u64, 37u);
    EXPECT_EQ(values[1].value.u64, 42u);
    EXPECT_EQ(values[2].value.u64, 77u);
    EXPECT_EQ(values[3].value.u64, 1234u);
    EXPECT_EQ(values[4].status,
              amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  }
}

TEST(MetricsTest, ExactIntegerEnergyAndIndependentClockNamespaces) {
  auto bytes = Fixed(1, 4, 288);
  Store(bytes, 24, 8, UINT64_C(0xdeadbeefcafe1234));
  Store(bytes, 32, 8, 1000);
  Store(bytes, 232, 8, 1000);
  amdf_gpu_kfd_metrics_t snapshot = {};
  ASSERT_EQ(amdf_gpu_kfd_metrics_decode(0x0d0006, bytes.size(), bytes.data(),
                                        &snapshot),
            AMDF_STATUS_OK);
  const auto metrics = Describe(snapshot);
  const auto* energy = Find(metrics, "energy_accumulator");
  ASSERT_NE(energy, nullptr);
  amdf_telemetry_value_t value = {};
  amdf_telemetry_sample_t sample = {};
  ASSERT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, 1, &energy->metric_id, &value,
                                      &sample),
            AMDF_STATUS_OK);
  EXPECT_EQ(value.value.u64, UINT64_C(0xdeadbeefcafe1234));
  EXPECT_EQ(sample.driver_time.value, sample.source_time.value);
  EXPECT_EQ(sample.driver_time.domain, AMDF_TELEMETRY_CLOCK_SYSTEM);
  EXPECT_EQ(sample.source_time.domain, AMDF_TELEMETRY_CLOCK_SOURCE);
  EXPECT_EQ(sample.driver_time.clock.frequency_hz, 1000000000u);
  EXPECT_EQ(sample.source_time.clock.frequency_hz, 100000000u);
}

TEST(MetricsTest, MalformedTaggedInputsPreserveTheValidatedOutput) {
  const auto original_bytes = FromHex(kSmu13);
  amdf_gpu_kfd_metrics_t original;
  std::memset(&original, 0xa5, sizeof(original));
  const auto check = [&](const std::vector<uint8_t>& bytes) {
    auto output = original;
    EXPECT_EQ(amdf_gpu_kfd_metrics_decode(0x0d0006, bytes.size(), bytes.data(),
                                          &output),
              amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EPROTO));
    EXPECT_EQ(std::memcmp(&output, &original, sizeof(output)), 0);
  };
  // Every truncated prefix, including a lying smaller header, is rejected.
  for (size_t length = 1; length < original_bytes.size(); ++length) {
    std::vector<uint8_t> bytes(original_bytes.begin(),
                               original_bytes.begin() + length);
    if (length >= 2) {
      Store(bytes, 0, 2, length);
    }
    check(bytes);
  }
  auto bytes = original_bytes;
  Store(bytes, 4, 4, UINT32_MAX);  // Negative signed attribute count.
  check(bytes);
  bytes = original_bytes;
  Store(bytes, 8, 8, UINT64_C(0x100220001));  // Reserved upper tag bits.
  check(bytes);
  bytes = original_bytes;
  Store(bytes, 8, 8, UINT64_C(0x02800001));  // Unknown type, no byte extent.
  check(bytes);
  bytes = original_bytes;
  Store(bytes, 8, 8, UINT64_C(0x02200000));  // Zero elements.
  check(bytes);
  bytes = original_bytes;
  Store(bytes, 8, 8, UINT64_C(0x03200001));  // Watts on temperature ID zero.
  check(bytes);
  bytes = original_bytes;
  Store(bytes, 18, 8, UINT64_C(0x02200001));  // Duplicate ID zero.
  check(bytes);
  bytes = original_bytes;
  bytes.push_back(0);
  Store(bytes, 0, 2, bytes.size());  // Bytes beyond the declared attributes.
  check(bytes);
}

TEST(MetricsTest, UnknownTaggedFieldsCanBeSkippedOnlyWithAnEstablishedExtent) {
  auto bytes = FromHex(kSmu13);
  const size_t offset = bytes.size();
  bytes.resize(offset + 10);
  Store(bytes, 0, 2, bytes.size());
  Store(bytes, 4, 4, 51);
  // Unknown ID 900, signed byte type, two inline bytes. Its semantics do not
  // become a guessed public quantity, but its understood extent is skippable.
  Store(bytes, offset, 8, (UINT64_C(1) << 20) | (UINT64_C(900) << 10) | 2);
  bytes[offset + 8] = 255;
  bytes[offset + 9] = 0;
  amdf_gpu_kfd_metrics_t snapshot = {};
  ASSERT_EQ(amdf_gpu_kfd_metrics_decode(0x0d0006, bytes.size(), bytes.data(),
                                        &snapshot),
            AMDF_STATUS_OK);
  const auto metrics = Describe(snapshot);
  ASSERT_NE(Find(metrics, "temperature_hotspot"), nullptr);
  auto original = FromHex(kSmu13);
  amdf_gpu_kfd_metrics_t baseline = {};
  ASSERT_EQ(amdf_gpu_kfd_metrics_decode(0x0d0006, original.size(),
                                        original.data(), &baseline),
            AMDF_STATUS_OK);
  EXPECT_EQ(metrics.size(),
            amdf_gpu_kfd_metrics_describe(&baseline, 0, nullptr));
}

TEST(MetricsTest, SelectionFailureNeverPublishesAnEarlierValue) {
  auto bytes = Fixed(3, 0, 264);
  Store(bytes, 4, 2, 0);
  amdf_gpu_kfd_metrics_t snapshot = {};
  ASSERT_EQ(amdf_gpu_kfd_metrics_decode(0x0f0000, bytes.size(), bytes.data(),
                                        &snapshot),
            AMDF_STATUS_OK);
  const auto metrics = Describe(snapshot);
  ASSERT_FALSE(metrics.empty());
  const amdf_telemetry_metric_id_t ids[] = {metrics[0].metric_id, UINT64_MAX};
  amdf_telemetry_value_t values[2];
  amdf_telemetry_sample_t sample;
  std::memset(values, 0xa5, sizeof(values));
  std::memset(&sample, 0x5a, sizeof(sample));
  const auto original_sample = sample;
  const std::array<amdf_telemetry_value_t, 2> original = {values[0], values[1]};
  EXPECT_EQ(amdf_gpu_kfd_metrics_validate_selection(snapshot.source_id, 2, ids),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(amdf_gpu_kfd_metrics_read(&snapshot, 2, ids, values, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(std::memcmp(values, original.data(), sizeof(values)), 0);
  EXPECT_EQ(std::memcmp(&sample, &original_sample, sizeof(sample)), 0);
  EXPECT_EQ(amdf_gpu_kfd_metrics_validate_selection(0, 1, ids),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
}

TEST(MetricsTest, UnknownProducerAndSchemaDoNotInheritAnotherInterpretation) {
  auto bytes = Fixed(3, 0, 264);
  amdf_gpu_kfd_metrics_t snapshot = {};
  for (uint32_t mp1 : {0u, 0xdeadbeu, 0x0d0000u}) {
    EXPECT_EQ(
        amdf_gpu_kfd_metrics_decode(mp1, bytes.size(), bytes.data(), &snapshot),
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  }
  bytes[3] = 1;
  EXPECT_EQ(amdf_gpu_kfd_metrics_decode(0x0e0000, bytes.size(), bytes.data(),
                                        &snapshot),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
}

}  // namespace
