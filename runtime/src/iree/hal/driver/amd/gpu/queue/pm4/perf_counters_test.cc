// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/perf_counters.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

constexpr uint32_t kGuard = 0xd8c37f91;
constexpr uint32_t kComputeFlush = 0xc0004600;
constexpr uint32_t kUconfig = 0xc0017900;
constexpr uint32_t kCopyData = 0xc0044000;
// Performance-register source, memory destination, write confirmation.
constexpr uint32_t kCopyRegisterControl = 0x00100204;

class PerfCountersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
        {11, 5, 1}, &profile_));
    config_ = {};
    config_.profile = &profile_;
    config_.engine_count = 2;
    Select("SQG_WAVES");
    Select("SQG_ITEMS");
    storage_.fill(kGuard);
  }

  void Select(const char* name) {
    const iree_hal_amd_gpu_pm4_perf_counter_event_t* event =
        iree_hal_amd_gpu_pm4_perf_counter_profile_find(
            &profile_, iree_make_cstring_view(name));
    ASSERT_NE(event, nullptr) << name;
    if (event->block == IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_ENGINE) {
      config_.engine_counters[config_.engine_counter_count++] = event;
    } else {
      config_.instance_counters[config_.instance_counter_count++] = event;
    }
  }

  // Adds the instruction counters on three workgroup processors.
  void SelectInstances() {
    Select("SQ_WAVES");
    Select("SQ_INSTS_ALL");
    config_.instances[config_.instance_count++] =
        iree_hal_amd_gpu_pm4_perf_counter_instance(0, 0, 0);
    config_.instances[config_.instance_count++] =
        iree_hal_amd_gpu_pm4_perf_counter_instance(0, 1, 4);
    config_.instances[config_.instance_count++] =
        iree_hal_amd_gpu_pm4_perf_counter_instance(1, 0, 2);
  }

  uint32_t* words() { return storage_.data() + 1; }

  std::vector<uint32_t> Emitted(uint32_t* end) {
    EXPECT_EQ(storage_.front(), kGuard);
    for (auto* word = end; word != storage_.data() + storage_.size(); ++word) {
      EXPECT_EQ(*word, kGuard);
    }
    return std::vector<uint32_t>(words(), end);
  }

  static uint32_t Register(uint32_t address) {
    return (address - 0x30000) >> 2;
  }

  // GFX11.5 register table under test.
  iree_hal_amd_gpu_pm4_perf_counter_profile_t profile_;
  // Two engine counters on two engines; tests add instance counters.
  iree_hal_amd_gpu_pm4_perf_counter_config_t config_;
  // Command storage with a guard on both sides of the emitted range.
  std::array<uint32_t, 512> storage_;
};

// Message of a failed status, consuming it.
std::string StatusText(iree_status_t status) {
  std::string message;
  const iree_allocator_t allocator = iree_allocator_system();
  char* buffer = nullptr;
  iree_host_size_t length = 0;
  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
    message.assign(buffer, length);
    iree_allocator_free(allocator, buffer);
  }
  iree_status_ignore(status);
  return message;
}

TEST_F(PerfCountersTest, OnlyGfx1151IsMarkedVerified) {
  // The one target the sequences ran on.
  EXPECT_TRUE(profile_.verified);
  // FORCE_EN would count every queue on the device.
  EXPECT_EQ(profile_.block_filter_default & 1u, 0u);
  // Sibling steppings and GFX11.0 were never run. They share the event
  // catalog and emit the same words.
  SelectInstances();
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_perf_counter_config_validate(&config_));
  const uint64_t address = UINT64_C(0x0000007700aa1000);
  const std::vector<uint32_t> start = Emitted(
      iree_hal_amd_gpu_pm4_perf_counter_emit_start(words(), &config_, address));
  storage_.fill(kGuard);
  const std::vector<uint32_t> sample =
      Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_sample(words(), &config_,
                                                            address));
  storage_.fill(kGuard);
  const std::vector<uint32_t> stop =
      Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_stop(words(), &config_));
  for (const iree_hal_amd_gpu_gfxip_version_t version :
       {iree_hal_amd_gpu_gfxip_version_t{11, 5, 0},
        iree_hal_amd_gpu_gfxip_version_t{11, 5, 2},
        iree_hal_amd_gpu_gfxip_version_t{11, 5, 3},
        iree_hal_amd_gpu_gfxip_version_t{11, 5, 40},
        iree_hal_amd_gpu_gfxip_version_t{11, 0, 0},
        iree_hal_amd_gpu_gfxip_version_t{11, 0, 3}}) {
    SCOPED_TRACE(::testing::Message()
                 << "gfx" << version.major << "." << version.minor << "."
                 << version.stepping);
    iree_hal_amd_gpu_pm4_perf_counter_profile_t profile;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
        version, &profile));
    EXPECT_FALSE(profile.verified);
    EXPECT_EQ(profile.events, profile_.events);
    EXPECT_EQ(profile.event_count, profile_.event_count);
    config_.profile = &profile;
    storage_.fill(kGuard);
    EXPECT_EQ(Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_start(
                  words(), &config_, address)),
              start);
    storage_.fill(kGuard);
    EXPECT_EQ(Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_sample(
                  words(), &config_, address)),
              sample);
    storage_.fill(kGuard);
    EXPECT_EQ(
        Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_stop(words(), &config_)),
        stop);
  }
  iree_hal_amd_gpu_pm4_perf_counter_profile_t gfx120;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
      {12, 0, 0}, &gfx120));
  EXPECT_FALSE(gfx120.verified);
}

TEST_F(PerfCountersTest, TargetsWithoutATableAreRefusedNamingTheGap) {
  struct Case {
    iree_hal_amd_gpu_gfxip_version_t version;
    const char* target;
    const char* gap;
  };
  for (const Case& c : {
           Case{{9, 0, 10}, "gfx9.0.10", "has no SQG block"},
           Case{{9, 4, 2}, "gfx9.4.2", "basic_counters.xml"},
           Case{{10, 3, 0}, "gfx10.3.0", "navi10_enum.h"},
           Case{{11, 7, 0}, "gfx11.7.0", "no register database or event table"},
           Case{{12, 5, 0}, "gfx12.5.0", "AID_PERFMON_CNTL"},
           Case{{13, 0, 0}, "gfx13.0.0", "no register table has been written"},
       }) {
    iree_hal_amd_gpu_pm4_perf_counter_profile_t profile;
    std::memset(&profile, 0xa5, sizeof(profile));
    iree_status_t status = iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
        c.version, &profile);
    EXPECT_TRUE(iree_status_is_unimplemented(status)) << c.target;
    const std::string message = StatusText(status);
    // Builds that keep status text name the target and what it lacks.
    if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) {
      EXPECT_NE(
          message.find(std::string("no PM4 performance counter table for ") +
                       c.target),
          std::string::npos)
          << message;
      EXPECT_NE(message.find(c.gap), std::string::npos) << message;
      EXPECT_EQ(message.find("gfx1151"), std::string::npos) << message;
    }
    // A refusal never leaves a half-filled table behind.
    EXPECT_EQ(profile.control, 0u);
    EXPECT_EQ(profile.events, nullptr);
    EXPECT_EQ(profile.event_count, 0u);
    EXPECT_FALSE(profile.verified);
  }
}

TEST_F(PerfCountersTest, Gfx120RenumbersEveryEventOnTheSameRegisters) {
  iree_hal_amd_gpu_pm4_perf_counter_profile_t gfx120;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
      {12, 0, 1}, &gfx120));
  // Same catalog, in the same order, with the GFX12 numbers.
  struct Event {
    const char* name;
    uint32_t gfx11;
    uint32_t gfx12;
  };
  const Event events[] = {
      {"SQG_WAVES", 0x14, 0x13},       {"SQG_WAVES_32", 0x15, 0x14},
      {"SQG_WAVES_64", 0x16, 0x15},    {"SQG_ITEMS", 0x18, 0x17},
      {"SQG_BUSY_CYCLES", 0x10, 0x0f}, {"SQG_WAVE_CYCLES", 0x11, 0x10},
      {"SQG_CYCLES", 0x0f, 0x0e},      {"SQG_WAVES_SAVED", 0x0d, 0x0c},
      {"SQ_WAVES", 0x04, 0x04},        {"SQ_INSTS_VALU", 0x3e, 0x32},
      {"SQ_INSTS_SALU", 0x3a, 0x2e},   {"SQ_INSTS_ALL", 0x30, 0x27},
  };
  ASSERT_EQ(profile_.event_count, IREE_ARRAYSIZE(events));
  ASSERT_EQ(gfx120.event_count, IREE_ARRAYSIZE(events));
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(events); ++i) {
    const iree_string_view_t name = iree_make_cstring_view(events[i].name);
    EXPECT_TRUE(iree_string_view_equal(profile_.events[i].name, name)) << i;
    EXPECT_TRUE(iree_string_view_equal(gfx120.events[i].name, name)) << i;
    EXPECT_EQ(profile_.events[i].select, events[i].gfx11) << events[i].name;
    EXPECT_EQ(gfx120.events[i].select, events[i].gfx12) << events[i].name;
    EXPECT_EQ(gfx120.events[i].block, profile_.events[i].block);
    EXPECT_EQ(gfx120.events[i].cycles, profile_.events[i].cycles);
  }

  // The sequences differ from GFX11.5 only in the select values.
  SelectInstances();
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_perf_counter_config_validate(&config_));
  auto gfx120_config = config_;
  gfx120_config.profile = &gfx120;
  for (uint32_t i = 0; i < config_.engine_counter_count; ++i) {
    gfx120_config.engine_counters[i] =
        iree_hal_amd_gpu_pm4_perf_counter_profile_find(
            &gfx120, config_.engine_counters[i]->name);
    ASSERT_NE(gfx120_config.engine_counters[i], nullptr);
  }
  for (uint32_t i = 0; i < config_.instance_counter_count; ++i) {
    gfx120_config.instance_counters[i] =
        iree_hal_amd_gpu_pm4_perf_counter_profile_find(
            &gfx120, config_.instance_counters[i]->name);
    ASSERT_NE(gfx120_config.instance_counters[i], nullptr);
  }
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_perf_counter_config_validate(&gfx120_config));
  const uint64_t state = UINT64_C(0x0000007700aa1000);
  std::vector<uint32_t> expected = Emitted(
      iree_hal_amd_gpu_pm4_perf_counter_emit_start(words(), &config_, state));
  storage_.fill(kGuard);
  const std::vector<uint32_t> start =
      Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_start(
          words(), &gfx120_config, state));
  ASSERT_EQ(start.size(), expected.size());
  // SQG_WAVES, SQG_ITEMS, then SQ_WAVES and SQ_INSTS_ALL on even selects.
  const uint32_t selects[][3] = {
      {Register(0x36740), 0x14, 0x13},
      {Register(0x36744), 0x18, 0x17},
      {Register(0x36700), 0x04, 0x04},
      {Register(0x36708), 0x30, 0x27},
  };
  uint32_t replaced = 0;
  for (size_t i = 0; i + 2 < expected.size(); ++i) {
    for (const auto& select : selects) {
      if (expected[i] == kUconfig && expected[i + 1] == select[0] &&
          expected[i + 2] == select[1]) {
        expected[i + 2] = select[2];
        ++replaced;
      }
    }
  }
  EXPECT_EQ(replaced, 4u);
  EXPECT_EQ(start, expected);
  // Sampling and stopping name no event: identical word for word.
  storage_.fill(kGuard);
  const std::vector<uint32_t> sample =
      Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_sample(
          words(), &config_, UINT64_C(0x0000007700bb2000)));
  storage_.fill(kGuard);
  EXPECT_EQ(Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_sample(
                words(), &gfx120_config, UINT64_C(0x0000007700bb2000))),
            sample);
  storage_.fill(kGuard);
  const std::vector<uint32_t> stop =
      Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_stop(words(), &config_));
  storage_.fill(kGuard);
  EXPECT_EQ(Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_stop(
                words(), &gfx120_config)),
            stop);
  // A recording prepared for one layout is not replayed with the other.
  EXPECT_NE(iree_hal_amd_gpu_pm4_perf_counter_config_signature(&gfx120_config),
            iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config_));
}

TEST_F(PerfCountersTest, EventsAreFoundByExactName) {
  const iree_hal_amd_gpu_pm4_perf_counter_event_t* waves =
      iree_hal_amd_gpu_pm4_perf_counter_profile_find(&profile_,
                                                     IREE_SV("SQG_WAVES"));
  ASSERT_NE(waves, nullptr);
  EXPECT_EQ(waves->select, 0x14u);
  EXPECT_EQ(waves->block, IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_ENGINE);
  EXPECT_FALSE(waves->cycles);
  const iree_hal_amd_gpu_pm4_perf_counter_event_t* busy =
      iree_hal_amd_gpu_pm4_perf_counter_profile_find(
          &profile_, IREE_SV("SQG_BUSY_CYCLES"));
  ASSERT_NE(busy, nullptr);
  EXPECT_TRUE(busy->cycles);
  const iree_hal_amd_gpu_pm4_perf_counter_event_t* instructions =
      iree_hal_amd_gpu_pm4_perf_counter_profile_find(&profile_,
                                                     IREE_SV("SQ_INSTS_ALL"));
  ASSERT_NE(instructions, nullptr);
  EXPECT_EQ(instructions->select, 0x30u);
  EXPECT_EQ(instructions->block,
            IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_INSTANCE);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_perf_counter_profile_find(
                &profile_, IREE_SV("sqg_waves")),
            nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_perf_counter_profile_find(&profile_,
                                                           IREE_SV("SQG_WAVE")),
            nullptr);
}

TEST_F(PerfCountersTest, ValidationRejectsUnusableSelections) {
  IREE_EXPECT_OK(iree_hal_amd_gpu_pm4_perf_counter_config_validate(&config_));

  iree_hal_amd_gpu_pm4_perf_counter_config_t config = config_;
  config.engine_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_gpu_pm4_perf_counter_config_validate(&config));

  config = config_;
  config.engine_counter_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_pm4_perf_counter_config_validate(&config));

  // An instance counter without workgroup processors has nothing to read.
  Select("SQ_WAVES");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_pm4_perf_counter_config_validate(&config_));

  // An event in the wrong block's list would program the wrong select.
  config = config_;
  config.instance_counter_count = 0;
  config.engine_counters[1] = config_.instance_counters[0];
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_pm4_perf_counter_config_validate(&config));
}

TEST_F(PerfCountersTest, SignatureFollowsTheSampleLayout) {
  const uint64_t base =
      iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config_);
  EXPECT_NE(base, 0u);
  EXPECT_EQ(base, iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config_));

  iree_hal_amd_gpu_pm4_perf_counter_config_t config = config_;
  std::swap(config.engine_counters[0], config.engine_counters[1]);
  EXPECT_NE(base, iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config));

  config = config_;
  config.engine_count = 1;
  EXPECT_NE(base, iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config));

  // Instances no sample reads do not change the layout.
  config = config_;
  config.instances[config.instance_count++] = 4;
  EXPECT_EQ(base, iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config));

  SelectInstances();
  const uint64_t with_instances =
      iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config_);
  EXPECT_NE(base, with_instances);
  config_.instances[1] = iree_hal_amd_gpu_pm4_perf_counter_instance(0, 1, 3);
  EXPECT_NE(with_instances,
            iree_hal_amd_gpu_pm4_perf_counter_config_signature(&config_));
}

TEST_F(PerfCountersTest, StartReadsStateThenProgramsAndCountsFromZero) {
  SelectInstances();
  const uint64_t state = UINT64_C(0x0000009a12345670);
  const std::vector<uint32_t> emitted = Emitted(
      iree_hal_amd_gpu_pm4_perf_counter_emit_start(words(), &config_, state));
  ASSERT_EQ(emitted.size(),
            iree_hal_amd_gpu_pm4_perf_counter_start_word_count(&config_));
  const std::vector<uint32_t> expected = {
      // Join earlier work, broadcast.
      kComputeFlush,
      0x80000407,
      kUconfig,
      Register(0x30800),
      0xe0000000,
      // What another collector would have left, before any write.
      kCopyData,
      kCopyRegisterControl,
      0x36020 >> 2,
      0,
      0x12345670,
      0x9a,
      kCopyData,
      kCopyRegisterControl,
      0x36760 >> 2,
      0,
      0x12345674,
      0x9a,
      kCopyData,
      kCopyRegisterControl,
      0x36740 >> 2,
      0,
      0x12345678,
      0x9a,
      // Disable and clear, stage enables, both filters at their default.
      kUconfig,
      Register(0x36020),
      0,
      kUconfig,
      Register(0x36780),
      0x55,
      kUconfig,
      Register(0x36788),
      0x1fffe,
      kUconfig,
      Register(0x36768),
      0x1fffe,
      // SQG_WAVES and SQG_ITEMS in engine slots 0 and 1.
      kUconfig,
      Register(0x36740),
      0x14,
      kUconfig,
      Register(0x36744),
      0x18,
      kUconfig,
      Register(0x36760),
      0x55,
      // SQ_WAVES and SQ_INSTS_ALL in the even instance selects 0 and 2.
      kUconfig,
      Register(0x36700),
      0x04,
      kUconfig,
      Register(0x36708),
      0x30,
      kUconfig,
      Register(0x30800),
      0xe0000000,
      // COMPUTE_PERFCOUNT_ENABLE = 1 on this queue.
      0xc0017602,
      0x20b,
      1,
      // Clear, then count.
      kUconfig,
      Register(0x36020),
      0,
      kUconfig,
      Register(0x36020),
      1,
      kComputeFlush,
      0x80000407,
  };
  EXPECT_EQ(emitted, expected);
}

TEST_F(PerfCountersTest, SampleLatchesThenCopiesEveryInstance) {
  SelectInstances();
  const uint64_t target = UINT64_C(0x0000007700001000);
  const std::vector<uint32_t> emitted = Emitted(
      iree_hal_amd_gpu_pm4_perf_counter_emit_sample(words(), &config_, target));
  ASSERT_EQ(emitted.size(),
            iree_hal_amd_gpu_pm4_perf_counter_sample_word_count(&config_));
  // Six values: two engine counters on two engines, then two instance
  // counters on three workgroup processors.
  EXPECT_EQ(iree_hal_amd_gpu_pm4_perf_counter_value_count(&config_), 10u);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_perf_counter_sample_byte_length(&config_),
            80u);
  auto copy = [&](uint32_t address, uint32_t byte_offset) {
    return std::vector<uint32_t>{
        kCopyData, kCopyRegisterControl,     address >> 2,
        0,         0x00001000 + byte_offset, 0x77};
  };
  std::vector<uint32_t> expected = {
      // The disable point: a compute join, then the sample strobe.
      kComputeFlush, 0x80000407, kUconfig,          Register(0x36020),
      0x401,         kUconfig,   Register(0x30800), 0xe0000000,
      kComputeFlush, 0x80000407,
  };
  auto append = [&](const std::vector<uint32_t>& more) {
    expected.insert(expected.end(), more.begin(), more.end());
  };
  // Engine 0: array and instance broadcast.
  append({kUconfig, Register(0x30800), 0x60000000});
  append(copy(0x34790, 0));   // SQG_WAVES low, value 0
  append(copy(0x34794, 4));   // SQG_WAVES high
  append(copy(0x34798, 16));  // SQG_ITEMS low, value 2
  append(copy(0x3479c, 20));
  // Engine 1.
  append({kUconfig, Register(0x30800), 0x60010000});
  append(copy(0x34790, 8));  // SQG_WAVES, value 1
  append(copy(0x34794, 12));
  append(copy(0x34798, 24));  // SQG_ITEMS, value 3
  append(copy(0x3479c, 28));
  // Workgroup processors: engine 0 array 0 unit 0, engine 0 array 1 unit 4,
  // engine 1 array 0 unit 2. Values 4-6 are SQ_WAVES, 7-9 SQ_INSTS_ALL.
  append({kUconfig, Register(0x30800), 0x00000000});
  append(copy(0x34700, 32));
  append(copy(0x34708, 56));
  append({kUconfig, Register(0x30800), 0x00000110});
  append(copy(0x34700, 40));
  append(copy(0x34708, 64));
  append({kUconfig, Register(0x30800), 0x00010008});
  append(copy(0x34700, 48));
  append(copy(0x34708, 72));
  append({kUconfig, Register(0x30800), 0xe0000000});
  EXPECT_EQ(emitted, expected);
}

TEST_F(PerfCountersTest, CopyTableRelocatesASequenceEmittedAtZero) {
  SelectInstances();
  const uint64_t target = UINT64_C(0x0000007700001000);
  std::array<uint32_t, 512> direct;
  direct.fill(kGuard);
  uint32_t* direct_end = iree_hal_amd_gpu_pm4_perf_counter_emit_sample(
      direct.data(), &config_, target);
  uint32_t* end =
      iree_hal_amd_gpu_pm4_perf_counter_emit_sample(words(), &config_, 0);
  const uint32_t copy_count =
      iree_hal_amd_gpu_pm4_perf_counter_sample_copy_count(&config_);
  EXPECT_EQ(copy_count, 2u * 2 * 2 + 3 * 2);
  for (uint32_t i = 0; i < copy_count; ++i) {
    uint32_t word_offset = 0, byte_offset = 0;
    iree_hal_amd_gpu_pm4_perf_counter_sample_copy(&config_, i, &word_offset,
                                                  &byte_offset);
    ASSERT_LT(byte_offset,
              iree_hal_amd_gpu_pm4_perf_counter_sample_byte_length(&config_));
    // Emitted at zero, a destination pair holds its own sample offset, and
    // it sits exactly where the table says.
    EXPECT_EQ(words()[word_offset - 4], kCopyData) << i;
    EXPECT_EQ(words()[word_offset], byte_offset) << i;
    EXPECT_EQ(words()[word_offset + 1], 0u) << i;
    const uint64_t address = target + byte_offset;
    words()[word_offset] = (uint32_t)address;
    words()[word_offset + 1] = (uint32_t)(address >> 32);
  }
  const std::vector<uint32_t> relocated = Emitted(end);
  EXPECT_EQ(relocated, std::vector<uint32_t>(direct.data(), direct_end));
}

TEST_F(PerfCountersTest, StopDisablesCountingAndRestoresResetDefaults) {
  SelectInstances();
  const std::vector<uint32_t> emitted =
      Emitted(iree_hal_amd_gpu_pm4_perf_counter_emit_stop(words(), &config_));
  ASSERT_EQ(emitted.size(),
            iree_hal_amd_gpu_pm4_perf_counter_stop_word_count(&config_));
  const std::vector<uint32_t> expected = {
      kUconfig,
      Register(0x30800),
      0xe0000000,
      // Stop, join, then this queue stops being counted.
      kUconfig,
      Register(0x36020),
      2,
      kComputeFlush,
      0x80000407,
      0xc0017602,
      0x20b,
      0,
      // Reset defaults.
      kUconfig,
      Register(0x36020),
      0,
      kUconfig,
      Register(0x36740),
      0x1ff,
      kUconfig,
      Register(0x36744),
      0x1ff,
      kUconfig,
      Register(0x36700),
      0x1ff,
      kUconfig,
      Register(0x36708),
      0x1ff,
      kUconfig,
      Register(0x36780),
      0,
      kUconfig,
      Register(0x36760),
      0,
      kUconfig,
      Register(0x36788),
      0x1fffe,
      kUconfig,
      Register(0x36768),
      0x1fffe,
      kUconfig,
      Register(0x30800),
      0xe0000000,
  };
  EXPECT_EQ(emitted, expected);
}

TEST_F(PerfCountersTest, DeltaSubtractsEachWidthAndRejectsUnwrittenSamples) {
  SelectInstances();
  const uint32_t length =
      iree_hal_amd_gpu_pm4_perf_counter_sample_byte_length(&config_);
  std::vector<uint8_t> begin(length), end(length);
  iree_hal_amd_gpu_pm4_perf_counter_sample_reset(&config_, begin.data());
  iree_hal_amd_gpu_pm4_perf_counter_sample_reset(&config_, end.data());
  std::vector<uint64_t> values(10, UINT64_C(0x1111111111111111));
  // Nothing was copied.
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_perf_counter_sample_delta(
      &config_, begin.data(), end.data(), values.data()));
  EXPECT_EQ(values, std::vector<uint64_t>(10, 0));

  auto store64 = [](std::vector<uint8_t>& sample, uint32_t value,
                    uint64_t raw) { memcpy(&sample[8 * value], &raw, 8); };
  auto store32 = [](std::vector<uint8_t>& sample, uint32_t value,
                    uint32_t raw) { memcpy(&sample[8 * value], &raw, 4); };
  for (uint32_t i = 0; i < 4; ++i) {
    store64(begin, i, 1000 + i);
    store64(end, i, 1000 + i + 128 * (i + 1));
  }
  // A 64-bit counter that crossed the low DWORD boundary.
  store64(begin, 3, UINT64_C(0x00000000fffffff0));
  store64(end, 3, UINT64_C(0x0000000100000010));
  for (uint32_t i = 4; i < 10; ++i) {
    store32(begin, i, 50 * i);
    store32(end, i, 50 * i + i);
  }
  // A 32-bit counter that wrapped.
  store32(begin, 9, 0xfffffff0);
  store32(end, 9, 0x00000010);
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_perf_counter_sample_delta(
      &config_, begin.data(), end.data(), values.data()));
  EXPECT_EQ(values,
            (std::vector<uint64_t>{128, 256, 384, 0x20, 4, 5, 6, 7, 8, 0x20}));

  // One copy of the end sample never ran.
  store32(end, 6, 0xffffffff);
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_perf_counter_sample_delta(
      &config_, begin.data(), end.data(), values.data()));
  EXPECT_EQ(values, std::vector<uint64_t>(10, 0));
}

TEST_F(PerfCountersTest, StateWordsTellAnotherCollector) {
  const uint32_t defaults[] = {0, 0, 0x1ff, 0xffffffff};
  EXPECT_TRUE(
      iree_hal_amd_gpu_pm4_perf_counter_state_is_default(&config_, defaults));
  const uint32_t counting[] = {1, 0x55, 0x14, 0xffffffff};
  EXPECT_FALSE(
      iree_hal_amd_gpu_pm4_perf_counter_state_is_default(&config_, counting));
  const uint32_t selected[] = {0, 0, 0x14, 0xffffffff};
  EXPECT_FALSE(
      iree_hal_amd_gpu_pm4_perf_counter_state_is_default(&config_, selected));
  // Never copied.
  const uint32_t unwritten[] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
  EXPECT_FALSE(
      iree_hal_amd_gpu_pm4_perf_counter_state_is_default(&config_, unwritten));
}

}  // namespace
}  // namespace iree::hal::amd::gpu
