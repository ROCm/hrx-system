// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/sampling.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ScopedDenoiseSchedule {
  ~ScopedDenoiseSchedule() {
    id4_ideogram4_denoise_schedule_deinitialize(&value,
                                                iree_allocator_system());
  }

  id4_ideogram4_denoise_schedule_t value = {};
};

TEST(Ideogram4SamplingTest, ParsesAdvertisedPresetNames) {
  const struct {
    iree_string_view_t name;
    id4_ideogram4_sampler_preset_t preset;
    uint32_t step_count;
  } cases[] = {
      {IREE_SV("V4_QUALITY_48"), ID4_IDEOGRAM4_SAMPLER_PRESET_V4_QUALITY_48,
       48},
      {IREE_SV("V4_DEFAULT_20"), ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20,
       20},
      {IREE_SV("V4_TURBO_12"), ID4_IDEOGRAM4_SAMPLER_PRESET_V4_TURBO_12, 12},
  };
  for (const auto& test_case : cases) {
    id4_ideogram4_sampler_preset_t preset = {};
    IREE_ASSERT_OK(id4_ideogram4_sampler_preset_parse(test_case.name, &preset));
    EXPECT_EQ(preset, test_case.preset);
    EXPECT_TRUE(iree_string_view_equal(
        id4_ideogram4_sampler_preset_name(preset), test_case.name));
    EXPECT_EQ(id4_ideogram4_sampler_preset_step_count(preset),
              test_case.step_count);
  }

  id4_ideogram4_sampler_preset_t preset =
      ID4_IDEOGRAM4_SAMPLER_PRESET_V4_QUALITY_48;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_sampler_preset_parse(IREE_SV("DEFAULT"), &preset));
  EXPECT_EQ(preset, ID4_IDEOGRAM4_SAMPLER_PRESET_INVALID);
  EXPECT_TRUE(iree_string_view_is_empty(
      id4_ideogram4_sampler_preset_name(ID4_IDEOGRAM4_SAMPLER_PRESET_INVALID)));
  EXPECT_EQ(id4_ideogram4_sampler_preset_step_count(
                ID4_IDEOGRAM4_SAMPLER_PRESET_INVALID),
            0u);
}

TEST(Ideogram4SamplingTest, LowersDefault1024ScheduleLikeReference) {
  // Reference values are LogitNormalSchedule applied to the official float32
  // intervals before the Python sampling loop reverses them.
  static const float reference_interval_times[] = {
      0.999447226524353f,    0.8989259004592896f,   0.8248513340950012f,
      0.7541020512580872f,   0.6856132745742798f,   0.6194489598274231f,
      0.5559036731719971f,   0.4952910542488098f,   0.43787524104118347f,
      0.3838496804237366f,   0.3333333432674408f,   0.2863751947879791f,
      0.24296267330646515f,  0.20303086936473846f,  0.16647091507911682f,
      0.1331367790699005f,   0.10284695774316788f,  0.07537547498941422f,
      0.050408974289894104f, 0.027341142296791077f, 0.00012339458044152707f,
  };

  ScopedDenoiseSchedule schedule;
  IREE_ASSERT_OK(id4_ideogram4_sampler_preset_lower_schedule(
      ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20, 1024, 1024,
      iree_allocator_system(), &schedule.value));
  ASSERT_EQ(schedule.value.step_count, 20u);
  for (uint32_t step_ordinal = 0; step_ordinal < schedule.value.step_count;
       ++step_ordinal) {
    const uint32_t current_interval = schedule.value.step_count - step_ordinal;
    const auto& step = schedule.value.steps[step_ordinal];
    EXPECT_NEAR(step.flow_time, reference_interval_times[current_interval],
                1e-7f);
    EXPECT_NEAR(step.next_flow_time,
                reference_interval_times[current_interval - 1], 1e-7f);
    EXPECT_FLOAT_EQ(step.guidance_scale, step_ordinal < 18 ? 7.0f : 3.0f);
  }
}

TEST(Ideogram4SamplingTest, ResolutionChangesScheduleMean) {
  ScopedDenoiseSchedule schedule_512;
  IREE_ASSERT_OK(id4_ideogram4_sampler_preset_lower_schedule(
      ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20, 512, 512,
      iree_allocator_system(), &schedule_512.value));
  ScopedDenoiseSchedule schedule_1024;
  IREE_ASSERT_OK(id4_ideogram4_sampler_preset_lower_schedule(
      ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20, 1024, 1024,
      iree_allocator_system(), &schedule_1024.value));

  ASSERT_EQ(schedule_512.value.step_count, schedule_1024.value.step_count);
  // Larger images raise the schedule mean and therefore lower interior flow
  // times. Clamped endpoints remain resolution-independent.
  EXPECT_GT(schedule_512.value.steps[9].flow_time,
            schedule_1024.value.steps[9].flow_time);
  EXPECT_FLOAT_EQ(schedule_512.value.steps[0].flow_time,
                  schedule_1024.value.steps[0].flow_time);
  EXPECT_FLOAT_EQ(schedule_512.value.steps[19].next_flow_time,
                  schedule_1024.value.steps[19].next_flow_time);
}

TEST(Ideogram4SamplingTest, RejectsInvalidInput) {
  ScopedDenoiseSchedule schedule;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_sampler_preset_lower_schedule(
                            (id4_ideogram4_sampler_preset_t)100, 512, 512,
                            iree_allocator_system(), &schedule.value));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_sampler_preset_lower_schedule(
                            ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20, 0, 512,
                            iree_allocator_system(), &schedule.value));
}

}  // namespace
