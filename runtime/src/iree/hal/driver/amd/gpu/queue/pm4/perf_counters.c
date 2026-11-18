// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/perf_counters.h"

#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"

#define IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(name_, select_, \
                                                       cycles_, text_) \
  {                                                                    \
      .name = IREE_SVL(name_),                                         \
      .block_name = IREE_SVL("SQG"),                                   \
      .description = IREE_SVL(text_),                                  \
      .block = IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_ENGINE,         \
      .select = select_,                                               \
      .cycles = cycles_,                                               \
  }
#define IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_INSTANCE_EVENT(name_, select_, \
                                                         text_)          \
  {                                                                      \
      .name = IREE_SVL(name_),                                           \
      .block_name = IREE_SVL("SQ"),                                      \
      .description = IREE_SVL(text_),                                    \
      .block = IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_INSTANCE,         \
      .select = select_,                                                 \
      .cycles = false,                                                   \
  }

// The selectable events, in catalog order, with one generation's SQG_PERF_SEL
// and SQ_PERF_SEL numbers. Names and meanings are the same on every
// generation that has the event; only the numbers move.
#define IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_EVENTS(                            \
    sqg_waves_, sqg_waves_32_, sqg_waves_64_, sqg_items_, sqg_busy_cycles_,  \
    sqg_wave_cycles_, sqg_cycles_, sqg_waves_saved_, sq_waves_,              \
    sq_insts_valu_, sq_insts_salu_, sq_insts_all_)                           \
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(                            \
      "SQG_WAVES", sqg_waves_, false, "Waves launched, per shader engine"),  \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(                        \
          "SQG_WAVES_32", sqg_waves_32_, false,                              \
          "32-lane waves launched, per shader engine"),                      \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(                        \
          "SQG_WAVES_64", sqg_waves_64_, false,                              \
          "64-lane waves launched, per shader engine"),                      \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(                        \
          "SQG_ITEMS", sqg_items_, false,                                    \
          "Workitems launched, per shader engine"),                          \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(                        \
          "SQG_BUSY_CYCLES", sqg_busy_cycles_, true,                         \
          "Clock cycles with at least one wave in the shader engine"),       \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(                        \
          "SQG_WAVE_CYCLES", sqg_wave_cycles_, true,                         \
          "Clock cycles summed over the waves in the shader engine"),        \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(                        \
          "SQG_CYCLES", sqg_cycles_, true,                                   \
          "Clock cycles while counting, per shader engine"),                 \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_ENGINE_EVENT(                        \
          "SQG_WAVES_SAVED", sqg_waves_saved_, false,                        \
          "Waves saved by a context switch, per shader engine"),             \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_INSTANCE_EVENT(                      \
          "SQ_WAVES", sq_waves_, "Waves launched, per workgroup processor"), \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_INSTANCE_EVENT(                      \
          "SQ_INSTS_VALU", sq_insts_valu_,                                   \
          "Vector ALU instructions issued, per workgroup processor"),        \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_INSTANCE_EVENT(                      \
          "SQ_INSTS_SALU", sq_insts_salu_,                                   \
          "Scalar ALU instructions issued, per workgroup processor"),        \
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_INSTANCE_EVENT(                      \
          "SQ_INSTS_ALL", sq_insts_all_,                                     \
          "Instructions issued, per workgroup processor")

// GFX11.0 and GFX11.5 event numbers (soc21). On gfx1151 the wave, workitem and
// instruction events matched predicted counts exactly; the cycle events were
// not compared with a clock.
static const iree_hal_amd_gpu_pm4_perf_counter_event_t
    iree_hal_amd_gpu_pm4_perf_counter_gfx11_events[] = {
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_EVENTS(0x14, 0x15, 0x16, 0x18, 0x10,
                                                 0x11, 0x0f, 0x0d, 0x04, 0x3e,
                                                 0x3a, 0x30),
};

// GFX12.0 event numbers (soc24): every SQG event and every SQ instruction
// event is renumbered against GFX11. Never counted on hardware.
static const iree_hal_amd_gpu_pm4_perf_counter_event_t
    iree_hal_amd_gpu_pm4_perf_counter_gfx12_events[] = {
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_EVENTS(0x13, 0x14, 0x15, 0x17, 0x0f,
                                                 0x10, 0x0e, 0x0c, 0x04, 0x32,
                                                 0x2e, 0x27),
};

// Registers and control values GFX11.0, GFX11.5 and GFX12.0 share: the
// register databases give all three the same addresses and field layouts for
// every register here. Programmed and read back on gfx1151 only.
static const iree_hal_amd_gpu_pm4_perf_counter_profile_t
    iree_hal_amd_gpu_pm4_perf_counter_gfx11_registers = {
        .graphics_index = 0x30800,
        .graphics_index_broadcast = UINT32_C(0xe0000000),
        .control = 0x36020,
        // Disable and clear; count; count and latch a sample; stop.
        .control_clear = 0,
        .control_start = 1,
        .control_sample = 0x401,
        .control_stop = 2,
        .instance_block_control = 0x36780,
        .engine_block_control = 0x36760,
        // PS, GS, HS and CS stage enables.
        .block_control_stages = 0x55,
        .instance_block_filter = 0x36788,
        .engine_block_filter = 0x36768,
        // VMIDs 0-15 enabled, FORCE_EN (bit 0) clear.
        .block_filter_default = 0x1fffe,
        .engine_select = 0x36740,
        .instance_select = 0x36700,
        .select_default = 0x1ff,
        .engine_result = 0x34790,
        .instance_result = 0x34700,
        .compute_enable = 0x2c00 + ((0xb82c - 0xb000) >> 2),
};

// One generation of the capability table.
typedef struct iree_hal_amd_gpu_pm4_perf_counter_target_t {
  // gfx IP major and minor version the row applies to.
  uint32_t major;
  uint32_t minor;
  // Steppings whose sequences ran on hardware, one bit per stepping.
  uint32_t verified_steppings;
  // Register table of the generation.
  const iree_hal_amd_gpu_pm4_perf_counter_profile_t* registers;
  // Event catalog of the generation.
  const iree_hal_amd_gpu_pm4_perf_counter_event_t* events;
  uint32_t event_count;
} iree_hal_amd_gpu_pm4_perf_counter_target_t;

// A generation without a table, and what a table for it would need.
typedef struct iree_hal_amd_gpu_pm4_perf_counter_gap_t {
  // gfx IP major version, and its minor or UINT32_MAX for every minor.
  uint32_t major;
  uint32_t minor;
  // What differs and where the missing values are documented.
  const char* reason;
} iree_hal_amd_gpu_pm4_perf_counter_gap_t;

// Only GFX11.5 stepping 1 (gfx1151) has run: no other GPU was available. Every
// other row and stepping is transcribed from references and unverified. What
// was measured there and is assumed elsewhere: the per-queue enable, the
// sample strobe, the block filters and the pairing of select 2k with result k.
static const iree_hal_amd_gpu_pm4_perf_counter_target_t
    iree_hal_amd_gpu_pm4_perf_counter_targets[] = {
        {11, 5, 1u << 1, &iree_hal_amd_gpu_pm4_perf_counter_gfx11_registers,
         iree_hal_amd_gpu_pm4_perf_counter_gfx11_events,
         IREE_ARRAYSIZE(iree_hal_amd_gpu_pm4_perf_counter_gfx11_events)},
        // aqlprofile drives GFX11.0 with the GFX11.5 registers and events.
        {11, 0, 0, &iree_hal_amd_gpu_pm4_perf_counter_gfx11_registers,
         iree_hal_amd_gpu_pm4_perf_counter_gfx11_events,
         IREE_ARRAYSIZE(iree_hal_amd_gpu_pm4_perf_counter_gfx11_events)},
        // Mesa's gfx12 database omits the perfmon control register; the
        // GFX12.0 kernel header names it CP_CP_PERFMON_CNTL at this address.
        {12, 0, 0, &iree_hal_amd_gpu_pm4_perf_counter_gfx11_registers,
         iree_hal_amd_gpu_pm4_perf_counter_gfx12_events,
         IREE_ARRAYSIZE(iree_hal_amd_gpu_pm4_perf_counter_gfx12_events)},
};

static const iree_hal_amd_gpu_pm4_perf_counter_gap_t
    iree_hal_amd_gpu_pm4_perf_counter_gaps[] = {
        {9, UINT32_MAX,
         "GFX9 (CDNA) has no SQG block, numbers its SQ events differently on "
         "gfx908, gfx90a and gfx940, and needs the RLC perfmon clock held; "
         "the values are in aqlprofile pmc_builder.h (GFXIP_LEVEL == 9) and "
         "rocprofiler-sdk basic_counters.xml"},
        {10, UINT32_MAX,
         "GFX10 has no SQG block and numbers its SQ events differently "
         "(navi10_enum.h); the sequence was not derived for it"},
        {11, 7, "no register database or event table for GFX11.7 is available"},
        {12, 5,
         "GFX12.5 has its own register set (gc_12_1), a second perfmon "
         "control (AID_PERFMON_CNTL) and per-XCC selection that the sequences "
         "do not model; see aqlprofile gfx12_primitives.h "
         "(GFX12_VARIANT_1250)"},
};

iree_status_t iree_hal_amd_gpu_pm4_perf_counter_profile_initialize(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_perf_counter_profile_t* out_profile) {
  *out_profile = (iree_hal_amd_gpu_pm4_perf_counter_profile_t){0};
  for (uint32_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amd_gpu_pm4_perf_counter_targets); ++i) {
    const iree_hal_amd_gpu_pm4_perf_counter_target_t* target =
        &iree_hal_amd_gpu_pm4_perf_counter_targets[i];
    if (target->major != gfxip.major || target->minor != gfxip.minor) {
      continue;
    }
    *out_profile = *target->registers;
    out_profile->events = target->events;
    out_profile->event_count = target->event_count;
    out_profile->verified =
        gfxip.stepping < 32 &&
        (target->verified_steppings & (1u << gfxip.stepping)) != 0;
    return iree_ok_status();
  }
  const char* reason = "no register table has been written for this generation";
  for (uint32_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amd_gpu_pm4_perf_counter_gaps); ++i) {
    const iree_hal_amd_gpu_pm4_perf_counter_gap_t* gap =
        &iree_hal_amd_gpu_pm4_perf_counter_gaps[i];
    if (gap->major == gfxip.major &&
        (gap->minor == UINT32_MAX || gap->minor == gfxip.minor)) {
      reason = gap->reason;
      break;
    }
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "no PM4 performance counter table for gfx%u.%u.%u: "
                          "%s",
                          gfxip.major, gfxip.minor, gfxip.stepping, reason);
}

const iree_hal_amd_gpu_pm4_perf_counter_event_t*
iree_hal_amd_gpu_pm4_perf_counter_profile_find(
    const iree_hal_amd_gpu_pm4_perf_counter_profile_t* profile,
    iree_string_view_t name) {
  for (uint32_t i = 0; i < profile->event_count; ++i) {
    if (iree_string_view_equal(profile->events[i].name, name)) {
      return &profile->events[i];
    }
  }
  return NULL;
}

iree_status_t iree_hal_amd_gpu_pm4_perf_counter_config_validate(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  if (!config->engine_count ||
      config->engine_count >
          IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_ENGINE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "performance counters address 1 to %d shader "
                            "engines; the device has %u",
                            IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_ENGINE_COUNT,
                            config->engine_count);
  }
  if (config->engine_counter_count >
          IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_ENGINE_COUNTERS ||
      config->instance_counter_count >
          IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_INSTANCE_COUNTERS) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "performance counter selection holds %u engine and %u instance "
        "counters; at most %d and %d fit",
        config->engine_counter_count, config->instance_counter_count,
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_ENGINE_COUNTERS,
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_INSTANCE_COUNTERS);
  }
  if (!config->engine_counter_count && !config->instance_counter_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "performance counter selection is empty");
  }
  if (config->instance_count >
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_INSTANCE_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "performance counters address at most %d workgroup processors; the "
        "device has %u",
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_MAX_INSTANCE_COUNT,
        config->instance_count);
  }
  if (config->instance_counter_count && !config->instance_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "instance performance counters need the device's workgroup "
        "processors");
  }
  for (uint32_t i = 0; i < config->engine_counter_count; ++i) {
    if (config->engine_counters[i]->block !=
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_ENGINE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s is not a shader-engine counter",
                              (int)config->engine_counters[i]->name.size,
                              config->engine_counters[i]->name.data);
    }
  }
  for (uint32_t i = 0; i < config->instance_counter_count; ++i) {
    if (config->instance_counters[i]->block !=
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_BLOCK_INSTANCE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s is not a workgroup-processor counter",
                              (int)config->instance_counters[i]->name.size,
                              config->instance_counters[i]->name.data);
    }
  }
  return iree_ok_status();
}

static uint64_t iree_hal_amd_gpu_pm4_perf_counter_mix(uint64_t hash,
                                                      uint32_t value) {
  // FNV-1a over the value's four bytes.
  for (uint32_t i = 0; i < 4; ++i) {
    hash = (hash ^ ((value >> (8 * i)) & 0xff)) * UINT64_C(0x100000001b3);
  }
  return hash;
}

uint64_t iree_hal_amd_gpu_pm4_perf_counter_config_signature(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  hash = iree_hal_amd_gpu_pm4_perf_counter_mix(hash, config->engine_count);
  hash =
      iree_hal_amd_gpu_pm4_perf_counter_mix(hash, config->engine_counter_count);
  for (uint32_t i = 0; i < config->engine_counter_count; ++i) {
    hash = iree_hal_amd_gpu_pm4_perf_counter_mix(
        hash, config->engine_counters[i]->select);
  }
  hash = iree_hal_amd_gpu_pm4_perf_counter_mix(hash,
                                               config->instance_counter_count);
  for (uint32_t i = 0; i < config->instance_counter_count; ++i) {
    hash = iree_hal_amd_gpu_pm4_perf_counter_mix(
        hash, config->instance_counters[i]->select);
  }
  const uint32_t instance_count =
      config->instance_counter_count ? config->instance_count : 0;
  hash = iree_hal_amd_gpu_pm4_perf_counter_mix(hash, instance_count);
  for (uint32_t i = 0; i < instance_count; ++i) {
    hash = iree_hal_amd_gpu_pm4_perf_counter_mix(hash, config->instances[i]);
  }
  return hash ? hash : 1;
}

// Instances read by a sample: none unless an instance counter is selected.
static uint32_t iree_hal_amd_gpu_pm4_perf_counter_instances(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return config->instance_counter_count ? config->instance_count : 0;
}

uint32_t iree_hal_amd_gpu_pm4_perf_counter_value_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return config->engine_counter_count * config->engine_count +
         config->instance_counter_count *
             iree_hal_amd_gpu_pm4_perf_counter_instances(config);
}

uint32_t iree_hal_amd_gpu_pm4_perf_counter_engine_value_offset(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config,
    uint32_t counter) {
  return counter * config->engine_count;
}

uint32_t iree_hal_amd_gpu_pm4_perf_counter_instance_value_offset(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config,
    uint32_t counter) {
  return config->engine_counter_count * config->engine_count +
         counter * iree_hal_amd_gpu_pm4_perf_counter_instances(config);
}

enum {
  // Compute join, sample strobe, broadcast and the join the strobe needs.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SAMPLE_PREFIX_WORDS = 2 + 3 + 3 + 2,
  // One graphics-index write.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SELECT_WORDS = 3,
  // One register copy.
  IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_COPY_WORDS = 6,
};

// DWORDs sampling one shader engine, and one workgroup processor.
static uint32_t iree_hal_amd_gpu_pm4_perf_counter_engine_words(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SELECT_WORDS +
         2 * IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_COPY_WORDS *
             config->engine_counter_count;
}
static uint32_t iree_hal_amd_gpu_pm4_perf_counter_instance_words(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SELECT_WORDS +
         IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_COPY_WORDS *
             config->instance_counter_count;
}

uint32_t iree_hal_amd_gpu_pm4_perf_counter_start_word_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return 52 +
         3 * (config->engine_counter_count + config->instance_counter_count);
}

uint32_t iree_hal_amd_gpu_pm4_perf_counter_sample_word_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SAMPLE_PREFIX_WORDS +
         config->engine_count *
             iree_hal_amd_gpu_pm4_perf_counter_engine_words(config) +
         iree_hal_amd_gpu_pm4_perf_counter_instances(config) *
             iree_hal_amd_gpu_pm4_perf_counter_instance_words(config) +
         IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SELECT_WORDS;
}

uint32_t iree_hal_amd_gpu_pm4_perf_counter_stop_word_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return 29 +
         3 * (config->engine_counter_count + config->instance_counter_count);
}

static uint32_t* iree_hal_amd_gpu_pm4_perf_counter_emit_register(
    uint32_t* words, uint32_t register_address, uint32_t value) {
  return iree_hal_amd_gpu_pm4_emit_uconfig_registers(words, register_address, 1,
                                                     &value);
}

static uint32_t* iree_hal_amd_gpu_pm4_perf_counter_emit_broadcast(
    uint32_t* words,
    const iree_hal_amd_gpu_pm4_perf_counter_profile_t* profile) {
  return iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->graphics_index, profile->graphics_index_broadcast);
}

uint32_t* iree_hal_amd_gpu_pm4_perf_counter_emit_start(
    uint32_t* words, const iree_hal_amd_gpu_pm4_perf_counter_config_t* config,
    uint64_t state_address) {
  const iree_hal_amd_gpu_pm4_perf_counter_profile_t* profile = config->profile;
  // Earlier work on this queue must not be counted.
  words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_broadcast(words, profile);
  // What another collector would have left, read before anything is written.
  words = iree_hal_amd_gpu_pm4_emit_copy_register32(words, profile->control,
                                                    state_address);
  words = iree_hal_amd_gpu_pm4_emit_copy_register32(
      words, profile->engine_block_control, state_address + 4);
  words = iree_hal_amd_gpu_pm4_emit_copy_register32(
      words, profile->engine_select, state_address + 8);
  // Counting stays off and every counter is cleared while selects change.
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->control, profile->control_clear);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->instance_block_control, profile->block_control_stages);
  // The filters decide whose waves count. The default admits every VMID and
  // leaves FORCE_EN clear, so only queues that set their own enable count.
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->instance_block_filter, profile->block_filter_default);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->engine_block_filter, profile->block_filter_default);
  for (uint32_t i = 0; i < config->engine_counter_count; ++i) {
    words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
        words, profile->engine_select + 4 * i,
        config->engine_counters[i]->select);
  }
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->engine_block_control, profile->block_control_stages);
  for (uint32_t i = 0; i < config->instance_counter_count; ++i) {
    words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
        words, profile->instance_select + 8 * i,
        config->instance_counters[i]->select);
  }
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_broadcast(words, profile);
  // Queue state: without it every counter reads zero.
  const uint32_t enable = 1;
  words = iree_hal_amd_gpu_pm4_emit_registers(words, profile->compute_enable, 1,
                                              &enable);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->control, profile->control_clear);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->control, profile->control_start);
  return iree_hal_amd_gpu_pm4_emit_compute_flush(words);
}

uint32_t* iree_hal_amd_gpu_pm4_perf_counter_emit_sample(
    uint32_t* words, const iree_hal_amd_gpu_pm4_perf_counter_config_t* config,
    uint64_t target_address) {
  const iree_hal_amd_gpu_pm4_perf_counter_profile_t* profile = config->profile;
  const uint32_t instance_count =
      iree_hal_amd_gpu_pm4_perf_counter_instances(config);
  // Every wave launched before this point has finished and is counted.
  words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->control, profile->control_sample);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_broadcast(words, profile);
  words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  for (uint32_t engine = 0; engine < config->engine_count; ++engine) {
    // Engine index with array and instance broadcast.
    words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
        words, profile->graphics_index, (engine << 16) | UINT32_C(0x60000000));
    for (uint32_t i = 0; i < config->engine_counter_count; ++i) {
      const uint64_t value_address =
          target_address +
          8ull * (iree_hal_amd_gpu_pm4_perf_counter_engine_value_offset(config,
                                                                        i) +
                  engine);
      words = iree_hal_amd_gpu_pm4_emit_copy_register32(
          words, profile->engine_result + 8 * i, value_address);
      words = iree_hal_amd_gpu_pm4_emit_copy_register32(
          words, profile->engine_result + 8 * i + 4, value_address + 4);
    }
  }
  for (uint32_t instance = 0; instance < instance_count; ++instance) {
    words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
        words, profile->graphics_index, config->instances[instance]);
    for (uint32_t i = 0; i < config->instance_counter_count; ++i) {
      const uint64_t value_address =
          target_address +
          8ull * (iree_hal_amd_gpu_pm4_perf_counter_instance_value_offset(
                      config, i) +
                  instance);
      words = iree_hal_amd_gpu_pm4_emit_copy_register32(
          words, profile->instance_result + 8 * i, value_address);
    }
  }
  return iree_hal_amd_gpu_pm4_perf_counter_emit_broadcast(words, profile);
}

uint32_t iree_hal_amd_gpu_pm4_perf_counter_sample_copy_count(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  return 2 * config->engine_count * config->engine_counter_count +
         iree_hal_amd_gpu_pm4_perf_counter_instances(config) *
             config->instance_counter_count;
}

void iree_hal_amd_gpu_pm4_perf_counter_sample_copy(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config, uint32_t index,
    uint32_t* out_word_offset, uint32_t* out_byte_offset) {
  // The destination pair is the last two DWORDs of a copy.
  const uint32_t address_word =
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_COPY_WORDS - 2;
  const uint32_t engine_copy_count =
      2 * config->engine_count * config->engine_counter_count;
  if (index < engine_copy_count) {
    const uint32_t half = index & 1;
    const uint32_t counter = (index / 2) % config->engine_counter_count;
    const uint32_t engine = (index / 2) / config->engine_counter_count;
    *out_word_offset =
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SAMPLE_PREFIX_WORDS +
        engine * iree_hal_amd_gpu_pm4_perf_counter_engine_words(config) +
        IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SELECT_WORDS +
        (2 * counter + half) * IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_COPY_WORDS +
        address_word;
    *out_byte_offset =
        8 * (iree_hal_amd_gpu_pm4_perf_counter_engine_value_offset(config,
                                                                   counter) +
             engine) +
        4 * half;
    return;
  }
  index -= engine_copy_count;
  const uint32_t counter = index % config->instance_counter_count;
  const uint32_t instance = index / config->instance_counter_count;
  *out_word_offset =
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SAMPLE_PREFIX_WORDS +
      config->engine_count *
          iree_hal_amd_gpu_pm4_perf_counter_engine_words(config) +
      instance * iree_hal_amd_gpu_pm4_perf_counter_instance_words(config) +
      IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_SELECT_WORDS +
      counter * IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_COPY_WORDS + address_word;
  *out_byte_offset =
      8 * (iree_hal_amd_gpu_pm4_perf_counter_instance_value_offset(config,
                                                                   counter) +
           instance);
}

uint32_t* iree_hal_amd_gpu_pm4_perf_counter_emit_stop(
    uint32_t* words, const iree_hal_amd_gpu_pm4_perf_counter_config_t* config) {
  const iree_hal_amd_gpu_pm4_perf_counter_profile_t* profile = config->profile;
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_broadcast(words, profile);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->control, profile->control_stop);
  words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  // This queue's later work is not counted by anyone.
  const uint32_t disable = 0;
  words = iree_hal_amd_gpu_pm4_emit_registers(words, profile->compute_enable, 1,
                                              &disable);
  // Reset defaults: the state the next collector expects to find.
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->control, profile->control_clear);
  for (uint32_t i = 0; i < config->engine_counter_count; ++i) {
    words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
        words, profile->engine_select + 4 * i, profile->select_default);
  }
  for (uint32_t i = 0; i < config->instance_counter_count; ++i) {
    words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
        words, profile->instance_select + 8 * i, profile->select_default);
  }
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->instance_block_control, 0);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->engine_block_control, 0);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->instance_block_filter, profile->block_filter_default);
  words = iree_hal_amd_gpu_pm4_perf_counter_emit_register(
      words, profile->engine_block_filter, profile->block_filter_default);
  return iree_hal_amd_gpu_pm4_perf_counter_emit_broadcast(words, profile);
}

void iree_hal_amd_gpu_pm4_perf_counter_sample_reset(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config, void* sample) {
  memset(sample, 0xff,
         iree_hal_amd_gpu_pm4_perf_counter_sample_byte_length(config));
}

bool iree_hal_amd_gpu_pm4_perf_counter_sample_delta(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config, const void* begin,
    const void* end, uint64_t* out_values) {
  const uint32_t value_count =
      iree_hal_amd_gpu_pm4_perf_counter_value_count(config);
  const uint32_t engine_value_count =
      config->engine_counter_count * config->engine_count;
  const uint32_t unwritten =
      (uint32_t)IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_UNWRITTEN;
  memset(out_values, 0, value_count * sizeof(*out_values));
  bool written = true;
  for (uint32_t i = 0; i < value_count && written; ++i) {
    uint32_t begin_words[2], end_words[2];
    memcpy(begin_words, (const uint8_t*)begin + 8 * i, sizeof(begin_words));
    memcpy(end_words, (const uint8_t*)end + 8 * i, sizeof(end_words));
    if (i < engine_value_count) {
      // A counter cannot reach the all-ones pattern in both halves.
      written = (begin_words[0] != unwritten || begin_words[1] != unwritten) &&
                (end_words[0] != unwritten || end_words[1] != unwritten);
      const uint64_t begin_value =
          begin_words[0] | ((uint64_t)begin_words[1] << 32);
      const uint64_t end_value = end_words[0] | ((uint64_t)end_words[1] << 32);
      out_values[i] = end_value - begin_value;
    } else {
      // Only the low DWORD is copied; the high one keeps the host pattern.
      written = begin_words[0] != unwritten && end_words[0] != unwritten;
      out_values[i] = (uint32_t)(end_words[0] - begin_words[0]);
    }
  }
  if (!written) {
    memset(out_values, 0, value_count * sizeof(*out_values));
  }
  return written;
}

bool iree_hal_amd_gpu_pm4_perf_counter_state_is_default(
    const iree_hal_amd_gpu_pm4_perf_counter_config_t* config,
    const uint32_t* state_words) {
  const iree_hal_amd_gpu_pm4_perf_counter_profile_t* profile = config->profile;
  return state_words[0] == profile->control_clear && state_words[1] == 0 &&
         state_words[2] == profile->select_default;
}
